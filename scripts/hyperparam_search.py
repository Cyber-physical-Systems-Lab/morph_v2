"""
hyperparam_search.py
====================
Bounded Bayesian hyperparameter optimization to close the Proximity gap at
medium and large scales.

Diagnosis: MORPH v2 uses MORE links than Proximity but fewer deliveries.
Problem is link quality, not cold-start. Search strategy:
  - Lower target_deg_frac  → fewer, more selective links
  - Raise reward_alpha     → delivery signal dominates Hebbian noise
  - Raise bcm_gain         → stale links decay faster
  - Raise theta_prune      → prune marginal links more aggressively

Enumerates integer-like settings and runs Optuna TPE over bounded continuous
parameters. The best config is verified on all 4 scales with 5 seeds.

Usage:
  python scripts/hyperparam_search.py
"""
import sys, os, warnings, itertools, pickle
import numpy as np
import optuna
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES_DIR = os.path.join(ROOT, 'results')
os.makedirs(RES_DIR, exist_ok=True)

import tarware, gymnasium as gym
from tarware.heuristic import AgentType, Mission, MissionType
from morph import MORPH
from collections import OrderedDict

T       = 400
SEARCH_SEEDS  = [42, 7, 123]          # fast sweep
VERIFY_SEEDS  = [42, 7, 123, 999, 2024]
N_TRIALS = 500
STUDY_DB = os.path.join(RES_DIR, 'hyperparam_bo.db')
STORAGE_URL = f"sqlite:///{STUDY_DB}"

SCALES = [
    ('tiny',   'tarware-tiny-5agvs-3pickers-partialobs-v1',       8),
    ('small',  'tarware-small-8agvs-4pickers-partialobs-v1',      12),
    ('medium', 'tarware-medium-12agvs-6pickers-partialobs-v1',    18),
    ('large',  'tarware-large-16agvs-8pickers-partialobs-v1',     24),
]

# Proximity reference (from Phase 1 calibration of main experiment)
PROXIMITY_RADII = {'tiny': 10, 'small': 8, 'medium': 6, 'large': 6}
PROXIMITY_DELIVERIES = {'tiny': 28.2, 'small': 32.8, 'medium': 48.0, 'large': 55.4}

# ── Bounded optimization domain ──────────────────────────────────────────────
# Continuous variables are optimized with Optuna TPE. Integer-like variables are
# enumerated as a small outer loop to keep the optimizer's domain continuous.
CONTINUOUS_BOUNDS = {
    'alpha':           (0.10, 0.30),
    'decay':           (0.94, 0.995),
    'beta':            (0.01, 0.10),
    'theta_form_end':  (0.30, 0.60),
    'theta_prune':     (0.002, 0.02),
    'target_deg_frac': (0.15, 0.55),
    'bcm_tau':         (0.85, 0.995),
    'reward_alpha':    (0.02, 0.20),
}

THETA_FORM_START_MIN_GAP = 0.10
THETA_FORM_START_MAX = 0.90

BASELINE_CONTINUOUS = {
    'alpha': 0.18,
    'decay': 0.98,
    'beta': 0.04,
    'theta_form_end': 0.45,
    'theta_form_start': 0.75,
    'theta_prune': 0.008,
    'target_deg_frac': 0.35,
    'bcm_tau': 0.95,
    'reward_alpha': 0.08,
}

INTEGER_SETTINGS = [
    {'grace_steps': grace_steps, 'k_slow': k_slow}
    for grace_steps, k_slow in itertools.product([10, 20, 40], [1, 3, 5])
]

# Fixed v2 params (not optimized)
FIXED = dict(
    theta_form_anneal=100,
    bcm_gain=0.5,
    neuromod_gain=0.4, neuromod_explore=0.10, neuromod_ema=0.03,
    pred_boost=0.4,
)
W_FLOOR = 0.25
EXPECTED_RATE = {'tiny': 0.07, 'small': 0.09, 'medium': 0.12, 'large': 0.14}


# ── Reuse run_episode from main experiment ───────────────────────────────────
def compute_coassign_jaccard(agents, assigned_agvs, assigned_pickers, N):
    targets = {}
    for a in agents:
        if a in assigned_agvs:
            m = assigned_agvs[a]; targets[a.id] = (m.location_x, m.location_y)
        elif a in assigned_pickers:
            m = assigned_pickers[a]; targets[a.id] = (m.location_x, m.location_y)
        else:
            targets[a.id] = None
    agent_ids = [a.id for a in agents]
    jac = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            ti = targets[agent_ids[i]]; tj = targets[agent_ids[j]]
            if ti and tj and ti == tj: jac[i,j] = jac[j,i] = 1.0
    return jac


def compute_pred_jac(agents, assigned_agvs, agent_idx, N):
    pred    = np.zeros((N, N))
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    for agv in [a for a in agents if a.type == AgentType.AGV]:
        if agv not in assigned_agvs: continue
        m = assigned_agvs[agv]
        if m.mission_type not in (MissionType.PICKING, MissionType.RETURNING): continue
        dists = [abs(p.x-m.location_x)+abs(p.y-m.location_y) for p in pickers]
        if not dists: continue
        nearest = pickers[int(np.argmin(dists))]
        i, j = agent_idx[agv.id], agent_idx[nearest.id]
        pred[i,j] = pred[j,i] = 1.0
    return pred


def run_morph(env_id, kw, T, seed, scale_name, w_floor=0.25):
    env = gym.make(env_id); u = env.unwrapped; env.reset(seed=seed)
    np.random.seed(seed)
    agents  = u.agents
    agvs    = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    N = len(agents)
    agent_idx    = {a.id: i for i, a in enumerate(agents)}
    coords_map   = {v: k for k, v in u.action_id_to_coords_map.items()}
    non_goal_ids = np.array([i for i, c in u.action_id_to_coords_map.items()
                              if (c[1], c[0]) not in u.goals])
    A = np.zeros((N,N)); W = np.zeros((N,N))
    morph_c = MORPH(N, **kw)
    assigned_agvs = OrderedDict(); assigned_pickers = OrderedDict()
    assigned_items = OrderedDict()
    deliveries = 0; link_sum = 0

    for t in range(T):
        rq = u.request_queue; gls = u.goals; actions = {a: 0 for a in agents}
        for item in rq:
            if item.id in assigned_items.values(): continue
            avail = [a for a in agvs if not a.busy and not a.carrying_shelf
                     and a not in assigned_agvs]
            if not avail: continue
            dists = [len(u.find_path((a.y,a.x),(item.y,item.x),a,care_for_agents=False))
                     for a in avail]
            best = avail[np.argmin(dists)]
            assigned_agvs[best] = Mission(MissionType.PICKING,
                                          coords_map[(item.y,item.x)], item.x,item.y, t)
            assigned_items[best] = item.id
        for agv in agvs:
            if agv in assigned_agvs:
                m = assigned_agvs[agv]
                assigned_agvs[agv].at_location = (agv.x==m.location_x and agv.y==m.location_y)
            if agv not in assigned_agvs or agv.busy: continue
            m = assigned_agvs[agv]
            if m.mission_type==MissionType.PICKING and m.at_location and agv.carrying_shelf:
                paths=[u.find_path((agv.y,agv.x),(y,x),agv,care_for_agents=False) for (x,y) in gls]
                bg=gls[np.argmin([len(p) for p in paths])]
                assigned_agvs[agv]=Mission(MissionType.DELIVERING,coords_map[(bg[1],bg[0])],bg[0],bg[1],t)
            elif m.mission_type==MissionType.DELIVERING and m.at_location and agv.carrying_shelf:
                empty=u.get_empty_shelf_information()
                eids=[i for i,e in zip(non_goal_ids,empty) if e>0]
                taken={m2.location_id for a2,m2 in assigned_agvs.items()
                       if a2 is not agv and m2.mission_type==MissionType.RETURNING}
                eids=[e for e in eids if e not in taken] or eids
                if eids:
                    elocs=[u.action_id_to_coords_map[i] for i in eids]
                    paths=[u.find_path((agv.y,agv.x),(y,x),agv,care_for_agents=False) for (y,x) in elocs]
                    be=eids[np.argmin([len(p) for p in paths])]; byx=u.action_id_to_coords_map[be]
                    assigned_agvs[agv]=Mission(MissionType.RETURNING,be,byx[1],byx[0],t)
            elif m.mission_type==MissionType.RETURNING and m.at_location and not agv.carrying_shelf:
                assigned_agvs.pop(agv); assigned_items.pop(agv,None)
            elif m.mission_type==MissionType.RETURNING and m.at_location and agv.carrying_shelf:
                empty=u.get_empty_shelf_information()
                eids=[i for i,e in zip(non_goal_ids,empty) if e>0]
                taken={m2.location_id for a2,m2 in assigned_agvs.items()
                       if a2 is not agv and m2.mission_type==MissionType.RETURNING}
                eids=[e for e in eids if e not in taken
                      and u.action_id_to_coords_map[e]!=(agv.y,agv.x)] or eids
                if eids:
                    elocs=[u.action_id_to_coords_map[i] for i in eids]
                    paths=[u.find_path((agv.y,agv.x),(y,x),agv,care_for_agents=False) for (y,x) in elocs]
                    be=eids[np.argmin([len(p) for p in paths])]; byx=u.action_id_to_coords_map[be]
                    assigned_agvs[agv]=Mission(MissionType.RETURNING,be,byx[1],byx[0],t)
        picker_locs={(mp.location_x,mp.location_y) for mp in assigned_pickers.values()}
        for agv,m in assigned_agvs.items():
            if m.mission_type not in [MissionType.PICKING,MissionType.RETURNING]: continue
            if (m.location_x,m.location_y) in picker_locs: continue
            avail_pickers=[p for p in pickers if p not in assigned_pickers]
            if not avail_pickers: continue
            dists=[abs(p.x-m.location_x)+abs(p.y-m.location_y) for p in avail_pickers]
            nat_picker=avail_pickers[int(np.argmin(dists))]
            idx_a=agent_idx[agv.id]; idx_p=agent_idx[nat_picker.id]
            w_ap=W[idx_a,idx_p]
            if np.random.random()<(w_floor+(1.0-w_floor)*w_ap):
                assigned_pickers[nat_picker]=Mission(MissionType.PICKING,
                                                     m.location_id,m.location_x,m.location_y,t)
                picker_locs.add((m.location_x,m.location_y))
        for picker in list(assigned_pickers.keys()):
            mp=assigned_pickers[picker]
            if not any(m.location_x==mp.location_x and m.location_y==mp.location_y
                       and m.mission_type in (MissionType.PICKING,MissionType.RETURNING)
                       for m in assigned_agvs.values()):
                assigned_pickers.pop(picker)
        for agv,m in assigned_agvs.items(): actions[agv]=m.location_id if not agv.busy else 0
        for p,m in assigned_pickers.items(): actions[p]=m.location_id
        result=env.step(list(actions[a] for a in agents)); rewards=result[1]
        step_del=sum(1 for rv in rewards[:u.num_agvs] if rv>0.5)
        deliveries+=step_del
        jac=compute_coassign_jaccard(agents,assigned_agvs,assigned_pickers,N)
        obs_mat=np.array([[a.x/u.grid_size[1],a.y/u.grid_size[0]] for a in agents],dtype=float)
        pred_jac=compute_pred_jac(agents,assigned_agvs,agent_idx,N)
        morph_c.step(obs_mat,jac,0.05,delivery=step_del,pred_jac=pred_jac)
        A=morph_c.A.copy(); W=morph_c.W.copy()
        link_sum+=morph_c.ne_h[-1]
    env.close()
    return deliveries, link_sum/T


def sample_continuous_params(trial):
    """Sample a bounded feasible continuous MORPH parameter vector."""
    params = {}
    for name, (lo, hi) in CONTINUOUS_BOUNDS.items():
        params[name] = trial.suggest_float(name, lo, hi)

    lo = params['theta_form_end'] + THETA_FORM_START_MIN_GAP
    params['theta_form_start'] = trial.suggest_float(
        'theta_form_start', lo, THETA_FORM_START_MAX
    )
    return params


def make_config(continuous_params, integer_setting):
    """Combine optimized and fixed settings into one full MORPH config."""
    return {
        **FIXED,
        **continuous_params,
        **integer_setting,
        'w_floor': W_FLOOR,
    }


def split_morph_kwargs(cfg):
    """Return response-floor and MORPH kwargs derived from a full config."""
    w_floor = cfg.get('w_floor', W_FLOOR)
    morph_params = {k: v for k, v in cfg.items() if k != 'w_floor'}
    return w_floor, morph_params


def scale_morph_kwargs(kw_base, scale_name, N_exp):
    return {
        **kw_base,
        'max_new': min(6, max(3, N_exp//6)),
        'expected_delivery_rate': EXPECTED_RATE[scale_name],
    }


def evaluate_config(cfg, search_scales, seeds):
    """Evaluate one bounded-optimization config on the search scales."""
    w_floor, kw_base = split_morph_kwargs(cfg)
    scores = {}
    links = {}
    for scale_name, env_id, N_exp in search_scales:
        kw = scale_morph_kwargs(kw_base, scale_name, N_exp)
        dels = []
        link_vals = []
        for seed in seeds:
            d, l = run_morph(env_id, kw, T, seed, scale_name, w_floor=w_floor)
            dels.append(d)
            link_vals.append(l)
        mean_d = np.mean(dels)
        prox = PROXIMITY_DELIVERIES[scale_name]
        gap = mean_d - prox        # positive = beats proximity
        scores[scale_name] = (mean_d, gap)
        links[scale_name] = np.mean(link_vals)

    combined = sum(v[1] for v in scores.values())
    return {'cfg': dict(cfg), 'scores': scores, 'links': links,
            'combined': combined}


def objective_for_setting(integer_setting, search_scales, seeds):
    def objective(trial):
        continuous_params = sample_continuous_params(trial)
        cfg = make_config(continuous_params, integer_setting)
        result = evaluate_config(cfg, search_scales, seeds)
        trial.set_user_attr('cfg', result['cfg'])
        trial.set_user_attr('scores', result['scores'])
        trial.set_user_attr('links', result['links'])
        return result['combined']

    return objective


def verify_config(best_cfg, scales, seeds):
    """Verify the selected full config on all scales."""
    w_floor, kw_base = split_morph_kwargs(best_cfg)
    verify_results = {}
    for scale_name, env_id, N_exp in scales:
        kw = scale_morph_kwargs(kw_base, scale_name, N_exp)
        dels = []
        links = []
        for seed in seeds:
            d, l = run_morph(env_id, kw, T, seed, scale_name, w_floor=w_floor)
            dels.append(d)
            links.append(l)
            print(f"  {scale_name:<8} seed={seed:4d}  del={d:3d}  links={l:.1f}")
        verify_results[scale_name] = {
            'mean': np.mean(dels),
            'std': np.std(dels),
            'links': np.mean(links),
        }
        prox = PROXIMITY_DELIVERIES[scale_name]
        gap = np.mean(dels) - prox
        print(f"  → {scale_name}: {np.mean(dels):.1f}±{np.std(dels):.1f}  "
              f"links={np.mean(links):.1f}  vs Proximity {prox} ({gap:+.1f})\n")
    return verify_results


def print_search_progress(idx, total, result):
    cfg = result['cfg']
    med_d, med_gap = result['scores']['medium']
    lrg_d, lrg_gap = result['scores']['large']
    marker = ' ★' if result['combined'] > 0 else ''
    print(f"[{idx:3d}/{total}] "
          f"wf={cfg['w_floor']:.2f} "
          f"tdf={cfg['target_deg_frac']:.2f} "
          f"ra={cfg['reward_alpha']:.2f} "
          f"bcm={cfg['bcm_gain']:.1f} "
          f"tp={cfg['theta_prune']:.3f} "
          f"| med={med_d:.1f}({med_gap:+.1f}) "
          f"lrg={lrg_d:.1f}({lrg_gap:+.1f}) "
          f"comb={result['combined']:+.1f}{marker}")


def study_name_for_setting(integer_setting):
    return (f"grace_{integer_setting['grace_steps']}"
            f"_kslow_{integer_setting['k_slow']}")


def trial_budget_by_setting(n_trials, integer_settings):
    base = n_trials // len(integer_settings)
    remainder = n_trials % len(integer_settings)
    return [
        base + (1 if idx < remainder else 0)
        for idx, _ in enumerate(integer_settings)
    ]


def create_or_load_study(integer_setting):
    sampler = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,
        group=True,
    )
    return optuna.create_study(
        study_name=study_name_for_setting(integer_setting),
        storage=STORAGE_URL,
        direction='maximize',
        sampler=sampler,
        load_if_exists=True,
    )


def enqueue_baseline_if_needed(study, integer_setting):
    if len(study.trials) > 0:
        return
    study.enqueue_trial(dict(BASELINE_CONTINUOUS))


def completed_results_from_study(study):
    results = []
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if {'cfg', 'scores', 'links'} <= set(trial.user_attrs):
            results.append({
                'cfg': trial.user_attrs['cfg'],
                'scores': trial.user_attrs['scores'],
                'links': trial.user_attrs['links'],
                'combined': trial.value,
                'number': trial.number,
                'study_name': study.study_name,
            })
    return results


def main():
    # ── Phase 1: Mixed bounded Bayesian optimization ───────────────────────
    print("="*65)
    print("MORPH v2 Bounded Bayesian Hyperparameter Optimization")
    print("="*65)

    search_scales = [s for s in SCALES if s[0] in ('medium','large')]
    budgets = trial_budget_by_setting(N_TRIALS, INTEGER_SETTINGS)
    total_runs = N_TRIALS * len(search_scales) * len(SEARCH_SEEDS)
    print(f"Optimizing {N_TRIALS} Optuna trials across "
          f"{len(INTEGER_SETTINGS)} integer settings")
    print(f"Each trial runs {len(search_scales)} scales × "
          f"{len(SEARCH_SEEDS)} seeds = {len(search_scales)*len(SEARCH_SEEDS)} episodes")
    print(f"Total planned search episodes = {total_runs}")
    print(f"Study DB: {STUDY_DB}\n")

    for setting, budget in zip(INTEGER_SETTINGS, budgets):
        study = create_or_load_study(setting)
        enqueue_baseline_if_needed(study, setting)
        completed = len(study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.COMPLETE,),
        ))
        n_to_run = max(0, budget - completed)
        print(f"{study.study_name}: target={budget} completed={completed} "
              f"remaining={n_to_run}")

        def log_completed_trial(study, trial):
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            if {'cfg', 'scores', 'links'} <= set(trial.user_attrs):
                result = {
                    'cfg': trial.user_attrs['cfg'],
                    'scores': trial.user_attrs['scores'],
                    'links': trial.user_attrs['links'],
                    'combined': trial.value,
                }
                print_search_progress(trial.number + 1, budget, result)

        if n_to_run > 0:
            study.optimize(
                objective_for_setting(setting, search_scales, SEARCH_SEEDS),
                n_trials=n_to_run,
                callbacks=[log_completed_trial],
            )

    # ── Top configs ─────────────────────────────────────────────────────────
    results = []
    for setting in INTEGER_SETTINGS:
        study = create_or_load_study(setting)
        results.extend(completed_results_from_study(study))

    if not results:
        raise RuntimeError("No completed Optuna trials found.")

    results.sort(key=lambda x: x['combined'], reverse=True)
    print(f"\n{'='*65}")
    print("TOP 5 OPTUNA TRIALS (combined gap vs Proximity, medium+large)")
    print(f"{'='*65}")
    for r in results[:5]:
        print(f"  {r['cfg']}  →  combined={r['combined']:+.1f}  "
              f"med={r['scores']['medium'][0]:.1f}  "
              f"lrg={r['scores']['large'][0]:.1f}")

    best_cfg = dict(results[0]['cfg'])
    best_w_floor = best_cfg.get('w_floor', W_FLOOR)
    print(f"\nBest config: {best_cfg}")

    # ── Phase 2: Verify best config on all 4 scales, 5 seeds ────────────────
    print(f"\n{'='*65}")
    print("Phase 2: Verification — all scales, 5 seeds")
    print(f"{'='*65}")
    verify_results = verify_config(best_cfg, SCALES, VERIFY_SEEDS)

    # ── Save best config ────────────────────────────────────────────────────
    out = {
        'best_cfg': best_cfg,
        'verify_results': verify_results,
        'all_search_results': results,
        'proximity_ref': PROXIMITY_DELIVERIES,
        'optimizer': 'optuna_tpe',
        'bounds': {
            **CONTINUOUS_BOUNDS,
            'theta_form_start': (
                'theta_form_end + 0.10',
                THETA_FORM_START_MAX,
            ),
        },
        'integer_settings': INTEGER_SETTINGS,
        'n_trials': N_TRIALS,
        'study_db': STUDY_DB,
    }
    with open(os.path.join(RES_DIR, 'hyperparam_search.pkl'), 'wb') as f:
        pickle.dump(out, f)

    print("\nSummary vs Proximity:")
    print(f"{'Scale':<8} {'MORPH v2 tuned':>16} {'Proximity':>10} {'Gap':>8}")
    print("-"*45)
    for scale_name, _, _ in SCALES:
        r = verify_results[scale_name]
        prox = PROXIMITY_DELIVERIES[scale_name]
        gap = r['mean'] - prox
        print(f"  {scale_name:<6}  {r['mean']:>8.1f}±{r['std']:<5.1f}  "
              f"{prox:>8.1f}  {gap:>+7.1f}")
    print(f"\nBest W_FLOOR: {best_w_floor}")
    print("\nSearch complete. Best config saved to results/hyperparam_search.pkl")


if __name__ == '__main__':
    main()
