"""
hyperparam_search.py
====================
Grid search to close the Proximity gap at medium and large scales.

Diagnosis: MORPH v2 uses MORE links than Proximity but fewer deliveries.
Problem is link quality, not cold-start. Search strategy:
  - Lower target_deg_frac  → fewer, more selective links
  - Raise reward_alpha     → delivery signal dominates Hebbian noise
  - Raise bcm_gain         → stale links decay faster
  - Raise theta_prune      → prune marginal links more aggressively

Runs on medium + large, 3 seeds each (fast sweep), then verifies top
configs on all 4 scales with 5 seeds.

Usage:
  python scripts/hyperparam_search.py
"""
import sys, os, warnings, itertools, pickle
import numpy as np
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tarware, gymnasium as gym
from tarware.heuristic import AgentType, Mission, MissionType
from morph import MORPH
from collections import OrderedDict

W_FLOOR = 0.25
T       = 400
SEARCH_SEEDS  = [42, 7, 123]          # fast sweep
VERIFY_SEEDS  = [42, 7, 123, 999, 2024]

SCALES = [
    ('tiny',   'tarware-tiny-5agvs-3pickers-partialobs-v1',       8),
    ('small',  'tarware-small-8agvs-4pickers-partialobs-v1',      12),
    ('medium', 'tarware-medium-12agvs-6pickers-partialobs-v1',    18),
    ('large',  'tarware-large-16agvs-8pickers-partialobs-v1',     24),
]

# Proximity reference (from Phase 1 calibration of main experiment)
PROXIMITY_RADII = {'tiny': 10, 'small': 8, 'medium': 6, 'large': 6}
PROXIMITY_DELIVERIES = {'tiny': 28.2, 'small': 32.8, 'medium': 48.0, 'large': 55.4}

# ── Search grid ──────────────────────────────────────────────────────────────
# Hypothesis: lower target_deg + stronger reward + faster BCM decay = better
GRID = {
    'target_deg_frac':  [0.20, 0.25, 0.30],
    'reward_alpha':     [0.10, 0.15, 0.20],
    'bcm_gain':         [0.5,  0.8,  1.2],
    'theta_prune':      [0.008, 0.012],
}

# Fixed v2 params (not searched)
FIXED = dict(
    alpha=0.18, beta=0.04, decay=0.98,
    theta_form_start=0.75, theta_form_end=0.45, theta_form_anneal=100,
    grace_steps=20, k_slow=3,
    bcm_tau=0.95,
    neuromod_gain=0.4, neuromod_explore=0.10, neuromod_ema=0.03,
    pred_boost=0.4,
)
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


def run_morph(env_id, kw, T, seed, scale_name):
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
            if np.random.random()<(W_FLOOR+(1.0-W_FLOOR)*w_ap):
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


# ── Phase 1: Grid search on medium+large, 3 seeds ───────────────────────────
print("="*65)
print("MORPH v2 Hyperparameter Search")
print("="*65)

search_scales = [s for s in SCALES if s[0] in ('medium','large')]
param_names   = sorted(GRID.keys())
param_values  = [GRID[k] for k in param_names]
configs       = list(itertools.product(*param_values))
print(f"Searching {len(configs)} configs × {len(search_scales)} scales × "
      f"{len(SEARCH_SEEDS)} seeds = {len(configs)*len(search_scales)*len(SEARCH_SEEDS)} runs\n")

results = []
for ci, vals in enumerate(configs):
    cfg = dict(zip(param_names, vals))
    kw_base = {**FIXED, **cfg}
    scale_scores = {}
    for scale_name, env_id, N_exp in search_scales:
        kw = {**kw_base,
              'max_new': min(6, max(3, N_exp//6)),
              'expected_delivery_rate': EXPECTED_RATE[scale_name]}
        dels = []
        for seed in SEARCH_SEEDS:
            d, l = run_morph(env_id, kw, T, seed, scale_name)
            dels.append(d)
        mean_d = np.mean(dels)
        prox   = PROXIMITY_DELIVERIES[scale_name]
        gap    = mean_d - prox        # positive = beats proximity
        scale_scores[scale_name] = (mean_d, gap)

    # Combined score: sum of gaps across medium+large
    combined = sum(v[1] for v in scale_scores.values())
    results.append({'cfg': cfg, 'scores': scale_scores, 'combined': combined})

    med_d, med_gap = scale_scores['medium']
    lrg_d, lrg_gap = scale_scores['large']
    marker = ' ★' if combined > 0 else ''
    print(f"[{ci+1:3d}/{len(configs)}] "
          f"tdf={cfg['target_deg_frac']:.2f} "
          f"ra={cfg['reward_alpha']:.2f} "
          f"bcm={cfg['bcm_gain']:.1f} "
          f"tp={cfg['theta_prune']:.3f} "
          f"| med={med_d:.1f}({med_gap:+.1f}) "
          f"lrg={lrg_d:.1f}({lrg_gap:+.1f}) "
          f"comb={combined:+.1f}{marker}")

# ── Top configs ──────────────────────────────────────────────────────────────
results.sort(key=lambda x: x['combined'], reverse=True)
print(f"\n{'='*65}")
print("TOP 5 CONFIGS (combined gap vs Proximity, medium+large)")
print(f"{'='*65}")
for r in results[:5]:
    print(f"  {r['cfg']}  →  combined={r['combined']:+.1f}  "
          f"med={r['scores']['medium'][0]:.1f}  lrg={r['scores']['large'][0]:.1f}")

best_cfg = results[0]['cfg']
print(f"\nBest config: {best_cfg}")

# ── Phase 2: Verify best config on all 4 scales, 5 seeds ────────────────────
print(f"\n{'='*65}")
print("Phase 2: Verification — all scales, 5 seeds")
print(f"{'='*65}")
verify_results = {}
for scale_name, env_id, N_exp in SCALES:
    kw = {**FIXED, **best_cfg,
          'max_new': min(6, max(3, N_exp//6)),
          'expected_delivery_rate': EXPECTED_RATE[scale_name]}
    dels = []; links = []
    for seed in VERIFY_SEEDS:
        d, l = run_morph(env_id, kw, T, seed, scale_name)
        dels.append(d); links.append(l)
        print(f"  {scale_name:<8} seed={seed:4d}  del={d:3d}  links={l:.1f}")
    verify_results[scale_name] = {'mean': np.mean(dels), 'std': np.std(dels),
                                  'links': np.mean(links)}
    prox = PROXIMITY_DELIVERIES[scale_name]
    gap  = np.mean(dels) - prox
    print(f"  → {scale_name}: {np.mean(dels):.1f}±{np.std(dels):.1f}  "
          f"links={np.mean(links):.1f}  vs Proximity {prox} ({gap:+.1f})\n")

# ── Save best config ─────────────────────────────────────────────────────────
out = {'best_cfg': best_cfg, 'verify_results': verify_results,
       'all_search_results': results, 'proximity_ref': PROXIMITY_DELIVERIES}
with open(os.path.join(ROOT,'results','hyperparam_search.pkl'), 'wb') as f:
    pickle.dump(out, f)

print("\nSummary vs Proximity:")
print(f"{'Scale':<8} {'MORPH v2 tuned':>16} {'Proximity':>10} {'Gap':>8}")
print("-"*45)
for scale_name, _, _ in SCALES:
    r    = verify_results[scale_name]
    prox = PROXIMITY_DELIVERIES[scale_name]
    gap  = r['mean'] - prox
    print(f"  {scale_name:<6}  {r['mean']:>8.1f}±{r['std']:<5.1f}  "
          f"{prox:>8.1f}  {gap:>+7.1f}")
print("\nSearch complete. Best config saved to results/hyperparam_search.pkl")
