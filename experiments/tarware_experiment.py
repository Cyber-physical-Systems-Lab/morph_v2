"""
MORPH v2 on TA-RWARE: Scale Study + Ablation
=============================================
Runs nine conditions across four warehouse scales (N=8→24).

Conditions
----------
  no_coord          — A=0, never respond (lower bound)
  proximity         — A_ij=1 if Manhattan dist ≤ r, soft gating
  tsg               — instantaneous co-assignment graph, soft gating, no memory
  morph_v1          — MORPH v1 (all v2 mechanisms disabled)
  morph_v2_no_bcm   — v2 without BCM metaplasticity
  morph_v2_no_reward— v2 without reward-modulated plasticity
  morph_v2_no_neuro — v2 without neuromodulation
  morph_v2          — full MORPH v2 (all mechanisms enabled)
  full              — all-to-all, always respond (upper bound)

Outputs
-------
  results/experiment_results.pkl     — all data for figure generation
  results/step_csv/<scale>_<cond>_seed<s>.csv
  results/summary_table.csv          — human-readable summary

Usage
-----
  python experiments/tarware_experiment.py
"""
import sys, os, warnings, pickle, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES_DIR = os.path.join(ROOT, 'results')
CSV_DIR = os.path.join(RES_DIR, 'step_csv')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

import tarware, gymnasium as gym
from tarware.heuristic import AgentType, Mission, MissionType
from morph import MORPH
from collections import OrderedDict

# ── Experiment config ────────────────────────────────────────────────────────
SEEDS   = [42, 7, 123, 999, 2024]
T_SCALE = 400
W_FLOOR = 0.25

SCALES = [
    ('tiny',   'tarware-tiny-5agvs-3pickers-partialobs-v1',       8),
    ('small',  'tarware-small-8agvs-4pickers-partialobs-v1',      12),
    ('medium', 'tarware-medium-12agvs-6pickers-partialobs-v1',    18),
    ('large',  'tarware-large-16agvs-8pickers-partialobs-v1',     24),
]

CONDITIONS = [
    'no_coord',
    'proximity',
    'tsg',
    'morph_v1',
    'morph_v2_no_bcm',
    'morph_v2_no_reward',
    'morph_v2_no_neuro',
    'morph_v2',
    'full',
]

# ── MORPH hyperparameters ────────────────────────────────────────────────────
# Shared v1/v2 base (same as v1 best config)
_V1_BASE = dict(
    alpha=0.18, beta=0.04, decay=0.98,
    theta_form_start=0.75, theta_form_end=0.45, theta_form_anneal=100,
    theta_prune=0.008, target_deg_frac=0.35,
    grace_steps=20, k_slow=3,
)
# v2 new mechanism params
_V2_NEW = dict(
    bcm_tau=0.95,   bcm_gain=0.5,
    reward_alpha=0.08,
    neuromod_gain=0.4, neuromod_explore=0.10,
    neuromod_ema=0.03,
    pred_boost=0.4,
)
# Expected delivery rate per step (used by neuromod; scale-dependent)
_EXPECTED_RATE = {'tiny': 0.07, 'small': 0.09, 'medium': 0.12, 'large': 0.14}


def morph_kw(cond, N, scale_name):
    """Return MORPH constructor kwargs for a given condition."""
    kw = dict(_V1_BASE)
    kw['max_new'] = min(6, max(3, N // 6))

    if cond == 'morph_v1':
        # Disable all v2 mechanisms
        kw.update(bcm_gain=0.0, reward_alpha=0.0,
                  neuromod_gain=0.0, neuromod_explore=0.0, pred_boost=0.0)

    elif cond == 'morph_v2_no_bcm':
        kw.update(_V2_NEW)
        kw['expected_delivery_rate'] = _EXPECTED_RATE[scale_name]
        kw['bcm_gain'] = 0.0

    elif cond == 'morph_v2_no_reward':
        kw.update(_V2_NEW)
        kw['expected_delivery_rate'] = _EXPECTED_RATE[scale_name]
        kw['reward_alpha'] = 0.0

    elif cond == 'morph_v2_no_neuro':
        kw.update(_V2_NEW)
        kw['expected_delivery_rate'] = _EXPECTED_RATE[scale_name]
        kw['neuromod_gain'] = 0.0
        kw['neuromod_explore'] = 0.0

    elif cond == 'morph_v2':
        kw.update(_V2_NEW)
        kw['expected_delivery_rate'] = _EXPECTED_RATE[scale_name]

    return kw


# ── Helper functions ─────────────────────────────────────────────────────────

def compute_coassign_jaccard(agents, assigned_agvs, assigned_pickers, N):
    targets    = {}
    for a in agents:
        if a in assigned_agvs:
            m = assigned_agvs[a];   targets[a.id] = (m.location_x, m.location_y)
        elif a in assigned_pickers:
            m = assigned_pickers[a]; targets[a.id] = (m.location_x, m.location_y)
        else:
            targets[a.id] = None
    agent_ids = [a.id for a in agents]
    jac = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ti = targets[agent_ids[i]]; tj = targets[agent_ids[j]]
            if ti and tj and ti == tj:
                jac[i, j] = jac[j, i] = 1.0
    return jac


def compute_pred_jac(agents, assigned_agvs, agent_idx, N):
    """
    Predictive formation hint: A_ij=1 if an AGV is newly assigned and
    agent j is the nearest available picker to its target.
    Returns an (N, N) hint matrix.
    """
    pred = np.zeros((N, N))
    agvs    = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    for agv in agvs:
        if agv not in assigned_agvs:
            continue
        m = assigned_agvs[agv]
        if m.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
            continue
        # Find nearest picker by Manhattan distance
        dists = [abs(p.x - m.location_x) + abs(p.y - m.location_y) for p in pickers]
        if not dists:
            continue
        nearest = pickers[int(np.argmin(dists))]
        i = agent_idx[agv.id]
        j = agent_idx[nearest.id]
        pred[i, j] = pred[j, i] = 1.0
    return pred


def compute_tsg_matrix(agents, assigned_agvs, assigned_pickers, N):
    """Instantaneous task-shared graph (no memory, no plasticity)."""
    return compute_coassign_jaccard(agents, assigned_agvs, assigned_pickers, N)


def compute_proximity_matrix(agents, r):
    """A_ij=1 if Manhattan distance between agents i and j ≤ r."""
    N = len(agents)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if abs(agents[i].x - agents[j].x) + abs(agents[i].y - agents[j].y) <= r:
                A[i, j] = A[j, i] = 1.0
    return A


def calibrate_proximity(env_id, T, seed, target_links):
    """Find r whose mean active links best matches target_links."""
    import gymnasium as gym
    print(f"    Calibrating proximity radius (target={target_links:.0f} links):")
    best_r, best_diff = 4, float('inf')
    for r in [2, 3, 4, 5, 6, 8, 10]:
        _, _, lh, _ = run_episode(env_id, 'proximity', None, T, seed, r=r,
                                  scale_name='tiny')
        diff = abs(lh.mean() - target_links)
        print(f"      r={r}: mean_links={lh.mean():.1f}  diff={diff:.1f}")
        if diff < best_diff:
            best_diff, best_r = diff, r
    print(f"    → best r={best_r}")
    return best_r


# ── Core episode runner ──────────────────────────────────────────────────────

def run_episode(env_id, cond, kw, T, seed, r=None, scale_name='tiny'):
    env = gym.make(env_id)
    u   = env.unwrapped
    env.reset(seed=seed)
    np.random.seed(seed)

    agents  = u.agents
    agvs    = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    N = len(agents)
    agent_idx    = {a.id: i for i, a in enumerate(agents)}
    coords_map   = {v: k for k, v in u.action_id_to_coords_map.items()}
    non_goal_ids = np.array([i for i, c in u.action_id_to_coords_map.items()
                              if (c[1], c[0]) not in u.goals])

    # Initialise topology
    if cond == 'full':
        A = np.ones((N, N)) - np.eye(N)
        W = np.ones((N, N)) - np.eye(N)
        morph_c = None
    elif cond == 'no_coord':
        A = np.zeros((N, N)); W = np.zeros((N, N)); morph_c = None
    elif cond == 'proximity':
        A = np.zeros((N, N)); W = np.zeros((N, N)); morph_c = None
    elif cond == 'tsg':
        A = np.zeros((N, N)); W = np.zeros((N, N)); morph_c = None
    else:
        # All morph variants
        A = np.zeros((N, N)); W = np.zeros((N, N))
        morph_c = MORPH(N, **(kw or morph_kw(cond, N, scale_name)))

    assigned_agvs    = OrderedDict()
    assigned_pickers = OrderedDict()
    assigned_items   = OrderedDict()
    cum_rew = 0.0; deliveries = 0
    reward_h = []; deliv_h = []; link_h = []; w_mean_h = []

    for t in range(T):
        rq  = u.request_queue
        gls = u.goals
        actions = {a: 0 for a in agents}

        # AGV assignment: nearest available AGV to each queued item
        for item in rq:
            if item.id in assigned_items.values():
                continue
            avail = [a for a in agvs
                     if not a.busy and not a.carrying_shelf
                     and a not in assigned_agvs]
            if not avail:
                continue
            dists = [len(u.find_path((a.y, a.x), (item.y, item.x), a,
                                     care_for_agents=False)) for a in avail]
            best = avail[np.argmin(dists)]
            assigned_agvs[best] = Mission(MissionType.PICKING,
                                          coords_map[(item.y, item.x)],
                                          item.x, item.y, t)
            assigned_items[best] = item.id

        # AGV state machine (all 7 coordination fixes)
        for agv in agvs:
            if agv in assigned_agvs:
                m = assigned_agvs[agv]
                if agv.x == m.location_x and agv.y == m.location_y:
                    assigned_agvs[agv].at_location = True
                else:
                    assigned_agvs[agv].at_location = False
            if agv not in assigned_agvs or agv.busy:
                continue
            m = assigned_agvs[agv]

            if m.mission_type == MissionType.PICKING and m.at_location and agv.carrying_shelf:
                paths = [u.find_path((agv.y, agv.x), (y, x), agv,
                                     care_for_agents=False) for (x, y) in gls]
                bg = gls[np.argmin([len(p) for p in paths])]
                assigned_agvs[agv] = Mission(MissionType.DELIVERING,
                                             coords_map[(bg[1], bg[0])],
                                             bg[0], bg[1], t)

            elif m.mission_type == MissionType.DELIVERING and m.at_location and agv.carrying_shelf:
                empty = u.get_empty_shelf_information()
                eids  = [i for i, e in zip(non_goal_ids, empty) if e > 0]
                taken = {m2.location_id for a2, m2 in assigned_agvs.items()
                         if a2 is not agv and m2.mission_type == MissionType.RETURNING}
                eids  = [e for e in eids if e not in taken] or eids
                if eids:
                    elocs = [u.action_id_to_coords_map[i] for i in eids]
                    paths = [u.find_path((agv.y, agv.x), (y, x), agv,
                                         care_for_agents=False) for (y, x) in elocs]
                    be  = eids[np.argmin([len(p) for p in paths])]
                    byx = u.action_id_to_coords_map[be]
                    assigned_agvs[agv] = Mission(MissionType.RETURNING, be,
                                                 byx[1], byx[0], t)

            elif (m.mission_type == MissionType.RETURNING
                  and m.at_location and not agv.carrying_shelf):
                assigned_agvs.pop(agv)
                assigned_items.pop(agv, None)

            elif (m.mission_type == MissionType.RETURNING
                  and m.at_location and agv.carrying_shelf):
                empty = u.get_empty_shelf_information()
                eids  = [i for i, e in zip(non_goal_ids, empty) if e > 0]
                taken = {m2.location_id for a2, m2 in assigned_agvs.items()
                         if a2 is not agv and m2.mission_type == MissionType.RETURNING}
                eids  = [e for e in eids
                         if e not in taken
                         and u.action_id_to_coords_map[e] != (agv.y, agv.x)] or eids
                if eids:
                    elocs = [u.action_id_to_coords_map[i] for i in eids]
                    paths = [u.find_path((agv.y, agv.x), (y, x), agv,
                                         care_for_agents=False) for (y, x) in elocs]
                    be  = eids[np.argmin([len(p) for p in paths])]
                    byx = u.action_id_to_coords_map[be]
                    assigned_agvs[agv] = Mission(MissionType.RETURNING, be,
                                                 byx[1], byx[0], t)

        # Dynamic topology for tsg/proximity
        if cond == 'tsg':
            A = compute_tsg_matrix(agents, assigned_agvs, assigned_pickers, N)
        elif cond == 'proximity':
            A = compute_proximity_matrix(agents, r or 4)

        # Picker assignment (1 picker per location cap)
        picker_locs = {(mp.location_x, mp.location_y)
                       for mp in assigned_pickers.values()}
        for agv, m in assigned_agvs.items():
            if m.mission_type not in [MissionType.PICKING, MissionType.RETURNING]:
                continue
            if (m.location_x, m.location_y) in picker_locs:
                continue
            avail_pickers = [p for p in pickers if p not in assigned_pickers]
            if not avail_pickers:
                continue
            dists = [abs(p.x - m.location_x) + abs(p.y - m.location_y)
                     for p in avail_pickers]
            nat_picker = avail_pickers[int(np.argmin(dists))]
            idx_a = agent_idx[agv.id]
            idx_p = agent_idx[nat_picker.id]

            if cond == 'no_coord':
                respond = False
            elif cond == 'full':
                respond = True
            elif cond in ('morph_v1', 'morph_v2_no_bcm', 'morph_v2_no_reward',
                          'morph_v2_no_neuro', 'morph_v2'):
                w_ap = W[idx_a, idx_p]
                respond = np.random.random() < (W_FLOOR + (1.0 - W_FLOOR) * w_ap)
            else:
                # proximity, tsg: soft gating using binary A
                a_ap = A[idx_a, idx_p]
                respond = np.random.random() < (W_FLOOR + (1.0 - W_FLOOR) * a_ap)

            if respond:
                assigned_pickers[nat_picker] = Mission(MissionType.PICKING,
                                                       m.location_id,
                                                       m.location_x, m.location_y, t)
                picker_locs.add((m.location_x, m.location_y))

        # Sticky picker release
        for picker in list(assigned_pickers.keys()):
            mp = assigned_pickers[picker]
            still_needed = any(
                m.location_x == mp.location_x and m.location_y == mp.location_y
                and m.mission_type in (MissionType.PICKING, MissionType.RETURNING)
                for m in assigned_agvs.values()
            )
            if not still_needed:
                assigned_pickers.pop(picker)

        # Build actions
        for agv, m in assigned_agvs.items():
            actions[agv] = m.location_id if not agv.busy else 0
        for p, m in assigned_pickers.items():
            actions[p] = m.location_id

        # Step environment
        result    = env.step(list(actions[a] for a in agents))
        rewards   = result[1]
        step_del  = sum(1 for rv in rewards[:u.num_agvs] if rv > 0.5)
        cum_rew  += sum(rewards)
        deliveries += step_del

        # MORPH update
        if morph_c is not None:
            jac     = compute_coassign_jaccard(agents, assigned_agvs,
                                               assigned_pickers, N)
            obs_mat = np.array([[a.x / u.grid_size[1], a.y / u.grid_size[0]]
                                 for a in agents], dtype=float)
            pred_jac = (compute_pred_jac(agents, assigned_agvs, agent_idx, N)
                        if cond == 'morph_v2' else None)
            morph_c.step(obs_mat, jac, 0.05,
                         delivery=step_del, pred_jac=pred_jac)
            A = morph_c.A.copy()
            W = morph_c.W.copy()
            n_links = morph_c.ne_h[-1]
            w_m     = morph_c.W_mean_h[-1]
        elif cond == 'full':
            n_links = N * (N - 1) // 2; w_m = 1.0
        elif cond == 'no_coord':
            n_links = 0; w_m = 0.0
        else:
            n_links = int(A.sum() // 2); w_m = 0.0

        reward_h.append(cum_rew)
        deliv_h.append(deliveries)
        link_h.append(n_links)
        w_mean_h.append(w_m)

    env.close()
    return (np.array(reward_h), np.array(deliv_h),
            np.array(link_h),   np.array(w_mean_h))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("MORPH v2  ·  TA-RWARE Scale Study + Ablation")
    print("=" * 65)
    print(f"Conditions : {CONDITIONS}")
    print(f"Seeds      : {SEEDS}")
    print(f"T          : {T_SCALE}  |  Scales: {[s[0] for s in SCALES]}")
    print()

    all_results    = {}
    proximity_radii = {}

    # ── Phase 1: Calibrate proximity radii ──────────────────────────────────
    print("Phase 1: Proximity Radius Calibration")
    print("-" * 40)
    for scale_name, env_id, N_exp in SCALES:
        print(f"  [{scale_name.upper()}]")
        # Target: match mean links of MORPH v2 (run one seed to estimate)
        _, _, lh, _ = run_episode(env_id, 'morph_v2',
                                  morph_kw('morph_v2', N_exp, scale_name),
                                  T_SCALE, SEEDS[0], scale_name=scale_name)
        target = lh.mean()
        print(f"    MORPH v2 reference links = {target:.1f}")
        r = calibrate_proximity(env_id, T_SCALE, SEEDS[0], target)
        proximity_radii[scale_name] = r
    print(f"\nProximity radii: {proximity_radii}\n")

    # ── Phase 2: Full experiment ─────────────────────────────────────────────
    print("Phase 2: Full Scale Study")
    print("-" * 40)
    for scale_name, env_id, N_exp in SCALES:
        print(f"\n[{scale_name.upper()}  N={N_exp}]")
        all_results[scale_name] = {}
        for cond in CONDITIONS:
            kw = (morph_kw(cond, N_exp, scale_name)
                  if cond not in ('no_coord', 'proximity', 'tsg', 'full')
                  else None)
            r  = proximity_radii[scale_name] if cond == 'proximity' else None
            all_results[scale_name][cond] = []
            for seed in SEEDS:
                out = run_episode(env_id, cond, kw, T_SCALE, seed, r=r,
                                  scale_name=scale_name)
                all_results[scale_name][cond].append(out)
                _, d, l, _ = out
                print(f"  {cond:22s}  seed={seed:4d}  "
                      f"del={d[-1]:4.0f}  links={l.mean():5.1f}")

                # Save per-step CSV
                csv_path = os.path.join(CSV_DIR,
                    f"{scale_name}_{cond}_seed{seed}.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'deliveries_cumulative',
                                     'active_links', 'condition', 'seed', 'scale'])
                    for step in range(T_SCALE):
                        writer.writerow([step, int(d[step]), int(l[step]),
                                         cond, seed, scale_name])

            # Per-condition summary
            mn = np.mean([all_results[scale_name][cond][s][1][-1]
                          for s in range(len(SEEDS))])
            lk = np.mean([all_results[scale_name][cond][s][2].mean()
                          for s in range(len(SEEDS))])
            print(f"  → {cond}: mean_del={mn:.1f}  mean_links={lk:.1f}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'condition':<24}" + "".join(
        f"  {s[0]:>6}" for s in SCALES) + "   avg_links"
    print(header)
    print("-" * 80)

    summary_rows = []
    for cond in CONDITIONS:
        row_dels  = []
        row_links = []
        for scale_name, _, N_exp in SCALES:
            mn = np.mean([all_results[scale_name][cond][s][1][-1]
                          for s in range(len(SEEDS))])
            lk = np.mean([all_results[scale_name][cond][s][2].mean()
                          for s in range(len(SEEDS))])
            row_dels.append(mn)
            row_links.append(lk)
        avg_links = np.mean(row_links)
        del_str   = "".join(f"  {d:>6.1f}" for d in row_dels)
        print(f"  {cond:<22}{del_str}   {avg_links:.1f}")
        summary_rows.append({'condition': cond,
                              **{s[0]: row_dels[i] for i, s in enumerate(SCALES)},
                              'avg_links': avg_links})

    # Save summary CSV
    sum_path = os.path.join(RES_DIR, 'summary_table.csv')
    with open(sum_path, 'w', newline='') as f:
        fields = ['condition'] + [s[0] for s in SCALES] + ['avg_links']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved: {sum_path}")

    # Save full results pkl
    pkl_path = os.path.join(RES_DIR, 'experiment_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'all_results':      all_results,
            'T':                T_SCALE,
            'SEEDS':            SEEDS,
            'SCALES':           SCALES,
            'CONDITIONS':       CONDITIONS,
            'proximity_radii':  proximity_radii,
        }, f)
    print(f"Saved: {pkl_path}")
    print("\nExperiment complete.")


if __name__ == '__main__':
    main()
