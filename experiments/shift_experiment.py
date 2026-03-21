"""
Task Distribution Shift Experiment
====================================
Demonstrates MORPH v2's adaptive rewiring under a genuine spatial task shift.

Protocol
--------
  Phase 1 (steps   1–400):  AGVs serve only LEFT-half shelves  (x < grid_width/2)
  Shift   (step  401):       task zone flips
  Phase 2 (steps 401–800):  AGVs serve only RIGHT-half shelves (x >= grid_width/2)

Same environment, same seed, same agents throughout — only WHICH half of the
warehouse generates demand changes.  This forces a genuine rewiring challenge:

  • Proximity  — links follow agent positions.  Agents that crowded left in
                 Phase 1 must move right and build new spatial connections.
                 No memory of who coordinated in Phase 1.
  • TSG        — no memory; instantly reflects current co-assignments.
  • MORPH v2   — W carries Phase-1 coordination memory (left-side pairs).
                 BCM prunes stale links; reward-modulation builds right-side
                 patterns. Neuromodulation detects the delivery-rate dip and
                 increases exploration.
  • MORPH v2 reset — weights zeroed at shift (upper bound for adaptation speed,
                     lower bound for memory benefit).

If a half has no items in the request queue the fallback is to serve any
available item (prevents complete starvation while keeping the bias strong).

Output
------
  figures/shift_adaptability.png
  results/shift_results.pkl
"""
import sys, os, warnings, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

import tarware, gymnasium as gym
from tarware.heuristic import AgentType, Mission, MissionType
from morph import MORPH

# ── Config ────────────────────────────────────────────────────────────────────
ENV_ID   = 'tarware-medium-12agvs-6pickers-partialobs-v1'
N_AGENTS = 18
T_PHASE  = 400          # steps per phase
T_TOTAL  = T_PHASE * 2
WINDOW   = 50           # rolling delivery window
W_FLOOR  = 0.25
PROX_R   = 6            # calibrated proximity radius for medium

SEEDS    = [42, 7, 123, 999, 2024]

CONDITIONS = ['proximity', 'tsg', 'morph_v2', 'morph_v2_reset']

COLORS = {
    'proximity':      '#2196F3',
    'tsg':            '#FF9800',
    'morph_v2':       '#4CAF50',
    'morph_v2_reset': '#9C27B0',
}
LABELS = {
    'proximity':      'Proximity (no memory)',
    'tsg':            'TSG (no memory)',
    'morph_v2':       'MORPH v2 (adaptive rewiring)',
    'morph_v2_reset': 'MORPH v2 (reset at shift)',
}

# MORPH v2 best config
_V2_KW = dict(
    alpha=0.18, beta=0.04, decay=0.98,
    theta_form_start=0.75, theta_form_end=0.45, theta_form_anneal=100,
    theta_prune=0.008, target_deg_frac=0.35,
    grace_steps=20, k_slow=3, max_new=5,
    bcm_tau=0.95, bcm_gain=0.5,
    reward_alpha=0.08,
    neuromod_gain=0.4, neuromod_explore=0.10,
    neuromod_ema=0.03, expected_delivery_rate=0.12,
    pred_boost=0.4,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        for j in range(i + 1, N):
            ti = targets[agent_ids[i]]; tj = targets[agent_ids[j]]
            if ti and tj and ti == tj:
                jac[i, j] = jac[j, i] = 1.0
    return jac


def compute_pred_jac(agents, assigned_agvs, agent_idx, N):
    pred    = np.zeros((N, N))
    agvs    = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    for agv in agvs:
        if agv not in assigned_agvs:
            continue
        m = assigned_agvs[agv]
        if m.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
            continue
        dists = [abs(p.x - m.location_x) + abs(p.y - m.location_y) for p in pickers]
        if not dists:
            continue
        nearest = pickers[int(np.argmin(dists))]
        i = agent_idx[agv.id]; j = agent_idx[nearest.id]
        pred[i, j] = pred[j, i] = 1.0
    return pred


def compute_proximity_matrix(agents, r):
    N = len(agents)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if abs(agents[i].x - agents[j].x) + abs(agents[i].y - agents[j].y) <= r:
                A[i, j] = A[j, i] = 1.0
    return A


# ── Two-phase episode runner ──────────────────────────────────────────────────

def run_shift_episode(cond, seed):
    """
    Run a single 800-step episode with a spatial task-zone flip at step 400.

    Phase 1: AGVs prefer left-half items  (item.x < half_x)
    Phase 2: AGVs prefer right-half items (item.x >= half_x)

    Returns
    -------
    step_deliveries : (T_TOTAL,) int array — deliveries per step
    zone_counts     : dict with left/right assignment counts per phase
    """
    env = gym.make(ENV_ID)
    u   = env.unwrapped
    env.reset(seed=seed)
    np.random.seed(seed)

    agents  = u.agents
    agvs    = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    agent_idx    = {a.id: i for i, a in enumerate(agents)}
    coords_map   = {v: k for k, v in u.action_id_to_coords_map.items()}
    non_goal_ids = np.array([i for i, c in u.action_id_to_coords_map.items()
                              if (c[1], c[0]) not in u.goals])

    # Spatial split threshold
    half_x = u.grid_size[1] // 2   # grid_size = (height, width)

    # Verify shelves exist in both halves
    all_items_x = [u.action_id_to_coords_map[i][1]   # col = x
                   for i in non_goal_ids]
    n_left  = sum(1 for x in all_items_x if x <  half_x)
    n_right = sum(1 for x in all_items_x if x >= half_x)

    # MORPH initialisation
    morph_c = MORPH(N_AGENTS, **_V2_KW) if cond in ('morph_v2', 'morph_v2_reset') else None

    assigned_agvs    = OrderedDict()
    assigned_pickers = OrderedDict()
    assigned_items   = OrderedDict()
    A = np.zeros((N_AGENTS, N_AGENTS))
    W = np.zeros((N_AGENTS, N_AGENTS))

    step_deliveries = np.zeros(T_TOTAL, dtype=int)
    zone_counts = {'ph1_left': 0, 'ph1_right': 0,
                   'ph2_left': 0, 'ph2_right': 0}

    for t in range(T_TOTAL):
        phase = 0 if t < T_PHASE else 1   # 0 = left half, 1 = right half

        # Reset MORPH weights at shift point for morph_v2_reset
        if t == T_PHASE and cond == 'morph_v2_reset':
            morph_c.W[:]         = 0.0
            morph_c.A[:]         = 0.0
            morph_c.theta_bcm[:] = 0.0
            morph_c.link_age[:]  = 0.0
            morph_c.jac_last[:]  = 0.0
            morph_c.H[:]         = _V2_KW['target_deg_frac']
            A[:] = 0.0; W[:] = 0.0

        rq      = u.request_queue
        gls     = u.goals
        actions = {a: 0 for a in agents}

        # ── AGV assignment with spatial preference ────────────────────────────
        # Preferred items: left half in Phase 1, right half in Phase 2
        preferred = [item for item in rq
                     if item.id not in assigned_items.values()
                     and (item.x <  half_x if phase == 0 else item.x >= half_x)]
        fallback  = [item for item in rq
                     if item.id not in assigned_items.values()
                     and item not in preferred]
        # Serve preferred first, fall back if preferred queue is empty
        item_queue = preferred if preferred else fallback

        for item in item_queue:
            avail = [a for a in agvs
                     if not a.busy and not a.carrying_shelf
                     and a not in assigned_agvs]
            if not avail:
                break
            dists = [len(u.find_path((a.y, a.x), (item.y, item.x), a,
                                      care_for_agents=False)) for a in avail]
            best = avail[np.argmin(dists)]
            assigned_agvs[best] = Mission(MissionType.PICKING,
                                          coords_map[(item.y, item.x)],
                                          item.x, item.y, t)
            assigned_items[best] = item.id
            # Track zone
            key = ('ph1' if phase == 0 else 'ph2') + ('_left' if item.x < half_x else '_right')
            zone_counts[key] += 1

        # ── AGV state machine ─────────────────────────────────────────────────
        for agv in agvs:
            if agv in assigned_agvs:
                m = assigned_agvs[agv]
                assigned_agvs[agv].at_location = (agv.x == m.location_x and
                                                   agv.y == m.location_y)
            if agv not in assigned_agvs or agv.busy:
                continue
            m = assigned_agvs[agv]

            if (m.mission_type == MissionType.PICKING
                    and m.at_location and agv.carrying_shelf):
                paths = [u.find_path((agv.y, agv.x), (y, x), agv,
                                     care_for_agents=False) for (x, y) in gls]
                bg = gls[np.argmin([len(p) for p in paths])]
                assigned_agvs[agv] = Mission(MissionType.DELIVERING,
                                             coords_map[(bg[1], bg[0])],
                                             bg[0], bg[1], t)

            elif (m.mission_type == MissionType.DELIVERING
                  and m.at_location and agv.carrying_shelf):
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
                eids  = ([e for e in eids if e not in taken
                           and u.action_id_to_coords_map[e] != (agv.y, agv.x)]
                         or eids)
                if eids:
                    elocs = [u.action_id_to_coords_map[i] for i in eids]
                    paths = [u.find_path((agv.y, agv.x), (y, x), agv,
                                          care_for_agents=False) for (y, x) in elocs]
                    be  = eids[np.argmin([len(p) for p in paths])]
                    byx = u.action_id_to_coords_map[be]
                    assigned_agvs[agv] = Mission(MissionType.RETURNING, be,
                                                 byx[1], byx[0], t)

        # ── Dynamic topology ──────────────────────────────────────────────────
        if cond == 'tsg':
            A = compute_coassign_jaccard(agents, assigned_agvs, assigned_pickers,
                                         N_AGENTS)
        elif cond == 'proximity':
            A = compute_proximity_matrix(agents, PROX_R)

        # ── Picker assignment ─────────────────────────────────────────────────
        picker_locs = {(mp.location_x, mp.location_y)
                       for mp in assigned_pickers.values()}
        for agv, m in assigned_agvs.items():
            if m.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
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

            if cond in ('morph_v2', 'morph_v2_reset'):
                w_ap    = W[idx_a, idx_p]
                respond = np.random.random() < (W_FLOOR + (1 - W_FLOOR) * w_ap)
            else:
                a_ap    = A[idx_a, idx_p]
                respond = np.random.random() < (W_FLOOR + (1 - W_FLOOR) * a_ap)

            if respond:
                assigned_pickers[nat_picker] = Mission(
                    MissionType.PICKING, m.location_id,
                    m.location_x, m.location_y, t)
                picker_locs.add((m.location_x, m.location_y))

        # ── Sticky picker release ─────────────────────────────────────────────
        for picker in list(assigned_pickers.keys()):
            mp = assigned_pickers[picker]
            still_needed = any(
                m.location_x == mp.location_x and m.location_y == mp.location_y
                and m.mission_type in (MissionType.PICKING, MissionType.RETURNING)
                for m in assigned_agvs.values()
            )
            if not still_needed:
                assigned_pickers.pop(picker)

        # ── Build actions and step ────────────────────────────────────────────
        for agv2, m in assigned_agvs.items():
            actions[agv2] = m.location_id if not agv2.busy else 0
        for p, m in assigned_pickers.items():
            actions[p] = m.location_id

        result   = env.step(list(actions[a] for a in agents))
        rewards  = result[1]
        step_del = sum(1 for rv in rewards[:u.num_agvs] if rv > 0.5)
        step_deliveries[t] = step_del

        # ── MORPH update ──────────────────────────────────────────────────────
        if morph_c is not None:
            jac      = compute_coassign_jaccard(agents, assigned_agvs,
                                                assigned_pickers, N_AGENTS)
            obs_mat  = np.array([[a.x / u.grid_size[1], a.y / u.grid_size[0]]
                                  for a in agents], dtype=float)
            pred_jac = compute_pred_jac(agents, assigned_agvs, agent_idx, N_AGENTS)
            morph_c.step(obs_mat, jac, 0.05,
                         delivery=step_del, pred_jac=pred_jac)
            A = morph_c.A.copy()
            W = morph_c.W.copy()

    env.close()
    return step_deliveries, zone_counts, n_left, n_right, half_x


# ── Windowed delivery rate ────────────────────────────────────────────────────

def windowed_rate(step_del, window=WINDOW):
    n = len(step_del) // window
    return np.array([step_del[i*window:(i+1)*window].sum() for i in range(n)])


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    n_windows = T_TOTAL // WINDOW
    x = np.arange(n_windows) * WINDOW + WINDOW // 2

    for cond in CONDITIONS:
        arr   = results[cond]   # (n_seeds, T_TOTAL)
        rates = np.array([windowed_rate(arr[s]) for s in range(arr.shape[0])])
        mean  = rates.mean(axis=0)
        std   = rates.std(axis=0)
        col   = COLORS[cond]
        lw    = 2.5 if cond == 'morph_v2' else 1.8
        ls    = '-'  if cond in ('proximity', 'morph_v2') else '--'
        ax.plot(x, mean, color=col, lw=lw, ls=ls, label=LABELS[cond])
        ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.15)

    # Shift marker
    ax.axvline(T_PHASE, color='black', ls=':', lw=1.8)
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    ax.text(T_PHASE + 6, ymax - (ymax - ymin) * 0.04,
            'Task zone\nflips →', fontsize=9, va='top', color='black')

    # Phase background shading
    ax.axvspan(0,       T_PHASE, alpha=0.04, color='blue',  label='_left zone active')
    ax.axvspan(T_PHASE, T_TOTAL, alpha=0.04, color='orange', label='_right zone active')

    # Phase labels
    ax.text(T_PHASE * 0.5,  ymin + (ymax-ymin)*0.04, 'Phase 1\n(left-half tasks)',
            fontsize=9, ha='center', color='steelblue', alpha=0.8)
    ax.text(T_PHASE * 1.5,  ymin + (ymax-ymin)*0.04, 'Phase 2\n(right-half tasks)',
            fontsize=9, ha='center', color='darkorange', alpha=0.8)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(f'Deliveries per {WINDOW} steps', fontsize=12)
    ax.set_title('Adaptability Under Spatial Task Distribution Shift (medium scale)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0, T_TOTAL)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'shift_adaptability.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Task Distribution Shift — Spatial Left/Right Zone Flip")
    print("=" * 65)
    print(f"Scale: medium ({ENV_ID})")
    print(f"T per phase: {T_PHASE}  |  Total: {T_TOTAL}")
    print(f"Phase 1: LEFT-half shelves only (x < grid_width/2)")
    print(f"Phase 2: RIGHT-half shelves only (x >= grid_width/2)")
    print(f"Seeds: {SEEDS}")
    print()

    results = {cond: [] for cond in CONDITIONS}

    for i, seed in enumerate(SEEDS):
        print(f"[Run {i+1}/{len(SEEDS)}]  seed={seed}")

        # Print zone info from first condition (same for all)
        step_del, zc, n_left, n_right, half_x = run_shift_episode(
            CONDITIONS[0], seed)
        results[CONDITIONS[0]].append(step_del)
        ph1 = step_del[:T_PHASE].sum(); ph2 = step_del[T_PHASE:].sum()
        print(f"  grid half_x={half_x}  shelves: left={n_left}  right={n_right}")
        print(f"  assignments: ph1 L={zc['ph1_left']} R={zc['ph1_right']} | "
              f"ph2 L={zc['ph2_left']} R={zc['ph2_right']}")
        print(f"  {CONDITIONS[0]:22s}  phase1={ph1:3d}  phase2={ph2:3d}")

        for cond in CONDITIONS[1:]:
            step_del, zc, _, _, _ = run_shift_episode(cond, seed)
            results[cond].append(step_del)
            ph1 = step_del[:T_PHASE].sum(); ph2 = step_del[T_PHASE:].sum()
            print(f"  {cond:22s}  phase1={ph1:3d}  phase2={ph2:3d}")
        print()

    # Stack
    for cond in CONDITIONS:
        results[cond] = np.stack(results[cond], axis=0)

    # Summary
    print("=" * 65)
    print("SUMMARY")
    print(f"{'Condition':<24}  {'Phase1':>7}  {'Phase2':>7}  {'Δ':>7}")
    print("-" * 55)
    for cond in CONDITIONS:
        arr = results[cond]
        ph1 = arr[:, :T_PHASE].sum(axis=1).mean()
        ph2 = arr[:, T_PHASE:].sum(axis=1).mean()
        print(f"  {cond:<22}  {ph1:7.1f}  {ph2:7.1f}  {ph2-ph1:+7.1f}")

    pkl_path = os.path.join(RES_DIR, 'shift_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    make_figure(results)
    print("\nShift experiment complete.")


if __name__ == '__main__':
    main()
