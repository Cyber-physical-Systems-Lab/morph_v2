"""
Task Distribution Shift Experiment
====================================
Demonstrates MORPH v2's adaptive rewiring under a repeated spatial task regime.

Protocol
--------
    Phase 1 (steps   1–400):  AGVs serve only LEFT-half shelves  (x < grid_width/2)
    Phase 2 (steps 401–800):  AGVs serve only RIGHT-half shelves (x >= grid_width/2)
    Phase 3 (steps 801–1200):  AGVs return to LEFT-half shelves   (x < grid_width/2)
    Phase 4 (steps 1201–1600):  AGVs serve only RIGHT-half shelves    (y < grid_height/2)
    Phase 5 (steps 1601–2000):  AGVs serve only LEFT-half shelves     (y >= grid_height/2)
    Phase 6 (steps 2001–2400):  AGVs return to RIGHT-half shelves     (y < grid_height/2)

Same environment, same seed, same agents throughout — only WHICH half of the
warehouse generates demand changes.  This forces a genuine rewiring challenge:

    • Proximity  — links follow agent positions.  Agents that crowded left in
                                 Phase 1 must move right and then up/down, rebuilding spatial
                                 connections each time. No memory of prior coordination.
    • TSG        — no memory; instantly reflects current co-assignments.
    • MORPH v2   — W carries Phase-1 coordination memory (left-side pairs),
                                 then reuses it when the left regime returns in Phase 3 and
                                 when the right regime returns in Phase 6. BCM prunes stale links;
                                 reward-modulation builds right-side and left-side patterns.
                                 Neuromodulation detects delivery-rate dips and increases
                                 exploration.
    • MORPH v2 reset — weights zeroed at each regime change (upper bound for
                                         adaptation speed, lower bound for memory benefit).

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
ENV_ID   = 'tarware-large-16agvs-8pickers-partialobs-v1'  # large scale with partial observability
T_PHASE  = 400          # steps per phase
T_TOTAL  = T_PHASE * 6
WINDOW   = 50           # rolling delivery window
W_FLOOR  = 0.25
PROX_R   = 8            # calibrated proximity radius for medium

# SEEDS    = [42, 7, 123, 999, 2024]
SEEDS = [
    13, 42, 87, 123, 256,
    512, 777, 1024, 1337, 2021,
    4096, 5555, 6789, 8192, 9999,
    12345, 22222, 31415, 42424, 65536
]

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
    'morph_v2_reset': 'MORPH v2 (reset at regime changes)',
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


def rolling_rate(step_del, window=WINDOW):
    """Causal rolling mean deliveries over the past `window` steps."""
    csum = np.concatenate(([0], np.cumsum(step_del)))
    out  = np.zeros(len(step_del), dtype=float)
    for t in range(len(step_del)):
        lo = max(0, t + 1 - window)
        out[t] = (csum[t + 1] - csum[lo]) / (t + 1 - lo)
    return out


def post_shift_rate(step_del, t_k, t_end, max_window=WINDOW):
    """
    Forward-looking rate within a single phase, free of pre-shift contamination.
    For offset δ, use the mean of deliveries in [t_k, t_k+δ) when δ ≤ max_window,
    then a sliding max_window window thereafter.
    """
    seg = step_del[t_k:t_end].astype(float)
    csum = np.concatenate(([0.0], np.cumsum(seg)))
    L = len(seg)
    out = np.zeros(L)
    for d in range(1, L + 1):
        lo = max(0, d - max_window)
        out[d - 1] = (csum[d] - csum[lo]) / (d - lo)
    return out


def compute_recovery_metrics(step_del, t_phase=T_PHASE, window=WINDOW, eps=0.10):
    """
    Compute τ_recover for every regime return in the LRLRLR cycle.

    Baseline comes from the previous same-zone phase; tau is the first
    post-shift step where post_shift_rate >= (1-eps) * baseline.
    Non-return phases are skipped.
    """
    metrics = {}

    n_phases = len(step_del) // t_phase
    for phase_idx in range(1, n_phases):
        phase_num = phase_idx + 1
        prior_same_zone = [p for p in range(phase_idx) if (p % 2) == (phase_idx % 2)]

        if not prior_same_zone:
            continue

        ps_phase = post_shift_rate(step_del, phase_idx * t_phase,
                                   (phase_idx + 1) * t_phase, window)
        base_phase = prior_same_zone[-1]
        base_rate = post_shift_rate(step_del, base_phase * t_phase,
                                    (base_phase + 1) * t_phase, window)
        baseline = float(base_rate[-1])
        hits = np.where(ps_phase >= (1 - eps) * baseline)[0]
        tau = float(hits[0] + 1) if (len(hits) > 0 and baseline > 1e-9) else float(t_phase)

        metrics[f'phase{phase_num}'] = {
            'tau': tau,
            'baseline': baseline,
            'censored': bool(len(hits) == 0 or baseline <= 1e-9),
        }
    return metrics


# ── Two-phase episode runner ──────────────────────────────────────────────────

def run_shift_episode(cond, seed):
    """
    Run a single 2400-step episode with a six-phase spatial regime.

    Phase 1: AGVs prefer left-half items   (item.x < half_x)
    Phase 2: AGVs prefer right-half items  (item.x >= half_x)
    Phase 3: AGVs prefer left-half items   (item.x < half_x)
    Phase 4: AGVs prefer right-half items  (item.x >= half_x)
    Phase 5: AGVs prefer left-half items   (item.x < half_x)
    Phase 6: AGVs prefer right-half items  (item.x >= half_x)

    Returns
    -------
    step_deliveries : (T_TOTAL,) int array — deliveries per step
    zone_counts     : dict with left/right assignment counts per phase
    recovery_steps  : dict with tau_recover for the two regime returns
    """
    env = gym.make(ENV_ID)
    u   = env.unwrapped
    env.reset(seed=seed)
    np.random.seed(seed)

    agents  = u.agents
    n_agents = len(agents)
    agvs    = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    agent_idx    = {a.id: i for i, a in enumerate(agents)}
    coords_map   = {v: k for k, v in u.action_id_to_coords_map.items()}
    non_goal_ids = np.array([i for i, c in u.action_id_to_coords_map.items()
                              if (c[1], c[0]) not in u.goals])

    # Spatial split thresholds
    half_x = u.grid_size[1] // 2   # grid_size = (height, width)
    half_y = u.grid_size[0] // 2

    # Verify shelves exist in both halves
    all_items_x = [u.action_id_to_coords_map[i][1]   # col = x
                   for i in non_goal_ids]
    n_left  = sum(1 for x in all_items_x if x <  half_x)
    n_right = sum(1 for x in all_items_x if x >= half_x)

    # MORPH initialisation
    morph_c = MORPH(n_agents, **_V2_KW) if cond in ('morph_v2', 'morph_v2_reset') else None

    assigned_agvs    = OrderedDict()
    assigned_pickers = OrderedDict()
    assigned_items   = OrderedDict()
    A = np.zeros((n_agents, n_agents))
    W = np.zeros((n_agents, n_agents))

    step_deliveries = np.zeros(T_TOTAL, dtype=int)
    zone_counts = {
        'ph1_left': 0, 'ph1_right': 0,
        'ph2_left': 0, 'ph2_right': 0,
        'ph3_left': 0, 'ph3_right': 0,
        'ph4_left': 0, 'ph4_right': 0,
        'ph5_left': 0, 'ph5_right': 0,
        'ph6_left': 0, 'ph6_right': 0,
    }

    for t in range(T_TOTAL):
        phase = t // T_PHASE

        # Reset MORPH weights at each regime change for morph_v2_reset
        if t in (T_PHASE, 2 * T_PHASE, 3 * T_PHASE, 4 * T_PHASE, 5 * T_PHASE) and cond == 'morph_v2_reset':
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
        prefer_left = (phase % 2 == 0)
        phase_key = f'ph{phase + 1}'
        preferred = [item for item in rq
                     if item.id not in assigned_items.values()
                     and ((item.x < half_x) if prefer_left else (item.x >= half_x))]

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
            key = phase_key + ('_left' if item.x < half_x else '_right')
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
                                         n_agents)
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
                                                assigned_pickers, n_agents)
            obs_mat  = np.array([[a.x / u.grid_size[1], a.y / u.grid_size[0]]
                                  for a in agents], dtype=float)
            pred_jac = compute_pred_jac(agents, assigned_agvs, agent_idx, n_agents)
            morph_c.step(obs_mat, jac, 0.05,
                         delivery=step_del, pred_jac=pred_jac)
            A = morph_c.A.copy()
            W = morph_c.W.copy()

    recovery_steps = compute_recovery_metrics(step_deliveries, t_phase=T_PHASE, window=WINDOW)

    env.close()
    return step_deliveries, zone_counts, n_left, n_right, half_x, recovery_steps


# ── Windowed delivery rate ────────────────────────────────────────────────────

def windowed_rate(step_del, window=WINDOW):
    n = len(step_del) // window
    return np.array([step_del[i*window:(i+1)*window].sum() for i in range(n)])


def make_time2rec_figure(recovery_results):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITIONS))
    width = 0.18
    phases = [('phase3', 'Return to LEFT (Phase 3)'),
              ('phase4', 'Return to RIGHT (Phase 4)'),
              ('phase5', 'Return to LEFT (Phase 5)'),
              ('phase6', 'Return to RIGHT (Phase 6)')]

    for i, (phase_key, label) in enumerate(phases):
        means = []
        errs = []
        for cond in CONDITIONS:
            vals = [v[phase_key]['tau'] for v in recovery_results[cond] if v[phase_key] is not None]
            if vals:
                means.append(float(np.mean(vals)))
                errs.append(float(np.std(vals)))
            else:
                means.append(np.nan)
                errs.append(0.0)
        offset = (i - (len(phases) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=errs, capsize=4, alpha=0.9, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], rotation=15, ha='right')
    ax.set_ylabel('Recovery step')
    ax.set_title('Time to Recover After Regime Return')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'time2rec.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


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
    for boundary in [T_PHASE, 2 * T_PHASE, 3 * T_PHASE, 4 * T_PHASE, 5 * T_PHASE]:
        ax.axvline(boundary, color='black', ls=':', lw=1.4)
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    ax.text(T_PHASE + 6, ymax - (ymax - ymin) * 0.04,
            'Task zones\nflip', fontsize=9, va='top', color='black')

    # Phase background shading
    ax.axvspan(0,             T_PHASE,          alpha=0.04, color='blue',   label='_left zone active')
    ax.axvspan(T_PHASE,       2 * T_PHASE,      alpha=0.04, color='orange', label='_right zone active')
    ax.axvspan(2 * T_PHASE,   3 * T_PHASE,      alpha=0.04, color='blue',   label='_left zone active again')
    ax.axvspan(3 * T_PHASE,   4 * T_PHASE,      alpha=0.04, color='blue',   label='_left zone active (y-axis)')
    ax.axvspan(4 * T_PHASE,   5 * T_PHASE,      alpha=0.04, color='orange', label='_right zone active (y-axis)')
    ax.axvspan(5 * T_PHASE,   T_TOTAL,          alpha=0.04, color='blue',   label='_left zone active again (y-axis)')

    # Phase labels
    ax.text(T_PHASE * 0.5,  ymin + (ymax-ymin)*0.04, 'Phase 1\n(left-half tasks)',
            fontsize=9, ha='center', color='steelblue', alpha=0.8)
    ax.text(T_PHASE * 1.5,  ymin + (ymax-ymin)*0.04, 'Phase 2\n(right-half tasks)',
            fontsize=9, ha='center', color='darkorange', alpha=0.8)
    ax.text(T_PHASE * 2.5,  ymin + (ymax-ymin)*0.04, 'Phase 3\n(return to left)',
            fontsize=9, ha='center', color='steelblue', alpha=0.8)
    ax.text(T_PHASE * 3.5,  ymin + (ymax-ymin)*0.04, 'Phase 4\n(left-half tasks, y-axis)',
            fontsize=9, ha='center', color='steelblue', alpha=0.8)
    ax.text(T_PHASE * 4.5,  ymin + (ymax-ymin)*0.04, 'Phase 5\n(right-half tasks, y-axis)',
            fontsize=9, ha='center', color='darkorange', alpha=0.8)
    ax.text(T_PHASE * 5.5,  ymin + (ymax-ymin)*0.04, 'Phase 6\n(return to left, y-axis)',
            fontsize=9, ha='center', color='steelblue', alpha=0.8)

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
    print(f"Phase 3: LEFT-half shelves again (x < grid_width/2)")
    print(f"Phase 4: RIGHT-half shelves only (x >= grid_width/2)")
    print(f"Phase 5: LEFT-half shelves only (x < grid_width/2)")
    print(f"Phase 6: RIGHT-half shelves again (x >= grid_width/2)")
    print(f"Seeds: {SEEDS}")
    print()

    results = {cond: [] for cond in CONDITIONS}
    recovery_results = {cond: [] for cond in CONDITIONS}

    for i, seed in enumerate(SEEDS):
        print(f"[Run {i+1}/{len(SEEDS)}]  seed={seed}")

        # Print zone info from first condition (same for all)
        step_del, zc, n_left, n_right, half_x, rec = run_shift_episode(
            CONDITIONS[0], seed)
        results[CONDITIONS[0]].append(step_del)
        recovery_results[CONDITIONS[0]].append(rec)
        ph1 = step_del[:T_PHASE].sum(); ph2 = step_del[T_PHASE:2 * T_PHASE].sum()
        ph3 = step_del[2 * T_PHASE:3 * T_PHASE].sum(); ph4 = step_del[3 * T_PHASE:4 * T_PHASE].sum()
        ph5 = step_del[4 * T_PHASE:5 * T_PHASE].sum(); ph6 = step_del[5 * T_PHASE:].sum()
        print(f"  grid half_x={half_x}  shelves: left={n_left}  right={n_right}")
        print(f"  assignments: ph1 L={zc['ph1_left']} R={zc['ph1_right']} | "
              f"ph2 L={zc['ph2_left']} R={zc['ph2_right']} | "
              f"ph3 L={zc['ph3_left']} R={zc['ph3_right']} | "
              f"ph4 L={zc['ph4_left']} R={zc['ph4_right']} | "
              f"ph5 L={zc['ph5_left']} R={zc['ph5_right']} | "
              f"ph6 L={zc['ph6_left']} R={zc['ph6_right']}")
        print(f"  {CONDITIONS[0]:22s}  phase1={ph1:3d}  phase2={ph2:3d}  phase3={ph3:3d}  phase4={ph4:3d}  phase5={ph5:3d}  phase6={ph6:3d}  rec3={rec['phase3']['tau']:.0f}  rec4={rec['phase4']['tau']:.0f}  rec5={rec['phase5']['tau']:.0f}  rec6={rec['phase6']['tau']:.0f}")

        for cond in CONDITIONS[1:]:
            step_del, zc, _, _, _, rec = run_shift_episode(cond, seed)
            results[cond].append(step_del)
            recovery_results[cond].append(rec)
            ph1 = step_del[:T_PHASE].sum()
            ph2 = step_del[T_PHASE:2 * T_PHASE].sum()
            ph3 = step_del[2 * T_PHASE:3 * T_PHASE].sum()
            ph4 = step_del[3 * T_PHASE:4 * T_PHASE].sum()
            ph5 = step_del[4 * T_PHASE:5 * T_PHASE].sum()
            ph6 = step_del[5 * T_PHASE:].sum()
            print(f"  {cond:22s}  phase1={ph1:3d}  phase2={ph2:3d}  phase3={ph3:3d}  phase4={ph4:3d}  phase5={ph5:3d}  phase6={ph6:3d}  rec3={rec['phase3']['tau']:.0f}  rec4={rec['phase4']['tau']:.0f}  rec5={rec['phase5']['tau']:.0f}  rec6={rec['phase6']['tau']:.0f}")
        print()

    # Stack
    for cond in CONDITIONS:
        results[cond] = np.stack(results[cond], axis=0)

    # Summary
    print("=" * 65)
    print("SUMMARY")
    print(f"{'Condition':<24}  {'P1':>6}  {'P2':>6}  {'P3':>6}  {'P4':>6}  {'P5':>6}  {'P6':>6}  {'Δ63':>7}  {'R3':>6}  {'R4':>6}  {'R5':>6}  {'R6':>6}")
    print("-" * 110)
    for cond in CONDITIONS:
        arr = results[cond]
        ph1 = arr[:, :T_PHASE].sum(axis=1).mean()
        ph2 = arr[:, T_PHASE:2 * T_PHASE].sum(axis=1).mean()
        ph3 = arr[:, 2 * T_PHASE:3 * T_PHASE].sum(axis=1).mean()
        ph4 = arr[:, 3 * T_PHASE:4 * T_PHASE].sum(axis=1).mean()
        ph5 = arr[:, 4 * T_PHASE:5 * T_PHASE].sum(axis=1).mean()
        ph6 = arr[:, 5 * T_PHASE:].sum(axis=1).mean()
        r3_vals = [v['phase3']['tau'] for v in recovery_results[cond] if v['phase3'] is not None]
        r4_vals = [v['phase4']['tau'] for v in recovery_results[cond] if v['phase4'] is not None]
        r5_vals = [v['phase5']['tau'] for v in recovery_results[cond] if v['phase5'] is not None]
        r6_vals = [v['phase6']['tau'] for v in recovery_results[cond] if v['phase6'] is not None]
        r3 = float(np.mean(r3_vals)) if r3_vals else float('nan')
        r4 = float(np.mean(r4_vals)) if r4_vals else float('nan')
        r5 = float(np.mean(r5_vals)) if r5_vals else float('nan')
        r6 = float(np.mean(r6_vals)) if r6_vals else float('nan')
        print(f"  {cond:<22}  {ph1:6.1f}  {ph2:6.1f}  {ph3:6.1f}  {ph4:6.1f}  {ph5:6.1f}  {ph6:6.1f}  {ph6-ph3:+7.1f}  {r3:6.1f}  {r4:6.1f}  {r5:6.1f}  {r6:6.1f}")

    pkl_path = os.path.join(RES_DIR, 'shift_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {pkl_path}")

    make_figure(results)
    make_time2rec_figure(recovery_results)
    print("\nShift experiment complete.")


if __name__ == '__main__':
    main()
