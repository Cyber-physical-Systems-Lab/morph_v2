"""
emergent_analysis.py
====================
Four emergent-behaviour analyses for MORPH v2 on TA-RWARE (medium scale).

Runs a single 800-step episode with full state recording, then generates:

  figures/emergent_w_vs_distance.png     — W[i,j] vs mean Manhattan distance
  figures/emergent_w_evolution.png       — W matrix heatmaps at 6 time snapshots
  figures/emergent_neuromod.png          — η time series with delivery annotations
  figures/emergent_link_persistence.png  — histogram of link lifetimes

Usage
-----
  python scripts/emergent_analysis.py
"""
import sys, os, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
from collections import OrderedDict

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, ROOT)

import tarware, gymnasium as gym
from tarware.heuristic import AgentType, Mission, MissionType
from morph import MORPH

# ── Config ────────────────────────────────────────────────────────────────────
SCALE_NAME = 'medium'
ENV_ID     = 'tarware-medium-12agvs-6pickers-partialobs-v1'
N_AGENTS   = 18
T          = 800
SEED       = 42
W_FLOOR    = 0.25

# MORPH v2 params (same as main experiment)
MORPH_KW = dict(
    alpha=0.18, beta=0.04, decay=0.98,
    theta_form_start=0.75, theta_form_end=0.45, theta_form_anneal=100,
    theta_prune=0.008, target_deg_frac=0.35,
    grace_steps=20, k_slow=3,
    max_new=3,
    bcm_tau=0.95, bcm_gain=0.5,
    reward_alpha=0.08,
    neuromod_gain=0.4, neuromod_explore=0.10,
    neuromod_ema=0.03, expected_delivery_rate=0.12,
    pred_boost=0.4,
)

SNAPSHOT_STEPS = [0, 100, 200, 400, 600, 800]

BG = 'white'

# ── Helpers copied from tarware_experiment ────────────────────────────────────

def compute_coassign_jaccard(agents, assigned_agvs, assigned_pickers, N):
    targets = {}
    for a in agents:
        if a in assigned_agvs:
            m = assigned_agvs[a];    targets[a.id] = (m.location_x, m.location_y)
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
        dists   = [abs(p.x - m.location_x) + abs(p.y - m.location_y) for p in pickers]
        if not dists:
            continue
        nearest = pickers[int(np.argmin(dists))]
        i = agent_idx[agv.id];  j = agent_idx[nearest.id]
        pred[i, j] = pred[j, i] = 1.0
    return pred


# ═════════════════════════════════════════════════════════════════════════════
# Episode runner with full state recording
# ═════════════════════════════════════════════════════════════════════════════

def run_recorded_episode():
    print(f"Running recorded episode: {SCALE_NAME}, T={T}, seed={SEED}")

    env = gym.make(ENV_ID)
    u   = env.unwrapped
    env.reset(seed=SEED)
    np.random.seed(SEED)

    agents   = u.agents
    agvs     = [a for a in agents if a.type == AgentType.AGV]
    pickers  = [a for a in agents if a.type == AgentType.PICKER]
    N = len(agents)
    agent_idx    = {a.id: i for i, a in enumerate(agents)}
    agent_types  = ['AGV' if a.type == AgentType.AGV else 'Picker' for a in agents]
    coords_map   = {v: k for k, v in u.action_id_to_coords_map.items()}
    non_goal_ids = np.array([i for i, c in u.action_id_to_coords_map.items()
                              if (c[1], c[0]) not in u.goals])

    morph_c = MORPH(N, **MORPH_KW)

    assigned_agvs    = OrderedDict()
    assigned_pickers = OrderedDict()
    assigned_items   = OrderedDict()

    # ── Recording buffers ─────────────────────────────────────────────────────
    pos_history   = np.zeros((T, N, 2), dtype=np.float32)  # (t, agent, xy)
    W_snapshots   = {}                                       # step → W matrix
    A_history     = np.zeros((T, N, N), dtype=np.uint8)    # adjacency per step
    delivery_h    = np.zeros(T, dtype=int)                  # deliveries per step

    deliveries = 0

    for t in range(T):
        # Record positions
        for i, a in enumerate(agents):
            pos_history[t, i] = [a.x, a.y]

        rq  = u.request_queue
        gls = u.goals
        actions = {a: 0 for a in agents}

        # AGV assignment
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

        # AGV state machine
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

        # Picker assignment
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
            idx_a = agent_idx[agv.id];  idx_p = agent_idx[nat_picker.id]
            w_ap  = morph_c.W[idx_a, idx_p]
            if np.random.random() < (W_FLOOR + (1.0 - W_FLOOR) * w_ap):
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
        result   = env.step(list(actions[a] for a in agents))
        rewards  = result[1]
        step_del = sum(1 for rv in rewards[:u.num_agvs] if rv > 0.5)
        deliveries += step_del
        delivery_h[t] = step_del

        # MORPH update
        jac      = compute_coassign_jaccard(agents, assigned_agvs, assigned_pickers, N)
        obs_mat  = np.array([[a.x / u.grid_size[1], a.y / u.grid_size[0]]
                              for a in agents], dtype=float)
        pred_jac = compute_pred_jac(agents, assigned_agvs, agent_idx, N)
        morph_c.step(obs_mat, jac, 0.05, delivery=step_del, pred_jac=pred_jac)

        # Record state
        A_history[t] = morph_c.A.astype(np.uint8)
        if (t + 1) in SNAPSHOT_STEPS:
            W_snapshots[t + 1] = morph_c.W.copy()

    # step 0 snapshot: zeros
    W_snapshots[0] = np.zeros((N, N))

    env.close()
    print(f"  Total deliveries: {deliveries}")

    return dict(
        pos_history  = pos_history,
        W_snapshots  = W_snapshots,
        A_history    = A_history,
        delivery_h   = delivery_h,
        eta_h        = np.array(morph_c.eta_h),
        ne_h         = np.array(morph_c.ne_h),
        final_W      = morph_c.W.copy(),
        agent_types  = agent_types,
        N            = N,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 1: W vs spatial distance
# ═════════════════════════════════════════════════════════════════════════════

def plot_w_vs_distance(data):
    print("Generating: W vs spatial distance ...")
    pos   = data['pos_history']   # (T, N, 2)
    W_fin = data['final_W']
    types = data['agent_types']
    N     = data['N']
    T_ep  = pos.shape[0]

    # Mean Manhattan distance for each pair over episode
    mean_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.mean(np.abs(pos[:, i, 0] - pos[:, j, 0])
                      + np.abs(pos[:, i, 1] - pos[:, j, 1]))
            mean_dist[i, j] = mean_dist[j, i] = d

    # Pair types
    pair_type = []
    xs, ys, cs = [], [], []
    PAIR_COLORS = {'AGV-AGV': '#d62728', 'AGV-Picker': '#2166ac', 'Picker-Picker': '#2ca02c'}
    for i in range(N):
        for j in range(i + 1, N):
            ti = types[i];  tj = types[j]
            pt = f"{ti}-{tj}" if ti <= tj else f"{tj}-{ti}"
            pt = pt.replace('Picker-AGV', 'AGV-Picker')
            xs.append(mean_dist[i, j])
            ys.append(W_fin[i, j])
            cs.append(PAIR_COLORS.get(pt, '#888888'))
            pair_type.append(pt)

    xs, ys = np.array(xs), np.array(ys)

    # Spearman correlation (all pairs)
    rho, pval = spearmanr(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(BG)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    for pt, col in PAIR_COLORS.items():
        mask = np.array([p == pt for p in pair_type])
        if mask.sum() == 0:
            continue
        ax.scatter(xs[mask], ys[mask], c=col, s=40, alpha=0.7,
                   edgecolors='none', label=pt, zorder=3)

    # Trend line
    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        xl = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xl, np.polyval(z, xl), color='#555555', lw=1.5,
                linestyle='--', zorder=4, label='linear fit (all)')

    ax.set_xlabel('Mean Manhattan distance over episode', fontsize=11)
    ax.set_ylabel('Final W[i,j]', fontsize=11)
    ax.set_title(f'MORPH: Does learned coordination reflect spatial structure?\n'
                 f'Spearman ρ = {rho:.3f}  (p = {pval:.3f})  '
                 f'[{SCALE_NAME}, seed={SEED}]',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.95)
    ax.yaxis.grid(True, alpha=0.3)

    out = os.path.join(FIG_DIR, 'emergent_w_vs_distance.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Spearman ρ = {rho:.3f} (p={pval:.3f})")
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 2: W matrix evolution heatmaps
# ═════════════════════════════════════════════════════════════════════════════

def plot_w_evolution(data):
    print("Generating: W matrix evolution ...")
    snaps  = data['W_snapshots']
    types  = data['agent_types']
    N      = data['N']
    steps  = sorted(snaps.keys())

    # Sort agents: AGVs first, pickers after
    agv_idx    = [i for i, t in enumerate(types) if t == 'AGV']
    picker_idx = [i for i, t in enumerate(types) if t == 'Picker']
    order      = agv_idx + picker_idx
    n_agv      = len(agv_idx)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)

    vmax = max(snaps[s].max() for s in steps if snaps[s].max() > 0)
    vmax = max(vmax, 0.1)

    for ax, step in zip(axes.flat, steps):
        W = snaps[step][np.ix_(order, order)]
        im = ax.imshow(W, vmin=0, vmax=vmax, cmap='Blues', aspect='auto')
        ax.set_title(f't = {step}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Agent index (AGV | Picker)', fontsize=8)
        ax.set_ylabel('Agent index (AGV | Picker)', fontsize=8)
        # Divider line between AGVs and Pickers
        ax.axhline(n_agv - 0.5, color='red', lw=1.0, linestyle='--', alpha=0.6)
        ax.axvline(n_agv - 0.5, color='red', lw=1.0, linestyle='--', alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate first subplot
    axes[0, 0].text(n_agv / 2, -1.5, 'AGVs', ha='center', va='bottom',
                    fontsize=8, color='#333333')
    axes[0, 0].text(n_agv + len(picker_idx) / 2, -1.5, 'Pickers',
                    ha='center', va='bottom', fontsize=8, color='#333333')

    fig.suptitle(f'MORPH W-matrix evolution — {SCALE_NAME} scale, seed={SEED}\n'
                 f'Red dashes divide AGV block from Picker block. '
                 f'Off-diagonal AGV–Picker weights represent learned coordination.',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'emergent_w_evolution.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 3: Neuromodulation time series
# ═════════════════════════════════════════════════════════════════════════════

def plot_neuromod(data):
    print("Generating: Neuromodulation time series ...")
    eta        = data['eta_h']
    delivery_h = data['delivery_h']
    ne_h       = data['ne_h']
    xs         = np.arange(len(eta))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor=BG,
                                    gridspec_kw={'height_ratios': [3, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    # ── Top: η signal ─────────────────────────────────────────────────────────
    # Shade exploration vs consolidation regions
    ax1.fill_between(xs, np.where(eta < 0, eta, 0), 0,
                     color='#d62728', alpha=0.25, label='Exploration (η < 0)')
    ax1.fill_between(xs, 0, np.where(eta > 0, eta, 0),
                     color='#2166ac', alpha=0.25, label='Consolidation (η > 0)')
    ax1.plot(xs, eta, color='#333333', lw=1.0, alpha=0.8, zorder=3)
    ax1.axhline(0, color='#888888', lw=0.8, linestyle=':')

    # Mark delivery events
    del_steps = np.where(delivery_h > 0)[0]
    ax1.scatter(del_steps, np.zeros(len(del_steps)), c='#ff7f0e',
                s=15, zorder=4, alpha=0.6, label='Delivery event')

    ax1.set_ylabel('Neuromodulation signal η', fontsize=10)
    ax1.set_title(f'MORPH Neuromodulation: exploration ↔ consolidation dynamics\n'
                  f'{SCALE_NAME} scale, seed={SEED}',
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.95, loc='lower right')
    ax1.set_xlim(0, len(eta))
    ax1.yaxis.grid(True, alpha=0.3)

    # ── Bottom: active links over time ────────────────────────────────────────
    ax2.fill_between(xs, ne_h, color='#2ca02c', alpha=0.5, step='mid')
    ax2.set_ylabel('Active links', fontsize=9)
    ax2.set_xlabel('Time step', fontsize=10)
    ax2.set_xlim(0, len(eta))
    ax2.yaxis.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'emergent_neuromod.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 4: Link persistence distribution
# ═════════════════════════════════════════════════════════════════════════════

def plot_link_persistence(data):
    print("Generating: Link persistence distribution ...")
    A_hist = data['A_history']   # (T, N, N)
    N      = data['N']
    T_ep   = A_hist.shape[0]

    # Track lifetimes of every link event
    lifetimes   = []
    still_alive = {}   # (i,j) → birth step

    for t in range(T_ep):
        for i in range(N):
            for j in range(i + 1, N):
                cur = A_hist[t, i, j]
                was = A_hist[t - 1, i, j] if t > 0 else 0
                if cur == 1 and was == 0:
                    still_alive[(i, j)] = t
                elif cur == 0 and was == 1:
                    birth = still_alive.pop((i, j), None)
                    if birth is not None:
                        lifetimes.append(t - birth)

    # Links still alive at end
    for (i, j), birth in still_alive.items():
        lifetimes.append(T_ep - birth)

    lifetimes = np.array(lifetimes)
    print(f"  Total link events: {len(lifetimes)}")
    if len(lifetimes) == 0:
        print("  No link events recorded — skipping figure.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    # ── Left: full histogram ──────────────────────────────────────────────────
    max_life = min(lifetimes.max(), T_ep)
    bins = np.arange(0, max_life + 10, 10)
    ax1.hist(lifetimes, bins=bins, color='#2166ac', alpha=0.8,
             edgecolor='white', linewidth=0.5, zorder=3)
    ax1.axvline(np.median(lifetimes), color='#d62728', lw=1.5,
                linestyle='--', label=f'Median = {np.median(lifetimes):.0f} steps')
    ax1.set_xlabel('Link lifetime (steps)', fontsize=10)
    ax1.set_ylabel('Number of link events', fontsize=10)
    ax1.set_title('Link lifetime distribution\n(all events)', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3)

    # ── Right: stable core vs dynamic periphery ────────────────────────────────
    # Define "stable core" = survived > T/2 = 400 steps
    threshold = T_ep // 2
    stable  = (lifetimes >= threshold).sum()
    dynamic = (lifetimes < threshold).sum()
    ax2.bar(['Stable core\n(≥ 400 steps)', f'Dynamic periphery\n(< 400 steps)'],
            [stable, dynamic],
            color=['#2166ac', '#d62728'], alpha=0.85,
            edgecolor='white', linewidth=1.0, zorder=3)
    for i, v in enumerate([stable, dynamic]):
        ax2.text(i, v + 0.5, f'{v}\n({100*v/max(len(lifetimes),1):.0f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Number of link events', fontsize=10)
    ax2.set_title('Stable core vs dynamic periphery\n'
                  '(lifetime threshold = T/2 = 400 steps)',
                  fontsize=10, fontweight='bold')
    ax2.yaxis.grid(True, alpha=0.3, zorder=0)

    fig.suptitle(f'MORPH link persistence — {SCALE_NAME} scale, seed={SEED}\n'
                 f'Short-lived exploratory links + long-lived stable coordination core',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'emergent_link_persistence.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

    # Summary stats
    pct_stable = 100 * stable / len(lifetimes)
    print(f"  Stable core (≥ {threshold} steps): {stable} events ({pct_stable:.1f}%)")
    print(f"  Dynamic periphery (< {threshold} steps): {dynamic} events ({100-pct_stable:.1f}%)")
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data = run_recorded_episode()

    plot_w_vs_distance(data)
    plot_w_evolution(data)
    plot_neuromod(data)
    plot_link_persistence(data)

    print("\nAll emergent behaviour figures saved to figures/")
