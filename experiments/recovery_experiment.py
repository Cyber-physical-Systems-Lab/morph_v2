"""
Recovery-Time Experiment for MORPH
====================================
Isolates the metric that the per-phase delivery total in shift_experiment.py
hides: how fast each method recovers after a distribution shift, and whether
that recovery time *shrinks on regime returns* (the core memory benefit MORPH
should provide).

Protocol
--------
    Phase 1 (1–400):    LEFT
    Phase 2 (401–800):  RIGHT       ← first shift
    Phase 3 (801–1200): LEFT        ← first return; tests memory once
    Phase 4 (1201–1600): RIGHT       ← tests pruning/relearn
    Phase 5 (1601–2000): LEFT        ← tests memory compounding (key cell)
    Phase 6 (2001–2400): RIGHT       ← tests pruning/relearn again

Same env, same agents, same seed throughout — only the preferred zone changes.

Primary metric — τ_recover(phase k):
    r_50(t)     = sum(step_deliveries[t-49 : t+1]) / 50
    baseline_k  = mean r_50 over the last 50 steps of the previous same-zone phase
    τ_recover_k = min{ t - t_k : t > t_k, r_50(t) ≥ 0.9 · baseline_k }
                  (right-censored at phase length if never reached)

Hypotheses
----------
    H1 (headline): τ_recover(morph_v2,        Phase 5) < τ_recover(morph_v2,        Phase 3)
    H2:            τ_recover(morph_v2_reset,  Phase 5) ≈ τ_recover(morph_v2_reset,  Phase 3)
    H3:            proximity τ_recover ~ constant across cycles (only re-positioning)
    H4:            tsg has smallest τ_recover but lowest plateau

Output
------
    figures/recovery_curves.png
    results/recovery_results.pkl
"""
import sys, os, warnings, pickle, argparse
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

# Reuse helpers and constants from the existing shift experiment so we don't
# duplicate the assignment logic that's already validated there.
from experiments.shift_experiment import (
    compute_coassign_jaccard,
    compute_pred_jac,
    compute_proximity_matrix,
    ENV_ID, W_FLOOR, PROX_R,
    SEEDS as SHIFT_SEEDS,
    # _V2_KW,
    COLORS, LABELS,
)

_V2_KW = dict(
    alpha=0.20968647794935405, beta=0.027724243515151663, decay=0.9705207750820916,
    theta_form_start=0.8733817583708194, theta_form_end=0.38632206314732803, theta_form_anneal=100,
    theta_prune=0.002460673290093817, target_deg_frac=0.30243297989234874,
    grace_steps=20, k_slow=2, max_new=5,
    bcm_tau=0.9222496815870475, bcm_gain=0.5,
    reward_alpha=0.18368593964285584,
    neuromod_gain=0.4, neuromod_explore=0.10,
    neuromod_ema=0.03, expected_delivery_rate=0.14,
    pred_boost=0.4,
)

# ── Config ────────────────────────────────────────────────────────────────────
T_PHASE       = 400
ZONE_SCHEDULE = ['L', 'R', 'L', 'R', 'L', 'R']  # 6 phases
T_TOTAL       = T_PHASE * len(ZONE_SCHEDULE)
WINDOW        = 50

CONDITIONS = ['proximity', 'tsg', 'morph_v2', 'morph_v2_reset']

# Phases (1-indexed in docs, 0-indexed internally) where a regime change occurs.
SHIFT_PHASE_INDICES = list(range(1, len(ZONE_SCHEDULE)))   # [1,2,3,4,5 for 6 phases]


# ── Episode runner ────────────────────────────────────────────────────────────

def run_cyclic_episode(cond, seed, zone_schedule=ZONE_SCHEDULE, t_phase=T_PHASE):
    """
    Run a single episode of length len(zone_schedule)*t_phase under a cyclical
    spatial preference schedule.

    Returns
    -------
    step_deliveries : (T_TOTAL,) int array — deliveries per step
    zone_counts     : dict — per-phase L/R assignment counts (preference check)
    half_x          : int  — grid mid-line x (for sanity printout)
    """
    n_phases = len(zone_schedule)
    T_total  = n_phases * t_phase

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

    half_x = u.grid_size[1] // 2

    morph_c = MORPH(n_agents, **_V2_KW) if cond in ('morph_v2', 'morph_v2_reset') else None

    assigned_agvs    = OrderedDict()
    assigned_pickers = OrderedDict()
    assigned_items   = OrderedDict()
    A = np.zeros((n_agents, n_agents))
    W = np.zeros((n_agents, n_agents))

    step_deliveries = np.zeros(T_total, dtype=int)
    zone_counts = {f'ph{p+1}_{side}': 0
                   for p in range(n_phases) for side in ('left', 'right')}

    for t in range(T_total):
        phase_idx = t // t_phase
        zone      = zone_schedule[phase_idx]

        # Reset MORPH weights at each regime change for morph_v2_reset.
        # Trigger on phase boundaries (t==t_phase, 2*t_phase, ...).
        if (t > 0 and t % t_phase == 0
                and cond == 'morph_v2_reset'):
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
        prefer_left = (zone == 'L')
        preferred = [item for item in rq
                     if item.id not in assigned_items.values()
                     and ((item.x < half_x) if prefer_left else (item.x >= half_x))]
        fallback  = [item for item in rq
                     if item.id not in assigned_items.values()
                     and item not in preferred]
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
            key = f'ph{phase_idx+1}_' + ('left' if item.x < half_x else 'right')
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

    env.close()
    return step_deliveries, zone_counts, half_x


# ── Recovery analysis ─────────────────────────────────────────────────────────

def rolling_rate(step_del, window=WINDOW):
    """r_window(t) = mean deliveries over the past `window` steps (causal)."""
    csum = np.concatenate(([0], np.cumsum(step_del)))
    out  = np.zeros(len(step_del), dtype=float)
    for t in range(len(step_del)):
        lo = max(0, t + 1 - window)
        out[t] = (csum[t + 1] - csum[lo]) / (t + 1 - lo)
    return out


def post_shift_rate(step_del, t_k, t_end, max_window):
    """
    Forward-looking rate within a single phase, free of pre-shift contamination.
    For offset δ ∈ [1, t_end - t_k], rate is the mean of deliveries in
    [t_k, t_k+δ) for δ ≤ max_window (growing window), then a sliding
    `max_window`-step window thereafter.

    Returns an array of length (t_end - t_k); index δ corresponds to post-shift
    step δ+1 (i.e. one delivery sample is required before any rate is defined).
    """
    seg = step_del[t_k:t_end].astype(float)
    csum = np.concatenate(([0.0], np.cumsum(seg)))
    L = len(seg)
    out = np.zeros(L)
    for d in range(1, L + 1):
        lo = max(0, d - max_window)
        out[d - 1] = (csum[d] - csum[lo]) / (d - lo)
    return out


def compute_recovery_metrics(step_del, zone_schedule=ZONE_SCHEDULE,
                             t_phase=T_PHASE, window=WINDOW, eps=0.10,
                             regret_horizon=150):
    """
    For each post-shift phase k>=1, compute recovery metrics relative to a
    prior same-zone baseline. τ_recover is only meaningful when the same zone
    has been visited before (a *return* phase) — for an initial-shift phase
    there is no prior same-zone baseline to recover to.

    The post-shift rate uses a forward-looking window inside the phase
    (`post_shift_rate`) to avoid contamination from pre-shift deliveries.

    Per-phase fields:
      is_return : True iff zone_schedule[k] appeared in zone_schedule[:k]
      tau       : post-shift step δ at which rate ≥ (1-eps)*baseline   (return only)
      regret    : Σ over first `regret_horizon` post-shift steps of
                  max(0, baseline - rate)                              (return only)
      plateau   : mean rate over the last `window` steps of phase k    (always)
      baseline  : reference rate from the prior same-zone phase        (return only)
      censored  : τ hit the right-censor (phase length)

    Non-return phases get NaN for tau/regret/baseline so figure code skips them.
    """
    n_ph = len(zone_schedule)
    out  = []
    for k in range(1, n_ph):
        zone_k = zone_schedule[k]
        prior  = [p for p in range(k) if zone_schedule[p] == zone_k]

        # Plateau: trailing rate inside this phase.
        ps_k = post_shift_rate(step_del, k * t_phase, (k + 1) * t_phase, window)
        plateau = float(ps_k[-1])

        if not prior:
            out.append(dict(
                phase=k + 1, zone=zone_k, is_return=False,
                tau=float('nan'), regret=float('nan'),
                plateau=plateau, baseline=float('nan'), censored=False,
            ))
            continue

        # Baseline: forward-looking rate at the END of the prior same-zone phase.
        base_phase = prior[-1]
        ps_base = post_shift_rate(step_del, base_phase * t_phase,
                                  (base_phase + 1) * t_phase, window)
        baseline = float(ps_base[-1])

        target = (1 - eps) * baseline
        hits = np.where(ps_k >= target)[0]
        censored = (len(hits) == 0) or (baseline <= 1e-9)
        # ps_k is 0-indexed for post-shift step; +1 → step count from boundary.
        tau = float(hits[0] + 1) if not censored else float(t_phase)

        horizon = min(regret_horizon, len(ps_k))
        regret = float(np.maximum(0.0, baseline - ps_k[:horizon]).sum())

        out.append(dict(
            phase=k + 1, zone=zone_k, is_return=True,
            tau=tau, regret=regret, plateau=plateau,
            baseline=baseline, censored=bool(censored),
        ))
    return out


def aggregate_metrics(per_seed_metrics, n_phases):
    """
    per_seed_metrics : list (n_seeds) of list (n_phases-1) of dicts
    Returns dict keyed by metric -> ndarray (n_phases-1, n_seeds), plus
    `is_return` (n_phases-1,) bool indicating which post-shift phases are returns.
    """
    keys = ('tau', 'regret', 'plateau', 'baseline', 'censored')
    n_seeds = len(per_seed_metrics)
    out = {k: np.full((n_phases - 1, n_seeds), np.nan) for k in keys}
    for s, ms in enumerate(per_seed_metrics):
        for ph_offset, m in enumerate(ms):
            for k in keys:
                out[k][ph_offset, s] = m[k]
    # is_return is the same for every seed; use the first seed's view.
    out['is_return'] = np.array([m['is_return'] for m in per_seed_metrics[0]],
                                dtype=bool)
    return out


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Return (median, lo, hi) over `values`."""
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = boots.mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(values)), float(lo), float(hi)


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(results, agg, t_phase=T_PHASE, zone_schedule=ZONE_SCHEDULE,
                window=WINDOW):
    n_phases = len(zone_schedule)
    T_total  = n_phases * t_phase
    fig, (ax_curve, ax_bar) = plt.subplots(2, 1, figsize=(11, 8),
                                            gridspec_kw={'height_ratios': [1.6, 1.0]})

    # Top: rolling-rate curves
    x = np.arange(T_total)
    for cond in CONDITIONS:
        traces = results[cond]                       # (n_seeds, T_total)
        rates  = np.stack([rolling_rate(traces[s], window)
                           for s in range(traces.shape[0])], axis=0)
        mean   = rates.mean(axis=0)
        sem    = rates.std(axis=0) / np.sqrt(rates.shape[0])
        col    = COLORS[cond]
        lw     = 2.4 if cond == 'morph_v2' else 1.7
        ls     = '-'  if cond in ('proximity', 'morph_v2') else '--'
        ax_curve.plot(x, mean, color=col, lw=lw, ls=ls, label=LABELS[cond])
        ax_curve.fill_between(x, mean - sem, mean + sem, color=col, alpha=0.15)

    # Phase boundaries + zone shading
    for p in range(1, n_phases):
        ax_curve.axvline(p * t_phase, color='black', ls=':', lw=1.2)
    for p, zone in enumerate(zone_schedule):
        color = '#dceaff' if zone == 'L' else '#ffe4c4'
        ax_curve.axvspan(p * t_phase, (p + 1) * t_phase, alpha=0.35, color=color, zorder=0)
    ymax = ax_curve.get_ylim()[1]
    for p, zone in enumerate(zone_schedule):
        ax_curve.text((p + 0.5) * t_phase, ymax * 0.95,
                      f'P{p+1} ({zone})', fontsize=9, ha='center',
                      color='steelblue' if zone == 'L' else 'darkorange')

    ax_curve.set_xlabel('Step')
    ax_curve.set_ylabel(f'Deliveries / step  (rolling {window}-step mean)')
    ax_curve.set_title('Recovery curves — cyclical L→R→L→R→L (Large scale)',
                       fontsize=12, fontweight='bold')
    ax_curve.legend(fontsize=9, loc='lower right')
    ax_curve.set_xlim(0, T_total)
    ax_curve.grid(True, alpha=0.3)

    # Bottom: τ_recover per RETURN phase, grouped by condition.
    # Initial-shift phases are excluded — τ is undefined there because the
    # rolling window at t_k inherits the pre-shift rate.
    is_return  = agg[CONDITIONS[0]]['is_return']
    return_idx = [k for k in range(n_phases - 1) if is_return[k]]
    width      = 0.22
    xpos       = np.arange(len(return_idx))
    for i, cond in enumerate(CONDITIONS):
        taus  = agg[cond]['tau']                     # (n_post, n_seeds)
        cis   = [bootstrap_ci(taus[k]) for k in return_idx]
        means = np.array([c[0] for c in cis])
        lo    = np.array([c[1] for c in cis])
        hi    = np.array([c[2] for c in cis])
        offset = (i - (len(CONDITIONS) - 1) / 2) * width
        ax_bar.bar(xpos + offset, means, width=width, color=COLORS[cond],
                   label=LABELS[cond],
                   yerr=[means - lo, hi - means], capsize=3, alpha=0.9)

    labels = [f'P{k+2}\n(return to {zone_schedule[k+1]})' for k in return_idx]
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel(f'τ_recover  (steps to {int((1-0.10)*100)}% of baseline)')
    ax_bar.set_title('Time-to-recover on regime returns '
                     '(lower is better; shrinking across cycles = memory compounding)',
                     fontsize=11)
    ax_bar.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'recovery_curves.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: 1 seed, T_PHASE=100')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Override number of seeds (default: 20)')
    args = parser.parse_args()

    if args.smoke:
        seeds   = [42]
        t_phase = 100
    else:
        seeds   = SHIFT_SEEDS if args.seeds is None else SHIFT_SEEDS[:args.seeds]
        t_phase = T_PHASE

    n_phases = len(ZONE_SCHEDULE)
    T_total  = n_phases * t_phase

    print("=" * 70)
    print("Recovery-Time Experiment — Cyclical L→R→L→R→L")
    print("=" * 70)
    print(f"Env: {ENV_ID}")
    print(f"Phases: {ZONE_SCHEDULE}  |  T per phase: {t_phase}  |  Total: {T_total}")
    print(f"Seeds: {seeds}")
    print(f"Conditions: {CONDITIONS}")
    print()

    results = {cond: [] for cond in CONDITIONS}
    metrics = {cond: [] for cond in CONDITIONS}     # per-seed metric dicts

    for i, seed in enumerate(seeds):
        print(f"[Run {i+1}/{len(seeds)}]  seed={seed}")
        for cond in CONDITIONS:
            step_del, zc, half_x = run_cyclic_episode(
                cond, seed, ZONE_SCHEDULE, t_phase)
            results[cond].append(step_del)
            ms = compute_recovery_metrics(step_del, ZONE_SCHEDULE, t_phase,
                                          window=WINDOW)
            metrics[cond].append(ms)

            # Per-phase totals + tau summary line (return phases only)
            ph_totals = [int(step_del[p*t_phase:(p+1)*t_phase].sum())
                         for p in range(n_phases)]
            tau_str = '  '.join(
                f'τ_P{m["phase"]}={int(m["tau"]):3d}{"*" if m["censored"] else ""}'
                for m in ms if m['is_return']
            )
            print(f"  {cond:22s}  totals={ph_totals}  {tau_str}")
        print()

    # Stack per-condition trace arrays
    for cond in CONDITIONS:
        results[cond] = np.stack(results[cond], axis=0)
    agg = {cond: aggregate_metrics(metrics[cond], n_phases) for cond in CONDITIONS}

    # ── Summary ───────────────────────────────────────────────────────────────
    is_return = agg[CONDITIONS[0]]['is_return']
    return_idx = [k for k in range(n_phases - 1) if is_return[k]]
    print("=" * 70)
    print("τ_recover SUMMARY  (mean across seeds; * = some seeds censored; "
          "return phases only)")
    print(f"{'Condition':<24}  " +
          '  '.join(f'P{k+2}({ZONE_SCHEDULE[k+1]})' for k in return_idx))
    print("-" * 70)
    for cond in CONDITIONS:
        taus = agg[cond]['tau']                      # (n_post, n_seeds)
        cens = agg[cond]['censored']
        cells = []
        for k in return_idx:
            mean = float(np.nanmean(taus[k]))
            cflag = '*' if np.nansum(cens[k]) > 0 else ' '
            cells.append(f'{mean:5.1f}{cflag}')
        print(f"  {cond:<22}  " + '  '.join(cells))

    # Plateau (asymptotic rate) summary — last-window mean per phase
    print()
    print("Plateau (last-window mean rate) per phase:")
    print(f"{'Condition':<24}  " +
          '  '.join(f'P{k+1}({z})' for k, z in enumerate(ZONE_SCHEDULE)))
    print("-" * 70)
    for cond in CONDITIONS:
        traces = results[cond]
        rates  = np.stack([rolling_rate(traces[s]) for s in range(traces.shape[0])])
        cells  = []
        for k in range(n_phases):
            seg_end = (k + 1) * t_phase
            v = rates[:, seg_end - WINDOW : seg_end].mean()
            cells.append(f'{v:5.3f}')
        print(f"  {cond:<22}  " + '  '.join(cells))
    print()

    # ── Persist & figure ──────────────────────────────────────────────────────
    pkl_path = os.path.join(RES_DIR,
                            'recovery_results_smoke.pkl' if args.smoke
                            else 'recovery_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dict(results=results, metrics=metrics, agg=agg,
                         zone_schedule=ZONE_SCHEDULE, t_phase=t_phase,
                         seeds=seeds), f)
    print(f"Saved: {pkl_path}")

    if not args.smoke:
        make_figure(results, agg, t_phase=t_phase,
                    zone_schedule=ZONE_SCHEDULE, window=WINDOW)
    print("Recovery experiment complete.")


if __name__ == '__main__':
    main()
