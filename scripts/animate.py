"""
animate.py
==========
Renders MORPH v2 warehouse animations for any TA-RWARE scale.

Usage:
  python scripts/animate.py tiny
  python scripts/animate.py small
  python scripts/animate.py medium
  python scripts/animate.py large
  python scripts/animate.py all        # renders all four

Output: figures/MORPH_v2_{scale}.gif
"""
import sys, warnings, io, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines   as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from collections import OrderedDict
from PIL import Image

import tarware, gymnasium as gym
from tarware.heuristic import AgentType, Mission, MissionType
from morph import MORPH

# ── Scale registry ────────────────────────────────────────────────────────────
SCALES = {
    'tiny':   ('tarware-tiny-5agvs-3pickers-partialobs-v1',       8),
    'small':  ('tarware-small-8agvs-4pickers-partialobs-v1',      12),
    'medium': ('tarware-medium-12agvs-6pickers-partialobs-v1',    18),
    'large':  ('tarware-large-16agvs-8pickers-partialobs-v1',     24),
}

SEED    = 42
T       = 300
W_FLOOR = 0.25
BG      = 'white'

# Best v2 config (from hyperparam search)
MORPH_V2_KW_BASE = dict(
    alpha=0.18, beta=0.04, decay=0.98,
    theta_form_start=0.75, theta_form_end=0.45, theta_form_anneal=100,
    theta_prune=0.008, grace_steps=20, k_slow=3,
    bcm_tau=0.95, bcm_gain=0.8,
    reward_alpha=0.10,
    neuromod_gain=0.4, neuromod_explore=0.10, neuromod_ema=0.03,
    pred_boost=0.4,
    target_deg_frac=0.20,
)
EXPECTED_RATE = {'tiny': 0.07, 'small': 0.09, 'medium': 0.12, 'large': 0.14}


# ── Helpers ───────────────────────────────────────────────────────────────────
def coassign_jac(agents, assigned_agvs, assigned_pickers, N):
    targets = {}
    for a in agents:
        if a in assigned_agvs:    targets[a.id] = (assigned_agvs[a].location_x,    assigned_agvs[a].location_y)
        elif a in assigned_pickers: targets[a.id] = (assigned_pickers[a].location_x, assigned_pickers[a].location_y)
        else: targets[a.id] = None
    ids = [a.id for a in agents]; jac = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            ti, tj = targets[ids[i]], targets[ids[j]]
            if ti and tj and ti == tj: jac[i,j] = jac[j,i] = 1.0
    return jac


def pred_jac_mat(agents, assigned_agvs, agent_idx, N):
    pred    = np.zeros((N, N))
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    for agv in [a for a in agents if a.type == AgentType.AGV]:
        if agv not in assigned_agvs: continue
        m = assigned_agvs[agv]
        if m.mission_type not in (MissionType.PICKING, MissionType.RETURNING): continue
        dists = [abs(p.x - m.location_x) + abs(p.y - m.location_y) for p in pickers]
        if not dists: continue
        near = pickers[int(np.argmin(dists))]
        i, j = agent_idx[agv.id], agent_idx[near.id]
        pred[i,j] = pred[j,i] = 1.0
    return pred


# ── Simulation ────────────────────────────────────────────────────────────────
def run_simulation(scale_name):
    env_id, _ = SCALES[scale_name]
    kw = {**MORPH_V2_KW_BASE,
          'expected_delivery_rate': EXPECTED_RATE[scale_name]}

    print(f"\n{'='*55}")
    print(f"Running simulation: {scale_name.upper()}  ({env_id})")
    print(f"{'='*55}")

    env = gym.make(env_id); u = env.unwrapped; env.reset(seed=SEED)
    np.random.seed(SEED)

    agents  = u.agents
    agvs    = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    N       = len(agents)
    ROWS, COLS = u.grid_size
    agent_idx  = {a.id: i for i, a in enumerate(agents)}
    coords_map = {v: k for k, v in u.action_id_to_coords_map.items()}
    non_goal_ids = np.array([i for i, c in u.action_id_to_coords_map.items()
                              if (c[1], c[0]) not in u.goals])

    kw['max_new'] = min(6, max(3, N // 6))
    morph_c = MORPH(N, **kw)
    A = np.zeros((N,N)); W = np.zeros((N,N))
    assigned_agvs    = OrderedDict()
    assigned_pickers = OrderedDict()
    assigned_items   = OrderedDict()

    frames     = []
    link_flash = {}
    prev_A     = A.copy()
    deliveries = 0

    for t in range(T):
        rq = u.request_queue; gls = u.goals; actions = {a: 0 for a in agents}

        for item in rq:
            if item.id in assigned_items.values(): continue
            avail = [a for a in agvs if not a.busy and not a.carrying_shelf
                     and a not in assigned_agvs]
            if not avail: continue
            dists = [len(u.find_path((a.y,a.x),(item.y,item.x),a,care_for_agents=False)) for a in avail]
            best  = avail[np.argmin(dists)]
            assigned_agvs[best]  = Mission(MissionType.PICKING, coords_map[(item.y,item.x)], item.x, item.y, t)
            assigned_items[best] = item.id

        for agv in agvs:
            if agv in assigned_agvs:
                m = assigned_agvs[agv]
                assigned_agvs[agv].at_location = (agv.x == m.location_x and agv.y == m.location_y)
            if agv not in assigned_agvs or agv.busy: continue
            m = assigned_agvs[agv]
            if m.mission_type == MissionType.PICKING and m.at_location and agv.carrying_shelf:
                paths = [u.find_path((agv.y,agv.x),(y,x),agv,care_for_agents=False) for (x,y) in gls]
                bg = gls[np.argmin([len(p) for p in paths])]
                assigned_agvs[agv] = Mission(MissionType.DELIVERING, coords_map[(bg[1],bg[0])], bg[0], bg[1], t)
            elif m.mission_type == MissionType.DELIVERING and m.at_location and agv.carrying_shelf:
                empty = u.get_empty_shelf_information()
                eids  = [i for i,e in zip(non_goal_ids,empty) if e>0]
                taken = {m2.location_id for a2,m2 in assigned_agvs.items()
                         if a2 is not agv and m2.mission_type == MissionType.RETURNING}
                eids  = [e for e in eids if e not in taken] or eids
                if eids:
                    elocs = [u.action_id_to_coords_map[i] for i in eids]
                    paths = [u.find_path((agv.y,agv.x),(y,x),agv,care_for_agents=False) for (y,x) in elocs]
                    be = eids[np.argmin([len(p) for p in paths])]; byx = u.action_id_to_coords_map[be]
                    assigned_agvs[agv] = Mission(MissionType.RETURNING, be, byx[1], byx[0], t)
            elif m.mission_type == MissionType.RETURNING and m.at_location and not agv.carrying_shelf:
                assigned_agvs.pop(agv); assigned_items.pop(agv, None)
            elif m.mission_type == MissionType.RETURNING and m.at_location and agv.carrying_shelf:
                empty = u.get_empty_shelf_information()
                eids  = [i for i,e in zip(non_goal_ids,empty) if e>0]
                taken = {m2.location_id for a2,m2 in assigned_agvs.items()
                         if a2 is not agv and m2.mission_type == MissionType.RETURNING}
                eids  = [e for e in eids if e not in taken
                         and u.action_id_to_coords_map[e] != (agv.y, agv.x)] or eids
                if eids:
                    elocs = [u.action_id_to_coords_map[i] for i in eids]
                    paths = [u.find_path((agv.y,agv.x),(y,x),agv,care_for_agents=False) for (y,x) in elocs]
                    be = eids[np.argmin([len(p) for p in paths])]; byx = u.action_id_to_coords_map[be]
                    assigned_agvs[agv] = Mission(MissionType.RETURNING, be, byx[1], byx[0], t)

        picker_locs = {(mp.location_x, mp.location_y) for mp in assigned_pickers.values()}
        for agv, m in assigned_agvs.items():
            if m.mission_type not in (MissionType.PICKING, MissionType.RETURNING): continue
            if (m.location_x, m.location_y) in picker_locs: continue
            avail_p = [p for p in pickers if p not in assigned_pickers]
            if not avail_p: continue
            dists = [abs(p.x-m.location_x)+abs(p.y-m.location_y) for p in avail_p]
            nat_p = avail_p[int(np.argmin(dists))]
            w_ap  = W[agent_idx[agv.id], agent_idx[nat_p.id]]
            if np.random.random() < (W_FLOOR + (1-W_FLOOR)*w_ap):
                assigned_pickers[nat_p] = Mission(MissionType.PICKING, m.location_id, m.location_x, m.location_y, t)
                picker_locs.add((m.location_x, m.location_y))

        for picker in list(assigned_pickers.keys()):
            mp = assigned_pickers[picker]
            if not any(m.location_x==mp.location_x and m.location_y==mp.location_y
                       and m.mission_type in (MissionType.PICKING, MissionType.RETURNING)
                       for m in assigned_agvs.values()):
                assigned_pickers.pop(picker)

        for agv,m in assigned_agvs.items(): actions[agv] = m.location_id if not agv.busy else 0
        for p,m   in assigned_pickers.items(): actions[p] = m.location_id

        result    = env.step(list(actions[a] for a in agents))
        rewards   = result[1]
        step_del  = sum(1 for rv in rewards[:u.num_agvs] if rv > 0.5)
        deliveries += step_del

        jac     = coassign_jac(agents, assigned_agvs, assigned_pickers, N)
        pjac    = pred_jac_mat(agents, assigned_agvs, agent_idx, N)
        obs_mat = np.array([[a.x/COLS, a.y/ROWS] for a in agents], dtype=float)
        morph_c.step(obs_mat, jac, 0.05, delivery=step_del, pred_jac=pjac)
        A = morph_c.A.copy(); W = morph_c.W.copy()

        for i in range(N):
            for j in range(i+1, N):
                if prev_A[i,j]==0 and A[i,j]==1: link_flash[(i,j)] = ('form', 14)
                elif prev_A[i,j]==1 and A[i,j]==0: link_flash[(i,j)] = ('prune', 14)
        link_flash = {k:(tp,max(0,fr-1)) for k,(tp,fr) in link_flash.items() if fr>0}
        prev_A = A.copy()

        if t % 2 == 0:
            frames.append({
                't': t, 'A': A.copy(), 'W': W.copy(),
                'agents':     [(a.id, a.type.name, a.x, a.y, a.busy, a.carrying_shelf is not None) for a in agents],
                'shelves':    [(s.x, s.y) for s in u.shelfs],
                'req_set':    {(s.x, s.y) for s in u.request_queue},
                'goals':      list(u.goals),
                'flash':      dict(link_flash),
                'cov_h':      list(morph_c.cov_h),
                'ne_h':       list(morph_c.ne_h),
                'form_h':     list(morph_c.form_h),
                'prune_h':    list(morph_c.prune_h),
                'eta_h':      list(morph_c.eta_h),
                'bcm_h':      list(morph_c.bcm_h),
                'deliveries': deliveries,
                'step_del':   step_del,
                'missions':   {a.id: (assigned_agvs[a].mission_type.name if a in assigned_agvs else 'IDLE') for a in agvs},
                'picker_busy':{p.id: (p in assigned_pickers) for p in pickers},
                'N': N, 'ROWS': ROWS, 'COLS': COLS,
                'target_deg': morph_c.target_deg,
            })

        if t % 50 == 0:
            ne = int(A.sum()//2)
            eta = morph_c.eta_h[-1] if morph_c.eta_h else 0
            print(f"  t={t:3d} | links={ne:3d}/{N*(N-1)//2} | "
                  f"del={deliveries} | η={eta:+.2f}")

    env.close()
    print(f"\nDone: {deliveries} deliveries  {morph_c.ne_h[-1]} final links  {len(frames)} frames")
    return frames


# ── Rendering ─────────────────────────────────────────────────────────────────
AGV_IDLE  = '#aaccee'; AGV_BUSY = '#2166ac'; AGV_CARRY = '#f4a320'
PCK_IDLE  = '#fdbb84'; PCK_BUSY = '#d73027'
SHELF_COL = '#c8b89a'; REQ_COL  = '#ff7700'; GOAL_COL  = '#1a9850'
FORM_COL  = (0.04, 0.68, 0.14, 0.90)
PRUNE_COL = (0.85, 0.10, 0.10, 0.75)

def syn_style(w):
    lw    = 0.4 + 3.5 * w
    alpha = 0.12 + 0.83 * w
    r = max(0, 0.55 - 0.45*w); g = max(0, 0.65 - 0.40*w); b = min(1, 0.90 - 0.05*w)
    return (r, g, b, alpha), lw


def draw_frame(fd, fidx, total, scale_name):
    t          = fd['t']
    A          = fd['A']; W_mat = fd['W']
    agents_s   = fd['agents']
    shelves    = fd['shelves']; req_set = fd['req_set']; goals = fd['goals']
    flash      = fd['flash']
    missions   = fd['missions']; picker_busy = fd['picker_busy']
    cov_h      = fd['cov_h']; ne_h = fd['ne_h']
    form_h     = fd['form_h']; prune_h = fd['prune_h']
    eta_h      = fd['eta_h']; bcm_h = fd['bcm_h']
    deliveries = fd['deliveries']
    N          = fd['N']; ROWS = fd['ROWS']; COLS = fd['COLS']
    target_deg = fd['target_deg']

    fig = plt.figure(figsize=(22, 10), facecolor=BG)
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            left=0.02, right=0.98, top=0.91, bottom=0.06,
                            hspace=0.45, wspace=0.09, width_ratios=[1.65, 1.0])
    ax_w  = fig.add_subplot(gs[:, 0])
    ax_m  = fig.add_subplot(gs[0:2, 1])
    ax_ne = fig.add_subplot(gs[2, 1])
    ax_eta= fig.add_subplot(gs[3, 1])
    ax_pl = fig.add_subplot(gs[4, 1])

    for ax in [ax_w, ax_ne, ax_eta, ax_pl]:
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_color('#cccccc')
    ax_m.set_facecolor('#f0f4ff')
    for sp in ax_m.spines.values(): sp.set_color('#4466aa'); sp.set_linewidth(1.2)

    # ── Warehouse ──
    ax_w.set_xlim(-0.5, COLS-0.5); ax_w.set_ylim(-0.5, ROWS-0.5)
    ax_w.set_aspect('equal'); ax_w.axis('off')
    ax_w.add_patch(mpatches.Rectangle((-0.5,-0.5), COLS, ROWS,
                                       facecolor='#f5f5f5', edgecolor='none', zorder=0))
    for x in range(COLS): ax_w.axvline(x-0.5, color='#e8e8e8', lw=0.3, zorder=0)
    for y in range(ROWS): ax_w.axhline(y-0.5, color='#e8e8e8', lw=0.3, zorder=0)

    for (gx, gy) in goals:
        ax_w.add_patch(mpatches.FancyBboxPatch((gx-0.45,gy-0.45), 0.9, 0.9,
                       boxstyle='round,pad=0.05', facecolor='#c7f0d0',
                       edgecolor=GOAL_COL, linewidth=1.5, alpha=0.85, zorder=1))
        ax_w.text(gx, gy, '▼', ha='center', va='center', fontsize=9, color=GOAL_COL, zorder=2)

    for (sx, sy) in shelves:
        is_req = (sx, sy) in req_set
        ax_w.add_patch(mpatches.FancyBboxPatch((sx-0.38,sy-0.38), 0.76, 0.76,
                       boxstyle='round,pad=0.04',
                       facecolor=REQ_COL if is_req else SHELF_COL,
                       edgecolor='#cc4400' if is_req else '#8a6a40',
                       linewidth=2.0 if is_req else 0.8, alpha=0.88, zorder=2))
        if is_req:
            ax_w.text(sx, sy, '!', ha='center', va='center',
                      fontsize=8, color='white', fontweight='bold', zorder=3)

    apos      = {aid: (ax_, ay_) for aid, atype, ax_, ay_, busy, carry in agents_s}
    agent_ids = [a[0] for a in agents_s]

    for i in range(N):
        for j in range(i+1, N):
            if A[i,j] != 1: continue
            xi, yi = apos[agent_ids[i]]; xj, yj = apos[agent_ids[j]]
            pair = (min(i,j), max(i,j))
            if pair in flash and flash[pair][0] == 'form':
                ttl = flash[pair][1]
                col = (0.04, 0.68, 0.14, min(1.0, 0.4+ttl*0.05))
                ax_w.plot([xi,xj],[yi,yj], color=col, lw=4.5, solid_capstyle='round', zorder=4)
            else:
                col, lw = syn_style(W_mat[i,j])
                ax_w.plot([xi,xj],[yi,yj], color=col, lw=lw, solid_capstyle='round', zorder=4)

    for (i,j),(ftype,ttl) in flash.items():
        if ftype=='prune' and ttl>0 and A[i,j]==0:
            xi,yi = apos[agent_ids[i]]; xj,yj = apos[agent_ids[j]]
            ax_w.plot([xi,xj],[yi,yj], color=(0.85,0.1,0.1,min(1.0,0.3+ttl*0.055)),
                      lw=3.0, linestyle='--', zorder=4)

    for aid, atype, ax_, ay_, busy, carry in agents_s:
        if atype == 'AGV':
            fc = AGV_CARRY if carry else (AGV_BUSY if busy else AGV_IDLE)
            ec = '#aa6600' if carry else ('#1a5a8a' if busy else '#4488aa')
            sz = 300; marker = 'h'
            ms = missions.get(aid, 'IDLE')
            lbl = {'PICKING':'→shelf','DELIVERING':'→dock','RETURNING':'→return','IDLE':''}.get(ms,'')
        else:
            is_b = picker_busy.get(aid, False)
            fc = PCK_BUSY if is_b else PCK_IDLE
            ec = '#a02020' if is_b else '#cc7744'
            sz = 180; marker = 'D'; lbl = ''
        ax_w.scatter(ax_, ay_, s=sz, marker=marker, color=fc, edgecolors=ec, linewidths=2.0, zorder=8)
        if carry:
            ax_w.scatter(ax_, ay_, s=sz*1.7, marker=marker, color='none',
                         edgecolors=AGV_CARRY, linewidths=2.5, zorder=7)
        if lbl:
            ax_w.text(ax_, ay_+0.6, lbl, ha='center', va='bottom',
                      fontsize=5.5, color='#333333', zorder=9)

    ax_w.text(COLS-0.3, ROWS-0.8, f'✓ {deliveries}\ndeliveries',
              ha='right', va='top', fontsize=12, fontweight='bold', color='#8b0000',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe0d0',
                        edgecolor='#d73027', linewidth=1.5), zorder=10)

    legend_handles = [
        mlines.Line2D([],[],marker='h',color='w',markerfacecolor=AGV_IDLE,markeredgecolor='#4488aa',markersize=11,label='AGV (idle)'),
        mlines.Line2D([],[],marker='h',color='w',markerfacecolor=AGV_BUSY,markeredgecolor='#1a5a8a',markersize=11,label='AGV (en route)'),
        mlines.Line2D([],[],marker='h',color='w',markerfacecolor=AGV_CARRY,markeredgecolor='#aa6600',markersize=11,label='AGV (carrying)'),
        mlines.Line2D([],[],marker='D',color='w',markerfacecolor=PCK_IDLE,markeredgecolor='#cc7744',markersize=9,label='Picker (idle)'),
        mlines.Line2D([],[],marker='D',color='w',markerfacecolor=PCK_BUSY,markeredgecolor='#a02020',markersize=9,label='Picker (busy)'),
        mpatches.Patch(facecolor=REQ_COL,edgecolor='#cc4400',label='Requested shelf'),
        mlines.Line2D([],[],color=FORM_COL,lw=4.0,label='Link formed (structural)'),
        mlines.Line2D([],[],color=PRUNE_COL,lw=2.5,linestyle='--',label='Link pruned (BCM)'),
        mlines.Line2D([],[],color=(0.10,0.25,0.85,0.85),lw=2.5,label='Stable link (W-weighted)'),
    ]
    ax_w.legend(handles=legend_handles, loc='lower left', fontsize=6, framealpha=0.95,
                facecolor='#f8f8fc', edgecolor='#ccccdd', ncol=2,
                handlelength=1.5, borderpad=0.5, labelspacing=0.3)

    ne = int(A.sum()//2)
    ax_w.set_title(f't = {t}   ·   Links: {ne}/{N*(N-1)//2}   ·   Deliveries: {deliveries}'
                   f'   ·   {scale_name.upper()}  N={N}',
                   fontsize=11, fontweight='bold', color='#111122', pad=5)

    # ── Mechanism box (v2) ──
    ax_m.clear(); ax_m.set_facecolor('#f0f4ff')
    for sp in ax_m.spines.values(): sp.set_color('#4466aa'); sp.set_linewidth(1.2)
    ax_m.set_xlim(0,1); ax_m.set_ylim(0,1); ax_m.axis('off')

    sf  = sum(1 for (tp,fr) in flash.values() if tp=='form'  and fr>0)
    spf = sum(1 for (tp,fr) in flash.values() if tp=='prune' and fr>0)
    sc  = '#04ac23' if sf>0 else ('#cc0000' if spf>0 else '#999999')
    st  = (f'+{sf} formed  ' if sf else '') + (f'−{spf} pruned' if spf else '') or 'dormant'
    eta_now   = eta_h[-1]  if eta_h  else 0.0
    bcm_now   = bcm_h[-1]  if bcm_h  else 0.0
    eta_col   = '#04ac23'  if eta_now > 0.05 else ('#cc0000' if eta_now < -0.05 else '#888888')
    eta_str   = f'η = {eta_now:+.2f}  ' + ('↑ consolidating' if eta_now>0.05 else ('↓ exploring' if eta_now<-0.05 else '≈ baseline'))
    cur_deg   = A.sum(axis=1).mean()

    ax_m.text(0.5, 0.97, 'MORPH v2 Active Mechanisms', ha='center', fontsize=9,
              color='#223366', fontweight='bold', transform=ax_m.transAxes)
    ax_m.axhline(0.91, color='#aabbdd', lw=0.8, xmin=0.04, xmax=0.96)

    ax_m.text(0.05,0.86,'⬡ Structural + BCM Metaplasticity', fontsize=7.5,
              color=sc, fontweight='bold', transform=ax_m.transAxes)
    ax_m.text(0.07,0.76,'Forms links via MI+Jaccard+pred hint\n'
              'BCM: high-W links decay faster (θ_ij sliding)',
              fontsize=6, color='#334455', transform=ax_m.transAxes)
    ax_m.text(0.07,0.68,f'Now: {st}   BCM θ̄={bcm_now:.3f}',
              fontsize=6, color=sc, fontweight='bold', transform=ax_m.transAxes)
    ax_m.axhline(0.63, color='#dde8ff', lw=0.6, xmin=0.04, xmax=0.96)

    ax_m.text(0.05,0.58,'~ Synaptic + Reward-modulated', fontsize=7.5,
              color='#2255aa', fontweight='bold', transform=ax_m.transAxes)
    ax_m.text(0.07,0.48,'W updated by Hebbian (co-assign) + delivery burst\n'
              f'P(respond) = {W_FLOOR:.2f} + {1-W_FLOOR:.2f}·W',
              fontsize=6, color='#334455', transform=ax_m.transAxes)
    ax_m.axhline(0.42, color='#dde8ff', lw=0.6, xmin=0.04, xmax=0.96)

    ax_m.text(0.05,0.37,'⚡ Neuromodulation', fontsize=7.5,
              color=eta_col, fontweight='bold', transform=ax_m.transAxes)
    ax_m.text(0.07,0.27, eta_str, fontsize=6.5, color=eta_col,
              fontweight='bold', transform=ax_m.transAxes)
    ax_m.text(0.07,0.19,'scales α and θ_form based on delivery rate vs expected',
              fontsize=6, color='#334455', transform=ax_m.transAxes)
    ax_m.axhline(0.13, color='#dde8ff', lw=0.6, xmin=0.04, xmax=0.96)

    ax_m.text(0.05,0.08,'⚖ Homeostatic', fontsize=7.5,
              color='#cc7700', fontweight='bold', transform=ax_m.transAxes)
    ax_m.text(0.07,0.01,f'target deg={target_deg:.1f}  ·  current mean={cur_deg:.1f}',
              fontsize=6, color='#334455', transform=ax_m.transAxes)

    # ── Link count ──
    ne_arr = np.array(ne_h, dtype=float); xne = np.arange(len(ne_arr))
    ax_ne.plot(xne, ne_arr, color='#1a8c5e', lw=1.8)
    ax_ne.fill_between(xne, ne_arr, alpha=0.14, color='#1a8c5e')
    ax_ne.axhline(N*(N-1)//2, color='#d73027', lw=0.8, linestyle=':', alpha=0.5)
    ax_ne.set_xlim(0,T); ax_ne.set_ylabel('Active links', fontsize=7, color='#1a8c5e')
    ax_ne.set_title(f'Topology size  (max={N*(N-1)//2})', fontsize=8, fontweight='bold')
    ax_ne.tick_params(colors='#888899', labelsize=6)
    if len(ne_arr)>0: ax_ne.scatter([xne[-1]],[ne_arr[-1]],color='#111133',s=20,zorder=5)

    # ── Neuromodulation signal η ──
    if eta_h:
        eta_arr = np.array(eta_h, dtype=float); xeta = np.arange(len(eta_arr))
        ax_eta.plot(xeta, eta_arr, color=eta_col, lw=1.8)
        ax_eta.fill_between(xeta, np.minimum(eta_arr,0), alpha=0.15, color='#cc0000')
        ax_eta.fill_between(xeta, np.maximum(eta_arr,0), alpha=0.15, color='#04ac23')
        ax_eta.axhline(0, color='#888888', lw=0.8, linestyle=':')
    ax_eta.set_xlim(0,T); ax_eta.set_ylim(-1.1,1.1)
    ax_eta.set_ylabel('η (neuromod)', fontsize=7, color='#555566')
    ax_eta.set_title('Neuromodulation signal  (+ = consolidate, − = explore)',
                     fontsize=8, fontweight='bold')
    ax_eta.tick_params(colors='#888899', labelsize=6)

    # ── Plasticity events ──
    fe = np.array(form_h, dtype=float); pe = np.array(prune_h, dtype=float)
    xpf = np.arange(len(fe))
    if fe.any():  ax_pl.bar(xpf[fe>0],  fe[fe>0], color='#04ac23', alpha=0.75, width=1, label='Formed')
    if pe.any():  ax_pl.bar(xpf[pe>0], -pe[pe>0], color='#cc0000', alpha=0.75, width=1, label='Pruned')
    ax_pl.axhline(0, color='#cccccc', lw=0.5)
    ax_pl.set_xlim(0,T); ax_pl.set_xlabel('Time step', fontsize=7)
    ax_pl.set_ylabel('Events/step', fontsize=7); ax_pl.set_title('Structural events', fontsize=8, fontweight='bold')
    ax_pl.tick_params(colors='#888899', labelsize=6); ax_pl.legend(fontsize=6, loc='upper left')

    # ── Progress bar ──
    prog = fidx / max(total-1, 1)
    ba   = fig.add_axes([0.02, 0.005, 0.96, 0.012])
    ba.barh(0, prog, height=1, color='#2255aa')
    ba.barh(0, 1-prog, height=1, color='#e8e8f0', left=prog)
    ba.set_xlim(0,1); ba.axis('off')

    fig.suptitle(
        f'MORPH v2  ·  {scale_name.upper()} warehouse  ·  N={N} agents  ·  '
        f'BCM + Reward-modulated + Neuromodulation + Predictive formation\n'
        f'AGVs (hexagons)  ·  Pickers (diamonds)  ·  '
        f'Soft gating P(respond) = 0.25 + 0.75·W  ·  A* path planning',
        color='#111122', fontsize=10, fontweight='bold', y=0.975)

    fig.canvas.draw()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=90, facecolor=BG, bbox_inches='tight')
    buf.seek(0); img = Image.open(buf).copy()
    plt.close(fig); buf.close()
    return img


def render_and_save(scale_name):
    frames   = run_simulation(scale_name)
    out_path = os.path.join(ROOT, 'figures', f'MORPH_v2_{scale_name}.gif')
    print(f"\nRendering {len(frames)} frames...")
    gif_frames = []
    for idx, fd in enumerate(frames):
        if idx % 10 == 0:
            print(f"  frame {idx:3d}/{len(frames)}  t={fd['t']}  "
                  f"links={int(fd['A'].sum()//2)}  del={fd['deliveries']}")
        gif_frames.append(draw_frame(fd, idx, len(frames), scale_name))

    durations = [280 if any(fr>8 for (_,fr) in fd['flash'].values()) else 120
                 for fd in frames]
    gif_frames = [gif_frames[0]]*5 + gif_frames + [gif_frames[-1]]*8
    durations  = [700]*5 + durations + [900]*8

    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:],
                       duration=durations, loop=0, optimize=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else 'tiny'
    targets = list(SCALES.keys()) if arg == 'all' else [arg]
    for scale in targets:
        if scale not in SCALES:
            print(f"Unknown scale '{scale}'. Choose from: {list(SCALES.keys())} or 'all'")
            sys.exit(1)
        render_and_save(scale)
    print("\nAll done.")
