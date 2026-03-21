"""
generate_figures.py
===================
Generates publication figures from experiment_results.pkl.

Figure 1  figures/scale_comparison.png
  6-panel: delivery curves for each scale (2×2) + throughput retention vs N
  + communication efficiency bar chart.
  Shows MORPH v2 vs v1 vs Proximity vs TSG vs Full-Graph.

Figure 2  figures/ablation.png
  4-panel bar chart: final deliveries per scale for all ablation conditions.
  Clearly shows contribution of each mechanism.

Figure 3  figures/pareto.png
  Pareto scatter: communication cost vs throughput for all conditions,
  across all scales.

Usage
-----
  python scripts/generate_figures.py
"""
import sys, os, pickle, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
with open(os.path.join(RES_DIR, 'experiment_results.pkl'), 'rb') as f:
    data = pickle.load(f)

all_results = data['all_results']
T           = data['T']
SEEDS       = data['SEEDS']
SCALES      = data['SCALES']
CONDITIONS  = data['CONDITIONS']
N_seeds     = len(SEEDS)

# ── Visual style ─────────────────────────────────────────────────────────────
BG = 'white'
sm = lambda x, w=15: uniform_filter1d(x.astype(float), w)

COLORS = {
    'no_coord':            '#bbbbbb',
    'proximity':           '#9467bd',
    'tsg':                 '#d62728',
    'morph_v1':            '#74add1',
    'morph_v2_no_bcm':     '#f4a320',
    'morph_v2_no_reward':  '#4dac26',
    'morph_v2_no_neuro':   '#e08080',
    'morph_v2':            '#2166ac',
    'full':                '#555555',
}
LABELS = {
    'no_coord':            'No-Coord',
    'proximity':           'Proximity-r',
    'tsg':                 'TSG',
    'morph_v1':            'MORPH v1',
    'morph_v2_no_bcm':     'v2 − BCM',
    'morph_v2_no_reward':  'v2 − Reward',
    'morph_v2_no_neuro':   'v2 − Neuromod',
    'morph_v2':            'MORPH v2 (full)',
    'full':                'Full-Graph',
}
LINESTYLES = {
    'no_coord':            ':',
    'proximity':           '--',
    'tsg':                 '-.',
    'morph_v1':            '--',
    'morph_v2_no_bcm':     '--',
    'morph_v2_no_reward':  '--',
    'morph_v2_no_neuro':   '--',
    'morph_v2':            '-',
    'full':                ':',
}


def agg(scale, cond, idx):
    arr = np.stack([all_results[scale][cond][s][idx] for s in range(N_seeds)])
    return arr.mean(0), arr.std(0)


def final_del(scale, cond):
    return np.array([all_results[scale][cond][s][1][-1] for s in range(N_seeds)])


def mean_links(scale, cond):
    return np.mean([all_results[scale][cond][s][2].mean() for s in range(N_seeds)])


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scale comparison (delivery curves + summary panels)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Scale comparison ...")
fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.patch.set_facecolor(BG)

# 2×2 delivery curves (top 4 panels)
curve_conds = ['no_coord', 'proximity', 'tsg', 'morph_v1', 'morph_v2', 'full']
xs = np.arange(T)

for pi, (scale, env_id, N_exp) in enumerate(SCALES):
    ax = fig.add_subplot(3, 3, pi + 1)
    ax.set_facecolor(BG)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for cond in curve_conds:
        mn, sd = agg(scale, cond, 1)
        lw = 2.5 if cond == 'morph_v2' else 1.5
        zo = 4 if cond == 'morph_v2' else 2
        ax.plot(xs, sm(mn), color=COLORS[cond], lw=lw,
                linestyle=LINESTYLES[cond], label=LABELS[cond], zorder=zo)
        ax.fill_between(xs, sm(np.maximum(mn - sd, 0)), sm(mn + sd),
                        alpha=0.10, color=COLORS[cond], zorder=zo - 1)
    fg_m = agg(scale, 'full', 1)[0][-1]
    v2_m = agg(scale, 'morph_v2', 1)[0][-1]
    lk   = mean_links(scale, 'morph_v2')
    ml   = N_exp * (N_exp - 1) // 2
    ax.set_title(f'{scale.upper()}  N={N_exp}\n'
                 f'MORPH v2: {v2_m:.1f} del  '
                 f'({100*v2_m/max(fg_m,0.01):.0f}% FG)  '
                 f'{lk:.0f}/{ml} links ({100*lk/ml:.0f}%)',
                 fontsize=9, fontweight='bold', loc='left')
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Cumulative deliveries', fontsize=9)
    ax.set_xlim(0, T)
    if pi == 0:
        ax.legend(fontsize=8, framealpha=0.95, loc='upper left')

# Panel 5: Throughput retention vs N
N_LIST    = [s[2] for s in SCALES]
MAX_LINKS = [N * (N - 1) // 2 for N in N_LIST]
ax5 = fig.add_subplot(3, 3, 5)
ax5.set_facecolor(BG)
for sp in ['top', 'right']: ax5.spines[sp].set_visible(False)
for cond in ['proximity', 'tsg', 'morph_v1', 'morph_v2']:
    retentions = []
    for scale, _, _ in SCALES:
        fg_m = agg(scale, 'full', 1)[0][-1]
        mn_m = agg(scale, cond, 1)[0][-1]
        retentions.append(100.0 * mn_m / max(fg_m, 0.01))
    lw = 2.5 if cond == 'morph_v2' else 1.5
    ax5.plot(N_LIST, retentions, color=COLORS[cond], lw=lw,
             linestyle=LINESTYLES[cond], marker='o', ms=7,
             label=LABELS[cond], zorder=4 if cond == 'morph_v2' else 2)
ax5.axhline(100, color='#888888', lw=0.8, linestyle=':', alpha=0.6)
ax5.set_xlabel('Number of agents (N)', fontsize=9)
ax5.set_ylabel('Deliveries as % of Full-Graph', fontsize=9)
ax5.set_title('Throughput Retention vs Scale', fontsize=10, fontweight='bold')
ax5.set_xticks(N_LIST)
ax5.legend(fontsize=8, framealpha=0.95)
ax5.yaxis.grid(True, alpha=0.3)

# Panel 6: Link efficiency vs N
ax6 = fig.add_subplot(3, 3, 6)
ax6.set_facecolor(BG)
for sp in ['top', 'right']: ax6.spines[sp].set_visible(False)
for cond in ['proximity', 'tsg', 'morph_v1', 'morph_v2']:
    link_pcts = [100.0 * mean_links(s[0], cond) / (s[2]*(s[2]-1)//2) for s in SCALES]
    lw = 2.5 if cond == 'morph_v2' else 1.5
    ax6.plot(N_LIST, link_pcts, color=COLORS[cond], lw=lw,
             linestyle=LINESTYLES[cond], marker='o', ms=7,
             label=LABELS[cond], zorder=4 if cond == 'morph_v2' else 2)
ax6.set_xlabel('Number of agents (N)', fontsize=9)
ax6.set_ylabel('Links used (% of max)', fontsize=9)
ax6.set_title('Communication Efficiency vs Scale', fontsize=10, fontweight='bold')
ax6.set_xticks(N_LIST)
ax6.legend(fontsize=8, framealpha=0.95)
ax6.yaxis.grid(True, alpha=0.3)

# Panel 7-9: per-scale final delivery bar chart (bottom row)
for pi, (scale, _, N_exp) in enumerate(SCALES):
    ax = fig.add_subplot(3, 4, 9 + pi)
    ax.set_facecolor(BG)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    fg_m = agg(scale, 'full', 1)[0][-1]
    bar_conds = ['no_coord', 'proximity', 'tsg', 'morph_v1', 'morph_v2', 'full']
    vals  = [final_del(scale, c).mean() for c in bar_conds]
    errs  = [final_del(scale, c).std()  for c in bar_conds]
    colors = [COLORS[c] for c in bar_conds]
    bars = ax.bar(range(len(bar_conds)), vals, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.0, width=0.65, zorder=3)
    ax.errorbar(range(len(bar_conds)), vals, yerr=errs, fmt='none',
                color='#333333', capsize=3, lw=1.2, zorder=4)
    for i, (bar, v, e) in enumerate(zip(bars, vals, errs)):
        pct = 100.0 * v / max(fg_m, 0.01)
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.2,
                f'{pct:.0f}%', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color=colors[i])
    ax.set_xticks(range(len(bar_conds)))
    ax.set_xticklabels([LABELS[c].replace(' ', '\n') for c in bar_conds],
                       fontsize=6.5, rotation=30, ha='right')
    ax.set_title(f'{scale.upper()} (N={N_exp})', fontsize=9, fontweight='bold')
    ax.set_ylabel('Final deliveries' if pi == 0 else '', fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)

fig.suptitle('MORPH v2: Scale Study — Delivery Performance and Communication Efficiency\n'
             f'T={T}  ·  {N_seeds} seeds  ·  Conditions: {N_seeds} seeds × 4 scales',
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
out1 = os.path.join(FIG_DIR, 'scale_comparison.png')
fig.savefig(out1, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out1}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Ablation bar chart
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Ablation ...")
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 6), facecolor=BG)
fig2.patch.set_facecolor(BG)

ablation_conds = ['full', 'morph_v1', 'morph_v2_no_bcm',
                  'morph_v2_no_reward', 'morph_v2_no_neuro', 'morph_v2']

for ax, (scale, _, N_exp) in zip(axes2, SCALES):
    ax.set_facecolor(BG)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    fg_m = agg(scale, 'full', 1)[0][-1]
    vals  = [final_del(scale, c).mean() for c in ablation_conds]
    errs  = [final_del(scale, c).std()  for c in ablation_conds]
    colors = [COLORS[c] for c in ablation_conds]
    bars = ax.bar(range(len(ablation_conds)), vals, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.2, width=0.65, zorder=3)
    ax.errorbar(range(len(ablation_conds)), vals, yerr=errs, fmt='none',
                color='#333333', capsize=4, lw=1.5, zorder=4)
    for i, (bar, v, e) in enumerate(zip(bars, vals, errs)):
        pct = 100.0 * v / max(fg_m, 0.01)
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.3,
                f'{v:.1f}\n({pct:.0f}%)', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=colors[i])
    ax.set_xticks(range(len(ablation_conds)))
    ax.set_xticklabels([LABELS[c] for c in ablation_conds],
                       fontsize=8, rotation=35, ha='right')
    ax.set_title(f'{scale.upper()}  N={N_exp}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Final deliveries (mean ± std)' if scale == 'tiny' else '',
                  fontsize=10)
    ax.set_ylim(0, max(vals) * 1.40)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    # Highlight MORPH v2 bar
    bars[-1].set_edgecolor('#2166ac')
    bars[-1].set_linewidth(2.5)

# Add mechanism labels as legend
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c], alpha=0.85)
           for c in ablation_conds]
labels  = [LABELS[c] for c in ablation_conds]
fig2.legend(handles, labels, loc='lower center', ncol=6,
            fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, -0.05))
fig2.suptitle('MORPH v2 Ablation: Contribution of Each Plasticity Mechanism\n'
              'Each "v2 − X" removes one mechanism; Full = all-to-all upper bound',
              fontsize=12, fontweight='bold', y=1.03)
fig2.tight_layout()
out2 = os.path.join(FIG_DIR, 'ablation.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out2}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Pareto — comm cost vs throughput (all conditions, all scales)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Pareto ...")
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5), facecolor=BG)
fig3.patch.set_facecolor(BG)
pareto_conds = ['no_coord', 'proximity', 'tsg',
                'morph_v1', 'morph_v2', 'full']

for ax, (scale, _, N_exp) in zip(axes3, SCALES):
    ax.set_facecolor(BG)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ml   = N_exp * (N_exp - 1) // 2
    fg_m = agg(scale, 'full', 1)[0][-1]
    for cond in pareto_conds:
        x = mean_links(scale, cond) / ml
        y = final_del(scale, cond).mean() / max(fg_m, 0.01)
        ms = 160 if cond == 'morph_v2' else 90
        ax.scatter(x, y, c=COLORS[cond], s=ms, zorder=4,
                   edgecolors='white', linewidths=1.5)
        offset = (0.01, 0.01)
        if cond == 'no_coord':   offset = (-0.06, -0.05)
        elif cond == 'full':     offset = (0.01, -0.05)
        elif cond == 'morph_v1': offset = (0.01, -0.05)
        ax.annotate(LABELS[cond], (x, y),
                    xytext=(x + offset[0], y + offset[1]),
                    fontsize=8, color=COLORS[cond], fontweight='bold')
    ax.axhline(1.0, color='#888888', lw=0.8, linestyle=':', alpha=0.5)
    ax.set_xlabel('Comm cost (links / max)', fontsize=9)
    ax.set_ylabel('Throughput (/ Full-Graph)' if scale == 'tiny' else '', fontsize=9)
    ax.set_title(f'{scale.upper()}  N={N_exp}', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.30)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

# Ideal region annotation on first panel
axes3[0].fill_between([0, 0.4], [0.95, 0.95], [1.30, 1.30],
                       alpha=0.06, color='#2166ac')
axes3[0].text(0.02, 1.27, 'ideal region\n(high throughput\nlow cost)',
              fontsize=7, color='#2166ac', va='top')

fig3.suptitle('MORPH v2: Pareto Frontier — Communication Cost vs Throughput\n'
              'Upper-left = better (high throughput, low communication overhead)',
              fontsize=12, fontweight='bold', y=1.03)
fig3.tight_layout()
out3 = os.path.join(FIG_DIR, 'pareto.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out3}")

print("\nAll figures saved.")
