"""
generate_figures.py
===================
Generates publication figures from experiment_results.pkl.

Figure 1  figures/scale_comparison.png
  Delivery curves for each scale (2×2) + throughput retention vs N
  + communication efficiency vs N.

Figure 2  figures/scale_bars.png
  Bar chart: final deliveries per condition across all 4 scales.

Figure 3  figures/pareto.png
  Pareto scatter: communication cost vs throughput, all 4 scales.

Conditions shown: Proximity, TSG, MORPH, Full-Graph.

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

# ── Load ──────────────────────────────────────────────────────────────────────
with open(os.path.join(RES_DIR, 'experiment_results.pkl'), 'rb') as f:
    data = pickle.load(f)

all_results = data['all_results']
T           = data['T']
SEEDS       = data['SEEDS']
SCALES      = data['SCALES']
N_seeds     = len(SEEDS)

# ── Conditions to show ────────────────────────────────────────────────────────
SHOW_CONDS = ['proximity', 'tsg', 'morph_v2', 'full']

COLORS = {
    'proximity': '#9467bd',
    'tsg':       '#d62728',
    'morph_v2':  '#2166ac',
    'full':      '#555555',
}
LABELS = {
    'proximity': 'Proximity',
    'tsg':       'TSG',
    'morph_v2':  'MORPH',
    'full':      'Full-Graph',
}
LINESTYLES = {
    'proximity': '--',
    'tsg':       '-.',
    'morph_v2':  '-',
    'full':      ':',
}
LINEWIDTHS = {
    'proximity': 1.8,
    'tsg':       1.8,
    'morph_v2':  2.5,
    'full':      1.5,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
BG = 'white'
sm = lambda x, w=15: uniform_filter1d(x.astype(float), w)


def agg(scale, cond, idx):
    arr = np.stack([all_results[scale][cond][s][idx] for s in range(N_seeds)])
    return arr.mean(0), arr.std(0)


def final_del(scale, cond):
    return np.array([all_results[scale][cond][s][1][-1] for s in range(N_seeds)])


def mean_links(scale, cond):
    return np.mean([all_results[scale][cond][s][2].mean() for s in range(N_seeds)])


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Delivery curves + summary panels
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Scale comparison ...")
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.patch.set_facecolor(BG)
xs = np.arange(T)

# 2×2 delivery curves (panels 1-4)
for pi, (scale, env_id, N_exp) in enumerate(SCALES):
    ax = fig.add_subplot(3, 3, pi + 1)
    ax.set_facecolor(BG)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    for cond in SHOW_CONDS:
        mn, sd = agg(scale, cond, 1)
        ax.plot(xs, sm(mn), color=COLORS[cond], lw=LINEWIDTHS[cond],
                linestyle=LINESTYLES[cond], label=LABELS[cond],
                zorder=4 if cond == 'morph_v2' else 2)
        ax.fill_between(xs, sm(np.maximum(mn - sd, 0)), sm(mn + sd),
                        alpha=0.12, color=COLORS[cond],
                        zorder=3 if cond == 'morph_v2' else 1)

    fg_m = agg(scale, 'full', 1)[0][-1]
    v2_m = agg(scale, 'morph_v2', 1)[0][-1]
    pr_m = agg(scale, 'proximity', 1)[0][-1]
    ax.set_title(f'{scale.upper()}  N={N_exp}\n'
                 f'MORPH {v2_m:.0f}  Proximity {pr_m:.0f}  '
                 f'({100*v2_m/max(pr_m,0.01):.0f}% of Proximity)',
                 fontsize=9, fontweight='bold', loc='left')
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Cumulative deliveries', fontsize=9)
    ax.set_xlim(0, T)
    ax.yaxis.grid(True, alpha=0.3)
    if pi == 0:
        ax.legend(fontsize=9, framealpha=0.95, loc='upper left')

# Panel 5: Throughput retention vs N (% of Full-Graph)
N_LIST = [s[2] for s in SCALES]
ax5 = fig.add_subplot(3, 3, 5)
ax5.set_facecolor(BG)
for sp in ['top', 'right']:
    ax5.spines[sp].set_visible(False)
for cond in SHOW_CONDS:
    retentions = []
    for scale, _, _ in SCALES:
        fg_m = agg(scale, 'full', 1)[0][-1]
        mn_m = agg(scale, cond, 1)[0][-1]
        retentions.append(100.0 * mn_m / max(fg_m, 0.01))
    ax5.plot(N_LIST, retentions, color=COLORS[cond], lw=LINEWIDTHS[cond],
             linestyle=LINESTYLES[cond], marker='o', ms=7,
             label=LABELS[cond], zorder=4 if cond == 'morph_v2' else 2)
ax5.axhline(100, color='#888888', lw=0.8, linestyle=':', alpha=0.6)
ax5.set_xlabel('Number of agents (N)', fontsize=9)
ax5.set_ylabel('Deliveries as % of Full-Graph', fontsize=9)
ax5.set_title('Throughput Retention vs Scale', fontsize=10, fontweight='bold')
ax5.set_xticks(N_LIST)
ax5.legend(fontsize=9, framealpha=0.95)
ax5.yaxis.grid(True, alpha=0.3)

# Panel 6: Link efficiency vs N
ax6 = fig.add_subplot(3, 3, 6)
ax6.set_facecolor(BG)
for sp in ['top', 'right']:
    ax6.spines[sp].set_visible(False)
for cond in SHOW_CONDS:
    link_pcts = [100.0 * mean_links(s[0], cond) / (s[2]*(s[2]-1)//2)
                 for s in SCALES]
    ax6.plot(N_LIST, link_pcts, color=COLORS[cond], lw=LINEWIDTHS[cond],
             linestyle=LINESTYLES[cond], marker='o', ms=7,
             label=LABELS[cond], zorder=4 if cond == 'morph_v2' else 2)
ax6.set_xlabel('Number of agents (N)', fontsize=9)
ax6.set_ylabel('Links used (% of max possible)', fontsize=9)
ax6.set_title('Communication Overhead vs Scale', fontsize=10, fontweight='bold')
ax6.set_xticks(N_LIST)
ax6.legend(fontsize=9, framealpha=0.95)
ax6.yaxis.grid(True, alpha=0.3)

# Panels 7-9: per-scale bar charts (bottom row, 4 bars each)
for pi, (scale, _, N_exp) in enumerate(SCALES):
    ax = fig.add_subplot(3, 4, 9 + pi)
    ax.set_facecolor(BG)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    fg_m  = agg(scale, 'full', 1)[0][-1]
    vals  = [final_del(scale, c).mean() for c in SHOW_CONDS]
    errs  = [final_del(scale, c).std()  for c in SHOW_CONDS]
    cols  = [COLORS[c] for c in SHOW_CONDS]
    bars  = ax.bar(range(len(SHOW_CONDS)), vals, color=cols, alpha=0.85,
                   edgecolor='white', linewidth=1.0, width=0.6, zorder=3)
    ax.errorbar(range(len(SHOW_CONDS)), vals, yerr=errs, fmt='none',
                color='#333333', capsize=3, lw=1.2, zorder=4)
    for bar, v, e, col in zip(bars, vals, errs, cols):
        pct = 100.0 * v / max(fg_m, 0.01)
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.3,
                f'{v:.0f}\n({pct:.0f}%)', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', color=col)
    # Highlight MORPH bar
    morph_idx = SHOW_CONDS.index('morph_v2')
    bars[morph_idx].set_edgecolor(COLORS['morph_v2'])
    bars[morph_idx].set_linewidth(2.5)
    ax.set_xticks(range(len(SHOW_CONDS)))
    ax.set_xticklabels([LABELS[c] for c in SHOW_CONDS],
                       fontsize=8, rotation=25, ha='right')
    ax.set_title(f'{scale.upper()} (N={N_exp})', fontsize=9, fontweight='bold')
    ax.set_ylabel('Deliveries' if pi == 0 else '', fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)

fig.suptitle(f'MORPH: Scale Study — Delivery Performance and Communication Overhead\n'
             f'T={T}  ·  {N_seeds} seeds  ·  4 scales',
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
out1 = os.path.join(FIG_DIR, 'scale_comparison.png')
fig.savefig(out1, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out1}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Grouped bar chart — all 4 conditions × all 4 scales
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Scale bar chart ...")
fig2, ax2 = plt.subplots(figsize=(12, 5), facecolor=BG)
ax2.set_facecolor(BG)
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)

n_conds  = len(SHOW_CONDS)
n_scales = len(SCALES)
width    = 0.18
x        = np.arange(n_scales)

for ci, cond in enumerate(SHOW_CONDS):
    offset = (ci - (n_conds - 1) / 2) * (width + 0.02)
    vals   = [final_del(s[0], cond).mean() for s in SCALES]
    errs   = [final_del(s[0], cond).std()  for s in SCALES]
    bars   = ax2.bar(x + offset, vals, width, color=COLORS[cond], alpha=0.85,
                     label=LABELS[cond], edgecolor='white', linewidth=0.8, zorder=3)
    ax2.errorbar(x + offset, vals, yerr=errs, fmt='none',
                 color='#333333', capsize=3, lw=1.2, zorder=4)

ax2.set_xticks(x)
ax2.set_xticklabels([f'{s[0].upper()}\n(N={s[2]})' for s in SCALES], fontsize=11)
ax2.set_ylabel('Final deliveries (mean ± std over 5 seeds)', fontsize=10)
ax2.set_title(f'Delivery Performance Across Scales  (T={T})',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.95, loc='upper left')
ax2.yaxis.grid(True, alpha=0.3, zorder=0)
fig2.tight_layout()
out2 = os.path.join(FIG_DIR, 'scale_bars.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out2}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Pareto — communication cost vs throughput
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Pareto ...")
fig3, axes3 = plt.subplots(1, 4, figsize=(18, 5), facecolor=BG)
fig3.patch.set_facecolor(BG)

for ax, (scale, _, N_exp) in zip(axes3, SCALES):
    ax.set_facecolor(BG)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ml   = N_exp * (N_exp - 1) // 2
    fg_m = agg(scale, 'full', 1)[0][-1]

    for cond in SHOW_CONDS:
        x_val = mean_links(scale, cond) / ml
        y_val = final_del(scale, cond).mean() / max(fg_m, 0.01)
        ms    = 200 if cond == 'morph_v2' else 100
        ax.scatter(x_val, y_val, c=COLORS[cond], s=ms, zorder=4,
                   edgecolors='white', linewidths=1.5, label=LABELS[cond])
        # Label offsets per condition
        dx, dy = 0.01, 0.02
        if cond == 'full':      dx, dy = 0.01, -0.06
        elif cond == 'tsg':     dx, dy = 0.01, -0.06
        ax.annotate(LABELS[cond], (x_val, y_val),
                    xytext=(x_val + dx, y_val + dy),
                    fontsize=9, color=COLORS[cond], fontweight='bold')

    # Ideal region shading
    ax.fill_between([0, 0.35], [0.9, 0.9], [1.25, 1.25],
                    alpha=0.06, color='#2166ac')
    if ax is axes3[0]:
        ax.text(0.02, 1.23, 'ideal\n(high throughput\nlow cost)',
                fontsize=7.5, color='#2166ac', va='top')

    ax.axhline(1.0, color='#888888', lw=0.8, linestyle=':', alpha=0.5)
    ax.set_xlabel('Comm cost (links / max possible)', fontsize=9)
    ax.set_ylabel('Throughput (fraction of Full-Graph)' if ax is axes3[0] else '',
                  fontsize=9)
    ax.set_title(f'{scale.upper()}  N={N_exp}', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(0.0, 1.35)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

fig3.suptitle('MORPH: Pareto Frontier — Communication Cost vs Throughput\n'
              'Upper-left = better (high throughput, low communication overhead)',
              fontsize=12, fontweight='bold', y=1.03)
fig3.tight_layout()
out3 = os.path.join(FIG_DIR, 'pareto.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  Saved: {out3}")

print("\nAll figures saved.")
