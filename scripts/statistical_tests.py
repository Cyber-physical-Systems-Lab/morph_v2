"""
statistical_tests.py
====================
Welch's t-tests comparing MORPH v2 against all other conditions.
Saves results to results/statistical_tests.csv.

Usage
-----
  python scripts/statistical_tests.py
"""
import sys, os, pickle, csv, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES_DIR = os.path.join(ROOT, 'results')

with open(os.path.join(RES_DIR, 'experiment_results.pkl'), 'rb') as f:
    data = pickle.load(f)

all_results = data['all_results']
SEEDS       = data['SEEDS']
SCALES      = data['SCALES']
N_seeds     = len(SEEDS)


def get_finals(scale, cond):
    return np.array([all_results[scale][cond][s][1][-1] for s in range(N_seeds)])


def welch(scale, ca, cb):
    a, b   = get_finals(scale, ca), get_finals(scale, cb)
    t, p   = stats.ttest_ind(a, b, equal_var=False)
    return dict(scale=scale, cond_A=ca, cond_B=cb,
                mean_A=float(a.mean()), mean_B=float(b.mean()),
                diff=float(a.mean()-b.mean()),
                t_stat=float(t), p_value=float(p),
                significant='YES' if p < 0.05 else 'NO')


rows = []
print(f"\n{'Scale':<8} {'Comparison':<40} {'mean_A':>7} {'mean_B':>7}"
      f" {'diff':>7} {'p':>8} {'sig':>4}")
print("-" * 80)

for scale, _, _ in SCALES:
    # MORPH v2 vs all others; skip conditions absent from results
    candidates = ['morph_v1', 'proximity', 'tsg',
                  'morph_v2_no_bcm', 'morph_v2_no_reward', 'morph_v2_no_neuro',
                  'full', 'no_coord']
    available = set(all_results.get(scale, {}).keys())
    for other in candidates:
        if other not in available:
            # Skip comparisons for conditions not present in this scale's results
            continue
        try:
            r = welch(scale, 'morph_v2', other)
        except Exception as e:
            # Don't fail the whole script for one bad comparison
            print(f"  Skipping comparison morph_v2 vs {other} on {scale}: {e}")
            continue
        rows.append(r)
        comp = f"morph_v2 vs {other}"
        sig  = '★' if r['significant'] == 'YES' else ''
        print(f"  {scale:<6}  {comp:<38}  {r['mean_A']:>7.2f}  {r['mean_B']:>7.2f}"
              f"  {r['diff']:>7.2f}  {r['p_value']:>8.4f}  {sig:>4}")
    print()

# Save
csv_path = os.path.join(RES_DIR, 'statistical_tests.csv')
fields   = ['scale', 'cond_A', 'cond_B', 'mean_A', 'mean_B',
            'diff', 't_stat', 'p_value', 'significant']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: (f'{r[k]:.4f}' if isinstance(r[k], float) else r[k])
                         for k in fields})
print(f"Saved: {csv_path}")
