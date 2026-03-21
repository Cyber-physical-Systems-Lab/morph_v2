# MORPH v2: Multi-agent Online Rewiring through Plasticity-guided Hierarchy

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MORPH v2 extends the original MORPH framework with four additional neuroplasticity
mechanisms, improving coordination quality and robustness over v1, Proximity, and TSG
baselines across all warehouse scales.

---

## What's new in v2

| Mechanism | Biological basis | What it adds |
|---|---|---|
| **BCM Metaplasticity** | BCM sliding threshold (BCM theory) | Strongly potentiated links decay faster; stale coordination forgotten quickly |
| **Reward-modulated Plasticity** | Dopamine-gated LTP | Links that *complete* deliveries are strengthened, not just links that co-assign |
| **Neuromodulation** | Dopamine/ACh arousal signal | System explores new links when delivery rate drops; consolidates when performing well |
| **Predictive Formation** | Anticipatory synaptogenesis | Links form before co-assignment occurs, closing the cold-start gap vs Proximity |

v1 behaviour is recovered exactly by setting all new mechanism gains to zero.

---

## Key results

*(populated after running experiments)*

| Condition | Tiny | Small | Medium | Large | Avg links |
|---|---|---|---|---|---|
| No-Coord | — | — | — | — | 0 |
| Proximity-r | — | — | — | — | — |
| TSG | — | — | — | — | — |
| MORPH v1 | — | — | — | — | — |
| MORPH v2 | — | — | — | — | — |
| Full-Graph | — | — | — | — | max |

---

## Repository structure

```
morph_v2/
├── morph/
│   ├── __init__.py
│   ├── morph.py          # MORPH v2 coordinator (all four new mechanisms)
│   └── morph_env.py      # Abstract environment and baselines (unchanged from v1)
│
├── experiments/
│   └── tarware_experiment.py   # Scale study + ablation (9 conditions, 4 scales)
│
├── scripts/
│   ├── generate_figures.py     # Figures 1–3 from experiment_results.pkl
│   └── statistical_tests.py    # Welch's t-tests, saves results/statistical_tests.csv
│
├── figures/                    # Generated outputs
├── results/                    # PKL + CSVs
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone <repo-url> morph_v2
cd morph_v2
pip install -r requirements.txt
```

### Run the full experiment (~45 min on 4 cores)

```bash
python experiments/tarware_experiment.py
```

Outputs:
- `results/experiment_results.pkl`
- `results/step_csv/` — per-step CSVs for all conditions/seeds/scales
- `results/summary_table.csv`

### Generate figures

```bash
python scripts/generate_figures.py
# → figures/scale_comparison.png
# → figures/ablation.png
# → figures/pareto.png
```

### Run statistical tests

```bash
python scripts/statistical_tests.py
# → results/statistical_tests.csv
```

---

## MORPH v2 parameters

### Inherited from v1

| Parameter | Default | Description |
|---|---|---|
| `alpha` | 0.18 | Synaptic learning rate |
| `beta` | 0.04 | Homeostatic correction rate |
| `decay` | 0.98 | Base synaptic weight decay per step |
| `theta_form_start` | 0.75 | Initial MI threshold for link formation |
| `theta_form_end` | 0.45 | Final MI threshold (after annealing) |
| `theta_prune` | 0.008 | Weight threshold for pruning |
| `target_deg_frac` | 0.35 | Target degree as fraction of N−1 |
| `grace_steps` | 20 | Min link age before eligible for pruning |
| `k_slow` | 3 | Structural update frequency |

### New in v2

| Parameter | Default | Description |
|---|---|---|
| `bcm_tau` | 0.95 | BCM threshold smoothing (higher = longer memory) |
| `bcm_gain` | 0.5 | Extra decay rate for over-potentiated links |
| `reward_alpha` | 0.08 | Learning rate for delivery-burst updates |
| `neuromod_gain` | 0.4 | Scales effective α when system performs above expectation |
| `neuromod_explore` | 0.10 | Reduces θ_form when delivery rate falls below expected |
| `neuromod_ema` | 0.03 | EMA smoothing for delivery rate (~33-step window) |
| `expected_delivery_rate` | 0.07 | Baseline deliveries/step for neuromodulation signal |
| `pred_boost` | 0.4 | Anticipatory hint weight in structural MI score |

---

## Conditions

| Condition | Description |
|---|---|
| `no_coord` | No communication (lower bound) |
| `proximity` | Manhattan distance ≤ r, r calibrated to match MORPH v2 link count |
| `tsg` | Instantaneous co-assignment graph, no memory (Task-Similarity Gate) |
| `morph_v1` | v2 code with all new mechanisms disabled — exact v1 behaviour |
| `morph_v2_no_bcm` | v2 without BCM metaplasticity |
| `morph_v2_no_reward` | v2 without reward-modulated plasticity |
| `morph_v2_no_neuro` | v2 without neuromodulation |
| `morph_v2` | Full MORPH v2 (all mechanisms enabled) |
| `full` | All-to-all graph, always respond (upper bound) |

---

## Citation

```bibtex
@inproceedings{morph2025,
  title     = {MORPH: Multi-agent Online Rewiring through Plasticity-guided Hierarchy},
  author    = {[Authors]},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
