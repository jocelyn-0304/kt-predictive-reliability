# Predictive reliability of knowledge tracing under sparse and noisy histories: code and saved predictions

This repository reproduces every number, table and figure of the paper *Predictive Reliability of Knowledge Tracing
Models Under Sparse and Noisy Student Response Data*. All data are simulated from two generators; no real learner
data are used or needed.

## Quick start

```bash
pip install -r requirements.txt
bash reproduce.sh quick     # about 5 minutes: metrics, bootstrap intervals, Tables 1-4 and Figures 1-3 from the saved predictions
bash reproduce.sh verify    # about 40 minutes on 2 CPU cores: rerun everything from the seeds and check it against the saved predictions
```

`bash reproduce.sh full` reruns from the seeds without the comparison (it needs empty `results/` and `results_ctrl/`); use it when the saved predictions are not included in your copy.

## What is in the repository

| File or folder | Content |
|---|---|
| `simulate.py` | Synthetic-KT-A (two-state learning, forgetting, guessing, slipping) and Synthetic-KT-B (dynamic item response process with asymptotes); learner-level 60/15/10/15 split |
| `classical.py` | Smoothed item-accuracy baseline, BKT with forgetting (EM), 2PL-IRT (MAP) |
| `neural.py`, `autodiff.py` | DKT, DKT-item (control: skill inputs, item outputs) and the AKT-style model on a small reverse-mode autodiff engine written in NumPy; full-sequence and window training |
| `core.py` | Fixed target events (50-59), visible-history windows, label flips, temperature fitting, model construction |
| `gradcheck.py` | Finite-difference check of the gradients of all three networks |
| `tune.py`, `tune_controls.py` | Hyperparameter grids on the development replicate (seed 9001) -> `hp_*.json`, `hp_ctrl_*.json` |
| `run_experiments.py` | Main grid: 2 generators x 5 guess/slip configurations x 5 seeds -> `results/*.npz` |
| `run_controls.py` | Control comparisons of Section 3.5 (DKT-item, clean window reference, position-matched windows, early targets; calibration-window predictions for temperature and Platt scaling) -> `results_ctrl/*.npz` |
| `analyze.py`, `summary.py`, `tables.py` | Metrics, learner-level bootstrap, paired differences, temperature scaling, ECE bin sensitivity -> `summary_full.txt`, `tables.json` |
| `summary_controls.py` | Section 3.5 and Table 4 -> `summary_controls.txt`, `tables_ctrl.json` |
| `make_tables.py`, `paper_tables.py` | Tables 1-4 as CSV and Markdown, laid out as in the paper -> `tables/` |
| `fig1.py`, `fig23.py` | Figures 1-3 -> `figs/` |
| `verify_predictions.py` | Bitwise comparison of two sets of saved predictions |
| `results/`, `results_ctrl/` | Saved test and calibration predictions of every model and condition (NumPy `.npz`) |

## Map from the paper to the code

| Paper | Numbers come from | Produced by |
|---|---|---|
| Table 1 | `run_experiments.py`, `simulate.py`, `hp_*.json`, `hp_ctrl_*.json` | (configuration) |
| Table 2, Figure 3, Sections 3.1-3.2 | `summary_full.txt`, `tables.json` | `summary.py`, `tables.py`, `fig23.py` |
| Table 3, Sections 3.3-3.4 | `summary_full.txt`, `tables.json` | `summary.py`, `tables.py` |
| Figure 2 | `results/` | `fig23.py` |
| Table 4, Section 3.5 | `summary_controls.txt`, `tables_ctrl.json` | `summary_controls.py` |
| Figure 1 | | `fig1.py` |

## Reproducibility

Every random draw is fixed by a seed (simulation, splits, initialisation, minibatch order, label flips, bootstrap).
With the library versions below, rerunning from the seeds reproduces the saved predictions exactly: `run_controls.py`
refits the test-only models and the clean window reference and compares 102 prediction arrays per replicate with
`results/`; the maximum absolute difference was 0 in all ten replicates. Other library versions may change the last
digits of some numbers.

Environment used for the paper: Python 3.13, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, matplotlib 3.11.2
(`requirements.txt`). No GPU is needed.

## Design constants

- 3,000 learners x 60 interactions per replicate, 10 skills x 20 items, seeds 1-5; development seed 9001.
- Target events 50-59 of each test learner (4,500 per replicate), identical for every condition.
- History length L in {3, 5, 10, 20, 50}; history label flips eta in {0, 0.05, 0.10, 0.20}; guess g in {0.10, 0.20, 0.30}, slip s in {0.05, 0.15, 0.25}.
- Full-sequence training supervises 108,000 predictions per epoch (1,800 learners x 60 positions); window training supervises 6,000 windows per epoch.
