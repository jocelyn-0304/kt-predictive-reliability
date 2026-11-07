# Predictive reliability of knowledge tracing under sparse and noisy histories: code

All results in the paper come from this code on simulated data. No real dataset is used.

## Requirements
Python 3 with the packages in `requirements.txt` (`pip install -r requirements.txt`). No deep learning framework is needed: DKT and the AKT-style model run on a small numpy reverse-mode autodiff engine (`autodiff.py`). Figure 1 uses the "Liberation Serif" font (falls back to a default font if missing).

## Order of use
| Step | Script | What it does | Output |
|---|---|---|---|
| 0 | `gradcheck.py` | Finite-difference check of the autodiff engine (DKT, AKT-style) | printed max relative error |
| 1 | `simulate.py` | Synthetic-KT-A (two-state learning/forgetting/guess/slip) and Synthetic-KT-B (dynamic 2PL-type IRT with asymptotes); learner-level 60/15/10/15 split | used by steps 2-3 |
| 2 | `tune.py` | Small hyperparameter grids on a development replicate (seed 9001) | `hp_A.json`, `hp_B.json` |
| 3 | `run_experiments.py` | Fits Baseline, BKT, 2PL-IRT, DKT, AKT-style; evaluates history lengths, label flips, guess/slip configurations; matched training for the neural models; temperature fits | `results/{A,B}_{config}_{seed}.npz` (50 files) |
| 4 | `summary.py` (uses `analyze.py`) | AUC, Brier, NLL, ECE, high-confidence errors, learner-level bootstrap, paired differences, temperature scaling, ECE bin sensitivity | `long_table.csv`, printed summary |
| 5 | `tables.py` | Numbers for Table 2 (model comparison) and Table 3 (high-confidence errors; ECE before/after temperature scaling) | `tables.json` |
| 6 | `fig1.py`, `fig23.py` | Figure 1 (workflow), Figure 2 (reliability diagrams), Figure 3 (metrics vs history length) | `figs/*.png` |

Table 1 is the experimental configuration; its values are defined in `run_experiments.py` (grid constants), `simulate.py` (generator parameters) and `hp_A.json`/`hp_B.json` (selected hyperparameters).

| 7 | `build_docx.py` (+ `docbuild.py`, `omml.py`, `text_*.py`) | Builds Table 1 (configuration), Table 2 and Table 3 from `tables.json`, inserts Figures 1-3, and writes the Word file | `paper.docx` |

`run_all.sh` runs steps 0-7. Step 7 additionally needs python-docx, lxml and pandoc (for editable equations). Core modules: `core.py` (windows, targets, temperature fit, model construction), `classical.py`, `neural.py`, `autodiff.py`.

## Design constants
- 3,000 learners x 60 interactions, 10 skills x 20 items, five seeds (1-5); target events 50-59.
- History length L in {3, 5, 10, 20, 50}; history label flips in {0, 0.05, 0.10, 0.20}; guess g in {0.10, 0.20, 0.30}, slip s in {0.05, 0.15, 0.25}.
- Seeds fix every random draw, so reruns on the same library versions reproduce the saved predictions.
