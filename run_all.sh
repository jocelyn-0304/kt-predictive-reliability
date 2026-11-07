#!/usr/bin/env bash
# Full pipeline, in order. Run from this directory. Requires python3 with numpy, scipy, pandas, matplotlib.
# Total time on 2 CPU cores: about 60-70 minutes (step 3 dominates).
set -euo pipefail
mkdir -p results figs logs

# 0. Verify the numpy autodiff engine against finite differences (DKT and AKT-style)
python3 gradcheck.py

# 1. Simulators (Synthetic-KT-A and Synthetic-KT-B) are in simulate.py and are called by steps 2-3.
#    The models are in classical.py (Baseline, BKT, 2PL-IRT) and neural.py (DKT, AKT-style).

# 2. Hyperparameter selection on the independent development replicate (seed 9001) -> hp_A.json, hp_B.json
python3 tune.py A > logs/tune_A.log
python3 tune.py B > logs/tune_B.log

# 3. Full experiment grid: 2 mechanisms x 5 configurations x 5 seeds = 50 runs -> results/*.npz
#    (two parallel workers; each run is resumable because finished files are skipped)
python3 run_experiments.py 0 2 > logs/w0.log 2>&1 &
python3 run_experiments.py 1 2 > logs/w1.log 2>&1 &
wait

# 4. Metrics, learner-level bootstrap, temperature scaling, ECE sensitivity -> long_table.csv, summary_full.txt
python3 summary.py > summary_full.txt

# 5. Table 1-3 inputs (Table 1 is configuration text; Tables 2 and 3 numbers) -> tables.json
python3 tables.py

# 6. Figures -> figs/fig1_workflow.png, figs/fig2_reliability.png, figs/fig3_history_length.png
python3 fig1.py
python3 fig23.py

# 7. Assemble the Word file with Tables 1-3 and Figures 1-3 (needs python-docx, lxml, pandoc)
python3 build_docx.py paper.docx
