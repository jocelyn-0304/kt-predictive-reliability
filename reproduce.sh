#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper.
#   bash reproduce.sh quick    recompute all metrics, intervals, tables and figures from the saved predictions
#                              in results/ and results_ctrl/ (about 5 minutes)
#   bash reproduce.sh full     simulate the data and fit every model from the seeds, then run the analysis
#                              (results/ and results_ctrl/ must be empty or absent; about 40-45 minutes on 2 CPU cores)
#   bash reproduce.sh verify   move the saved predictions aside, rerun everything from the seeds, check that the new
#                              predictions are identical to the saved ones, then run the analysis
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-quick}"
mkdir -p logs figs tables

experiments() {
  python3 gradcheck.py | tee logs/gradcheck.log                 # finite-difference check of the autodiff engine
  python3 tune.py A > logs/tune_A.log                           # hyperparameters on the development replicate (seed 9001)
  python3 tune.py B > logs/tune_B.log                           #   -> hp_A.json, hp_B.json
  python3 tune_controls.py A > logs/tune_ctrl_A.log             # same procedure for the control model DKT-item
  python3 tune_controls.py B > logs/tune_ctrl_B.log             #   -> hp_ctrl_A.json, hp_ctrl_B.json
  mkdir -p results results_ctrl
  python3 run_experiments.py 0 2 > logs/w0.log 2>&1 &           # main grid: 2 generators x 5 configurations x 5 seeds
  python3 run_experiments.py 1 2 > logs/w1.log 2>&1 &
  wait
  python3 run_controls.py 0 2 > logs/ctrl_w0.log 2>&1 &         # control comparisons (Section 3.5)
  python3 run_controls.py 1 2 > logs/ctrl_w1.log 2>&1 &
  wait
}

analysis() {
  python3 summary.py > summary_full.txt                         # Sections 3.1-3.4 (writes long_table.csv)
  python3 tables.py > logs/tables.log                           # Tables 2-3 -> tables.json
  python3 summary_controls.py > summary_controls.txt            # Section 3.5 and Table 4 -> tables_ctrl.json
  python3 make_tables.py                                        # Tables 2-4 as CSV and Markdown -> tables/
  python3 fig1.py                                               # Figure 1 -> figs/fig1_workflow.png
  python3 fig23.py                                              # Figures 2-3 -> figs/fig2_reliability.png, figs/fig3_history_length.png
}

case "$MODE" in
  quick)
    analysis ;;
  full)
    if [ -n "$(ls -A results 2>/dev/null)" ] || [ -n "$(ls -A results_ctrl 2>/dev/null)" ]; then
      echo "results/ or results_ctrl/ is not empty; move them away first, or use: bash reproduce.sh verify"; exit 1
    fi
    experiments; analysis ;;
  verify)
    mv results results_published
    mv results_ctrl results_ctrl_published
    experiments
    python3 verify_predictions.py results_published results
    python3 verify_predictions.py results_ctrl_published results_ctrl
    analysis ;;
  *)
    echo "usage: bash reproduce.sh [quick|full|verify]"; exit 1 ;;
esac
echo "done ($MODE)"
