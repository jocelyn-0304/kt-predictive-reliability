"""Table 1 rows and the Table 4 (control comparisons) layout, shared by build_ieee.py and build_jedm.py."""

TABLE1_CAPTION = "Table 1. Experimental conditions and model configurations [Data mode: simulated; configuration of the executed experiments]."
TABLE1 = [
    ["Component", "Setting"],
    ["Learners and items", "3,000 learners × 60 interactions per replicate; 200 items in 10 skills; five replicates (seeds 1–5); separate development replicate (seed 9001) for hyperparameters"],
    ["Splits (by learner)", "Training 60% (1,800), validation 15% (450), calibration 10% (300), test 15% (450)"],
    ["Target events", "Events 50–59 of each test learner (events numbered 0–59, so the last ten responses; 4,500 per replicate), identical for every condition"],
    ["Generators", "Synthetic-KT-A: two-state learning, forgetting, guessing, slipping (Eq. 1). Synthetic-KT-B: dynamic item response process with asymptotes (Eq. 2)"],
    ["Guess g / slip s", "Centre (0.20, 0.15); g = 0.10, 0.30 with s = 0.15; s = 0.05, 0.25 with g = 0.20"],
    ["History length L", "3, 5, 10, 20, 50 (centre configuration); 3, 10, 50 (other configurations)"],
    ["History label flips η", "0, 0.05, 0.10, 0.20 (centre configuration); 0, 0.10, 0.20 at L = 50 (other configurations)"],
    ["Training settings", "Test-only shift (all models; full-sequence loss for DKT and AKT-style); matched training with the window loss (DKT, AKT-style; centre configuration)"],
    ["Control comparisons", "Centre configuration: DKT-item; clean window reference (L = 50, targets 50–59); position-matched windows (L = 3–20, targets 50–59); early targets (event L with its complete history)"],
    ["Baseline", "Uses the target item only. Smoothed item accuracy; α = 50 (KT-A), 10 (KT-B)"],
    ["BKT with forgetting", "Uses the target skill and the visible responses on that skill. Per-skill initial mastery, learning, forgetting (≤ 0.3), guess, slip; upper bound on g and s 0.3 (KT-A), 0.5 (KT-B)"],
    ["2PL-IRT", "Uses the identities of the visible and target items and all visible responses. Item discrimination and difficulty (MAP); SD of log-discrimination 0.3 (both)"],
    ["DKT", "Uses the skill and response of each visible event and the target skill. GRU; token embedding 24; hidden size 32 (KT-A), 64 (KT-B); learning rate 0.01"],
    ["AKT-style", "Uses skills, one difficulty value per item, and all visible responses. Rasch-type embeddings 16 (both); attention with distance decay; head width 32; learning rate 0.01"],
    ["DKT-item (control)", "As DKT, with one output per item (the target item’s entry is used). Hidden size 64 (KT-A), 16 (KT-B); learning rate 0.003"],
    ["Neural training", "Adam; batch 64; weight decay 1e-5; gradient clipping 5; at most 40 epochs; early stopping (patience 5) on validation NLL. Full-sequence loss: 108,000 predictions per epoch; window loss: 6,000 windows per epoch"],
]


# ---------------------------------------------------------------- Table 4 (control comparisons), from tables_ctrl.json
TABLE4_CAPTION = ("Table 4. Control comparisons in the centre configuration [Data mode: computed on simulated data]. "
                  "Panel A: item information (test-only models, clean histories; mean (SD) over five replicates).")
TABLE4_NOTE_A = ("Panel A. DKT-item = DKT with skill inputs and one output per item. Targets are events 50–59 with the last L events "
                 "visible. AUC = ROC-AUC; ECE = expected calibration error (10 bins).")
TABLE4_TITLE_B = "Panel B. Three visible events (L = 3, η = 0): mean offset and ECE (means over five replicates)."
TABLE4_NOTE_B = ("Panel B. Mean offset = mean predicted probability minus observed proportion correct (negative = under-prediction). "
                 "Truncated: targets are events 50–59 and only the three events before each target are visible (test-only models). "
                 "Complete: the target is event 3 of each test learner (its fourth response; events are numbered from 0), and events 0–2 are its complete record. "
                 "Matched and Position-matched: window loss with three history events and training targets at events 3–59 or at "
                 "events 50–59 (neural models only; a dash marks cells that do not apply). ECE columns refer to the test-only models on truncated "
                 "windows: Raw = unscaled; TS = temperature scaling; Platt = Platt scaling; target = fitted on calibration windows of "
                 "the same condition; ref = fitted on clean L = 50 windows.")
T4_MODELS_A = ["DKT", "DKT-item", "AKT"]
T4_MODELS_B = ["Baseline", "BKT", "2PL-IRT", "DKT", "AKT", "DKT-item"]
NM4 = {"AKT": "AKT-style"}
GEN4 = {"A": "Synthetic-KT-A", "B": "Synthetic-KT-B"}


def s3(x):
    return f"{x:.3f}".replace("-", "−")


def table4_rows(C):
    """Returns (rows_A, spans_A, rows_B, spans_B) for Table 4; C = json.load(open('tables_ctrl.json'))."""
    hdr0 = ["Model", "L = 3", "", "", "L = 50", "", ""]
    hdr1 = ["", "AUC", "Brier", "ECE", "AUC", "Brier", "ECE"]
    ra, sa = [hdr0, hdr1], [(0, 1, 3), (0, 4, 6)]
    for m in "AB":
        ra.append([GEN4[m]] + [""] * 6); sa.append((len(ra) - 1, 0, 6))
        for mo in T4_MODELS_A:
            row = [NM4.get(mo, mo)]
            for L in (3, 50):
                v = C[m][f"item|{mo}|{L}"]
                row += [f"{v[k][0]:.3f} ({v[k][1]:.3f})" for k in ("auc", "brier", "ece10")]
            ra.append(row)
    hdr0 = ["Model", "Mean offset", "", "", "", "ECE", "", "", ""]
    hdr1 = ["", "Truncated", "Complete", "Matched", "Position-matched", "Raw", "TS-target", "Platt-target", "Platt-ref"]
    rb, sb = [hdr0, hdr1], [(0, 1, 4), (0, 5, 8)]
    for m in "AB":
        rb.append([GEN4[m]] + [""] * 8); sb.append((len(rb) - 1, 0, 8))
        for mo in T4_MODELS_B:
            rec = lambda v: C[m][f"recal|{mo}|3|0.0|{v}"]
            row = [NM4.get(mo, mo), s3(rec("raw")["citl"][0]), s3(C[m][f"early|{mo}|3"]["citl_point"][0])]
            for st in ("M", "Mpos"):
                k = f"scheme|{st}|{mo}|3|0.0"
                row.append(s3(C[m][k]["citl"][0]) if k in C[m] else "–")
            row += [s3(rec(v)["ece10"][0]) for v in ("raw", "ts_target", "pl_target", "pl_ref")]
            rb.append(row)
    return ra, sa, rb, sb
