import json, sys
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docbuild import *
import text_front as F, text_back as B, text_refs as R

TJ = json.load(open("tables.json"))
MODELS = ["Baseline", "BKT", "2PL-IRT", "DKT", "AKT"]
MN = {"AKT": "AKT-style", "Baseline": "Baseline"}
nm = lambda m: MN.get(m, m)
GEN = {"A": "KT-A", "B": "KT-B"}
d = new_doc()

# ---- footer page number
sec = d.sections[0]
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run()
for t, txt in (("begin", None), (None, " PAGE "), ("end", None)):
    if t:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); r._r.append(e)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)
r.font.name = FONT; r.font.size = Pt(10)

title(d, F.TITLE)
author_line(d, F.AUTHORS)
label_para(d, "Abstract. ", B.ABSTRACT)
label_para(d, "Keywords: ", F.KEYWORDS)

def f3(x): return f"{x:.3f}"

def table1():
    rows = [["Component", "Setting"],
            ["Learners and items", "3,000 learners × 60 interactions per replicate; 200 items in 10 skills; five replicates (seeds 1–5); separate development replicate (seed 9001) for hyperparameters"],
            ["Splits (by learner)", "Training 60% (1,800), validation 15% (450), calibration 10% (300), test 15% (450)"],
            ["Target events", "Events 50–59 of each test learner (4,500 per replicate), identical for every condition"],
            ["Generators", "Synthetic-KT-A: two-state learning, forgetting, guessing, slipping (Eq. 1). Synthetic-KT-B: dynamic item response process with asymptotes (Eq. 2)"],
            ["Guess g / slip s", "Centre (0.20, 0.15); g = 0.10, 0.30 with s = 0.15; s = 0.05, 0.25 with g = 0.20"],
            ["History length L", "3, 5, 10, 20, 50 (centre configuration); 3, 10, 50 (other configurations)"],
            ["History label flips η", "0, 0.05, 0.10, 0.20 (centre configuration); 0, 0.10, 0.20 at L = 50 (other configurations)"],
            ["Training regimes", "Test-only shift (all models); matched training (DKT, AKT-style; centre configuration)"],
            ["Baseline", "Smoothed item accuracy; α = 50 (KT-A), 10 (KT-B)"],
            ["BKT with forgetting", "Per-skill initial mastery, learning, forgetting (≤ 0.3), guess, slip; upper bound on g and s 0.3 (KT-A), 0.5 (KT-B)"],
            ["2PL-IRT", "Item discrimination and difficulty (MAP); SD of log-discrimination 0.3 (both)"],
            ["DKT", "GRU; token embedding 24; hidden size 32 (KT-A), 64 (KT-B); learning rate 0.01"],
            ["AKT-style", "Rasch-type embeddings 16 (both); attention with distance decay; head width 32; learning rate 0.01"],
            ["Neural training", "Adam; batch 64; weight decay 1e-5; gradient clipping 5; at most 40 epochs; early stopping (patience 5) on validation NLL"]]
    caption(d, "Table 1. Experimental conditions and model configurations [Data mode: simulated; configuration of the executed experiments].")
    table(d, rows, [4.0, 12.66], size=9, align_first="left")
    d.add_paragraph().paragraph_format.space_after = Pt(4)

def table2():
    hdr0 = ["Generator", "Model", "L = 3", "", "", "", "L = 50", "", "", ""]
    hdr1 = ["", ""] + ["AUC", "Brier", "NLL", "ECE"] * 2
    rows = [hdr0, hdr1]
    for mech in "AB":
        for mo in MODELS:
            row = [GEN[mech] if mo == "Baseline" else "", nm(mo)]
            for L in (3, 50):
                v = TJ["T2"][f"{mech}|{mo}|{L}"]
                for k in ("auc", "brier", "nll", "ece10"):
                    row.append(f"{v[k][0]:.3f} ({v[k][1]:.3f})")
            rows.append(row)
    caption(d, "Table 2. Predictive discrimination and calibration across models with clean histories (η = 0), at visible history lengths L = 3 and L = 50 [Data mode: computed on simulated data; mean (SD) over five replicates].")
    table(d, rows, [1.4, 1.9] + [1.67] * 8, size=7.5, header_rows=2, merge_runs=[(0, 2, 5), (0, 6, 9)], keep=True)
    note(d, "AUC = ROC-AUC; NLL = negative log-likelihood; ECE = expected calibration error (10 equal-width bins). Targets are events 50–59 of 450 test learners per replicate. Lower is better for Brier, NLL, and ECE.")

def table3():
    conds = [("c1", "L = 50, η = 0"), ("c2", "L = 3, η = 0"), ("c3", "L = 50, η = 0.20"), ("c4", "L = 50, η = 0, s = 0.05")]
    hdr0 = ["Model"]; hdr1 = [""]
    for _, lab in conds:
        hdr0 += [lab, ""]; hdr1 += ["n", "Error"]
    rows = [hdr0, hdr1]; spans = [(0, 1, 2), (0, 3, 4), (0, 5, 6), (0, 7, 8)]
    for mech in "AB":
        rows.append([{"A": "Synthetic-KT-A", "B": "Synthetic-KT-B"}[mech]] + [""] * 8); spans.append((len(rows) - 1, 0, 8))
        for mo in MODELS:
            row = [nm(mo)]
            for c, _ in conds:
                v = TJ["T3A"][f"{mech}|{mo}|{c}"]
                row += [f"{v['n']:,}", "NA" if v["err"] is None else f3(v["err"])]
            rows.append(row)
    caption(d, "Table 3. High-confidence errors and calibration changes under sparse or noisy conditions [Data mode: computed on simulated data]. Panel A: high-confidence predictions (confidence ≥ 0.90).")
    table(d, rows, [3.2] + [1.69] * 8, size=8, header_rows=2, merge_runs=spans, keep=True)
    note(d, "Panel A. n = number of predictions with confidence max(p, 1 − p) ≥ 0.90 among 22,500 target events (five replicates pooled); Error = proportion of these predictions whose predicted class was wrong; NA = no such prediction. Coverage is n / 22,500. Baseline is the smoothed item-accuracy model.")
    conds2 = [(3, 0.0, "L = 3, η = 0"), (3, 0.2, "L = 3, η = 0.20"), (50, 0.2, "L = 50, η = 0.20")]
    hdr0 = ["Model"]; hdr1 = [""]; sp = []
    for k, (_, _, lab) in enumerate(conds2):
        hdr0 += [lab, "", ""]; hdr1 += ["Raw", "TS-target", "TS-ref"]; sp.append((0, 1 + 3 * k, 3 + 3 * k))
    rows = [hdr0, hdr1]
    for mech in "AB":
        rows.append([{"A": "Synthetic-KT-A", "B": "Synthetic-KT-B"}[mech]] + [""] * 9); sp.append((len(rows) - 1, 0, 9))
        for mo in MODELS:
            row = [nm(mo)]
            for (L, e, _) in conds2:
                for var in ("raw", "ts_target", "ts_ref"):
                    row.append(f3(TJ["T3B"][f"{mech}|{mo}|{L}|{e}|{var}"]["ece"][0]))
            rows.append(row)
    p = d.add_paragraph(); p.paragraph_format.space_before = Pt(2); p.paragraph_format.keep_with_next = True
    rich(p, "Panel B. Expected calibration error (10 bins) before and after temperature scaling.", 10.5, True)
    table(d, rows, [2.6] + [1.56] * 9, size=8, header_rows=2, merge_runs=sp, keep=True)
    note(d, "Panel B. ECE, mean over five replicates. Raw = unscaled; TS-target = temperature fitted on calibration windows with the same L and η as the test windows; TS-ref = temperature fitted on clean L = 50 windows and applied unchanged. The baseline ignores history, so its TS-target and TS-ref columns coincide.")

FIGS = {"fig1": ("figs/fig1_workflow.png", 15.0, "Figure 1. Study design and evaluation workflow [Data mode: schematic of the executed simulation protocol]."),
        "fig2": ("figs/fig2_reliability.png", 16.0, "Figure 2. Reliability diagrams under representative conditions: clean full history (L = 50), three visible events (L = 3), and 20% history flips (L = 50), for Synthetic-KT-A (top) and Synthetic-KT-B (bottom). Predictions are pooled over five replicates; bins with fewer than 50 events are omitted [Data mode: computed on simulated data]."),
        "fig3": ("figs/fig3_history_length.png", 16.0, "Figure 3. Change in AUC, Brier score, NLL, and ECE with visible history length (η = 0). Points are means of the five replicate estimates; bars are 95% learner-level bootstrap half-widths [Data mode: computed on simulated data].")}

def emit(blocks):
    for x in blocks:
        k = x[0]
        if k == "h1": heading(d, x[1], 1)
        elif k == "h2": heading(d, x[1], 2)
        elif k == "p":
            indent = not (len(x) > 2 and x[2] is False) and not x[1].startswith("where ")
            p = body(d, x[1], indent)
            if len(x) > 2 and x[2] is False: p.paragraph_format.left_indent = Pt(18)
        elif k == "eq": equation(d, x[1], x[2])
        elif k == "table1": table1()
        elif k == "table2": table2()
        elif k == "table3": table3()
        elif k in FIGS:
            png, w, cap = FIGS[k]
            figure(d, png, w); caption(d, cap, above=False)

emit(F.INTRO); emit(F.METHODS); emit(B.RESULTS); emit(B.DISCUSSION); emit(B.CONCLUSION)
heading(d, "References", 1)
for r_ in R.REFS: reference(d, r_)
out = sys.argv[1] if len(sys.argv) > 1 else "paper.docx"
d.save(out); print("saved", out)
