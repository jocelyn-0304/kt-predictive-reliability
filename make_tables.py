"""Write the numbers of Tables 2-4 as CSV and Markdown (tables/), so they can be checked against the paper without
building the manuscript. Reads tables.json (tables.py) and tables_ctrl.json (summary_controls.py)."""
import csv, json, os

os.makedirs("tables", exist_ok=True)
T = json.load(open("tables.json"))
C = json.load(open("tables_ctrl.json"))
MODELS = ["Baseline", "BKT", "2PL-IRT", "DKT", "AKT"]
NM = {"AKT": "AKT-style"}
GEN = {"A": "Synthetic-KT-A", "B": "Synthetic-KT-B"}


def write(name, header, rows):
    with open(f"tables/{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(f"tables/{name}.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")


ms = lambda v: f"{v[0]:.3f} ({v[1]:.3f})"
# Table 2: mean (SD) over five replicates, clean histories
rows = []
for m in "AB":
    for mo in MODELS:
        r = [GEN[m], NM.get(mo, mo)]
        for L in (3, 50):
            v = T["T2"][f"{m}|{mo}|{L}"]
            r += [ms(v[k]) for k in ("auc", "brier", "nll", "ece10")]
        rows.append(r)
write("table2", ["Generator", "Model"] + [f"{k} L={L}" for L in (3, 50) for k in ("AUC", "Brier", "NLL", "ECE")], rows)
# Table 3A: pooled high-confidence predictions
conds = {"c1": "L=50 eta=0", "c2": "L=3 eta=0", "c3": "L=50 eta=0.20", "c4": "L=50 eta=0 s=0.05"}
rows = []
for m in "AB":
    for mo in MODELS:
        r = [GEN[m], NM.get(mo, mo)]
        for c in conds:
            v = T["T3A"][f"{m}|{mo}|{c}"]
            r += [v["n"], "NA" if v["err"] is None else f"{v['err']:.3f}"]
        rows.append(r)
write("table3a", ["Generator", "Model"] + [f"{h} {c}" for c in conds.values() for h in ("n", "error")], rows)
# Table 3B: ECE before/after temperature scaling
rows = []
for m in "AB":
    for mo in MODELS:
        r = [GEN[m], NM.get(mo, mo)]
        for (L, e) in ((3, 0.0), (3, 0.2), (50, 0.2)):
            r += [f"{T['T3B'][f'{m}|{mo}|{L}|{e}|{v}']['ece'][0]:.3f}" for v in ("raw", "ts_target", "ts_ref")]
        rows.append(r)
write("table3b", ["Generator", "Model"] + [f"{v} L={L} eta={e}" for (L, e) in ((3, 0.0), (3, 0.2), (50, 0.2)) for v in ("raw", "TS-target", "TS-ref")], rows)
# Table 1 (configuration) and Table 4 (control comparisons), laid out as in the paper
import paper_tables as PT
write("table1", PT.TABLE1[0], PT.TABLE1[1:])
ra, sa, rb, sb = PT.table4_rows(C)
write("table4a", [f"{a} {b}".strip() for a, b in zip(["Model", "L=3", "L=3", "L=3", "L=50", "L=50", "L=50"], ra[1])], ra[2:])
write("table4b", [f"{a} {b}".strip() for a, b in zip(["Model", "Offset", "Offset", "Offset", "Offset", "ECE", "ECE", "ECE", "ECE"], rb[1])], rb[2:])
print("wrote", sorted(os.listdir("tables")))
