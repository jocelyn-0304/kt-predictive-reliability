import sys, json
import numpy as np, pandas as pd
from analyze import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
runs = load_runs()
seeds = sorted({k[2] for k in runs if k[1] == "center"})
print("seeds complete for center:", seeds, "| runs:", len(runs))
df = long_table(runs)
df.to_csv("long_table.csv", index=False)
ms = lambda s: f"{s.mean():.3f}({s.std(ddof=1) if len(s) > 1 else 0:.3f})"
C = df[(df.cfg == "center")]

print("\n=== S1: center, eta=0, raw — mean(SD across seeds)")
for met in ("auc", "brier", "nll", "ece10"):
    t = C[(C.setting == "S1") & (C.eta == 0) & (C.variant == "raw")].pivot_table(index=["mech", "model"], columns="L", values=met, aggfunc=ms, sort=False)
    print("--", met); print(t)

B = 500
print("\n=== bootstrap CI (pooled over seeds): Brier/NLL/AUC/ECE for L=3 and L=50, eta=0, raw")
for mech in "AB":
    for L in (3, 50):
        for mo in MODELS:
            r = pooled_boot(runs, mech, "center", "S1", mo, L, 0.0, "raw", B=B)
            print(mech, L, f"{mo:9s}", {k: tuple(round(x, 3) for x in v) for k, v in r.items()})

print("\n=== paired: Brier(model) - Brier(Baseline), eta=0, raw")
for mech in "AB":
    for L in (3, 50):
        for mo in MODELS[1:]:
            r = pooled_boot(runs, mech, "center", "S1", mo, L, 0.0, "raw", B=B, metrics=("brier", "nll", "auc"),
                            ref=dict(setting="S1", model="Baseline", L=L, eta=0.0, variant="raw"))
            print(mech, L, f"{mo:9s}", {k: tuple(round(x, 4) for x in v) for k, v in r.items()})

print("\n=== paired: Brier(L=3) - Brier(L=50) within model (history-length effect), raw")
for mech in "AB":
    for mo in MODELS:
        r = pooled_boot(runs, mech, "center", "S1", mo, 3, 0.0, "raw", B=B, metrics=("brier", "nll", "auc", "ece10"),
                        ref=dict(setting="S1", model=mo, L=50, eta=0.0, variant="raw"))
        print(mech, f"{mo:9s}", {k: tuple(round(x, 4) for x in v) for k, v in r.items()})

print("\n=== matched vs test-only (neural), eta=0: Brier/NLL/ECE difference (matched - test-only)")
for mech in "AB":
    for mo in ("DKT", "AKT"):
        for L in (3, 5, 10, 20, 50):
            r = pooled_boot(runs, mech, "center", "M", mo, L, 0.0, "raw", B=B, metrics=("brier", "nll", "ece10", "auc"),
                            ref=dict(setting="S1", model=mo, L=L, eta=0.0, variant="raw"))
            print(mech, mo, L, {k: tuple(round(x, 4) for x in v) for k, v in r.items()})

print("\n=== label flips eta (test-only S1), center, raw: mean(SD) Brier / ECE10 / AUC / hc_err / hc_cov")
for L in (50, 3):
    for met in ("auc", "brier", "ece10", "hc_err", "hc_cov"):
        t = C[(C.setting == "S1") & (C.L == L) & (C.variant == "raw")].pivot_table(index=["mech", "model"], columns="eta", values=met, aggfunc=ms, sort=False)
        print(f"-- L={L} {met}"); print(t)
print("\n=== matched eta (M, L=50): mean(SD)")
for met in ("auc", "brier", "ece10", "hc_err", "hc_cov"):
    t = C[(C.setting == "M") & (C.L == 50) & (C.variant == "raw")].pivot_table(index=["mech", "model"], columns="eta", values=met, aggfunc=ms, sort=False)
    print("--", met); print(t)

print("\n=== g/s grid at L=50 and L=3 (eta=0, raw): mean(SD)")
G = df[(df.setting == "S1") & (df.eta == 0) & (df.variant == "raw")]
for L in (50, 3):
    for met in ("auc", "brier", "ece10", "hc_err", "hc_cov", "base"):
        t = G[G.L == L].pivot_table(index=["mech", "model"], columns="cfg", values=met, aggfunc=ms, sort=False)
        print(f"-- L={L} {met}"); print(t)

print("\n=== temperature scaling: NLL & ECE10 by variant; T; AUC change")
for (L, eta) in ((50, 0.0), (3, 0.0), (50, 0.20), (3, 0.20)):
    X = C[(C.setting == "S1") & (C.L == L) & (C.eta == eta)]
    for met in ("nll", "ece10", "hc_err"):
        t = X.pivot_table(index=["mech", "model"], columns="variant", values=met, aggfunc=ms, sort=False)
        print(f"-- L={L} eta={eta} {met}"); print(t)
    t = X[X.variant != "raw"].pivot_table(index=["mech", "model"], columns="variant", values="T", aggfunc=ms, sort=False)
    print("-- T"); print(t)
raw = C[(C.setting == "S1") & (C.variant == "raw")].set_index(["mech", "seed", "model", "L", "eta"]).auc
for v in ("ts_target", "ts_ref"):
    tv = C[(C.setting == "S1") & (C.variant == v)].set_index(["mech", "seed", "model", "L", "eta"]).auc
    print("max |AUC change| for", v, (tv - raw.loc[tv.index]).abs().max())

print("\n=== paired TS effect (ts_target - raw) NLL/ECE with CI, center eta=0.2/L=3 etc.")
for mech in "AB":
    for (L, eta) in ((50, 0.0), (3, 0.0), (50, 0.2)):
        for mo in MODELS:
            out = {}
            for v in ("ts_target", "ts_ref"):
                r = pooled_boot(runs, mech, "center", "S1", mo, L, eta, v, B=B, metrics=("nll", "ece10"),
                                ref=dict(setting="S1", model=mo, L=L, eta=eta, variant="raw"))
                out[v] = {k: tuple(round(x, 4) for x in y) for k, y in r.items()}
            print(mech, L, eta, f"{mo:9s}", out)

print("\n=== ECE bin sensitivity (raw, center, eta=0): ranks by ECE (1=lowest)")
for (L, eta) in ((50, 0.0), (3, 0.0), (50, 0.2)):
    X = C[(C.setting == "S1") & (C.L == L) & (C.eta == eta) & (C.variant == "raw")]
    for mech in "AB":
        t = X[X.mech == mech].groupby("model")[["ece5", "ece10", "ece15", "ece20", "ece_mass10"]].mean().round(4)
        print(f"-- {mech} L={L} eta={eta}"); print(t); print(t.rank().astype(int).T.to_string())
