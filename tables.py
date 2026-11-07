import json, numpy as np, pandas as pd
from analyze import *
runs = load_runs()
df = pd.read_csv("long_table.csv")
ms = lambda s: (float(s.mean()), float(s.std(ddof=1)))
out = {}
# Table 2: mean (SD) across 5 seeds
T2 = {}
C = df[(df.cfg == "center") & (df.setting == "S1") & (df.eta == 0) & (df.variant == "raw")]
for mech in "AB":
    for mo in MODELS:
        for L in (3, 50):
            s = C[(C.mech == mech) & (C.model == mo) & (C.L == L)]
            T2[f"{mech}|{mo}|{L}"] = {m: ms(s[m]) for m in ("auc", "brier", "nll", "ece10")}
out["T2"] = T2
# Table 3A pooled hc
conds = {"c1": ("center", 50, 0.0), "c2": ("center", 3, 0.0), "c3": ("center", 50, 0.2), "c4": ("s05", 50, 0.0)}
T3A = {}
for mech in "AB":
    for mo in MODELS:
        for cn, (cfg, L, eta) in conds.items():
            n = e = N = 0
            for (m, c, s), R in runs.items():
                if m != mech or c != cfg: continue
                key = f"S1|{mo}|L{L}|e{eta}"
                p, y = R["p"][key], R["y"]
                cc = np.maximum(p, 1 - p) >= 0.9
                n += int(cc.sum()); N += len(p)
                e += int((((p[cc] >= 0.5).astype(int)) != y[cc]).sum())
            T3A[f"{mech}|{mo}|{cn}"] = dict(n=n, err=(e / n if n else None), N=N, nerr=e)
out["T3A"] = T3A
# Table 3B: ECE raw / ts_target / ts_ref, NLL too
T3B = {}
D = df[(df.cfg == "center") & (df.setting == "S1")]
for mech in "AB":
    for mo in MODELS:
        for (L, eta) in [(3, 0.0), (3, 0.2), (50, 0.2)]:
            for var in ("raw", "ts_target", "ts_ref"):
                s = D[(D.mech == mech) & (D.model == mo) & (D.L == L) & (D.eta == eta) & (D.variant == var)]
                T3B[f"{mech}|{mo}|{L}|{eta}|{var}"] = dict(ece=ms(s.ece10), nll=ms(s.nll), T=ms(s["T"]))
out["T3B"] = T3B
# base rates and training info
info = {}
for (m, c, s), R in runs.items():
    if c == "center":
        i = R["meta"]["info"]
        info[f"{m}{s}"] = {k: i[k] for k in ("base_rate_all", "base_rate_target", "n_train", "n_val", "n_calib", "n_test")}
        for k in i:
            if k.endswith("_full"): info[f"{m}{s}"][k] = i[k]
out["info"] = info
json.dump(out, open("tables.json", "w"), indent=1)
print(json.dumps({k: out["T3A"][k] for k in out["T3A"]}, indent=0)[:3000])
for k, v in info.items():
    print(k, v["base_rate_target"], v["base_rate_all"], v["n_test"], v.get("DKT_full"), v.get("AKT_full"))
