"""Small, predefined hyper-parameter grids per model family, selected on the validation learners of an
independent development dataset (seed 9001). Criterion: NLL at fixed targets 50-59 with L=50 clean history."""
import sys, json, time, itertools
import numpy as np
from core import *

mech = sys.argv[1]
seed = 9001
D = simulate(mech, seed)
sp = split_students(seed, 3000)
tr, va = subset(D, sp["train"]), subset(D, sp["val"])
Wv = windows(D, sp["val"], 50, 0.0, seed, "val")
yv = Wv[3]
res = {"mech": mech, "seed": seed, "grids": {}}
t0 = time.time()

def val_nll(p): return nll(p, yv)

g = {}
g["baseline"] = {a: val_nll(ItemMean(a).fit(*tr, D["nI"]).predict(*Wv[:3])) for a in (2.0, 10.0, 50.0)}
g["bkt"] = {b: val_nll(BKT(bound=b).fit(*tr, D["nI"]).predict(*Wv[:3])) for b in (0.3, 0.5)}
g["irt"] = {s: val_nll(IRT2PL(sd_loga=s).fit(*tr, D["nI"]).predict(*Wv[:3])) for s in (0.3, 0.5, 1.0)}
dk, ak = {}, {}
for H, lr in itertools.product((16, 32, 64), (3e-3, 1e-2)):
    m = DKT(D["K"], D["nI"], E=24, H=H, seed=seed)
    r = train_neural(m, tr, va, dict(lr=lr, epochs=40, bs=64, patience=5, wd=1e-5), seed, mode="full")
    dk[f"{H}|{lr}"] = dict(nll=val_nll(predictor(m)(Wv)), epochs=r["epochs_run"])
for E, lr in itertools.product((16, 24, 32), (3e-3, 1e-2)):
    m = AKT(D["K"], D["nI"], E=E, hid=32, seed=seed)
    r = train_neural(m, tr, va, dict(lr=lr, epochs=40, bs=64, patience=5, wd=1e-5), seed, mode="full")
    ak[f"{E}|{lr}"] = dict(nll=val_nll(predictor(m)(Wv)), epochs=r["epochs_run"])
g["dkt"], g["akt"] = dk, ak
res["grids"] = g
best = {}
best["alpha"] = min(g["baseline"], key=g["baseline"].get)
best["bkt_bound"] = min(g["bkt"], key=g["bkt"].get)
best["irt_sd"] = min(g["irt"], key=g["irt"].get)
bd = min(dk, key=lambda k: dk[k]["nll"]); ba = min(ak, key=lambda k: ak[k]["nll"])
best["dkt_H"], best["dkt_lr"] = int(bd.split("|")[0]), float(bd.split("|")[1])
best["akt_E"], best["akt_lr"] = int(ba.split("|")[0]), float(ba.split("|")[1])
res["best"] = best
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"hp_{mech}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
