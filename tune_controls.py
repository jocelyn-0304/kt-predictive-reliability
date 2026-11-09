"""Hyperparameters for the control model DKT-item (skill inputs, item outputs), chosen exactly as for DKT in tune.py:
the same grid (hidden size 16/32/64 x learning rate 0.003/0.01), the development replicate (seed 9001), and validation NLL
at the fixed targets 50-59 with a clean L=50 history.  Usage: python3 tune_controls.py A|B  -> hp_ctrl_{A,B}.json"""
import sys, json, time, itertools
from core import *

mech = sys.argv[1]
seed = 9001
D = simulate(mech, seed)
sp = split_students(seed, 3000)
tr, va = subset(D, sp["train"]), subset(D, sp["val"])
Wv = windows(D, sp["val"], 50, 0.0, seed, "val")
t0 = time.time()
grid = {}
for H, lr in itertools.product((16, 32, 64), (3e-3, 1e-2)):
    m = DKTItem(D["K"], D["nI"], E=24, H=H, seed=seed)
    r = train_neural(m, tr, va, dict(lr=lr, epochs=40, bs=64, patience=5, wd=1e-5), seed, mode="full")
    grid[f"{H}|{lr}"] = dict(nll=nll(predictor(m)(Wv), Wv[3]), epochs=r["epochs_run"])
b = min(grid, key=lambda k: grid[k]["nll"])
res = dict(mech=mech, seed=seed, grids={"dkt_item": grid},
           best=dict(dkti_H=int(b.split("|")[0]), dkti_lr=float(b.split("|")[1])), seconds=round(time.time() - t0, 1))
json.dump(res, open(f"hp_ctrl_{mech}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
