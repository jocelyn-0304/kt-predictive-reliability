"""Run the full experimental grid. Usage: python3 run_experiments.py <worker_id> <n_workers>"""
import sys, os, json, time
import numpy as np
from core import *

N_LEARNERS, T_LEN = 3000, 60
SEEDS = [1, 2, 3, 4, 5]
CONFIGS = {"center": (0.20, 0.15), "g10": (0.10, 0.15), "g30": (0.30, 0.15), "s05": (0.20, 0.05), "s25": (0.20, 0.25)}
LS, ETAS = [3, 5, 10, 20, 50], [0.0, 0.05, 0.10, 0.20]
S1_CENTER = [(L, e) for L in LS for e in ETAS]
S1_OTHER = [(3, 0.0), (10, 0.0), (50, 0.0), (50, 0.10), (50, 0.20)]
MATCH_L, MATCH_ETA = [3, 5, 10, 20, 50], [0.05, 0.10, 0.20]
HP = {m: json.load(open(f"hp_{m}.json"))["best"] for m in "AB"}


def run_one(mech, cfg, seed):
    path = f"results/{mech}_{cfg}_{seed}.npz"
    if os.path.exists(path):
        return
    t0 = time.time()
    hp = HP[mech]
    g, s = CONFIGS[cfg]
    D = simulate(mech, seed, N=N_LEARNERS, T=T_LEN, g=g, s=s)
    sp = split_students(seed, N_LEARNERS)
    tr, va = subset(D, sp["train"]), subset(D, sp["val"])
    models, info = {}, {}
    for m in build_classical(tr, D["nI"], hp):
        models[m.name] = predictor(m)
        if m.name == "BKT":
            info["bkt_params"] = m.P.round(4).tolist()
    for name in ("DKT", "AKT"):
        t1 = time.time()
        net = new_neural(name, D["K"], D["nI"], hp, seed)
        r = train_neural(net, tr, va, neural_cfg(name, hp), seed, mode="full")
        info[f"{name}_full"] = dict(r, sec=round(time.time() - t1, 1))
        models[name] = predictor(net)
    conds = S1_CENTER if cfg == "center" else S1_OTHER
    Wref = windows(D, sp["calib"], 50, 0.0, seed, "calib")
    Tref = {n: fit_T(fn(Wref), Wref[3]) for n, fn in models.items()}
    store, Ts = {}, {}
    y_te = sidx_te = None

    def evaluate(setting, name, fn, L, eta, with_ref):
        nonlocal y_te, sidx_te
        Wte = windows(D, sp["test"], L, eta, seed, "test")
        Wca = windows(D, sp["calib"], L, eta, seed, "calib")
        if y_te is None:
            y_te, sidx_te = Wte[3], Wte[4]
        key = f"{setting}|{name}|L{L}|e{eta}"
        store["p::" + key] = fn(Wte).astype(np.float32)
        Ts[key] = dict(T_target=fit_T(fn(Wca), Wca[3]), T_ref=Tref[name] if with_ref else None)

    for (L, eta) in conds:
        for name, fn in models.items():
            evaluate("S1", name, fn, L, eta, True)
    if cfg == "center":
        for L in MATCH_L:
            for name in ("DKT", "AKT"):
                t1 = time.time()
                net = new_neural(name, D["K"], D["nI"], hp, seed)
                r = train_neural(net, tr, va, neural_cfg(name, hp), seed, mode="window", L=L, eta=0.0)
                info[f"{name}_matchL{L}"] = dict(r, sec=round(time.time() - t1, 1))
                evaluate("M", name, predictor(net), L, 0.0, False)
        for eta in MATCH_ETA:
            for name in ("DKT", "AKT"):
                t1 = time.time()
                net = new_neural(name, D["K"], D["nI"], hp, seed)
                r = train_neural(net, tr, va, neural_cfg(name, hp), seed, mode="window", L=50, eta=eta)
                info[f"{name}_matchE{eta}"] = dict(r, sec=round(time.time() - t1, 1))
                evaluate("M", name, predictor(net), 50, eta, False)
    info.update(base_rate_all=float(D["resp"].mean()), base_rate_target=float(D["resp"][:, 50:].mean()),
                n_train=len(sp["train"]), n_val=len(sp["val"]), n_calib=len(sp["calib"]), n_test=len(sp["test"]),
                hp=hp, seconds=round(time.time() - t0, 1))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, y=y_te, sidx=sidx_te, meta=json.dumps(dict(T=Ts, info=info)), **store)
    os.replace(tmp, path)
    print(f"done {mech} {cfg} seed{seed} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    wid, nw = int(sys.argv[1]), int(sys.argv[2])
    jobs = [(m, "center", s) for s in SEEDS for m in "AB"]
    jobs += [(m, c, s) for c in CONFIGS if c != "center" for s in SEEDS for m in "AB"]
    for j in jobs[wid::nw]:
        run_one(*j)
    print("worker finished", flush=True)
