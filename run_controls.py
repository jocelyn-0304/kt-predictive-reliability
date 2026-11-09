"""Control comparisons for the centre configuration (g = 0.20, s = 0.15), both generators, seeds 1-5.

S1    test-only models refitted exactly as in run_experiments.py (Baseline, BKT, 2PL-IRT, DKT, AKT-style) plus the
      control model DKT-item (skill inputs, item outputs), evaluated at every centre condition (L x eta).
W0    DKT and AKT-style trained with the window loss on clean L = 50 windows ending at events 50-59 (the same model as
      the matched L = 50 model of run_experiments.py), evaluated at every centre condition. It shares the training
      scheme and budget of the matched models but sees clean, full-length inputs.
Mpos  window training with exactly L history events and targets restricted to events 50-59 (the test positions),
      L = 3, 5, 10, 20, evaluated at (L, eta = 0).
E     early targets: the target is event L and all L preceding events are visible, i.e. a learner who has answered
      only L items; S1 models, L = 3, 5, 10, 20.
Predictions on the calibration windows are stored as well, so temperature and Platt maps can be fitted afterwards
(summary_controls.py). The refitted S1 and W0 predictions are compared with results/*.npz as a reproducibility check.
Output: results_ctrl/{A,B}_{seed}.npz.   Usage: python3 run_controls.py <worker_id> <n_workers>"""
import sys, os, json, time
import numpy as np
from core import *
from run_experiments import N_LEARNERS, T_LEN, SEEDS, S1_CENTER, HP

HPC = {m: {**HP[m], **json.load(open(f"hp_ctrl_{m}.json"))["best"]} for m in "AB"}
LS_POS = [3, 5, 10, 20]
LS_EARLY = [3, 5, 10, 20]
TPOS = np.arange(50, 60)


def early_windows(D, idx, L):
    """Target = event L of each learner; the visible history is the complete record before it (events 0..L-1)."""
    it, sk, rs, sidx = make_windows(D["items"][idx], D["skills"][idx], D["resp"][idx], L, [L])
    y = rs[:, -1].copy()
    rin = rs.copy()
    rin[:, -1] = 0
    return it, sk, rin, y, sidx


def run_one(mech, seed):
    path = f"results_ctrl/{mech}_{seed}.npz"
    if os.path.exists(path):
        return
    t0 = time.time()
    hp = HPC[mech]
    D = simulate(mech, seed, N=N_LEARNERS, T=T_LEN, g=0.20, s=0.15)
    sp = split_students(seed, N_LEARNERS)
    tr, va = subset(D, sp["train"]), subset(D, sp["val"])
    info, store, s1, win = {}, {}, {}, {}
    for m in build_classical(tr, D["nI"], hp):
        s1[m.name] = predictor(m)
    for name in ("DKT", "AKT", "DKT-item"):
        t1 = time.time()
        net = new_neural(name, D["K"], D["nI"], hp, seed)
        r = train_neural(net, tr, va, neural_cfg(name, hp), seed, mode="full")
        info[f"{name}_full"] = dict(r, sec=round(time.time() - t1, 1))
        s1[name] = predictor(net)
    for name in ("DKT", "AKT"):
        t1 = time.time()
        net = new_neural(name, D["K"], D["nI"], hp, seed)
        r = train_neural(net, tr, va, neural_cfg(name, hp), seed, mode="window", L=50, eta=0.0)
        info[f"{name}_W0"] = dict(r, sec=round(time.time() - t1, 1))
        win[("W0", name)] = predictor(net)
        for L in LS_POS:
            t1 = time.time()
            net = new_neural(name, D["K"], D["nI"], hp, seed)
            r = train_neural(net, tr, va, neural_cfg(name, hp), seed, mode="window", L=L, eta=0.0, tpos=TPOS)
            info[f"{name}_MposL{L}"] = dict(r, sec=round(time.time() - t1, 1))
            win[("Mpos", name, L)] = predictor(net)
    cache, lab = {}, {}

    def W(split, L, eta):
        if (split, L, eta) not in cache:
            cache[(split, L, eta)] = windows(D, sp[split], L, eta, seed, split)
        return cache[(split, L, eta)]

    def put(tag, fn, L, eta):
        Wte, Wca = W("test", L, eta), W("calib", L, eta)
        if not lab:
            lab.update(y=Wte[3], sidx=Wte[4], yc=Wca[3])
        assert np.array_equal(lab["y"], Wte[3]) and np.array_equal(lab["yc"], Wca[3])
        store[f"p::{tag}|L{L}|e{eta}"] = fn(Wte).astype(np.float32)
        store[f"c::{tag}|L{L}|e{eta}"] = fn(Wca).astype(np.float32)

    for (L, eta) in S1_CENTER:
        for name, fn in s1.items():
            put(f"S1|{name}", fn, L, eta)
        for name in ("DKT", "AKT"):
            put(f"W0|{name}", win[("W0", name)], L, eta)
    for name in ("DKT", "AKT"):
        for L in LS_POS:
            put(f"Mpos|{name}", win[("Mpos", name, L)], L, 0.0)
    for L in LS_EARLY:
        We = early_windows(D, sp["test"], L)
        store[f"yE::L{L}"] = We[3]
        store[f"sE::L{L}"] = We[4]
        for name, fn in s1.items():
            store[f"q::E|{name}|L{L}"] = fn(We).astype(np.float32)
    # reproducibility check against the main experiment
    ref = np.load(f"results/{mech}_center_{seed}.npz")
    diffs = {}
    for k in ref.files:
        if k.startswith("p::S1|") or k in ("p::M|DKT|L50|e0.0", "p::M|AKT|L50|e0.0"):
            diffs[k] = float(np.abs(ref[k] - store[k.replace("p::M|", "p::W0|")]).max())
    info["repro"] = dict(n_compared=len(diffs), max_abs_diff=max(diffs.values()),
                         labels_equal=bool(np.array_equal(ref["y"], lab["y"])))
    info.update(hp=hp, seconds=round(time.time() - t0, 1))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, y=lab["y"], sidx=lab["sidx"], yc=lab["yc"], meta=json.dumps(info), **store)
    os.replace(tmp, path)
    print(f"done {mech} seed{seed} in {time.time() - t0:.0f}s repro={info['repro']}", flush=True)


if __name__ == "__main__":
    wid, nw = int(sys.argv[1]), int(sys.argv[2])
    jobs = [(m, s) for s in SEEDS for m in "AB"]
    for j in jobs[wid::nw]:
        run_one(*j)
    print("worker finished", flush=True)
