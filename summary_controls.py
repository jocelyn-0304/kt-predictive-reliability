"""Control comparisons (Section 3.5): item-output DKT, window-trained references, position-matched window training,
early targets, calibration-in-the-large, and temperature versus Platt (slope + intercept) recalibration.
Reads results_ctrl/*.npz (run_controls.py) and results/*.npz (run_experiments.py, for the matched models).
Prints the summary used in the paper (summary_controls.txt) and writes tables_ctrl.json (Table 4)."""
import glob, json, re
import numpy as np
from scipy.optimize import minimize
from analyze import EPS, sig, logit, auc, ece_eq, boot_weights, boot_metrics, load_runs

B = 500
MOD = ["Baseline", "BKT", "2PL-IRT", "DKT", "AKT", "DKT-item"]
NM = {"AKT": "AKT-style"}


def load_ctrl():
    R = {}
    for f in sorted(glob.glob("results_ctrl/*.npz")):
        m = re.match(r".*/(A|B)_(\d+)\.npz", f)
        z = np.load(f)
        R[(m.group(1), int(m.group(2)))] = dict(
            y=z["y"].astype(int), sidx=z["sidx"], yc=z["yc"].astype(int),
            p={k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("p::")},
            c={k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("c::")},
            q={k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("q::")},
            yE={int(k[5:]): z[k].astype(int) for k in z.files if k.startswith("yE::")},
            sE={int(k[5:]): z[k] for k in z.files if k.startswith("sE::")},
            meta=json.loads(str(z["meta"])))
    return R


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_T(p, y):
    from scipy.optimize import minimize_scalar
    z = logit(p)
    r = minimize_scalar(lambda lt: nll(sig(z / np.exp(lt)), y), bounds=(-2.5, 2.5), method="bounded", options=dict(xatol=1e-4))
    return float(np.exp(r.x))


def fit_platt(p, y):
    """p' = sigmoid(a * logit(p) + b), fitted by maximum likelihood."""
    z = logit(p)

    def f(w):
        s = w[0] * z + w[1]
        e = sig(s) - y
        return float(np.mean(np.logaddexp(0, s) - y * s)), np.array([np.mean(e * z), np.mean(e)])
    r = minimize(f, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


RC = load_ctrl()
RM = load_runs()                                   # main experiment (matched models "M|...")
SEEDS = sorted({s for (_, s) in RC})
print("control runs:", sorted(RC), "\nreproducibility vs results/*.npz:")
for k, R in sorted(RC.items()):
    print(" ", k, R["meta"]["repro"])


def key(setting, model, L, eta):
    return f"{setting}|{model}|L{L}|e{float(eta)}"


def pred(R, setting, model, L, eta, var="raw"):
    """Test predictions of one replicate under a recalibration variant (raw, ts_target, ts_ref, pl_target, pl_ref).
    Target-environment maps are fitted on calibration windows of the same condition; reference maps on the clean
    L = 50 calibration windows of the same test-only model."""
    k = key(setting, model, L, eta)
    if k not in R["p"]:
        return None
    p = R["p"][k]
    if var == "raw":
        return p
    src = k if var.endswith("target") else key("S1", model, 50, 0.0)
    pc = R["c"][src]
    if var.startswith("ts"):
        return sig(logit(p) / fit_T(pc, R["yc"]))
    a, b = fit_platt(pc, R["yc"])
    return sig(a * logit(p) + b)


def get(mech, s, spec):
    """spec = (setting, model, L, eta, var) for late targets, or ("E", model, L) for early targets."""
    R = RC[(mech, s)]
    if spec[0] == "E":
        return R["q"][f"E|{spec[1]}|L{spec[2]}"], R["yE"][spec[2]], R["sE"][spec[2]]
    setting, model, L, eta, var = spec
    if setting == "M":
        p = RM[(mech, "center", s)]["p"].get(key("M", model, L, eta))
        assert np.array_equal(RM[(mech, "center", s)]["y"], R["y"])
    else:
        p = pred(R, setting, model, L, eta, var)
    return p, R["y"], R["sidx"]


def point(mech, spec, metric):
    vals = []
    for s in SEEDS:
        p, y, _ = get(mech, s, spec)
        if p is None:
            return None
        vals.append({"auc": lambda: auc(y, p), "brier": lambda: float(((p - y) ** 2).mean()), "nll": lambda: nll(p, y),
                     "ece10": lambda: ece_eq(p, y, 10), "citl": lambda: float(p.mean() - y.mean())}[metric]())
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def boot(mech, spec, metrics=("auc", "brier", "nll", "ece10", "citl"), ref=None):
    """Mean over replicates of per-replicate learner-level bootstrap distributions (as in analyze.pooled_boot);
    with ref, paired differences on the same resampled learners."""
    acc = {k: [] for k in metrics}
    for s in SEEDS:
        p, y, sidx = get(mech, s, spec)
        if p is None:
            return None
        Wb = boot_weights(int(sidx.max()) + 1, B, 1000 + s)
        bm = boot_metrics(p, y, sidx, Wb, metrics)
        if ref is not None:
            pr, yr, sr = get(mech, s, ref)
            assert np.array_equal(yr, y) and np.array_equal(sr, sidx)
            br = boot_metrics(pr, yr, sr, Wb, metrics)
            bm = {k: bm[k] - br[k] for k in metrics}
        for k in metrics:
            acc[k].append(bm[k])
    out = {}
    for k in metrics:
        mb = np.nanmean(np.stack(acc[k]), axis=0)
        out[k] = (float(np.nanmean(mb)), float(np.nanpercentile(mb, 2.5)), float(np.nanpercentile(mb, 97.5)))
    return out


fmt = lambda r, d=4: {k: tuple(round(x, d) for x in v) for k, v in r.items()} if r else None
TJ = {"A": {}, "B": {}}

print("\n=== 1. Item information: DKT-item vs DKT (test-only, full-sequence training), eta = 0")
for mech in "AB":
    for L in (3, 50):
        for mo in ("DKT", "DKT-item", "AKT", "2PL-IRT", "Baseline"):
            r = {m: point(mech, ("S1", mo, L, 0.0, "raw"), m) for m in ("auc", "brier", "nll", "ece10", "citl")}
            TJ[mech][f"item|{mo}|{L}"] = r
            print(mech, L, f"{mo:9s}", {k: (round(v[0], 3), round(v[1], 3)) for k, v in r.items()})
        print(mech, L, "paired DKT-item - DKT", fmt(boot(mech, ("S1", "DKT-item", L, 0.0, "raw"), ("auc", "brier", "nll", "ece10"), ref=("S1", "DKT", L, 0.0, "raw"))))
        print(mech, L, "paired DKT-item - Baseline", fmt(boot(mech, ("S1", "DKT-item", L, 0.0, "raw"), ("auc", "brier", "nll"), ref=("S1", "Baseline", L, 0.0, "raw"))))
        print(mech, L, "paired DKT-item - AKT", fmt(boot(mech, ("S1", "DKT-item", L, 0.0, "raw"), ("auc", "brier", "nll"), ref=("S1", "AKT", L, 0.0, "raw"))))
    L = 50
    print(mech, "L=50 eta=0.2 paired DKT-item - DKT", fmt(boot(mech, ("S1", "DKT-item", 50, 0.2, "raw"), ("auc", "brier"), ref=("S1", "DKT", 50, 0.2, "raw"))))

print("\n=== 2. Calibration-in-the-large (mean prediction - observed proportion), raw, eta = 0; late targets (events 50-59)")
for mech in "AB":
    for mo in MOD:
        row = {}
        for L in (3, 5, 10, 20, 50):
            r = boot(mech, ("S1", mo, L, 0.0, "raw"), ("citl",))
            row[L] = r["citl"]
            TJ[mech][f"citl|S1|{mo}|{L}"] = r["citl"]
        print(mech, f"{mo:9s}", {L: tuple(round(x, 3) for x in v) for L, v in row.items()})

print("\n=== 3. Early targets (target = event L, complete history) vs late windows (targets 50-59, last L events); raw")
for mech in "AB":
    for mo in MOD:
        for L in (3, 5, 10, 20):
            e = boot(mech, ("E", mo, L), ("citl", "ece10", "brier", "auc"))
            e["citl_point"] = point(mech, ("E", mo, L), "citl")
            TJ[mech][f"early|{mo}|{L}"] = e
            print(mech, f"{mo:9s}", L, "early", fmt(e, 3))
    for L in (3, 10):
        ny = [len(RC[(mech, s)]["yE"][L]) for s in SEEDS]
        print(mech, "early events per replicate", L, ny[:1], "observed proportion correct",
              round(float(np.mean([RC[(mech, s)]["yE"][L].mean() for s in SEEDS])), 3))
    print(mech, "late observed proportion correct", round(float(np.mean([RC[(mech, s)]["y"].mean() for s in SEEDS])), 3))

print("\n=== 4. Window starting state and training scheme, eta = 0 (DKT, AKT-style)")
print("settings: S1 = full-sequence, test-only; W0 = window loss on clean L=50 windows (targets 50-59);")
print("          M = window loss with L events, targets L..59 (main experiment); Mpos = window loss with L events, targets 50-59")
for mech in "AB":
    for mo in ("DKT", "AKT"):
        for L in (3, 5, 10, 20, 50):
            row = {}
            for st in ("S1", "W0", "M", "Mpos"):
                if st == "Mpos" and L == 50:
                    continue
                r = {m: point(mech, (st, mo, L, 0.0, "raw"), m) for m in ("auc", "brier", "ece10", "citl")}
                if r["brier"] is None:
                    continue
                row[st] = r
                TJ[mech][f"scheme|{st}|{mo}|{L}|0.0"] = r
            print(mech, mo, L, {st: {k: round(v[0], 3) for k, v in r.items()} for st, r in row.items()})
        for L in (3, 5):
            for a, b_ in (("Mpos", "M"), ("Mpos", "W0"), ("M", "W0"), ("Mpos", "S1"), ("W0", "S1")):
                r = boot(mech, (a, mo, L, 0.0, "raw"), ("brier", "ece10", "citl", "auc"), ref=(b_, mo, L, 0.0, "raw"))
                TJ[mech][f"pair|{a}-{b_}|{mo}|{L}|0.0"] = r
                print(mech, mo, L, f"{a}-{b_}", fmt(r))
        r = boot(mech, ("W0", mo, 50, 0.0, "raw"), ("brier", "ece10", "citl", "auc"), ref=("S1", mo, 50, 0.0, "raw"))
        TJ[mech][f"pair|W0-S1|{mo}|50|0.0"] = r
        print(mech, mo, 50, "W0-S1", fmt(r))

print("\n=== 5. Label noise under one training scheme (L = 50, eta = 0.20): matched M vs clean window reference W0")
for mech in "AB":
    for mo in ("DKT", "AKT"):
        for eta in (0.05, 0.1, 0.2):
            row = {st: {m: point(mech, (st, mo, 50, eta, "raw"), m) for m in ("auc", "brier", "ece10", "citl")} for st in ("S1", "W0", "M")}
            for st, r in row.items():
                TJ[mech][f"scheme|{st}|{mo}|50|{eta}"] = r
            pr = boot(mech, ("M", mo, 50, eta, "raw"), ("brier", "ece10", "auc", "nll"), ref=("W0", mo, 50, eta, "raw"))
            TJ[mech][f"pair|M-W0|{mo}|50|{eta}"] = pr
            print(mech, mo, eta, {st: {k: round(v[0], 3) for k, v in r.items()} for st, r in row.items()}, "M-W0", fmt(pr))

print("\n=== 6. Temperature versus Platt scaling (S1 models): ECE / NLL / citl by variant")
VARS = ("raw", "ts_target", "pl_target", "ts_ref", "pl_ref")
for mech in "AB":
    for (L, eta) in ((3, 0.0), (3, 0.2), (50, 0.2), (50, 0.0)):
        for mo in MOD:
            row = {v: {m: point(mech, ("S1", mo, L, eta, v), m) for m in ("ece10", "nll", "citl")} for v in VARS}
            for v, r in row.items():
                TJ[mech][f"recal|{mo}|{L}|{eta}|{v}"] = r
            print(mech, L, eta, f"{mo:9s}", {v: {k: round(x[0], 4) for k, x in r.items()} for v, r in row.items()})
        for mo in ("BKT", "2PL-IRT", "DKT", "AKT", "DKT-item"):
            pt = boot(mech, ("S1", mo, L, eta, "pl_target"), ("nll", "ece10"), ref=("S1", mo, L, eta, "ts_target"))
            pr = boot(mech, ("S1", mo, L, eta, "pl_ref"), ("nll", "ece10"), ref=("S1", mo, L, eta, "raw"))
            pr2 = boot(mech, ("S1", mo, L, eta, "pl_target"), ("nll", "ece10"), ref=("S1", mo, L, eta, "raw"))
            TJ[mech][f"pair|pl_target-ts_target|{mo}|{L}|{eta}"] = pt
            TJ[mech][f"pair|pl_ref-raw|{mo}|{L}|{eta}"] = pr
            TJ[mech][f"pair|pl_target-raw|{mo}|{L}|{eta}"] = pr2
            print(mech, L, eta, f"{mo:9s}", "pl_target-ts_target", fmt(pt), "| pl_target-raw", fmt(pr2), "| pl_ref-raw", fmt(pr))

print("\n=== 7. Fitted Platt parameters (a, b) and temperatures, mean over replicates")
for mech in "AB":
    for mo in ("BKT", "2PL-IRT", "DKT", "AKT", "DKT-item", "Baseline"):
        out = {}
        for (L, eta) in ((50, 0.0), (3, 0.0), (50, 0.2)):
            ab = np.array([fit_platt(RC[(mech, s)]["c"][key("S1", mo, L, eta)], RC[(mech, s)]["yc"]) for s in SEEDS])
            T = np.array([fit_T(RC[(mech, s)]["c"][key("S1", mo, L, eta)], RC[(mech, s)]["yc"]) for s in SEEDS])
            out[f"L{L}e{eta}"] = dict(a=round(float(ab[:, 0].mean()), 3), b=round(float(ab[:, 1].mean()), 3), T=round(float(T.mean()), 3))
        TJ[mech][f"platt|{mo}"] = out
        print(mech, f"{mo:9s}", out)

print("\n=== 8. Training budget: epochs run (incl. 5 patience epochs), mean over replicates")
for mech in "AB":
    ep = {}
    for s in SEEDS:
        for k, v in RC[(mech, s)]["meta"].items():
            if isinstance(v, dict) and "epochs_run" in v:
                ep.setdefault(k, []).append(v["epochs_run"])
    print(mech, {k: round(float(np.mean(v)), 1) for k, v in sorted(ep.items())})
json.dump(TJ, open("tables_ctrl.json", "w"), indent=1)
print("\nwrote tables_ctrl.json")
