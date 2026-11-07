"""Aggregate saved predictions: point metrics, seed-level mean/SD, student-level bootstrap."""
import glob, json, os, re
import numpy as np
import pandas as pd

EPS = 1e-6
MODELS = ["Baseline", "BKT", "2PL-IRT", "DKT", "AKT"]


def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def recal(p, T):
    return sig(logit(p) / T)


def auc(y, p):
    from scipy.stats import rankdata
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece_eq(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    cnt = np.bincount(b, minlength=nb); sp = np.bincount(b, weights=p, minlength=nb); sy = np.bincount(b, weights=y, minlength=nb)
    return np.abs(sy - sp).sum() / len(p)


def ece_mass(p, y, nb=10):
    o = np.argsort(p, kind="stable"); parts = np.array_split(o, nb)
    return sum(abs(y[i].sum() - p[i].sum()) for i in parts) / len(p)


def hc_stats(p, y, thr=0.90):
    c = np.maximum(p, 1 - p); m = c >= thr
    n = int(m.sum())
    if n == 0:
        return np.nan, 0.0, 0
    err = ((p[m] >= 0.5).astype(int) != y[m]).mean()
    return float(err), n / len(p), n


def point_metrics(p, y):
    pc = np.clip(p, EPS, 1 - EPS)
    e, cov, n = hc_stats(p, y)
    return dict(auc=auc(y, p), brier=float(((p - y) ** 2).mean()),
                nll=float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
                ece10=ece_eq(p, y, 10), ece5=ece_eq(p, y, 5), ece15=ece_eq(p, y, 15), ece20=ece_eq(p, y, 20),
                ece_mass10=ece_mass(p, y, 10), hc_err=e, hc_cov=cov, hc_n=n, mean_p=float(p.mean()), base=float(y.mean()))


def load_runs(pattern="results/*.npz"):
    runs = {}
    for f in sorted(glob.glob(pattern)):
        m = re.match(r".*/(A|B)_(\w+)_(\d+)\.npz", f)
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        runs[(m.group(1), m.group(2), int(m.group(3)))] = dict(
            y=z["y"].astype(int), sidx=z["sidx"],
            p={k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("p::")}, meta=meta)
    return runs


def variants(p, Tinfo):
    out = {"raw": p, "ts_target": recal(p, Tinfo["T_target"])}
    if Tinfo.get("T_ref"):
        out["ts_ref"] = recal(p, Tinfo["T_ref"])
    return out


def long_table(runs):
    rows = []
    for (mech, cfg, seed), R in runs.items():
        for key, p in R["p"].items():
            setting, model, Ls, es = key.split("|")
            L, eta = int(Ls[1:]), float(es[1:])
            for var, pv in variants(p, R["meta"]["T"][key]).items():
                rows.append(dict(mech=mech, cfg=cfg, seed=seed, setting=setting, model=model, L=L, eta=eta, variant=var,
                                 T=R["meta"]["T"][key]["T_target" if var != "ts_ref" else "T_ref"] if var != "raw" else 1.0,
                                 **point_metrics(pv, R["y"])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ weighted bootstrap (student-level)
def boot_weights(n_students, B, seed):
    return np.random.default_rng(seed).multinomial(n_students, np.full(n_students, 1.0 / n_students), size=B).astype(np.float64)


def boot_metrics(p, y, sidx, Wb, metrics=("auc", "brier", "nll", "ece10", "hc_err", "hc_cov")):
    """Metrics for each bootstrap replicate; weights = multiplicity of the learner."""
    w = Wb[:, sidx]                    # (B, n_events)
    sw = w.sum(1)
    out = {}
    pc = np.clip(p, EPS, 1 - EPS)
    if "brier" in metrics: out["brier"] = (w * (p - y) ** 2).sum(1) / sw
    if "nll" in metrics: out["nll"] = (w * -(y * np.log(pc) + (1 - y) * np.log(1 - pc))).sum(1) / sw
    if "ece10" in metrics:
        b = np.minimum((p * 10).astype(int), 9); oh = np.eye(10)[b]
        out["ece10"] = np.abs((w * y) @ oh - (w * p) @ oh).sum(1) / sw
    if "hc_err" in metrics or "hc_cov" in metrics:
        c = np.maximum(p, 1 - p); m = (c >= 0.90).astype(float); er = ((p >= 0.5).astype(int) != y).astype(float)
        wm = (w * m).sum(1)
        out["hc_cov"] = wm / sw
        with np.errstate(invalid="ignore", divide="ignore"):
            out["hc_err"] = np.where(wm > 0, (w * m * er).sum(1) / wm, np.nan)
    if "auc" in metrics:
        o = np.argsort(p, kind="stable"); ps = p[o]
        st = np.r_[0, np.flatnonzero(np.diff(ps) != 0) + 1]
        ws = w[:, o]; ys = y[o].astype(float)
        pos = np.add.reduceat(ws * ys, st, axis=1); neg = np.add.reduceat(ws * (1 - ys), st, axis=1)
        cneg = np.cumsum(neg, 1) - neg
        out["auc"] = (pos * (cneg + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))
    return out


def pooled_boot(runs, mech, cfg, setting, model, L, eta, variant, B=1000, metrics=("auc", "brier", "nll", "ece10"), ref=None):
    """Average over seeds of per-seed bootstrap replicates. `ref` = dict(setting, model, L, eta, variant) for a paired
    difference (metric(model) - metric(ref)) computed on the same resampled learners."""
    acc = {k: [] for k in metrics}
    for (m, c, s), R in sorted(runs.items()):
        if m != mech or c != cfg:
            continue
        key = f"{setting}|{model}|L{L}|e{eta}"
        if key not in R["p"]:
            continue
        n_st = int(R["sidx"].max()) + 1
        Wb = boot_weights(n_st, B, 1000 + s)
        p = variants(R["p"][key], R["meta"]["T"][key])[variant]
        bm = boot_metrics(p, R["y"], R["sidx"], Wb, metrics)
        if ref is not None:
            rk = f"{ref['setting']}|{ref['model']}|L{ref['L']}|e{ref['eta']}"
            pr = variants(R["p"][rk], R["meta"]["T"][rk])[ref["variant"]]
            br = boot_metrics(pr, R["y"], R["sidx"], Wb, metrics)
            bm = {k: bm[k] - br[k] for k in metrics}
        for k in metrics:
            acc[k].append(bm[k])
    res = {}
    for k in metrics:
        if not acc[k]:
            continue
        mean_b = np.nanmean(np.stack(acc[k]), axis=0)
        res[k] = (float(np.nanmean(mean_b)), float(np.nanpercentile(mean_b, 2.5)), float(np.nanpercentile(mean_b, 97.5)))
    return res
