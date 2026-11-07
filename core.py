import json, time
import numpy as np
from scipy.optimize import minimize_scalar
import autodiff as ad
from simulate import simulate, split_students, subset
from classical import ItemMean, BKT, IRT2PL
from neural import DKT, AKT, make_windows, predict_logits, corrupt_history, train_neural

TARGETS = np.arange(50, 60)       # fixed target events (identical for every history length)
EPS = 1e-6
SPLIT_ID = {"test": 1, "calib": 2, "val": 3}


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_T(p, y):
    z = logit(p)
    f = lambda lt: nll(sig(z / np.exp(lt)), y)
    r = minimize_scalar(f, bounds=(-2.5, 2.5), method="bounded", options=dict(xatol=1e-4))
    return float(np.exp(r.x))


def windows(D, idx, L, eta, seed, split):
    """Visible-history windows (last L events before each fixed target) with optional label flips on the
    history only; the target label y is always the clean response."""
    it, sk, rs, sidx = make_windows(D["items"][idx], D["skills"][idx], D["resp"][idx], L, TARGETS)
    y = rs[:, -1].copy()
    rin = rs.copy()
    rin[:, -1] = 0
    if eta > 0:
        u = np.random.default_rng(seed * 100003 + L * 101 + SPLIT_ID[split]).random(rs[:, :-1].shape)
        rin[:, :-1] = corrupt_history(rs[:, :-1], eta, u)
    return it, sk, rin, y, sidx


def build_classical(tr, nI, hp):
    return [ItemMean(hp["alpha"]).fit(*tr, nI),
            BKT(bound=hp["bkt_bound"]).fit(*tr, nI),
            IRT2PL(sd_loga=hp["irt_sd"]).fit(*tr, nI)]


def new_neural(name, K, nI, hp, seed):
    if name == "DKT":
        return DKT(K, nI, E=24, H=hp["dkt_H"], seed=seed)
    return AKT(K, nI, E=hp["akt_E"], hid=32, seed=seed)


def neural_cfg(name, hp):
    return dict(lr=hp["dkt_lr"] if name == "DKT" else hp["akt_lr"], epochs=hp.get("epochs", 40),
                bs=64, patience=5, wd=1e-5, win_per_epoch=hp.get("win_per_epoch", 6000))


def predictor(model):
    if hasattr(model, "forward"):
        return lambda W: sig(predict_logits(model, W[0], W[1], W[2]))
    return lambda W: model.predict(W[0], W[1], W[2])
