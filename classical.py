"""Smoothed item-accuracy baseline, BKT with forgetting (EM), and 2PL-IRT (MAP ability).
All global parameters are fitted on training learners only; for unseen learners only the
visible history window is used to update the learner state."""
import numpy as np
from scipy.optimize import minimize


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


# ---------------------------------------------------------------- baseline
class ItemMean:
    name = "Baseline"

    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, items, skills, resp, nI):
        self.gm = resp.mean()
        n = np.bincount(items.ravel(), minlength=nI).astype(float)
        c = np.bincount(items.ravel(), weights=resp.ravel().astype(float), minlength=nI)
        self.p = (c + self.alpha * self.gm) / (n + self.alpha)       # unseen items -> global mean
        return self

    def predict(self, it, sk, rs):
        return self.p[it[:, -1]]


# ---------------------------------------------------------------- BKT
class BKT:
    """Per-skill two-state HMM with learning T and forgetting F; guess g, slip s bounded by `bound`.
    Fitted by (bounded) EM with several restarts; one set of parameters per skill."""
    name = "BKT"

    def __init__(self, bound=0.5, iters=60, restarts=3):
        self.bound, self.iters, self.restarts = bound, iters, restarts

    @staticmethod
    def _seqs(items, skills, resp, k):
        rows, ys = [], []
        for i in range(skills.shape[0]):
            m = skills[i] == k
            if m.any():
                ys.append(resp[i][m])
        if not ys:
            return None, None
        L = max(len(y) for y in ys)
        Y = np.zeros((len(ys), L)); V = np.zeros((len(ys), L), bool)
        for i, y in enumerate(ys):
            Y[i, :len(y)] = y; V[i, :len(y)] = True
        return Y, V

    def _em(self, Y, V, init):
        pi, T, F, g, s = init
        n, L = Y.shape
        ll_old = -np.inf
        for _ in range(self.iters):
            e1 = np.where(Y == 1, 1 - s, s); e0 = np.where(Y == 1, g, 1 - g)
            e1 = np.where(V, e1, 1.0); e0 = np.where(V, e0, 1.0)
            Tm = np.where(V, T, 0.0); Fm = np.where(V, F, 0.0)      # identity transition on padding
            a1 = np.zeros((n, L)); a0 = np.zeros((n, L)); c = np.zeros((n, L))
            x1 = pi * e1[:, 0]; x0 = (1 - pi) * e0[:, 0]
            c[:, 0] = x1 + x0; a1[:, 0] = x1 / c[:, 0]; a0[:, 0] = x0 / c[:, 0]
            for t in range(1, L):
                x1 = (a1[:, t - 1] * (1 - Fm[:, t]) + a0[:, t - 1] * Tm[:, t]) * e1[:, t]
                x0 = (a1[:, t - 1] * Fm[:, t] + a0[:, t - 1] * (1 - Tm[:, t])) * e0[:, t]
                c[:, t] = x1 + x0; a1[:, t] = x1 / c[:, t]; a0[:, t] = x0 / c[:, t]
            ll = np.log(c).sum()
            b1 = np.ones((n, L)); b0 = np.ones((n, L))
            for t in range(L - 2, -1, -1):
                f1 = e1[:, t + 1] * b1[:, t + 1]; f0 = e0[:, t + 1] * b0[:, t + 1]
                b1[:, t] = ((1 - Fm[:, t + 1]) * f1 + Fm[:, t + 1] * f0) / c[:, t + 1]
                b0[:, t] = (Tm[:, t + 1] * f1 + (1 - Tm[:, t + 1]) * f0) / c[:, t + 1]
            g1 = a1 * b1; g0 = a0 * b0
            gs = g1 + g0; g1 /= gs; g0 /= gs
            # transition posteriors for t>=1 (only where step t is a real observation)
            f1 = e1[:, 1:] * b1[:, 1:]; f0 = e0[:, 1:] * b0[:, 1:]
            Vt = V[:, 1:]
            xi00 = a0[:, :-1] * (1 - T) * f0 / c[:, 1:]
            xi01 = a0[:, :-1] * T * f1 / c[:, 1:]
            xi10 = a1[:, :-1] * F * f0 / c[:, 1:]
            xi11 = a1[:, :-1] * (1 - F) * f1 / c[:, 1:]
            xi00, xi01, xi10, xi11 = [np.where(Vt, x, 0) for x in (xi00, xi01, xi10, xi11)]
            pi = np.clip(g1[:, 0].mean(), 0.01, 0.99)
            T = np.clip(xi01.sum() / max(xi00.sum() + xi01.sum(), 1e-9), 0.005, 0.8)
            F = np.clip(xi10.sum() / max(xi10.sum() + xi11.sum(), 1e-9), 0.0, 0.3)
            w0 = np.where(V, g0, 0); w1 = np.where(V, g1, 0)
            g = np.clip((w0 * Y).sum() / max(w0.sum(), 1e-9), 0.01, self.bound)
            s = np.clip((w1 * (1 - Y)).sum() / max(w1.sum(), 1e-9), 0.01, self.bound)
            if abs(ll - ll_old) < 1e-4:
                break
            ll_old = ll
        return (pi, T, F, g, s), ll

    def fit(self, items, skills, resp, nI):
        K = int(skills.max()) + 1
        self.K = K
        P = np.zeros((K, 5))
        inits = [(0.3, 0.15, 0.03, 0.2, 0.1), (0.5, 0.3, 0.1, 0.1, 0.2), (0.2, 0.08, 0.0, 0.3, 0.05)]
        for k in range(K):
            Y, V = self._seqs(items, skills, resp, k)
            if Y is None:
                P[k] = (0.3, 0.15, 0.03, 0.2, 0.1); continue
            best, bp = -np.inf, None
            for init in inits[:self.restarts]:
                p, ll = self._em(Y, V, init)
                if ll > best:
                    best, bp = ll, p
            P[k] = bp
        self.P = P
        return self

    def predict(self, it, sk, rs):
        kt = sk[:, -1]
        pi, T, F, g, s = [self.P[kt, j] for j in range(5)]
        p = pi.copy()
        for j in range(sk.shape[1] - 1):
            m = sk[:, j] == kt
            y = rs[:, j]
            l1 = np.where(y == 1, 1 - s, s); l0 = np.where(y == 1, g, 1 - g)
            post = p * l1 / (p * l1 + (1 - p) * l0)
            p = np.where(m, post * (1 - F) + (1 - post) * T, p)
        return p * (1 - s) + (1 - p) * g


# ---------------------------------------------------------------- 2PL IRT
class IRT2PL:
    """P(y=1)=sigmoid(a_i(theta - b_i)); item parameters fitted jointly with training-learner abilities
    (MAP, theta~N(0,1)); at prediction time theta is re-estimated from the visible history window only."""
    name = "2PL-IRT"

    def __init__(self, sd_loga=0.5, sd_b=2.0):
        self.sd_loga, self.sd_b = sd_loga, sd_b

    def fit(self, items, skills, resp, nI):
        N, T = items.shape
        it = items.ravel(); y = resp.ravel().astype(float); st = np.repeat(np.arange(N), T)
        n_ev = len(y)

        def unpack(x):
            return x[:N], x[N:N + nI], x[N + nI:]

        def f(x):
            th, la, b = unpack(x)
            a = np.exp(la)
            z = a[it] * (th[st] - b[it])
            p = sig(z)
            nll = (np.logaddexp(0, z) - y * z).sum()
            pri = 0.5 * (th ** 2).sum() + 0.5 * ((la / self.sd_loga) ** 2).sum() + 0.5 * ((b / self.sd_b) ** 2).sum()
            dz = p - y
            gth = np.bincount(st, weights=dz * a[it], minlength=N) + th
            gb = np.bincount(it, weights=-dz * a[it], minlength=nI) + b / self.sd_b ** 2
            gla = np.bincount(it, weights=dz * a[it] * (th[st] - b[it]), minlength=nI) + la / self.sd_loga ** 2
            return (nll + pri) / n_ev, np.concatenate([gth, gla, gb]) / n_ev

        x0 = np.concatenate([np.zeros(N), np.zeros(nI), np.zeros(nI)])
        r = minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=300))
        th, la, b = unpack(r.x)
        self.a, self.b = np.exp(la), b
        self.nI = nI
        return self

    def predict(self, it, sk, rs):
        a_h = self.a[it[:, :-1]]; b_h = self.b[it[:, :-1]]; y = rs[:, :-1].astype(float)
        th = np.zeros(len(it))
        for _ in range(12):
            p = sig(a_h * (th[:, None] - b_h))
            g = (a_h * (y - p)).sum(1) - th
            h = -(a_h ** 2 * p * (1 - p)).sum(1) - 1.0
            th = th - np.clip(g / h, -2, 2)
        t = it[:, -1]
        return sig(self.a[t] * (th - self.b[t]))
