"""Synthetic-KT-A (two-state learning/forgetting/guess/slip process) and
Synthetic-KT-B (dynamic multidimensional IRT with four-parameter asymptotes)."""
import numpy as np

K_SKILLS, ITEMS_PER_SKILL = 10, 20


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def _schedule(rng, N, T, K, ips, p_stay=0.6):
    sk = np.zeros((N, T), dtype=np.int64)
    sk[:, 0] = rng.integers(0, K, N)
    for t in range(1, T):
        stay = rng.random(N) < p_stay
        sk[:, t] = np.where(stay, sk[:, t - 1], rng.integers(0, K, N))
    items = sk * ips + rng.integers(0, ips, (N, T))
    return sk, items


def simulate_A(seed, N=3000, T=60, g=0.20, s=0.15, K=K_SKILLS, ips=ITEMS_PER_SKILL,
               mu0=0.35, tau=0.18, lam=0.006, item_sd=0.3):
    rng = np.random.default_rng(seed)
    nI = K * ips
    sk, items = _schedule(rng, N, T, K, ips)
    gi = sig(logit(g) + item_sd * rng.normal(size=nI))     # item-specific guessing
    si = sig(logit(s) + item_sd * rng.normal(size=nI))     # item-specific slipping
    u = rng.normal(0, 0.7, N)                              # student propensity
    nu = rng.normal(0, 0.5, K)                             # skill difficulty (shared)
    om = rng.normal(0, 0.3, K)
    P0 = sig(logit(mu0) + u[:, None] + nu[None, :] + 0.3 * rng.normal(size=(N, K)))
    TAU = sig(logit(tau) + 0.5 * u[:, None] + om[None, :])
    z = rng.random((N, K)) < P0
    last = -np.ones((N, K), dtype=np.int64)
    resp = np.zeros((N, T), dtype=np.int64)
    zlat = np.zeros((N, T), dtype=np.int64)
    rows = np.arange(N)
    for t in range(T):
        k = sk[:, t]
        has = last[rows, k] >= 0
        gap = np.where(has, t - last[rows, k], 0)
        phi = 1 - np.exp(-lam * gap)
        zc = z[rows, k]
        zc = zc & ~((rng.random(N) < phi) & zc & has)       # forgetting since last practice
        it = items[:, t]
        pc = np.where(zc, 1 - si[it], gi[it])
        resp[:, t] = rng.random(N) < pc
        zlat[:, t] = zc
        zc = zc | ((~zc) & (rng.random(N) < TAU[rows, k]))   # learning after the attempt
        z[rows, k] = zc
        last[rows, k] = t
    return dict(items=items, skills=sk, resp=resp, zlat=zlat, K=K, nI=nI, mech="A",
                params=dict(g=g, s=s, mu0=mu0, tau=tau, lam=lam, item_sd=item_sd))


def simulate_B(seed, N=3000, T=60, g=0.20, s=0.15, K=K_SKILLS, ips=ITEMS_PER_SKILL,
               item_sd=0.3, ar=0.9, sig_eps=0.15, growth_mean=1.0, growth_sd=0.4, shift=0.0):
    rng = np.random.default_rng(seed)
    nI = K * ips
    sk, items = _schedule(rng, N, T, K, ips)
    gi = sig(logit(g) + item_sd * rng.normal(size=nI))
    si = sig(logit(s) + item_sd * rng.normal(size=nI))
    a = np.exp(0.3 * rng.normal(size=nI))
    b = rng.normal(0, 1.0, nI)
    alpha = rng.normal(0, 1.0, N)
    nu = rng.normal(0, 0.5, (N, K))
    beta = rng.normal(growth_mean, growth_sd, N)
    eps = np.zeros((N, T))
    eps[:, 0] = rng.normal(0, sig_eps / np.sqrt(1 - ar ** 2), N)
    for t in range(1, T):
        eps[:, t] = ar * eps[:, t - 1] + sig_eps * rng.normal(size=N)
    rows = np.arange(N)[:, None]
    theta = alpha[:, None] + nu[rows, sk] + beta[:, None] * (np.arange(T)[None, :] / (T - 1)) + eps + shift
    pc = gi[items] + (1 - gi[items] - si[items]) * sig(a[items] * (theta - b[items]))
    resp = (rng.random((N, T)) < pc).astype(np.int64)
    return dict(items=items, skills=sk, resp=resp, K=K, nI=nI, mech="B",
                params=dict(g=g, s=s, ar=ar, sig_eps=sig_eps, growth_mean=growth_mean))


def simulate(mech, seed, **kw):
    return simulate_A(seed, **kw) if mech == "A" else simulate_B(seed, **kw)


def split_students(seed, N, frac=(0.60, 0.15, 0.10, 0.15)):
    rng = np.random.default_rng(seed + 424242)
    perm = rng.permutation(N)
    cuts = np.cumsum([int(round(f * N)) for f in frac])[:-1]
    tr, va, ca, te = np.split(perm, cuts)
    return dict(train=tr, val=va, calib=ca, test=te)


def subset(D, idx):
    return D["items"][idx], D["skills"][idx], D["resp"][idx]
