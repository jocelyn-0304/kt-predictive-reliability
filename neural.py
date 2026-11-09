"""DKT (GRU) and an AKT-style attention model implemented on the numpy autodiff engine."""
import copy
import numpy as np
import autodiff as ad
from autodiff import Tensor, param


class DKT:
    """Deep knowledge tracing (Piech et al., 2015) with a GRU: input = (skill, response) token,
    output = per-skill probability of a correct response; the entry of the target skill is used."""
    name = "DKT"
    out_by_item = False

    def __init__(self, K, n_items, E=24, H=32, seed=0):
        rng = np.random.default_rng(seed)
        self.K, self.H = K, H
        self.Emb = param((2 * K, E), rng, scale=0.3, kind="normal")
        self.Wx = param((E, 3 * H), rng)
        self.bx = param((3 * H,), rng, kind="zeros")
        self.Wh = param((H, 3 * H), rng)
        self.bh = param((3 * H,), rng, kind="zeros")
        self.Wo = param((H, K), rng)
        self.bo = param((K,), rng, kind="zeros")

    def params(self):
        return [self.Emb, self.Wx, self.bx, self.Wh, self.bh, self.Wo, self.bo]

    def forward(self, items, skills, resp_in):
        B, T = skills.shape
        tok = skills * 2 + resp_in
        emb = ad.take(self.Emb, tok)
        xp = ad.matmul(emb, self.Wx) + self.bx
        h = Tensor(np.zeros((B, self.H)))
        hs = []
        for t in range(T):
            hs.append(h)                      # state summarising events < t
            h = ad.gru_cell(xp, t, h, self.Wh, self.bh)
        Hs = ad.stack(hs, axis=1)
        logits = ad.matmul(Hs, self.Wo) + self.bo
        return ad.gather_last(logits, items if self.out_by_item else skills)


class DKTItem(DKT):
    """Control variant: DKT with skill (KC) inputs and one output per item; the entry of the target item is used.
    Skill inputs with item outputs was the combination that worked best on most datasets in Gervet et al. (2020)."""
    name = "DKT-item"
    out_by_item = True

    def __init__(self, K, n_items, E=24, H=32, seed=0):
        super().__init__(K, n_items, E=E, H=H, seed=seed)
        rng = np.random.default_rng(seed + 2000)
        self.Wo = param((H, n_items), rng)
        self.bo = param((n_items,), rng, kind="zeros")


class AKT:
    """AKT-style knowledge tracing: Rasch-type question embeddings, one attention head with
    monotonic exponential distance decay on the attention logits, MLP prediction head.
    Simplified relative to Ghosh et al. (2020): single block, no question-encoder stack,
    temporal (not context-aware) distance."""
    name = "AKT"

    def __init__(self, K, n_items, E=24, hid=32, seed=0):
        rng = np.random.default_rng(seed + 1000)
        self.K, self.E = K, E
        self.C = param((K, E), rng, scale=0.3, kind="normal")
        self.Dm = param((K, E), rng, scale=0.1, kind="normal")
        self.mu = param((n_items, 1), rng, scale=0.1, kind="normal")
        self.Ey = param((2 * K, E), rng, scale=0.3, kind="normal")
        self.Fy = param((2 * K, E), rng, scale=0.1, kind="normal")
        self.Wq = param((E, E), rng)
        self.Wk = param((E, E), rng)
        self.Wv = param((E, E), rng)
        self.th = Tensor(np.zeros((1,)), requires_grad=True)
        self.W1 = param((2 * E, hid), rng)
        self.b1 = param((hid,), rng, kind="zeros")
        self.W2 = param((hid, 1), rng)
        self.b2 = param((1,), rng, kind="zeros")
        self._cache = {}

    def params(self):
        return [self.C, self.Dm, self.mu, self.Ey, self.Fy, self.Wq, self.Wk, self.Wv,
                self.th, self.W1, self.b1, self.W2, self.b2]

    def _tm(self, T):
        if T not in self._cache:
            tt = np.arange(T)
            dist = np.where(tt[None, :] < tt[:, None], tt[:, None] - tt[None, :], 0).astype(ad.DT)
            mask = ~(tt[None, :] < tt[:, None])           # True = not visible (j >= t)
            self._cache[T] = (Tensor(dist), mask)
        return self._cache[T]

    def forward(self, items, skills, resp_in):
        B, T = skills.shape
        dist, mask = self._tm(T)
        mu_q = ad.take(self.mu, items)                     # (B,T,1)
        x = ad.take(self.C, skills) + mu_q * ad.take(self.Dm, skills)
        tok = skills * 2 + resp_in
        y = ad.take(self.Ey, tok) + mu_q * ad.take(self.Fy, tok)
        q = ad.matmul(x, self.Wq)
        k = ad.matmul(x, self.Wk)
        v = ad.matmul(y, self.Wv)
        sc = ad.matmul(q, ad.swapaxes(k, -1, -2)) * (1.0 / np.sqrt(self.E))
        theta = ad.softplus(self.th)                       # (1,)
        decay = ad.exp(ad.mul(ad.mul(theta, dist), -1.0))  # (T,T)
        sc = sc * decay
        alpha = ad.softmax_masked(sc, mask)
        ctx = ad.matmul(alpha, v)
        feat = ad.concat([ctx, x], axis=-1)
        hid = ad.relu(ad.matmul(feat, self.W1) + self.b1)
        out = ad.matmul(hid, self.W2) + self.b2
        return ad.reshape(out, (B, T))


# --------------------------------------------------------------------------------
def predict_logits(model, items, skills, resp_in, last_only=True, bs=512):
    outs = []
    for i in range(0, len(skills), bs):
        z = model.forward(items[i:i + bs], skills[i:i + bs], resp_in[i:i + bs])
        outs.append(z.data[:, -1] if last_only else z.data)
    return np.concatenate(outs, axis=0).astype(np.float64)


def make_windows(items, skills, resp, L, targets):
    """Windows of L history events + the target event, for every (student, target) pair.
    Returns arrays (N*len(targets), L+1) plus the clean label of the target and the student index."""
    n = items.shape[0]
    tg = np.asarray(targets)
    sidx = np.repeat(np.arange(n), len(tg))
    tpos = np.tile(tg, n)
    cols = tpos[:, None] + np.arange(-L, 1)[None, :]
    it = items[sidx[:, None], cols]
    sk = skills[sidx[:, None], cols]
    rs = resp[sidx[:, None], cols]
    return it, sk, rs, sidx


def corrupt_history(resp_hist, eta, u):
    """Flip history labels where common random numbers u < eta (nested across eta)."""
    flip = u < eta
    return np.where(flip, 1 - resp_hist, resp_hist)


def train_neural(model, tr, va, cfg, seed, mode="full", L=None, eta=0.0, verbose=False, tpos=None):
    """Train on clean next-response labels.
    tr/va = (items, skills, resp) full sequences (N,T).
    mode='full'  : teacher-forced loss at every position (standard KT training).
    mode='window': loss only at the last position of random windows with exactly L history events
                   (training distribution matched to test-time truncation).
    eta: label-flip rate applied to *input* histories during training (matched corruption);
         the targets used in the loss are always the clean responses.
    tpos: optional array of allowed target positions for mode='window' (training and validation windows);
          default None = every position from L to the end of the sequence."""
    rng = np.random.default_rng(seed + 7)
    opt = ad.Adam(model.params(), lr=cfg["lr"], wd=cfg.get("wd", 1e-5), clip=5.0)
    bs = cfg.get("bs", 64)
    Ttr = tr[0].shape[1]

    def make_val():
        if mode == "full":
            it, sk, rs = va
            u = np.random.default_rng(seed + 99).random(rs.shape)
            rin = corrupt_history(rs, eta, u) if eta > 0 else rs
            return it, sk, rin, rs, np.ones_like(rs, dtype=ad.DT)
        tg = np.arange(L, Ttr, max(1, (Ttr - L) // 6)) if tpos is None else np.asarray(tpos)
        it, sk, rs, _ = make_windows(*va, L, tg)
        u = np.random.default_rng(seed + 99).random(rs.shape)
        rin = rs.copy()
        rin[:, :-1] = corrupt_history(rs[:, :-1], eta, u[:, :-1]) if eta > 0 else rs[:, :-1]
        rin[:, -1] = 0
        w = np.zeros_like(rs, dtype=ad.DT)
        w[:, -1] = 1
        return it, sk, rin, rs, w

    V = make_val()

    def val_loss():
        it, sk, rin, rs, w = V
        tot, den = 0.0, 0.0
        for i in range(0, len(sk), 512):
            z = model.forward(it[i:i + 512], sk[i:i + 512], rin[i:i + 512]).data
            y = rs[i:i + 512]
            ww = w[i:i + 512]
            per = np.logaddexp(0, z) - y * z
            tot += float((per * ww).sum())
            den += float(ww.sum())
        return tot / den

    best, best_state, bad = np.inf, None, 0
    hist = []
    for ep in range(cfg["epochs"]):
        if mode == "full":
            order = rng.permutation(len(tr[1]))
            batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        else:
            nS = len(tr[1])
            tg_all = np.arange(L, Ttr) if tpos is None else np.asarray(tpos)
            m = min(cfg.get("win_per_epoch", 6000), nS * len(tg_all))
            s_idx = rng.integers(0, nS, m)
            t_idx = rng.choice(tg_all, m)
            batches = [(s_idx[i:i + bs], t_idx[i:i + bs]) for i in range(0, m, bs)]
        for b in batches:
            if mode == "full":
                it, sk, rs = tr[0][b], tr[1][b], tr[2][b]
                rin = corrupt_history(rs, eta, rng.random(rs.shape)) if eta > 0 else rs
                w = np.ones(rs.shape, dtype=ad.DT)
            else:
                s_b, t_b = b
                cols = t_b[:, None] + np.arange(-L, 1)[None, :]
                it, sk, rs = tr[0][s_b[:, None], cols], tr[1][s_b[:, None], cols], tr[2][s_b[:, None], cols]
                rin = rs.copy()
                if eta > 0:
                    rin[:, :-1] = corrupt_history(rs[:, :-1], eta, rng.random(rs[:, :-1].shape))
                rin[:, -1] = 0
                w = np.zeros(rs.shape, dtype=ad.DT)
                w[:, -1] = 1
            z = model.forward(it, sk, rin)
            loss = ad.bce_logits(z, rs.astype(ad.DT), w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = val_loss()
        hist.append(vl)
        if verbose:
            print(f"  epoch {ep + 1:2d} val_nll {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = [p.data.copy() for p in model.params()]
        else:
            bad += 1
            if bad >= cfg.get("patience", 4):
                break
    if best_state is not None:
        for p, s in zip(model.params(), best_state):
            p.data = s
    return {"val_nll": best, "epochs_run": len(hist)}
