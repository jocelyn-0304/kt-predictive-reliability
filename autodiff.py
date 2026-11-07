"""Minimal reverse-mode autodiff on numpy (no torch available in this environment).

Supports exactly the operations needed for the GRU-based DKT and the attention-based
AKT-style model used in this study. Verified against finite differences in gradcheck.py.
"""
import numpy as np

DT = np.float32


def set_dtype(dt):
    global DT
    DT = dt


class SliceGrad:
    """Sparse gradient contribution: parent.grad[idx] += val."""
    __slots__ = ("idx", "val")

    def __init__(self, idx, val):
        self.idx = idx
        self.val = val


class Tensor:
    __slots__ = ("data", "grad", "parents", "backward_fn", "requires_grad")

    def __init__(self, data, requires_grad=False, parents=(), backward_fn=None):
        self.data = np.asarray(data, dtype=DT)
        self.grad = None
        self.requires_grad = requires_grad
        self.parents = parents
        self.backward_fn = backward_fn

    # ---- operator sugar -------------------------------------------------
    def __add__(self, o): return add(self, o)
    __radd__ = __add__
    def __sub__(self, o): return sub(self, o)
    def __rsub__(self, o): return sub(as_t(o), self)
    def __mul__(self, o): return mul(self, o)
    __rmul__ = __mul__
    def __neg__(self): return mul(self, -1.0)
    def __matmul__(self, o): return matmul(self, o)

    @property
    def shape(self): return self.data.shape

    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node.parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node.backward_fn is None or node.grad is None:
                continue
            grads = node.backward_fn(node.grad)
            for p, g in zip(node.parents, grads):
                if g is None or not p.requires_grad:
                    continue
                if isinstance(g, SliceGrad):
                    if p.grad is None:
                        p.grad = np.zeros_like(p.data)
                    p.grad[g.idx] += g.val
                else:
                    p.grad = g if p.grad is None else p.grad + g
            if node.parents:
                node.grad = None  # free intermediates; leaves keep .grad


def as_t(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _make(data, parents, fn):
    rg = any(p.requires_grad for p in parents)
    if not rg:
        return Tensor(data)
    return Tensor(data, True, tuple(parents), fn)


def _unb(g, shape):
    """Sum gradient g down to `shape` (undo numpy broadcasting)."""
    if g.shape == tuple(shape):
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


# ---- elementwise ----------------------------------------------------------
def add(a, b):
    a, b = as_t(a), as_t(b)
    return _make(a.data + b.data, (a, b), lambda g: (_unb(g, a.shape), _unb(g, b.shape)))


def sub(a, b):
    a, b = as_t(a), as_t(b)
    return _make(a.data - b.data, (a, b), lambda g: (_unb(g, a.shape), _unb(-g, b.shape)))


def mul(a, b):
    a, b = as_t(a), as_t(b)
    return _make(a.data * b.data, (a, b),
                 lambda g: (_unb(g * b.data, a.shape), _unb(g * a.data, b.shape)))


def sigmoid(a):
    s = 1.0 / (1.0 + np.exp(-np.clip(a.data, -30, 30)))
    s = s.astype(DT)
    return _make(s, (a,), lambda g: (g * s * (1 - s),))


def tanh(a):
    t = np.tanh(a.data)
    return _make(t, (a,), lambda g: (g * (1 - t * t),))


def relu(a):
    m = a.data > 0
    return _make(a.data * m, (a,), lambda g: (g * m,))


def exp(a):
    e = np.exp(np.clip(a.data, -30, 30)).astype(DT)
    return _make(e, (a,), lambda g: (g * e,))


def softplus(a):
    x = a.data
    out = np.logaddexp(0, x).astype(DT)
    sg = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    return _make(out, (a,), lambda g: (g * sg,))


# ---- reductions / shape -----------------------------------------------------
def tsum(a, axis=None, keepdims=False):
    out = a.data.sum(axis=axis, keepdims=keepdims)

    def fn(g):
        if axis is None:
            return (np.broadcast_to(g, a.shape).copy(),)
        gg = g if keepdims else np.expand_dims(g, axis)
        return (np.broadcast_to(gg, a.shape).copy(),)
    return _make(out, (a,), fn)


def reshape(a, shape):
    return _make(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),))


def swapaxes(a, i, j):
    return _make(np.swapaxes(a.data, i, j), (a,), lambda g: (np.swapaxes(g, i, j),))


def concat(ts, axis=-1):
    out = np.concatenate([t.data for t in ts], axis=axis)
    sizes = [t.data.shape[axis] for t in ts]

    def fn(g):
        outs, start = [], 0
        for s in sizes:
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(start, start + s)
            outs.append(g[tuple(sl)])
            start += s
        return tuple(outs)
    return _make(out, tuple(ts), fn)


def stack(ts, axis=1):
    out = np.stack([t.data for t in ts], axis=axis)
    return _make(out, tuple(ts), lambda g: tuple(np.take(g, i, axis=axis) for i in range(len(ts))))


def take(E, idx):
    """Embedding lookup: E[idx]."""
    out = E.data[idx]

    def fn(g):
        gE = np.zeros_like(E.data)
        np.add.at(gE, idx, g)
        return (gE,)
    return _make(out, (E,), fn)


def gather_last(x, idx):
    """out[b,t] = x[b,t,idx[b,t]]."""
    out = np.take_along_axis(x.data, idx[..., None], axis=-1)[..., 0]

    def fn(g):
        gx = np.zeros_like(x.data)
        np.put_along_axis(gx, idx[..., None], g[..., None], axis=-1)
        return (gx,)
    return _make(out, (x,), fn)


def matmul(a, b):
    a, b = as_t(a), as_t(b)
    out = np.matmul(a.data, b.data)

    def fn(g):
        ga = np.matmul(g, np.swapaxes(b.data, -1, -2)) if a.requires_grad else None
        gb = np.matmul(np.swapaxes(a.data, -1, -2), g) if b.requires_grad else None
        return (_unb(ga, a.shape) if ga is not None else None,
                _unb(gb, b.shape) if gb is not None else None)
    return _make(out, (a, b), fn)


def softmax_masked(x, mask):
    """Softmax over last axis; mask True = excluded. Fully masked rows -> zeros."""
    z = np.where(mask, -1e9, x.data)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z) * (~mask)
    den = e.sum(axis=-1, keepdims=True)
    s = (e / np.maximum(den, 1e-30)).astype(DT)

    def fn(g):
        return (s * (g - (g * s).sum(axis=-1, keepdims=True)),)
    return _make(s, (x,), fn)


def bce_logits(z, y, w):
    """Weighted mean binary cross-entropy from logits. y, w are constant arrays."""
    zd = z.data
    per = np.logaddexp(0, zd) - y * zd
    sw = max(float(w.sum()), 1e-12)
    out = np.array((per * w).sum() / sw, dtype=DT)

    def fn(g):
        sg = 1.0 / (1.0 + np.exp(-np.clip(zd, -30, 30)))
        return (g * (sg - y) * w / sw,)
    return _make(out, (z,), fn)


def gru_cell(xp, t, h, Wh, bh):
    """Fused GRU step. xp: (B,T,3H) input projections (includes input bias), uses xp[:,t,:].
    Gate layout [r | z | n]; h' = (1-z)*n + z*h ; n = tanh(xn + r*(Whn h + bhn))."""
    H = h.data.shape[1]
    x = xp.data[:, t, :]
    hp = h.data @ Wh.data + bh.data
    r = 1.0 / (1.0 + np.exp(-np.clip(x[:, :H] + hp[:, :H], -30, 30)))
    z = 1.0 / (1.0 + np.exp(-np.clip(x[:, H:2 * H] + hp[:, H:2 * H], -30, 30)))
    hn = hp[:, 2 * H:]
    n = np.tanh(x[:, 2 * H:] + r * hn)
    hnew = ((1 - z) * n + z * h.data).astype(DT)

    def fn(g):
        dn = g * (1 - z)
        dz = g * (h.data - n)
        dh_direct = g * z
        dan = dn * (1 - n * n)
        dr = dan * hn
        dhn = dan * r
        daz = dz * z * (1 - z)
        dar = dr * r * (1 - r)
        dhp = np.concatenate([dar, daz, dhn], axis=1)
        dx = np.concatenate([dar, daz, dan], axis=1)
        dh = dh_direct + dhp @ Wh.data.T
        dWh = h.data.T @ dhp
        dbh = dhp.sum(axis=0)
        return (SliceGrad((slice(None), t, slice(None)), dx), dh, dWh, dbh)
    return _make(hnew, (xp, h, Wh, bh), fn)


# ---- parameters / optimiser ------------------------------------------------------
def param(shape, rng, scale=None, kind="glorot"):
    shape = tuple(shape)
    if kind == "zeros":
        d = np.zeros(shape)
    elif kind == "normal":
        d = rng.normal(0, scale if scale else 0.1, size=shape)
    else:
        fan = (shape[0] + shape[-1]) if len(shape) > 1 else shape[0] * 2
        lim = np.sqrt(6.0 / fan)
        d = rng.uniform(-lim, lim, size=shape)
    return Tensor(d, requires_grad=True)


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, clip=5.0):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.wd, self.clip = lr, b1, b2, eps, wd, clip
        self.m = [np.zeros_like(q.data) for q in params]
        self.v = [np.zeros_like(q.data) for q in params]
        self.t = 0

    def zero_grad(self):
        for q in self.p:
            q.grad = None

    def step(self):
        self.t += 1
        gs = [q.grad if q.grad is not None else np.zeros_like(q.data) for q in self.p]
        tot = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in gs))
        sc = min(1.0, self.clip / (tot + 1e-12))
        for i, (q, g) in enumerate(zip(self.p, gs)):
            g = g * sc + self.wd * q.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            q.data = (q.data - self.lr * mh / (np.sqrt(vh) + self.eps)).astype(DT)
