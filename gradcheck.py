import numpy as np, autodiff as ad
ad.set_dtype(np.float64)
from neural import DKT, AKT
rng = np.random.default_rng(0)
B,T,K,NI = 5,7,3,12
skills = rng.integers(0,K,(B,T)); items = rng.integers(0,NI,(B,T)); resp = rng.integers(0,2,(B,T))
w = rng.random((B,T)); y = resp.astype(float)
for M in (DKT, AKT):
    m = M(K, NI, seed=3)
    def loss():
        return ad.bce_logits(m.forward(items, skills, resp), y, w)
    L = loss(); 
    for p in m.params(): p.grad = None
    L.backward()
    worst = 0
    for pi,p in enumerate(m.params()):
        g = p.grad if p.grad is not None else np.zeros_like(p.data)
        flat = p.data.reshape(-1)
        for j in rng.choice(flat.size, min(6, flat.size), replace=False):
            old = flat[j]; eps=1e-6
            flat[j] = old+eps; lp = float(loss().data)
            flat[j] = old-eps; lm = float(loss().data)
            flat[j] = old
            num = (lp-lm)/(2*eps); ana = g.reshape(-1)[j]
            rel = abs(num-ana)/max(1e-8, abs(num)+abs(ana))
            worst = max(worst, rel if abs(num)+abs(ana)>1e-7 else 0)
    print(M.name, 'loss', float(L.data), 'max rel err', worst)
