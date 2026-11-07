import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import *

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "axes.linewidth": 0.7, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.7, "ytick.major.width": 0.7})
COL = {"Baseline": "#2a78d6", "BKT": "#eb6834", "2PL-IRT": "#1baf7a", "DKT": "#6250d6", "AKT": "#e87ba4"}
MRK = {"Baseline": "o", "BKT": "s", "2PL-IRT": "^", "DKT": "D", "AKT": "v"}
LS = {"Baseline": ":", "BKT": "-", "2PL-IRT": "--", "DKT": "-.", "AKT": "-"}
MECH = {"A": "Synthetic-KT-A", "B": "Synthetic-KT-B"}
runs = load_runs()
Ls = [3, 5, 10, 20, 50]

# ------------------------------------------------ Figure 2: reliability diagrams (pooled over seeds)
def pooled_bins(mech, model, L, eta, nb=10):
    cnt = np.zeros(nb); sp = np.zeros(nb); sy = np.zeros(nb)
    for (m, c, s), R in runs.items():
        if m != mech or c != "center":
            continue
        key = f"S1|{model}|L{L}|e{eta}"
        p, y = R["p"][key], R["y"]
        b = np.minimum((p * nb).astype(int), nb - 1)
        cnt += np.bincount(b, minlength=nb); sp += np.bincount(b, weights=p, minlength=nb); sy += np.bincount(b, weights=y, minlength=nb)
    return cnt, sp, sy

conds = [(50, 0.0, "L = 50, no label flips"), (3, 0.0, "L = 3, no label flips"), (50, 0.20, "L = 50, 20% history flips")]
fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.5), sharex=True, sharey=True)
MINN = 50
for r, mech in enumerate("AB"):
    for c, (L, eta, ttl) in enumerate(conds):
        ax = axes[r, c]
        ax.plot([0, 1], [0, 1], color="#8d8c86", lw=0.8, ls="--", zorder=1)
        for mo in MODELS:
            cnt, sp, sy = pooled_bins(mech, mo, L, eta)
            ok = cnt >= MINN
            ax.plot(sp[ok] / cnt[ok], sy[ok] / cnt[ok], color=COL[mo], marker=MRK[mo], ms=3.6, lw=1.1, ls=LS[mo], mec="white", mew=0.4, label=mo, zorder=3)
        ax.set_xlim(0.1, 0.95); ax.set_ylim(0.1, 0.95)
        ax.set_title(f"{MECH[mech]}\n{ttl}", fontsize=7.6)
        ax.grid(color="#e6e5e0", lw=0.5)
        if c == 0: ax.set_ylabel("Observed proportion correct")
        if r == 1: ax.set_xlabel("Mean predicted probability")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figs/fig2_reliability.png", dpi=300, facecolor="white")
plt.close(fig)

# ------------------------------------------------ Figure 3: metrics vs history length
cache = "figs/fig3_cache.json"
res = {}
for mech in "AB":
    for mo in MODELS:
        for L in Ls:
            r = pooled_boot(runs, mech, "center", "S1", mo, L, 0.0, "raw", B=400, metrics=("auc", "brier", "nll", "ece10"))
            res[f"{mech}|{mo}|{L}"] = r
json.dump(res, open(cache, "w"))
import pandas as pd
_d = pd.read_csv("long_table.csv")
_d = _d[(_d.cfg == "center") & (_d.setting == "S1") & (_d.eta == 0) & (_d.variant == "raw")]
PT = {(k[0], k[1], k[2]): v for k, v in _d.groupby(["mech", "model", "L"])[["auc", "brier", "nll", "ece10"]].mean().to_dict("index").items()}
mets = [("auc", "ROC-AUC"), ("brier", "Brier score"), ("nll", "Negative log-likelihood"), ("ece10", "ECE (10 bins)")]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharex=True)
x = np.arange(len(Ls))
for r_, mech in enumerate("AB"):
    for c_, (met, lab) in enumerate(mets):
        ax = axes[r_, c_]
        for i, mo in enumerate(MODELS):
            m = np.array([PT[(mech, mo, L)][met] for L in Ls])
            bm = np.array([res[f"{mech}|{mo}|{L}"][met][0] for L in Ls])
            lo = m - (bm - np.array([res[f"{mech}|{mo}|{L}"][met][1] for L in Ls]))
            hi = m + (np.array([res[f"{mech}|{mo}|{L}"][met][2] for L in Ls]) - bm)
            xo = x + (i - 2) * 0.06
            ax.errorbar(xo, m, yerr=[m - lo, hi - m], color=COL[mo], marker=MRK[mo], ms=3.4, lw=1.0, ls=LS[mo], elinewidth=0.6, capsize=1.2,
                        mec="white", mew=0.3, label=mo)
        ax.set_xticks(x); ax.set_xticklabels([str(L) for L in Ls])
        ax.grid(color="#e6e5e0", lw=0.5)
        if r_ == 0: ax.set_title(lab, fontsize=8)
        if r_ == 1: ax.set_xlabel("Visible history length L")
        if c_ == 0: ax.set_ylabel(MECH[mech], fontsize=8)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("figs/fig3_history_length.png", dpi=300, facecolor="white")
print("figures written")
