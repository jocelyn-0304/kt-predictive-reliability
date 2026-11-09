import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"] = "Liberation Serif"
fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(-2, 70); ax.axis("off")
INK, SEC = "#0b0b0b", "#3d3c3a"
def box(x, y, w, h, title, body, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc="#f3f2ef", ec="#52514e", lw=0.9))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=fs, color=SEC, linespacing=1.4)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK, shrinkA=0, shrinkB=0))
W = 30.5
box(0.5, 46, W, 23, "1  Data generation", "Synthetic-KT-A: two-state learning,\nforgetting, guessing, slipping\nSynthetic-KT-B: dynamic IRT\n3,000 learners × 60 events per seed;\n5 independent seeds")
box(34.7, 46, W, 23, "2  Learner-level split", "train 60% · validation 15%\ncalibration 10% · test 15%\nfixed target events 50–59, shared\nby every history length")
box(68.9, 46, W, 23, "3  Models (identical splits)", "Baseline (smoothed item rate)\nBKT with forgetting · 2PL-IRT\nDKT (GRU) · AKT-style attention\ncontrol: DKT-item (item outputs)\nvalidation learners: tuning, stopping")
arrow(31.4, 57.5, 34.2, 57.5); arrow(65.6, 57.5, 68.4, 57.5)
box(0.5, 16.5, 48, 26, "4  Evaluation conditions", "history length L = 3, 5, 10, 20, 50 latest events\nlabel flips η = 0, .05, .10, .20 (visible history only)\nguess g = .10, .20, .30; slip s = .05, .15, .25\ntest-only shift vs. matched window training\ncontrols: clean window reference, position-\nmatched windows, complete short histories", fs=7.4)
box(51.5, 16.5, 48, 26, "5  Post-hoc calibration", "temperature scaling, p′ = σ(logit(p) / T)\nPlatt scaling, p′ = σ(a logit(p) + b) (controls)\nfitted on the calibration learners only:\n(a) target-environment recalibration\n(b) direct transfer from clean, L = 50 windows\ntest labels never enter any fitting step", fs=7.4)
arrow(50, 46, 50, 43.1); arrow(88, 46, 88, 43.1)
arrow(49, 29.5, 51, 29.5)
box(0.5, -1.5, 99, 14.5, "6  Reliability analysis (learner-level resampling; seeds as independent replicates)", "ROC-AUC · Brier score · negative log-likelihood · ECE (10 bins, bin-choice sensitivity) · mean offset\nreliability diagrams · error rate among predictions with max(p, 1−p) ≥ 0.90, with coverage and count · paired differences", fs=7.4)
arrow(75.5, 16.5, 75.5, 13.6)
fig.savefig("figs/fig1_workflow.png", dpi=300, bbox_inches="tight", facecolor="white")
