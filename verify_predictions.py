"""Compare two sets of saved predictions, e.g. a fresh rerun from the seeds against the published results.
Every array except the metadata is compared; the exit code is 0 only if all arrays are present and identical.
Usage: python3 verify_predictions.py <reference_dir> <new_dir>"""
import sys, glob, os
import numpy as np

ref_dir, new_dir = sys.argv[1], sys.argv[2]
worst, n, missing = 0.0, 0, []
for f in sorted(glob.glob(os.path.join(ref_dir, "*.npz"))):
    g = os.path.join(new_dir, os.path.basename(f))
    if not os.path.exists(g):
        missing.append(os.path.basename(f))
        continue
    a, b = np.load(f), np.load(g)
    for k in a.files:
        if k == "meta":
            continue
        if k not in b.files:
            missing.append(f"{os.path.basename(f)}:{k}")
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print("shape mismatch:", os.path.basename(f), k)
            worst = np.inf
            continue
        if x.size:
            worst = max(worst, float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()))
        n += 1
print(f"{ref_dir} vs {new_dir}: compared {n} arrays, maximum absolute difference {worst:g}, missing {len(missing)}")
if missing:
    print("missing:", missing[:20])
sys.exit(0 if worst == 0 and not missing else 1)
