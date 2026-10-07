"""V55 feature ablation on cached chain features: refit (on synth+mlp0 pooled, layers >= 6) with each feature subset, report mean
pattern corr per net. Usage: python experiments/probes/k4c_drop.py"""
import numpy as np, os, itertools
S = np.load("data/dumps/k4c_feats.npz")
NETS = {"mlp0": "data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "data/kaggle_out/kaggle_out_k4/k4teach"}
names = ["s3sq", "y21", "y21T", "c2", "absa", "chain", "1"]
data = {n: [] for n in NETS}
for k in S.files:
    net, L, full = k.split("_"); l = int(L[1:])
    if full != "1" or l < 6: continue
    f = os.path.join(NETS[net], f"layer{l}_seed0.npz")
    if not os.path.exists(f): continue
    T = np.load(f); v = np.diag(T["C"]).astype(np.float64); mu = T["mean"].astype(np.float64); d = np.abs(mu / np.sqrt(v)) < 1.5
    data[net].append((S[k][:, d], T["K4d"].astype(np.float64)[d] / v[d] ** 2))
def score(keep, fit_on):
    A = np.vstack([X[keep].T for net in fit_on for X, t in data[net]]); y = np.concatenate([t for net in fit_on for X, t in data[net]])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return {net: np.mean([np.corrcoef(b @ X[keep], t)[0, 1] for X, t in data[net]]) for net in data}
for drop in [(), (1,), (2,), (3,), (4,), (2, 3), (2, 3, 4), (1, 2, 3, 4), (0,), (5,)]:
    keep = [i for i in range(7) if i not in drop]
    a = score(keep, ["mlp0"]); b = score(keep, ["synth"])
    print(f"drop {[names[i] for i in drop]!s:35s} fit mlp0 -> synth {a['synth']:.3f} (mlp0 {a['mlp0']:.3f}) | fit synth -> mlp0 {b['mlp0']:.3f} (synth {b['synth']:.3f})")
