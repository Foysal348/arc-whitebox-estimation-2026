"""V55 lean map: one coefficient set for all layers >= 6 vs separate sets per layer group, lever-weighted (|sigma phi(a)(a^2-1)|),
cross-net (fit on one network, evaluate on the other). Metric: lever-weighted pattern error relative to the chain's (lower is better).
Usage: python experiments/probes/k4c_groups.py"""
import numpy as np, os
from scipy.stats import norm
S = np.load("data/dumps/k4c_feats.npz")
NETS = {"mlp0": "data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "data/kaggle_out/kaggle_out_k4/k4teach"}
KEEP = [0, 1, 3, 5, 6]
data = {n: {} for n in NETS}
for k in S.files:
    net, L, full = k.split("_"); l = int(L[1:])
    if full != "1" or l < 4: continue
    f = os.path.join(NETS[net], f"layer{l}_seed0.npz")
    if not os.path.exists(f): continue
    T = np.load(f); v = np.diag(T["C"]).astype(np.float64); mu = T["mean"].astype(np.float64); a = mu / np.sqrt(v)
    data[net][l] = (S[k][KEEP], T["K4d"].astype(np.float64) / v ** 2, np.abs(np.sqrt(v) * norm.pdf(a) * (a * a - 1)))
def fit(net, layers):
    A = np.vstack([data[net][l][0].T * data[net][l][2][:, None] for l in layers if l in data[net]])
    y = np.concatenate([data[net][l][1] * data[net][l][2] for l in layers if l in data[net]])
    return np.linalg.lstsq(A, y, rcond=None)[0]
def err(b, net, l):
    X, t, lev = data[net][l]; ch = X[3]; p = b @ X; p = p * ch.mean() / p.mean(); tt = t * ch.mean() / t.mean()
    return np.sum((lev * (p - tt)) ** 2), np.sum((lev * (ch - tt)) ** 2)
GROUPS = {"one": [list(range(6, 16))], "three": [[6, 7, 8, 9], [10, 11, 12], [13, 14, 15]], "two": [list(range(6, 11)), list(range(11, 16))]}
for name, gs in GROUPS.items():
    for src, dst in (("mlp0", "synth"), ("synth", "mlp0"), ("mlp0", "mlp0")):
        e = e0 = 0.0
        for g in gs:
            b = fit(src, g)
            for l in g:
                if l in data[dst]:
                    a_, b_ = err(b, dst, l); e += a_; e0 += b_
        print(f"{name:5s} fit {src:5s} -> {dst:5s}: {e / e0:.3f}")
# LMIN 4: does the >=6 map also help at layers 4-5?
b = fit("mlp0", range(6, 16))
for net in data:
    print(net, "layers 4-5 with the >=6 map:", [f"L{l} {err(b, net, l)[0] / err(b, net, l)[1]:.3f}" for l in (4, 5) if l in data[net]])
