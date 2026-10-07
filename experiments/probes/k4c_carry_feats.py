"""Does a CARRIED feature (previous layer's features / fitted pattern transported by W∘W) add to the V55 kappa4-pattern map?
Uses the cached chain features (data/dumps/k4c_feats.npz) and the weights. For layer l: transported features
T_k = (W∘W) X_{l-1}[k] * v_{l-1}^2 / v_l^2 style (both W orientations tried), plus a recursive carried pattern
p_l = beta·X_l + gamma (W∘W)(p_{l-1} v_{l-1}^2)/v_l^2.  Reports pattern corr (decisive neurons) on both nets.
Usage: python experiments/probes/k4c_carry_feats.py
"""
import numpy as np, os, glob
S = np.load("data/dumps/k4c_feats.npz")
NETS = {"mlp0": "data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "data/kaggle_out/kaggle_out_k4/k4teach"}
data = {}
for net, d in NETS.items():
    W = np.load(glob.glob(d + "/weights_seed*.npz")[0])["W"].astype(np.float64)
    rows = {}
    for k in S.files:
        nn, L, full = k.split("_")
        if nn != net or full != "1": continue
        rows[int(L[1:])] = S[k]
    tt = {}
    for l in sorted(rows):
        f = os.path.join(d, f"layer{l}_seed0.npz")
        if os.path.exists(f):
            T = np.load(f); v = np.diag(T["C"]).astype(np.float64); mu = T["mean"].astype(np.float64)
            tt[l] = (T["K4d"].astype(np.float64) / v ** 2, np.abs(mu / np.sqrt(v)) < 1.5)
    data[net] = (W, rows, tt)
# orientation check: which of W[l-1] / W[l-1].T maps the chain kappa4 of layer l-1 best to layer l's?  (chain ch = X[5])
for net, (W, rows, tt) in data.items():
    for orient in (0, 1):
        cs = []
        for l in range(7, 16):
            if l not in rows or l - 1 not in rows: continue
            M = W[l - 1] if orient == 0 else W[l - 1].T
            pr = (M * M) @ rows[l - 1][5]
            cs.append(np.corrcoef(pr, rows[l][5])[0, 1])
        print(net, "orient", orient, "corr(transported prev chain, chain) mean %.2f" % np.mean(cs))

from scipy.stats import norm
def build(net, mode):
    W, rows, tt = data[net]
    T = {}
    for l in rows:
        f = os.path.join(NETS[net], f"layer{l}_seed0.npz")
        if os.path.exists(f):
            Z = np.load(f); v = np.diag(Z["C"]).astype(np.float64); mu = Z["mean"].astype(np.float64); T[l] = (v, mu)
    out = {}
    for l in sorted(tt):
        if l < 7 or l not in rows or l - 1 not in rows or l - 1 not in T or l not in T: continue
        X = rows[l]; Xp = rows[l - 1]; vp, mup = T[l - 1]; v, _ = T[l]
        phi = norm.cdf(mup / np.sqrt(vp))
        wgt = {"none": 1.0, "phi": phi, "phi2": phi * phi, "phi4": phi ** 4}[mode]
        M = (W[l - 1] * W[l - 1]).T
        tr = [M @ (Xp[k] * vp ** 2 * wgt) / v ** 2 for k in (0, 1, 2, 3, 5)]
        out[l] = (np.vstack([X] + tr), tt[l][0], tt[l][1])
    return out
for mode in ("none", "phi", "phi2", "phi4"):
    B = {net: build(net, mode) for net in data}
    for fit_on in ("mlp0", "synth"):
        A = np.vstack([F[:, d].T for F, t, d in B[fit_on].values()]); y = np.concatenate([t[d] for F, t, d in B[fit_on].values()])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        base = np.linalg.lstsq(A[:, :7], y, rcond=None)[0]
        for net in data:
            c = [np.corrcoef((beta @ F)[d], t[d])[0, 1] for F, t, d in B[net].values()]
            c0 = [np.corrcoef((base @ F[:7])[d], t[d])[0, 1] for F, t, d in B[net].values()]
            print(f"{mode:5s} fit {fit_on:5s} -> {net:5s}: carried {np.mean(c):.3f}  base {np.mean(c0):.3f}  ({' '.join(f'{x:.2f}' for x in c)})")
