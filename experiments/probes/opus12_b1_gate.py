"""Opus R12, thread B1 (old-content surrogate), first gate.

Part 1 (captures r7_cap_L14 / L15, every source dense, index s = same source): the OLD content at L15 (sources of age
>= 5, the ones the surrogate must supply) is an EXACT linear image of the L14 sources of age >= 4. The cheapest carried
n x n object that could produce it is its own (2,1) slice, transported by the exact repeated-index slice map
    S'(i,k) = [(V o V)^T S V](i,k) + 2 [(V o (S V))^T V](i,k),   V = diag(w1_14) W[14]   (2 products + 1)
(all index coincidences of the (2,1) slice; the all-distinct part of the old tensor is what it cannot see).
Gate: ||S'-D21_old15|| / ||D21_total15|| <= 2% (after the best scalar); also reported per decisive rows.
Part 2 (teacher, true post-ReLU state at L14 and L15): the one-step MISSING part of the next D21 (true next slice minus
the exact repeated-index transport of the true post slice) - its share, its numerical rank, and whether the previous
step's missing part, transported by the same slice map, predicts it (alignment).

Usage:  python experiments/probes/opus12_b1_gate.py
"""
import os, json, numpy as np
CAP = "data/dumps"; PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
f64 = lambda x: np.asarray(x, np.float64)
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
n = 1024; off = ~np.eye(n, dtype=bool)
def blocks_of_source(c, s, rres, rfb):          # identical to r9_horizon_oracle.py
    A = f64(c["A_st"][s]); P = f64(c["P_st"][s]); Z = f64(c["Z_st"][s]); L = f64(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f64(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f64(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f64(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f64(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = Z[:, rres + 1]; Y3 = np.repeat(y[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]
def slice_of(c, s, rres, rfb):
    sl = np.zeros((n, n))
    for (X, P, Y) in blocks_of_source(c, s, rres, rfb):
        sl += ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0
    return sl                                  # diagonal = kappa3_iii contribution
def slice_map(S, V):
    """repeated-index transport of a (2,1) slice S(a,c)=k3(a,a,c) (diagonal = k3(a,a,a)) under z' = V^T z."""
    Tp = S.copy(); np.fill_diagonal(Tp, np.diag(S) / 3.0)   # avoid triple-counting a=b=c
    U = Tp @ V
    return (V * V).T @ U + 2 * (V * U).T @ V
out = {}
c14 = np.load(os.path.join(CAP, "r7_cap_L14.npz")); c15 = np.load(os.path.join(CAP, "r7_cap_L15.npz"))
k14, k15 = c14["A_st"].shape[0], c15["A_st"].shape[0]
rres, rfb = int(c14["rres"]), int(c14["rfb"])
w1 = f64(c14["W_all"])[:, 4]; V = w1[:, None] * Wall[int(c14["li"]) + 1]
# sanity: the L14 P leg of source 0 transported equals the L15 one
chk = float(np.linalg.norm(V.T @ f64(c14["P_st"][0]) - f64(c15["P_st"][0])) / np.linalg.norm(f64(c15["P_st"][0])))
old14 = sum(slice_of(c14, s, rres, rfb) for s in range(k14) if k14 - s >= 4)
old15 = sum(slice_of(c15, s, rres, rfb) for s in range(k15) if k15 - s >= 5)
D21_15 = f64(c15["D21"]).copy(); np.fill_diagonal(D21_15, 0.0)
pred = slice_map(old14, V)
a = float(np.sum(pred[off] * old15[off]) / np.sum(pred[off] ** 2))
mu, var = f64(c15["mu"]), f64(c15["var"]); dec = np.abs(mu / np.sqrt(var)) < 1.5; M = off & dec[:, None]
p1 = {"transport_check": chk, "n_old15": int(sum(1 for s in range(k15) if k15 - s >= 5)),
      "old_share_of_D21": float(np.linalg.norm(old15[off]) / np.linalg.norm(D21_15[off])),
      "slice_map_scale": a,
      "err_vs_total_D21": float(np.linalg.norm((a * pred - old15)[off]) / np.linalg.norm(D21_15[off])),
      "err_vs_old": float(np.linalg.norm((a * pred - old15)[off]) / np.linalg.norm(old15[off])),
      "err_vs_total_decisive_rows": float(np.linalg.norm((a * pred - old15)[M]) / np.linalg.norm(D21_15[M])),
      "d3_err_vs_old": float(np.linalg.norm(a * np.diag(pred) - np.diag(old15)) / np.linalg.norm(np.diag(old15)))}
out["part1_captures"] = p1
print("Part 1 (captures L14 -> L15, old = age >= 5 at L15):", {k: round(v, 4) if isinstance(v, float) else v for k, v in p1.items()}, flush=True)
# ---- part 2: teacher one-step missing part ----
p2 = {}
miss = {}
for l in (14, 15):
    Yp = f64(np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))["Y21"])
    tr = f64(np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz"))["Y21"])
    Vt = Wall[l]
    rep = slice_map(Yp, Vt)
    m_ = tr - rep; np.fill_diagonal(m_, 0.0); miss[l + 1] = m_
    s = np.linalg.svd(m_, compute_uv=False); e = np.cumsum(s ** 2) / np.sum(s ** 2)
    trz = tr.copy(); np.fill_diagonal(trz, 0.0)
    p2[l + 1] = {"missing_share": float(np.linalg.norm(m_) / np.linalg.norm(trz)),
                 "rank90": int(np.searchsorted(e, 0.90) + 1), "rank98": int(np.searchsorted(e, 0.98) + 1)}
# alignment: does last step's missing part, transported, predict this one? (post-ReLU dressing by Phi of L15)
T15 = np.load(os.path.join(PRE, "layer15_seed0.npz")); m15, v15 = f64(T15["mean"]), np.diag(f64(T15["C"]))
from math import erf
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); d1 = Phi(m15 / np.sqrt(v15))
dressed = (d1 ** 2)[:, None] * miss[15] * d1[None, :]
pr = slice_map(dressed, Wall[15]); np.fill_diagonal(pr, 0.0)
b = float(np.sum(pr * miss[16]) / np.sum(pr * pr))
p2["align_16_from_15"] = {"scale": b, "corr": float(np.corrcoef(pr[off], miss[16][off])[0, 1]),
                          "rel_err_after_scale": float(np.linalg.norm(b * pr - miss[16]) / np.linalg.norm(miss[16]))}
out["part2_teacher"] = p2
print("Part 2 (teacher, missing = true next slice - exact repeated-index transport of the true post slice):", p2, flush=True)
json.dump(out, open("results/summaries/opus12_b1_gate.json", "w"), indent=1)
