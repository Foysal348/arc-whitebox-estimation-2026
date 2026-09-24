"""Round 11 re-verification (24 Sep): stress-test my own Round 11 gates and look for what was missed.

V1  d3 independence: besides the relative error, the norm ratio and correlation of sum_p Q^3 d3 with the true next d3
    (is the diagonal channel small, or orthogonal?).
V2  covariance 'diag + rank r': the eigen-truncation of C is not the optimal diag + low-rank fit of the OFF-diagonal
    entries; redo it by alternating principal-axis factoring (fill the diagonal with the low-rank prediction, 40 iters).
V3  rank-1 D21: the SVD of the zero-diagonal slice is not the best rank-1 fit of the off-diagonal entries; redo it by
    alternating diagonal filling (60 iters).
V4  RSC with LONG memory: add the pre-activation slices of layers 11..15 (teacher, every layer) transported to L16 with
    the ReLU dressing, one free coefficient each, to repeat + tree; in-sample fit (upper bound).
V5  (missed) kappa4 diagonal: how much of the next layer's kappa4 diagonal is the exact repeated-index part that the
    stored post-ReLU (4), (3,1), (2,2) cumulants give, and how much is the (2,1,1) + all-distinct remainder?
    K4d'_a = sum_p Q^4 K4d + 4 sum_{p!=q} Q_p^3 Q_q K31(p,q) + 3 sum_{p!=q} Q_p^2 Q_q^2 K22(p,q) + [(2,1,1) + (1,1,1,1)].
    Compare with the chain's own free-running kappa4 diagonal error (32-34% at depth).

Usage:  python experiments/probes/r11_verify.py
"""
import numpy as np, os, json
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
n = 1024; off = ~np.eye(n, dtype=bool); I = np.eye(n, dtype=bool); f = lambda k: np.asarray(k, np.float64)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
pre = lambda l: np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); post = lambda l: np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
def dress(l):
    T0 = pre(l); mu = f(T0["mean"]); s = np.sqrt(np.diag(f(T0["C"]))); a = Phi(mu / s); b = pdf(mu / s) / s
    return a, b / np.maximum(a * a, 1e-6)
out = {}

print("V1 d3 diagonal channel: next layer | rel err | ||A||/||truth|| | corr(A, truth)")
out["V1"] = []
for l in (3, 7, 11, 14, 15):
    Y = f(post(l)["Y21"]); Q = W[l]; truth = np.diag(f(pre(l + 1)["Y21"])); A = (Q ** 3).T @ np.diag(Y)
    r = {"next": l + 1, "rel": rel(A, truth), "norm_ratio": float(np.linalg.norm(A) / np.linalg.norm(truth)), "corr": float(np.corrcoef(A, truth)[0, 1])}
    out["V1"].append(r); print(f"  L{l+1:2d} | {r['rel']:.3f} | {r['norm_ratio']:.3f} | {r['corr']:.3f}", flush=True)

print("V2 best diag + rank-r fit of the off-diagonal covariance (alternating), L15: r | eigen-trunc | alternating")
C = f(pre(15)["C"]); out["V2"] = {}
for r in (16, 64, 128, 256):
    lam, V = np.linalg.eigh(C); Lr = (V[:, -r:] * lam[-r:]) @ V[:, -r:].T; e0 = rel(Lr[off], C[off])
    X = C.copy()
    for _ in range(40):
        lam, V = np.linalg.eigh(X); Lr = (V[:, -r:] * np.maximum(lam[-r:], 0)) @ V[:, -r:].T
        X = C.copy(); X[I] = Lr[I]
    e1 = rel(Lr[off], C[off]); out["V2"][r] = [e0, e1]; print(f"  r{r:3d} | {e0:.3f} | {e1:.3f}", flush=True)

print("V3 best rank-1 fit of the off-diagonal D21 (alternating diagonal fill): layer | SVD of zero-diag | alternating")
out["V3"] = {}
for L in (11, 15, 16):
    Y = f(pre(L)["Y21"]).copy(); Y0 = Y.copy(); Y0[I] = 0.0
    u, s, vt = np.linalg.svd(Y0); e0 = rel((s[0] * np.outer(u[:, 0], vt[0]))[off], Y[off])
    X = Y0.copy()
    for _ in range(60):
        u, s, vt = np.linalg.svd(X); R1 = s[0] * np.outer(u[:, 0], vt[0]); X = Y0.copy(); X[I] = R1[I]
    e1 = rel(R1[off], Y[off]); out["V3"][L] = [e0, e1]; print(f"  L{L} | {e0:.3f} | {e1:.3f}", flush=True)

print("V4 RSC with long memory at L16 (in-sample upper bound): features | off-diag D21 rel err | d3 rel err")
Q15 = W[15]; truth = f(pre(16)["Y21"]); P15 = post(15); a15, tau15 = dress(15)
Y = f(P15["Y21"]); Tp = Y.copy(); np.fill_diagonal(Tp, np.diag(Y) / 3.0); U = Tp @ Q15
F = {"repeat": (Q15 * Q15).T @ U + 2 * (Q15 * U).T @ Q15}
E = f(P15["C"]).copy(); np.fill_diagonal(E, 0.0); R_ = E @ Q15; X = (tau15[:, None] * Q15) * R_
F["tree"] = (tau15[:, None] * (R_ * R_) + 2 * E @ X).T @ Q15
for l0 in (15, 14, 13, 12, 11):          # pre slice at l0, dressed and transported l0 -> 16
    S = f(pre(l0)["Y21"])
    for l in range(l0, 16):
        a, _ = dress(l); S = (a ** 2)[:, None] * S * a[None, :]; S = (W[l] * W[l]).T @ S @ W[l]
    F[f"mem{l0}"] = S
out["V4"] = {}
for name, cols in (("repeat+tree", ["repeat", "tree"]), ("+mem15..14", ["repeat", "tree", "mem15", "mem14"]),
                   ("+mem15..11 (long memory)", ["repeat", "tree", "mem15", "mem14", "mem13", "mem12", "mem11"])):
    res = []
    for mask in (off, I):
        Am = np.stack([F[c][mask] for c in cols], 1); y = truth[mask]; cf = np.linalg.lstsq(Am, y, rcond=None)[0]
        res.append(float(np.linalg.norm(y - Am @ cf) / np.linalg.norm(y)))
    out["V4"][name] = res; print(f"  {name:26s} | {res[0]:.3f} | {res[1]:.3f}", flush=True)

print("V5 kappa4 diagonal of the next layer from the exact post (4),(3,1),(2,2) cumulants: next | rel err (4) only | (4)+(3,1)+(2,2) | scalar-fit | share of remainder")
out["V5"] = []
for l in (3, 7, 11, 14, 15):
    P = post(l); Q = W[l]; truth = f(pre(l + 1)["K4d"])
    K4 = np.nan_to_num(f(P["K4d"])); K31 = np.nan_to_num(f(P["K31"])).copy(); K22 = np.nan_to_num(f(P["K22"])).copy()
    np.fill_diagonal(K31, 0.0); np.fill_diagonal(K22, 0.0)
    t4 = (Q ** 4).T @ K4
    t31 = 4 * ((Q ** 3) * (K31 @ Q)).sum(0)
    t22 = 3 * ((Q ** 2) * (K22 @ (Q ** 2))).sum(0)
    rep = t4 + t31 + t22; c = float(rep @ truth / (rep @ rep))
    r = {"next": l + 1, "only4": rel(t4, truth), "rep": rel(rep, truth), "rep_scaled": rel(c * rep, truth), "scale": c,
         "remainder_norm_share": float(np.linalg.norm(truth - rep) / np.linalg.norm(truth)),
         "parts_norm": [float(np.linalg.norm(x) / np.linalg.norm(truth)) for x in (t4, t31, t22)]}
    out["V5"].append(r)
    print(f"  L{l+1:2d} | {r['only4']:.3f} | {r['rep']:.3f} | {r['rep_scaled']:.3f} (x{c:.2f}) | parts |t4|,|t31|,|t22| / |truth| = "
          + ", ".join(f"{x:.2f}" for x in r["parts_norm"]), flush=True)
json.dump(out, open("results/summaries/r11_verify.json", "w"), indent=1)
