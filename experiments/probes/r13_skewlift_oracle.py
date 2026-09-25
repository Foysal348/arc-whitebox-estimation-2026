"""Round 13 falsifier (26 Sep): 'Source-Coupled Skew-Lift Quartic Carrier', the proposal's own teacher-forced oracle.
Carried kappa4 memory = sum_s Sym4(T_s (x) r_s), T_s = the chain's kappa3 source s, r_s a free n-vector per source.
  1. At capture L14 (pre-activation z_14, 13 live sources) build each source's slice D21_s and diagonal d3_s
     (blocks_of_source, as in r9_horizon_oracle.py), dress to post-ReLU h_14 at first order: D21p = Phi_a^2 Phi_b D21, d3p = Phi^3 d3.
  2. Fit R = {r_s} by least squares so that K22_hat = sum_s 1/2 (D21p_s[a,b] r_b + D21p_s[b,a] r_a) and
     K31_hat = sum_s 1/4 (d3p_s[a] r_b + 3 D21p_s[a,b] r_a) match the TEACHER post-ReLU K22 / K31 off-diagonals at layer 14.
  3. Predict the next pre-activation kappa4 diagonal q4[i] = sum_s d3'_s[i] (W_14^T r_s)[i], with d3'_s the per-source
     diagonal of the same source at capture L15, and compare with the teacher's K4d at layer 15 on the decisive neurons.
Gate (proposal): pattern corr >= 0.70. Baselines on the same mask: chain 0.26, exact linear carry 0.785 (Opus R12).

Usage:  python experiments/probes/r13_skewlift_oracle.py
"""
import numpy as np, os, json, sys
from math import erf
from scipy.sparse.linalg import LinearOperator, lsqr
sys.path.insert(0, "experiments/probes")
CAP = "data/dumps"; PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
f64 = lambda x: np.asarray(x, np.float64)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5)))

def blocks_of_source(c, s, rres, rfb, n):          # identical to r9_horizon_oracle.py
    A = f64(c["A_st"][s]); P = f64(c["P_st"][s]); Z = f64(c["Z_st"][s]); L = f64(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f64(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f64(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f64(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f64(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = Z[:, rres + 1]; Y3 = np.repeat(y[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]

def source_slices(path):
    c = np.load(path); rres = int(c["rres"]); rfb = int(c["rfb"]); k = c["A_st"].shape[0]; n = c["A_st"].shape[1]
    out = []
    for s in range(k):
        sl = np.zeros((n, n))
        for (X, P, Y) in blocks_of_source(c, s, rres, rfb, n):
            sl += ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0
        out.append(sl)
    return out, int(c["li"])

S14, li14 = source_slices(os.path.join(CAP, "r7_cap_L14.npz")); L = li14 + 1       # pre-activation layer 14
S15, li15 = source_slices(os.path.join(CAP, "r7_cap_L15.npz")); assert li15 + 1 == L + 1
k = len(S14); n = S14[0].shape[0]; off = ~np.eye(n, dtype=bool)
tp = np.load(os.path.join(PRE, f"layer{L}_seed0.npz")); ph = Phi(f64(tp["mean"]) / np.sqrt(np.diag(f64(tp["C"]))))
Dp = [(ph ** 2)[:, None] * S * ph[None, :] for S in S14]
d3p = [np.diag(S14[s]).copy() * ph ** 3 for s in range(k)]
for S in Dp: np.fill_diagonal(S, 0.0)
post = np.load(os.path.join(POST, f"layer{L}_post_seed0.npz"))
K22t = np.nan_to_num(f64(post["K22"])).copy(); K31t = np.nan_to_num(f64(post["K31"])).copy()
np.fill_diagonal(K22t, 0.0); np.fill_diagonal(K31t, 0.0)
s22, s31 = np.linalg.norm(K22t), np.linalg.norm(K31t)

def mv(x):
    R = x.reshape(k, n); K22 = np.zeros((n, n)); K31 = np.zeros((n, n))
    for s in range(k):
        X = Dp[s] * R[s][None, :]; K22 += 0.5 * (X + X.T)
        K31 += 0.25 * np.outer(d3p[s], R[s]) + 0.75 * Dp[s] * R[s][:, None]
    np.fill_diagonal(K22, 0.0); np.fill_diagonal(K31, 0.0)
    return np.concatenate([(K22 / s22).ravel(), (K31 / s31).ravel()])
def rmv(y):
    G22 = y[:n * n].reshape(n, n) / s22; G31 = y[n * n:].reshape(n, n) / s31
    G22 = G22.copy(); G31 = G31.copy(); np.fill_diagonal(G22, 0.0); np.fill_diagonal(G31, 0.0)
    g = np.zeros((k, n)); Gs = G22 + G22.T
    for s in range(k):
        g[s] = 0.5 * (Gs * Dp[s]).sum(0) + 0.25 * (G31.T @ d3p[s]) + 0.75 * (G31 * Dp[s]).sum(1)
    return g.ravel()
A = LinearOperator((2 * n * n, k * n), matvec=mv, rmatvec=rmv, dtype=np.float64)
b = np.concatenate([(K22t / s22).ravel(), (K31t / s31).ravel()])
sol = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=int(os.environ.get("R13_ITERS", "300")), show=False)
R = sol[0].reshape(k, n)
fit = mv(sol[0]); r22 = np.linalg.norm(fit[:n * n] - b[:n * n]) / np.linalg.norm(b[:n * n]); r31 = np.linalg.norm(fit[n * n:] - b[n * n:]) / np.linalg.norm(b[n * n:])
print(f"fit (layer {L} post): rel resid K22 {r22:.3f}, K31 {r31:.3f}, lsqr iters {sol[2]}", flush=True)

W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)[L]          # z_{L+1} = h_L @ W[L]
d3n = [np.diag(S15[s]).copy() for s in range(k)]                                        # same sources at L+1
q4 = sum(d3n[s] * (W.T @ R[s]) for s in range(k))
tn = np.load(os.path.join(PRE, f"layer{L + 1}_seed0.npz")); mu = f64(tn["mean"]); v = np.diag(f64(tn["C"])); tk4 = f64(tn["K4d"])
dec = np.abs(mu / np.sqrt(v)) < 1.5
ex = lambda x: x / v ** 2
D = np.load("data/dumps/errbudget_v29plain.npz"); ck4 = D[f"L{L + 1}_ck4"]
corr = lambda a, b_: float(np.corrcoef(ex(a)[dec], ex(b_)[dec])[0, 1])
res = {"layer_fit": L, "layer_pred": L + 1, "n_sources": k, "fit_resid_K22": r22, "fit_resid_K31": r31,
       "corr_q4_vs_teacher": corr(q4, tk4), "corr_chain_vs_teacher": corr(ck4, tk4),
       "corr_chain_plus_q4": corr(ck4 + q4, tk4), "level_q4_over_teacher": float(np.mean(ex(q4)[dec]) / np.mean(ex(tk4)[dec]))}
cf = np.linalg.lstsq(np.stack([ex(ck4)[dec], ex(q4)[dec], np.ones(dec.sum())], 1), ex(tk4)[dec], rcond=None)[0]
res["best_blend_coef_chain_q4_const"] = cf.tolist()
res["corr_best_blend_insample"] = float(np.corrcoef(cf[0] * ex(ck4)[dec] + cf[1] * ex(q4)[dec], ex(tk4)[dec])[0, 1])
print(json.dumps(res, indent=1))
json.dump(res, open("results/summaries/r13_skewlift_oracle_it{}.json".format(os.environ.get("R13_ITERS", "300"))+"", "w"), indent=1)
