"""Round 7 (Fable, 23 Sep), part 4: is the chain's LINEAR TRANSPORT of the old sources too strong?

Per source s alive at layer l, its all-distinct contribution to the next slice is computed exactly as the chain does
(legs W^T diag(w1) leg, then (1/3)[(X*P)Y^T + (P*Y)X^T + (X*Y)P^T] per hub block) minus the transport of the source's own
(2,1) / (3) slice entries at l (which the closure handles exactly).  The all-distinct slice error e_dist(l+1) is then
regressed on these per-source contributions (one coefficient per source, i.e. per AGE), on the newborn hub of layer l and
on the fresh Edgeworth terms.  A positive coefficient growing with age = the chain over-propagates old kappa3 content
(the true transport is damped by the K_dist x C curvature, which acts coherently on the source's own structure).

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r7_srcgain.py [layers...]
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]; n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64); f32 = lambda x: np.asarray(x, np.float32)
IW = {1: 4, 2: 6, 3: 8}; off = ~np.eye(n, dtype=bool)
TARGETS = [int(a) for a in sys.argv[1:]] or [3, 7, 11, 14, 15]

def S_diag(W, k3): return ((W * W) * k3[:, None]).T @ W
def S_21(W, Y):
    YW = Y @ W; return (W * W).T @ YW + 2.0 * (W * YW).T @ W
def hub(W, g, L, M):
    LW = L @ W; MW = M @ W; Wg = W * g[:, None]
    return (Wg * LW).T @ MW + (Wg * MW).T @ LW + (LW * MW).T @ Wg
def stats(x, y):
    s = float(np.dot(x, y) / np.dot(x, x)); ss = float(np.sum(y * y))
    return dict(slope=s, r2_fit=float(1 - np.sum((y - s * x) ** 2) / ss), r2_1to1=float(1 - np.sum((y - x) ** 2) / ss), corr=float(np.corrcoef(x, y)[0, 1]))
def multi(X, y, names):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
    return dict(r2=r2, coef={nm: float(c) for nm, c in zip(names, coef)})

def blk_slices(X, P, Y):
    """(2,1) slice [i,c] = T_iic and diagonal T_iii of Sym(X x P x Y) with dense legs (float32 in, float64 out)"""
    D21 = ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0
    D3 = np.sum(X * P * Y, axis=1)
    return f64(D21), f64(D3)

def blocks_of_source(c, s, rres, rfb):
    A = f32(c["A_st"][s]); P = f32(c["P_st"][s]); Z = f32(c["Z_st"][s]); L = f32(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f32(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f32(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f32(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f32(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = f32(Z[:, rres + 1]); Y3 = np.repeat(y[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]

out = []
for l in TARGETS:
    t0 = time.time()
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P_ = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz")); c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); cn = np.load(os.path.join(CAP, f"r7_cap_L{l+1}.npz"))
    W = f64(Wall[l]); Wf = f32(W); rres = int(c["rres"]); rfb = int(c["rfb"]); k = c["A_st"].shape[0]
    Wa = f64(c["W_all"]); w1 = f32(Wa[:, 4]); w2 = f32(Wa[:, 6]); w3 = f32(Wa[:, 8])
    k3_post_true = np.diag(f64(P_["Y21"])).copy(); Y21_post_true = f64(P_["Y21"]).copy(); np.fill_diagonal(Y21_post_true, 0.0)
    Dtrue = f64(Tn["Y21"]).copy(); np.fill_diagonal(Dtrue, 0.0); Dchain = f64(cn["D21"]).copy(); np.fill_diagonal(Dchain, 0.0)
    K3c = f64(c["K3"]); K21c = f64(c["K21"]).copy(); np.fill_diagonal(K21c, 0.0)
    Sx_t = Dtrue - S_diag(W, k3_post_true) - S_21(W, Y21_post_true); Sx_c = Dchain - S_diag(W, K3c) - S_21(W, K21c)
    e_x = Sx_c - Sx_t; e_tot = Dchain - Dtrue
    WT1 = (Wf * w1[:, None]).T                                 # transport: leg' = W^T diag(w1) leg
    # ---- per-source all-distinct contributions to D21(l+1) ----
    contrib = []; d21_old = np.zeros((n, n)); d3_old = np.zeros(n)
    for s in range(k):
        tot = np.zeros((n, n)); sl21 = np.zeros((n, n)); sl3 = np.zeros(n)
        for (X, P, Y) in blocks_of_source(c, s, rres, rfb):
            D21b, D3b = blk_slices(X, P, Y); sl21 += D21b; sl3 += D3b                     # the source's own slices at l (pre-activation, unscaled)
            Xn, Pn, Yn = WT1 @ X, WT1 @ P, WT1 @ Y
            Dn, _ = blk_slices(Xn, Pn, Yn); tot += Dn                                   # its full (2,1) contribution at l+1
        np.fill_diagonal(sl21, 0.0)
        sl21w = sl21 * (f64(w1) ** 2)[:, None] * f64(w1)[None, :]; sl3w = sl3 * f64(w1) ** 3   # post-ReLU (wick-scaled) slice entries
        alld = tot - S_21(W, sl21w) - S_diag(W, sl3w); np.fill_diagonal(alld, 0.0)
        contrib.append(alld); d21_old += sl21; d3_old += sl3
    # check: the summed source dslices must reproduce the chain's D3 / D21 at layer l
    rec = {"post_layer": l, "n_sources": k,
           "check_D21_sources_vs_chain": stats(d21_old[off], f64(c["D21"])[off]), "check_D3_sources_vs_chain": stats(d3_old, f64(c["D3"]))}
    # ---- newborn hub of layer l (chain's own state: X1 = 3 a_b + Xt, Y1 = a_b w2 + Yt with the full D21; P = I) ----
    Co_c = f32(c["C_off"]); D21c = f32(f64(c["D21"])); np.fill_diagonal(D21c, 0.0)
    a_b = w1[:, None] * Co_c; Xt = 1.5 * w2[:, None] * D21c; Yt = 0.5 * w1[:, None] * D21c.T * w3[None, :]
    X1 = 3.0 * a_b + Xt; Y1 = a_b * w2[None, :] + Yt
    D21n, D3n = blk_slices(X1, np.eye(n, dtype=np.float32), Y1); np.fill_diagonal(D21n, 0.0)
    Xn, Pn, Yn = Wf.T @ X1, Wf.T.copy(), Wf.T @ Y1
    nb, _ = blk_slices(Xn, Pn, Yn); nb = nb - S_21(W, D21n) - S_diag(W, D3n); np.fill_diagonal(nb, 0.0)
    recon = sum(contrib) + nb
    rec["recon_check"] = stats(recon[off], Sx_c[off]); rec["recon_rms"] = float(np.sqrt(np.mean(recon[off] ** 2))); rec["Sx_chain_rms"] = float(np.sqrt(np.mean(Sx_c[off] ** 2)))
    # ---- fresh Edgeworth terms (true state) ----
    mu_t, C_t = f64(T["mean"]), f64(T["C"]); s_ = np.sqrt(np.diag(C_t)); a_ = mu_t / s_
    wt = {1: Phi(a_), 2: pdf(a_) / s_, 3: -a_ * pdf(a_) / s_ ** 2}
    Co_t = C_t.copy(); np.fill_diagonal(Co_t, 0.0); K_t = f64(T["Y21"]).copy(); np.fill_diagonal(K_t, 0.0)
    Cw1 = Co_t * wt[1][None, :]; Kw1 = K_t * wt[1][None, :]; KTw2 = K_t.T * wt[2][None, :]; C2w2 = Co_t * Co_t * wt[2][None, :]
    famA = hub(W, wt[3], Cw1, Kw1); famB = hub(W, wt[2], KTw2, Cw1); dedge = hub(W, wt[3], C2w2, Cw1)
    # ---- regressions ----
    y = e_x[off]
    names = [f"src{s}(age{k - s})" for s in range(k)]
    Xs = np.stack([cc[off] for cc in contrib], 1)
    rec["rms_per_source"] = {nm: float(np.sqrt(np.mean(cc[off] ** 2))) for nm, cc in zip(names, contrib)}
    rec["e_dist_rms"] = float(np.sqrt(np.mean(y ** 2))); rec["nb_rms"] = float(np.sqrt(np.mean(nb[off] ** 2)))
    rec["per_source_alone"] = {nm: stats(cc[off], y) for nm, cc in zip(names, contrib)}
    rec["joint_sources"] = multi(Xs, y, names)
    tot_old = sum(contrib); rec["total_old_alone"] = stats(tot_old[off], y); rec["nb_alone"] = stats(nb[off], y)
    Xall = np.concatenate([Xs, np.stack([nb[off], famA[off], famB[off], dedge[off]], 1)], 1)
    rec["joint_all"] = multi(Xall, y, names + ["newborn_hub", "famA", "famB", "dedge"])
    rec["joint_totold_nb_fresh"] = multi(np.stack([tot_old[off], nb[off], famA[off], famB[off], dedge[off]], 1), y, ["total_old", "newborn_hub", "famA", "famB", "dedge"])
    # the same on the TOTAL slice error (what the chain's D21 is off by)
    rec["joint_all_on_e_tot"] = multi(Xall, e_tot[off], names + ["newborn_hub", "famA", "famB", "dedge"])
    out.append(rec)
    rc = rec["recon_check"]; cd = rec["check_D21_sources_vs_chain"]
    print(f"post L{l} -> pre L{l+1} ({time.time()-t0:.0f}s, {k} sources): source dslices vs chain D21: slope {cd['slope']:.4f} R2 {cd['r2_1to1']:.5f} | recon of S_dist_chain: slope {rc['slope']:.3f} R2(1:1) {rc['r2_1to1']:.4f} (rms {rec['recon_rms']:.2e} vs {rec['Sx_chain_rms']:.2e})", flush=True)
    print(f"   e_dist rms {rec['e_dist_rms']:.2e} | per-source rms: " + " ".join(f"{v:.1e}" for v in rec["rms_per_source"].values()) + f" | newborn hub {rec['nb_rms']:.2e}", flush=True)
    print("   per-source ALONE slope/R2: " + " ".join(f"{v['slope']:+.3f}/{v['r2_fit']:.3f}" for v in rec["per_source_alone"].values()), flush=True)
    j = rec["joint_sources"]; print(f"   joint sources: R2 {j['r2']:.3f} coefs " + " ".join(f"{v:+.3f}" for v in j["coef"].values()), flush=True)
    j = rec["joint_all"]; print(f"   joint all: R2 {j['r2']:.3f} | " + " ".join(f"{k_}:{v:+.3f}" for k_, v in j["coef"].items()), flush=True)
    j = rec["joint_totold_nb_fresh"]; print(f"   total_old + newborn + fresh: R2 {j['r2']:.3f} | " + " ".join(f"{k_}:{v:+.3f}" for k_, v in j["coef"].items()) + f" | total_old alone: slope {rec['total_old_alone']['slope']:+.4f} R2 {rec['total_old_alone']['r2_fit']:.3f}", flush=True)
json.dump(out, open("../results/summaries/r7_srcgain.json", "w"), indent=1)
