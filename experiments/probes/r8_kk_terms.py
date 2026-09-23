"""Round 8 (Fable, 23 Sep): the SECOND order in the pre-activation kappa3 of the post-ReLU all-distinct kappa3 tensor.

Round 7 found that 91-96% of the chain's (2,1)-slice error is the error of the ALL-DISTINCT entries of the post-ReLU
kappa3 tensor, and that the first-order-in-K Edgeworth terms (families A, B at half coefficient in the port, doubled
edge, triangle, curvature) explain only 35-45% of it.  The next order is K x K.  For distinct i, j, c the joint cumulant
kappa3(h_i, h_j, h_c) depends only on the trivariate cumulants of (z_i, z_j, z_c); at second order in the (2,1)-slice
entries K[x, y] = kappa3(z_x, z_x, z_y) the connected terms are the 12 pairs of slice hyperedges (xxy) that cover
{i, j, c}, with coefficient (1/2)(1/2)(1/2) x 2 = 1/4 each and Wick weights w_k(h) = E[h^(k)] set by the vertex degrees:
    type P  (1/4) w2_h (K_xxh w2_x)(K_yyh w2_y)                     hub h, both doubles point at the hub      [3 terms]
    type Q  (1/4) w4_h (K_hhx w1_x)(K_hhy w1_y)                     hub h doubled in both hyperedges          [3 terms]
    type R  (1/4) w3_h [(K_xxh w2_x)(K_hhy w1_y) + (x<->y)]         mixed                                     [6 terms]
The ported newborn Sym(X1 x I x Y1) with X1 = 3 a_b + Xt, Y1 = a_b w2 + Yt (Xt = 1.5 d(w2) D21, Yt = 0.5 d(w1) D21^T d(w3))
carries, through its Xt x Yt cross term, exactly HALF of type R (coefficient 1/8) and nothing of P and Q.  The R7_FB_SCALE
knob S multiplies Xt and Yt, so it scales families A, B by S and type R by S^2: at S = 2 the families are exact but type
R is DOUBLED (1/2 instead of 1/4).  Predicted regression coefficients of the chain's fresh all-distinct error on the
exact terms: famA -1/2, famB -1/2, R -1/2, P -1, Q -1 (the chain's deficit of each).

This probe computes every term from the TRUE layer-l pre-activation cumulants (teacher), contracts each to the next
layer's (2,1) slice in 3 n^3 (hub forms), REMOVES each term's own (2,1)-slice content (the closure handles the slice
exactly; round 7 left it in), and regresses the measured all-distinct error e_dist(l+1) on them.  Also: the same terms
with K replaced by its rank-16 range-finder approximation (what thin legs would carry).

Usage (from whest-starterkit, needs data/dumps/r7_cap_L*.npz):  python -m uv run python ../experiments/probes/r8_kk_terms.py
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]
n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64)
off = ~np.eye(n, dtype=bool)
LAYERS = [int(x) for x in os.environ.get("R8_LAYERS", "3,7,11,14,15").split(",")]

def wicks(mu, var):
    s = np.sqrt(var); a = mu / s; ph = pdf(a)
    return dict(w1=Phi(a), w2=ph / s, w3=-a * ph / s ** 2, w4=(a * a - 1) * ph / s ** 3)

def S_diag(W, k3):
    return ((W * W) * k3[:, None]).T @ W

def S_21(W, Y):
    """Y[i,j] = kappa3(h_i,h_i,h_j), zero diagonal."""
    YW = Y @ W
    return (W * W).T @ YW + 2.0 * (W * YW).T @ W

def hub_dist(W, g, L, M):
    """All-distinct part of the D21' contraction of T_ijc = (1/2) sum_{hub h} g_h (L[h,x] M[h,y] + M[h,x] L[h,y]).
    L, M have zero diagonals, so T has no diagonal and its (2,1) slice is Y[i, h] = g_h L[h,i] M[h,i] (i doubled)."""
    LW = L @ W; MW = M @ W; Wg = W * g[:, None]
    full = (Wg * LW).T @ MW + (Wg * MW).T @ LW + (LW * MW).T @ Wg
    Y21 = (L * M * g[:, None]).T                      # Y21[i, h] = g_h L[h,i] M[h,i]
    return full - S_21(W, Y21)

def stats(x, y):
    s = float(np.dot(x, y) / np.dot(x, x)); ss = float(np.sum(y * y))
    return dict(slope=s, r2_fit=float(1 - np.sum((y - s * x) ** 2) / ss), corr=float(np.corrcoef(x, y)[0, 1]))

def multi(X, y, names):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
    return dict(r2=r2, coef={nm: float(c) for nm, c in zip(names, coef)})

def r2_fixed(pred, y):
    return float(1 - np.sum((y - pred) ** 2) / np.sum(y * y))

def rank16(K, Om):
    """the chain's range finder: one power iteration, sketch = 16 columns of the layer weight"""
    Qf, _ = np.linalg.qr(K @ Om); Qf, _ = np.linalg.qr(K @ (K.T @ Qf)); return Qf @ (Qf.T @ K)

def terms(W, Co, K, w):
    Cw1 = Co * w["w1"][None, :]                       # L[h,x] = C_hx w1_x
    Kw1 = K * w["w1"][None, :]                        # M[h,y] = K_hhy w1_y
    KTw2 = K.T * w["w2"][None, :]                     # L[h,x] = K_xxh w2_x
    C2w2 = (Co * Co) * w["w2"][None, :]
    return {"hub": hub_dist(W, w["w2"], Cw1, Cw1),
            "famA": hub_dist(W, w["w3"], Cw1, Kw1),
            "famB": hub_dist(W, w["w2"], KTw2, Cw1),
            "dedge": hub_dist(W, w["w3"], C2w2, Cw1),
            "R": hub_dist(W, 0.5 * w["w3"], KTw2, Kw1),
            "P": hub_dist(W, 0.25 * w["w2"], KTw2, KTw2),
            "Q": hub_dist(W, 0.25 * w["w4"], Kw1, Kw1)}

out = []
for l in LAYERS:
    t0 = time.time()
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz")); c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); cn = np.load(os.path.join(CAP, f"r7_cap_L{l+1}.npz"))
    W = f64(Wall[l])
    rec = {"post_layer": l}
    # ---- the all-distinct error of the chain's D21'(l+1) (as in r7_slice_terms.py) ----
    k3_post_true = np.diag(f64(P["Y21"])).copy(); Y21_post_true = f64(P["Y21"]).copy(); np.fill_diagonal(Y21_post_true, 0.0)
    Dtrue = f64(Tn["Y21"]).copy(); np.fill_diagonal(Dtrue, 0.0)
    Dchain = f64(cn["D21"]).copy(); np.fill_diagonal(Dchain, 0.0)
    K3c = f64(c["K3"]); K21c = f64(c["K21"]).copy(); np.fill_diagonal(K21c, 0.0)
    Sx_t = Dtrue - S_diag(W, k3_post_true) - S_21(W, Y21_post_true)
    Sx_c = Dchain - S_diag(W, K3c) - S_21(W, K21c)
    e_x = Sx_c - Sx_t; y = e_x[off]
    rms = lambda A: float(np.sqrt(np.mean(A[off] ** 2)))
    rec["rms"] = {"S_dist_true": rms(Sx_t), "e_dist": rms(e_x), "e_tot": rms(Dchain - Dtrue)}
    # ---- terms from the TRUE pre-activation cumulants of layer l ----
    mu_t, C_t = f64(T["mean"]), f64(T["C"]); var_t = np.diag(C_t).copy(); Co_t = C_t.copy(); np.fill_diagonal(Co_t, 0.0)
    K_t = f64(T["Y21"]).copy(); np.fill_diagonal(K_t, 0.0)
    wt = wicks(mu_t, var_t)
    s_t = np.sqrt(var_t); a_t = mu_t / s_t
    beta = np.sum(K_t * Co_t, 1) / np.sum(Co_t * Co_t, 1)
    rec["r_beta_alpha_over_sigma"] = {"median": float(np.median(beta * a_t / s_t)), "q25": float(np.percentile(beta * a_t / s_t, 25)), "q75": float(np.percentile(beta * a_t / s_t, 75))}
    Tt = terms(W, Co_t, K_t, wt)
    rec["term_rms"] = {k: rms(v) for k, v in Tt.items()}
    rec["single_on_e_dist"] = {k: stats(v[off], y) for k, v in Tt.items()}
    names = ["famA", "famB", "R", "P", "Q", "dedge"]
    X = np.stack([Tt[k][off] for k in names], 1)
    rec["joint"] = multi(X, y, names)
    rec["joint_AB"] = multi(X[:, :2], y, names[:2])
    rec["joint_ABR"] = multi(X[:, :3], y, names[:3])
    rec["joint_ABRPQ"] = multi(X[:, :5], y, names[:5])
    fixed = {"AB_half": -0.5 * (Tt["famA"] + Tt["famB"]),
             "ABR_half": -0.5 * (Tt["famA"] + Tt["famB"] + Tt["R"]),
             "ABR_half_PQ_full": -0.5 * (Tt["famA"] + Tt["famB"] + Tt["R"]) - Tt["P"] - Tt["Q"],
             "all_theory": -0.5 * (Tt["famA"] + Tt["famB"] + Tt["R"]) - Tt["P"] - Tt["Q"] - Tt["dedge"],
             "S2_change": +0.5 * (Tt["famA"] + Tt["famB"]) + 1.5 * Tt["R"]}       # what R7_FB_SCALE = 2 ADDS to the newborn (A, B +1/2 each; R from 1/8 to 4/8 = +3/2 of the exact R)
    rec["fixed_coef_r2"] = {k: r2_fixed(v[off], y) for k, v in fixed.items()}
    rec["fixed_coef_slope"] = {k: stats(v[off], y)["slope"] for k, v in fixed.items()}
    # the S=2 change against the error: how much of the added content is wanted (ideal slope -1 of e on the added content)
    # ---- rank-16 (thin-leg) versions of the K x K terms ----
    K16 = rank16(K_t, W[:, :16].copy()); np.fill_diagonal(K16, 0.0)
    T16 = terms(W, Co_t, K16, wt)
    rec["rank16"] = {k: {"rel_diff_to_dense": float(np.sqrt(np.sum((T16[k] - Tt[k])[off] ** 2) / np.sum(Tt[k][off] ** 2))), **stats(T16[k][off], y)} for k in ("famA", "famB", "R", "P", "Q")}
    # ---- the same terms from the CHAIN's own state at layer l (deployable inputs) ----
    mu_c, var_c = f64(c["mu"]), f64(c["var"]); Co_c = f64(c["C_off"]); K_c = f64(c["D21"]).copy(); np.fill_diagonal(K_c, 0.0)
    wc = wicks(mu_c, var_c)
    Tc = terms(W, Co_c, K_c, wc)
    rec["chainstate_single_on_e_dist"] = {k: stats(Tc[k][off], y) for k in ("famA", "famB", "R", "P", "Q")}
    rec["chainstate_fixed_r2"] = {"ABR_half_PQ_full": r2_fixed((-0.5 * (Tc["famA"] + Tc["famB"] + Tc["R"]) - Tc["P"] - Tc["Q"])[off], y)}
    out.append(rec)
    r = rec["rms"]; tr = rec["term_rms"]
    print(f"post L{l} -> pre L{l+1} ({time.time()-t0:.0f}s): S_dist {r['S_dist_true']:.2e}, e_dist {r['e_dist']:.2e} | r = beta alpha/sigma median {rec['r_beta_alpha_over_sigma']['median']:.3f}", flush=True)
    print("   term rms (all-distinct part): " + ", ".join(f"{k} {v:.2e}" for k, v in tr.items()), flush=True)
    print("   single: " + ", ".join(f"{k} slope {v['slope']:+.2f} R2 {v['r2_fit']:.3f}" for k, v in rec["single_on_e_dist"].items() if k != "hub"), flush=True)
    j = rec["joint"]; print(f"   joint(6) R2 {j['r2']:.3f}: " + ", ".join(f"{k} {v:+.2f}" for k, v in j["coef"].items()) + f" | AB {rec['joint_AB']['r2']:.3f}, ABR {rec['joint_ABR']['r2']:.3f}, ABRPQ {rec['joint_ABRPQ']['r2']:.3f}", flush=True)
    print("   fixed-coefficient R2: " + ", ".join(f"{k} {v:.3f} (slope {rec['fixed_coef_slope'][k]:+.2f})" for k, v in rec["fixed_coef_r2"].items()), flush=True)
    print("   rank-16 K: " + ", ".join(f"{k} diff {v['rel_diff_to_dense']:.2f} slope {v['slope']:+.2f} R2 {v['r2_fit']:.3f}" for k, v in rec["rank16"].items()), flush=True)
    print("   chain-state terms: " + ", ".join(f"{k} slope {v['slope']:+.2f} R2 {v['r2_fit']:.3f}" for k, v in rec["chainstate_single_on_e_dist"].items()) + f" | fixed ABR/2+P+Q R2 {rec['chainstate_fixed_r2']['ABR_half_PQ_full']:.3f}", flush=True)
json.dump(out, open("../results/summaries/r8_kk_terms.json", "w"), indent=1)
