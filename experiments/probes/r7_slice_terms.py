"""Round 7 (Fable, 23 Sep): WHICH dropped term of the post-ReLU kappa3 tensor makes the chain's (2,1)-slice error?

Setting.  z_{l+1} = h_l W[l], so the pre-activation slice at l+1 is EXACTLY
    D21'(a,b) = kappa3(z'_a, z'_a, z'_b) = sum_{ijc} W_ia W_ja W_cb T_ijc,   T = kappa3 tensor of h_l = relu(z_l).
Split T by index pattern: diagonal (i=j=c), the (2,1) slice (two equal), all-distinct.  The chain represents the first
two EXACTLY (its own K3 / K21 closure values ride the M leg); the all-distinct part is the transported sources + the
newborn hub Sym(X1 x I x Y1) with X1 = 3 a_b + 1.5 d(w2) D21, Y1 = a_b d(w2) + 0.5 d(w1) D21^T d(w3).
Edgeworth to first order in the pre-activation kappa3 K and first order in C_off gives, for i, j, c distinct:
    T_ijc = w1_i w1_j w1_c K_ijc                                   (linear transport of the all-distinct part)
          + sum_{hub h} w2_h (C_hx w1_x)(C_hy w1_y)                  (Gaussian hub, O(C^2))              [chain: coef 1]
          + sum_{hub h} (1/2) w3_h [ (C_hx w1_x)(K_hhy w1_y) + (x<->y) ]      (family A, "w3-type")     [chain: coef 1/4]
          + sum_{hub h} (1/2) w2_h [ (K_xxh w2_x)(C_hy w1_y) + (x<->y) ]      (family B, "w2 w2-type")  [chain: coef 1/4]
          + sum_{hub h} (1/2) w3_h [ (C_hx^2 w2_x)(C_hy w1_y) + (x<->y) ]     (doubled edge, O(C^3))    [chain: 0]
          + w2_i w2_j w2_c C_ij C_jc C_ic                              (triangle, O(C^3))          [chain: 0]
          + (1/2) sum_{hub h} w2_h w1_x w1_y kappa4(h,h,x,y)          (kappa4 (2,1,1) slice)       [chain: lam-model feed]
          + K_ijc [C_ij w2_i w2_j w1_c + ...]                          (K_dist x C, transport curvature) [chain: 0]
The port's newborn carries exactly HALF of families A and B (B2 shadowed; the lean chain's c_w2 r_w2 WK12 term of Y1 was
not ported), so the prediction is: chain error = -(1/2) (A + B) + ...  Every hub-form term contracts to D21' in 3 n^3.

Part 1 (this file): decomposition of the chain's D21'(l+1) error into the three index patterns (true post slices from the
post-ReLU teacher at l = 3, 7, 11, 14, 15), the chain's post (2,1) slice accuracy, and the regression of the error on the
hub-form candidates computed from the TRUE layer-l pre-activation data (and from the chain's own state).

Usage (from whest-starterkit, after r7_capture.py):  python -m uv run python ../experiments/probes/r7_slice_terms.py
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]
n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
IW = {(1, 1): 4, (2, 1): 6, (3, 1): 8, (4, 1): 12}      # WICK_PAIRS index of w(k,1) = E[h^(k)]
f64 = lambda x: np.asarray(x, np.float64)
f32 = lambda x: np.asarray(x, np.float32)
off = ~np.eye(n, dtype=bool)

def wicks(mu, var):
    s = np.sqrt(var); a = mu / s; ph = pdf(a)
    return dict(w1=Phi(a), w2=ph / s, w3=-a * ph / s ** 2, w4=(a * a - 1) * ph / s ** 3)

def S_diag(W, k3):
    """sum_i W_ia^2 k3_i W_ib"""
    return ((W * W) * k3[:, None]).T @ W

def S_21(W, Y):
    """Y[i,j] = kappa3(h_i,h_i,h_j), zero diagonal.  sum_{i!=k} W_ia^2 W_kb Y_ik + 2 sum_{i!=j} W_ia W_ja W_ib Y_ij"""
    YW = Y @ W
    return (W * W).T @ YW + 2.0 * (W * YW).T @ W

def hub(W, g, L, M):
    """D21' contraction of T_ijc = (1/2) sum_{hub h in {i,j,c}} g_h (L[h,x] M[h,y] + M[h,x] L[h,y]), x,y the other two.
    With L == M this is T = sum_h g_h L[h,x] L[h,y] (the Gaussian hub)."""
    LW = L @ W; MW = M @ W; Wg = W * g[:, None]
    return (Wg * LW).T @ MW + (Wg * MW).T @ LW + (LW * MW).T @ Wg

def stats(x, y):
    """y regressed on x (through the origin): slope, R^2 of the fit, and R^2 of x as a 1:1 predictor of y."""
    s = float(np.dot(x, y) / np.dot(x, x)); ss = float(np.sum(y * y))
    return dict(slope=s, r2_fit=float(1 - np.sum((y - s * x) ** 2) / ss), r2_1to1=float(1 - np.sum((y - x) ** 2) / ss), corr=float(np.corrcoef(x, y)[0, 1]))

def multi(X, y, names):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
    return dict(r2=r2, coef={nm: float(c) for nm, c in zip(names, coef)})

out = []
for l in (3, 7, 11, 14, 15):
    t0 = time.time()
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz")); c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); cn = np.load(os.path.join(CAP, f"r7_cap_L{l+1}.npz"))
    W = f64(Wall[l])                                   # z_{l+1} = h_l @ W
    rec = {"post_layer": l}
    # ---- sanity: weight convention and sample consistency: C_pre(l+1) == W^T C_post(l) W ----
    Cn_true = f64(Tn["C"]); Cp_true = f64(P["C"]); chk = W.T @ Cp_true @ W
    rec["check_Cpre_from_Cpost_relerr"] = float(np.sqrt(np.mean((chk - Cn_true) ** 2) / np.mean(Cn_true ** 2)))
    # ---- true and chain quantities ----
    k3_post_true = np.diag(f64(P["Y21"])).copy(); Y21_post_true = f64(P["Y21"]).copy(); np.fill_diagonal(Y21_post_true, 0.0)
    Dtrue = f64(Tn["Y21"]).copy(); np.fill_diagonal(Dtrue, 0.0)
    Dchain = f64(cn["D21"]).copy(); np.fill_diagonal(Dchain, 0.0)
    K3c = f64(c["K3"]); K21c = f64(c["K21"]).copy(); np.fill_diagonal(K21c, 0.0)
    e_tot = Dchain - Dtrue
    # chain's post (2,1) slice and diagonal against the post teacher
    rec["post_K21_chain_vs_true"] = stats(K21c[off], Y21_post_true[off])
    rec["post_K21_relerr"] = float(np.sqrt(np.mean((K21c - Y21_post_true)[off] ** 2) / np.mean(Y21_post_true[off] ** 2)))
    rec["post_k3diag_chain_vs_true"] = stats(K3c, k3_post_true)
    rec["post_k3diag_relerr"] = float(np.sqrt(np.mean((K3c - k3_post_true) ** 2) / np.mean(k3_post_true ** 2)))
    # ---- decomposition of D21'(l+1) by index pattern of the post tensor ----
    Sd_t, S2_t = S_diag(W, k3_post_true), S_21(W, Y21_post_true); Sx_t = Dtrue - Sd_t - S2_t
    Sd_c, S2_c = S_diag(W, K3c), S_21(W, K21c); Sx_c = Dchain - Sd_c - S2_c
    e_d, e_2, e_x = Sd_c - Sd_t, S2_c - S2_t, Sx_c - Sx_t
    rms = lambda A: float(np.sqrt(np.mean(A[off] ** 2)))
    rec["rms"] = {"Dtrue": rms(Dtrue), "S_diag_true": rms(Sd_t), "S_21_true": rms(S2_t), "S_dist_true": rms(Sx_t),
                  "e_tot": rms(e_tot), "e_diag": rms(e_d), "e_21": rms(e_2), "e_dist": rms(e_x)}
    rec["rel_err_D21"] = rms(e_tot) / rms(Dtrue)
    # variance budget of e_tot (fractions incl. cross terms sum to 1)
    v = float(np.sum(e_tot[off] ** 2))
    rec["err_var_share"] = {"diag": float(np.sum(e_tot[off] * e_d[off]) / v), "21": float(np.sum(e_tot[off] * e_2[off]) / v), "dist": float(np.sum(e_tot[off] * e_x[off]) / v)}
    # ---- candidate terms from the TRUE layer-l pre-activation data ----
    mu_t, C_t = f64(T["mean"]), f64(T["C"]); var_t = np.diag(C_t).copy(); Co_t = C_t.copy(); np.fill_diagonal(Co_t, 0.0)
    K_t = f64(T["Y21"]).copy(); np.fill_diagonal(K_t, 0.0)          # K_t[i,j] = kappa3(z_i, z_i, z_j)
    wt = wicks(mu_t, var_t)
    # and from the chain's own state at layer l
    mu_c, var_c = f64(c["mu"]), f64(c["var"]); Co_c = f64(c["C_off"]); K_c = f64(c["D21"]).copy(); np.fill_diagonal(K_c, 0.0)
    wc = {k: f64(c["W_all"])[:, IW[(kk, 1)]] for k, kk in (("w1", 1), ("w2", 2), ("w3", 3), ("w4", 4))}
    def terms(Co, K, w):
        Cw1 = Co * w["w1"][None, :]                       # L[h,x] = C_hx w1_x
        Kw1 = K * w["w1"][None, :]                        # M[h,y] = K_hhy w1_y     (K[h,y] = kappa3(h,h,y))
        KTw2 = K.T * w["w2"][None, :]                     # L[h,x] = K_xxh w2_x
        C2w2 = (Co * Co) * w["w2"][None, :]               # L[h,x] = C_hx^2 w2_x
        return {"hub": hub(W, w["w2"], Cw1, Cw1),         # Gaussian hub (in the chain, coef 1)
                "famA": hub(W, w["w3"], Cw1, Kw1),        # (1/2) w3_h [C K + K C]   (chain has 1/4)
                "famB": hub(W, w["w2"], KTw2, Cw1),       # (1/2) w2_h [K^T C + C K^T]  (chain has 1/4)
                "dedge": hub(W, w["w3"], C2w2, Cw1)}      # doubled edge (chain has 0)
    Tt = terms(Co_t, K_t, wt); Tc = terms(Co_c, K_c, wc)
    # kappa4 (2,1,1) hub from the post kappa4 diagonal of layer l-1 (chain's own K4_post, accurate) through W' = W[l-1]
    cp = np.load(os.path.join(CAP, f"r7_cap_L{l-1}.npz")); Q4 = f64(cp["K4"]); Wp = f64(Wall[l - 1])
    U = (Wp * Wp) * (0.5 * wt["w2"])[None, :]; V = Wp * wt["w1"][None, :]; UW = U @ W; VW = V @ W
    Tt["k4hub"] = 2.0 * (UW * VW).T @ (Q4[:, None] * VW) + (VW * VW).T @ (Q4[:, None] * UW)
    # inherited proxy: the newborn hub with the chain's C vs the true C
    inh_hub = Tc["hub"] - Tt["hub"]
    rec["term_rms"] = {k: rms(v_) for k, v_ in Tt.items()}; rec["term_rms"]["inh_hub"] = rms(inh_hub)
    # ---- regressions: e_x (all-distinct error) and e_tot on the candidates ----
    y = e_x[off]; yt = e_tot[off]
    rec["single_on_e_dist"] = {k: stats(v_[off], y) for k, v_ in Tt.items()}; rec["single_on_e_dist"]["inh_hub"] = stats(inh_hub[off], y)
    rec["single_on_e_dist_chainstate"] = {k: stats(v_[off], y) for k, v_ in Tc.items() if k != "hub"}
    AB = Tt["famA"] + Tt["famB"]
    rec["famA+B_on_e_dist"] = stats(AB[off], y); rec["famA+B_on_e_tot"] = stats(AB[off], yt)
    rec["half_famA+B_1to1_r2_on_e_dist"] = float(1 - np.sum((y + 0.5 * AB[off]) ** 2) / np.sum(y * y))
    rec["half_famA+B_1to1_r2_on_e_tot"] = float(1 - np.sum((yt + 0.5 * AB[off]) ** 2) / np.sum(yt * yt))
    names = ["famA", "famB", "dedge", "k4hub", "inh_hub"]
    X = np.stack([Tt[k][off] if k != "inh_hub" else inh_hub[off] for k in names], 1)
    rec["joint_on_e_dist"] = multi(X, y, names); rec["joint_on_e_tot"] = multi(X, yt, names)
    rec["joint_AB_only_on_e_dist"] = multi(X[:, :2], y, names[:2])
    # residual after removing the predicted -(1/2)(A+B): what is left of the all-distinct error
    resid = y + 0.5 * AB[off]
    rec["resid_after_half_AB"] = {"rms": float(np.sqrt(np.mean(resid ** 2))), "on_dedge": stats(Tt["dedge"][off], resid), "on_k4hub": stats(Tt["k4hub"][off], resid), "on_inh_hub": stats(inh_hub[off], resid)}
    out.append(rec)
    r = rec["rms"]; sh = rec["err_var_share"]
    print(f"post L{l} -> pre L{l+1} ({time.time()-t0:.0f}s): check {rec['check_Cpre_from_Cpost_relerr']:.1e} | D21' true rms {r['Dtrue']:.2e} = diag {r['S_diag_true']:.2e} + (2,1) {r['S_21_true']:.2e} + dist {r['S_dist_true']:.2e}"
          f" | err rms {r['e_tot']:.2e} ({rec['rel_err_D21']*100:.1f}%): diag {r['e_diag']:.2e} 21 {r['e_21']:.2e} dist {r['e_dist']:.2e} | var share diag {sh['diag']:.2f} 21 {sh['21']:.2f} dist {sh['dist']:.2f}", flush=True)
    k = rec["post_K21_chain_vs_true"]; print(f"   chain post K21 vs true: slope {k['slope']:.3f} R2(1:1) {k['r2_1to1']:.4f} rel err {rec['post_K21_relerr']*100:.1f}% | post k3 diag: slope {rec['post_k3diag_chain_vs_true']['slope']:.3f} rel err {rec['post_k3diag_relerr']*100:.1f}%", flush=True)
    tr = rec["term_rms"]; print("   term rms: " + ", ".join(f"{k_} {v_:.2e}" for k_, v_ in tr.items()), flush=True)
    s = rec["single_on_e_dist"]; print("   e_dist on each (true-state terms): " + ", ".join(f"{k_}: slope {v_['slope']:+.2f} R2 {v_['r2_fit']:.3f}" for k_, v_ in s.items()), flush=True)
    print(f"   e_dist on famA+famB: slope {rec['famA+B_on_e_dist']['slope']:+.3f} R2 {rec['famA+B_on_e_dist']['r2_fit']:.3f} | e_tot: slope {rec['famA+B_on_e_tot']['slope']:+.3f} R2 {rec['famA+B_on_e_tot']['r2_fit']:.3f}"
          f" | 1:1 test err = -(A+B)/2: R2 e_dist {rec['half_famA+B_1to1_r2_on_e_dist']:.3f}, e_tot {rec['half_famA+B_1to1_r2_on_e_tot']:.3f}", flush=True)
    j = rec["joint_on_e_dist"]; print(f"   joint on e_dist: R2 {j['r2']:.3f} " + ", ".join(f"{k_} {v_:+.2f}" for k_, v_ in j["coef"].items()) + f" | joint on e_tot: R2 {rec['joint_on_e_tot']['r2']:.3f}", flush=True)
    rr = rec["resid_after_half_AB"]; print(f"   residual after -(A+B)/2: rms {rr['rms']:.2e}; on dedge slope {rr['on_dedge']['slope']:+.2f} R2 {rr['on_dedge']['r2_fit']:.3f}; on k4hub slope {rr['on_k4hub']['slope']:+.2f} R2 {rr['on_k4hub']['r2_fit']:.3f}; on inh_hub slope {rr['on_inh_hub']['slope']:+.2f} R2 {rr['on_inh_hub']['r2_fit']:.3f}", flush=True)
json.dump(out, open("../results/summaries/r7_slice_terms.json", "w"), indent=1)
