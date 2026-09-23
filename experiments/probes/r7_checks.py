"""Round 7 (Fable, 23 Sep), part 5: three quick checks on stored arrays.

(1) The rank-R_RES truncation of the slice fix-up: the chain's tensor carries S21 = S_sep + Q Bm instead of S21, so its
    (2,1) slice is K21 - tau with tau = Rres - Q Bm; -S_21(tau) hides inside the measured 'all-distinct' error.  tau is
    rebuilt from the captured state with the chain's own range-finder recipe (sketch = first R_RES columns of W[l-1]).
(2) Is the (2,1) slice K[h,y] = kappa3(h,h,y) coherently proportional to C_hy?  (beta = <K,C>/<C,C>, R^2.)  If so, famA has
    a coherent part beta w3 (C C) w1 w1 that RESCALES the Gaussian hub by (1 + beta w3/w2) = (1 - beta alpha/sigma).
(3) The K4->K3 feed's all-distinct content (lam-model): (lam/12) w1_h [w1_x C_hx w2_y + (x<->y)] per hub h, regressed on
    the all-distinct error (slope -1 = the feed's all-distinct content is wrong, 0 = right).

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r7_checks.py
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]; n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64); f32 = lambda x: np.asarray(x, np.float32)
off = ~np.eye(n, dtype=bool); R_RES = 16; R_FB = 16; METRIC_C = 2.0

def S_diag(W, k3): return ((W * W) * k3[:, None]).T @ W
def S_21(W, Y):
    YW = Y @ W; return (W * W).T @ YW + 2.0 * (W * YW).T @ W
def hub(W, g, L, M):
    LW = L @ W; MW = M @ W; Wg = W * g[:, None]
    return (Wg * LW).T @ MW + (Wg * MW).T @ LW + (LW * MW).T @ Wg
def stats(x, y):
    s = float(np.dot(x, y) / np.dot(x, x)); ss = float(np.sum(y * y))
    return dict(slope=s, r2_fit=float(1 - np.sum((y - s * x) ** 2) / ss), r2_1to1=float(1 - np.sum((y - x) ** 2) / ss), corr=float(np.corrcoef(x, y)[0, 1]))
def rangefinder(R, Om, r):
    Y = R @ Om; Q, _ = np.linalg.qr(Y); Y = R @ (R.T @ Q); Q, _ = np.linalg.qr(Y); return Q, Q.T @ R

# lam(b) of every layer from the c1_b = w1(b) lam(b) of the sources captured at layer 15
c15 = np.load(os.path.join(CAP, "r7_cap_L15.npz")); LAM = {}
for s in range(c15["c1_b"].shape[0]):
    b = s + 1; w1b = f64(np.load(os.path.join(CAP, f"r7_cap_L{b}.npz"))["W_all"])[:, 4]; m = w1b > 0.3
    LAM[b] = float(np.median(f64(c15["c1_b"][s])[m] / w1b[m]))
print("lam per birth layer:", {b: round(v, 5) for b, v in LAM.items()}, flush=True)

out = []
for l in (3, 7, 11, 14, 15):
    t0 = time.time()
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P_ = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz")); c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); cn = np.load(os.path.join(CAP, f"r7_cap_L{l+1}.npz"))
    W = f64(Wall[l]); Wf = f32(W)
    Wa = f64(c["W_all"]); w1 = Wa[:, 4]; w2 = Wa[:, 6]; w12 = Wa[:, 5]; w3 = Wa[:, 8]
    k3_post_true = np.diag(f64(P_["Y21"])).copy(); Y21_post_true = f64(P_["Y21"]).copy(); np.fill_diagonal(Y21_post_true, 0.0)
    Dtrue = f64(Tn["Y21"]).copy(); np.fill_diagonal(Dtrue, 0.0); Dchain = f64(cn["D21"]).copy(); np.fill_diagonal(Dchain, 0.0)
    K3c = f64(c["K3"]); K21c = f64(c["K21"]).copy(); np.fill_diagonal(K21c, 0.0)
    e_x = (Dchain - S_diag(W, K3c) - S_21(W, K21c)) - (Dtrue - S_diag(W, k3_post_true) - S_21(W, Y21_post_true)); e_tot = Dchain - Dtrue
    y = e_x[off]; rec = {"post_layer": l}
    # ---- (1) tau: the truncated part of the slice fix-up ----
    Co = f64(c["C_off"]); D21 = f64(c["D21"]).copy(); np.fill_diagonal(D21, 0.0); K1 = f64(c["K1"])
    a_b = w1[:, None] * Co
    Omf = f64(Wall[l - 1][:, :R_FB]); Qf, Bf = rangefinder(D21, Omf, R_FB); D21r = Qf @ Bf
    Xt = 1.5 * w2[:, None] * D21r; Yt = 0.5 * w1[:, None] * D21r.T * w3[None, :]
    X1 = 3.0 * a_b + Xt; Y1 = a_b * w2[None, :] + Yt; xd = np.diag(Xt); yd = np.diag(Yt)
    D21_new = (xd[:, None] * Y1.T + X1 * Y1 + yd[:, None] * X1.T) / 3.0
    dG = f64(c["g4row"]) / METRIC_C; lam = LAM.get(l, 0.0); y_b = w2 * (0.25 * METRIC_C); dgw = w1 * w1 * dG; u_b = y_b * dG; c1_b = w1 * lam
    D21_new += (dgw[:, None] * y_b[None, :]) / 3.0 + (y_b * (2.0 / 3.0))[:, None] * (a_b * c1_b[None, :]) + (w1 * w1)[:, None] * (u_b / 3.0)[None, :]
    D21_w = (w1 ** 2)[:, None] * D21 * w1[None, :]
    rep21 = D21_w + D21_new; np.fill_diagonal(rep21, 0.0)
    S21 = K21c - rep21; np.fill_diagonal(S21, 0.0)
    e_b = w12 - 2.0 * K1 * w1; S_sep = e_b[:, None] * Co * w1[None, :]; np.fill_diagonal(S_sep, 0.0)
    Rres = S21 - S_sep
    Om = f64(Wall[l - 1][:, :R_RES]); Q, Bm = rangefinder(Rres, Om, R_RES); tau = Rres - Q @ Bm
    St = S_21(W, tau); np.fill_diagonal(St, 0.0)
    rms = lambda A: float(np.sqrt(np.mean(A[off] ** 2)))
    rec["tau"] = {"K21_rms": rms(K21c), "S21_rms": rms(S21), "S_sep_rms": rms(S_sep), "Rres_rms": rms(Rres), "tau_rms": rms(tau), "captured_frac": float(1 - rms(tau) ** 2 / rms(Rres) ** 2),
                  "S21_of_tau_rms": rms(St), "e_dist_rms": float(np.sqrt(np.mean(y ** 2))), "e_dist_on_S21tau": stats(St[off], y), "e_tot_on_S21tau": stats(St[off], e_tot[off])}
    # the same for a rank-64 / rank-128 truncation (what a bigger R_RES would leave)
    for rr in (64, 128):
        Q2, B2 = rangefinder(Rres, f64(Wall[l - 1][:, :rr]), rr); t2 = Rres - Q2 @ B2; S2 = S_21(W, t2); np.fill_diagonal(S2, 0.0)
        rec["tau"][f"rank{rr}"] = {"tau_rms": rms(t2), "S21_of_tau_rms": rms(S2), "e_dist_on": stats(S2[off], y)}
    # ---- (2) coherence of the (2,1) slice with C ----
    C_t = f64(T["C"]); Co_t = C_t.copy(); np.fill_diagonal(Co_t, 0.0); K_t = f64(T["Y21"]).copy(); np.fill_diagonal(K_t, 0.0)
    mu_t = f64(T["mean"]); s_ = np.sqrt(np.diag(C_t)); a_ = mu_t / s_
    beta = float(np.dot(K_t[off], Co_t[off]) / np.dot(Co_t[off], Co_t[off])); r2b = float(1 - np.sum((K_t[off] - beta * Co_t[off]) ** 2) / np.sum(K_t[off] ** 2))
    # row-wise (per hub h) beta: K[h,:] vs C[h,:]
    bh = np.sum(K_t * Co_t, 1) / np.maximum(np.sum(Co_t * Co_t, 1), 1e-30)
    rec["K_on_C"] = {"beta": beta, "r2": r2b, "beta_row_median": float(np.median(bh)), "beta_row_iqr": [float(np.percentile(bh, 25)), float(np.percentile(bh, 75))],
                     "corr_betarow_alpha": float(np.corrcoef(bh, a_)[0, 1]), "eff_hub_factor_median": float(np.median(1 + bh * (-a_ / s_)))}
    # ---- (3) the feed's all-distinct content ----
    g = (lam / 6.0) * w1; Lc = Co * w1[None, :]; Mv = np.repeat(w2[None, :], n, axis=0)
    feed = hub(W, g, Lc, Mv); np.fill_diagonal(feed, 0.0)
    rec["feed"] = {"lam": lam, "rms": rms(feed), "e_dist_on_feed": stats(feed[off], y)}
    # the true kappa4 (2,1,1) hub from the post diagonal of l-1 for comparison of scale
    out.append(rec)
    tt = rec["tau"]; kc = rec["K_on_C"]; fd = rec["feed"]
    print(f"post L{l} ({time.time()-t0:.0f}s): K21 {tt['K21_rms']:.2e} S21 {tt['S21_rms']:.2e} S_sep {tt['S_sep_rms']:.2e} Rres {tt['Rres_rms']:.2e} tau(r16) {tt['tau_rms']:.2e} (captured {tt['captured_frac']*100:.0f}%) | S_21(tau) rms {tt['S21_of_tau_rms']:.2e} vs e_dist {tt['e_dist_rms']:.2e}: slope {tt['e_dist_on_S21tau']['slope']:+.2f} R2 {tt['e_dist_on_S21tau']['r2_fit']:.3f} (e_tot: slope {tt['e_tot_on_S21tau']['slope']:+.2f} R2 {tt['e_tot_on_S21tau']['r2_fit']:.3f})", flush=True)
    print(f"   rank64: tau {tt['rank64']['tau_rms']:.2e} S_21 {tt['rank64']['S21_of_tau_rms']:.2e} slope {tt['rank64']['e_dist_on']['slope']:+.2f} R2 {tt['rank64']['e_dist_on']['r2_fit']:.3f} | rank128: tau {tt['rank128']['tau_rms']:.2e} slope {tt['rank128']['e_dist_on']['slope']:+.2f} R2 {tt['rank128']['e_dist_on']['r2_fit']:.3f}", flush=True)
    print(f"   K21 on C_off: beta {kc['beta']:+.4f} R2 {kc['r2']:.3f} | per-row beta median {kc['beta_row_median']:+.4f} IQR {kc['beta_row_iqr'][0]:+.4f}..{kc['beta_row_iqr'][1]:+.4f}, corr with alpha {kc['corr_betarow_alpha']:+.2f} | median effective hub factor 1 - beta_h alpha_h/sigma_h = {kc['eff_hub_factor_median']:.3f}", flush=True)
    print(f"   feed (lam {fd['lam']:.4f}): rms {fd['rms']:.2e}, e_dist on feed: slope {fd['e_dist_on_feed']['slope']:+.2f} R2 {fd['e_dist_on_feed']['r2_fit']:.3f}", flush=True)
json.dump(out, open("../results/summaries/r7_checks.json", "w"), indent=1)
