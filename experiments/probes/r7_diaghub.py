"""Round 7 (Fable, 23 Sep), part 7: the SELF-LOOP hubs.  In the cumulant expansion the marginal cumulants of the hub neuron h
also generate all-distinct entries when h is connected to x and y by two covariance edges:
    kappa3_hhh:  (1/6)  K3_h  w4_h (C_hx w1_x)(C_hy w1_y)      ("D3 hub")
    kappa4_hhhh: (1/24) K4_h  w5_h (C_hx w1_x)(C_hy w1_y)      ("K4 hub")
    kappa3^2:    (1/72) K3_h^2 w6_h (...)                        (skipped, second order)
i.e. the Gaussian hub weight w2_h is really the full derivative series w2_h + K3_h w4_h/6 + K4_h w5_h/24 + ... = E[h''] of the
NON-Gaussian marginal of z_h.  At depth sigma ~ 0.43 and K3_h/sigma^3 ~ 0.12, so these are a few % of the hub on the
mean-dominated neurons.  The chain's newborn uses w2_h only.  Both are hub forms (3 n^3) with the legs the chain already has.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r7_diaghub.py
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]; n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64); off = ~np.eye(n, dtype=bool)

def He(k, x):
    h0, h1 = np.ones_like(x), x
    if k == 0: return h0
    for j in range(1, k): h0, h1 = h1, x * h1 - j * h0
    return h1
def wk(k, a, s):
    """E[h^(k)] for h = relu(z), z ~ N(mu, s^2), a = mu/s:  w1 = Phi, wk = (-1)^k He_{k-2}(a) phi(a) / s^(k-1)"""
    if k == 1: return Phi(a)
    return (-1) ** k * He(k - 2, a) * pdf(a) / s ** (k - 1)
def S_diag(W, k3): return ((W * W) * k3[:, None]).T @ W
def S_21(W, Y):
    YW = Y @ W; return (W * W).T @ YW + 2.0 * (W * YW).T @ W
def hub(W, g, L, M):
    LW = L @ W; MW = M @ W; Wg = W * g[:, None]
    return (Wg * LW).T @ MW + (Wg * MW).T @ LW + (LW * MW).T @ Wg
def stats(x, y):
    s = float(np.dot(x, y) / np.dot(x, x)); ss = float(np.sum(y * y))
    return dict(slope=s, r2_fit=float(1 - np.sum((y - s * x) ** 2) / ss), corr=float(np.corrcoef(x, y)[0, 1]))
def multi(X, y, names):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
    return dict(r2=r2, coef={nm: float(c) for nm, c in zip(names, coef)})

out = []
for l in (3, 7, 11, 14, 15):
    t0 = time.time()
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P_ = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz")); c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); cn = np.load(os.path.join(CAP, f"r7_cap_L{l+1}.npz"))
    W = f64(Wall[l])
    k3_post_true = np.diag(f64(P_["Y21"])).copy(); Y21_post_true = f64(P_["Y21"]).copy(); np.fill_diagonal(Y21_post_true, 0.0)
    Dtrue = f64(Tn["Y21"]).copy(); np.fill_diagonal(Dtrue, 0.0); Dchain = f64(cn["D21"]).copy(); np.fill_diagonal(Dchain, 0.0)
    K3c = f64(c["K3"]); K21c = f64(c["K21"]).copy(); np.fill_diagonal(K21c, 0.0)
    e_x = (Dchain - S_diag(W, K3c) - S_21(W, K21c)) - (Dtrue - S_diag(W, k3_post_true) - S_21(W, Y21_post_true)); y = e_x[off]
    C_t = f64(T["C"]); mu = f64(T["mean"]); s = np.sqrt(np.diag(C_t)); a = mu / s
    Co = C_t.copy(); np.fill_diagonal(Co, 0.0); K = f64(T["Y21"]).copy(); D3 = np.diag(K).copy(); np.fill_diagonal(K, 0.0); K4 = f64(T["K4d"])
    w = {k: wk(k, a, s) for k in range(1, 8)}
    Cw1 = Co * w[1][None, :]; Kw1 = K * w[1][None, :]
    terms = {"hub": hub(W, w[2], Cw1, Cw1),
             "d3hub": hub(W, D3 * w[4] / 6.0, Cw1, Cw1),
             "k4hub": hub(W, K4 * w[5] / 24.0, Cw1, Cw1),
             "d3sqhub": hub(W, D3 * D3 * w[6] / 72.0, Cw1, Cw1),
             "d3k4hub": hub(W, D3 * K4 * w[7] / 144.0, Cw1, Cw1),
             "famA": hub(W, w[3], Cw1, Kw1),
             # family A with the non-Gaussian hub derivative: w3 -> w3 + K3 w5/6 + K4 w6/24 (same structure, next order)
             "famA_ng": hub(W, D3 * w[5] / 6.0 + K4 * w[6] / 24.0, Cw1, Kw1)}
    rms = lambda A: float(np.sqrt(np.mean(A[off] ** 2)))
    rec = {"post_layer": l, "e_dist_rms": rms(e_x), "rms": {k_: rms(v) for k_, v in terms.items()},
           "single": {k_: stats(v[off], y) for k_, v in terms.items() if k_ != "hub"}}
    names = ["famA", "d3hub", "k4hub", "d3sqhub", "d3k4hub", "famA_ng"]
    rec["joint"] = multi(np.stack([terms[k_][off] for k_ in names], 1), y, names)
    # the effective hub weight: E[h''] of the non-Gaussian marginal vs the Gaussian w2 -- on the neurons that matter
    eff = w[2] + D3 * w[4] / 6.0 + K4 * w[5] / 24.0 + D3 * D3 * w[6] / 72.0
    m = np.abs(a) < 1.5
    rec["hub_weight_ratio_median_all"] = float(np.median(eff / w[2])); rec["hub_weight_ratio_median_a<1.5"] = float(np.median((eff / w[2])[m]))
    rec["hub_weight_ratio_iqr_all"] = [float(np.percentile(eff / w[2], 25)), float(np.percentile(eff / w[2], 75))]
    out.append(rec)
    print(f"post L{l} ({time.time()-t0:.0f}s): e_dist {rec['e_dist_rms']:.2e} | rms hub {rec['rms']['hub']:.2e} d3hub {rec['rms']['d3hub']:.2e} k4hub {rec['rms']['k4hub']:.2e} d3sqhub {rec['rms']['d3sqhub']:.2e} d3k4hub {rec['rms']['d3k4hub']:.2e} famA_ng {rec['rms']['famA_ng']:.2e}", flush=True)
    print("   single: " + ", ".join(f"{k_}: slope {v['slope']:+.2f} R2 {v['r2_fit']:.3f}" for k_, v in rec["single"].items()), flush=True)
    j = rec["joint"]; print(f"   joint R2 {j['r2']:.3f}: " + ", ".join(f"{k_} {v:+.2f}" for k_, v in j["coef"].items()), flush=True)
    print(f"   non-Gaussian hub weight / w2: median {rec['hub_weight_ratio_median_all']:.3f} (IQR {rec['hub_weight_ratio_iqr_all'][0]:.3f}..{rec['hub_weight_ratio_iqr_all'][1]:.3f}), on |alpha|<1.5: {rec['hub_weight_ratio_median_a<1.5']:.3f}", flush=True)
json.dump(out, open("../results/summaries/r7_diaghub.json", "w"), indent=1)
