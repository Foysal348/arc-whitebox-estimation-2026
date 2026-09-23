"""Round 8 (Fable, 24 Sep): does the kernel's remaining depth deficit come from its GAUSSIAN vertex weights?

The quartic kernel (probe_astra7_k4.py) uses d_k = E_G[h^(k)] of the Gaussian marginal N(mu, s^2) at every vertex.  At
depth the pre-activation marginals carry a sizeable kappa3 / kappa4 (the mean closure needs them: -14% / -30%), and the
same Edgeworth expansion that makes the mean closure exact gives the non-Gaussian vertex weights
    E[h^(k)] = w_k + (kappa3/6) w_{k+3} + (kappa4/24) w_{k+4} + (kappa3^2/72) w_{k+6},   w_j = E_G[h^(j)] = (-1)^j He_{j-2}(a) phi(a)/s^(j-1)
with the chain's own D3 (kappa3 diagonal) and g4 (kappa4 diagonal).  This probe recomputes the kernel F0, its pair
polynomial P_H, Sol's K3C_rep and (from r8_k4_1111.py's saved all-distinct term, when present) the full non-pair
prediction with those weights and compares slope / correlation / errG4 against the Gaussian-weight version at the
pre-activation layers 4, 8, 12, 15, 16.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r8_k4_ngvertex.py
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64); n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64)
D = np.load(os.path.join(CAP, "astra7_k4_mlp0.npz"))
DIST = json.load(open("../results/summaries/r8_k4_1111.json")) if os.path.exists("../results/summaries/r8_k4_1111.json") else None
TARGETS = [int(x) for x in os.environ.get("R8_LAYERS", "4,8,12,15,16").split(",")]
METRIC = 2.0
def He(k, x):
    h0, h1 = np.ones_like(x), x
    if k == 0: return h0
    for j in range(1, k): h0, h1 = h1, x * h1 - j * h0
    return h1
def wG(k, a, s):
    """E_G[h^(k)] for h = relu(z), z ~ N(a s, s^2): k = 1 -> Phi; k >= 2 -> (-1)^k He_{k-2}(a) phi(a) / s^(k-1)"""
    if k == 1: return Phi(a)
    return (-1) ** k * He(k - 2, a) * pdf(a) / s ** (k - 1)
def vertex(a, s, k3, k4, ng):
    d = {}
    for k in (1, 2, 3):
        v = wG(k, a, s)
        if ng: v = v + k3 / 6 * wG(k + 3, a, s) + k4 / 24 * wG(k + 4, a, s) + k3 ** 2 / 72 * wG(k + 6, a, s)
        d[k] = v
    return d
def pairpoly(diag, coef22, W):
    W2 = W * W
    return (W2 * W2).T @ diag + 0.5 * np.einsum("ji,ji->i", W2, coef22 @ W2)
def kernel(c, W, ng):
    mu, var = f64(c["mu"]), f64(c["var"]); s = np.sqrt(var); a = mu / s
    dv = vertex(a, s, f64(c["D3"]), f64(c["g4row"]), ng); d1, d2, d3 = dv[1], dv[2], dv[3]
    Co = f64(c["C_off"]); C = Co.copy(); np.fill_diagonal(C, var); Cd = var
    g = f64(c["g4row"]) / METRIC; lam = float(c["lam"]) if "lam" in c.files else None
    U = d1[:, None] * W; Bm = d2[:, None] * W; Cm = d3[:, None] * W
    Q = C @ U; T = Bm * Q; R = C @ T
    f2 = 4 * np.sum(Cm * Q ** 3, 0); f3 = 12 * np.sum(T * R, 0)
    o = np.outer
    dH = 4 * d3 * Cd ** 3 * d1 ** 3 + 12 * Cd ** 3 * d2 ** 2 * d1 ** 2
    cH = (12 * Co ** 2 * (o(d3 * Cd * d1, d1 ** 2) + o(d1 ** 2, d3 * Cd * d1)) + 12 * Co ** 2 * (o(Cd * d2 ** 2, d1 ** 2) + o(d1 ** 2, Cd * d2 ** 2))
          + 24 * Co * (o(Cd * d1 * d2, Cd * d1 * d2) + Co ** 2 * o(d1 * d2, d1 * d2)))
    np.fill_diagonal(cH, 0.0)
    nonpair_G = f2 + f3 - pairpoly(dH, cH, W)                    # star + path minus their own pair content (the f1 edge term is vertex-weight-free at order d1^4 and left out here)
    D21 = f64(c["D21"]).copy(); np.fill_diagonal(D21, 0.0)
    Xc = U; Ss = Bm * (Co @ Xc)
    Trep = D21.T @ (Xc * Xc) + 2 * Xc * (D21 @ Xc) - 2 * f64(c["D3"])[:, None] * (Xc * Xc)
    K3C = 12 * np.sum(Ss * Trep, 0)
    f1 = METRIC * np.sum(U * U, 0) * np.sum(U * ((g - 0 * Cd)[:, None] * U), 0)   # placeholder (lam unknown here); the f1 edge is identical in both versions up to d1
    return nonpair_G, K3C, dv
out = []
for Lt in TARGETS:
    l = Lt - 1; c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); W = Wall[l]
    t4, Ptrue, Pchain, pred, K3Cd, g4, w, a1 = (f64(D[f"L{Lt}_{k}"]) for k in ("t4", "Ptrue", "Pchain", "pred", "K3C", "g4", "w", "a1"))
    target = t4 - Ptrue; m = np.abs(a1) < 1.5
    npG, K3CG, dG_ = kernel(c, W, False); npN, K3CN, dN_ = kernel(c, W, True)
    # the dump's pred includes the f1 (input-kappa4 edge) term; isolate it as pred - (star+path nonpair) with Gaussian weights, and reuse it for the NG version
    f1_np = pred - npG
    predN = f1_np + npN
    rec = {"layer": Lt, "vertex_ratio_ng_over_g": {f"d{k}": {"median": float(np.median(dN_[k] / dG_[k])), "q10": float(np.percentile(dN_[k] / dG_[k], 10)), "q90": float(np.percentile(dN_[k] / dG_[k], 90)),
                                                          "median_decisive": float(np.median((dN_[k] / dG_[k])[m]))} for k in (1, 2, 3)},
           "check_f1_isolation_rms": float(np.sqrt(np.mean(f1_np ** 2))), "check_K3C_G_vs_dump": float(np.max(np.abs(K3CG - K3Cd)) / np.max(np.abs(K3Cd)))}
    def fit(X, y):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None); return list(map(float, coef)), float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
    for nm, P_, K_ in (("gauss", pred, K3CG), ("nongauss", predN, K3CN)):
        cf, r2 = fit(np.stack([P_, K_], 1), target); tot = P_ + K_
        e = Pchain + tot - t4
        rec[nm] = {"fit_coef": cf, "fit_r2": r2, "nonpair_slope": float(np.dot(tot, target) / np.dot(tot, tot)), "nonpair_corr": float(np.corrcoef(tot, target)[0, 1]),
                   "nonpair_corr_decisive": float(np.corrcoef(tot[m], target[m])[0, 1]), "errG4_deploy_S1": float(np.sqrt(np.sum(w * e * e) / np.sum(w * t4 * t4))),
                   "rms_pred": float(np.sqrt(np.mean(P_ ** 2))), "rms_K3C": float(np.sqrt(np.mean(K_ ** 2)))}
    out.append(rec)
    vr = rec["vertex_ratio_ng_over_g"]
    print(f"L{Lt}: NG/G vertex-weight ratio medians d1 {vr['d1']['median']:.3f} d2 {vr['d2']['median']:.3f} d3 {vr['d3']['median']:.3f} (d3 q10/q90 {vr['d3']['q10']:.2f}/{vr['d3']['q90']:.2f}; decisive d3 {vr['d3']['median_decisive']:.3f}) | K3C check {rec['check_K3C_G_vs_dump']:.1e}", flush=True)
    for nm in ("gauss", "nongauss"):
        r = rec[nm]; print(f"   {nm:8s}: target on [kernel, K3C] coef {r['fit_coef'][0]:.2f},{r['fit_coef'][1]:.2f} R2 {r['fit_r2']:.3f} | non-pair slope {r['nonpair_slope']:.2f} corr {r['nonpair_corr']:.3f} (decisive {r['nonpair_corr_decisive']:.3f}) | errG4 deploy S1 {r['errG4_deploy_S1']:.3f} | rms kernel {r['rms_pred']:.2e} K3C {r['rms_K3C']:.2e}", flush=True)
json.dump(out, open("../results/summaries/r8_k4_ngvertex.json", "w"), indent=1, default=float)
