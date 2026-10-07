"""Round 16 (Astra, SBQM): teacher-forced one-step ceiling of the Signed Bilinear Quartic Memory.
Post-ReLU teacher at layer l (h_l = relu(z_l)), z_{l+1} = h_l @ W[l]. Fit a symmetric H so that Sym(H x C) matches the teacher's
post-ReLU K31 (K31_ij = k(i,i,i,j)) and K22 (K22_ij = k(i,i,j,j)) slices with C = teacher post covariance; H_ii = K4d_i / (3 C_ii).
Transport H_next = W[l]^T H W[l], read q4 = 3 diag(H_next) diag(C_next) and correlate with the next pre-activation teacher's K4d / v^2
on decisive neurons (|mu/sigma| < 1.5). Controls: H = C (Sym(C x C)), H = diag only, and the slice-fit quality.
Gate (Astra): >= 0.70 at every deep target (L12, L15, L16) -> PASS; else kill.
Usage: python experiments/probes/r16_sbqm_ceiling.py
"""
import numpy as np, json

POST = "data/kaggle_out/kaggle_out_k4post/k4post"
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(f"{POST}/weights_seed0.npz")["W"].astype(np.float64)
Wp = np.load(f"{PRE}/weights_seed0.npz")["W"].astype(np.float64)
assert np.allclose(W, Wp), "teacher weight files differ"
out = {}
for l in (3, 7, 11, 14, 15):
    P = np.load(f"{POST}/layer{l}_post_seed0.npz"); N = np.load(f"{PRE}/layer{l + 1}_seed0.npz")
    C = P["C"].astype(np.float64); K31 = P["K31"].astype(np.float64); K22 = P["K22"].astype(np.float64); k4 = P["K4d"].astype(np.float64)
    c = np.diag(C).copy()
    live = c > 1e-12 * c.max()                       # dead post-ReLU neurons (zero variance) carry no memory
    cs = np.where(live, c, 1.0)
    # sanity: the linear map is exact for the covariance
    Cn_pred = W[l].T @ C @ W[l]; Cn = N["C"].astype(np.float64)
    cov_rel = float(np.linalg.norm(Cn_pred - Cn) / np.linalg.norm(Cn))
    vn = np.diag(Cn); mun = N["mean"].astype(np.float64); dec = np.abs(mun / np.sqrt(vn)) < 1.5
    t = N["K4d"].astype(np.float64) / vn ** 2
    corr = lambda q: float(np.corrcoef((q / vn ** 2)[dec], t[dec])[0, 1])
    off = ~np.eye(len(c), dtype=bool)
    # SBQM fit
    Hd = np.where(live, k4 / (3 * cs), 0.0)
    a1 = 1.5 * c[:, None] * np.ones_like(C); b1 = K31 - 1.5 * Hd[:, None] * C            # equation from K31_ij
    a2 = a1.T; b2 = K31.T - 1.5 * Hd[None, :] * C                                          # from K31_ji
    a3 = 2 * C; b3 = K22 - 0.5 * (Hd[:, None] * c[None, :] + c[:, None] * Hd[None, :])
    H = (a1 * b1 + a2 * b2 + a3 * b3) / (a1 ** 2 + a2 ** 2 + a3 ** 2 + 1e-30)
    H = 0.5 * (H + H.T); np.fill_diagonal(H, Hd); H[~live] = 0; H[:, ~live] = 0
    # slice-fit quality (off-diagonal)
    K31f = 1.5 * (Hd[:, None] * C + c[:, None] * H); K22f = 0.5 * (Hd[:, None] * c[None, :] + c[:, None] * Hd[None, :] + 4 * H * C)
    r2 = lambda f, g: float(1 - np.sum((f - g)[off] ** 2) / np.sum((g - g[off].mean())[off] ** 2))
    def readout(Hs):
        Hn = W[l].T @ Hs @ W[l]
        return 3 * np.diag(Hn) * vn
    q_sbqm = readout(H)
    # fairness variants of the H fit: K31 equations only, K22 equation only (ridge-guarded where C_ij ~ 0)
    H31 = (a1 * b1 + a2 * b2) / (a1 ** 2 + a2 ** 2 + 1e-30); H31 = 0.5 * (H31 + H31.T); np.fill_diagonal(H31, Hd); H31[~live] = 0; H31[:, ~live] = 0
    eps = 1e-3 * np.mean(np.abs(C[off]))
    H22 = a3 * b3 / (a3 ** 2 + eps ** 2); np.fill_diagonal(H22, Hd); H22[~live] = 0; H22[:, ~live] = 0
    q_31, q_22 = readout(H31), readout(H22)
    Hc = C * (np.mean((k4 / (3 * cs * cs))[live]))           # H proportional to C (Sym(C x C) control), level irrelevant for corr
    q_cc = readout(Hc)
    q_diag = readout(np.diag(Hd))                  # diagonal-only memory (K4 slice carried, no all-distinct sector)
    # true-slice transport for reference: diagonal of the one-step carry using the TRUE (4), (3,1), (2,2) slices only
    W2 = W[l] ** 2
    # exact contribution of the (4), (3,1) and (2,2) slices to the next diagonal: sum_a W^4 k4_a + 4 sum_{a!=b} W_a^3 W_b K31_ab + 3 sum_{a!=b} W_a^2 W_b^2 K22_ab
    K31o = K31 * off; K22o = K22 * off
    W3 = W[l] ** 3
    q_slices = (W2 ** 2).T @ k4 + 4 * ((K31o @ W[l]) * W3).sum(0) + 3 * ((K22o @ W2) * W2).sum(0)
    out[l + 1] = dict(cov_map_rel_err=cov_rel, n_decisive=int(dec.sum()),
                      fit_r2_K31=r2(K31f, K31), fit_r2_K22=r2(K22f, K22),
                      corr_sbqm=corr(q_sbqm), corr_sbqm_K31only=corr(q_31), corr_sbqm_K22only=corr(q_22), corr_H_eq_C=corr(q_cc), corr_diag_only=corr(q_diag), corr_true_slices=corr(q_slices))
    print(f"post L{l} -> pre L{l + 1}: " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in out[l + 1].items()), flush=True)
json.dump(out, open("results/summaries/r16_sbqm_ceiling.json", "w"), indent=1)
