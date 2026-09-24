"""Round 9 (GPT 5.6 Sol) - DHMP one-step kill gate, on the teacher arrays (24 Sep).

DHMP drops the kappa3 source bank and rebuilds the next layer's (2,1) slice at every ReLU from pair state only:
  all-distinct post-ReLU kappa3 = dressed Gaussian tree  K3_tree(i,j,k) = tau_i E_ij E_ik + (2 perms),  tau = b / a^2,
  a = Phi(mu/s) = E[relu'], b = phi(mu/s)/s = E[relu''], E = post-ReLU covariance (off-diagonal);
  repeated-index part = the exact post-ReLU slices T(i,i,j) and d3 (from the closure).
Contracted with the next weights Q (z_next = h @ Q):
  R = E Q,  X = (tau Q) o R,  H = tau o R^2 + 2 E X,  D21_tree = H^T Q
  Repeat = (Q^2)^T U + 2 (Q o U)^T Q with U = T' Q  (T' = T off-diagonal + diag d3/3)
Gate (Sol): relative RMS of D21_next - true <= 2.5% at mid/deep layers; kill if > 4% at two of them.
Inputs are TRUE (teacher-forced): post-ReLU E, T, d3 at layers 3, 7, 11, 14, 15; a, b from the pre-activation teacher;
compared with the true next pre-activation slice kappa3(z_p, z_p, z_q). Variants: bare edge G = a a^T o C_pre instead of E;
and the tree switched off (repeated part only) to show what the tree contributes.

Usage:  python experiments/probes/r9_dhmp_gate.py
"""
import numpy as np, os, json
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
off = ~np.eye(1024, dtype=bool)
out = []
for l in (3, 7, 11, 14, 15):
    T0 = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l + 1}_seed0.npz"))
    mu = np.asarray(T0["mean"], np.float64); Cpre = np.asarray(T0["C"], np.float64); s = np.sqrt(np.diag(Cpre)); al = mu / s
    a = Phi(al); b = pdf(al) / s
    tau = b / np.maximum(a * a, 1e-6)
    E = np.asarray(P["C"], np.float64).copy(); np.fill_diagonal(E, 0.0)
    G = np.outer(a, a) * Cpre; np.fill_diagonal(G, 0.0)
    Y = np.asarray(P["Y21"], np.float64)                      # post-ReLU kappa3(h_i, h_i, h_j) (centred third moment)
    d3 = np.diag(Y).copy(); Tp = Y.copy(); np.fill_diagonal(Tp, d3 / 3.0)
    Q = W[l]                                                    # z_(l+1) = h_l @ W[l]
    truth = np.asarray(Tn["Y21"], np.float64)                  # kappa3(z_p, z_p, z_q)
    U = Tp @ Q
    rep = (Q * Q).T @ U + 2 * (Q * U).T @ Q
    def tree(Ed):
        R = Ed @ Q; X = (tau[:, None] * Q) * R; H = tau[:, None] * (R * R) + 2 * Ed @ X
        return H.T @ Q
    tE = tree(E); tG = tree(G)
    err = lambda D: float(np.sqrt(np.mean((D - truth)[off] ** 2) / np.mean(truth[off] ** 2)))
    dist = truth - rep                                          # the true all-distinct contribution
    cc = lambda x: float(np.corrcoef(x[off], dist[off])[0, 1]); sl = lambda x: float(np.dot(x[off], dist[off]) / np.dot(x[off], x[off]))
    rec = {"layer_next": l + 1, "err_repeat_only": err(rep), "err_dressed": err(rep + tE), "err_bare": err(rep + tG),
           "alldistinct_share": float(np.sqrt(np.mean(dist[off] ** 2) / np.mean(truth[off] ** 2))),
           "tree_vs_alldistinct": {"corr_dressed": cc(tE), "slope_dressed": sl(tE), "corr_bare": cc(tG), "slope_bare": sl(tG)}}
    out.append(rec)
    print(f"next L{l+1}: rel err  repeat-only {rec['err_repeat_only']:.3f} | +dressed tree {rec['err_dressed']:.3f} | +bare tree {rec['err_bare']:.3f} "
          f"|| all-distinct share {rec['alldistinct_share']:.3f}; tree vs true all-distinct: corr {cc(tE):.3f} slope {sl(tE):.2f} (bare {cc(tG):.3f}, {sl(tG):.2f})", flush=True)
json.dump(out, open("results/summaries/r9_dhmp_gate.json", "w"), indent=1)
