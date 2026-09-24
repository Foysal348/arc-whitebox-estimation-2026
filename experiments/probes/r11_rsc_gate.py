"""Round 11 triage (24 Sep): Copilot's Response-State Closure (RSC).
RSC carries C, the slice Y21 = kappa3(z_i,z_i,z_j), K4d and one response matrix R updated linearly,
R' = (W o W) R W^T + alpha Y21 + beta C, and reads the next channels from linear combinations of them.
Most generous one-step gate (teacher-forced, TRUE post-ReLU states, coefficients fitted IN-SAMPLE on the target layer):
target = the teacher's next pre-activation slice at layer 16 (from layer 15), off-diagonal and diagonal (d3); features
  F1 exact repeated-index transport of the true post-15 slice   (Q o Q)^T Tp Q + 2 (Q o U)^T Q
  F2 dressed Gaussian tree from the true post-15 covariance      (r9_dhmp_gate)
  F3 R: the true post-14 slice slice-transported twice with the ReLU dressing a^2 (x) a in between (RSC's memory term)
  F4 Q^T C_post Q and F5 (Q o Q)^T C_post Q                      (RSC's beta C terms)
Least squares over the off-diagonal entries (and separately the diagonal). In-sample fit = an upper bound on RSC.
Bar: the chain's own free-running D21 6.6% (L15), d3 4.7% (L16).

Usage:  python experiments/probes/r11_rsc_gate.py
"""
import numpy as np, os, json
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
n = 1024; off = ~np.eye(n, dtype=bool); f = lambda k: np.asarray(k, np.float64)
def dress(l):
    T0 = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); mu = f(T0["mean"]); s = np.sqrt(np.diag(f(T0["C"])))
    a = Phi(mu / s); b = pdf(mu / s) / s; return a, b / np.maximum(a * a, 1e-6)
def slice_T(S, Q): return (Q * Q).T @ S @ Q
P15 = np.load(os.path.join(POST, "layer15_post_seed0.npz")); P14 = np.load(os.path.join(POST, "layer14_post_seed0.npz"))
Q15, Q14 = W[15], W[14]
truth = f(np.load(os.path.join(PRE, "layer16_seed0.npz"))["Y21"])
a15, tau15 = dress(15)
Y = f(P15["Y21"]); d3 = np.diag(Y).copy(); Tp = Y.copy(); np.fill_diagonal(Tp, d3 / 3.0); U = Tp @ Q15
F1 = (Q15 * Q15).T @ U + 2 * (Q15 * U).T @ Q15
E = f(P15["C"]).copy(); np.fill_diagonal(E, 0.0); R_ = E @ Q15; X = (tau15[:, None] * Q15) * R_
F2 = (tau15[:, None] * (R_ * R_) + 2 * E @ X).T @ Q15
S15 = slice_T(f(P14["Y21"]), Q14); S15d = (a15 ** 2)[:, None] * S15 * a15[None, :]
F3 = slice_T(S15d, Q15)
Cp = f(P15["C"]); F4 = Q15.T @ Cp @ Q15; F5 = (Q15 * Q15).T @ Cp @ Q15
feats = {"F1 repeat": F1, "F2 tree": F2, "F3 R (2-step slice memory)": F3, "F4 QtCQ": F4, "F5 (QoQ)tCQ": F5}
out = {}
for part, mask in (("offdiag", off), ("diag_d3", np.eye(n, dtype=bool))):
    y = truth[mask]; res = {}
    for name, cols in (("F1", ["F1 repeat"]), ("F1+F2", ["F1 repeat", "F2 tree"]),
                       ("F1+F2+F3", ["F1 repeat", "F2 tree", "F3 R (2-step slice memory)"]),
                       ("all five (RSC upper bound)", list(feats))):
        A = np.stack([feats[c][mask] for c in cols], 1); coef = np.linalg.lstsq(A, y, rcond=None)[0]
        res[name] = {"rel_err": float(np.linalg.norm(y - A @ coef) / np.linalg.norm(y)), "coef": [float(c) for c in coef]}
        print(f"{part:8s} {name:28s} rel err {res[name]['rel_err']:.3f}  coef {np.round(coef, 3)}", flush=True)
    out[part] = res
json.dump(out, open("results/summaries/r11_rsc_gate.json", "w"), indent=1)
