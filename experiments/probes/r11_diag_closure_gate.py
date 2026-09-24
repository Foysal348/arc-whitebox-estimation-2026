"""Round 11 triage (24 Sep): the three answers (SciMaster 'matrix sketch', 'independent-cumulant', 'diagonal-channel
transport') all carry only the DIAGONAL of kappa3 (d3) and propagate it through the next weights by Hadamard powers,
plus (some) a covariance-driven term, plus a low-rank (+diagonal) covariance or a rank-one kappa3.
One-step, teacher-forced gates on MLP #0 (true post-ReLU state of layer l -> next pre-activation d3 at l+1), z' = h @ Q:
  A  independence:     d3'_q = sum_p Q_pq^3 d3_p                         (what 'independent-cumulant' carries)
  B  A + dressed tree: + diag of the Gaussian-tree all-distinct part built from the post covariance (r9_dhmp_gate)
  C  exact repeated-index part (needs the full post Y21 slice, i.e. NOT diagonal-only) + tree
  truth: diag of the teacher's next Y21.  Bar: the chain's own free-running d3 error 3.2-4.7% (r11_chain_channel_ref).
Also the structural ceilings of the other two carried objects, on the TRUE arrays:
  best rank-1 approximation of the off-diagonal D21 (the 's t x t x t' age-mode kappa3 -> D21 = s (Wt)_i^2 (Wt)_j is
  at best rank one), and best 'diag + rank-r' covariance (r = 4, 16, 32, 64) for the off-diagonal C (chain: 0.4-0.7%).

Usage:  python experiments/probes/r11_diag_closure_gate.py
"""
import numpy as np, os, json
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
n = 1024; off = ~np.eye(n, dtype=bool)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
out = {"d3": [], "rank": []}
print("next layer | d3 rel err: A indep | B indep+tree | C exact-repeat+tree | (share of truth from off-diag repeat + all-distinct)")
for l in (3, 7, 11, 14, 15):
    T0 = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l + 1}_seed0.npz"))
    mu = np.asarray(T0["mean"], np.float64); Cpre = np.asarray(T0["C"], np.float64); s = np.sqrt(np.diag(Cpre)); al = mu / s
    a = Phi(al); b = pdf(al) / s; tau = b / np.maximum(a * a, 1e-6)
    E = np.asarray(P["C"], np.float64).copy(); np.fill_diagonal(E, 0.0)
    Y = np.asarray(P["Y21"], np.float64); d3p = np.diag(Y).copy()
    Q = W[l]; truth = np.diag(np.asarray(Tn["Y21"], np.float64)).copy()
    A = (Q ** 3).T @ d3p
    R = E @ Q; X = (tau[:, None] * Q) * R; tree = np.diag((tau[:, None] * (R * R) + 2 * E @ X).T @ Q)
    Tp = Y.copy(); np.fill_diagonal(Tp, d3p / 3.0); U = Tp @ Q
    rep = np.diag((Q * Q).T @ U + 2 * (Q * U).T @ Q)
    r = {"layer_next": l + 1, "A_indep": rel(A, truth), "B_indep_tree": rel(A + tree, truth), "C_rep_tree": rel(rep + tree, truth),
         "C_rep_only": rel(rep, truth), "share_not_diag": rel(A, truth)}
    out["d3"].append(r)
    print(f"L{l+1:2d} | {r['A_indep']:.3f} | {r['B_indep_tree']:.3f} | {r['C_rep_tree']:.3f} (repeat only {r['C_rep_only']:.3f})", flush=True)
print("layer | best rank-1 D21 off-diag err | best diag+rank-r C off-diag err (r=4/16/32/64)")
for L in (7, 11, 14, 15, 16):
    Tn = np.load(os.path.join(PRE, f"layer{L}_seed0.npz"))
    Y = np.asarray(Tn["Y21"], np.float64).copy(); np.fill_diagonal(Y, 0.0)
    sv = np.linalg.svd(Y, compute_uv=False); r1 = float(np.sqrt(max(0.0, 1 - sv[0] ** 2 / np.sum(sv ** 2))))
    C = np.asarray(Tn["C"], np.float64); lam, V = np.linalg.eigh(C); lam, V = lam[::-1], V[:, ::-1]
    Co = C[off]; errs = {}
    for rr in (4, 16, 32, 64):
        Lr = (V[:, :rr] * lam[:rr]) @ V[:, :rr].T
        errs[rr] = rel(Lr[off], Co)                     # the diagonal is carried exactly; this is the off-diagonal error
    out["rank"].append({"layer": L, "D21_rank1": r1, "C_diag_plus_rank": errs})
    print(f"L{L:2d} | {r1:.3f} | " + " / ".join(f"{v:.3f}" for v in errs.values()), flush=True)
json.dump(out, open("results/summaries/r11_diag_closure_gate.json", "w"), indent=1)
