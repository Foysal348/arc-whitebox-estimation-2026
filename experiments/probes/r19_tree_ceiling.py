"""New-architecture step 1 (8 Oct): can Hermite-tree inputs replace the kappa3 source chain in the mean closure?

Teacher-forced, per layer t (public MLP #0): the post-ReLU mean is predicted by the univariate Edgeworth closure from the
teacher's pre-activation mean / variance and a (kappa3, kappa4) diagonal pair:
    E[relu z] ~ sigma (phi + a Phi) - k3 a phi / (6 s^2) + k4 (a^2 - 1) phi / (24 s^3) + k3^2 He4(-a) phi / (72 s^5)
and compared with the teacher's post-ReLU mean (relu_mean). Inputs compared:
  zero          Gaussian closure (k3 = k4 = 0)
  teacher       the 6e7-sample teacher's own k3 / k4 diagonal (floor of this closure)
  chain         the plain chain's k3 / k4 (errbudget_v29plain ck3 / ck4)
  tree_h        Gaussian Hermite-tree births at every layer b, carried linearly by P_(b->t) = W[b] U[b+1]...U[t-1] for at most h ages:
                  k3 star : 6  sum_a g_a P_ai X_ai^2                         (g = phi / (2 sigma), X = C_b (Phi_b P))
                  k4 star : 24 sum_a c_a P_ai X_ai^3, path: 48 v_i^T C_b v_i   (c = -a phi / (6 sigma^2), v = (g P) o X)
                built from the TEACHER mu / C (oracle inputs; the cost question comes later)
  mixes         chain k3 + tree k4, tree k3 + chain k4
Reported: closure MSE vs relu_mean per layer, and the share of the (zero -> teacher) gap that each input closes.
Usage: python experiments/probes/r19_tree_ceiling.py
"""
import numpy as np, json, time
from scipy.special import ndtr

d = np.load("kaggle/r17/data/r17_data.npz")
W = d["W"].astype(np.float64); MU = d["mean"].astype(np.float64); C = d["C"].astype(np.float64)
E = np.load("data/dumps/errbudget_v29plain.npz")
n = W.shape[1]; L = 16
T = {l: np.load(f"data/kaggle_out/kaggle_out_k4pub/k4teach/layer{l}_seed0.npz") for l in range(1, 17)}
v = {l: np.diag(C[l]).copy() for l in range(1, 17)}; s = {l: np.sqrt(v[l]) for l in v}
al = {l: MU[l] / s[l] for l in v}; phi = {l: np.exp(-0.5 * al[l] ** 2) / np.sqrt(2 * np.pi) for l in v}; Phi = {l: ndtr(al[l]) for l in v}
g = {l: phi[l] / (2 * s[l]) for l in v}; c3 = {l: -al[l] * phi[l] / (6 * v[l]) for l in v}
He4 = lambda x: x ** 4 - 6 * x ** 2 + 3


def closure(l, k3, k4):
    a, p, sg = al[l], phi[l], s[l]
    return sg * (p + a * Phi[l]) - k3 * a * p / (6 * sg ** 2) + k4 * (a * a - 1) * p / (24 * sg ** 3) + k3 ** 2 * He4(-a) * p / (72 * sg ** 5)


# Hermite-tree diagonals: births at b, read at t = b+1 .. 16 (age = t - b)
t0 = time.time()
k3t = {(b, t): None for b in range(1, 16) for t in range(b + 1, 17)}
k4t = {}
for b in range(1, 16):
    P = W[b].copy()                                   # h_b -> z_{b+1}
    for t in range(b + 1, 17):
        X = C[b] @ (Phi[b][:, None] * P)
        k3t[(b, t)] = 6.0 * np.sum((g[b][:, None] * P) * X * X, axis=0)
        star = 24.0 * np.sum((c3[b][:, None] * P) * X ** 3, axis=0)
        V = (g[b][:, None] * P) * X
        k4t[(b, t)] = star + 48.0 * np.sum(V * (C[b] @ V), axis=0)
        if t < 16:
            P = P @ (Phi[t][:, None] * W[t])
    print(f"birth {b} done ({time.time() - t0:.0f}s)", flush=True)


def tree(t, h, which):
    src = k3t if which == 3 else k4t
    return sum(src[(b, t)] for b in range(max(1, t - h), t))


out = {}
for t in range(4, 17):
    trelu = T[t]["relu_mean"].astype(np.float64)
    tk3 = T[t]["Y21"].diagonal().astype(np.float64); tk4 = T[t]["K4d"].astype(np.float64)
    ck3 = E[f"L{t}_ck3"].astype(np.float64); ck4 = E[f"L{t}_ck4"].astype(np.float64)
    err = lambda k3, k4: float(np.mean((closure(t, k3, k4) - trelu) ** 2))
    row = {"zero": err(0 * tk3, 0 * tk4), "teacher": err(tk3, tk4), "chain": err(ck3, ck4),
           "teacher_k3+zero_k4": err(tk3, 0 * tk4), "chain_k3+teacher_k4": err(ck3, tk4)}
    for h in (2, 3, 4, 6, 16):
        row[f"tree_h{h}"] = err(tree(t, h, 3), tree(t, h, 4))
    row["chain_k3+tree_k4_h6"] = err(ck3, tree(t, 6, 4))
    row["tree_k3_h6+chain_k4"] = err(tree(t, 6, 3), ck4)
    row["tree_k3_h16+teacher_k4"] = err(tree(t, 16, 3), tk4)
    gap = row["zero"] - row["teacher"]
    row["k3_corr_tree_h6"] = float(np.corrcoef(tree(t, 6, 3), tk3)[0, 1]); row["k3_corr_chain"] = float(np.corrcoef(ck3, tk3)[0, 1])
    out[t] = row
    print(f"L{t:2d} zero {row['zero']:.2e} teacher {row['teacher']:.2e} chain {row['chain']:.2e} | tree h2 {row['tree_h2']:.2e} h3 {row['tree_h3']:.2e} "
          f"h6 {row['tree_h6']:.2e} all {row['tree_h16']:.2e} | chainK3+treeK4 {row['chain_k3+tree_k4_h6']:.2e} treeK3+chainK4 {row['tree_k3_h6+chain_k4']:.2e} "
          f"| gap closed: chain {(row['zero']-row['chain'])/gap:.2f} tree_h6 {(row['zero']-row['tree_h6'])/gap:.2f} | k3 corr tree {row['k3_corr_tree_h6']:.2f} chain {row['k3_corr_chain']:.2f}", flush=True)
json.dump(out, open("results/summaries/r19_tree_ceiling.json", "w"), indent=1)
np.savez("data/dumps/r19_tree_diags.npz", **{f"k3_b{b}_t{t}": x for (b, t), x in k3t.items()},
         **{f"k4_b{b}_t{t}": x for (b, t), x in k4t.items()})
