"""Step 1 fairness check: the tree kappa3 diagonal has pattern corr ~0.98 with the teacher, so is its closure failure only a LEVEL
error? Per layer, rescale tree k3 (h = 6 and all history) and tree k4 by ONE scalar each, fitted (oracle, least squares) to the
teacher's k3 / k4, and recompute the closure error vs relu_mean. Also the chain's k3 with an oracle scalar, for symmetry.
Usage: python experiments/probes/r19_tree_rescale.py   (needs data/dumps/r19_tree_diags.npz from r19_tree_ceiling.py)
"""
import numpy as np, json
exec(open("experiments/probes/r19_tree_ceiling.py", encoding="utf-8").read().split("# Hermite-tree diagonals")[0])
D = np.load("data/dumps/r19_tree_diags.npz")
tree = lambda t, h, k: sum(D[f"k{k}_b{b}_t{t}"] for b in range(max(1, t - h), t))
sc = lambda x, y: x * (x @ y) / (x @ x)
out = {}
for t in range(4, 17):
    trelu = T[t]["relu_mean"].astype(np.float64)
    tk3 = T[t]["Y21"].diagonal().astype(np.float64); tk4 = T[t]["K4d"].astype(np.float64)
    ck3 = E[f"L{t}_ck3"].astype(np.float64); ck4 = E[f"L{t}_ck4"].astype(np.float64)
    err = lambda k3, k4: float(np.mean((closure(t, k3, k4) - trelu) ** 2))
    r = {"chain": err(ck3, ck4), "teacher": err(tk3, tk4)}
    for h in (6, 16):
        k3, k4 = tree(t, h, 3), tree(t, h, 4)
        r[f"tree_h{h}_raw"] = err(k3, k4)
        r[f"tree_h{h}_k3scaled+chain_k4"] = err(sc(k3, tk3), ck4)
        r[f"tree_h{h}_both_scaled"] = err(sc(k3, tk3), sc(k4, tk4))
        r[f"tree_h{h}_k3_level_ratio"] = float((k3 @ tk3) / (k3 @ k3))
    r["chain_k3_scaled+chain_k4"] = err(sc(ck3, tk3), ck4)
    out[t] = r
    print(f"L{t:2d} chain {r['chain']:.2e} | tree h6: raw {r['tree_h6_raw']:.2e}, k3 scaled (x{r['tree_h6_k3_level_ratio']:.2f}) + chain k4 {r['tree_h6_k3scaled+chain_k4']:.2e}, both scaled {r['tree_h6_both_scaled']:.2e} | all-history k3 scaled + chain k4 {r['tree_h16_k3scaled+chain_k4']:.2e}")
json.dump(out, open("results/summaries/r19_tree_rescale.json", "w"), indent=1)
