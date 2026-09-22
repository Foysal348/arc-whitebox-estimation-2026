"""Own idea (21 Sep): adjoint / linear-response formulation of the old tier.
The covariance sensitivity of the final mean, J_{l->16} = W_16 D_15 W_15 ... D_{l+1} W_{l+1} (D = diag of E[phi'] = P(z>0)),
is a product of (16-l) random matrices. If its effective rank rho_l (for a 2% Frobenius truncation, i.e. 99.96% energy)
is << n, the per-source-layer products J~ (P o P) and J~ (P o U) cost n^2 rho instead of n^3.
This script measures the singular-value profile of J_{l->16} on the seed-7 MLP with D from a 2e5-sample Monte Carlo
(P(z_i > 0) per layer), and reports the ranks for 90 / 99 / 99.9 / 99.96 % energy, plus the same for the end-to-end
propagators T_{b->16} (their rank bounds the cost of the fixed per-source factors Y_b = T_{b->16} a_b).
Usage: python probe_adjoint_rank.py
"""
import numpy as np, time, sys, os
sys.path.insert(0, os.path.abspath("whest-starterkit"))
n, L, seed = 1024, 16, 7
rng = np.random.default_rng(seed)
scale = (2.0 / n) ** 0.5
W = [(rng.standard_normal((n, n)) * scale).astype(np.float32) for _ in range(L)]   # same recipe as local_engine.build_mlp
# Monte Carlo: fraction of active units per layer (D_l) -- h_l = relu(h_{l-1} @ W_l), row-vector convention
N = 200000; x = np.random.default_rng(1).standard_normal((N, n)).astype(np.float32)
d = []; h = x
for l in range(L):
    z = h @ W[l]; d.append((z > 0).mean(axis=0).astype(np.float64)); h = np.maximum(z, 0)
d = np.array(d)   # d[l] = P(z_l > 0), l = 0..15  (layer l+1 in 1-based)
print(f"active fractions: layer1 mean {d[0].mean():.3f} std {d[0].std():.3f}; layer16 mean {d[-1].mean():.3f} std {d[-1].std():.3f}", flush=True)
# Column-vector convention: z_l = W_l^T h_{l-1}. Covariance propagation to leading order: C_l ~ D_l (W_l^T C_{l-1} W_l) D_l
# so the sensitivity of pre-activation layer 16 to post-activation covariance at layer l is
#   J_{l->16} = W_16^T D_15 W_15^T ... D_{l+1} W_{l+1}^T   (n x n), l in 1-based post-activation layers 1..15
def ranks(sv):
    e = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return {q: int(np.searchsorted(e, q) + 1) for q in (0.9, 0.99, 0.999, 0.9996)}
J = W[L - 1].T.astype(np.float64)      # J_{15->16} = W_16^T
print("J_{l->16}: sensitivity of the final pre-activation covariance to the post-activation covariance at layer l", flush=True)
for l in range(L - 1, 0, -1):      # l = 15 .. 1 (post-activation layer, 1-based)
    if l < L - 1:
        J = J @ (np.diag(d[l]) @ W[l].T.astype(np.float64))   # J_{l->16} = J_{l+1->16} D_{l+1} W_{l+1}^T
    sv = np.linalg.svd(J, compute_uv=False)
    r = ranks(sv)
    print(f"  l={l:2d} ({L - l} factors): sigma_max/sigma_med {sv[0]/np.median(sv):8.1f}  rank for 90/99/99.9/99.96% energy: {r[0.9]:4d} {r[0.99]:4d} {r[0.999]:4d} {r[0.9996]:4d}", flush=True)
