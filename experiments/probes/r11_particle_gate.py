"""Round 11 triage (24 Sep): 'Deterministic Particle Propagation' (SciMaster). Replace the source bank by K = k*M
deterministic, moment-matched particles pushed through the exact network (particle-wise ReLU), and read every channel
(variance, d3, K4d, C off-diagonal, D21) as empirical cumulants of the cloud. Cost ~K/n u per layer.
Direct free-run test on MLP #0 against the 6e7-sample teacher, for the particle designs the answer describes:
  xpoly   : rotated cross-polytope +-sqrt(n) q_i (2n = 2048 points: exact mean, covariance and odd moments at the input)
  xpoly2r : the same directions with two radial levels matching the Gaussian 4th moment (4n = 4096 points)
  orth448 : 224 random orthonormal directions, antithetic (K = 448 = k M with k = 14, M = 32, the answer's size)
  gauss   : i.i.d. Gaussian draws of the same sizes (plain Monte Carlo reference)
Gate: the chain's own free-running errors (variance 0.14%, C off 0.73%, d3 4.6%, K4d 32%, D21 6.6% at L15).

Usage:  python experiments/probes/r11_particle_gate.py
"""
import numpy as np, os, json
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
n = 1024; off = ~np.eye(n, dtype=bool); rng = np.random.default_rng(0)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
Qr, _ = np.linalg.qr(rng.standard_normal((n, n)))
def designs():
    X = np.sqrt(n) * Qr.T; yield "xpoly 2048", np.vstack([X, -X]), None
    # two radial levels: r1^2 w1 + r2^2 w2 = 1 (per unit direction scaled by sqrt n), E r^4 = 1 + 2/n (Gaussian |x|^4 / n^2)
    m4 = 1.0 + 2.0 / n; d = np.sqrt(m4 - 1.0); r1, r2 = np.sqrt(1 - d), np.sqrt(1 + d)
    yield "xpoly2r 4096", np.vstack([r1 * X, -r1 * X, r2 * X, -r2 * X]), None
    D = np.sqrt(n) * Qr.T[:224]; yield "orth448", np.vstack([D, -D]), None
    for K in (448, 2048, 4096): yield f"gauss {K}", rng.standard_normal((K, n)), None
def cum(Z):
    mu = Z.mean(0); Y = Z - mu; K = Z.shape[0]
    C = Y.T @ Y / K; Y21 = (Y * Y).T @ Y / K; k4 = (Y ** 4).mean(0) - 3 * np.diag(C) ** 2
    return C, Y21, k4
T = {L: np.load(os.path.join(PRE, f"layer{L}_seed0.npz")) for L in (7, 11, 15)}
out = {}
print("design | layer | var | C off | d3 | K4d | D21 off")
for name, X0, _ in designs():
    h = X0; res = {}
    for l in range(1, 16):
        z = h @ W[l - 1]
        if l in T:
            C, Y21, k4 = cum(z); t = T[l]; tC = t["C"].astype(np.float64); tY = t["Y21"].astype(np.float64)
            r = {"var": rel(np.diag(C), np.diag(tC)), "C_off": rel(C[off], tC[off]), "d3": rel(np.diag(Y21), np.diag(tY)),
                 "K4d": rel(k4, t["K4d"].astype(np.float64)), "D21": rel(Y21[off], tY[off])}
            res[l] = r
            print(f"{name:13s} | L{l:2d} | {r['var']:.4f} | {r['C_off']:.4f} | {r['d3']:.3f} | {r['K4d']:.3f} | {r['D21']:.3f}", flush=True)
        h = np.maximum(z, 0.0)
    out[name] = res
json.dump(out, open("results/summaries/r11_particle_gate.json", "w"), indent=1)
