"""GPT R5 gate 3: TT rank-growth probe of the network function in INPUT coordinates.
For a small He-init ReLU MLP (width n, depth L) and one output neuron j, approximate f_j(x) = h_L(x)_j on a tensor grid of
Gauss-Hermite nodes (q per input dimension) by rank-adaptive TT-cross (teneva), then integrate the TT against the Gauss-Hermite
weights to estimate E[f_j(x)] and compare with a Monte-Carlo truth. Report the TT ranks and the integral error as depth grows.
Kill (GPT): ranks grow ~linearly with width, or cross evaluations exceed a matched MC count with no accuracy gain.
Usage: python probe_tt_rank.py [n=64] [q=8] [rmax=32]
"""
import sys, time, numpy as np, teneva
n = int(sys.argv[1]) if len(sys.argv) > 1 else 64; q = int(sys.argv[2]) if len(sys.argv) > 2 else 8; rmax = int(sys.argv[3]) if len(sys.argv) > 3 else 32
nodes, wts = np.polynomial.hermite_e.hermegauss(q); wts = wts / wts.sum()      # probabilists' Hermite: nodes for N(0,1)
rng = np.random.default_rng(3)
def mlp(L):
    return [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float64) for _ in range(L)]
for L in (1, 2, 4, 8, 16):
    Ws = mlp(L); j = 0
    def fwd(X):
        h = X
        for W in Ws: h = np.maximum(h @ W, 0.0)
        return h
    # MC truth (2e6 samples)
    N = 2_000_000; acc = 0.0; acc2 = 0.0
    for _ in range(20):
        X = rng.standard_normal((N // 20, n)); v = fwd(X)[:, j]; acc += v.sum(); acc2 += (v * v).sum()
    truth = acc / N; se = np.sqrt((acc2 / N - truth ** 2) / N)
    def f(I):   # I: (samples, n) multi-indices -> values of f_j at the grid points
        return fwd(nodes[I])[:, j]
    calls = {"k": 0}
    def fc(I):
        calls["k"] += len(I); return f(I)
    t0 = time.time()
    Y0 = teneva.rand([q] * n, r=2)
    info = {}
    Y = teneva.cross(fc, Y0, m=int(2e6), e=None, nswp=int(sys.argv[4]) if len(sys.argv) > 4 else 2, dr_max=2, info=info, log=False)
    # integral: contract each core with the GH weights
    est = teneva.sum(teneva.mul_scalar(Y, 1.0) if False else [G * wts[None, :, None] for G in Y])
    ranks = teneva.ranks(Y)
    print(f"n={n} q={q} L={L:2d}: TT ranks max {max(ranks)} (erank {teneva.erank(Y):.1f}), cross evals {calls['k']:,} ({time.time()-t0:.0f}s), "
          f"E[f_j]: TT {est:.5f} vs MC {truth:.5f} +- {se:.1e}  -> rel err {abs(est-truth)/abs(truth):.2e};  matched-MC error at the same eval count: {np.sqrt(acc2/N-truth**2)/np.sqrt(max(calls['k'],1))/abs(truth):.2e}", flush=True)
