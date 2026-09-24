"""Q18 (25 Sep): the kappa4 memory horizon.
At target layer L, replace the pre-activation z_{L-h} by a Gaussian with the TEACHER's mean and covariance (all
non-Gaussian history older than h steps erased), push it through the TRUE network (exact ReLUs, true weights) for h steps,
and measure the kappa4 diagonal of z_L. h = 1 is 'fresh physics only' (Opus R12: 1-4% of the level from L8); as h grows
the result approaches the truth. Compared on the decisive neurons (|mu/sigma| < 1.5) with the independent 6e7-sample
teacher K4d, alongside the shipped-chain plain V29 dump (errbudget_v29plain.npz).
Gate (queue Q18): if h = 2 holds >= half of the missing pattern, a young-source kappa4 carrier is worth building.

Usage:  python experiments/probes/q18_k4_horizon.py [N=1000000]
"""
import numpy as np, os, sys, json, time
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float32)
D = np.load("data/dumps/errbudget_v29plain.npz")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
B = 20000; nb = N // B; n = 1024
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
tch = lambda L: np.load(os.path.join(PRE, f"layer{L}_seed0.npz"))
res = {}
for L in (12, 16):
    t = tch(L); mu = np.asarray(t["mean"], np.float64); v = np.diag(np.asarray(t["C"], np.float64)).copy()
    tk4 = np.asarray(t["K4d"], np.float64); s = np.sqrt(v); a = mu / s; dec = np.abs(a) < 1.5
    wG = (a * a - 1) * pdf(a) / s ** 3 / 24
    ex = lambda k: k / v ** 2
    ck4 = D[f"L{L}_ck4"]
    def score(k4, name):
        e, et = ex(k4)[dec], ex(tk4)[dec]
        sc = float(np.sum(wG * k4 * wG * tk4) / np.sum((wG * k4) ** 2))
        r = {"level": float(e.mean() / et.mean()), "corr": float(np.corrcoef(e, et)[0, 1]),
             "errG4": float(np.linalg.norm(wG * (k4 - tk4)) / np.linalg.norm(wG * tk4)),
             "errG4_scaled": float(np.linalg.norm(wG * (sc * k4 - tk4)) / np.linalg.norm(wG * tk4)), "scale": sc}
        print(f"  L{L} {name:10s} level {r['level']:.3f} corr {r['corr']:.3f} errG4 {r['errG4']:.3f} (best scale x{sc:.2f}: {r['errG4_scaled']:.3f})", flush=True)
        return r
    res[f"L{L}"] = {"chain": score(ck4, "chain")}
    for h in (1, 2, 3, 4, 6):
        l0 = L - h; t0 = tch(l0); m0 = np.asarray(t0["mean"], np.float64); C0 = np.asarray(t0["C"], np.float64)
        lam, V = np.linalg.eigh(C0); R = (V * np.sqrt(np.maximum(lam, 0))).astype(np.float32)
        rng = np.random.default_rng(100 + h); acc = np.zeros((4, n)); tt = time.time()
        for b in range(nb):
            z = rng.standard_normal((B, n), dtype=np.float32) @ R.T + m0.astype(np.float32)
            for li in range(l0, L):
                z = np.maximum(z, 0) @ W[li]
            y = (z - mu.astype(np.float32)); y2 = y * y
            acc += np.stack([y.sum(0, dtype=np.float64), y2.sum(0, dtype=np.float64), (y2 * y).sum(0, dtype=np.float64), (y2 * y2).sum(0, dtype=np.float64)])
        m1, m2, m3, m4 = acc / (nb * B); c2 = m2 - m1 ** 2
        k4 = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4 - 3 * c2 ** 2
        r = score(k4, f"gauss h={h}"); r["var_rel"] = float(np.linalg.norm(c2 - v) / np.linalg.norm(v)); r["sec"] = time.time() - tt
        res[f"L{L}"][f"h{h}"] = r
    json.dump(res, open("results/summaries/q18_k4_horizon.json", "w"), indent=1)
