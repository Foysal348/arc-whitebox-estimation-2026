"""Opus R12 review, point 4 follow-up: does the YOUNG kappa4 content (born in the last h ReLUs) add pattern to the chain?

Q18 / opus12_q18_noise: the Gaussian-reset object K_h (z_{L-h} ~ N(teacher mean, C), true net for h steps) holds only
2-5% of the deep kappa4 LEVEL at h = 2, but its per-neuron pattern correlates 0.52-0.58 (noise-corrected) with the truth,
above the chain's 0.32-0.34. The level is what the chain already gets right. So the carrier question is: does
chain + a K_h + b var^2 beat the chain OUT OF SAMPLE? (a, b fitted G4-weighted on one layer, applied to the other.)
This is the ideal young carrier (exact births of the last h layers, exact transport); a deployable one can only be worse.

Usage:  python experiments/probes/opus12_k4_young_blend.py [N=1000000]
"""
import numpy as np, os, sys, json, time
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float32)
D = np.load("data/dumps/errbudget_v29plain.npz")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
B = 20000; nb = N // B; n = 1024
pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
tch = lambda L: np.load(os.path.join(PRE, f"layer{L}_seed0.npz"))
def cum(M, cnt):
    m1, m2, m3, m4 = M / cnt; c2 = m2 - m1 ** 2
    return m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4 - 3 * c2 ** 2
K = {}; ST = {}
for L in (12, 16):
    t = tch(L); mu = np.asarray(t["mean"], np.float64); v = np.diag(np.asarray(t["C"], np.float64)).copy()
    a = mu / np.sqrt(v); ST[L] = dict(v=v, tk4=np.asarray(t["K4d"], np.float64), ck4=D[f"L{L}_ck4"], dec=np.abs(a) < 1.5,
                                    w=(a * a - 1) * pdf(a) / np.sqrt(v) ** 3 / 24)
    for h in (1, 2, 3):
        l0 = L - h; t0 = tch(l0); m0 = np.asarray(t0["mean"], np.float64); C0 = np.asarray(t0["C"], np.float64)
        lam, V = np.linalg.eigh(0.5 * (C0 + C0.T)); R = (V * np.sqrt(np.maximum(lam, 0))).astype(np.float32)
        rng = np.random.default_rng(5000 + 10 * L + h); acc = np.zeros((4, n)); tt = time.time()
        for b in range(nb):
            z = rng.standard_normal((B, n), dtype=np.float32) @ R.T + m0.astype(np.float32)
            for li in range(l0, L):
                z = np.maximum(z, 0) @ W[li]
            y = z - mu.astype(np.float32); y2 = y * y
            acc += np.stack([y.sum(0, dtype=np.float64), y2.sum(0, dtype=np.float64), (y2 * y).sum(0, dtype=np.float64), (y2 * y2).sum(0, dtype=np.float64)])
        K[(L, h)] = cum(acc, nb * B); print(f"  L{L} h={h} done ({time.time()-tt:.0f}s)", flush=True)
def ev(L, k4):
    s = ST[L]; w, tk4, v, dec = s["w"], s["tk4"], s["v"], s["dec"]
    return {"errG4": float(np.linalg.norm(w * (k4 - tk4)) / np.linalg.norm(w * tk4)),
            "dec_corr": float(np.corrcoef((k4 / v ** 2)[dec], (tk4 / v ** 2)[dec])[0, 1])}
def fit(L, h):
    s = ST[L]; w = s["w"]
    A_ = np.stack([w * K[(L, h)], w * s["v"] ** 2], 1)
    return np.linalg.lstsq(A_, w * (s["tk4"] - s["ck4"]), rcond=None)[0]
res = {}
for h in (1, 2, 3):
    for Lfit, Ltest in ((12, 16), (16, 12)):
        cf = fit(Lfit, h); s = ST[Ltest]
        new = s["ck4"] + cf[0] * K[(Ltest, h)] + cf[1] * s["v"] ** 2
        cin = fit(Ltest, h); newin = s["ck4"] + cin[0] * K[(Ltest, h)] + cin[1] * s["v"] ** 2
        r = {"coef_fit": cf.tolist(), "chain": ev(Ltest, s["ck4"]), "blend_out_of_sample": ev(Ltest, new),
             "blend_in_sample": ev(Ltest, newin), "coef_in_sample": cin.tolist(), "K_h_alone": ev(Ltest, K[(Ltest, h)])}
        res[f"h{h}_fit{Lfit}_test{Ltest}"] = r
        print(f"h={h} fit L{Lfit} -> test L{Ltest}: coef a {cf[0]:.2f} b {cf[1]:.4f} | errG4 chain {r['chain']['errG4']:.3f} -> "
              f"out-of-sample {r['blend_out_of_sample']['errG4']:.3f} (in-sample {r['blend_in_sample']['errG4']:.3f}) | dec corr chain {r['chain']['dec_corr']:.3f} -> "
              f"{r['blend_out_of_sample']['dec_corr']:.3f} (in-sample {r['blend_in_sample']['dec_corr']:.3f}); K_h alone corr {r['K_h_alone']['dec_corr']:.3f}", flush=True)
json.dump(res, open("results/summaries/opus12_k4_young_blend.json", "w"), indent=1)
