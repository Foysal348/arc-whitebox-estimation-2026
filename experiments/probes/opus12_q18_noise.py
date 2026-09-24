"""Opus R12 review, point 4: audit of experiments/probes/q18_k4_horizon.py (the kappa4 memory horizon).

Same method as Q18 (z_{L-h} ~ N(teacher mean, teacher C), true net for h steps, kappa4 diagonal of z_L, MLP #0), same
N = 1e6 and decisive set (|mu/sigma| < 1.5, unweighted level and correlation), but with the sample split into two
independent halves, so that the MC noise of each reported number is MEASURED:
  - level (decisive mean excess kurtosis / truth) with its half-to-half spread,
  - split-half reliability of the per-neuron pattern (Spearman-Brown to full N) and the noise-corrected correlation
    corr / sqrt(reliability) with the independent 6e7 teacher,
  - per-neuron noise std vs signal std on the decisive set.
Also an index check of the forward loop: h = 0 (Gaussian z_L itself) must give level ~0.

Usage:  python experiments/probes/opus12_q18_noise.py [N=1000000]
"""
import numpy as np, os, sys, json, time
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float32)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
B = 20000; nb = N // B; n = 1024
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
tch = lambda L: np.load(os.path.join(PRE, f"layer{L}_seed0.npz"))
def cum(M, cnt):
    m1, m2, m3, m4 = M / cnt; c2 = m2 - m1 ** 2
    return m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4 - 3 * c2 ** 2
res = {}
for L, hs in ((16, (0, 2, 6)), (12, (2, 6))):
    t = tch(L); mu = np.asarray(t["mean"], np.float64); v = np.diag(np.asarray(t["C"], np.float64)).copy()
    tk4 = np.asarray(t["K4d"], np.float64); a = mu / np.sqrt(v); dec = np.abs(a) < 1.5
    ex = lambda k: (k / v ** 2)[dec]
    for h in hs:
        l0 = L - h; t0 = tch(l0); m0 = np.asarray(t0["mean"], np.float64); C0 = np.asarray(t0["C"], np.float64)
        lam, V = np.linalg.eigh(0.5 * (C0 + C0.T)); R = (V * np.sqrt(np.maximum(lam, 0))).astype(np.float32)
        rng = np.random.default_rng(900 + h + L); acc = [np.zeros((4, n)), np.zeros((4, n))]; tt = time.time()
        for b in range(nb):
            z = rng.standard_normal((B, n), dtype=np.float32) @ R.T + m0.astype(np.float32)
            for li in range(l0, L):
                z = np.maximum(z, 0) @ W[li]
            y = z - mu.astype(np.float32); y2 = y * y
            acc[b % 2] += np.stack([y.sum(0, dtype=np.float64), y2.sum(0, dtype=np.float64), (y2 * y).sum(0, dtype=np.float64), (y2 * y2).sum(0, dtype=np.float64)])
        half = nb // 2 * B
        k1, k2 = cum(acc[0], half), cum(acc[1], half); kf = cum(acc[0] + acc[1], 2 * half)
        e1, e2, ef, et = ex(k1), ex(k2), ex(kf), ex(tk4)
        rhh = float(np.corrcoef(e1, e2)[0, 1]); rel = 2 * rhh / (1 + rhh) if rhh > -1 else float("nan")
        c = float(np.corrcoef(ef, et)[0, 1])
        noise_sd = float(np.std(e1 - e2) / 2)                      # std of the full-N estimate's noise per neuron
        sig_sd = float(np.sqrt(max(np.var(ef) - noise_sd ** 2, 0.0)))
        r = {"level": float(ef.mean() / et.mean()), "level_half_spread": float(abs(e1.mean() - e2.mean()) / et.mean() / 2),
             "corr_teacher": c, "split_half_r": rhh, "reliability": rel,
             "corr_teacher_noise_corrected": (c / np.sqrt(rel)) if rel > 0 else None,
             "noise_sd_exkurt": noise_sd, "signal_sd_exkurt": sig_sd, "truth_sd_exkurt": float(np.std(et)),
             "truth_mean_exkurt": float(et.mean()), "sec": time.time() - tt}
        res[f"L{L}_h{h}"] = r
        print(f"L{L} h={h}: level {r['level']:.3f} (+-{r['level_half_spread']:.3f}) | corr {c:.3f}, split-half r {rhh:.3f}, reliability {rel:.3f}, "
              f"noise-corrected corr {r['corr_teacher_noise_corrected'] if r['corr_teacher_noise_corrected'] is None else round(r['corr_teacher_noise_corrected'], 3)} | "
              f"per-neuron exkurt: noise sd {noise_sd:.4f}, signal sd {sig_sd:.4f}, truth sd {r['truth_sd_exkurt']:.4f} (truth mean {r['truth_mean_exkurt']:.4f})  ({r['sec']:.0f}s)", flush=True)
json.dump(res, open("results/summaries/opus12_q18_noise.json", "w"), indent=1)
