"""Opus R12 review, point 1(a): the kappa4 carry test (opus12_k4_carry.py) on the SECOND network (synthetic He net,
data/kaggle_out/kaggle_out_k4/k4teach, 6e7-sample teacher at layers 2,4,...,14,15; chain = errbudget_v29plain_synth.npz).

Exact Monte Carlo through the true synthetic net (Gaussian input, float32 forward, float64 accumulation):
  T    exact kappa4 diagonal of z_L
  LIN  kappa4 of the linear response y = (Phi(a_{L-1}) o (z_{L-1} - m_{L-1})) @ W[L-1]  (the pre-activation kappa4
       tensor of layer L-1 carried linearly; no fresh ReLU physics). m, var of z_{L-1} from an independent pilot batch.
Compared with the independent teacher K4d on the decisive neurons (|mu/sigma| < 1.5 at L). Split-group reliability
(10 groups) gives the MC-noise ceiling of each correlation; 'corr_nc' = corr / sqrt(reliability of the MC estimate).

Usage:  python experiments/probes/opus12_k4_carry_synth.py [N=1600000]
"""
import numpy as np, os, sys, json, time
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4/k4teach"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float32)
D = np.load("data/dumps/errbudget_v29plain_synth.npz")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_600_000
TARGETS = (8, 12, 14, 15)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
n = 1024; B = 20000; nb = N // B; G = 10; gs = nb // G
rng = np.random.default_rng(4242)
# pilot: linearisation points at L-1
x = rng.standard_normal((40000, n), dtype=np.float32); h = x; pm, pv = {}, {}
for li in range(15):
    z = h @ W[li]; L = li + 1
    if L + 1 in TARGETS: pm[L] = z.mean(0, dtype=np.float64); pv[L] = z.var(0, dtype=np.float64)
    h = np.maximum(z, 0)
tch = {L: np.load(os.path.join(PRE, f"layer{L}_seed0.npz")) for L in TARGETS}
tmu = {L: np.asarray(tch[L]["mean"], np.float64) for L in TARGETS}
acc = {(L, k, g): np.zeros((4, n)) for L in TARGETS for k in ("T", "LIN") for g in range(G)}
def mom(y):
    y2 = y * y
    return np.stack([y.sum(0, dtype=np.float64), y2.sum(0, dtype=np.float64), (y2 * y).sum(0, dtype=np.float64), (y2 * y2).sum(0, dtype=np.float64)])
t0 = time.time()
for b in range(gs * G):
    g = b // gs
    h = rng.standard_normal((B, n), dtype=np.float32); zprev = None
    for li in range(15):
        L = li + 1
        if L in TARGETS:
            m = pm[L - 1]; s = np.sqrt(pv[L - 1]); d1 = Phi(m / s).astype(np.float32)
            acc[(L, "LIN", g)] += mom(((zprev - m.astype(np.float32)) * d1) @ W[li])
        z = h @ W[li]
        if L in TARGETS: acc[(L, "T", g)] += mom(z - tmu[L].astype(np.float32))
        zprev = z; h = np.maximum(z, 0)
    if b % 20 == 19: print(f"  batch {b+1}/{gs*G} ({time.time()-t0:.0f}s)", flush=True)
def cum(M, cnt):
    m1, m2, m3, m4 = M / cnt; c2 = m2 - m1 ** 2
    return c2, m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4 - 3 * c2 ** 2
res = {"N": gs * G * B}
for L in TARGETS:
    v = np.diag(np.asarray(tch[L]["C"], np.float64)).copy(); mu = tmu[L]; s = np.sqrt(v); a = mu / s; dec = np.abs(a) < 1.5
    w = (a * a - 1) * pdf(a) / s ** 3 / 24; tk4 = np.asarray(tch[L]["K4d"], np.float64); ck4 = D[f"L{L}_ck4"]
    ex = lambda k: (k / v ** 2)[dec]
    r = {"n_dec": int(dec.sum())}
    for k in ("T", "LIN"):
        full = cum(sum(acc[(L, k, g)] for g in range(G)), gs * G * B)[1]
        # split-half reliability: groups 0-4 vs 5-9, Spearman-Brown to full N
        h1 = cum(sum(acc[(L, k, g)] for g in range(G // 2)), gs * (G // 2) * B)[1]
        h2 = cum(sum(acc[(L, k, g)] for g in range(G // 2, G)), gs * (G // 2) * B)[1]
        rhh = float(np.corrcoef(ex(h1), ex(h2))[0, 1]); rel = 2 * rhh / (1 + rhh)
        c = float(np.corrcoef(ex(full), ex(tk4))[0, 1])
        r[k] = {"level_vs_teacher": float(ex(full).mean() / ex(tk4).mean()), "dec_corr_teacher": c, "reliability": rel,
                "dec_corr_teacher_nc": c / np.sqrt(max(rel, 1e-9)),
                "errG4_teacher": float(np.linalg.norm(w * (full - tk4)) / np.linalg.norm(w * tk4))}
        if k == "LIN":
            A_ = np.stack([w[dec] * full[dec], w[dec] * v[dec] ** 2], 1); cf = np.linalg.lstsq(A_, w[dec] * tk4[dec], rcond=None)[0]
            r[k]["affine_errG4_teacher"] = float(np.linalg.norm(w * (cf[0] * full + cf[1] * v ** 2 - tk4)) / np.linalg.norm(w * tk4))
    r["chain"] = {"level_vs_teacher": float(ex(ck4).mean() / ex(tk4).mean()), "dec_corr_teacher": float(np.corrcoef(ex(ck4), ex(tk4))[0, 1]),
                  "errG4_teacher": float(np.linalg.norm(w * (ck4 - tk4)) / np.linalg.norm(w * tk4))}
    res[f"L{L}"] = r
    print(f"synth L{L} (dec {r['n_dec']}): decisive corr with teacher  T {r['T']['dec_corr_teacher']:.3f} (rel {r['T']['reliability']:.3f})  "
          f"LIN {r['LIN']['dec_corr_teacher']:.3f} (rel {r['LIN']['reliability']:.3f}, noise-corr {r['LIN']['dec_corr_teacher_nc']:.3f})  chain {r['chain']['dec_corr_teacher']:.3f} | "
          f"level LIN {r['LIN']['level_vs_teacher']:.3f} T {r['T']['level_vs_teacher']:.3f} chain {r['chain']['level_vs_teacher']:.3f} | "
          f"errG4 chain {r['chain']['errG4_teacher']:.3f} T {r['T']['errG4_teacher']:.3f} LIN(affine) {r['LIN']['affine_errG4_teacher']:.3f}", flush=True)
json.dump(res, open("results/summaries/opus12_k4_carry_synth.json", "w"), indent=1)
