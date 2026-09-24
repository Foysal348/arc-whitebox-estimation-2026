"""Opus R12, kappa4 thread, falsifier 1: is the missing kappa4 pattern GAUSSIAN physics (all orders in C) or memory?

One step, teacher-forced: z ~ N(mu_l, C_l) with the TEACHER's pre-activation mean and covariance at layer l (MLP #0),
h = relu(z), y = h @ W[l] = next pre-activation. The kappa4 diagonal of y is estimated by Monte Carlo (N samples,
float64 accumulation, batch-split standard errors). This is what an ideal, non-perturbative GAUSSIAN closure would
give for the next kappa4 diagonal (every tree and loop order in C, exact ReLU), i.e. the ceiling of every
'fresh kernel' (Astra's quartic kernel is its leading order in C). Compared with the teacher's true kappa4 diagonal at
l+1 and with the chain's free-running kappa4 (errbudget_v29plain.npz), with the G4 weight E''''-closure factor
w = G4(mu, sigma)/24 of the NEXT layer (the weight with which kappa4 enters the mean).

If the Gaussian MC is close to the truth on the decisive neurons, the kernel's deficit is higher order in C (cheap
physics exists); if not, the missing part is non-Gaussian input (kappa3 / kappa4 memory).

Usage:  python experiments/probes/opus12_k4_gaussmc.py [N=3000000] [layers=7,11,14,15]
"""
import numpy as np, os, sys, json, time
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]
D = np.load("data/dumps/errbudget_v29plain.npz")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
LAYERS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [7, 11, 14, 15]
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
def G4w(mu, var):
    s = np.sqrt(var); a = mu / s
    return (a * a - 1) * pdf(a) / s ** 3 / 24.0
rng = np.random.default_rng(12345)
out = {}
for l in LAYERS:
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz"))
    mu = np.asarray(T["mean"], np.float64); C = np.asarray(T["C"], np.float64)
    C = 0.5 * (C + C.T); lam, V = np.linalg.eigh(C); lmin = float(lam.min())
    Lc = (V * np.sqrt(np.maximum(lam, 0.0))).astype(np.float32)   # C = Lc Lc^T (tiny negative MC eigenvalues clipped)
    Wl = W[l].astype(np.float32)
    ctr = np.asarray(Tn["mean"], np.float64)          # centre near the true next mean for stable raw moments
    B = 20000; nb = N // B; t0 = time.time()
    S = np.zeros((4, 1024)); Sb = []
    for b in range(nb):
        z = rng.standard_normal((B, 1024), dtype=np.float32) @ Lc.T + mu.astype(np.float32)
        y = (np.maximum(z, 0) @ Wl).astype(np.float64) - ctr
        p = np.stack([y.sum(0), (y ** 2).sum(0), (y ** 3).sum(0), (y ** 4).sum(0)])
        S += p; Sb.append(p)
    def k4_of(M, n):
        m1, m2, m3, m4 = M / n
        c2 = m2 - m1 ** 2; c4 = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4
        return c4 - 3 * c2 ** 2, c2, m1
    k4mc, varmc, m1 = k4_of(S, nb * B)
    # standard error from 10 groups of batches
    G = 10; gs = nb // G
    k4g = np.stack([k4_of(sum(Sb[g * gs:(g + 1) * gs]), gs * B)[0] for g in range(G)])
    se = k4g.std(0, ddof=1) / np.sqrt(G)
    tk4 = np.asarray(Tn["K4d"], np.float64); tvar = np.diag(np.asarray(Tn["C"], np.float64)); tmu = np.asarray(Tn["mean"], np.float64)
    ck4 = D[f"L{l+1}_ck4"]
    w = G4w(tmu, tvar); dec = np.abs(tmu / np.sqrt(tvar)) < 1.5
    def st(x):
        e = float(np.linalg.norm(w * (x - tk4)) / np.linalg.norm(w * tk4))
        sl = float(np.sum(w ** 2 * x * tk4) / np.sum(w ** 2 * x * x))       # least-squares scale that would map x onto the truth
        # pattern: correlation on decisive neurons of kappa4 / var^2 (excess kurtosis), unweighted
        c = float(np.corrcoef((x / tvar ** 2)[dec], (tk4 / tvar ** 2)[dec])[0, 1])
        return e, sl, c
    r = {"N": nb * B, "sec": time.time() - t0,
         "mc_noise_rel_G4": float(np.linalg.norm(w * se) / np.linalg.norm(w * tk4)),
         "var_rel_err_gauss": float(np.linalg.norm(varmc - tvar) / np.linalg.norm(tvar)),
         "gauss": st(k4mc), "chain": st(ck4), "blend_half": st(0.5 * (k4mc + ck4))}
    # best 2-way blend (in-sample) and the gain available in principle
    Am = np.stack([w * k4mc, w * ck4], 1); cf = np.linalg.lstsq(Am, w * tk4, rcond=None)[0]
    r["dec_mean_ratio_gauss"] = float(np.mean(k4mc[dec]) / np.mean(tk4[dec])); r["dec_mean_ratio_chain"] = float(np.mean(ck4[dec]) / np.mean(tk4[dec]))
    r["dec_rel_rms_gauss"] = float(np.linalg.norm((k4mc - tk4)[dec]) / np.linalg.norm(tk4[dec])); r["dec_rel_rms_chain"] = float(np.linalg.norm((ck4 - tk4)[dec]) / np.linalg.norm(tk4[dec]))
    r["dec_rel_rms_noise"] = float(np.linalg.norm(se[dec]) / np.linalg.norm(tk4[dec]))
    print(f"   decisive: mean ratio gauss {r['dec_mean_ratio_gauss']:.3f} chain {r['dec_mean_ratio_chain']:.3f}; rel rms err gauss {r['dec_rel_rms_gauss']:.3f} chain {r['dec_rel_rms_chain']:.3f} (MC noise {r['dec_rel_rms_noise']:.3f})")
    r["blend_ls_coef"] = cf.tolist(); r["blend_ls"] = st(cf[0] * k4mc + cf[1] * ck4)
    out[l + 1] = r
    print(f"next L{l+1}: N {nb*B:.1e} ({r['sec']:.0f}s)  MC noise {r['mc_noise_rel_G4']:.3f}  var err (Gauss) {r['var_rel_err_gauss']:.4f}\n"
          f"   errG4 / LS-scale / dec-corr:  gaussMC {r['gauss'][0]:.3f} {r['gauss'][1]:.2f} {r['gauss'][2]:.3f} | chain {r['chain'][0]:.3f} {r['chain'][1]:.2f} {r['chain'][2]:.3f}"
          f" | 50/50 {r['blend_half'][0]:.3f} | LS blend {cf.round(2)} -> {r['blend_ls'][0]:.3f}", flush=True)
    np.savez(os.path.join(os.environ.get('OPUS_SCRATCH', '.'), f"opus12_gaussmc_L{l+1}.npz"), k4mc=k4mc, varmc=varmc, se=se)
json.dump(out, open("results/summaries/opus12_k4_gaussmc.json", "w"), indent=1)
