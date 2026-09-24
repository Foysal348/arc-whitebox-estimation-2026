"""Opus R12, kappa4 thread, falsifier 2: WHAT carries the deep kappa4 pattern?

Exact Monte Carlo through the true MLP #0 (N samples, Gaussian input AND the shipped chain's spherical input
x_hat sqrt(n); float32 forward, float64 accumulation). At a target layer L = l+1, with exact samples of z_l:
  T    exact kappa4 diagonal of z_L                          (sanity: vs the 6e7-sample teacher for the Gaussian input)
  LIN  kappa4 of the LINEAR response y = (Phi(a_l) o (z_l - mu_l)) @ W_l     = every sector of kappa4(z_l), carried
       linearly (no fresh ReLU physics at all)
  QUAD kappa4 of y2 = [Phi o dz + 0.5 phi/s o (dz^2 - var)] @ W_l            (adds the leading ReLU curvature:
       kappa3 x C births of kappa4 and the Gaussian star/path trees at second order)
Patterns are compared on the decisive neurons (|mu/sigma| < 1.5 at L) and with the G4 weight of the mean closure.
If LIN reproduces the decisive pattern, the deep kappa4 is linear memory of the full pre-activation tensor; the
gap T - LIN is what the ReLU adds in one step.

Also tested (zero-cost structured predictors of LIN, from the true l-layer mean/covariance only):
  SMM  3 eps (u^T M u)^2 with M = C + mu mu^T   (scale-mixture / radial form; eps fitted by one scalar)
  SCC  3 (u^T C u)^2                              (Sym(C x C) with covariance only; one scalar)

Usage:  python experiments/probes/opus12_k4_carry.py [N=2000000]
"""
import numpy as np, os, sys, json, time
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float32)
D = np.load("data/dumps/errbudget_v29plain.npz")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
TARGETS = (4, 8, 12, 15, 16)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
n = 1024; B = 20000; nb = N // B
tch = {L: np.load(os.path.join(PRE, f"layer{L}_seed0.npz")) for L in range(1, 17)}
tmu = {L: np.asarray(tch[L]["mean"], np.float64) for L in tch}
tvar = {L: np.diag(np.asarray(tch[L]["C"], np.float64)).copy() for L in tch}
res = {}
INPUTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["gauss", "sphere"]
for inp in INPUTS:
    rng = np.random.default_rng(777)
    acc = {(L, k): np.zeros((4, n)) for L in TARGETS for k in ("T", "LIN", "QUAD")}
    t0 = time.time()
    for b in range(nb):
        x = rng.standard_normal((B, n), dtype=np.float32)
        if inp == "sphere":
            x *= (np.sqrt(n) / np.linalg.norm(x, axis=1))[:, None].astype(np.float32)
        h = x
        for li in range(16):
            L = li + 1
            if L in TARGETS and L >= 2:
                # z_{L-1} is the pre-activation of the previous layer; h holds relu(z_{L-1})
                zp = zprev; m = tmu[L - 1].astype(np.float32); v = tvar[L - 1]
                s = np.sqrt(v); a = tmu[L - 1] / s
                d1 = Phi(a).astype(np.float32); d2 = (pdf(a) / s).astype(np.float32)
                dz = zp - m
                ylin = (dz * d1) @ W[li]
                yq = ylin + (0.5 * (dz * dz - v.astype(np.float32)) * d2) @ W[li]
                for k, y in (("LIN", ylin), ("QUAD", yq)):
                    y2 = y * y
                    acc[(L, k)] += np.stack([y.sum(0, dtype=np.float64), y2.sum(0, dtype=np.float64),
                                             (y2 * y).sum(0, dtype=np.float64), (y2 * y2).sum(0, dtype=np.float64)])
            z = h @ W[li]
            if L in TARGETS:
                y = z - tmu[L].astype(np.float32); y2 = y * y
                acc[(L, "T")] += np.stack([y.sum(0, dtype=np.float64), y2.sum(0, dtype=np.float64),
                                           (y2 * y).sum(0, dtype=np.float64), (y2 * y2).sum(0, dtype=np.float64)])
            zprev = z
            h = np.maximum(z, 0)
        if b % 20 == 19: print(f"  {inp}: batch {b+1}/{nb} ({time.time()-t0:.0f}s)", flush=True)
    def cum(M):
        m1, m2, m3, m4 = M / (nb * B)
        c2 = m2 - m1 ** 2; c4 = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4 - 3 * c2 ** 2
        return c2, c4
    for L in TARGETS:
        cT2, cT4 = cum(acc[(L, "T")]); _, cL4 = cum(acc[(L, "LIN")]); _, cQ4 = cum(acc[(L, "QUAD")])
        v = tvar[L]; mu = tmu[L]; s = np.sqrt(v); a = mu / s; dec = np.abs(a) < 1.5
        w = (a * a - 1) * pdf(a) / s ** 3 / 24
        ex = lambda k4: k4 / v ** 2
        r = {"var_rel_vs_teacher": float(np.linalg.norm(cT2 - v) / np.linalg.norm(v))}
        tk4 = np.asarray(tch[L]["K4d"], np.float64)
        r["T_vs_teacher_errG4"] = float(np.linalg.norm(w * (cT4 - tk4)) / np.linalg.norm(w * tk4)) if inp == "gauss" else None
        r["dec_mean_exkurt"] = {k: float(np.mean(ex(x)[dec])) for k, x in (("T", cT4), ("LIN", cL4), ("QUAD", cQ4))}
        for k, x in (("LIN", cL4), ("QUAD", cQ4)):
            e1, e2 = ex(x)[dec], ex(cT4)[dec]
            r[f"{k}_dec_corr"] = float(np.corrcoef(e1, e2)[0, 1])
            r[f"{k}_errG4"] = float(np.linalg.norm(w * (x - cT4)) / np.linalg.norm(w * cT4))
        if inp == "gauss":
            ck4 = D[f"L{L}_ck4"]
            r["chain_dec_corr_vs_T"] = float(np.corrcoef(ex(ck4)[dec], ex(cT4)[dec])[0, 1])
            r["teacher_dec_corr_vs_T"] = float(np.corrcoef(ex(tk4)[dec], ex(cT4)[dec])[0, 1])
            r["chain_errG4_vs_T"] = float(np.linalg.norm(w * (ck4 - cT4)) / np.linalg.norm(w * cT4))
            # independent-sample comparisons against the 6e7 teacher (no shared MC noise)
            for k, x in (("T", cT4), ("LIN", cL4), ("QUAD", cQ4), ("chain", ck4)):
                r[f"{k}_vs_teacher_dec_corr"] = float(np.corrcoef(ex(x)[dec], ex(tk4)[dec])[0, 1])
            for k, x in (("LIN", cL4), ("QUAD", cQ4)):
                # best affine map of x onto the teacher on decisive neurons (level + scale), then G4 error
                A_ = np.stack([w[dec] * x[dec], w[dec] * v[dec] ** 2], 1); cf = np.linalg.lstsq(A_, w[dec] * tk4[dec], rcond=None)[0]
                r[f"{k}_affine_coef"] = cf.tolist()
                r[f"{k}_affine_errG4_teacher"] = float(np.linalg.norm(w * (cf[0] * x + cf[1] * v ** 2 - tk4)) / np.linalg.norm(w * tk4))
            r["chain_errG4_teacher"] = float(np.linalg.norm(w * (ck4 - tk4)) / np.linalg.norm(w * tk4))
            print(f"   vs teacher (independent): dec-corr T {r['T_vs_teacher_dec_corr']:.3f} LIN {r['LIN_vs_teacher_dec_corr']:.3f} QUAD {r['QUAD_vs_teacher_dec_corr']:.3f} chain {r['chain_vs_teacher_dec_corr']:.3f} | "
                  f"errG4: chain {r['chain_errG4_teacher']:.3f} LIN(affine) {r['LIN_affine_errG4_teacher']:.3f} QUAD(affine) {r['QUAD_affine_errG4_teacher']:.3f}", flush=True)
        # structured zero-cost predictors of LIN, from layer L-1 mean / covariance (teacher)
        Cp = np.asarray(tch[L - 1]["C"], np.float64); mp = tmu[L - 1]; sp = np.sqrt(np.diag(Cp)); d1 = Phi(mp / sp)
        U = d1[:, None] * W[L - 1].astype(np.float64)
        uCu = np.sum(U * (Cp @ U), 0); uMu = uCu + (U.T @ mp) ** 2
        for k, f in (("SMM", 3 * uMu ** 2), ("SCC", 3 * uCu ** 2)):
            sc = float(np.sum(f[dec] * cL4[dec]) / np.sum(f[dec] ** 2))
            r[f"{k}_scale"] = sc
            r[f"{k}_vs_LIN_dec_corr"] = float(np.corrcoef(ex(f)[dec], ex(cL4)[dec])[0, 1])
            r[f"{k}_vs_T_dec_corr"] = float(np.corrcoef(ex(f)[dec], ex(cT4)[dec])[0, 1])
        res[f"{inp}_L{L}"] = r
        print(f"[{inp}] L{L}: exkurt(dec mean) T {r['dec_mean_exkurt']['T']:.4f} LIN {r['dec_mean_exkurt']['LIN']:.4f} QUAD {r['dec_mean_exkurt']['QUAD']:.4f} | "
              f"dec-corr with T: LIN {r['LIN_dec_corr']:.3f} QUAD {r['QUAD_dec_corr']:.3f} "
              + (f"chain {r['chain_dec_corr_vs_T']:.3f} teacher {r['teacher_dec_corr_vs_T']:.3f} | errG4 vs T: chain {r['chain_errG4_vs_T']:.3f} T-vs-teacher {r['T_vs_teacher_errG4']:.3f} " if inp == "gauss" else "")
              + f"| errG4 LIN {r['LIN_errG4']:.3f} QUAD {r['QUAD_errG4']:.3f} | SMM->LIN corr {r['SMM_vs_LIN_dec_corr']:.3f} SCC->LIN {r['SCC_vs_LIN_dec_corr']:.3f}", flush=True)
    json.dump(res, open("results/summaries/opus12_k4_carry.json", "w"), indent=1)
