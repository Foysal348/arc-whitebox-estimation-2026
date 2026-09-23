"""Moonshot probe (23 Sep): the kappa4 diagonal as a SUM OVER STAR SOURCES that share the kappa3 legs.

Motivation. The corrected oracle: the exact pre-activation kappa4 diagonal is worth -30% final MSE; the chain's memoryless
regeneration has corr ~0.8 with it, and no combination of the chain's cheap quantities predicts the rest (probe_k4feat).

Theory (Mehler expansion of relu of a Gaussian vector g ~ N(mu, C), leading order in the off-diagonal covariance). With
x = h - m, Phi_r = P(g_r > 0), phi_r = pdf(mu_r / s_r), the fourth cumulant born at a ReLU layer has star terms

    k(h_r, h_r, h_k, h_m) = beta_r  a_kr a_mr     beta_r  = 2 Phi_r (1 - Phi_r) - 2 m_r phi_r / s_r
    k(h_r, h_r, h_r, h_k) = gamma_r a_kr          gamma_r = 3 (1 - Phi_r) (E[h_r^2] - 2 m_r^2)
    k(h_r, h_r, h_r, h_r) = delta_r               (the marginal kappa4 of relu)
with a = diag(Phi) C_off - EXACTLY the leg of the kappa3 source born at the same layer (a_b = w1 C_off). Transported
linearly to a later pre-activation by T (z = T h), the diagonal of the fourth cumulant is

    k4(z_i) = sum_b sum_r [ 6 beta_r T_ir^2 At_ir^2 + 4 gamma_r T_ir^3 At_ir + delta_r T_ir^4 ],   At = T a

i.e. elementwise functions of the SAME two legs (T, At) the chain forms for its kappa3 slices: O(n^2) per source-layer.
This probe computes that sum with exact dense transport T = W^T D ... W^T (D = diag Phi) from the chain's own
mu, var, C_off, and compares it with the teacher's kappa4 diagonal (overall and G4^2-weighted), alone and combined with
the chain's regenerated diagonal.

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_k4star.py
"""
import os, sys, types, json, time, glob
import numpy as np
from math import erf
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
assert src.count(a1) == 1
src = src.replace(a1, "            _CAPW(li, mu, var, g4row, None if trim else C_off)\n" + a1)
S = {}
def _CAPW(li, mu, var, g4row, C_off):
    S[li] = dict(mu=np.asarray(mu, np.float64).copy(), var=np.asarray(var, np.float64).copy(),
                 g4=None if g4row is None else np.asarray(g4row, np.float64).copy(),
                 C_off=None if C_off is None else np.asarray(C_off, np.float32).copy())
mod = types.ModuleType("est_k4s"); mod.__dict__["_CAPW"] = _CAPW
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)

NETS = {"mlp0": "../data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "../data/kaggle_out/kaggle_out_k4/k4teach"}
if os.environ.get("NETS"): NETS = {k: v for k, v in NETS.items() if k in os.environ["NETS"].split(",")}
res = {}
for nk, DIR in NETS.items():
    S.clear()
    Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"].astype(np.float64)
    mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l].astype(np.float32)) for l in range(Wall.shape[0])], seed=0, name=nk)
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        est.predict(mlp, 2**41)
    print(f"{nk}: chain run {time.time()-t0:.0f}s", flush=True)
    n = Wall.shape[1]; L = Wall.shape[0]
    # per chain index li: Gaussian-closure birth coefficients of the post-ReLU h_(li+1)
    B = {}
    for li in range(L - 1):
        mu, var = S[li]["mu"], S[li]["var"]; s = np.sqrt(var); a_ = mu / s; P = Phi(a_); ph = pdf(a_)
        m = s * ph + mu * P; Eh2 = (mu * mu + var) * P + mu * s * ph
        beta = 2 * P * (1 - P) - 2 * m * ph / s
        gamma = 3 * (1 - P) * (Eh2 - 2 * m * m)
        # marginal kappa4 of relu(N(mu, s^2)) from its raw moments
        Eh3 = (mu ** 3 + 3 * mu * var) * P + s * (mu * mu + 2 * var) * ph
        Eh4 = (mu ** 4 + 6 * mu * mu * var + 3 * var * var) * P + s * mu * (mu * mu + 5 * var) * ph
        m2c = Eh2 - m * m; m4c = Eh4 - 4 * m * Eh3 + 6 * m * m * Eh2 - 3 * m ** 4
        delta = m4c - 3 * m2c * m2c
        Coff = S[li]["C_off"]
        if Coff is None:                                  # layer 1: C_pre = W^T W exactly (input covariance I)
            W0 = Wall[0]; Coff = (W0.T @ W0); np.fill_diagonal(Coff, 0.0)
        Coff = Coff.astype(np.float64)
        B[li] = dict(beta=beta, gamma=gamma, delta=delta, Phi=P, a=P[:, None] * Coff)
    star = {lt: np.zeros(n) for lt in range(1, L)}
    parts = {lt: np.zeros((3, n)) for lt in range(1, L)}
    for lb in range(L - 1):                               # birth at the post-ReLU of chain index lb
        T = Wall[lb + 1].T.copy()                         # z_(lb+1) = h_lb @ W[lb+1]  ->  T = W^T
        At = T @ B[lb]["a"]
        for lt in range(lb + 1, L):
            if lt > lb + 1:
                D = B[lt - 1]["Phi"]
                T = Wall[lt].T @ (D[:, None] * T); At = Wall[lt].T @ (D[:, None] * At)
            b = B[lb]
            p1 = 6 * ((T * T) * (At * At)) @ b["beta"]; p2 = 4 * ((T ** 3) * At) @ b["gamma"]; p3 = (T ** 4) @ b["delta"]
            parts[lt] += np.stack([p1, p2, p3]); star[lt] += p1 + p2 + p3
    out = []
    for lt in range(1, L):
        f = os.path.join(DIR, f"layer{lt + 1}_seed0.npz")
        if not os.path.exists(f) or S[lt]["g4"] is None: continue
        t4 = np.asarray(np.load(f)["K4d"], np.float64); c4 = S[lt]["g4"]
        mu, var = S[lt]["mu"], S[lt]["var"]; s = np.sqrt(var); al = mu / s; w = ((al * al - 1) * pdf(al) / s ** 3) ** 2
        def ev(v):
            e = v - t4
            return dict(corr=float(np.corrcoef(v, t4)[0, 1]), slope=float(np.dot(v, t4) / np.dot(v, v)),
                        errG4=float(np.sqrt(np.sum(w * e * e) / np.sum(w * t4 * t4))))
        X = np.stack([c4, star[lt]], 1); sw = np.sqrt(w); cb, *_ = np.linalg.lstsq(X * sw[:, None], t4 * sw, rcond=None)
        rec = dict(layer=lt + 1, chain=ev(c4), star=ev(star[lt]), blend=ev(X @ cb), blend_coef=list(map(float, cb)),
                   part_rms=[float(np.sqrt(np.mean(p ** 2))) for p in parts[lt]], t4_rms=float(np.sqrt(np.mean(t4 ** 2))))
        out.append(rec)
        print(f"{nk} L{lt + 1:2d} | chain corr {rec['chain']['corr']:.3f} errG4 {rec['chain']['errG4']:.3f} | star corr {rec['star']['corr']:.3f} "
              f"slope {rec['star']['slope']:.2f} errG4 {rec['star']['errG4']:.3f} | blend errG4 {rec['blend']['errG4']:.3f} coef {cb[0]:.2f},{cb[1]:.2f} "
              f"| parts rms {rec['part_rms'][0]:.1e} {rec['part_rms'][1]:.1e} {rec['part_rms'][2]:.1e} vs t4 {rec['t4_rms']:.1e}", flush=True)
    res[nk] = out
json.dump(res, open("../results/summaries/k4star_probe.json", "w"), indent=1)
