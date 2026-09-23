"""Round 7 (Astra) falsifier 1 (23 Sep): the covariance terms absent from V29's (1,1) closure.

Astra's code audit: the (1,1) table keeps the Gaussian C and C^2 terms and a standalone kappa31 term, but omits the pure
Gaussian C^3 / C^4 terms of the Mehler series sum_r C^r d_r,i d_r,j / r! and the first C x kappa31 dressing
(kappa_iiij / 6) C_ij d_4,i d_2,j (+ transpose), with d_r = E[relu^(r)(z)] (d1 = Phi, d2 = phi/s, d3 = -a phi/s^2,
d4 = (a^2 - 1) phi/s^3). Test, without fitting: add dC = C^3 d3 d3 / 6 + C^4 d4 d4 / 24 + C (B + B^T) / 6,
B = d4 K31 d2, to the chain's post-ReLU covariance and measure the error against the post-ReLU teacher, entry-wise and
projected onto the next layer's variances (weighted by the mean-closure sensitivity phi(a')/(2 s')). K31 is taken both
from the chain's own model (lam C_off) and from the teacher.

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_astra7_cov.py
"""
import os, sys, types, json, time, glob
import numpy as np
from math import erf
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"
SRC = "../submissions/v31ang/estimator.py"
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a3 = "            C = flops.as_symmetric(C, symmetry=(0, 1))\n"
for a in (a1, a3): assert src.count(a) == 1
src = src.replace(a1, "            _CAPW(li, mu, var, None if trim else C_off, lam_prev)\n" + a1)
src = src.replace(a3, a3 + "            _CAPC(li, C)\n")
S = {}
POSTL = (3, 7, 11, 14, 15)
def _CAPW(li, mu, var, C_off, lam):
    d = S.setdefault(li, {}); d.update(mu=np.asarray(mu, np.float64).copy(), var=np.asarray(var, np.float64).copy(), lam=float(lam))
    if li + 1 in POSTL and C_off is not None: d["C_off"] = np.asarray(C_off, np.float64).copy()
def _CAPC(li, C):
    if li + 1 in POSTL: S.setdefault(li, {})["C"] = np.asarray(C, np.float64).copy()
mod = types.ModuleType("est_a7"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPC": _CAPC})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
print(f"chain run {time.time()-t0:.0f}s", flush=True)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)

def missing_cov_terms(C, d2, d3, d4, K31):          # Astra round 7, appendix C (verbatim)
    C = C.copy(); np.fill_diagonal(C, 0)
    C2 = C * C
    out = (C2 * C) * (d3[:, None] * d3[None, :]) / 6
    out += (C2 * C2) * (d4[:, None] * d4[None, :]) / 24
    B = d4[:, None] * K31 * d2[None, :]
    out += C * (B + B.T) / 6
    return out
off = ~np.eye(1024, dtype=bool)
out = []
for l in POSTL:
    li = l - 1; c = S[li]
    mu, var = c["mu"], c["var"]; s = np.sqrt(var); a = mu / s; ph = pdf(a)
    d2, d3, d4 = ph / s, -a * ph / s ** 2, (a * a - 1) * ph / s ** 3
    Coff = c["C_off"]
    K31c = c["lam"] * Coff                                                      # the chain's (3,1) model
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); K31t = np.asarray(T["K31"], np.float64).copy(); np.fill_diagonal(K31t, 0)
    P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz")); Ct = np.asarray(P["C"], np.float64)
    E0 = c["C"] - Ct; np.fill_diagonal(E0, 0.0)
    v = np.diag(Ct); noise = np.sqrt((np.outer(v, v) + Ct * Ct)[off] / 6e7)
    # next-layer projection and sensitivity
    W = Wall[li + 1].astype(np.float64); n_ = S[li + 1]; s1 = np.sqrt(n_["var"]); a1 = n_["mu"] / s1; wsens = (pdf(a1) / (2 * s1)) ** 2
    proj = lambda E: np.sum(W * (E @ W), axis=0)
    def score(E):
        pv = proj(E); return float(np.sqrt(np.mean(E[off] ** 2))), float(np.sum(wsens * pv * pv))
    r0, p0 = score(E0)
    rec = {"post_layer": l, "E0_rms": r0, "noise_rms": float(np.sqrt(np.mean(noise ** 2))), "proj0": p0}
    for tag, K31 in (("chainK31", K31c), ("trueK31", K31t)):
        dC = missing_cov_terms(Coff, d2, d3, d4, K31)
        parts = {"C3": (Coff ** 3) * np.outer(d3, d3) / 6, "C4": (Coff ** 4) * np.outer(d4, d4) / 24}
        B = d4[:, None] * K31 * d2[None, :]; parts["CK31"] = Coff * (B + B.T) / 6
        for p_ in parts.values(): np.fill_diagonal(p_, 0.0)
        r1, p1 = score(E0 + dC)
        rec[tag] = {"dC_rms": float(np.sqrt(np.mean(dC[off] ** 2))), "corr_dC_minusE0": float(np.corrcoef(dC[off], -E0[off])[0, 1]),
                    "E1_rms": r1, "proj1": p1, "proj_ratio": p1 / p0,
                    "parts_rms": {k: float(np.sqrt(np.mean(v_[off] ** 2))) for k, v_ in parts.items()},
                    "parts_proj_ratio": {k: score(E0 + v_)[1] / p0 for k, v_ in parts.items()}}
    out.append(rec)
    ch, tr = rec["chainK31"], rec["trueK31"]
    print(f"post L{l}: E0 rms {r0:.2e} (noise {rec['noise_rms']:.1e}) | patch(chain K31): dC rms {ch['dC_rms']:.2e} corr(dC,-E0) {ch['corr_dC_minusE0']:+.3f} "
          f"E rms -> {ch['E1_rms']:.2e}, weighted proj. var err x{ch['proj_ratio']:.3f} | patch(true K31): dC rms {tr['dC_rms']:.2e} corr {tr['corr_dC_minusE0']:+.3f} x{tr['proj_ratio']:.3f}", flush=True)
    print("    parts (chain K31): " + ", ".join(f"{k}: rms {ch['parts_rms'][k]:.1e} proj x{ch['parts_proj_ratio'][k]:.3f}" for k in ch["parts_rms"])
          + " | (true K31) CK31 proj x" + f"{tr['parts_proj_ratio']['CK31']:.3f}", flush=True)
json.dump(out, open("../results/summaries/astra7_cov.json", "w"), indent=1)
