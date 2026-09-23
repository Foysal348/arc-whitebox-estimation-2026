"""Follow-up to probe_postslices.py (23 Sep): are the two remaining error sources NEW at each layer or INHERITED?

(A) Covariance. dC_post (off-diagonal) is the source of ~95% of the next layer's variance error, and the chain's own
    closure terms explain only 3-8% of it. The linear closure term carries the pre-activation covariance error forward:
    dC_post_ij ~ Phi_i Phi_j dC_pre_ij. How much of dC_post is that inherited part?
(B) kappa4 diagonal. The pairwise transport of the TRUE post slices sets the level (slope 1.00) but not the per-neuron
    pattern (corr 0.24-0.59); the chain's regeneration dG = t_g + lam t_v gets corr 0.75-0.83. Which of its two parts
    carries the pattern, and does the residual t4 - pairwise line up with kappa3 quantities the chain has?

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_inherit.py
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
a4 = "                        dG = t_g + t_v * lam_prev                 # var == diag(C_pre)\n"
for a in (a1, a3, a4): assert src.count(a) == 1, a
src = src.replace(a1, "            _CAPW(li, mu, var, g4row, D3, None if trim else C_off)\n" + a1)
src = src.replace(a3, a3 + "            _CAPC(li, C)\n")
src = src.replace(a4, a4 + "                        _CAPG(li, t_g, t_v, lam_prev)\n")
S = {}
f64 = lambda x: None if x is None else np.asarray(x, dtype=np.float64).copy()
def _CAPW(li, mu, var, g4, D3, C_off):
    S.setdefault(li, {}).update(mu=f64(mu), var=f64(var), g4=f64(g4), D3=f64(D3), C_off=None if C_off is None else np.asarray(C_off, np.float32).copy())
def _CAPC(li, C): S.setdefault(li, {})["C"] = np.asarray(C, np.float32).copy()
def _CAPG(li, t_g, t_v, lam): S.setdefault(li, {}).update(t_g=f64(t_g), t_v=f64(t_v), lam=float(lam))
mod = types.ModuleType("est_inh"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPC": _CAPC, "_CAPG": _CAPG})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
print(f"chain run {time.time()-t0:.0f}s", flush=True)
Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
iu = np.triu_indices(1024, 1)
r2 = lambda y, x: float(1 - np.sum((y - x * (np.dot(x, y) / np.dot(x, x))) ** 2) / np.sum(y * y))
out = []; SAVE = {}
for l in (3, 7, 11, 14, 15):
    li = l - 1; c = S[li]; rec = {"post_layer": l}
    P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz")); T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz"))
    # (A) inherited covariance error
    dCpost = (c["C"].astype(np.float64) - np.asarray(P["C"], np.float64))[iu]
    Cpre = c["C_off"].astype(np.float64); np.fill_diagonal(Cpre, c["var"])
    dCpre = Cpre - np.asarray(T["C"], np.float64)
    ph = Phi(c["mu"] / np.sqrt(c["var"]))
    inh = (ph[:, None] * dCpre * ph[None, :])[iu]
    rec["dCpost_off_rms"] = float(np.sqrt(np.mean(dCpost ** 2))); rec["inherited_rms"] = float(np.sqrt(np.mean(inh ** 2)))
    rec["R2_inherited"] = r2(dCpost, inh); rec["slope_inherited"] = float(np.dot(inh, dCpost) / np.dot(inh, inh))
    rec["fresh_rms"] = float(np.sqrt(np.mean((dCpost - inh * rec["slope_inherited"]) ** 2)))
    # (B) kappa4 diagonal at the next pre-activation
    nx = li + 1; t4 = np.asarray(np.load(os.path.join(PRE, f"layer{l + 1}_seed0.npz"))["K4d"], np.float64); n_ = S[nx]
    W = Wall[nx].astype(np.float64); W2 = W * W
    K22 = np.asarray(P["K22"], np.float64); np.fill_diagonal(K22, 0.0)
    pair = (W2 * W2).T @ np.asarray(P["K4d"], np.float64) + 3 * np.einsum("ji,ji->i", W2, K22 @ W2)
    resid = t4 - pair
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    rec["k4"] = {"corr_t4_tg": cc(t4, n_["t_g"]), "corr_t4_lamtv": cc(t4, n_["lam"] * n_["t_v"]), "corr_t4_g4": cc(t4, n_["g4"]),
                 "corr_resid_tv": cc(resid, n_["t_v"]), "corr_resid_tg": cc(resid, n_["t_g"]), "corr_resid_D3": cc(resid, n_["D3"]),
                 "corr_resid_absD3": cc(resid, np.abs(n_["D3"])), "corr_resid_mu": cc(resid, n_["mu"]),
                 "rms_t4": float(np.sqrt(np.mean(t4 ** 2))), "rms_resid": float(np.sqrt(np.mean(resid ** 2))), "std_t4": float(np.std(t4)), "std_resid": float(np.std(resid)),
                 "std_pair": float(np.std(pair)), "corr_resid_pair": cc(resid, pair)}
    SAVE.update({f"L{l+1}_{k_}": v_ for k_, v_ in dict(t4=t4, pair=pair, D3=n_["D3"], var=n_["var"], mu=n_["mu"], t_v=n_["t_v"], t_g=n_["t_g"], g4=n_["g4"], lam=np.full(1024, n_["lam"])).items()})
    out.append(rec)
    k = rec["k4"]
    print(f"post L{l}: dC_post_off rms {rec['dCpost_off_rms']:.2e} | inherited Phi.dCpre.Phi rms {rec['inherited_rms']:.2e} R2 {rec['R2_inherited']:.3f} slope {rec['slope_inherited']:.2f} "
          f"-> fresh rms {rec['fresh_rms']:.2e}", flush=True)
    print(f"   k4(L{l+1}): corr t4 with t_g {k['corr_t4_tg']:.3f}, lam t_v {k['corr_t4_lamtv']:.3f}, g4 {k['corr_t4_g4']:.3f} | std t4 {k['std_t4']:.2e} pair {k['std_pair']:.2e} resid {k['std_resid']:.2e} "
          f"corr(resid,pair) {k['corr_resid_pair']:+.2f} | resid vs t_v {k['corr_resid_tv']:+.3f} t_g {k['corr_resid_tg']:+.3f} D3 {k['corr_resid_D3']:+.3f} |D3| {k['corr_resid_absD3']:+.3f} mu {k['corr_resid_mu']:+.3f}", flush=True)
json.dump(out, open("../results/summaries/inherit_probe.json", "w"), indent=1)
np.savez_compressed("../data/dumps/inherit_k4.npz", **SAVE)
