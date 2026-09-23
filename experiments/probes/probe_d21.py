"""Moonshot probe (23 Sep): is the chain's kappa3 (2,1) SLICE right, entry by entry?

The channel oracles (meanfix_oracle.py) put the largest single lever on the pre-activation VARIANCE (-47% if exact in
the mean closure), and probe_postslices/probe_inherit found that the variance error is ~95% the off-diagonal post-ReLU
covariance error, of which ~4.2e-5 rms per entry is FRESH at every layer and unexplained by the chain's closure terms.
The off-diagonal covariance closure is driven by the kappa3 slice D21_ij = k3(z_i, z_i, z_j) (terms d21, d21T, d21*d21T...),
which the chain builds from its sum over sources with truncated old tiers. The pre-activation teacher stores the TRUE
slice E[y_i^2 y_j] at every layer. Compare them, overall and split by the neuron's |mu/sigma|.

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_d21.py
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"
SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
assert src.count(a1) == 1
src = src.replace(a1, "            _CAP(li, mu, var, D3, None if trim else D21)\n" + a1)
S = {}
LAY = {3, 5, 7, 9, 11, 13, 14, 15}
def _CAP(li, mu, var, D3, D21):
    if li + 1 in LAY and D21 is not None:
        S[li] = dict(mu=np.asarray(mu, np.float64).copy(), var=np.asarray(var, np.float64).copy(), D3=np.asarray(D3, np.float64).copy(), D21=np.asarray(D21, np.float32).copy())
mod = types.ModuleType("est_d21"); mod.__dict__["_CAP"] = _CAP
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
if os.environ.get("RFB"): est.R_FB = int(os.environ["RFB"])
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    pred = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
ALM = np.asarray(truth["all_layer_means"], np.float64)
print(f"chain run {time.time()-t0:.0f}s  R_FB {est.R_FB}  util {ctx.flops_used/2**41:.4f}  final MSE {np.mean((pred[-1]-ALM[-1])**2):.4e}", flush=True)
off = ~np.eye(1024, dtype=bool)
out = []
for li in sorted(S):
    l = li + 1; T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); c = S[li]
    Y = np.asarray(T["Y21"], np.float64); D = c["D21"].astype(np.float64); v = np.diag(np.asarray(T["C"], np.float64))
    # orientation: the chain's D21 may be k3(i,i,j) or its transpose; pick the better match
    best = None
    for nm, Dm in (("D21", D), ("D21^T", D.T)):
        x = Dm[off]; y = Y[off]; s = float(np.dot(x, y) / np.dot(x, x)); r2 = float(1 - np.sum((y - s * x) ** 2) / np.sum((y - y.mean()) ** 2))
        if best is None or r2 > best[2]: best = (nm, s, r2, Dm)
    nm, s, r2, Dm = best
    noise = float(np.sqrt(np.mean((np.outer(v, np.sqrt(v)) ** 2)[off] * 2.0 / 6e7)))     # ~ sqrt(Var(y_i^2 y_j)/N)
    e = (Dm - Y)[off]; rec = {"layer": l, "orient": nm, "slope_true_on_chain": s, "r2": r2, "rel_err": float(np.sqrt(np.mean(e ** 2) / np.mean(Y[off] ** 2))),
                             "err_rms": float(np.sqrt(np.mean(e ** 2))), "true_rms": float(np.sqrt(np.mean(Y[off] ** 2))), "noise_est": noise,
                             "d3_slope": float(np.dot(c["D3"], np.diag(Y)) / np.dot(c["D3"], c["D3"]))}
    a = c["mu"] / np.sqrt(c["var"])
    for lo, hi in ((0, 0.75), (0.75, 1.5), (1.5, 99)):
        m = (np.abs(a) >= lo) & (np.abs(a) < hi); M = off & m[:, None]
        rec[f"rel_err_rows_a{lo}-{hi}"] = float(np.sqrt(np.mean((Dm - Y)[M] ** 2) / np.mean(Y[M] ** 2)))
    out.append(rec)
    print(f"L{l:2d}: chain {nm} vs true k3(i,i,j): slope {s:.3f} R2 {r2:.3f} rel err {rec['rel_err']:.3f} (err rms {rec['err_rms']:.2e}, true rms {rec['true_rms']:.2e}, noise ~{noise:.1e}) "
          f"| rows |a|<0.75 {rec['rel_err_rows_a0-0.75']:.3f}, 0.75-1.5 {rec['rel_err_rows_a0.75-1.5']:.3f}, >1.5 {rec['rel_err_rows_a1.5-99']:.3f} | D3 slope {rec['d3_slope']:.3f}", flush=True)
json.dump(out, open("../results/summaries/d21_probe.json", "w"), indent=1)
