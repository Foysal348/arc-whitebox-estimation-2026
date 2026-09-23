"""Moonshot diagnostic (23 Sep): WHERE does the chain's final error come from?

The per-layer MSE of V31 against the baked truth grows almost linearly from layer 3 to layer 16 (0.7e-9 noise floor ->
2.29e-8), so the error is accumulated, not made at the last layer. The last-layer mean is a function of the MARGINAL of each
pre-activation only, E[relu(z_i)] = Edgeworth(mu_i, var_i, k3_i, k4_i), so it can be split exactly by substituting the
MC teacher's true marginal cumulants one at a time into the chain's own closure:

    mean error  =  (error of mu)  +  (error of var)  +  (error of k3_iii)  +  (error of k4_iiii)  +  closure floor

The teacher (kaggle/k4, 6e7 samples, public MLP #0 logan-fitzgerald) stores for every layer the pre-activation mean, the
covariance, E[y_i^2 y_j] (whose diagonal is k3_iii), the k4 diagonal and the MC ReLU mean; the baked truth has 1e9 samples.

Usage (from whest-starterkit):
    python -m uv run python ../experiments/probes/probe_errbudget.py
"""
import os, sys, types, json, time, glob
import numpy as np
from math import factorial
from math import erf, sqrt as _sqrt, pi as _pi
class norm:
    pdf = staticmethod(lambda a: np.exp(-0.5 * a * a) / np.sqrt(2 * _pi))
    cdf = staticmethod(np.vectorize(lambda a: 0.5 * (1.0 + erf(a / _sqrt(2.0)))))
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

DIR = os.environ.get("TEACHER", "../data/kaggle_out/kaggle_out_k4pub/k4teach")
SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a2 = "            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]\n"
assert src.count(a1) == 1 and src.count(a2) == 1
src = src.replace(a1, "            _CAP(li, mu, var, D3, g4row)\n" + a1)
src = src.replace(a2, a2 + "            _CAP2(li, pk1v)\n")

CAP = {}
def _f(x): return None if x is None else np.asarray(x, dtype=np.float64).copy()
def _CAP(li, mu, var, D3, g4row):
    CAP.setdefault(li, {}).update(mu=_f(mu), var=_f(var), k3=_f(D3), k4=_f(g4row))
def _CAP2(li, pk1v):
    CAP.setdefault(li, {})["pred"] = _f(pk1v)

mod = types.ModuleType("est_eb"); mod.__dict__.update({"_CAP": _CAP, "_CAP2": _CAP2})
exec(compile(src, SRC, "exec"), mod.__dict__)
_tf = os.path.join(DIR, "truth.npz")
truth = np.load(_tf) if os.path.exists(_tf) else {"mlp_seed": 0, "mlp_name": "synthetic"}   # synthetic teacher: no baked truth
Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])],
          seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    pred = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
print(f"chain run {time.time()-t0:.0f}s  util {ctx.flops_used/2**41:.4f}", flush=True)
ALM = np.asarray(truth["all_layer_means"], dtype=np.float64) if "all_layer_means" in truth else None

def He(k, a):
    h0, h1 = np.ones_like(a), a
    if k == 0: return h0
    for j in range(1, k): h0, h1 = h1, a * h1 - j * h0
    return h1
def dG(k, mu, s):
    """k-th mu-derivative of G(mu, s) = E relu(mu + s Z)."""
    a = mu / s
    if k == 0: return s * norm.pdf(a) + mu * norm.cdf(a)
    if k == 1: return norm.cdf(a)
    return (-1) ** k * He(k - 2, a) * norm.pdf(a) / s ** (k - 1)
def edge(mu, var, k3, k4):
    s = np.sqrt(var)
    return dG(0, mu, s) + k3 / 6 * dG(3, mu, s) + k4 / 24 * dG(4, mu, s) + k3 ** 2 / 72 * dG(6, mu, s) + k3 * k4 / 144 * dG(7, mu, s)

rows = []; SAVE = {}
for li in sorted(CAP):
    l = li + 1
    f = os.path.join(DIR, f"layer{l}_seed0.npz")
    if not os.path.exists(f): continue
    T = np.load(f); c = CAP[li]
    tm = np.asarray(T["mean"], np.float64); tv = np.diag(np.asarray(T["C"], np.float64)).copy()
    t3 = np.diag(np.asarray(T["Y21"], np.float64)).copy(); t4 = np.asarray(T["K4d"], np.float64)
    trelu = np.asarray(T["relu_mean"], np.float64); truth_l = ALM[li] if ALM is not None else trelu
    k3 = c["k3"] if c["k3"] is not None else np.zeros_like(tm)
    k4 = c["k4"] if c["k4"] is not None else np.zeros_like(tm)
    mse = lambda p: float(np.mean((p - truth_l) ** 2))
    r = {"layer": l,
         "chain_pred": mse(c["pred"]) if "pred" in c else None,
         "chain_edge": mse(edge(c["mu"], c["var"], k3, k4)),
         "teacher_relu_vs_truth": mse(trelu),                       # MC noise of the 6e7-sample teacher
         "floor_edge_teacher": mse(edge(tm, tv, t3, t4)),            # closure floor with the true marginal cumulants
         "floor_edge_teacher_vs_mc": float(np.mean((edge(tm, tv, t3, t4) - trelu) ** 2)),   # same samples: noise-reduced
         "sub_mu": mse(edge(tm, c["var"], k3, k4)),
         "sub_var": mse(edge(c["mu"], tv, k3, k4)),
         "sub_k3": mse(edge(c["mu"], c["var"], t3, k4)),
         "sub_k4": mse(edge(c["mu"], c["var"], k3, t4)),
         "sub_mu_var": mse(edge(tm, tv, k3, k4)),
         "err_mu_rms": float(np.sqrt(np.mean((c["mu"] - tm) ** 2))),
         "err_var_rms": float(np.sqrt(np.mean((c["var"] - tv) ** 2))),
         "err_k3_rms": float(np.sqrt(np.mean((k3 - t3) ** 2))), "k3_rms": float(np.sqrt(np.mean(t3 ** 2))),
         "err_k4_rms": float(np.sqrt(np.mean((k4 - t4) ** 2))), "k4_rms": float(np.sqrt(np.mean(t4 ** 2))),
         # first-order attribution of the mean error to each cumulant's error (chain point)
         }
    s = np.sqrt(c["var"])
    lin = {"mu": dG(1, c["mu"], s) * (tm - c["mu"]),
           "var": dG(2, c["mu"], s) / 2 * (tv - c["var"]),
           "k3": dG(3, c["mu"], s) / 6 * (t3 - k3),
           "k4": dG(4, c["mu"], s) / 24 * (t4 - k4)}
    for k, v in lin.items(): r["lin_" + k] = float(np.mean(v * v))
    r["lin_sum"] = float(np.mean(sum(lin.values()) ** 2))
    rows.append(r)
    SAVE.update({f"L{l}_{k}": v for k, v in dict(cmu=c["mu"], cvar=c["var"], ck3=k3, ck4=k4, cpred=c.get("pred", tm * 0), tmu=tm, tvar=tv, tk3=t3, tk4=t4, trelu=trelu, truth=truth_l).items()})
    print(f"L{l:2d} chain {r['chain_pred'] if r['chain_pred'] is not None else float('nan'):.3e} edge {r['chain_edge']:.3e} | teacher-noise {r['teacher_relu_vs_truth']:.2e} "
          f"floor {r['floor_edge_teacher']:.2e} (vs MC {r['floor_edge_teacher_vs_mc']:.2e}) | sub mu {r['sub_mu']:.2e} var {r['sub_var']:.2e} "
          f"k3 {r['sub_k3']:.2e} k4 {r['sub_k4']:.2e} mu+var {r['sub_mu_var']:.2e} | lin mu {r['lin_mu']:.2e} var {r['lin_var']:.2e} "
          f"k3 {r['lin_k3']:.2e} k4 {r['lin_k4']:.2e}", flush=True)
TAG = os.environ.get("TAG", "v31")
json.dump(rows, open(f"../results/summaries/errbudget_{TAG}.json", "w"), indent=1)
np.savez_compressed(f"../data/dumps/errbudget_{TAG}.npz", **SAVE)
