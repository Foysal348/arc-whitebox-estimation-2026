"""V34 oracle ceiling: run V31 on a public mini MLP with its kappa4 (2,2)/(3,1)/diagonal slices replaced by the MC-measured
TRUE ones (k4teacher output), and compare the final-layer MSE against the baked 1e9-sample truth.
The chain models the kappa4 core as G = diag(g) + lambda C_off and feeds:
   g4row = dG * c            (dG = kappa4 diagonal of the pre-activation)
   wk4m  = (dG_i + dG_j) c/6 (the (2,2) slice)
   wk431 = (c/2) lambda C_off (the (3,1) slice)
so the oracle substitutes, from the teacher's measured slices at the same layer:
   dG_true = K4d,  wk4m_true = K22_off * (c/6) / <scale>,  wk431_true = K31 * (c/2) / <scale>
with the scale chosen so that the substitution is exact when the model is exact (calibrated on the model's own relation).
Modes (env V34_MODE): all | diag | k22 | k31   (which part to take from the teacher)
Usage (from whest-starterkit): V34_K4_ORACLE=<dir> python -m uv run python ../experiments/chain/k4_oracle_run.py
"""
import os, sys, time, types, glob, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
DIR = os.environ["V34_K4_ORACLE"]; MODE = os.environ.get("V34_MODE", "all")
T = {}
for f in glob.glob(os.path.join(DIR, "layer*_seed*.npz")):
    l = int(os.path.basename(f).split("_")[0][5:]); T[l] = f
truth = np.load(os.path.join(DIR, "truth.npz"))
W = None
wf = glob.glob(os.path.join(DIR, "weights_seed*.npz"))
Wall = np.load(wf[0])["W"] if wf else None
def _K4_ORACLE(li, n, var, C_off, g4row, wk4m, wk431, trim):
    """li is 0-based: the pre-activation being processed is that of layer li+1 (1-based) = the teacher's 'layer li+1'."""
    l = li + 1
    if l not in T: return g4row, wk4m, wk431
    z = np.load(T[l]); METRIC_C = 2.0
    K4d = np.asarray(z["K4d"], dtype=np.float32); K22 = np.asarray(z["K22"], dtype=np.float32); K31 = np.asarray(z["K31"], dtype=np.float32)
    np.fill_diagonal(K22, 0.0); np.fill_diagonal(K31, 0.0)
    if MODE in ("all", "diag"):
        g4row = fnp.asarray(K4d * METRIC_C)
    if MODE in ("all", "k22") and wk4m is not None:
        wk4m = fnp.asarray(K22 * (METRIC_C / 6.0))
    if MODE in ("all", "k31") and wk431 is not None:
        wk431 = fnp.asarray(K31 * (0.5 * METRIC_C))
    return g4row, wk4m, wk431
SRC = "../experiments/variants/estimator_k4oracle.py"
mod = types.ModuleType("est_k4o"); mod.__dict__["_K4_ORACLE"] = _K4_ORACLE
exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
tr = np.asarray(truth["final_means"], dtype=np.float64)
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    pred = est.predict(mlp, 2**41)
p = np.asarray(pred, dtype=np.float64)
mse = float(np.mean((p[-1] - tr) ** 2))
print(f"[{MODE}] final-layer MSE vs baked truth {mse:.4e}   util {ctx.flops_used/2**41:.4f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump({MODE: mse}, open(f"../data/dumps/k4oracle_{MODE}.json", "w"), indent=1)
