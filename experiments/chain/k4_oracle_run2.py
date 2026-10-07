"""V34b: the kappa4 oracle with the UNITS FIXED (23 Sep). Supersedes k4_oracle_run.py.

The first oracle (K3x) injected the teacher's slices as g4row = K4d * METRIC_C, wk4m = K22 * METRIC_C / 6 and
wk431 = K31 * METRIC_C / 2, on the ANGULAR chain. probe_errbudget.py measured the chain's own objects against the same
teacher and found that they are already in cumulant units:
    g4row  vs K4d (diagonal):          slope 0.98 at layer 2 (falling to 0.75 at depth: the chain is too small there)
    wk4m   vs K22 (off-diagonal):      slope 1.00 at layer 4, R^2 0.985 (1.17 at layer 16)
    wk431  vs K31:                     wk431 = lam C_off is the (3,1) slice model itself
so the first oracle doubled the diagonal and cut the (2,2) slice by 3, and it did so on a chain whose internal state
describes the SPHERICAL input (V31), not the Gaussian one the teacher sampled. Its 6-15x degradation measured the bug.

This run uses the plain Gaussian chain (V31_ANGULAR=0) and injects the slices 1:1. The teacher's sampling noise on the
kappa4 diagonal is sqrt(24 sigma^8 / N): 2.3e-3 at layer 1 (where the true value is 0) but 1.7e-5 (<1%) at layer 16, so
'all' is a clean oracle from layer 3 on; 'allfit' uses the chain's own diagonal times the V39 depth table instead.
Modes (V34_MODE): none | diag | k22 | k31 | k22k31 | all | allfit
Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/chain/k4_oracle_run2.py none k22 k31 ...
"""
import os, sys, time, types, glob, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
assert os.environ.get("V31_ANGULAR") == "0", "run the plain Gaussian chain: V31_ANGULAR=0"
DIR = os.environ.get("TEACHER", "../data/kaggle_out/kaggle_out_k4pub/k4teach")
T = {int(os.path.basename(f).split("_")[0][5:]): f for f in glob.glob(os.path.join(DIR, "layer*_seed*.npz"))}
truth = np.load(os.path.join(DIR, "truth.npz"))
Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"]
K4T = [1.0, 1.007, 1.005, 1.007, 1.011, 1.021, 1.034, 1.036, 1.058, 1.065, 1.072, 1.109, 1.132, 1.170, 1.184, 1.206]
MODE = ["none"]
CACHE = {}

def _K4_ORACLE(li, n, var, C_off, g4row, wk4m, wk431, trim):
    m = MODE[0]; l = li + 1
    if m == "none" or l not in T: return g4row, wk4m, wk431
    if l not in CACHE:
        z = np.load(T[l]); K22 = np.asarray(z["K22"], np.float32).copy(); K31 = np.asarray(z["K31"], np.float32).copy()
        np.fill_diagonal(K22, 0.0); np.fill_diagonal(K31, 0.0)
        CACHE.clear(); CACHE[l] = (np.asarray(z["K4d"], np.float32), K22, K31)
    K4d, K22, K31 = CACHE[l]
    if m in ("diag", "all"):
        # K4O_LMIN / K4O_LMAX: inject only at layers in [lmin, lmax]; K4O_BLEND: (1-b) chain + b teacher;
        # K4O_PART: "pattern" = teacher pattern at the chain's layer-mean level, "level" = chain pattern at the teacher's level
        lmin = int(os.environ.get("K4O_LMIN", "1")); lmax = int(os.environ.get("K4O_LMAX", "16"))
        if lmin <= l <= lmax:
            ch = np.asarray(g4row, np.float64); tt = np.asarray(K4d, np.float64)
            part = os.environ.get("K4O_PART", "")
            if part == "pattern":
                tt = tt * (ch.mean() / tt.mean())
            elif part == "level":
                tt = ch * (tt.mean() / ch.mean())
            b = float(os.environ.get("K4O_BLEND", "1"))
            g4row = fnp.asarray(((1 - b) * ch + b * tt).astype(np.float32))
    if m == "allfit": g4row = g4row * float(K4T[min(li, 15)])
    if m in ("k22", "k22k31", "all", "allfit") and wk4m is not None: wk4m = fnp.asarray(K22)
    if m in ("k31", "k22k31", "all", "allfit") and wk431 is not None: wk431 = fnp.asarray(K31)
    return g4row, wk4m, wk431

SRC = os.environ.get("ORACLE_EST", "../experiments/variants/estimator_k4oracle.py")
mod = types.ModuleType("est_k4o2"); mod.__dict__["_K4_ORACLE"] = _K4_ORACLE
os.environ["V34_K4_ORACLE"] = DIR
exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])],
          seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
ALM = np.asarray(truth["all_layer_means"], np.float64)
res = {}
for m in sys.argv[1:]:
    MODE[0] = m; CACHE.clear()
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        p = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    per = [float(np.mean((p[k] - ALM[k]) ** 2)) for k in range(16)]
    res[m] = {"final": per[-1], "per_layer": per, "util": ctx.flops_used / 2**41}
    print(f"[{m}] util {ctx.flops_used / 2**41:.4f} final-layer MSE {per[-1]:.4e}  (L8 {per[7]:.3e}, L12 {per[11]:.3e})  ({time.time()-t0:.0f}s)", flush=True)
b = res.get("none", {}).get("final")
if b:
    for m in res: print(f"  {m:8s} ratio to none {res[m]['final']/b:.4f}")
json.dump(res, open(f"../results/summaries/k4oracle2{os.environ.get('TAG', '')}.json", "w"), indent=1)
