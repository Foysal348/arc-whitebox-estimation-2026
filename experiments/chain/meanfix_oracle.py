"""Channel oracles for the MEAN closure (23 Sep): what is each input of E[relu(z_i)] worth if it were exact?

The split kappa4 oracle showed that the whole -30% of the exact kappa4 diagonal comes through the mean closure alone
(true kappa4 only there: 1.602e-8; everywhere else only: 2.311e-8; chain: 2.290e-8). The mean closure is the Edgeworth
function E(mu, var, k3, k4) (reproduced to 4 digits by probe_errbudget.py). This oracle adds, at every layer and right
after the chain forms its mean, E(with the teacher's value of the chosen inputs) - E(chain inputs), and lets the
corrected mean propagate. Modes (argv): any '+'-joined subset of var, k3, k4 (k4 alone must reproduce ~1.60e-8).

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/chain/meanfix_oracle.py none k4 var k3 var+k3 var+k3+k4
"""
import os, sys, types, json, time, glob
import numpy as np
from math import erf, sqrt as _sqrt, pi as _pi
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
assert os.environ.get("V31_ANGULAR") == "0"
PRE = os.environ.get("TEACHER", "../data/kaggle_out/kaggle_out_k4pub/k4teach")
SRC = "../submissions/v31ang/estimator.py"
src = open(SRC, encoding="utf-8").read()
a = "            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]\n"
assert src.count(a) == 1
src = src.replace(a, a + "            pk1v = _MEANFIX(li, pk1v, mu, var, D3, g4row)\n")
Phi = np.vectorize(lambda x: 0.5 * (1.0 + erf(x / _sqrt(2.0)))); pdf = lambda x: np.exp(-0.5 * x * x) / np.sqrt(2 * _pi)
def He(k, x):
    h0, h1 = np.ones_like(x), x
    if k == 0: return h0
    for j in range(1, k): h0, h1 = h1, x * h1 - j * h0
    return h1
def dG(k, mu, s):
    x = mu / s
    if k == 0: return s * pdf(x) + mu * Phi(x)
    if k == 1: return Phi(x)
    return (-1) ** k * He(k - 2, x) * pdf(x) / s ** (k - 1)
def edge(mu, var, k3, k4):
    s = np.sqrt(var)
    return dG(0, mu, s) + k3 / 6 * dG(3, mu, s) + k4 / 24 * dG(4, mu, s) + k3 ** 2 / 72 * dG(6, mu, s) + k3 * k4 / 144 * dG(7, mu, s)
TCH = {}
def teacher(l):
    if l not in TCH:
        f = os.path.join(PRE, f"layer{l}_seed0.npz")
        if not os.path.exists(f): TCH[l] = None
        else:
            T = np.load(f); TCH[l] = dict(var=np.diag(np.asarray(T["C"], np.float64)).copy(), k3=np.diag(np.asarray(T["Y21"], np.float64)).copy(), k4=np.asarray(T["K4d"], np.float64))
    return TCH[l]
MODE = [set()]
def _MEANFIX(li, pk1v, mu, var, D3, g4row):
    if not MODE[0] or D3 is None or g4row is None: return pk1v
    t = teacher(li + 1)
    if t is None: return pk1v
    m, v, k3, k4 = (np.asarray(x, np.float64) for x in (mu, var, D3, g4row))
    v2 = t["var"] if "var" in MODE[0] else v; k32 = t["k3"] if "k3" in MODE[0] else k3; k42 = t["k4"] if "k4" in MODE[0] else k4
    delta = edge(m, v2, k32, k42) - edge(m, v, k3, k4)
    return pk1v + fnp.asarray(delta.astype(np.float32))
mod = types.ModuleType("est_mf"); mod.__dict__["_MEANFIX"] = _MEANFIX
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
ALM = np.asarray(truth["all_layer_means"], np.float64)
res = {}
for m in sys.argv[1:]:
    MODE[0] = set() if m == "none" else set(m.split("+"))
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        p = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    per = [float(np.mean((p[k] - ALM[k]) ** 2)) for k in range(16)]
    res[m] = per
    print(f"[{m}] final-layer MSE {per[-1]:.4e}  ratio {per[-1] / res.get('none', per)[-1]:.3f}  (L8 {per[7]:.3e}, L12 {per[11]:.3e})  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open("../results/summaries/meanfix_oracle.json", "w"), indent=1)
