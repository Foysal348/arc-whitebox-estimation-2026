"""Round 7 (Fable, 23 Sep): END-TO-END test of the Edgeworth coefficient of the newborn's K x C cross terms.

Runs experiments/variants/r7_estimator_fbscale.py (the shipped chain with the D21 feedback constants 1.5 / 0.5 scaled by
R7_FB_SCALE: 1.0 = shipped, 2.0 = the full Edgeworth famA + famB in the all-distinct entries) on public MLP #0, plain
Gaussian chain, and reports per scale: the (2,1)-slice relative error against the pre-activation teacher at layers
4, 8, 12, 15, 16, the per-layer MSE and the final-layer MSE against the baked truth.

Usage (from whest-starterkit):  PYTHONUTF8=1 V31_ANGULAR=0 V19_FULL_LAST=1 V26_STRASSEN=0 python -m uv run python ../experiments/probes/r7_fbscale_run.py 1.0 2.0 [more scales]
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
assert os.environ.get("V31_ANGULAR") == "0"
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"
SRC = os.environ.get("EST", "../experiments/variants/r7_estimator_fbscale.py")
LAY = {4, 8, 12, 15, 16}
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
assert src.count(a1) == 1
src = src.replace(a1, "            _CAP(li, var, D21)\n" + a1)
S = {}
def _CAP(li, var, D21):
    if li + 1 in LAY and D21 is not None: S[li + 1] = (np.asarray(var, np.float64).copy(), np.asarray(D21, np.float32).copy())
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
ALM = np.asarray(truth["all_layer_means"], np.float64); off = ~np.eye(1024, dtype=bool)
TRUE = {l: np.asarray(np.load(os.path.join(PRE, f"layer{l}_seed0.npz"))["Y21"], np.float64) for l in LAY}
res = {}
for sc in sys.argv[1:]:
    os.environ["R7_FB_SCALE"] = sc; S.clear()
    mod = types.ModuleType(f"est_r7_{sc}"); mod.__dict__["_CAP"] = _CAP
    exec(compile(src, SRC, "exec"), mod.__dict__)
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    per = [float(np.mean((pred[k] - ALM[k]) ** 2)) for k in range(16)]
    d21 = {}
    for l in sorted(S):
        v, D = S[l]; Y = TRUE[l]; e = (D.astype(np.float64) - Y)[off]
        d21[l] = {"rel_err": float(np.sqrt(np.mean(e ** 2) / np.mean(Y[off] ** 2))), "err_rms": float(np.sqrt(np.mean(e ** 2)))}
    res[sc] = {"final": per[-1], "per_layer": per, "d21": d21, "util": ctx.flops_used / 2**41}
    print(f"[scale {sc}] final MSE {per[-1]:.4e} (L8 {per[7]:.3e}, L12 {per[11]:.3e}) util {ctx.flops_used/2**41:.4f} ({time.time()-t0:.0f}s) | D21 rel err: " + ", ".join(f"L{l} {d21[l]['rel_err']*100:.2f}%" for l in sorted(d21)), flush=True)
    del est, mod
b = res.get("1.0", {}).get("final")
if b:
    for sc in res: print(f"  scale {sc}: ratio to shipped {res[sc]['final']/b:.4f}")
json.dump(res, open(f"../results/summaries/r7_fbscale{os.environ.get('TAG', '')}.json", "w"), indent=1)
