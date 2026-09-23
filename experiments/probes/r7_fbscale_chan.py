"""Round 7 (Fable, 23 Sep): per-CHANNEL errors of the chain vs the feedback coefficient scale.

For each R7_FB_SCALE in argv, runs experiments/variants/r7_estimator_fbscale.py (plain Gaussian chain) and reports per layer
the relative rms error against the pre-activation teacher of: var, kappa3_iii (D3), the (2,1) slice (D21), the kappa4
diagonal (g4row), the pre-activation mean, plus the off-diagonal C error, and the per-layer / final MSE against the baked truth.
Also stores D21 at layers 4, 8, 12, 15, 16 so that the CHANGE of the slice between scales can be regressed on the error.

Usage (from whest-starterkit):  PYTHONUTF8=1 V31_ANGULAR=0 V19_FULL_LAST=1 V26_STRASSEN=0 python -m uv run python ../experiments/probes/r7_fbscale_chan.py 0.0 1.0 1.5 2.0
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
KEEP = {4, 8, 12, 15, 16}
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
assert src.count(a1) == 1
src = src.replace(a1, "            _CAP(li, mu, var, D3, D21, C_off, g4row)\n" + a1)
S = {}
f64 = lambda x: None if x is None else np.asarray(x, np.float64).copy()
def _CAP(li, mu, var, D3, D21, C_off, g4row):
    l = li + 1
    S[l] = dict(mu=f64(mu), var=f64(var), D3=f64(D3), g4=f64(g4row), D21=(np.asarray(D21, np.float32).copy() if (D21 is not None and l in KEEP) else None),
                D21_err=None, Coff=None)
    T = TRUE[l]
    if D21 is not None:
        e = (np.asarray(D21, np.float64) - T["Y21"])[off]; S[l]["D21_err"] = float(np.sqrt(np.mean(e ** 2) / np.mean(T["Y21"][off] ** 2)))
    if C_off is not None:
        e = (np.asarray(C_off, np.float64) - T["Coff"])[off]; S[l]["Coff_err"] = float(np.sqrt(np.mean(e ** 2) / np.mean(T["Coff"][off] ** 2)))
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
n = Wall.shape[1]; off = ~np.eye(n, dtype=bool)
TRUE = {}
for l in range(1, 17):
    z = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); C = np.asarray(z["C"], np.float64); Co = C.copy(); np.fill_diagonal(Co, 0.0)
    TRUE[l] = dict(mu=np.asarray(z["mean"], np.float64), var=np.diag(C).copy(), D3=np.diag(np.asarray(z["Y21"], np.float64)).copy(), K4=np.asarray(z["K4d"], np.float64),
                   Y21=np.asarray(z["Y21"], np.float64), Coff=Co)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=n, depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
ALM = np.asarray(truth["all_layer_means"], np.float64)
rel = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2)))
res = {}; D21S = {}
for sc in sys.argv[1:]:
    os.environ["R7_FB_SCALE"] = sc; S.clear()
    mod = types.ModuleType(f"est_r7c_{sc}"); mod.__dict__["_CAP"] = _CAP
    exec(compile(src, SRC, "exec"), mod.__dict__)
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    per = [float(np.mean((pred[k] - ALM[k]) ** 2)) for k in range(16)]
    rows = {}
    for l in sorted(S):
        s = S[l]; T = TRUE[l]
        rows[l] = {"mu": rel(s["mu"], T["mu"]), "var": rel(s["var"], T["var"]), "D3": (rel(s["D3"], T["D3"]) if s["D3"] is not None else None),
                   "K4": (rel(s["g4"], T["K4"]) if s["g4"] is not None else None), "D21": s["D21_err"], "Coff": s.get("Coff_err"), "mse": per[l - 1]}
        if s["D21"] is not None: D21S[(sc, l)] = s["D21"]
    res[sc] = {"final": per[-1], "per_layer": per, "rows": rows, "util": ctx.flops_used / 2**41}
    print(f"[scale {sc}] final MSE {per[-1]:.4e} ({time.time()-t0:.0f}s)", flush=True)
    print("   L : " + " ".join(f"{l:>7d}" for l in sorted(rows)), flush=True)
    for ch in ("var", "D3", "D21", "K4", "Coff", "mu"):
        print(f"   {ch:4s}: " + " ".join((f"{rows[l][ch]*100:7.2f}" if rows[l][ch] is not None else "      -") for l in sorted(rows)) + "  (% rel err)", flush=True)
    print("   mse : " + " ".join(f"{rows[l]['mse']:7.1e}" for l in sorted(rows)), flush=True)
    del est, mod
# the change of the slice between scales, against the shipped chain's slice error
if "1.0" in res:
    for sc in res:
        if sc == "1.0": continue
        for l in sorted(KEEP):
            if ("1.0", l) in D21S and (sc, l) in D21S:
                e1 = (D21S[("1.0", l)].astype(np.float64) - TRUE[l]["Y21"])[off]; d = (D21S[(sc, l)].astype(np.float64) - D21S[("1.0", l)].astype(np.float64))[off]
                slope = float(np.dot(d, e1) / np.dot(d, d)); r2 = float(1 - np.sum((e1 - slope * d) ** 2) / np.sum(e1 ** 2))
                res[sc]["rows"][l]["delta_vs_err"] = {"slope_err_on_delta": slope, "r2": r2, "delta_rms": float(np.sqrt(np.mean(d ** 2))), "err_rms": float(np.sqrt(np.mean(e1 ** 2)))}
                print(f"  scale {sc} L{l}: shipped slice error regressed on (D21_scale - D21_shipped): slope {slope:+.3f} (ideal -1), R2 {r2:.3f}, |delta| {np.sqrt(np.mean(d**2)):.2e} vs |err| {np.sqrt(np.mean(e1**2)):.2e}", flush=True)
b = res.get("1.0", {}).get("final")
if b:
    for sc in res: print(f"  scale {sc}: final ratio to shipped {res[sc]['final']/b:.4f}")
json.dump(res, open(f"../results/summaries/r7_fbscale_chan{os.environ.get('TAG', '')}.json", "w"), indent=1)
