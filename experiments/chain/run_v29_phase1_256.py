"""Pilot G (companion): V29 on the SAME 256x16 networks the ARC reference notebook uses
(Phase-1 public MLPs, first 16 weight matrices, truth = all_layer_means[15], 1e9 samples).
At n=256 the shared-basis ranks (384/224) exceed n, so V29 runs every source dense (its most accurate form).
Usage (from whest-starterkit): python -m uv run python ../experiments/chain/run_v29_phase1_256.py [N_MLPS] [env knobs apply, e.g. V17_NO_REGEN=1]
"""
import os, sys, time, types, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DEPTH = int(os.environ.get("DEPTH", "16"))
TAG = os.environ.get("TAG", "v29")
SRC = os.environ.get("EST", "../submissions/v29m8f5/estimator.py")
mod = types.ModuleType("est_p1"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)

ds = load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini").with_format("numpy")
class _Ctx: seed = 0; width = 256; depth = DEPTH; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
res = {}
for i in range(N):
    r = ds[i]; W = np.asarray(r["weights"][:DEPTH], dtype=np.float32); truth = np.asarray(r["all_layer_means"][DEPTH - 1], dtype=np.float64)
    mlp = MLP(width=W.shape[1], depth=DEPTH, weights=[fnp.array(W[l]) for l in range(DEPTH)], seed=int(r["mlp_seed"]), name=str(r["mlp_name"]))
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = est.predict(mlp, 2**41)
    pred = np.asarray(pred, dtype=np.float64); d = pred[-1] - truth; mse = float(np.mean(d * d))
    units = ctx.flops_used / 2**31
    res[str(r["mlp_name"])] = {"mse": mse, "units": units, "secs": time.time() - t0}
    print(f"{r['mlp_name']}|{TAG}: raw MSE {mse:.3e}  {units:.1f} u  {time.time()-t0:.0f}s", flush=True)
json.dump(res, open(f"../data/dumps/phase1_256_{TAG}.json", "w"), indent=1)
