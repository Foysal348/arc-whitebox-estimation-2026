"""Run an estimator on a slice of the mini split and save per-MLP feature dumps + truths.
Env: EST (estimator file), DUMP_DIR, START, END, plus the estimator's own knobs (V33_DUMP_DIR is set here).
"""
import os, sys, time, importlib.util, json
import numpy as np
EST = os.environ["EST"]; DUMP_DIR = os.environ.get("DUMP_DIR", "./dumps"); START = int(os.environ.get("START", "0")); END = int(os.environ.get("END", "1"))
os.environ["V33_DUMP_DIR"] = DUMP_DIR; os.makedirs(DUMP_DIR, exist_ok=True)
import flopscope as flops
import whestbench
spec = importlib.util.spec_from_file_location("est", EST); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ds = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision=os.environ.get("REV", "v2-phase2"), split="mini")
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
est = m.Estimator(); est.setup(_Ctx())
log = {}
for i, mlp in enumerate(whestbench.iter_mlps(ds)):
    if i < START: continue
    if i >= END: break
    truth = np.asarray(ds[i]["all_layer_means"], dtype=np.float64)
    t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = est.predict(mlp, 2**41)
    pred = np.asarray(pred, dtype=np.float64)
    mse = float(np.mean((pred[-1] - truth[-1]) ** 2))
    np.savez_compressed(os.path.join(DUMP_DIR, f"truth_{mlp.name}_{mlp.seed}.npz"), truth=truth, pred=pred, mse=mse, util=ctx.flops_used / 2**41)
    log[mlp.name] = {"mse": mse, "util": ctx.flops_used / 2**41, "secs": time.time() - t0}
    print(f"{i} {mlp.name}: raw {mse:.3e} util {ctx.flops_used/2**41:.4f} {time.time()-t0:.0f}s", flush=True)
    json.dump(log, open(os.path.join(DUMP_DIR, f"log_{START}_{END}.json"), "w"), indent=1)
