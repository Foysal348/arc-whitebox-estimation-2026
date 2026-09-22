"""Compare V29 variants on one locally built MLP: FLOPs, wall, residual, and prediction drift vs the reference config.
Usage (from whest-starterkit): python -m uv run python ../experiments/chain/knob_test.py <tag> KEY=VAL ...   (env knobs of estimator_v29)
Writes ../knob_<tag>.npz with the prediction so tags can be diffed."""
import os, sys, time, importlib.util
tag = sys.argv[1]
for kv in sys.argv[2:]:
    k, v = kv.split("="); os.environ[k] = v
import numpy as np, ctypes
import psutil
def peak_gb():
    mi=psutil.Process().memory_info(); return getattr(mi,"peak_wset",0)/2**30, getattr(mi,"peak_pagefile",0)/2**30
import flopscope as flops
import flopscope.numpy as fnp
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
spec = importlib.util.spec_from_file_location("est", os.environ.get("EST", "../submissions/v29/estimator.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
est = m.Estimator(); est.setup(_Ctx())
mlp = build_mlp(width=1024, depth=16, seed=7)
B = 2**41
res = {}
for call in range(2):   # first call is warm-up (shallower Strassen); second is steady state
    t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = est.predict(mlp, B)
    res[call] = dict(flops=ctx.flops_used, util=ctx.flops_used / B, wall=time.time() - t0, residual=ctx.residual_wall_time_s)
    print(f"[{tag}] call {call}: util {res[call]['util']:.4f} flops {ctx.flops_used:,} wall {res[call]['wall']:.1f}s residual {ctx.residual_wall_time_s:.2f}s peakWS {peak_gb()[0]:.2f}GB peakCommit {peak_gb()[1]:.2f}GB", flush=True)
p = np.asarray(pred, dtype=np.float64)
np.savez(f"../data/dumps/knob_{tag}.npz", pred=p, util=res[1]["util"])
ref = "../data/dumps/knob_ref.npz"
if tag != "ref" and os.path.exists(ref):
    r = np.load(ref)["pred"]
    d = p[-1] - r[-1]
    print(f"[{tag}] final-layer drift vs ref: rms {np.sqrt(np.mean(d*d)):.3e}  max {np.max(np.abs(d)):.3e}  (final mean rms {np.sqrt(np.mean(r[-1]**2)):.3f}); MSE(ref, tag) = {np.mean(d*d):.3e}")
