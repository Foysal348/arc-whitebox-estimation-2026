"""Back-to-back residual A/B of two V29 knob sets in one process on the seed-7 MLP.
Usage (from whest-starterkit): python -m uv run python ../experiments/chain/resid_ab.py "tagA KEY=VAL ..." "tagB KEY=VAL ..." [rounds]
Each config gets its own Estimator (pools), warmed with one call; then `rounds` alternating steady calls are timed.
"""
import os, sys, time, importlib.util
import numpy as np
import flopscope as flops
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
cfgs = [a.split() for a in sys.argv[1:3]]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
mlp = build_mlp(width=1024, depth=16, seed=7)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
ests = {}
base_env = dict(os.environ)
for c in cfgs:
    os.environ.clear(); os.environ.update(base_env)
    for kv in c[1:]:
        k, v = kv.split("="); os.environ[k] = v
    spec = importlib.util.spec_from_file_location("est_" + c[0], os.environ.get("EST", "../submissions/v29m8f5/estimator.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    e = m.Estimator(); e.setup(_Ctx()); ests[c[0]] = e
res = {c[0]: [] for c in cfgs}
for r in range(rounds + 1):
    for c in cfgs:
        t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            ests[c[0]].predict(mlp, 2**41)
        if r > 0:
            res[c[0]].append((ctx.flops_used / 2**41, ctx.residual_wall_time_s, time.time() - t0))
        print(f"round {r} {c[0]}: util {ctx.flops_used/2**41:.4f} residual {ctx.residual_wall_time_s:.2f}s wall {time.time()-t0:.0f}s", flush=True)
for c in cfgs:
    a = np.array(res[c[0]])
    print(f"[{c[0]}] steady: util {a[:,0].mean():.4f}  residual mean {a[:,1].mean():.2f}s (min {a[:,1].min():.2f})  wall {a[:,2].mean():.0f}s")
