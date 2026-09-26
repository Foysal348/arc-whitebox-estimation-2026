"""Grader-like timing probe: ONE estimator instance, consecutive predict() calls on different networks; reports per call
the wall time, the residual wall time (Python outside flopscope; Phase 2 cap 0.4 s on the grader), flopscope backend /
overhead time and the op count. The laptop is slower than the grader, so compare variants relative to each other.
Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/residual_probe.py <estimator.py> [label]
"""
import os, sys, types, glob, time, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.dirname(SRC))
nets = []
for d in ("../data/kaggle_out/kaggle_out_k4pub/k4teach", "../data/kaggle_out/kaggle_out_k4/k4teach"):
    W = np.load(glob.glob(os.path.join(d, "weights_seed*.npz"))[0])["W"].astype(np.float32)
    nets.append(MLP(width=W.shape[1], depth=W.shape[0], weights=[fnp.array(W[l]) for l in range(W.shape[0])], seed=0, name=d.split("/")[-2]))
rng = np.random.default_rng(11)
W = (rng.standard_normal((16, 1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32)
nets.append(MLP(width=1024, depth=16, weights=[fnp.array(W[l]) for l in range(16)], seed=0, name="rand"))


class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


mod = types.ModuleType("est_var"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
est = mod.Estimator(); est.setup(_Ctx())
rows = []
for i, mlp in enumerate(nets):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        p = est.predict(mlp, 2**41)
        _ = np.asarray(p)
    r = dict(call=i, net=mlp.name, util=ctx.flops_used / 2**41, wall=ctx.wall_time_s, residual=ctx.residual_wall_time_s,
             backend=ctx.flopscope_backend_time_s, overhead=ctx.flopscope_overhead_time_s, ops=len(ctx.op_log))
    rows.append(r)
    print(f"[{label}] call {i} {mlp.name:12s} util {r['util']:.4f} wall {r['wall']:.1f}s residual {r['residual']:.3f}s "
          f"backend {r['backend']:.1f}s overhead {r['overhead']:.2f}s ops {r['ops']}", flush=True)
os.makedirs("../results/summaries", exist_ok=True)
json.dump(rows, open(f"../results/summaries/residual_probe_{label}.json", "w"), indent=1)
