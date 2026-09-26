"""FLOP profile of one predict() on public MLP #0: op_log aggregated by (op, shapes) and by op, largest first.
Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/flop_profile.py <estimator.py> [NAME=VALUE,...]
"""
import os, sys, types, glob, collections
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC = sys.argv[1]
if len(sys.argv) > 2:
    os.environ.update(dict(kv.split("=", 1) for kv in sys.argv[2].split(",") if kv))
W = np.load(glob.glob("../data/kaggle_out/kaggle_out_k4pub/k4teach/weights_seed*.npz")[0])["W"].astype(np.float32)
mlp = MLP(width=W.shape[1], depth=W.shape[0], weights=[fnp.array(W[l]) for l in range(W.shape[0])], seed=0, name="mlp0")


class _Ctx: seed = 0; width = W.shape[1]; depth = W.shape[0]; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


mod = types.ModuleType("est_var"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
est = mod.Estimator(); est.setup(_Ctx())
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
U = 2.0 ** 31
tot = ctx.flops_used / U
by_shape = collections.Counter(); by_op = collections.Counter(); cnt = collections.Counter()
for r in ctx.op_log:
    key = (r.op_name, r.subscripts or "", str(tuple(tuple(s) if hasattr(s, "__len__") else s for s in r.shapes))[:90])
    by_shape[key] += r.flop_cost; cnt[key] += 1; by_op[r.op_name] += r.flop_cost
print(f"total {tot:.2f} u  (util {ctx.flops_used / 2**41:.4f}), {len(ctx.op_log)} ops")
print("\nby op:")
for k, v in by_op.most_common(15):
    print(f"  {v / U:8.2f} u  {100 * v / ctx.flops_used:5.1f}%  {k}")
print("\nby (op, subscripts, shapes):")
for k, v in by_shape.most_common(45):
    print(f"  {v / U:7.2f} u  x{cnt[k]:<4d} {k[0]:12s} {k[1]:18s} {k[2]}")
