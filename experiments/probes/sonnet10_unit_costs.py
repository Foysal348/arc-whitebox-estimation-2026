"""Round 10 (Sonnet, 24 Sep): unit FLOP cost of one dense n x n x n Strassen-level-5 product
and of the batched 'hub' kernel (sum_k X_k Y_k^T), as actually metered by flopscope on the
shipped v40fb _Strassen class -- needed to convert the young-tier operation COUNTS (from
sonnet10_young_tier_count.py) into u totals and to check whether batching many same-size
products into one Strassen family changes the per-matrix cost (the "batching = same FLOPs"
claim in the task background).

Usage (from whest-starterkit): python -m uv run python ../experiments/probes/sonnet10_unit_costs.py
"""
import os, sys, types, json
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

SRC = os.path.abspath("../submissions/v40fb/estimator.py")
code = open(SRC, encoding="utf-8").read()
mod = types.ModuleType("est_unit_costs")
mod.__file__ = SRC
exec(compile(code, SRC, "exec"), mod.__dict__)

n = 1024
f32 = np.float32
out = {"n": n, "mm_by_level": {}, "hub_by_level_k1": {}, "mm_batch_scaling": {}, "hub_batch_scaling": {}}

smm = mod._Strassen(f32, 16)

# 1) mm / hub cost vs Strassen level, single (1,1,n,n) product
X = fnp.zeros((1, 1, n, n), dtype=f32); Y = fnp.zeros((1, 1, n, n), dtype=f32); O = fnp.zeros((1, 1, n, n), dtype=f32)
for lev in range(0, 6):
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        smm.mm(X, Y, O, lev)
    out["mm_by_level"][lev] = ctx.flops_used / 2**31

Xh = fnp.zeros((1, 1, n, n), dtype=f32); Yh = fnp.zeros((1, 1, n, n), dtype=f32); Oh = fnp.zeros((1, n, n), dtype=f32)
for lev in range(0, 6):
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        smm.hub(Xh, Yh, Oh, lev)
    out["hub_by_level_k1"][lev] = ctx.flops_used / 2**31

# 2) mm batch scaling at level 5 (does batching many (n,n) products in ONE Strassen family
#    change the per-matrix cost?)
lev = 5
for P in (1, 2, 4, 6, 8, 10, 12):
    Xb = fnp.zeros((1, P, n, n), dtype=f32); Yb = fnp.zeros((1, P, n, n), dtype=f32); Ob = fnp.zeros((1, P, n, n), dtype=f32)
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        smm.mm(Xb, Yb, Ob, lev)
    u = ctx.flops_used / 2**31
    out["mm_batch_scaling"][P] = {"total_u": u, "per_matrix_u": u / P}

# 3) hub batch scaling at level 5 (sum_k X_k Y_k^T, k sources)
for k in (1, 2, 3, 4, 8):
    Xk = fnp.zeros((k, 1, n, n), dtype=f32); Yk = fnp.zeros((k, 1, n, n), dtype=f32); Ok = fnp.zeros((1, n, n), dtype=f32)
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        smm.hub(Xk, Yk, Ok, lev)
    u = ctx.flops_used / 2**31
    out["hub_batch_scaling"][k] = {"total_u": u, "per_source_u": u / k}

print(json.dumps(out, indent=1))
os.makedirs("../results/summaries", exist_ok=True)
json.dump(out, open("../results/summaries/sonnet10_unit_costs.json", "w"), indent=1)
