"""D2 memory census: where do the pooled buffers go? Runs a V29 variant (env knobs) twice on the local seed-7 MLP,
then lists the largest pooled buffers (estimator _Pool + Strassen pools) and the totals.
Usage (from whest-starterkit): python -m uv run python ../experiments/chain/pool_census.py <tag> KEY=VAL ...
"""
import os, sys, time, importlib.util
tag = sys.argv[1]
for kv in sys.argv[2:]:
    k, v = kv.split("="); os.environ[k] = v
import numpy as np, psutil
import flopscope as flops
import flopscope.numpy as fnp
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
spec = importlib.util.spec_from_file_location("est", os.environ.get("EST", "../submissions/v29m8f5/estimator.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
est = m.Estimator(); est.setup(_Ctx())
mlp = build_mlp(width=1024, depth=16, seed=7)
for call in range(2):
    t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = est.predict(mlp, 2**41)
    mi = psutil.Process().memory_info()
    print(f"[{tag}] call {call}: util {ctx.flops_used/2**41:.4f} wall {time.time()-t0:.0f}s residual {ctx.residual_wall_time_s:.2f}s peakCommit {getattr(mi,'peak_pagefile',0)/2**30:.2f} GB", flush=True)

def nbytes(x):
    if isinstance(x, tuple): x = x[0]
    a = np.asarray(x); return a.nbytes
rows = []
for key, b in est._pool.bufs.items():
    rows.append(("POOL", key, nbytes(b)))
smm = getattr(est, "_smm", None)
if smm is not None:
    for key, b in smm.pool.items():
        rows.append(("STRASSEN", key, nbytes(b)))
rows.sort(key=lambda r: -r[2])
tot = {"POOL": 0, "STRASSEN": 0}
for k, key, nb in rows: tot[k] += nb
print(f"[{tag}] totals: estimator pool {tot['POOL']/2**30:.2f} GB, Strassen pools {tot['STRASSEN']/2**30:.2f} GB, buffers {len(rows)}")
# group Strassen buffers by (kind, level P)
byP = {}
for k, key, nb in rows:
    if k == "STRASSEN":
        g = (key[0], key[1]); byP[g] = byP.get(g, 0) + nb
for g, nb in sorted(byP.items(), key=lambda t: -t[1])[:20]:
    print(f"   Strassen group kind={g[0]} P={g[1]}: {nb/2**20:.0f} MB")
for k, key, nb in rows[:25]:
    print(f"   {k} {key}: {nb/2**20:.0f} MB")
