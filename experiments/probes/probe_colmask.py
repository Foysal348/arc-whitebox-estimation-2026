"""Pilot F: end-to-end effect of birth-column screening. For each source, when it first appears (age 1), rank its
birth columns by e_j = w2_j^2 |A_j|^2 |P_j|^2 and FIX a keep-mask of the top-q fraction (q depends on the birth layer);
at every later layer zero the dropped columns of its dense legs before the slice contractions (equivalent to
never storing them).  Metric: final-layer MSE vs the full chain on the same MLP (chain's own error 2.1e-8).
Usage (from whest-starterkit): python -m uv run python ../experiments/probes/probe_colmask.py "q_early,q_mid,q_deep" ...   e.g. 1.0,0.8,0.65
  q_early: births at layers <= 5 ; q_mid: births 6..9 ; q_deep: births >= 10
"""
import os, sys, types, json, time
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
import flopscope as flops
import flopscope.numpy as fnp

SRC = "../submissions/v29m8/estimator.py"
src = open(SRC, encoding="utf-8").read()
anchor = "                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,\n"
assert src.count(anchor) == 1
src = src.replace(anchor, "                _MASK(li, ka, k, A_st, P_st, w2b_list)\n" + anchor)

CFG = {"q": (1.0, 1.0, 1.0), "masks": {}, "birth": {}}
def _MASK(li, ka, k, A_st, P_st, w2b_list):
    nslots = k + 1
    for s in range(nslots):
        if s not in CFG["masks"] or os.environ.get("RERANK") == "1":
            # first appearance -> this is its birth layer (slot index = birth order); q by birth layer li
            q = CFG["q"][0] if li <= 5 else (CFG["q"][1] if li <= 9 else CFG["q"][2])
            if s not in CFG["birth"]: CFG["birth"][s] = li
            q = CFG["q"][0] if CFG["birth"][s] <= 5 else (CFG["q"][1] if CFG["birth"][s] <= 9 else CFG["q"][2])
            if q >= 1.0:
                CFG["masks"][s] = None; continue
            A = np.asarray(A_st[s], dtype=np.float64); P = np.asarray(P_st[s], dtype=np.float64); w2 = np.asarray(w2b_list[s], dtype=np.float64)
            e = (w2 ** 2) * np.sum(A * A, axis=0) * np.sum(P * P, axis=0)
            n = A.shape[1]; keep = np.zeros(n, bool); keep[np.argsort(e)[::-1][:int(round(q * n))]] = True
            CFG["masks"][s] = keep
        m = CFG["masks"][s]
        if m is None: continue
        drop = fnp.asarray((~m).astype(np.float32))            # 1 on dropped columns
        keepv = fnp.asarray(m.astype(np.float32))
        fnp.multiply(A_st[s], keepv[None, :], out=A_st[s])      # zero dropped columns (all contractions for those j vanish)
        fnp.multiply(P_st[s], keepv[None, :], out=P_st[s])

mod = types.ModuleType("est_mask"); mod.__dict__["_MASK"] = _MASK
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = build_mlp(width=1024, depth=16, seed=7)

def run(q):
    CFG["q"] = q; CFG["masks"] = {}; CFG["birth"] = {}
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = est.predict(mlp, 2**41)
    return np.asarray(pred, dtype=np.float64), time.time() - t0

ref, t = run((1.0, 1.0, 1.0)); print(f"[ref] {t:.0f}s", flush=True)
res = {}
for arg in sys.argv[1:]:
    q = tuple(float(x) for x in arg.split(","))
    p, t = run(q); d = p[-1] - ref[-1]; mse = float(np.mean(d * d))
    kept = {s: (None if m is None else int(m.sum())) for s, m in CFG["masks"].items()}
    res[arg] = mse
    print(f"[q={arg}] MSE vs full chain {mse:.3e}   ({t:.0f}s)  kept columns per slot: {kept}", flush=True)
json.dump(res, open("../data/dumps/colmask_results.json", "w"), indent=1)
