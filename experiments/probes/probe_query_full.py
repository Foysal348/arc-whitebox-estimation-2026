"""Pilot C: compressibility of the CONTRACTED old-source query family (GPT-5.6 / Astra convergent Q3).
Step 1 (this script, run from whest-starterkit): run V29m8 on one local MLP and dump, at chosen layers, the shared
basis Qc (n x r), the dense old legs A_s, P_s (n x n), the per-source birth coefficients w2_s (n,), for tier-1 old sources.
Step 2 (probe_query_analyze.py): offline numpy analysis of U = sum_s (A_s o P_s) diag(w2_s) FA_s^T (n x r) against
cheap surrogates (isotropic trace, diagonal Hermite, TensorSketch of width d).
Usage: python -m uv run python ../experiments/probes/probe_query.py 10,14
"""
import os, sys, types
import numpy as np
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "10,14").split(",")]
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
import flopscope as flops

SRC = "../submissions/v29m8/estimator.py"
src = open(SRC, encoding="utf-8").read()
anchor = ('                legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]\n'
          '                legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]\n'
          '            else:\n')
assert src.count(anchor) == 1
src = src.replace(anchor, '                _DUMP(li, legs4["AP1"], ka, kb, k, Qc, w2b_list, None, s_list, e_list, Z_st, L_st)\n' + anchor)

def _DUMP(li, legs4, ka, kb, k, Qc, w2b_list, U2, s_list=None, e_list=None, Z_st=None, L_st=None):
    if li not in LAYERS or ka == 0:
        return
    arr = np.asarray(legs4)  # (2*(L-1), 1, n, n): slot 2s = A_s, 2s+1 = P_s
    out = {"ka": ka, "kb": kb, "k": k, "Qc": np.asarray(Qc, dtype=np.float64)}
    if U2 is not None:
        out["U2"] = np.asarray(U2, dtype=np.float64)
    for s in range(ka):
        out[f"A{s}"] = arr[2 * s, 0].astype(np.float32)
        out[f"P{s}"] = arr[2 * s + 1, 0].astype(np.float32)
        out[f"w2_{s}"] = np.asarray(w2b_list[s], dtype=np.float64)
        if s_list is not None and s < len(s_list):
            out[f"s_{s}"] = np.asarray(s_list[s], dtype=np.float64); out[f"e_{s}"] = np.asarray(e_list[s], dtype=np.float64)
            out[f"Z{s}"] = np.asarray(Z_st[s], dtype=np.float32); out[f"L{s}"] = np.asarray(L_st[s], dtype=np.float32)
    np.savez_compressed(f"../data/dumps/query_dump_full_L{li}.npz", **out)
    print(f"dumped layer {li}: ka={ka} (tier2 {kb}), k={k}, Qc {out['Qc'].shape}", flush=True)

mod = types.ModuleType("est_dump"); mod.__dict__["_DUMP"] = _DUMP
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
est = mod.Estimator(); est.setup(_Ctx())
mlp = build_mlp(width=1024, depth=16, seed=7)
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
print("done")
