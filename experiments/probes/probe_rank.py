"""Question A pilot: numerical rank of the Hadamard products of the K3 source legs.
Runs one local MLP through a probed copy of submissions/v29m8/estimator.py and, at chosen layers,
takes the dense legs (A_s, P_s) of every live source and computes the singular spectrum of
A*P, A*A, P*P (elementwise). Reports how many singular directions hold 90/99/99.9% of the energy,
separately for old (shared-basis, rank<=384) and young (dense) sources. Analysis only: numpy SVD
outside flopscope accounting (this is NOT submission code).
Usage (from whest-starterkit): python -m uv run python ../experiments/probes/probe_rank.py [layers=6,10,14] [max_src=3]
"""
import os, sys, time, types, importlib.util
import numpy as np
os.environ.setdefault("PYTHONWARNINGS", "ignore")
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "6,10,14").split(",")]
MAX_SRC = int(sys.argv[2]) if len(sys.argv) > 2 else 3
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
import flopscope as flops

SRC = "../submissions/v29m8/estimator.py"
src = open(SRC, encoding="utf-8").read()
anchor = ('                legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]\n'
          '                legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]\n'
          '            else:\n')
assert src.count(anchor) == 1, "anchor not unique"
probe_line = '                _PROBE(li, legs4["AP1"], ka, kb, k)\n'
src = src.replace(anchor, probe_line + anchor)

REPORT = {}
def energy_ranks(s):
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return {q: int(np.searchsorted(e, q) + 1) for q in (0.9, 0.99, 0.999)}

def _PROBE(li, legs4, ka, kb, k):
    if li not in LAYERS or k == 0:
        return
    t0 = time.time()
    arr = np.asarray(legs4)  # (2*(L-1), 1, n, n): slot 2s = A_s, 2s+1 = P_s ; [0:2kb] tier-2 old, [2kb:2ka] tier-1 old, [2ka:2k] young
    rows = []
    groups = [("old2", 0, kb), ("old1", kb, ka), ("young", ka, k)]
    for name, lo, hi in groups:
        for s in range(lo, min(hi, lo + MAX_SRC)):
            A = arr[2 * s, 0].astype(np.float64); P = arr[2 * s + 1, 0].astype(np.float64)
            for pname, M in (("A", A), ("P", P), ("A*P", A * P), ("A*A", A * A), ("P*P", P * P)):
                sv = np.linalg.svd(M, compute_uv=False)
                r = energy_ranks(sv)
                rows.append((name, s, pname, r[0.9], r[0.99], r[0.999], float(sv[0]), float(np.linalg.norm(M))))
    REPORT[li] = rows
    print(f"\n== layer {li}: ka={ka} old (tier2 {kb}), young {k-ka}; probe {time.time()-t0:.1f}s", flush=True)
    print(f"{'group':6s} {'src':>3s} {'matrix':6s} {'r90':>5s} {'r99':>5s} {'r999':>5s} {'top sv':>10s} {'fro':>10s}")
    for g, s, pn, r90, r99, r999, top, fro in rows:
        print(f"{g:6s} {s:3d} {pn:6s} {r90:5d} {r99:5d} {r999:5d} {top:10.3e} {fro:10.3e}", flush=True)

mod = types.ModuleType("est_probe"); mod.__dict__["_PROBE"] = _PROBE
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
est = mod.Estimator(); est.setup(_Ctx())
mlp = build_mlp(width=1024, depth=16, seed=7)
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
np.save("../data/dumps/probe_rank_report.npy", np.array([REPORT], dtype=object), allow_pickle=True)
print("\ndone; util", ctx.flops_used / 2**41)
