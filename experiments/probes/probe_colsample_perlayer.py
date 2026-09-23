"""Moonshot probe (23 Sep): PER-LAYER unbiased column sampling - the state stays exact, only each layer's slice
contraction is sampled, with an independent draw every layer.

Why this differs from every earlier truncation. Pilot F (deterministic, fixed at birth) and probe_colsample.py
(unbiased, fixed at birth) both perturb the SOURCE, so one source's error is carried coherently through every later
layer: the error accumulates like L. Here the transported legs are untouched - the mask is applied to a copy for the
slice call only and restored immediately - so each layer contributes an INDEPENDENT zero-mean error and the total
should accumulate like sqrt(L), i.e. ~L times better in MSE.

Cost model: a source-layer costs four dense products, two transports and two slice contractions. Sampling a fraction q
of the birth columns in the contractions only saves (1 - q) of the slice half, i.e. (1 - q)/2 of the source-layer.
(In a real implementation the dropped columns are simply never contracted; here we zero them, which measures the
error faithfully but not the saving - the saving is computed from the column count.)

Env: MODE (sample | topq), ALPHA (importance exponent on the energy, default 0.5), PFLOOR, SEEDS, LAYERWISE (1 = redraw
every layer, 0 = draw once at birth).
Usage (from whest-starterkit):
    python -m uv run python ../experiments/probes/probe_colsample_perlayer.py "1.0,1.0,0.7" "1.0,0.7,0.5"
"""
import os, sys, types, json, time
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
import flopscope as flops
import flopscope.numpy as fnp

SRC = "../submissions/v31ang/estimator.py"
src = open(SRC, encoding="utf-8").read()
a1 = "                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,\n"
a2 = "                                        sb2=(fap24[:2 * kb] if kb > 0 else None), s_sb=s_sb)\n"
assert src.count(a1) == 1 and src.count(a2) == 1
src = src.replace(a1, "                _SAVED = _MASK_PRE(li, ka, k, A_st, P_st, w2b_list)\n" + a1)
src = src.replace(a2, a2 + "                _MASK_POST(_SAVED, A_st, P_st)\n")

MODE = os.environ.get("MODE", "sample")
ALPHA = float(os.environ.get("ALPHA", "0.5"))
PFLOOR = float(os.environ.get("PFLOOR", "0.1"))
LAYERWISE = os.environ.get("LAYERWISE", "1") == "1"
CFG = {"q": (1.0, 1.0, 1.0), "birth": {}, "plan": {}, "rng": np.random.default_rng(0), "cols": 0, "tot": 0}

def _plan(e, q, n):
    if MODE == "topq":
        keep = np.zeros(n, bool); keep[np.argsort(e)[::-1][:int(round(q * n))]] = True
        return keep, np.ones(n)
    w = np.power(np.maximum(e, 0.0), ALPHA)
    if w.sum() <= 0: w = np.ones(n)
    p = np.clip(w / w.sum() * (q * n), PFLOOR, 1.0)
    for _ in range(50):
        free = p < 1.0
        d = q * n - p.sum()
        if abs(d) < 1e-9 or not free.any(): break
        p[free] = np.clip(p[free] * (1 + d / max(p[free].sum(), 1e-12)), PFLOOR, 1.0)
    keep = CFG["rng"].random(n) < p
    scale = np.ones(n); scale[keep] = np.power(1.0 / np.maximum(p[keep], 1e-12), 1.0 / 3.0)
    return keep, scale

def _MASK_PRE(li, ka, k, A_st, P_st, w2b_list):
    saved = []
    for s in range(k + 1):
        if s not in CFG["birth"]: CFG["birth"][s] = li
        b = CFG["birth"][s]
        q = CFG["q"][0] if b <= 5 else (CFG["q"][1] if b <= 9 else CFG["q"][2])
        if q >= 1.0: continue
        A = np.asarray(A_st[s], dtype=np.float64); P = np.asarray(P_st[s], dtype=np.float64)
        w2 = np.asarray(w2b_list[s], dtype=np.float64)
        if LAYERWISE or s not in CFG["plan"]:
            e = (w2 ** 2) * np.sum(A * A, axis=0) * np.sum(P * P, axis=0)
            CFG["plan"][s] = _plan(e, q, A.shape[1])
        keep, scale = CFG["plan"][s]
        CFG["cols"] += int(keep.sum()); CFG["tot"] += keep.size
        saved.append((s, A.astype(np.float32), P.astype(np.float32)))     # exact legs, restored after the call
        v = fnp.asarray((keep * scale).astype(np.float32))
        fnp.multiply(A_st[s], v[None, :], out=A_st[s])
        fnp.multiply(P_st[s], v[None, :], out=P_st[s])
    return saved

def _MASK_POST(saved, A_st, P_st):
    for s, A, P in saved:                                                  # the state is left exact
        fnp.copyto(A_st[s], fnp.asarray(A)); fnp.copyto(P_st[s], fnp.asarray(P))

mod = types.ModuleType("est_cspl"); mod.__dict__.update({"_MASK_PRE": _MASK_PRE, "_MASK_POST": _MASK_POST})
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = build_mlp(width=1024, depth=16, seed=7)

def run(q, seed=0):
    CFG.update({"q": q, "birth": {}, "plan": {}, "rng": np.random.default_rng(seed), "cols": 0, "tot": 0})
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = est.predict(mlp, 2**41)
    frac = CFG["cols"] / max(CFG["tot"], 1)
    return np.asarray(pred, dtype=np.float64), ctx.flops_used / 2**41, frac, time.time() - t0

ref, u0, _, t = run((1.0, 1.0, 1.0)); print(f"[ref] util {u0:.4f} ({t:.0f}s)", flush=True)
NS = int(os.environ.get("SEEDS", "2"))
res = {}
for arg in sys.argv[1:]:
    q = tuple(float(x) for x in arg.split(",")); dr = []
    for sd in range(NS):
        p, u, frac, t = run(q, sd)
        d = p[-1] - ref[-1]; dr.append(float(np.mean(d * d)))
        print(f"  [{MODE} layerwise={int(LAYERWISE)} q={arg} seed={sd}] drift {dr[-1]:.3e}  kept fraction {frac:.3f}  ({t:.0f}s)", flush=True)
    res[arg] = {"mode": MODE, "layerwise": LAYERWISE, "alpha": ALPHA, "drift_mean": float(np.mean(dr)), "drifts": dr}
    print(f"[{MODE} layerwise={int(LAYERWISE)} q={arg}] mean drift {np.mean(dr):.3e}", flush=True)
json.dump(res, open(f"../results/summaries/colsample_perlayer_{MODE}_{int(LAYERWISE)}.json", "w"), indent=1)
