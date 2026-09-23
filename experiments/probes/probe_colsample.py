"""Moonshot probe (23 Sep): UNBIASED column sampling of the sources, against the deterministic screening of pilot F.

Pilot F kept the top-q birth columns of every source by energy e_j = w2_j^2 |a_j|^2 |p_j|^2 and died end to end: a per-source
D21 error of ~1% became a 32% error on the final mean (drift 6.7e-9 at q = 0.7 for deep births). That amplification is the
signature of a COHERENT BIAS - the same columns are dropped at every layer, so the error adds up along the chain.

Every slice term is CUBIC in a source's columns (H = (A o P) diag(w2) A^T and its siblings are degree 3 in the birth index),
so a Horvitz-Thompson estimator is available: keep column j with probability p_j and rescale its columns by (1/p_j)^(1/3).
Then every term is an unbiased estimate of the exact one, and the error is zero-mean noise that partially cancels along the
chain instead of accumulating. Same cost as the deterministic mask at the same keep-fraction.

Modes (env MODE): topq (reproduce pilot F) | sample (unbiased, p proportional to e^ALPHA) | sample_fixed (unbiased, fixed size)
Env: ALPHA (default 0.5), SEEDS (number of independent draws to average over, default 1)
Usage (from whest-starterkit):
    python -m uv run python ../experiments/probes/probe_colsample.py "1.0,1.0,0.7" "1.0,0.9,0.75" ...
"""
import os, sys, types, json, time
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
import flopscope as flops
import flopscope.numpy as fnp

SRC = "../submissions/v31ang/estimator.py"
src = open(SRC, encoding="utf-8").read()
anchor = "                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,\n"
assert src.count(anchor) == 1
src = src.replace(anchor, "                _MASK(li, ka, k, A_st, P_st, w2b_list)\n" + anchor)

MODE = os.environ.get("MODE", "sample")
ALPHA = float(os.environ.get("ALPHA", "0.5"))
CFG = {"q": (1.0, 1.0, 1.0), "masks": {}, "birth": {}, "scaled": set(), "rng": np.random.default_rng(0)}

def _plan(e, q, n):
    """Return (keep mask, per-column scale) for one source."""
    if MODE == "topq":
        keep = np.zeros(n, bool); keep[np.argsort(e)[::-1][:int(round(q * n))]] = True
        return keep, np.ones(n)
    w = np.power(np.maximum(e, 0.0), ALPHA)
    if w.sum() <= 0: w = np.ones(n)
    p = w / w.sum() * (q * n)                      # expected keep count = q n
    PFLOOR = float(os.environ.get('PFLOOR', '0.05'))   # bound the HT variance: no column is rescaled by more than (1/PFLOOR)^(1/3)
    p = np.clip(p, PFLOOR, 1.0)
    # renormalise so that the expected count is exactly q n even after clipping
    for _ in range(50):
        free = p < 1.0
        deficit = q * n - p.sum()
        if abs(deficit) < 1e-9 or not free.any(): break
        p[free] = np.clip(p[free] * (1 + deficit / max(p[free].sum(), 1e-12)), 0.0, 1.0)
    keep = CFG["rng"].random(n) < p
    scale = np.ones(n)
    scale[keep] = np.power(1.0 / np.maximum(p[keep], 1e-12), 1.0 / 3.0)   # cubic terms -> unbiased
    return keep, scale

def _MASK(li, ka, k, A_st, P_st, w2b_list):
    for s in range(k + 1):
        if s not in CFG["masks"]:
            if s not in CFG["birth"]: CFG["birth"][s] = li
            b = CFG["birth"][s]
            q = CFG["q"][0] if b <= 5 else (CFG["q"][1] if b <= 9 else CFG["q"][2])
            if q >= 1.0:
                CFG["masks"][s] = None; continue
            A = np.asarray(A_st[s], dtype=np.float64); P = np.asarray(P_st[s], dtype=np.float64)
            w2 = np.asarray(w2b_list[s], dtype=np.float64)
            na2 = np.sum(A * A, axis=0); np2 = np.sum(P * P, axis=0)
            e = (w2 ** 2) * na2 * np2
            if os.environ.get("IMP", "energy") == "termnorm":
                # variance-optimal importance for a sum of matrices: p_j proportional to ||X_j||_F.
                # X_j = w2_j (a_j o p_j) a_j^T  =>  ||X_j||_F = |w2_j| * ||a_j o p_j|| * ||a_j||, computed exactly.
                nap = np.sqrt(np.sum((A * P) ** 2, axis=0))
                e = (np.abs(w2) * nap * np.sqrt(na2)) ** (1.0 / max(ALPHA, 1e-9)) if ALPHA > 0 else e
            CFG["masks"][s] = _plan(e, q, A.shape[1])
        m = CFG["masks"][s]
        if m is None: continue
        keep, scale = m
        # The rescaling is applied ONCE, at the source's birth: the transport is linear, so a column scaled at birth
        # stays scaled through every later layer and every cubic term remains unbiased. Applying it again at each
        # layer would compound it geometrically (s^14) - that is what blew the first version up.
        first = s not in CFG["scaled"]
        v = fnp.asarray(((keep * scale) if first else keep.astype(np.float64)).astype(np.float32))
        CFG["scaled"].add(s)
        fnp.multiply(A_st[s], v[None, :], out=A_st[s])
        fnp.multiply(P_st[s], v[None, :], out=P_st[s])

mod = types.ModuleType("est_cs"); mod.__dict__["_MASK"] = _MASK
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = build_mlp(width=1024, depth=16, seed=7)

def run(q, seed=0):
    CFG["q"] = q; CFG["masks"] = {}; CFG["birth"] = {}; CFG["scaled"] = set(); CFG["rng"] = np.random.default_rng(seed)
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = est.predict(mlp, 2**41)
    kept = sum(int(m[0].sum()) for m in CFG["masks"].values() if m is not None)
    return np.asarray(pred, dtype=np.float64), ctx.flops_used / 2**41, kept, time.time() - t0

ref, u0, _, t = run((1.0, 1.0, 1.0)); print(f"[ref] util {u0:.4f} ({t:.0f}s)", flush=True)
NS = int(os.environ.get("SEEDS", "1"))
res = {}
for arg in sys.argv[1:]:
    q = tuple(float(x) for x in arg.split(","))
    drifts = []
    for sd in range(NS):
        p, u, kept, t = run(q, sd)
        d = p[-1] - ref[-1]; drifts.append(float(np.mean(d * d)))
        print(f"  [{MODE} q={arg} seed={sd}] drift MSE {drifts[-1]:.3e}  util {u:.4f} ({u/u0-1:+.2%})  kept cols {kept}  ({t:.0f}s)", flush=True)
    res[arg] = {"mode": MODE, "alpha": ALPHA, "drift_mean": float(np.mean(drifts)), "drifts": drifts}
    print(f"[{MODE} q={arg}] mean drift {np.mean(drifts):.3e}", flush=True)
json.dump(res, open(f"../results/summaries/colsample_{MODE}.json", "w"), indent=1)
