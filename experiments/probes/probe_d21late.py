"""Which part of the OLD sources' slices matters at late layers: the (2,1) slice D21 (n^3 per source) or the diagonal D3 (n^2)?
At the chosen layer the dslices are computed twice (all sources; young sources only, old legs zeroed) and mixed:
   dropD21old_L<l> : D3 from all sources, D21 from the young only   (old D21 removed at layer l)
   dropD3old_L<l>  : D21 from all sources, D3 from the young only   (old D3 removed at layer l)
Metric: final-layer MSE against the full chain. If dropD21old is cheap in accuracy at late layers, the old tier's slice
products can be skipped there (V19's D3-only mode extended to old sources), saving 2 of the 4 products per old source-layer.
Usage (from whest-starterkit): python -m uv run python ../experiments/probes/probe_d21late.py "12,13,14"
"""
import os, sys, time, types, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
import flopscope as flops
import flopscope.numpy as fnp
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "12,13,14").split(",")]
SRC = "../submissions/v29m8/estimator.py"
src = open(SRC, encoding="utf-8").read()
a1 = "                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,\n"
a2 = "                                        sb2=(fap24[:2 * kb] if kb > 0 else None), s_sb=s_sb)\n"
assert src.count(a1) == 1 and src.count(a2) == 1
i1 = src.index(a1); i2 = src.index(a2) + len(a2)
call = src[i1:i2]
call2 = call.replace("D3, D21 = self._dslices(", "D3y, D21y = self._dslices(")
patched = (call +
           "                _f = _KEEP(li, ka, D3, D21)\n"
           "                if _f is not None:\n"
           "                    _sv = _ZERO_OLD(ka, A_st, P_st, Z_st, Zf_st)\n" +
           "\n".join("    " + ln if ln.strip() else ln for ln in call2.splitlines()) + "\n"
           "                    _RESTORE(_sv, Z_st, Zf_st)\n"
           "                    D3, D21 = _MIX(_f, D3y, D21y)\n")
src = src[:i1] + patched + src[i2:]
CFG = {"layer": None, "mode": None}
def _KEEP(li, ka, D3, D21):
    if li != CFG["layer"]: return None
    return (np.array(D3, dtype=np.float32).copy(), np.array(D21, dtype=np.float32).copy())
def _ZERO_OLD(ka, A_st, P_st, Z_st, Zf_st):
    idx = list(range(ka))
    zs = np.array(Z_st)[idx].copy() if Z_st is not None else None
    zfs = np.array(Zf_st)[idx].copy() if Zf_st is not None else None
    for s in idx:
        fnp.multiply(A_st[s], 0.0, out=A_st[s]); fnp.multiply(P_st[s], 0.0, out=P_st[s])
        if Z_st is not None: fnp.multiply(Z_st[s], 0.0, out=Z_st[s])
        if Zf_st is not None: fnp.multiply(Zf_st[s], 0.0, out=Zf_st[s])
    return (idx, zs, zfs)
def _RESTORE(sv, Z_st, Zf_st):
    idx, zs, zfs = sv
    for j, s in enumerate(idx):
        if Z_st is not None: fnp.copyto(Z_st[s], zs[j])
        if Zf_st is not None: fnp.copyto(Zf_st[s], zfs[j])
def _MIX(f, D3y, D21y):
    D3f, D21f = f
    if CFG["mode"] == "dropD21old":
        return fnp.asarray(D3f), D21y
    if CFG["mode"] == "dropD21all":
        return fnp.asarray(D3f), fnp.asarray(np.zeros_like(D21f))
    return D3y, fnp.asarray(D21f)
mod = types.ModuleType("est_d21"); mod.__dict__.update({"_KEEP": _KEEP, "_ZERO_OLD": _ZERO_OLD, "_RESTORE": _RESTORE, "_MIX": _MIX})
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = build_mlp(width=1024, depth=16, seed=7)
def run(layer, mode):
    CFG["layer"] = layer; CFG["mode"] = mode
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
        p = est.predict(mlp, 2**41)
    return np.asarray(p, dtype=np.float64)[-1], time.time() - t0
ref, t = run(None, None); print(f"[ref] {t:.0f}s", flush=True)
res = {}
for l in LAYERS:
    for mode in (os.environ.get("MODES", "dropD21old,dropD3old").split(",")):
        p, t = run(l, mode); mse = float(np.mean((p - ref) ** 2)); res[f"{mode}_L{l}"] = mse
        print(f"[{mode}_L{l}] MSE vs full chain {mse:.3e}  ({t:.0f}s)", flush=True)
json.dump(res, open("../data/dumps/d21late_results_" + os.environ.get("MODES", "default").replace(",", "_") + ".json", "w"), indent=1)
