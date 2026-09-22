"""Sensitivity map for the old-source (third-cumulant) content of the K3 chain (Fable-5 Q3 design).
Runs a patched copy of submissions/v29m8/estimator.py on one local MLP:
  ref                : full V29 (reference prediction)
  dropold_L<l>       : at layer l only, zero the confined old sources' dense legs + thin legs before the ReLU slices
  dropslot_<b>       : at every layer, zero source slot b (birth order; slot 0 = oldest) before the slices
  keeplast_<k>       : drop old contributions at every layer except the last k (C1 proposal)
Reports final-layer MSE(variant, ref) and the FLOPs of each variant. Analysis only (not submission code).
Usage (from whest-starterkit): python -m uv run python ../experiments/probes/sens_probe.py [which=all|ref,dropold,dropslot,keeplast]
"""
import os, sys, time, types, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
import flopscope as flops
import flopscope.numpy as fnp

which = (sys.argv[1] if len(sys.argv) > 1 else "all")
SRC = "../submissions/v29m8/estimator.py"
src = open(SRC, encoding="utf-8").read()
anchor = "                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,\n"
assert src.count(anchor) == 1
hook = (
"                _saved = _SENS_PRE(li, ka, k, A_st, P_st, Z_st, Zf_st)\n"
)
src = src.replace(anchor, hook + anchor)
# restore thin legs right after the call (they are transported in place next layer)
anchor2 = "                                        sb2=(fap24[:2 * kb] if kb > 0 else None), s_sb=s_sb)\n"
assert src.count(anchor2) == 1
src = src.replace(anchor2, anchor2 + "                _SENS_POST(_saved, Z_st, Zf_st)\n")

CFG = {"drop_old_layers": set(), "drop_slot": None}
def _zero(x):
    fnp.multiply(x, 0.0, out=x)
def _SENS_PRE(li, ka, k, A_st, P_st, Z_st, Zf_st):
    saved = None
    idx = []
    if li in CFG["drop_old_layers"] and ka > 0:
        idx = list(range(ka))
    if CFG["drop_slot"] is not None and CFG["drop_slot"] < k + 1:
        idx = sorted(set(idx) | {CFG["drop_slot"]})
    if not idx:
        return None
    zs = np.array(Z_st)[idx].copy() if Z_st is not None else None
    zfs = np.array(Zf_st)[idx].copy() if Zf_st is not None else None
    for s in idx:
        _zero(A_st[s]); _zero(P_st[s])
        if Z_st is not None: _zero(Z_st[s])
        if Zf_st is not None: _zero(Zf_st[s])
    return (idx, zs, zfs)
def _SENS_POST(saved, Z_st, Zf_st):
    if saved is None: return
    idx, zs, zfs = saved
    # restore thin legs for slots that are still transported (young); old slots are re-formed anyway
    for j, s in enumerate(idx):
        if CFG["drop_slot"] == s:      # a dropped birth stays dropped
            continue
        if Z_st is not None: fnp.copyto(Z_st[s], zs[j])
        if Zf_st is not None: fnp.copyto(Zf_st[s], zfs[j])

mod = types.ModuleType("est_sens"); mod.__dict__.update({"_SENS_PRE": _SENS_PRE, "_SENS_POST": _SENS_POST})
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = build_mlp(width=1024, depth=16, seed=7)

def run(tag, drop_layers=(), drop_slot=None):
    CFG["drop_old_layers"] = set(drop_layers); CFG["drop_slot"] = drop_slot
    est = mod.Estimator(); est.setup(_Ctx())
    t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = est.predict(mlp, 2**41)
    p = np.asarray(pred, dtype=np.float64)
    return p, ctx.flops_used / 2**41, time.time() - t0

results = {}
ref, u, t = run("ref"); results["ref"] = {"util": u}
print(f"[ref] util {u:.4f} ({t:.0f}s)", flush=True)
def report(tag, p, u):
    d = p[-1] - ref[-1]; mse = float(np.mean(d * d))
    results[tag] = {"util": u, "mse_vs_ref": mse}
    print(f"[{tag}] util {u:.4f} mse_vs_ref {mse:.3e}", flush=True)
    json.dump(results, open("../data/dumps/sens_results.json", "w"), indent=1)

if which in ("all", "keeplast"):
    for kk in (1, 2, 3):
        p, u, t = run(f"keeplast_{kk}", drop_layers=range(0, 16 - kk)); report(f"keeplast_{kk}", p, u)
if which in ("all", "dropold"):
    for l in range(5, 16):
        p, u, t = run(f"dropold_L{l}", drop_layers=[l]); report(f"dropold_L{l}", p, u)
if which in ("all", "dropslot"):
    for b in range(0, 11):
        p, u, t = run(f"dropslot_{b}", drop_slot=b); report(f"dropslot_{b}", p, u)
print("done")
