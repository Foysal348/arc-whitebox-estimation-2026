"""Moonshot A1 (24 Sep): where do the 220 u of matmul go? A per-call-site ledger of the shipped chain.

Wraps flopscope.numpy.matmul / einsum with a recorder that attributes each call's metered FLOPs (from the BudgetContext
counter before/after the call) to the calling source line of the estimator and to the operand shapes. Output: a table by
source line (function + line + the code text), sorted by FLOPs, and a per-layer total.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/m1_matmul_ledger.py
"""
import os, sys, types, time, json, traceback, collections, linecache
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from local_engine import build_mlp

SRC = os.environ.get("EST", "../submissions/v40fb/estimator.py")
code = open(SRC, encoding="utf-8").read()
mod = types.ModuleType("est_led")
exec(compile(code, SRC, "exec"), mod.__dict__)
CTX = {"ctx": None}
LED = collections.defaultdict(lambda: [0.0, 0, collections.Counter()])

def wrap(name, fn):
    def w(*a, **k):
        c = CTX["ctx"]; before = c.flops_used if c is not None else 0
        out = fn(*a, **k)
        after = c.flops_used if c is not None else 0
        st = traceback.extract_stack(limit=14)
        INNER = {"_leaf_mm", "mm", "hub", "_leaf_hub", "prod", "_combos", "_assemble", "_sym_product", "_hub2"}
        fr = next((f for f in reversed(st[:-1]) if f.filename == SRC and f.name not in INNER), None)
        key = (name, fr.lineno if fr else -1, fr.name if fr else "?")
        shp = " x ".join(str(tuple(getattr(x, "shape", ()))) for x in a[:3] if hasattr(x, "shape"))
        LED[key][0] += after - before; LED[key][1] += 1; LED[key][2][shp] += 1
        return out
    return w
for nm in ("matmul", "einsum"):
    mod.__dict__["fnp"].__dict__  # noqa
setattr_targets = [("matmul", fnp.matmul), ("einsum", fnp.einsum)]
for nm, fn in setattr_targets:
    setattr(fnp, nm, wrap(nm, fn))
# the estimator bound fnp at import: patch its module reference too (same module object, attributes now wrapped)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = build_mlp(width=1024, depth=16, seed=7)
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    CTX["ctx"] = ctx
    est.predict(mlp, 2**41)
tot = ctx.flops_used
print(f"run {time.time()-t0:.0f}s, total {tot/2**31:.2f} u; recorded matmul+einsum {sum(v[0] for v in LED.values())/2**31:.2f} u")
rows = sorted(LED.items(), key=lambda kv: -kv[1][0])
out = []
for (nm, ln, fnname), (fl, n, shp) in rows[:40]:
    txt = linecache.getline(SRC, ln).strip()[:95] if ln > 0 else "?"
    top = shp.most_common(1)[0][0] if shp else ""
    print(f"{fl/2**31:8.2f} u {fl/tot*100:5.1f}%  {nm:6s} {fnname[:12]:12s} L{ln:<5d} calls {n:4d}  {top[:44]:44s} | {txt}")
    out.append({"op": nm, "line": ln, "u": fl / 2**31, "calls": n, "shapes": dict(shp.most_common(5)), "code": txt})
json.dump(out, open("../results/summaries/m1_matmul_ledger_callers.json", "w"), indent=1)
