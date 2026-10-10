"""Call-site FLOP + op ledger of one predict() (10 Oct 2026): where do the ~190 u of the shipped V56 go?

Every public callable of flopscope.numpy (and .linalg) is wrapped. The outermost wrapped call (re-entrancy guarded) is charged
its metered FLOPs (BudgetContext.flops_used before/after) and one op, and attributed to the deepest frame of the estimator file
that is NOT inside the _Strassen / _Pool helpers, i.e. the estimator line that asked for the product. Output: a table by
estimator line (function:line + code text), by function, and by named region (nearest preceding '# ----' header or comment
block in _predict_core). Writes <out>.json.
Usage: python site_ledger.py <estimator.py> <weights_seed0.npz> [out] [NAME=VALUE,...]
"""
import os, sys, types, json, collections, linecache, inspect
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC, WPATH = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "site_ledger"
if len(sys.argv) > 4:
    os.environ.update(dict(kv.split("=", 1) for kv in sys.argv[4].split(",") if kv))
SRC_ABS = os.path.abspath(SRC)
W = np.load(WPATH)["W"].astype(np.float32)
mlp = MLP(width=W.shape[1], depth=W.shape[0], weights=[fnp.array(W[l]) for l in range(W.shape[0])], seed=0, name="mlp0")
HELPER = {"_buf", "_buf2", "_flat", "level", "_ok", "_combos", "_assemble", "mm", "_leaf_mm", "hub", "_leaf_hub", "get", "get_pair", "__init__"}
STATE = {"ctx": None, "depth": 0}
LED = collections.defaultdict(lambda: [0.0, 0])        # (func, line) -> [flops, ops]


def site():
    f = inspect.currentframe().f_back.f_back
    best = None
    while f is not None:
        if os.path.abspath(f.f_code.co_filename) == SRC_ABS and f.f_code.co_name not in HELPER:
            best = (f.f_code.co_name, f.f_lineno); break
        f = f.f_back
    return best or ("<outside>", 0)


def wrap(fn):
    def w(*a, **k):
        if STATE["depth"] or STATE["ctx"] is None:
            return fn(*a, **k)
        STATE["depth"] += 1
        c = STATE["ctx"]; before = c.flops_used
        try:
            return fn(*a, **k)
        finally:
            STATE["depth"] -= 1
            s = site(); LED[s][0] += c.flops_used - before; LED[s][1] += 1
    w.__wrapped__ = fn
    return w


for mod in (fnp, fnp.linalg):
    for nm in dir(mod):
        if nm.startswith("_"): continue
        obj = getattr(mod, nm)
        if callable(obj) and not isinstance(obj, type) and not inspect.ismodule(obj):
            try: setattr(mod, nm, wrap(obj))
            except Exception: pass


class _Ctx: seed = 0; width = W.shape[1]; depth = W.shape[0]; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


est_mod = types.ModuleType("est_led"); est_mod.__file__ = SRC_ABS
exec(compile(open(SRC, encoding="utf-8").read(), SRC_ABS, "exec"), est_mod.__dict__)
est = est_mod.Estimator(); est.setup(_Ctx())
for rep in range(2):                                   # call 1 = cold (first-predict path), call 2 = warm (what most MLPs pay)
    LED.clear()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        STATE["ctx"] = ctx
        est.predict(mlp, 2**41)
        STATE["ctx"] = None
    total = ctx.flops_used
U = 2.0 ** 31
src_lines = open(SRC, encoding="utf-8").read().splitlines()


def region(line):
    for j in range(line - 1, 0, -1):
        t = src_lines[j - 1].strip()
        if t.startswith("# ----") or t.startswith("# ===="):
            return f"L{j}: " + t.strip("# -=")[:70]
    return "<top>"


rows = sorted(((v[0], v[1], f, l) for (f, l), v in LED.items()), reverse=True)
by_func = collections.Counter(); by_reg = collections.Counter(); ops_reg = collections.Counter()
for fl, op, f, l in rows:
    by_func[f] += fl
    r = region(l) if f == "_predict_core" else f"[{f}]"
    by_reg[r] += fl; ops_reg[r] += op
print(f"TOTAL {total / U:.2f} u (util {total / 2**41:.4f}), attributed {sum(v[0] for v in LED.values()) / U:.2f} u, ops {sum(v[1] for v in LED.values())}")
print("\n== by function ==")
for f, v in by_func.most_common(): print(f"{v / U:8.2f} u  {f}")
print("\n== by region ==")
for r, v in by_reg.most_common(40): print(f"{v / U:8.2f} u  {ops_reg[r]:6d} ops  {r}")
print("\n== top 60 lines ==")
for fl, op, f, l in rows[:60]:
    print(f"{fl / U:8.3f} u {op:6d} ops  {f}:{l}  {src_lines[l - 1].strip()[:110] if l else ''}")
json.dump({"total_u": total / U, "by_function": {k: v / U for k, v in by_func.items()},
           "by_region": {k: [v / U, ops_reg[k]] for k, v in by_reg.items()},
           "lines": [[f, l, fl / U, op, src_lines[l - 1].strip()[:160] if l else ""] for fl, op, f, l in rows]},
          open(OUT + ".json", "w"), indent=1)
