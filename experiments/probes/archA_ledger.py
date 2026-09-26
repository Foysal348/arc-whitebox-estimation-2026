"""Phase A costing (26 Sep 2026): the metered ledger of the current best chain (submissions/v44rfb8) and the unit prices
used to cost every Phase-A candidate at n = 1024, L = 16.

(1) Per-call-site ledger of v44rfb8 (same wrapper as m1_matmul_ledger.py): every metered matmul/einsum attributed to its
    calling source line; the non-matmul remainder (Strassen additions, elementwise, qr, stats) is total - recorded.
    Two predict() calls on the same process: the ledger is taken on the SECOND (steady-state, STRASSEN_FIRST no longer
    applies; this is what the grader bills for all but the first MLP of a worker).
(2) Unit prices, metered with flopscope on the chain's own Strassen class: a plain dense 1024^3 product, the same product
    through _Strassen.mm at level 5 (MIN 8), the block-symmetric sandwich _sym_product (W C W^T), a (1024 x 1024) x (1024 x r)
    product for r = 384 / 224 / 32, a (1024 x 384) QR, and the n^2 elementwise op.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/archA_ledger.py
Writes ../results/summaries/archA_ledger_v44rfb8.json
"""
import os, sys, types, time, json, traceback, collections, linecache
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from local_engine import build_mlp

SRC = os.environ.get("EST", "../submissions/v44rfb8/estimator.py")
mod = types.ModuleType("est_led"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
CTX = {"ctx": None}
LED = collections.defaultdict(lambda: [0.0, 0, collections.Counter()])
_orig = {nm: getattr(fnp, nm) for nm in ("matmul", "einsum")}
def wrap(name, fn):
    def w(*a, **k):
        c = CTX["ctx"]; before = c.flops_used if c is not None else 0
        out = fn(*a, **k)
        after = c.flops_used if c is not None else 0
        st = traceback.extract_stack(limit=16)
        INNER = {"_leaf_mm", "mm", "hub", "_leaf_hub", "prod", "_combos", "_assemble", "_sym_product", "_hub2"}
        fr = next((f for f in reversed(st[:-1]) if f.filename == SRC and f.name not in INNER), None)
        key = (name, fr.lineno if fr else -1, fr.name if fr else "?")
        shp = " x ".join(str(tuple(getattr(x, "shape", ()))) for x in a[:3] if hasattr(x, "shape"))
        LED[key][0] += after - before; LED[key][1] += 1; LED[key][2][shp] += 1
        return out
    return w
for nm, fn in _orig.items():
    setattr(fnp, nm, wrap(nm, fn))
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
U = 2.0 ** 31
mlp = build_mlp(width=1024, depth=16, seed=7)
est = mod.Estimator(); est.setup(_Ctx())
tot = {}
for call in (1, 2):
    LED.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        CTX["ctx"] = ctx
        est.predict(mlp, 2**41)
    CTX["ctx"] = None
    tot[call] = ctx.flops_used
    print(f"predict call {call}: {time.time()-t0:.0f}s, total {ctx.flops_used/U:.2f} u = util {ctx.flops_used/2**41:.4f}", flush=True)
rec = sum(v[0] for v in LED.values())
rows = sorted(LED.items(), key=lambda kv: -kv[1][0]); out_rows = []
for (nm, ln, fnname), (fl, n_, shp) in rows[:45]:
    txt = linecache.getline(SRC, ln).strip()[:100] if ln > 0 else "?"
    top = shp.most_common(1)[0][0] if shp else ""
    print(f"{fl/U:8.2f} u {fl/tot[2]*100:5.1f}%  {nm:6s} {fnname[:12]:12s} L{ln:<5d} calls {n_:4d}  {top[:46]:46s} | {txt}")
    out_rows.append({"op": nm, "line": ln, "u": fl / U, "calls": n_, "shapes": dict(shp.most_common(4)), "code": txt})
print(f"recorded matmul+einsum {rec/U:.2f} u; remainder (Strassen additions, elementwise, qr, stats) {(tot[2]-rec)/U:.2f} u")

# ---- unit prices ----
for nm, fn in _orig.items():
    setattr(fnp, nm, fn)
f32 = fnp.float32; n = 1024
rng = np.random.default_rng(0)
A = fnp.asarray(rng.standard_normal((n, n)).astype(np.float32)); B = fnp.asarray(rng.standard_normal((n, n)).astype(np.float32))
Csym = flops.as_symmetric(fnp.asarray((lambda M: (M + M.T) / 2)(rng.standard_normal((n, n))).astype(np.float32)), symmetry=(0, 1))
def meter(f):
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as c:
        f()
    return c.flops_used / U
S = est._smm
price = {}
price["plain_matmul_1024^3"] = meter(lambda: fnp.matmul(A, B))
outb = fnp.empty((1, 1, n, n), dtype=f32)
price["strassen_L5_1024^3"] = meter(lambda: S.mm(A[None, None], B[None, None], outb, S.level(n, n, n, 5)))
price["sym_sandwich_WCWt (2 products)"] = meter(lambda: est._sym_product(fnp.matmul(A, Csym), A.T, n, fnp.empty((n, n), dtype=f32), 5))
price["sym_sandwich_WCWt_second_half_only"] = price["sym_sandwich_WCWt (2 products)"] - price["plain_matmul_1024^3"]
for r in (384, 224, 64, 32, 8):
    Br = fnp.asarray(rng.standard_normal((n, r)).astype(np.float32)); outr = fnp.empty((1, 1, n, r), dtype=f32)
    price[f"strassen_(1024x1024)x(1024x{r})"] = meter(lambda: S.mm(A[None, None], Br[None, None], outr, S.level(n, n, r, 5)))
    price[f"plain_(1024x1024)x(1024x{r})"] = meter(lambda: fnp.matmul(A, Br))
price["qr_1024x384"] = meter(lambda: fnp.linalg.qr(fnp.asarray(rng.standard_normal((n, 384)).astype(np.float32))))
price["elementwise_mult_1024^2"] = meter(lambda: fnp.multiply(A, B))
price["einsum_rowdot_kij,kij->i (k=1)"] = meter(lambda: fnp.einsum("kij,kij->i", A[None], B[None]))
for k, v in price.items():
    print(f"  price {k:42s} {v:.4f} u")
json.dump({"estimator": SRC, "total_u_call1": tot[1] / U, "total_u_call2": tot[2] / U, "util_call2": tot[2] / 2**41,
           "recorded_matmul_einsum_u": rec / U, "remainder_u": (tot[2] - rec) / U, "rows": out_rows, "unit_prices_u": price},
          open("../results/summaries/archA_ledger_v44rfb8.json", "w"), indent=1)
