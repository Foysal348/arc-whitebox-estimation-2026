"""Round 10 (Sonnet, 24 Sep): reproduce the young-tier ledger lines (transport L1019, D21 hub
L1680/L1692) with flopscope, AND simultaneously trace the exact per-layer (k, ka, extra) young-
source-count bookkeeping via sys.settrace on the shipped v40fb estimator (read only -- no edits),
so the op-count model behind the "young transport ~54u + D21 contraction ~50u = ~104u" ledger
claim can be checked against ground truth from the SAME run.

Usage (from whest-starterkit): python -m uv run python ../experiments/probes/sonnet10_young_tier_count.py
"""
import sys, os, types, json, collections, traceback
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from local_engine import build_mlp

SRC = os.path.abspath("../submissions/v40fb/estimator.py")
code = open(SRC, encoding="utf-8").read()
mod = types.ModuleType("est_young_tier_count")
mod.__file__ = SRC
exec(compile(code, SRC, "exec"), mod.__dict__)

# ---- 1) flopscope ledger, attributed to source line (same technique as m1_matmul_ledger.py) ----
CTX = {"ctx": None}
LED = collections.defaultdict(float)
fnp_mod = fnp
def wrap(fn):
    def w(*a, **k):
        c = CTX["ctx"]; before = c.flops_used if c is not None else 0
        out = fn(*a, **k)
        after = c.flops_used if c is not None else 0
        st = traceback.extract_stack(limit=14)
        INNER = {"_leaf_mm", "mm", "hub", "_leaf_hub", "prod", "_combos", "_assemble", "_sym_product", "_hub2"}
        fr = next((f for f in reversed(st[:-1]) if f.filename == SRC and f.name not in INNER), None)
        if fr is not None:
            LED[fr.lineno] += after - before
        return out
    return w
for nm in ("matmul", "einsum"):
    setattr(fnp_mod, nm, wrap(getattr(fnp_mod, nm)))

# ---- 2) settrace: k/ka/extra at the transport gate (L1018) and k/ka at the two _hub2 call
#         sites (L1680 confined-present branch, L1692 all-young branch) ----
rows_transport, rows_hub = [], []
LINE_TRANSPORT, LINE_HUB_A, LINE_HUB_B = 1018, 1680, 1692

def local_trace(frame, event, arg):
    if event == "line":
        ln = frame.f_lineno
        loc = frame.f_locals
        if ln == LINE_TRANSPORT:
            rows_transport.append({"li": loc.get("li"), "k": loc.get("k"), "ka": loc.get("ka"), "extra": loc.get("extra")})
        elif ln == LINE_HUB_A:
            rows_hub.append({"site": "A", "k": loc.get("k"), "ka": loc.get("ka")})
        elif ln == LINE_HUB_B:
            rows_hub.append({"site": "B", "k": loc.get("k"), "ka": 0})
    return local_trace

def global_trace(frame, event, arg):
    if event == "call" and frame.f_code.co_filename == SRC:
        return local_trace
    return None

class _Ctx:
    seed = 0; width = 1024; depth = 16; flop_budget = 2**41
    submission_dir = None; scratch_dir = None; api_version = "x"

mlp = build_mlp(width=1024, depth=16, seed=7)
est = mod.Estimator(); est.setup(_Ctx())
sys.settrace(global_trace)
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    CTX["ctx"] = ctx
    est.predict(mlp, 2**41)
sys.settrace(None)
tot_u = ctx.flops_used / 2**31

# ---- 3) op-count model: how many (n,n,n) dense products does the transport / D21-hub line
#         actually issue, from the traced (k, ka, extra)? ----
transport_ops = sum(2 * (r["k"] - r["ka"]) + r["extra"] for r in rows_transport)
hub_ops = sum(2 * (r["k"] - r["ka"]) for r in rows_hub)

ledger_u = {str(ln): LED.get(ln, 0.0) / 2**31 for ln in (1019, 1680, 1692)}
transport_u_measured = ledger_u["1019"]
d21_u_measured = ledger_u["1680"] + ledger_u["1692"]

print(f"total metered: {tot_u:.2f} u")
print(f"L1019 transport (measured): {transport_u_measured:.3f} u  |  op-count model: {transport_ops} dense (n,n,n) products")
print(f"L1680+L1692 D21 hub (measured): {d21_u_measured:.3f} u  |  op-count model: {hub_ops} dense (n,n,n) products")
print(f"young tier total (measured): {transport_u_measured + d21_u_measured:.3f} u  (background claim: ~104 u)")
print(f"implied avg u/product -- transport: {transport_u_measured / transport_ops:.4f}, "
      f"D21 hub: {d21_u_measured / hub_ops:.4f}  (isolated single-product Strassen-L5 unit cost "
      f"measured separately at ~0.55-0.63 u -- see sonnet10_unit_costs.json)")

out = {
    "total_u": tot_u,
    "transport_rows": rows_transport, "hub_rows": rows_hub,
    "ledger_u": ledger_u,
    "young_tier_u_measured": transport_u_measured + d21_u_measured,
    "op_count_model": {"transport_ops": transport_ops, "hub_ops": hub_ops,
                        "transport_u_measured": transport_u_measured, "hub_u_measured": d21_u_measured,
                        "implied_avg_u_per_product_transport": transport_u_measured / transport_ops,
                        "implied_avg_u_per_product_hub": d21_u_measured / hub_ops},
}
os.makedirs("../results/summaries", exist_ok=True)
json.dump(out, open("../results/summaries/sonnet10_young_tier_count.json", "w"), indent=1)
