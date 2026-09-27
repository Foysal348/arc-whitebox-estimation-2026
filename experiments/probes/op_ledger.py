"""Op-count ledger: for every top-level Strassen call (mm / hub) of one predict() on public MLP #0, the number of
flopscope ops it issues and the FLOPs it costs, grouped by (kind, m, kd, w, level, caller line). The op count is a
deterministic proxy of residual wall time (Python glue per op). Also reports the ops outside Strassen.
Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/op_ledger.py <estimator.py> [warm]
"""
import os, sys, types, glob, collections, inspect
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC = sys.argv[1]
WARM = len(sys.argv) > 2
W = np.load(glob.glob("../data/kaggle_out/kaggle_out_k4pub/k4teach/weights_seed*.npz")[0])["W"].astype(np.float32)
mlp = MLP(width=W.shape[1], depth=W.shape[0], weights=[fnp.array(W[l]) for l in range(W.shape[0])], seed=0, name="mlp0")


class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


mod = types.ModuleType("est_var"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
S = mod._Strassen
state = {"depth": 0, "ctx": None}
led = collections.defaultdict(lambda: [0, 0, 0.0])   # calls, ops, flops


def wrap(name):
    orig = getattr(S, name)

    def w(self, X, Y, out, lev):
        top = state["depth"] == 0
        if top:
            ctx = state["ctx"]; o0 = len(ctx.op_log); f0 = ctx.flops_used
            fr = inspect.stack()[1]
            caller = f"{fr.function}:{fr.lineno}"
        state["depth"] += 1
        try:
            return orig(self, X, Y, out, lev)
        finally:
            state["depth"] -= 1
            if top:
                state.setdefault("ranges", []).append((o0, len(ctx.op_log)))
                shp = (name, tuple(X.shape[-2:]), tuple(Y.shape[-2:]), int(X.shape[0]) if name == "hub" else int(Y.shape[0]), lev, caller)
                e = led[shp]; e[0] += 1; e[1] += len(ctx.op_log) - o0; e[2] += (ctx.flops_used - f0) / 2**31
    setattr(S, name, w)


wrap("mm"); wrap("hub")
est = mod.Estimator(); est.setup(_Ctx())
if WARM:
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as c0:
        state["ctx"] = c0; est.predict(mlp, 2**41)
    led.clear()
    state["ranges"] = []
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    state["ctx"] = ctx
    est.predict(mlp, 2**41)
tot_ops = len(ctx.op_log); tot_u = ctx.flops_used / 2**31
s_ops = sum(v[1] for v in led.values()); s_u = sum(v[2] for v in led.values())
print(f"total ops {tot_ops}  u {tot_u:.2f} | Strassen top-level calls {sum(v[0] for v in led.values())}: ops {s_ops} ({100*s_ops/tot_ops:.0f}%), u {s_u:.2f}")
print(f"{'kind':4s} {'X':>12s} {'Y':>12s} {'b':>3s} {'lev':>3s} {'caller':22s} {'calls':>5s} {'ops':>6s} {'ops/call':>8s} {'u':>7s}")
for k, (c, o, u) in sorted(led.items(), key=lambda kv: -kv[1][1])[:40]:
    print(f"{k[0]:4s} {str(k[1]):>12s} {str(k[2]):>12s} {k[3]:>3d} {k[4]:>3d} {k[5][:22]:22s} {c:>5d} {o:>6d} {o/c:>8.0f} {u:>7.2f}")
by_caller = collections.defaultdict(lambda: [0, 0, 0.0])
for k, (c, o, u) in led.items():
    b = by_caller[k[5]]; b[0] += c; b[1] += o; b[2] += u
print("\nby caller:")
for k, (c, o, u) in sorted(by_caller.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k:28s} calls {c:4d} ops {o:6d} u {u:7.2f}")

# ops outside the top-level Strassen calls, by op name and first-operand shape
inside = set()
for a, b in state.get("ranges", []):
    inside.update(range(a, b))
outside = collections.Counter(); ou = collections.Counter()
for i, r in enumerate(ctx.op_log):
    if i in inside: continue
    shp = r.shapes[0] if r.shapes else ()
    key = (r.op_name, str(tuple(shp))[:40])
    outside[key] += 1; ou[key] += r.flop_cost
print()
print(f"ops outside Strassen: {tot_ops - len(inside)}")
for k, c in outside.most_common(30):
    print(f"  {c:5d}  {ou[k] / 2**31:6.2f} u  {k[0]:14s} {k[1]}")
