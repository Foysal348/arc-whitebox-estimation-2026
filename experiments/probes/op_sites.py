"""Op count by estimator source line: one warm-up predict() on the synthetic net, then a recorded predict() on public
MLP #0 in which every flopscope op is attributed to the innermost estimator line outside the Strassen helper.
The op count is the residual-time driver (each op ~0.07 ms laptop / ~0.02 ms grader).
Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/op_sites.py <estimator.py> [top]
"""
import os, sys, types, glob, collections
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC = os.path.abspath(sys.argv[1])
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 45
nets = []
for d in ("../data/kaggle_out/kaggle_out_k4/k4teach", "../data/kaggle_out/kaggle_out_k4pub/k4teach"):
    W = np.load(glob.glob(os.path.join(d, "weights_seed*.npz"))[0])["W"].astype(np.float32)
    nets.append(MLP(width=1024, depth=16, weights=[fnp.array(W[l]) for l in range(16)], seed=0, name=d))


class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


mod = types.ModuleType("est_var"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
HELPERS = {"_combos", "_assemble", "_leaf_mm", "_leaf_hub", "mm", "hub", "_buf", "_buf2", "_flat", "level", "_ok", "get", "get_pair", "NN"}
lines = open(SRC, encoding="utf-8").read().split("\n")
count = collections.Counter(); ucost = collections.Counter(); names = collections.defaultdict(collections.Counter)


class RecList(list):
    def append(self, rec):
        f = sys._getframe(1); site = None; strassen = False
        while f is not None:
            if os.path.abspath(f.f_code.co_filename) == SRC:
                if f.f_code.co_name in HELPERS:
                    strassen = strassen or f.f_code.co_name in ("mm", "hub", "_combos", "_assemble", "_leaf_mm", "_leaf_hub")
                else:
                    site = (f.f_lineno, f.f_code.co_name); break
            f = f.f_back
        key = (site, strassen)
        count[key] += 1; ucost[key] += rec.flop_cost; names[key][rec.op_name] += 1
        super().append(rec)


est = mod.Estimator(); est.setup(_Ctx())
with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
    est.predict(nets[0], 2**41)
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    ctx._op_log = RecList()
    est.predict(nets[1], 2**41)
tot = sum(count.values())
print(f"total ops {tot} (op_log {len(ctx.op_log)}), u {ctx.flops_used / 2**31:.2f}")
print(f"{'ops':>6s} {'u':>7s} {'S':1s} line  function          source / op names")
SORT_U = os.environ.get("SORT_U") == "1"
order = sorted(count, key=lambda k: -ucost[k])[:TOP] if SORT_U else [k for k, _ in count.most_common(TOP)]
for key in order:
    c = count[key]
    (ln, fn), st = (key[0] or (0, "?")), key[1]
    src = lines[ln - 1].strip()[:70] if ln else ""
    top_ops = ",".join(f"{k}:{v}" for k, v in names[key].most_common(3))
    print(f"{c:6d} {ucost[key] / 2**31:7.2f} {'S' if st else ' '} {ln:5d} {fn[:16]:16s} {src} [{top_ops}]")
