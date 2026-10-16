"""Local pre-flight for round 2 (16 Oct): combinations on the blend-1.1 A and B estimators.

Each variant is run in a fresh subprocess, because the knobs are read at import time. The run uses MLP #0 (k4pub
weights) and reports:
- the final-layer prediction change against the default, which proves the knob takes effect;
- the metered FLOPs in u (2^31);
- the final-layer MSE against the 6e7-sample teacher (truth.npz) where available.

Usage: python kaggle/tune/local_check.py [tag ...]   (no tags: default + every variant)
"""
import os, sys, json, subprocess
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V0 = json.load(open(os.path.join(os.path.dirname(__file__), "variants_all.json")))
V = {t: v["env"] for t, v in V0.items()}
FILE = {t: v["file"] for t, v in V0.items()}
SRC = {"v56As_fsb11.py": "submissions/v56As_fsb11/estimator.py", "v56s_fsb11.py": "submissions/v56s_fsb11/estimator.py", "v56s_fs.py": "submissions/v56s_fs/estimator.py"}
CHILD = r'''
import os, sys, types, numpy as np, flopscope as flops, flopscope.numpy as fnp
from whestbench import MLP
EST = os.environ["LC_EST"]
mod = types.ModuleType("est"); mod.__file__ = os.path.abspath(EST)
exec(compile(open(EST, encoding="utf-8").read(), os.path.abspath(EST), "exec"), mod.__dict__)
W = np.load("data/kaggle_out/kaggle_out_k4pub/k4teach/weights_seed0.npz")["W"]; n, L = W.shape[1], W.shape[0]
class C: seed = 0; width = n; depth = L; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
e = mod.Estimator(); e.setup(C())
mlp = MLP(width=n, depth=L, weights=[fnp.array(w) for w in W], seed=0, name="m")
with flops.BudgetContext(flop_budget=2**41, quiet=True) as b:
    p = e.predict(mlp, 2**41)
ol = b.op_log
np.savez(sys.argv[1], pred=np.asarray(p, np.float64), flops=float(b.flops_used), ops=float(len(ol) if ol is not None else -1))
'''
tags = sys.argv[1:] or ["default"] + list(V)
truth = None
tp = os.path.join(ROOT, "data/kaggle_out/kaggle_out_k4pub/k4teach/truth.npz")
if os.path.exists(tp):
    z = np.load(tp); truth = np.asarray(z[z.files[0]], np.float64)
res = {}
for t in tags:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore"); env.update(V.get(t, {})); env["LC_EST"] = SRC[FILE[t]] if t in FILE else "submissions/v56As_fs/estimator.py"
    out = os.path.join(ROOT, f"kaggle/tune/_lc_{t}.npz")
    r = subprocess.run([sys.executable, "-c", CHILD, out], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(t, "FAILED", r.stderr[-1500:]); res[t] = None; continue
    with np.load(out) as z:
        p = z["pred"]; res[t] = dict(flops_u=float(z["flops"]) / 2**31, ops=int(z["ops"]))
    final = p[-1] if p.ndim == 2 else p
    res[t]["final"] = final
    msg = f"{t:11s} flops {res[t]['flops_u']:.2f} u, ops {res[t]['ops']}"
    if truth is not None:
        tf = truth[-1] if truth.ndim == 2 else truth
        res[t]["mse"] = float(np.mean((final - tf) ** 2)); msg += f" | final MSE vs teacher {res[t]['mse']:.4e}"
    if "default" in res and res["default"] is not None and t != "default":
        d = res["default"]["final"]; msg += f" | rel change vs default {np.linalg.norm(final - d) / np.linalg.norm(d):.2e}"
        if "mse" in res[t]: msg += f" | MSE ratio {res[t]['mse'] / res['default']['mse']:.4f}"
    print(msg, flush=True)
    os.remove(out)
json.dump({t: {k: v for k, v in r.items() if k != "final"} for t, r in res.items() if r}, open(os.path.join(ROOT, "results/summaries/tune2_local_check.json"), "w"), indent=1)
