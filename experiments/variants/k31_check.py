"""E299 vs low-op variants (16 Oct): two predict() calls on MLP #0 in one process (cold + warm), each in a fresh subprocess
per variant. Reports per-call FLOPs (u = 2^31), flopscope op count, residual time, and the final-layer MSE vs the 6e7 teacher.
Usage: python kaggle/e299b/warm_check.py [tag ...]
"""
import os, sys, json, subprocess
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V = {"base": {}, "k31": {"X2_K31TWO": "1"}, "k31_mix05": {"X2_K31TWO": "1", "X2_K31_MIX": "0.5"}}
CHILD = r'''
import os, sys, types, json, numpy as np, flopscope as flops, flopscope.numpy as fnp
from whestbench import MLP
EST = os.environ["WC_EST"]
mod = types.ModuleType("est"); mod.__file__ = os.path.abspath(EST)
exec(compile(open(EST, encoding="utf-8").read(), os.path.abspath(EST), "exec"), mod.__dict__)
W = np.load("data/kaggle_out/kaggle_out_k4pub/k4teach/weights_seed0.npz")["W"]; n, L = W.shape[1], W.shape[0]
class C: seed = 0; width = n; depth = L; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
e = mod.Estimator(); e.setup(C())
mlp = MLP(width=n, depth=L, weights=[fnp.array(w) for w in W], seed=0, name="m")
out = []
for call in range(2):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as b:
        p = e.predict(mlp, 2**41)
    out.append(dict(flops_u=float(b.flops_used) / 2**31, ops=len(b.op_log or []), resid_s=float(b.residual_wall_time_s or 0), wall_s=float(b.wall_time_s or 0)))
np.save(sys.argv[1] + ".npy", np.asarray(p, np.float64)); json.dump(out, open(sys.argv[1] + ".json", "w"))
'''
truth = np.load(os.path.join(ROOT, "data/kaggle_out/kaggle_out_k4pub/k4teach/truth.npz"))["final_means"].astype(np.float64)
tags = sys.argv[1:] or list(V)
res = {}
for t in tags:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore", WC_EST="experiments/variants/e299_k31.py"); env.update(V[t])
    o = os.path.join(ROOT, f"kaggle/e299b/_k31_{t}")
    r = subprocess.run([sys.executable, "-c", CHILD, o], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(t, "FAILED", r.stderr[-1500:], flush=True); continue
    calls = json.load(open(o + ".json")); p = np.load(o + ".npy"); final = p[-1] if p.ndim == 2 else p
    mse = float(np.mean((final - truth) ** 2)); res[t] = dict(calls=calls, mse=mse)
    msg = f"{t:18s} cold {calls[0]['flops_u']:.2f} u {calls[0]['ops']} ops resid {calls[0]['resid_s']:.3f} s | warm {calls[1]['flops_u']:.2f} u {calls[1]['ops']} ops resid {calls[1]['resid_s']:.3f} s wall {calls[1]['wall_s']:.1f} s | MSE {mse:.4e}"
    if "base" in res and t != "base":
        b = res["base"]; msg += f" | MSE ratio {mse / b['mse']:.5f} | warm util ratio {calls[1]['flops_u'] / b['calls'][1]['flops_u']:.4f} | score ratio {mse * calls[1]['flops_u'] / (b['mse'] * b['calls'][1]['flops_u']):.4f}"
    print(msg, flush=True)
    os.remove(o + ".json"); os.remove(o + ".npy")
json.dump(res, open(os.path.join(ROOT, "results/summaries/exp2_k31_local.json"), "w"), indent=1)
