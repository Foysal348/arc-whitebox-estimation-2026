"""EXP4 (17 Oct 2026): E299 vs the e30_a fixed-basis old level port (experiments/variants/e299_fija.py).
Template: kaggle/e299b/warm_check.py. Two predict() calls on MLP #0 in a fresh subprocess per variant (cold + warm);
reports warm FLOPs (u = 2^31), flopscope op count, residual time, final-layer MSE vs the 6e7 teacher.
LOCAL RUN RULE: each child takes kaggle/_runlock (mkdir; retry every 30 s) and releases it right after.
Usage: python experiments/variants/exp4_warm_check.py [tag ...]
"""
import os, sys, json, time, subprocess
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EST = "experiments/variants/e299_fija.py"
V = {"base": {}, "fija": {"X4_FIJA": "1"}, "hot3": {"X4_HOT": "3.0"}, "fija_hot3": {"X4_FIJA": "1", "X4_HOT": "3.0"},
     "fija_hot2.5": {"X4_FIJA": "1", "X4_HOT": "2.5"}, "hot2.5": {"X4_HOT": "2.5"}}
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
LOCK = os.path.join(ROOT, "kaggle", "_runlock")
OUTD = os.environ.get("EXP4_OUT", os.path.join(ROOT, "results", "summaries", "exp4_preds"))
os.makedirs(OUTD, exist_ok=True)
truth = np.load(os.path.join(ROOT, "data/kaggle_out/kaggle_out_k4pub/k4teach/truth.npz"))["final_means"].astype(np.float64)
JS = os.path.join(ROOT, "results/summaries/exp4_fixed_basis_port.json")
res = json.load(open(JS)) if os.path.exists(JS) else {}
for t in sys.argv[1:] or list(V):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore", WC_EST=EST); env.update(V[t])
    o = os.path.join(OUTD, f"_wc_{t}")
    while True:
        try:
            os.mkdir(LOCK); break
        except FileExistsError:
            time.sleep(30)
    try:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-c", CHILD, o], cwd=ROOT, env=env, capture_output=True, text=True)
    finally:
        os.rmdir(LOCK)
    if r.returncode != 0:
        print(t, "FAILED", r.stderr[-2500:], flush=True); continue
    calls = json.load(open(o + ".json")); p = np.load(o + ".npy"); final = p[-1] if p.ndim == 2 else p
    mse = float(np.mean((final - truth) ** 2)); res[t] = dict(env=V[t], calls=calls, mse=mse, run_s=time.time() - t0)
    msg = (f"{t:12s} cold {calls[0]['flops_u']:.2f} u {calls[0]['ops']} ops | warm {calls[1]['flops_u']:.3f} u {calls[1]['ops']} ops "
           f"resid {calls[1]['resid_s']:.3f} s wall {calls[1]['wall_s']:.1f} s | MSE {mse:.6e}")
    if "base" in res and t != "base":
        b = res["base"]
        msg += (f" | dFLOPs {100 * (calls[1]['flops_u'] / b['calls'][1]['flops_u'] - 1):+.2f}% | dops {calls[1]['ops'] - b['calls'][1]['ops']:+d}"
                f" | MSE ratio {mse / b['mse']:.5f} | score ratio {mse * calls[1]['flops_u'] / (b['mse'] * b['calls'][1]['flops_u']):.4f}")
    print(msg, flush=True)
    json.dump(res, open(JS, "w"), indent=1)
