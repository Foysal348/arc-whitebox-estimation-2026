"""Experiment 5 (17 Oct): e30_a vs e30a_k4c (E299 kappa4 corrector / rider ported). Template: kaggle/e299b/warm_check.py.
Two predict() calls on MLP #0 in one process (cold + warm), a fresh subprocess per variant, run one at a time under the
shared kaggle/_runlock directory lock. Reports per-call FLOPs (u = 2^31), flopscope op count, residual time and the
final-layer MSE vs the k4teach truth.
Usage: python experiments/variants/e30a_k4c_warm_check.py [tag ...]
"""
import os, sys, json, time, subprocess
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EST_BASE = "external/e30_a_estimator.py"
EST_X5 = "experiments/variants/e30a_k4c.py"
V = {"e30a": (EST_BASE, {}),
     "x5_off": (EST_X5, {}),
     "k4c_s050": (EST_X5, {"X5_K4C": "1", "X5_K4C_SCALE": "0.5"}),
     "k4c_s075": (EST_X5, {"X5_K4C": "1", "X5_K4C_SCALE": "0.75"}),
     "k4c_s100": (EST_X5, {"X5_K4C": "1", "X5_K4C_SCALE": "1"}),
     "k4c_s025": (EST_X5, {"X5_K4C": "1", "X5_K4C_SCALE": "0.25"}),
     "k4c_s100_nostar": (EST_X5, {"X5_K4C": "1", "X5_K4S": "0"}),
     "k4c_s050_nostar": (EST_X5, {"X5_K4C": "1", "X5_K4S": "0", "X5_K4C_SCALE": "0.5"}),
     "k4c_s050_rider": (EST_X5, {"X5_K4C": "1", "X5_K4C_SCALE": "0.5", "X5_RIDER": "1"}),
     "k4c_s100_lvl": (EST_X5, {"X5_K4C": "1", "X5_K4LVL": "1"}),
     "rider": (EST_X5, {"X5_RIDER": "1"}),
     "lvl": (EST_X5, {"X5_K4LVL": "1"}),
     "k4c_s030": (EST_X5, {"X5_K4C": "1", "X5_K4C_SCALE": "0.3"}),
     "k4c_s030_lvl": (EST_X5, {"X5_K4C": "1", "X5_K4C_SCALE": "0.3", "X5_K4LVL": "1"}),
     "k4c_s030_nostar": (EST_X5, {"X5_K4C": "1", "X5_K4C_SCALE": "0.3", "X5_K4S": "0"}),
     "k4c_s100_rider": (EST_X5, {"X5_K4C": "1", "X5_RIDER": "1"})}
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


def lock():
    while True:
        try:
            os.mkdir(LOCK); return
        except FileExistsError:
            print("  runlock busy, waiting 30 s", flush=True); time.sleep(30)


truth = np.load(os.path.join(ROOT, "data/kaggle_out/kaggle_out_k4pub/k4teach/truth.npz"))["final_means"].astype(np.float64)
tags = sys.argv[1:] or list(V)
OUTJ = os.path.join(ROOT, "results/summaries/exp5_k4c_into_e30a.json")
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for t in tags:
    est, kn = V[t]
    env = {k: v for k, v in os.environ.items() if not k.startswith("X5_")}
    env.update(PYTHONUTF8="1", PYTHONWARNINGS="ignore", WC_EST=est); env.update(kn)
    o = os.path.join(ROOT, f"experiments/variants/_wc_{t}")
    lock()
    try:
        r = subprocess.run([sys.executable, "-c", CHILD, o], cwd=ROOT, env=env, capture_output=True, text=True)
    finally:
        os.rmdir(LOCK)
    if r.returncode != 0:
        print(t, "FAILED", r.stderr[-2500:], flush=True); continue
    calls = json.load(open(o + ".json")); p = np.load(o + ".npy"); final = p[-1] if p.ndim == 2 else p
    mse = float(np.mean((final - truth) ** 2)); res[t] = dict(est=est, knobs=kn, calls=calls, mse=mse)
    np.save(o + "_final.npy", final)
    msg = (f"{t:18s} cold {calls[0]['flops_u']:.3f} u {calls[0]['ops']} ops | warm {calls[1]['flops_u']:.3f} u {calls[1]['ops']} ops "
           f"resid {calls[1]['resid_s']:.3f} s wall {calls[1]['wall_s']:.1f} s | MSE {mse:.5e}")
    if "e30a" in res and t != "e30a":
        b = res["e30a"]
        msg += (f" | MSE ratio {mse / b['mse']:.5f} | dwarm {calls[1]['flops_u'] - b['calls'][1]['flops_u']:+.3f} u"
                f" | dops {calls[1]['ops'] - b['calls'][1]['ops']:+d} | score ratio {mse * calls[1]['flops_u'] / (b['mse'] * b['calls'][1]['flops_u']):.5f}")
    print(msg, flush=True)
    os.remove(o + ".json"); os.remove(o + ".npy")
    json.dump(res, open(OUTJ, "w"), indent=1)
