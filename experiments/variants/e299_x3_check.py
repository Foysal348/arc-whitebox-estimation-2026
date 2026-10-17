"""E299-X3 knob check (17 Oct 2026; template kaggle/e299b/warm_check.py): two predict() calls on MLP #0 in one fresh
subprocess per variant (cold + warm). Reports per-call FLOPs (u = 2^31), flopscope op count, residual time, final-layer MSE
vs the 6e7 teacher, and the deviation of all 16 layer outputs from the all-knobs-off base (exactness check).
Each subprocess runs under the shared laptop run lock (mkdir kaggle/_runlock, retry every 30 s, rmdir right after).
Usage: python experiments/variants/e299_x3_check.py [tag ...]      (tags from V below; results merged into the JSON)
"""
import os, sys, json, time, subprocess
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EST = "experiments/variants/e299_x3.py"
LOCK = os.path.join(ROOT, "kaggle", "_runlock")
OUTJ = os.path.join(ROOT, "results/summaries/exp3_e30a_cost_port.json")
PRED = os.environ.get("X3_PRED_DIR", os.path.join(ROOT, "kaggle/e299b/_x3pred"))
EXACT = {"X3_NOSYMC": "1", "X3_FOLD2": "1", "X3_SGDEAD": "1"}
ROUND = {"X3_DDMM": "1", "X3_KVEC": "1", "X3_PWR": "1", "X3_TQR": "1"}
V = {"base": {}, "E299_ref": None,
     "nosymc": {"X3_NOSYMC": "1"}, "fold2": {"X3_FOLD2": "1"}, "ddmm": {"X3_DDMM": "1"}, "kvec": {"X3_KVEC": "1"},
     "pwr": {"X3_PWR": "1"}, "reloj": {"X3_RELOJ": "1"}, "sgdead": {"X3_SGDEAD": "1"}, "tqr": {"X3_TQR": "1"},
     "hot3": {"X3_HOT": "3.0"},
     "exact_all": dict(EXACT), "exact_round": dict(EXACT, **ROUND),
     "exact_round_hot3": dict(EXACT, **ROUND, X3_HOT="3.0")}
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


def lock():
    while True:
        try:
            os.mkdir(LOCK); return
        except FileExistsError:
            print("  run lock busy, waiting 30 s", flush=True); time.sleep(30)


truth = np.load(os.path.join(ROOT, "data/kaggle_out/kaggle_out_k4pub/k4teach/truth.npz"))["final_means"].astype(np.float64)
tags = sys.argv[1:] or [t for t in V if t != "E299_ref"]
os.makedirs(PRED, exist_ok=True)
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for t in tags:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore", WC_EST=EST)
    for k in list(env):
        if k.startswith("X3_") and k != "X3_PRED_DIR":
            del env[k]
    if V[t] is None:
        env["WC_EST"] = "kaggle/e299b/E299_estimator.py"
    else:
        env.update(V[t])
    o = os.path.join(PRED, t)
    lock()
    try:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-c", CHILD, o], cwd=ROOT, env=env, capture_output=True, text=True)
    finally:
        os.rmdir(LOCK)
    if r.returncode != 0:
        print(t, "FAILED", r.stderr[-2500:], flush=True); continue
    calls = json.load(open(o + ".json")); p = np.load(o + ".npy"); final = p[-1] if p.ndim == 2 else p
    mse = float(np.mean((final - truth) ** 2))
    rec = dict(env={k: v for k, v in (V[t] or {}).items()}, calls=calls, mse=mse, run_s=time.time() - t0)
    if t != "base" and os.path.exists(os.path.join(PRED, "base.npy")):
        p0 = np.load(os.path.join(PRED, "base.npy"))
        rec["max_rel_dev_all_layers"] = float(np.max(np.abs(p - p0)) / np.max(np.abs(p0)))
        rec["rel_l2_dev_final"] = float(np.linalg.norm(final - p0[-1]) / np.linalg.norm(p0[-1]))
        rec["bit_identical"] = bool(np.array_equal(p, p0))
    res[t] = rec
    msg = (f"{t:18s} cold {calls[0]['flops_u']:.3f} u {calls[0]['ops']} ops | warm {calls[1]['flops_u']:.3f} u {calls[1]['ops']} ops "
           f"resid {calls[1]['resid_s']:.3f} s wall {calls[1]['wall_s']:.1f} s | MSE {mse:.5e}")
    if "base" in res and t != "base":
        b = res["base"]
        msg += (f" | dWarm {100 * (calls[1]['flops_u'] / b['calls'][1]['flops_u'] - 1):+.3f}% dOps {calls[1]['ops'] - b['calls'][1]['ops']:+d}"
                f" | MSE ratio {mse / b['mse']:.6f} | maxdev {rec.get('max_rel_dev_all_layers', float('nan')):.2e} bit {rec.get('bit_identical')}")
    print(msg, flush=True)
    json.dump(res, open(OUTJ, "w"), indent=1)
