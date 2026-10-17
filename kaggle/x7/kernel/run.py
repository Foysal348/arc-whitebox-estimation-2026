# Experiment 7 (17 Oct): e299_x7 (e299xrk + e30_a dead-neuron compactions, LordProcciuto MIT) on public MLP #0, flopscope 0.12.3.
# Per variant: a fresh subprocess, two predict() calls (cold + warm); warm FLOPs (u = 2^31), op count, final MSE vs the 6e7 teacher.
# Then call-site ledgers of the compacted variants.
import subprocess, sys, glob, os, shutil, json
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.3,<0.13", "whestbench>=0.16.2,<0.17"],
                     capture_output=True, text=True).stdout[-300:], flush=True)
import numpy as np
D = os.path.dirname(glob.glob("/kaggle/input/**/site_ledger.py", recursive=True)[0])
W = "/kaggle/working"
for f in os.listdir(D):
    shutil.copy(os.path.join(D, f), W)
os.chdir(W)
V = [("ref_e299xrk", "e299xrk.py", {}),
     ("base", "e299_x7.py", {}),
     ("yrow", "e299_x7.py", {"X7_YROW": "1"}),
     ("cpre", "e299_x7.py", {"X7_CPRE": "1"}),
     ("both", "e299_x7.py", {"X7_YROW": "1", "X7_CPRE": "1"}),
     ("yrow_arriba", "e299_x7.py", {"X7_YROW": "1", "X7_DEAD_ROUND": "arriba"}),
     ("yrow_noz", "e299_x7.py", {"X7_YROW": "1", "X7_YROW_Z": "0"})]
CHILD = r'''
import os, sys, types, json, numpy as np, flopscope as flops, flopscope.numpy as fnp
from whestbench import MLP
EST = os.environ["WC_EST"]
mod = types.ModuleType("est"); mod.__file__ = os.path.abspath(EST)
exec(compile(open(EST, encoding="utf-8").read(), os.path.abspath(EST), "exec"), mod.__dict__)
W = np.load("weights_seed0.npz")["W"]; n, L = W.shape[1], W.shape[0]
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
truth = np.load("truth.npz")["final_means"].astype(np.float64)
res, finals = {}, {}
for tag, f, env_ in V:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore", WC_EST=f); env.update(env_)
    o = os.path.join(W, "_wc_" + tag)
    r = subprocess.run([sys.executable, "-c", CHILD, o], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(tag, "FAILED", r.stderr[-3000:], flush=True); res[tag] = dict(failed=r.stderr[-3000:]); continue
    calls = json.load(open(o + ".json")); p = np.load(o + ".npy"); final = p[-1] if p.ndim == 2 else p
    finals[tag] = final
    mse = float(np.mean((final - truth) ** 2)); res[tag] = dict(calls=calls, mse=mse, env=env_)
    msg = (f"{tag:12s} cold {calls[0]['flops_u']:.2f} u {calls[0]['ops']} ops | warm {calls[1]['flops_u']:.3f} u "
           f"{calls[1]['ops']} ops resid {calls[1]['resid_s']:.3f} s wall {calls[1]['wall_s']:.1f} s | MSE {mse:.5e}")
    if "base" in res and tag != "base" and "calls" in res["base"]:
        b = res["base"]; fr = calls[1]['flops_u'] / b['calls'][1]['flops_u']
        msg += (f" | MSE ratio {mse / b['mse']:.5f} | FLOPs ratio {fr:.4f} | score ratio {mse / b['mse'] * fr:.4f}"
                f" | max|d| vs base {float(np.max(np.abs(final - finals['base']))):.3e}")
    print(msg, flush=True)
    res[tag]["nonfinite"] = int(np.sum(~np.isfinite(final)))
json.dump(res, open("x7_results.json", "w"), indent=1)
if "ref_e299xrk" in finals and "base" in finals:
    print("base vs e299xrk max|d|", float(np.max(np.abs(finals["base"] - finals["ref_e299xrk"]))), flush=True)
for tag, envs in (("yrow", "X7_YROW=1"), ("both", "X7_YROW=1,X7_CPRE=1")):
    r = subprocess.run([sys.executable, "site_ledger.py", "e299_x7.py", "weights_seed0.npz", "ledger_x7_" + tag, envs],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUTF8="1"))
    open("ledger_x7_" + tag + ".txt", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr[-3000:])
    print("ledger", tag, "rc", r.returncode); print(r.stdout[:5000], flush=True)
for f in ("weights_seed0.npz",):
    os.remove(f)
for f in glob.glob("_wc_*"):
    os.remove(f)
