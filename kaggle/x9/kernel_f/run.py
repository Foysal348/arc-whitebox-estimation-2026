# X9 (20 Oct): e30k_x9 = e30k_x8 + X9_ANG (our V31 spherical input: layer-0 kappa4 core + outputs x a1) + X9_LAM, on public MLP #0,
# flopscope 0.12.3. Per variant: a fresh subprocess, two predict() calls (cold + warm); warm FLOPs (u = 2^31), op count, final MSE vs truth.
import subprocess, sys, glob, os, shutil, json
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.3,<0.13", "whestbench>=0.16.2,<0.17"],
                     capture_output=True, text=True).stdout[-300:], flush=True)
import numpy as np
W = "/kaggle/working"
for pat in ("weights_seed0.npz", "truth.npz", "e30k_x8.py", "e30k_x9.py"):
    src = sorted(glob.glob("/kaggle/input/**/" + pat, recursive=True))[0]
    print("copy", src, flush=True)
    shutil.copy(src, W)
os.chdir(W)
V = [("ref_x8", "e30k_x8.py", {}),
     ("base", "e30k_x9.py", {}),
     ("ang", "e30k_x9.py", {"X9_ANG": "1"}),
     ("ang_lam09", "e30k_x9.py", {"X9_ANG": "1", "X9_LAM": "0.9"}),
     ("ang_lam08", "e30k_x9.py", {"X9_ANG": "1", "X9_LAM": "0.8"}),
     ("ang_lam11", "e30k_x9.py", {"X9_ANG": "1", "X9_LAM": "1.1"}),
     ("lam09", "e30k_x9.py", {"X9_LAM": "0.9"})]
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
res, fulls = {}, {}
for tag, f, env_ in V:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore", WC_EST=f); env.update(env_)
    o = os.path.join(W, "_wc_" + tag)
    r = subprocess.run([sys.executable, "-c", CHILD, o], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(tag, "FAILED", r.stderr[-3000:], flush=True); res[tag] = dict(failed=r.stderr[-3000:]); continue
    calls = json.load(open(o + ".json")); p = np.load(o + ".npy"); final = p[-1] if p.ndim == 2 else p
    fulls[tag] = p
    mse = float(np.mean((final - truth) ** 2)); res[tag] = dict(calls=calls, mse=mse, env=env_)
    res[tag]["nonfinite"] = int(np.sum(~np.isfinite(p)))
    msg = (f"{tag:10s} cold {calls[0]['flops_u']:.2f} u {calls[0]['ops']} ops | warm {calls[1]['flops_u']:.3f} u "
           f"{calls[1]['ops']} ops resid {calls[1]['resid_s']:.3f} s wall {calls[1]['wall_s']:.1f} s | MSE {mse:.5e} | nonfinite {res[tag]['nonfinite']}")
    if "base" in res and tag != "base" and "calls" in res["base"]:
        b = res["base"]; fr = calls[1]['flops_u'] / b['calls'][1]['flops_u']
        msg += (f" | MSE ratio {mse / b['mse']:.5f} | FLOPs ratio {fr:.5f} | score ratio {mse / b['mse'] * fr:.5f}"
                f" | max|d| vs base {float(np.max(np.abs(p - fulls['base']))):.3e}")
    print(msg, flush=True)
json.dump(res, open("x9_results.json", "w"), indent=1)
if "ref_x8" in fulls and "base" in fulls:
    print("x9 default vs x8: max|d| (all layers)", float(np.max(np.abs(fulls["base"] - fulls["ref_x8"]))),
          "bit-identical", bool(np.array_equal(fulls["base"], fulls["ref_x8"])),
          "| warm FLOPs/ops x8", res["ref_x8"]["calls"][1]["flops_u"], res["ref_x8"]["calls"][1]["ops"],
          "x9", res["base"]["calls"][1]["flops_u"], res["base"]["calls"][1]["ops"], flush=True)
for f in ("weights_seed0.npz", "truth.npz", "e30k_x8.py", "e30k_x9.py"):
    os.remove(f)
for f in glob.glob("_wc_*"):
    os.remove(f)
