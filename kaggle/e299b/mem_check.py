"""Peak memory of E299, E299-B and E295 (16 Oct): two predict() calls on MLP #0 in a fresh subprocess each; the parent samples
the child's working set / RSS every 50 ms and the child reports its own peak (Windows peak_wset). Laptop-only, one MLP.
Usage: python kaggle/e299b/mem_check.py
"""
import os, sys, json, time, subprocess, psutil
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = {"E299": ("kaggle/e299b/E299_estimator.py", {}), "E299B": ("submissions/e299b/estimator.py", {}),
        "E295": ("kaggle/e299b/E295_estimator.py", {})}
CHILD = r'''
import os, sys, types, numpy as np, flopscope as flops, flopscope.numpy as fnp, psutil
from whestbench import MLP
EST = os.environ["MC_EST"]
mod = types.ModuleType("est"); mod.__file__ = os.path.abspath(EST)
exec(compile(open(EST, encoding="utf-8").read(), os.path.abspath(EST), "exec"), mod.__dict__)
W = np.load("data/kaggle_out/kaggle_out_k4pub/k4teach/weights_seed0.npz")["W"]; n, L = W.shape[1], W.shape[0]
class C: seed = 0; width = n; depth = L; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
e = mod.Estimator(); e.setup(C())
mlp = MLP(width=n, depth=L, weights=[fnp.array(w) for w in W], seed=0, name="m")
base = psutil.Process().memory_info().rss
for call in range(2):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as b:
        p = e.predict(mlp, 2**41)
mi = psutil.Process().memory_info()
print("CHILD_PEAK_GB", getattr(mi, "peak_wset", 0) / 2**30, "BASE_GB", base / 2**30)
'''
res = {}
for tag, (est, env_extra) in RUNS.items():
    env = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore", MC_EST=est); env.update(env_extra)
    p = subprocess.Popen([sys.executable, "-c", CHILD], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ps = psutil.Process(p.pid); peak = 0
    while p.poll() is None:
        try: peak = max(peak, ps.memory_info().rss)
        except psutil.Error: pass
        time.sleep(0.05)
    out, err = p.communicate()
    line = [l for l in out.splitlines() if l.startswith("CHILD_PEAK_GB")]
    res[tag] = dict(sampled_peak_rss_gb=peak / 2**30, child=line[0] if line else err[-400:])
    print(f"{tag:6s} sampled peak RSS {peak / 2**30:.2f} GB | {res[tag]['child']}", flush=True)
json.dump(res, open(os.path.join(ROOT, "results/summaries/e299b_mem_check.json"), "w"), indent=1)
