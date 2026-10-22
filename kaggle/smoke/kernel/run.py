# Reproduce the grader smoke (whestbench's own generated contest, graded shape, no dataset) for v56s / v56As with full
# tracebacks; v55s125 (graded fine) as control. Also an in-process call with the exception printed.
import glob, os, subprocess, sys, traceback
env = dict(os.environ, PYTHONUTF8="1")
def sh(cmd):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("$", " ".join(cmd)[:220], "-> rc", r.returncode, flush=True)
    print(r.stdout[-5000:], flush=True); print(r.stderr[-15000:], flush=True)
sh([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"])
sh([sys.executable, "-m", "pip", "list"])
d = os.path.dirname(glob.glob("/kaggle/input/**/v56s.py", recursive=True)[0])
probe = r'''
import sys, traceback, types, resource
import numpy as np
from whestbench import cli
from whestbench.scoring import make_contest, ContestSpec
spec = cli._smoke_test_contest_spec(); spec = ContestSpec(width=spec.width, depth=spec.depth, n_mlps=1, flop_budget=spec.flop_budget, ground_truth_samples=spec.ground_truth_samples)
data = make_contest(spec)
mlp = data.mlps[0]
print("mlp types:", type(mlp.weights[0]), getattr(mlp.weights[0], "dtype", None), len(mlp.weights), flush=True)
src = sys.argv[1]
mod = types.ModuleType("est"); exec(compile(open(src).read(), src, "exec"), mod.__dict__)
import flopscope as flops
class C: seed = 0; width = spec.width; depth = spec.depth; flop_budget = spec.flop_budget; submission_dir = None; scratch_dir = None; api_version = "x"
try:
    e = mod.Estimator(); e.setup(C())
    with flops.BudgetContext(flop_budget=spec.flop_budget, quiet=True):
        out = e.predict(mlp, spec.flop_budget)
    print("OK", np.asarray(out).shape, "maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, flush=True)
except Exception:
    traceback.print_exc()
'''
open("/kaggle/working/probe.py", "w").write(probe)
for name in ("v56s", "v55s125"):
    print("=" * 30, name, flush=True)
    sh([sys.executable, "/kaggle/working/probe.py", f"{d}/{name}.py"])
    sh(["whest", "run", "--estimator", f"{d}/{name}.py", "--n-mlps", "3", "--runner", "subprocess", "--debug", "--fail-fast", "--format", "plain"])
