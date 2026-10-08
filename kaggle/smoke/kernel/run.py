# Reproduce the grader smoke failure (ESTIMATOR_EXCEPTION) of v56s / v56As with full tracebacks; v55s125 (graded fine) as control.
import glob, os, subprocess, sys
env = dict(os.environ, PYTHONUTF8="1", HF_XET_HIGH_PERFORMANCE="1")
def sh(cmd):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("$", " ".join(cmd)[:200], "-> rc", r.returncode, flush=True)
    print(r.stdout[-6000:], flush=True); print(r.stderr[-12000:], flush=True)
    return r
sh([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"])
sh(["whest", "version"])
sh(["whest", "dataset", "download", "aicrowd/arc-whestbench-public-2026", "--revision", "v2-phase2", "--split", "mini"])
d = os.path.dirname(glob.glob("/kaggle/input/**/v56s.py", recursive=True)[0])
for name in ("v56s", "v56As"):
    print("=" * 30, name, flush=True)
    sh(["whest", "validate", "--estimator", f"{d}/{name}.py"])
    sh(["whest", "run", "--estimator", f"{d}/{name}.py", "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
        "--split", "mini", "--n-mlps", "1", "--runner", "subprocess", "--debug", "--fail-fast", "--format", "plain"])
