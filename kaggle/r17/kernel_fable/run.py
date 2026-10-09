# Fable LKV ceiling (experiments/probes/fable_ceiling_2026-10-09.py, smoke-run by Fable at N=2000), N = 2e6 on GPU.
import glob, os, subprocess, sys
src = glob.glob("/kaggle/input/**/fable_ceiling_2026-10-09.py", recursive=True)[0]
data = glob.glob("/kaggle/input/**/r17_data.npz", recursive=True)[0]
tdir = "/kaggle/working/k4teach"; os.makedirs(tdir, exist_ok=True)
for f in glob.glob("/kaggle/input/**/mlp0__layer*_seed0.npz", recursive=True):
    t = os.path.join(tdir, os.path.basename(f).split("__", 1)[1])
    if not os.path.exists(t): os.symlink(f, t)
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
r = subprocess.run([sys.executable, src, "2e6", "cuda", "--data", data, "--teacher-dir", tdir, "--targets", "all", "--diag-targets", "12,15,16", "--out", "/kaggle/working/fable_ceiling_all"])
sys.exit(r.returncode)
