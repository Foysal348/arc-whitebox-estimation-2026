# Fable kappa3-feed ceiling (10 Oct 2026), CPU split (no GPU quota): targets 13,14, N = 1e6, no sampled C.T feed.
import glob, os, subprocess, sys
import torch; torch.set_num_threads(os.cpu_count() or 4); print("threads", torch.get_num_threads(), flush=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/fable_k3feed_ceiling_2026-10-10.py", recursive=True)[0])
data = glob.glob("/kaggle/input/**/r17_data.npz", recursive=True)[0]
teach = "/kaggle/working/teach"; os.makedirs(teach, exist_ok=True)
for f in glob.glob("/kaggle/input/**/mlp0__layer*_seed0.npz", recursive=True):
    t = os.path.join(teach, os.path.basename(f).split("__", 1)[1])
    if not os.path.exists(t): os.symlink(f, t)
cmd = [sys.executable, f"{src}/fable_k3feed_ceiling_2026-10-10.py", "1e6", "cpu", "--targets", "13,14", "--J", "4,6", "--ct-targets", "none",
       "--data", data, "--teacher-dir", teach, "--out", "/kaggle/working/k3feed_mlp0_part3", "--batch", "2000"]
print(cmd, flush=True)
sys.exit(subprocess.run(cmd).returncode)
