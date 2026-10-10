# Round-20 student GPU smoke (11 Oct 2026): zero-theta (pure slice recursion) baseline on the validation dump; timing.
import glob, os, subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"], capture_output=True, text=True).stdout[-300:], flush=True)
import numpy as np, torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/train_pm.py", recursive=True)[0])
data = os.path.dirname(glob.glob("/kaggle/input/**/net_300000.npz", recursive=True)[0])
np.save("/kaggle/working/zero.npy", np.zeros((4, 91), np.float32))
env = dict(os.environ, PYTHONUTF8="1")
r = subprocess.run([sys.executable, f"{code}/train_pm.py", "--data", data, "--est", f"{code}/v56_dump.py", "--out", "/kaggle/working/smoke",
                    "--eval-only", "/kaggle/working/zero.npy"], env=env)
# one timed forward+backward on the same network
sys.path.insert(0, code)
from pm_student import PairMemoryStudent
import time
m = PairMemoryStudent(f"{code}/v56_dump.py", device="cuda"); m.theta.data.normal_(0, 0.01)
rng = np.random.default_rng(300000)
Ws = [torch.tensor((rng.standard_normal((1024, 1024)) * np.sqrt(2 / 1024)).astype(np.float32), device="cuda") for _ in range(16)]
for k in range(3):
    torch.cuda.synchronize(); t0 = time.time()
    out = m(Ws); loss = (out[-1] ** 2).mean(); loss.backward(); torch.cuda.synchronize()
    print(f"fwd+bwd {time.time() - t0:.2f}s, peak mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
sys.exit(r.returncode)
