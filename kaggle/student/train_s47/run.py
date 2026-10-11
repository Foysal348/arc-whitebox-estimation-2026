# Round-20 PM364 training, seed 47 (11 Oct 2026): 512 train / 128 val / 128 test, free-running, 16 epochs max.
import glob, os, subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"], capture_output=True, text=True).stdout[-300:], flush=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/train_pm.py", recursive=True)[0])
env = dict(os.environ, PYTHONUTF8="1")
sys.exit(subprocess.run([sys.executable, f"{code}/train_pm.py", "--data", "/kaggle/input", "--est", f"{code}/v56_dump.py",
                         "--out", "/kaggle/working/run_s47", "--seed", "47", "--epochs", "16", "--val-from-train", "64"], env=env).returncode)
