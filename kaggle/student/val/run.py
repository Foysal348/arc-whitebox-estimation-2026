# Student teacher-dump validation (11 Oct 2026): 2 held-out seeds with full matrices (parity data) + timing.
import glob, os, subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"], capture_output=True, text=True).stdout[-300:], flush=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/teacher_dump.py", recursive=True)[0])
env = dict(os.environ, PYTHONUTF8="1")
sys.exit(subprocess.run([sys.executable, f"{code}/teacher_dump.py", f"{code}/v56_dump.py", "/kaggle/working/dump", "300000", "300001", "32", "300000"], env=env).returncode)
