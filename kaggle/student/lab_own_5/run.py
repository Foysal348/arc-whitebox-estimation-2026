# Student teacher labels (11 Oct 2026): two worker processes, 2 threads each. Ranges [(100416, 100468), (100468, 100512)]
import glob, os, subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"], capture_output=True, text=True).stdout[-300:], flush=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/teacher_dump.py", recursive=True)[0])
env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
procs = [subprocess.Popen([sys.executable, f"{code}/teacher_dump.py", f"{code}/v56_dump.py", "/kaggle/working/dump", str(a), str(b), "32", ""], env=env)
         for a, b in [(100416, 100468), (100468, 100512)]]
sys.exit(max(p.wait() for p in procs))
