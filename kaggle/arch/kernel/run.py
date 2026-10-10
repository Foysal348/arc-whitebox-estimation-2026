# Architecture analysis (10 Oct 2026): (1) call-site FLOP ledger of the shipped V56 (v56As_fs); (2) single-pool kappa3 ceiling.
import glob, os, subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"], capture_output=True, text=True).stdout[-300:], flush=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/site_ledger.py", recursive=True)[0])
caps = os.path.dirname(glob.glob("/kaggle/input/**/r7_cap_L14.npz", recursive=True)[0])
wts = glob.glob("/kaggle/input/**/teach__weights_seed0.npz", recursive=True)[0]
os.chdir("/kaggle/working")
r1 = subprocess.run([sys.executable, f"{code}/site_ledger.py", f"{code}/v56As_fs.py", wts, "/kaggle/working/ledger_v56As_fs"])
print("ledger rc", r1.returncode, flush=True)
r2 = subprocess.run([sys.executable, f"{code}/pool_ceiling.py", "--layers", "11,14,15", "--ranks", "128,256,384,512,768", "--cap", caps, "--out", "/kaggle/working/pool_ceiling.json"])
sys.exit(r1.returncode or r2.returncode)
