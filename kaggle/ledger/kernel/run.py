# Call-site cost ledgers of e299xrk (team best #336757) and e30_a (LordProcciuto, MIT) on public MLP #0, flopscope 0.12.3 (17 Oct).
import subprocess, sys, glob, os, shutil
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.3,<0.13", "whestbench>=0.16.2,<0.17"], capture_output=True, text=True).stdout[-300:])
D = os.path.dirname(glob.glob("/kaggle/input/**/site_ledger.py", recursive=True)[0])
W = "/kaggle/working"
for f in os.listdir(D): shutil.copy(os.path.join(D, f), W)
os.chdir(W)
for est in ("e299xrk.py", "e30a.py"):
    r = subprocess.run([sys.executable, "site_ledger.py", est, "weights_seed0.npz", "ledger_" + est[:-3]], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUTF8="1"))
    open("ledger_" + est[:-3] + ".txt", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr[-3000:])
    print(est, "rc", r.returncode); print(r.stdout[-6000:])
for f in ("weights_seed0.npz",): os.remove(f)
