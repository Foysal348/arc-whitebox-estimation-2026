# Fable feed oracle E2E (10 Oct 2026): chain feed replaced by ['cf100', 'cf110', 'cf120'] in L9-16, MLP #0.
import glob, os, subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"], capture_output=True, text=True).stdout[-300:], flush=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/r20_joint_oracle.py", recursive=True)[0])
base = "/kaggle/working"
for net, sub in (("mlp0", "kaggle_out_k4pub"), ("synth", "kaggle_out_k4")):
    d = f"{base}/data/kaggle_out/{sub}/k4teach"; os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{src}/{net}__*"):
        t = os.path.join(d, os.path.basename(f).split("__", 1)[1])
        if not os.path.exists(t): os.symlink(f, t)
os.makedirs(f"{base}/run", exist_ok=True); os.makedirs(f"{base}/results/summaries", exist_ok=True)
fsrc = os.path.dirname(glob.glob("/kaggle/input/**/r20_feed_oracle.py", recursive=True)[0])
cfgs = []
for nm in ['cf100', 'cf110', 'cf120']:
    hits = glob.glob(f"/kaggle/input/**/{nm}__layer12_seed0.npz", recursive=True)
    fd = os.path.dirname(hits[0]); d = f"/kaggle/working/feeds/{nm}"; os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{fd}/{nm}__*"):
        os.symlink(f, os.path.join(d, os.path.basename(f).split("__", 1)[1]))
    cfgs.append(f"V31_ANGULAR=0,FEED_ORACLE=inject,FEED_LMIN=9,FEED_LMAX=16,FEED_DIR={d}")
env = dict(os.environ, PYTHONUTF8="1", TAG="feedor5", NETS="mlp0")
r = subprocess.run([sys.executable, f"{src}/run_variant_teacher.py", f"{fsrc}/r20_feed_oracle.py"] + cfgs, cwd=f"{base}/run", env=env)
sys.exit(r.returncode)
