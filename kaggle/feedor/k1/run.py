# Fable feed oracle (10 Oct 2026): hook validity + feed removal, MLP #0. Configs: ['V31_ANGULAR=0']
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
z = "/kaggle/working/zero"; os.makedirs(z, exist_ok=True)
for f in glob.glob(f"{fsrc}/zero__*"):
    t = os.path.join(z, os.path.basename(f).split("__", 1)[1])
    if not os.path.exists(t): os.symlink(f, t)
env = dict(os.environ, PYTHONUTF8="1", TAG="feedor1", NETS="mlp0", FEED_DUMP="/kaggle/working/feeddump")
r = subprocess.run([sys.executable, f"{src}/run_variant_teacher.py", f"{fsrc}/r20_feed_oracle.py"] + ['V31_ANGULAR=0'], cwd=f"{base}/run", env=env)
sys.exit(r.returncode)
