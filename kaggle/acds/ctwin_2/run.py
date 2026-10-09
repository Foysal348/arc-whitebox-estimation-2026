# C.T layer windows (10 Oct): oracle-affine C.T kappa4 feature injected only in a layer window, MLP #0. Specs var:LMIN:LMAX ['ceil:15:99', 'eig64:11:99', 'eig64:13:99']
# K4d/v^2 ~ a chain_ck4/v^2 + b pred/v^2 + c fitted to the teacher (3 numbers per layer) is injected into the plain chain's
# mean closure on MLP #0. chain_only = same affine map without the prediction (fair baseline).
import glob, os, subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"], capture_output=True, text=True).stdout[-300:], flush=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/r20_joint_oracle.py", recursive=True)[0])
fit = os.path.dirname(glob.glob("/kaggle/input/**/ceil__layer12_seed0.npz", recursive=True)[0])
base = "/kaggle/working"
for net, sub in (("mlp0", "kaggle_out_k4pub"), ("synth", "kaggle_out_k4")):
    d = f"{base}/data/kaggle_out/{sub}/k4teach"; os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{src}/{net}__*"):
        t = os.path.join(d, os.path.basename(f).split("__", 1)[1])
        if not os.path.exists(t): os.symlink(f, t)
os.makedirs(f"{base}/run", exist_ok=True); os.makedirs(f"{base}/results/summaries", exist_ok=True)
cfgs = []
for spec in ['ceil:15:99', 'eig64:11:99', 'eig64:13:99']:
    var, lo, hi = spec.split(":")
    d = f"{base}/fits/{var}"
    if not os.path.isdir(d):
        os.makedirs(d)
        for f in glob.glob(f"{fit}/{var}__*"):
            os.symlink(f, os.path.join(d, os.path.basename(f).split("__", 1)[1]))
    cfgs.append(f"V31_ANGULAR=0,R20_K4=full,R20_K4_LMIN={lo},R20_K4_LMAX={hi},OPUS_TEACHER={d}")
env = dict(os.environ, PYTHONUTF8="1", TAG="ctwin", NETS="mlp0")
sys.exit(subprocess.run([sys.executable, f"{src}/run_variant_teacher.py", f"{src}/r20_joint_oracle.py"] + cfgs, cwd=f"{base}/run", env=env).returncode)
