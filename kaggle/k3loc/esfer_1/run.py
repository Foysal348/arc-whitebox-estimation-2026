# kappa3 localisation (10 Oct): teacher kappa3 (D21 slice / d3) injected in a layer window, MLP #0. Configs: ['V31_ANGULAR=0,OPUS_D21ORACLE=all_d3,OPUS_OR_LMIN=1,OPUS_OR_LMAX=4', 'V31_ANGULAR=0,OPUS_D21ORACLE=all_d3,OPUS_OR_LMIN=5,OPUS_OR_LMAX=8', 'V31_ANGULAR=0,OPUS_D21ORACLE=all_d3,OPUS_OR_LMIN=9,OPUS_OR_LMAX=12']
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
env = dict(os.environ, PYTHONUTF8="1", TAG="k3loc", NETS="mlp0")
r = subprocess.run([sys.executable, f"{src}/run_variant_teacher.py", f"{src}/r20_joint_oracle.py"] + ['V31_ANGULAR=0,OPUS_D21ORACLE=all_d3,OPUS_OR_LMIN=1,OPUS_OR_LMAX=4', 'V31_ANGULAR=0,OPUS_D21ORACLE=all_d3,OPUS_OR_LMIN=5,OPUS_OR_LMAX=8', 'V31_ANGULAR=0,OPUS_D21ORACLE=all_d3,OPUS_OR_LMIN=9,OPUS_OR_LMAX=12'], cwd=f"{base}/run", env=env)
sys.exit(r.returncode)
