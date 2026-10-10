# kappa3 level scale test (10 Oct 2026), plain chain, MLP #0 + synthetic. Configs: ['V31_ANGULAR=0,CBA_MODE=scale,CBA_D3S=1.02,CBA_LMIN=9,CBA_LMAX=16', 'V31_ANGULAR=0,CBA_MODE=scale,CBA_D3S=0.98,CBA_LMIN=9,CBA_LMAX=16']
# kappa4-diagonal PATTERN (chain level) into the plain chain's mean closure on MLP #0, vs the true teacher pattern.
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
csrc = os.path.dirname(glob.glob("/kaggle/input/**/r20_cba.py", recursive=True)[0])
env = dict(os.environ, PYTHONUTF8="1", TAG="k3scale1")
sys.exit(subprocess.run([sys.executable, f"{src}/run_variant_teacher.py", f"{csrc}/r20_cba.py"] + ['V31_ANGULAR=0,CBA_MODE=scale,CBA_D3S=1.02,CBA_LMIN=9,CBA_LMAX=16', 'V31_ANGULAR=0,CBA_MODE=scale,CBA_D3S=0.98,CBA_LMIN=9,CBA_LMAX=16'], cwd=f"{base}/run", env=env).returncode)
