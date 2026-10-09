# End-to-end value of the Fable/LKV kappa4 prediction (star + C.T [+ T.T], exact kappa3 from samples) injected as the
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
preds = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/fable_ceiling_all_k4pred_*", recursive=True) if os.path.isdir(p)}
print("prediction dirs:", sorted(preds), flush=True)
cfgs = ["V31_ANGULAR=0", "V31_ANGULAR=0,R20_K4=pattern"]
for name in ("fable_ceiling_all_k4pred_ARCH_J6", "fable_ceiling_all_k4pred_ARCH_J4", "fable_ceiling_all_k4pred_ARCH_r64_J6", "fable_ceiling_all_k4pred_ARCH_r64_ry16_J4"):
    if name in preds:
        cfgs.append(f"V31_ANGULAR=0,R20_K4=pattern,OPUS_TEACHER={preds[name]}")
env = dict(os.environ, PYTHONUTF8="1", TAG="fable_e2e", NETS="mlp0")
sys.exit(subprocess.run([sys.executable, f"{src}/run_variant_teacher.py", f"{src}/r20_joint_oracle.py"] + cfgs, cwd=f"{base}/run", env=env).returncode)
