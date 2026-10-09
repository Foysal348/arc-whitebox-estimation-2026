# One-step channel errors of Hermite-K ReLU responses vs the exact ReLU (r20_onestep_channels.py), validated locally at N=4000.
import glob, os, subprocess, sys
src = glob.glob("/kaggle/input/**/r20_onestep_channels.py", recursive=True)[0]
data = glob.glob("/kaggle/input/**/r17_data.npz", recursive=True)[0]
env = dict(os.environ, R17_DATA=data, R20_OUT="/kaggle/working/r20_onestep_channels", R20_LAYERS="2,4,8,12,15")
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
sys.exit(subprocess.run([sys.executable, src, "2e6", "cuda"], env=env).returncode)
