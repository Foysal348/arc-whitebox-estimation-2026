# Round 17 family ceiling on GPU: runs r17_hermite_family_ceiling.py from the attached dataset (set C (births per age), validated locally at N=2500).
import glob, os, subprocess, sys
src = glob.glob("/kaggle/input/**/r17_hermite_family_ceiling.py", recursive=True)[0]
data = glob.glob("/kaggle/input/**/r17_data.npz", recursive=True)[0]
env = dict(os.environ, R17_DATA=data, R17_OUT="/kaggle/working/r17_hermite_family_ceiling_C", R17_SET="C")
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
r = subprocess.run([sys.executable, src, "4e6", "cuda"], env=env)
sys.exit(r.returncode)
