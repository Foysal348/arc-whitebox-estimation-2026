# Round 17 Q1 hub-pair carrier ceiling (Claude's falsifier with the project metric), validated locally at N=4000.
import glob, subprocess, sys
src = glob.glob("/kaggle/input/**/r17_q1_hubpair_falsifier.py", recursive=True)[0]
data = glob.glob("/kaggle/input/**/r17_data.npz", recursive=True)[0]
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
r = subprocess.run([sys.executable, src, "--weights", data, "--teacher-npz", data, "--N", "2000000", "--device", "cuda",
                    "--out", "/kaggle/working/r17_q1_hubpair.json"])
sys.exit(r.returncode)
