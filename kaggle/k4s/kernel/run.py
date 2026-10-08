# V56 star-feature dump on Kaggle (CPU work; validated locally at width 256).
import glob, os, subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flopscope>=0.12.1,<0.13", "whestbench>=0.16.1,<0.17"],
                     capture_output=True, text=True).stdout[-300:], flush=True)
d = os.path.dirname(glob.glob("/kaggle/input/**/k4s_dump.py", recursive=True)[0])
r = subprocess.run([sys.executable, os.path.join(d, "k4s_dump.py"), d, "/kaggle/working/k4s_feats.npz"])
sys.exit(r.returncode)
