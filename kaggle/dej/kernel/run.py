# DEJ-4 falsifier (10 Oct 2026): D21 / d3 error when the youngest `young` sources are kept exact and every older source is
# projected (from its EXACT legs, i.e. the best case of an exact age-(young+1) reset) onto a rank-r basis. young = 1..4.
import glob, os, subprocess, sys
code = os.path.dirname(glob.glob("/kaggle/input/**/pool_ceiling.py", recursive=True)[0])
caps = os.path.dirname(glob.glob("/kaggle/input/**/r7_cap_L14.npz", recursive=True)[0])
rc = 0
for y in (1, 2, 3, 4):
    rc |= subprocess.run([sys.executable, f"{code}/pool_ceiling.py", "--layers", "11,14,15", "--ranks", "352,448,512", "--young", str(y),
                          "--cap", caps, "--out", f"/kaggle/working/dej_young{y}.json"]).returncode
sys.exit(rc)
