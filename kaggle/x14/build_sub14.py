"""Build submissions/<name>/estimator.py from submissions/e30kc1n3/estimator.py (current best, N3 baked in) + X14 patches
(no dump code) + an optional new table baked in as literals (no json) + X14_T21 default.
Usage: python kaggle/x14/build_sub14.py <table.json or -> <out_dir> <t21 0|1> "<header line>"
"""
import os, sys, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import x14_patches
tabf, outd, t21, header = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SRC = os.path.join(ROOT, "submissions/e30kc1n3")
src = open(os.path.join(SRC, "estimator.py"), encoding="utf-8").read()
src = x14_patches.apply(src, dump=False, t21_default=t21)
if tabf != "-":
    tab = json.load(open(tabf))
    feats, beta = list(tab["feats"]), [[float(x) for x in r] for r in tab["beta"]]
    assert len(beta) == 16 and all(len(r) == len(feats) for r in beta)
    lines = src.split("\n")
    i_f = [k for k, l in enumerate(lines) if l.startswith("CORR_FEATS = [") and lines[k - 1].startswith("X10_TAB = ''")]
    assert len(i_f) == 1, i_f
    k = i_f[0]
    assert lines[k + 1].startswith("CORR_BETA_NEW = [[")
    lines[k] = "CORR_FEATS = " + repr(feats)
    lines[k + 1] = "CORR_BETA_NEW = " + repr(beta)
    src = "\n".join(lines)
    src = src.replace("X10_TAB = ''   # e30kc1n3: refitted table", "X10_TAB = ''   # %s: refitted table" % os.path.basename(outd.rstrip("/\\")), 1)
src = "# " + header + "\n" + src
assert "import json" not in src and "json." not in src.replace("_x10json", "")
os.makedirs(outd, exist_ok=True)
open(os.path.join(outd, "estimator.py"), "w", encoding="utf-8", newline="\n").write(src)
shutil.copy(os.path.join(SRC, "LICENSE"), os.path.join(outd, "LICENSE"))
print("wrote", os.path.join(outd, "estimator.py"), "T21 default", t21, "table", tabf)
