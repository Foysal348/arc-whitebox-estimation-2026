"""Build submissions/<name>/estimator.py = submissions/e30kc1n2t/estimator.py with a new mean-correction table baked in as
Python literals (no json import). If the table uses the N3 modulators (st2 / s3st / sl), the e30k_x12 patch is applied too.
Usage: python kaggle/x12/build_sub.py <table.json> <out_dir> "<header line>"
"""
import os, sys, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
tab = json.load(open(sys.argv[1])); outd = sys.argv[2]; header = sys.argv[3]
SRC = os.path.join(ROOT, "submissions/e30kc1n2t")
src = open(os.path.join(SRC, "estimator.py"), encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, (c, old[:80])
    src = src.replace(old, new)


feats, beta = list(tab["feats"]), [[float(x) for x in r] for r in tab["beta"]]
assert len(beta) == 16 and all(len(r) == len(feats) for r in beta)
lines = src.split("\n")
i_f = [k for k, l in enumerate(lines) if l.startswith("CORR_FEATS = [") and "X10_TAB" in lines[k - 1]]
assert len(i_f) == 1, i_f
k = i_f[0]
assert lines[k + 1].startswith("CORR_BETA_NEW = [[")
lines[k] = "CORR_FEATS = " + repr(feats)
lines[k + 1] = "CORR_BETA_NEW = " + repr(beta)
src = "\n".join(lines)
rep("X10_TAB = ''   # e30kc1n2: refitted table N2 baked in below as literals (no json)",
    "X10_TAB = ''   # %s: refitted table baked in below as literals (no json)" % os.path.basename(outd.rstrip("/\\")))
n3 = any(nm.endswith(("*st2", "*s3st", "*sl")) for nm in feats)
if n3:
    rep('''or any(nm.endswith(("*st", "*sd")) for nm in (CORR_FEATS or [])))''',
        '''or any(nm.endswith(("*st", "*sd", "*st2", "*s3st", "*sl")) for nm in (CORR_FEATS or [])))''')
    rep('''    _MODS = ["", "s3", "s4", "s33", "dn", "v", "ds", "cr", "c2", "k4", "st", "sd"]''',
        '''    _MODS = ["", "s3", "s4", "s33", "dn", "v", "ds", "cr", "c2", "k4", "st", "sd", "st2", "s3st", "sl"]''')
    rep('''              "st": 0.0, "sd": 2.0}''', '''              "st": 0.0, "sd": 2.0, "st2": 0.0, "s3st": 1.5, "sl": 2.0}''')
    rep('''            x10_st = None
            last = li == L - 1''', '''            x10_st = None
            x12_g4l = None
            last = li == L - 1''')
    rep('''                    g4ns = g4ns * (fnp.mean(g4row) / fnp.mean(g4ns))
''', '''                    g4ns = g4ns * (fnp.mean(g4row) / fnp.mean(g4ns))
                    if X10_STMOD:
                        x12_g4l = g4ns
''')
    rep('''                    if "s33" in _FZ["mods"]:
                        fm["s33"] = D3f * D3f
''', '''                    if "s33" in _FZ["mods"]:
                        fm["s33"] = D3f * D3f
                    if "st2" in _FZ["mods"]:
                        fm["st2"] = (x10_st * x10_st) if x10_st is not None else zeros_n
                    if "s3st" in _FZ["mods"]:
                        fm["s3st"] = (D3f * x10_st) if x10_st is not None else zeros_n
                    if "sl" in _FZ["mods"]:
                        fm["sl"] = (fnp.subtract(x12_g4l, g4row) if (x12_g4l is not None and g4row is not None) else zeros_n)
''')
src = "# " + header + "\n" + src
assert "import json" not in src and "json." not in src.replace("_x10json", "")
os.makedirs(outd, exist_ok=True)
open(os.path.join(outd, "estimator.py"), "w", encoding="utf-8", newline="\n").write(src)
shutil.copy(os.path.join(SRC, "LICENSE"), os.path.join(outd, "LICENSE"))
print("wrote", os.path.join(outd, "estimator.py"), "feats", len(feats), "N3 modulators" if n3 else "")
