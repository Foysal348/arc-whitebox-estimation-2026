"""Builds experiments/variants/e30k_x12.py from e30k_x10.py (X12, 20 Oct): three extra table modulators for the N3 fit
  st2  = st^2                    (E 0)    h_k*st2
  s3st = D3 * st  (/ var^1.5)    (E 1.5)  h_k*s3st
  sl   = g4ns - g4row (/ var^2)  (E 2)    h_k*sl   (the corrector's local part, star coefficient zeroed, before scaling)
and the X10 dump also records g4l = g4ns (zeros where the split corrector is off). Defaults (no new features in the
table) are untouched: only names that appear in the table are computed."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
src = open(os.path.join(ROOT, "experiments/variants/e30k_x10.py"), encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, (c, old)
    src = src.replace(old, new)


rep('''# e30k_x10 (20 Oct): e30k_x8 + X10_* knobs''', '''# e30k_x12 (20 Oct): e30k_x10 + N3 modulators st2 / s3st / sl (kaggle/x12/patch_x12.py); dump records g4l.
# e30k_x10 (20 Oct): e30k_x8 + X10_* knobs''')
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
rep('''                             g4c=(g4mean if (mode == 1 and g4mean is not None) else zeros_n), g4=_g4d, var=var,''',
    '''                             g4c=(g4mean if (mode == 1 and g4mean is not None) else zeros_n), g4=_g4d, var=var,
                             g4l=(x12_g4l if (mode == 1 and x12_g4l is not None) else _g4d),''')
open(os.path.join(ROOT, "experiments/variants/e30k_x12.py"), "w", encoding="utf-8", newline="\n").write(src)
print("wrote e30k_x12.py", len(src))
