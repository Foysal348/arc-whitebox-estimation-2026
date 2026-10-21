"""X14 (21 Oct, experiment 14) source patches, shared by the variant builder (patch_x14.py -> experiments/variants/e30k_x14.py)
and the submission builder (build_sub14.py, on submissions/e30kc1n3/estimator.py).

A) N4 table modulators (each enters the table as h_k * m, k = 0..6, like the N3 ones; only computed if the table uses them):
   zr   = sum_j D21_ij^2 / var_j            (/ var_i^2  -> E 2)   row energy of the standardised D21
   zc   = sum_j D21_ji^2 / var_j^2          (/ var_i    -> E 1)   column energy of the standardised D21
   rc2  = sum_j C_off_ij^2 / var_j          (/ var_i    -> E 1)   C_pre correlation-row energy
   ast  = |alpha| * st                      (E 0)
   pdl  = W @ delta_prev                    (/ var^0.5  -> E 0.5) previous-layer table delta carried to this layer
   pst  = W @ st_prev                       (E 0)                 previous-layer corrector star carried to this layer
   All are n^2 (no n^3 work). The X10 dump records the raw sums zr, zc, rc2, pdl, pst.
B) X14_T21=1: the second (2,1) use of the (3,1) slot (wl = (3,2), wr = (1,1)) reads wk431^T (forum topic 18246; al07's
   EF_K31T21 in external/al07_two_term/v29_experiments/estimator_v29_ef.py). Our X2_K31 two-term slot is asymmetric:
   wk431_ij = K(j^3, i) = b1 v_j C_ij + b2 D3_j D21[j, i] / v_j (mixed 0.4 with the symmetric e30_a slot).
"""

NEW_MODS = ["zr", "zc", "rc2", "ast", "pdl", "pst"]
NEW_E = {"zr": 2.0, "zc": 1.0, "rc2": 1.0, "ast": 0.0, "pdl": 0.5, "pst": 0.0}


def _rep(src, old, new, n=1):
    c = src.count(old)
    assert c == n, (c, old[:120])
    return src.replace(old, new)


def apply(src, dump=True, t21_default="0"):
    r = _rep
    # --- knobs
    src = r(src, '''or any(nm.endswith(("*st", "*sd", "*st2", "*s3st", "*sl")) for nm in (CORR_FEATS or [])))''',
            '''or any(nm.endswith(("*st", "*sd", "*st2", "*s3st", "*sl", "*ast", "*pst")) for nm in (CORR_FEATS or [])))''')
    src = r(src, '''_X10_REC = None
''', '''_X10_REC = None
# X14 (21 Oct): T21 orientation fix of the (3,1) slot and the N4 table modulators (kaggle/x14/x14_patches.py)
X14_T21 = _os.environ.get("X14_T21", "%s") == "1"   # (2,1) use with wl=(3,2), wr=(1,1) reads wk431^T
X14_PREV = X10_DUMP or any(nm.endswith(("*pdl", "*pst")) for nm in (CORR_FEATS or []))
''' % t21_default)
    # --- B) T21
    src = r(src, '''TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))''',
            '''TERM_SPECS[(2, 1)].append(('wk431T' if X14_T21 else 'wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))''')
    src = r(src, '''                 "wk431"}''', '''                 "wk431", "wk431T"}''')
    src = r(src, '''{"wk431"} if mode == 2''', '''{"wk431", "wk431T"} if mode == 2''')
    src = r(src, '''                        b2["wk431"] = wk431
''', '''                        b2["wk431"] = wk431
                        if X14_T21:
                            b2["wk431T"] = wk431.T
''')
    # --- A) N4 modulators
    src = r(src, '''    _MODS = ["", "s3", "s4", "s33", "dn", "v", "ds", "cr", "c2", "k4", "st", "sd", "st2", "s3st", "sl"]''',
            '''    _MODS = ["", "s3", "s4", "s33", "dn", "v", "ds", "cr", "c2", "k4", "st", "sd", "st2", "s3st", "sl",
             "zr", "zc", "rc2", "ast", "pdl", "pst"]''')
    src = r(src, '''              "st": 0.0, "sd": 2.0, "st2": 0.0, "s3st": 1.5, "sl": 2.0}''',
            '''              "st": 0.0, "sd": 2.0, "st2": 0.0, "s3st": 1.5, "sl": 2.0,
              "zr": 2.0, "zc": 1.0, "rc2": 1.0, "ast": 0.0, "pdl": 0.5, "pst": 0.0}''')
    src = r(src, '''        w1_prev = None  # wick w(1) of the previous layer, folded into WD
''', '''        w1_prev = None  # wick w(1) of the previous layer, folded into WD
        x14_dprev = x14_stprev = None   # X14: previous layer's table delta / corrector star
''')
    src = r(src, '''            mu = W @ mu
''', '''            mu = W @ mu
            x14_pdl = (W @ x14_dprev) if (X14_PREV and x14_dprev is not None) else None
            x14_pst = (W @ x14_stprev) if (X14_PREV and x14_stprev is not None) else None
''')
    src = r(src, '''            mu = K1v if delta is None else K1v + delta
''', '''            mu = K1v if delta is None else K1v + delta
            if X14_PREV:
                x14_dprev = delta
                x14_stprev = x10_st
''')
    src = r(src, '''                        fm["sl"] = (fnp.subtract(x12_g4l, g4row) if (x12_g4l is not None and g4row is not None) else zeros_n)
''', '''                        fm["sl"] = (fnp.subtract(x12_g4l, g4row) if (x12_g4l is not None and g4row is not None) else zeros_n)
                    if "zr" in _FZ["mods"] or "zc" in _FZ["mods"]:
                        if mode == 1 and D21 is not None:
                            _x14q = D21 * D21
                            _x14iv = 1.0 / var
                            fm["zr"] = _x14q @ _x14iv
                            fm["zc"] = (_x14iv * _x14iv) @ _x14q
                        else:
                            fm["zr"] = fm["zc"] = zeros_n
                    if "rc2" in _FZ["mods"]:
                        fm["rc2"] = ((C_off * C_off) @ (1.0 / var)) if C_off is not None else zeros_n
                    if "ast" in _FZ["mods"]:
                        fm["ast"] = (fnp.abs(alpha) * x10_st) if x10_st is not None else zeros_n
                    if "pdl" in _FZ["mods"]:
                        fm["pdl"] = x14_pdl if x14_pdl is not None else zeros_n
                    if "pst" in _FZ["mods"]:
                        fm["pst"] = x14_pst if x14_pst is not None else zeros_n
''')
    if dump:
        src = r(src, '''                    _cr2 = fnp.einsum("ij,ij->i", C_off, C_off) if C_off is not None else zeros_n
''', '''                    _cr2 = fnp.einsum("ij,ij->i", C_off, C_off) if C_off is not None else zeros_n
                    _x14iv = 1.0 / var
                    if mode == 1 and D21 is not None:
                        _x14q = D21 * D21
                        _x14zr = _x14q @ _x14iv
                        _x14zc = (_x14iv * _x14iv) @ _x14q
                    else:
                        _x14zr = _x14zc = zeros_n
                    _x14rc2 = ((C_off * C_off) @ _x14iv) if C_off is not None else zeros_n
''')
        src = r(src, '''alpha=alpha, sp=sigma * phi, delta=delta,''',
                '''alpha=alpha, sp=sigma * phi, delta=delta,
                             zr=_x14zr, zc=_x14zc, rc2=_x14rc2,
                             pdl=(x14_pdl if x14_pdl is not None else zeros_n), pst=(x14_pst if x14_pst is not None else zeros_n),''')
    return src
