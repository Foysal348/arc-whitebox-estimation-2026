"""Build experiments/variants/estimator_v31ang.py from submissions/v29m8f5/estimator.py: the ANGULAR (spherical-input) variant.
X = (R/sqrt n) Y with Y uniform on the sphere of radius sqrt n, R ~ chi_n independent; the bias-free ReLU net is degree-1
homogeneous, so E[h_l(X)] = a1 E[h_l(Y)], a1 = E[R/sqrt n] = sqrt(2/n) Gamma((n+1)/2)/Gamma(n/2).  Y has C = I, kappa3 = 0 and
kappa4(Y) = c Sym3(I x I), c = -2/(n+2).  In V29's harmonic (matrix-core) representation this is a previous-layer core with
g_prev = -6/(n+2) (constant), var_prev = 1, lambda_prev = g_prev, so the layer-0 pre-activation core is g * C_pre exactly:
dG = g var, off-diagonal g C_off.  The patch runs V29's own kappa4 terms at layer 0 with these values and rescales the output.
Env knobs of the built file:  V31_ANGULAR=1 (kappa4 injection at layer 0), V31_A1=1 (output rescaling by a1). Default both on.
"""
import re
src = open("submissions/v29m8f5/estimator.py", encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)

# knobs + constants
rep('STRASSEN_LEVELS = int(_os.environ.get("V26_STRASSEN", "5"))',
    'ANGULAR = _os.environ.get("V31_ANGULAR", "1") == "1"   # V31: spherical-input reparameterization (kappa4 injection at layer 0)\n'
    'ANG_A1 = _os.environ.get("V31_A1", "1") == "1"          # V31: output rescaling by a1 = E[R/sqrt n]\n'
    'import math as _math\n'
    'STRASSEN_LEVELS = int(_os.environ.get("V26_STRASSEN", "5"))')
# term program mode 3: layer-0 program plus the kappa4 terms
rep('MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4",\n                 "wk431"}',
    'MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4",\n                 "wk431"}\n'
    'MODE3_MISSING = {"d21", "d21T", "d3col", "d3row", "d3"}   # V31: layer 0 with the input kappa4 terms')
rep('    missing = MODE0_MISSING if mode == 0 else ({"wk431"} if mode == 2 else set())',
    '    missing = MODE0_MISSING if mode == 0 else (MODE3_MISSING if mode == 3 else ({"wk431"} if mode == 2 else set()))')
rep('        self._progs = {0: _term_prog(0), 1: _term_prog(1), 2: _term_prog(2)}',
    '        self._progs = {0: _term_prog(0), 1: _term_prog(1), 2: _term_prog(2), 3: _term_prog(3)}')
# layer-0 kappa4 slices from the spherical input
rep('            else:\n                D3 = D21 = g4row = wk4m = None\n',
    '            else:\n                D3 = D21 = g4row = wk4m = None\n'
    '                ang0 = ANGULAR and riders and regen and not trim\n'
    '                if ang0:\n'
    '                    # V31: exact layer-0 core of the spherical input, g C_pre with g = -6/(n+2)\n'
    '                    ang_g = -6.0 / (n + 2.0)\n'
    '                    dG = var * ang_g\n'
    '                    g4row = dG * METRIC_C\n'
    '                    g22c = (dG * (METRIC_C / 6.0))[:, None]\n'
    '                    wk4m = _zero_diag(fnp.add(g22c, g22c.T, out=NN("wk4m")))\n'
    '                    wk431 = fnp.multiply(C_off, (0.5 * METRIC_C * ang_g), out=NN("wk431"))\n')
rep('            pmode = mode if (mode == 0 or (regen and not NO_WK431)) else 2',
    '            pmode = mode if (mode == 0 or (regen and not NO_WK431)) else 2\n'
    '            if mode == 0 and ANGULAR and riders and regen and not trim:\n'
    '                pmode = 3')
rep('                b1.update({"d3": D3, "g4": g4row})\n            ab_cache = {}',
    '                b1.update({"d3": D3, "g4": g4row})\n'
    '            elif pmode == 3:\n'
    '                g4col = fnp.multiply((g4row)[:, None], ones_row, out=NN("g4col"))\n'
    '                b2.update({"wk4m": wk4m, "g4col": g4col, "g4row": g4col.T, "wk431": wk431})\n'
    '                b1.update({"g4": g4row})\n'
    '            ab_cache = {}')
# output rescaling
rep('        return fnp.stack(rows, axis=0)',
    '        out = fnp.stack(rows, axis=0)\n'
    '        if ANGULAR and ANG_A1 and riders:\n'
    '            a1 = _math.exp(_math.lgamma((n + 1) / 2.0) - _math.lgamma(n / 2.0)) * _math.sqrt(2.0 / n)\n'
    '            out = out * float(a1)\n'
    '        return out')
open("experiments/variants/estimator_v31ang.py", "w", encoding="utf-8").write(src)
import math; n = 1024
print("a1 =", math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2)) * math.sqrt(2 / n), " g =", -6 / (n + 2))
print("written experiments/variants/estimator_v31ang.py")
