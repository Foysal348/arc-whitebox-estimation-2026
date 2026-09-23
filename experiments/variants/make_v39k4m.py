"""Build experiments/variants/estimator_v39k4m.py: per-layer multiplier on the chain's kappa4 DIAGONAL (zero FLOPs).

Finding (probe_errbudget.py, 23 Sep): against the Monte-Carlo teacher, the chain's pre-activation kappa4 diagonal
(g4row = dG * METRIC_C, already in kappa4 units: regression slope 0.98 at layer 2) is systematically too SMALL at
depth. The least-squares multiplier <chain, true> / <chain, chain> rises smoothly with depth and is the same on two
unrelated networks (public MLP #0 and a synthetic He network):

    layer   2      4      6      8      10     12     14     15     16
    MLP #0  1.007  1.003  1.019  1.037  1.064  1.091  1.170  1.183  1.206
    synth   1.007  1.010  1.023  1.034  1.065  1.127  1.169  1.185   -

The table below is their mean (odd layers from MLP #0). Knobs:
    V39_K4MULT  strength s: multiplier = 1 + s (T[l] - 1); 0 = inert (bit-identical to the parent)
    V39_K4MODE  mean | all   mean: only the kappa4 terms of the MEAN closure use the corrected diagonal
                             all:  g4row itself is corrected (mean, second-moment and covariance terms, and through
                                   the post-ReLU closure also the next layer's regenerated core)
"""
SRC = "submissions/v31ang/estimator.py"
DST = "experiments/variants/estimator_v39k4m.py"
s = open(SRC, encoding="utf-8").read()

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)

T = [1.0, 1.007, 1.005, 1.007, 1.011, 1.021, 1.034, 1.036, 1.058, 1.065, 1.072, 1.109, 1.132, 1.170, 1.184, 1.206]

# ---- knobs + table ----------------------------------------------------------------------------------------------
rep('ANGULAR = _os.environ.get("V31_ANGULAR", "1") == "1"',
    'K4MULT_S = float(_os.environ.get("V39_K4MULT", "0"))   # V39: strength of the kappa4-diagonal depth correction\n'
    'K4MODE = _os.environ.get("V39_K4MODE", "mean")\n'
    f'K4MULT_T = {T!r}\n'
    'ANGULAR = _os.environ.get("V31_ANGULAR", "1") == "1"')

# ---- mean-closure kappa4 terms read their own key 'g4m' ------------------------------------------------------------
rep("TERM_SPECS = {(1,): [('ones1', 'ones1', 0, None, 1.0),\n        ('d3', 'ones1', 8, None, 0.16666666666666666),\n        ('g4', 'ones1', 12, None, 0.041666666666666664),\n        ('d3', 'd3', 18, None, 0.013888888888888888),\n        ('d3', 'g4', 20, None, 0.006944444444444444)],",
    "TERM_SPECS = {(1,): [('ones1', 'ones1', 0, None, 1.0),\n        ('d3', 'ones1', 8, None, 0.16666666666666666),\n        ('g4m', 'ones1', 12, None, 0.041666666666666664),\n        ('d3', 'd3', 18, None, 0.013888888888888888),\n        ('d3', 'g4m', 20, None, 0.006944444444444444)],")
rep('MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4",',
    'MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4", "g4m",')

# ---- 'all' mode: correct g4row where it is formed (regen branch, both code paths) --------------------------------------
rep("""                    g4row = dG * METRIC_C
                    wk4m = wk431 = None""",
    """                    g4row = dG * METRIC_C
                    if K4MULT_S and K4MODE == "all":
                        g4row = g4row * float(1.0 + K4MULT_S * (K4MULT_T[min(li, 15)] - 1.0))
                    wk4m = wk431 = None""")
rep("""                    g4row = dG * METRIC_C
                    g22c = (dG * (METRIC_C / 6.0))[:, None]
                    wk4m = _zero_diag(fnp.add(g22c, g22c.T, out=NN("wk4m")))
                    wk431 = None if trim""",
    """                    g4row = dG * METRIC_C
                    if K4MULT_S and K4MODE == "all":
                        g4row = g4row * float(1.0 + K4MULT_S * (K4MULT_T[min(li, 15)] - 1.0))
                    g22c = (dG * (METRIC_C / 6.0))[:, None]
                    wk4m = _zero_diag(fnp.add(g22c, g22c.T, out=NN("wk4m")))
                    wk431 = None if trim""")

# ---- the mean-closure key ------------------------------------------------------------------------------------------
rep("""                b1.update({"d3": D3, "g4": g4row})""",
    """                _m = float(1.0 + K4MULT_S * (K4MULT_T[min(li, 15)] - 1.0)) if (K4MULT_S and K4MODE == "mean") else 1.0
                b1.update({"d3": D3, "g4": g4row, "g4m": (g4row * _m) if _m != 1.0 else g4row})""")
rep("""                b1.update({"g4": g4row})""",
    """                b1.update({"g4": g4row, "g4m": g4row})""")

open(DST, "w", encoding="utf-8").write(s)
import ast; ast.parse(s)
print("wrote", DST)
