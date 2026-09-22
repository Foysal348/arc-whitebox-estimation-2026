"""Fit V29's kappa4 ansatz against the TRUE cumulant slices measured by k4teacher.py.
V29 (F68) models the kappa4 matrix core as G = diag(g) + lambda_l * C_off and feeds the Wick terms with
    g4row_i  ~ dG_i                      (the (4,) / diagonal channel)
    wk4m_ij  ~ (dG_i + dG_j) * c / 6     (the (2,2) slice, i != j)
    wk431_ij ~ (c / 2) * lambda * C_off  (the (3,1) slice)
so the ansatz to test on the true slices is
    K22[i,j] ~ a (K4d_i + K4d_j) + b C_off[i,j]                     (+ richer terms)
    K31[i,j] ~ p K4d_i + q C_off[i,j] (+ r K4d_i C_off[i,j]/var ...)
Reports, per layer, the R^2 of V29's two-term form and of richer but still n^2-cheap forms, plus the MC noise floor.
Usage: python k4_law_fit.py <dir> [seed]
"""
import sys, glob, os, numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "kaggle_k4teach/_local"; seed = sys.argv[2] if len(sys.argv) > 2 else "0"
files = sorted(glob.glob(os.path.join(d, f"layer*_seed{seed}.npz")), key=lambda f: int(os.path.basename(f).split("_")[0][5:]))
def r2(y, X):
    c, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - np.sum((y - X @ c) ** 2) / np.sum(y ** 2), c
for f in files:
    z = np.load(f); l = int(os.path.basename(f).split("_")[0][5:])
    C = z["C"].astype(np.float64); K22 = z["K22"].astype(np.float64); K31 = z["K31"].astype(np.float64); K4d = z["K4d"]; N = int(z["N"])
    n = C.shape[0]; var = np.diag(C).copy(); Coff = C - np.diag(var)
    off = ~np.eye(n, dtype=bool)
    # MC noise floor: Var(y_i^2 y_j^2)/N ~ (3 var_i var_j)^2 * (something); use the Gaussian-surrogate estimate
    noise22 = np.sqrt((3.0 * np.outer(var, var) ** 2 * 8).mean() / N)
    noise31 = np.sqrt((15.0 * var ** 3 * var).mean() / N)
    s, sq = K4d[:, None] + K4d[None, :], np.sqrt(np.outer(var, var))
    feats22 = {"V29: (K4d_i + K4d_j), C_off": [s, Coff],
               "+ C_off^2 / sqrt(var var)": [s, Coff, Coff ** 2 / sq],
               "+ (K4d_i+K4d_j) * C_off/sqrt(vv), sqrt(vv)": [s, Coff, Coff ** 2 / sq, s * Coff / sq, sq]}
    feats31 = {"V29: C_off (times a per-layer scalar)": [Coff],
               "+ K4d_i, K4d_i C_off/var_i": [Coff, np.repeat(K4d[:, None], n, 1), np.repeat(K4d[:, None], n, 1) * Coff / var[:, None]],
               "+ C_off^2/sqrt(vv), var_i": [Coff, np.repeat(K4d[:, None], n, 1), np.repeat(K4d[:, None], n, 1) * Coff / var[:, None], Coff ** 2 / sq, np.repeat(var[:, None], n, 1)]}
    y22 = K22[off]; y31 = K31[off]
    out22 = {k: r2(y22, np.stack([f_[off] for f_ in v], 1))[0] for k, v in feats22.items()}
    out31 = {k: r2(y31, np.stack([f_[off] for f_ in v], 1))[0] for k, v in feats31.items()}
    print(f"L{l:2d}  |K22off| rms {np.sqrt((y22**2).mean()):.3e} (MC noise ~{noise22:.1e})   |K31| rms {np.sqrt((y31**2).mean()):.3e} (noise ~{noise31:.1e})   kappa4 diag mean {K4d.mean():.3e}")
    for k, v in out22.items(): print(f"      K22 R^2 {v:7.4f}   {k}")
    for k, v in out31.items(): print(f"      K31 R^2 {v:7.4f}   {k}")
