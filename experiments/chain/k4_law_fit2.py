"""Second-stage fit: can the part of the true kappa4 (3,1) slice that V29's lambda*C_off misses be built from objects the chain
ALREADY carries at that layer - the third-cumulant (2,1) slice kappa3(i,i,j) = E[y_i^2 y_j] (the teacher's Y21), the covariance,
the kappa4 diagonal and the mean? All candidate terms are n^2 elementwise products (free next to an n^3 matmul).
Usage: python k4_law_fit2.py <dir> [seed]
"""
import sys, glob, os, numpy as np
d = sys.argv[1]; seed = sys.argv[2] if len(sys.argv) > 2 else "0"
files = sorted(glob.glob(os.path.join(d, f"layer*_seed{seed}.npz")), key=lambda f: int(os.path.basename(f).split("_")[0][5:]))
def r2(y, X):
    c, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - np.sum((y - X @ c) ** 2) / np.sum(y ** 2), c
print(f"{'layer':>5} {'V29 lam*Coff':>13} {'+K4d terms':>11} {'+kappa3 (2,1)':>14} {'+k3 x mean/var':>15} {'all':>7}   (R^2 of the true (3,1) slice)")
for f in files:
    z = np.load(f); l = int(os.path.basename(f).split("_")[0][5:])
    if "Y21" not in z.files: print(f"L{l}: no Y21 in this dump (rerun the teacher)"); continue
    C = z["C"].astype(np.float64); K31 = z["K31"].astype(np.float64); K4d = z["K4d"]; Y21 = z["Y21"].astype(np.float64); m = z["mean"]
    n = C.shape[0]; var = np.diag(C).copy(); Coff = C - np.diag(var); off = ~np.eye(n, dtype=bool)
    ki = np.repeat(K4d[:, None], n, 1); mi = np.repeat(m[:, None], n, 1); vi = np.repeat(var[:, None], n, 1)
    y = K31[off]
    base = [Coff]
    k4t = base + [ki, ki * Coff / vi]
    k3t = base + [Y21, Y21 * mi / vi]
    k3m = k3t + [Y21 * mi, Y21 / np.sqrt(vi), Y21 * np.sqrt(vi)]
    allt = k4t + [Y21, Y21 * mi / vi, Y21 * mi, Coff ** 2 / np.sqrt(np.outer(var, var)), vi]
    vals = [r2(y, np.stack([t[off] for t in feats], 1))[0] for feats in (base, k4t, k3t, k3m, allt)]
    print(f"L{l:>4} " + " ".join(f"{v:13.4f}" for v in vals))
