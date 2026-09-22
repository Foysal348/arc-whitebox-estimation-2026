"""Fitted coefficients of the improved (3,1) law across layers (and their stability), plus the same for (2,2).
Law tested:  K31[i,j] ~ lam * C_off + b1 * K4d_i + b2 * K4d_i C_off/var_i + b3 * kappa3(i,i,j) + b4 * kappa3(i,i,j) mean_i/var_i
Everything is normalised by the layer's own scale so the coefficients are comparable across depth.
Usage: python k4_law_coef.py <dir> [seed]
"""
import sys, glob, os, numpy as np
d = sys.argv[1]; seed = sys.argv[2] if len(sys.argv) > 2 else "0"
files = sorted(glob.glob(os.path.join(d, f"layer*_seed{seed}.npz")), key=lambda f: int(os.path.basename(f).split("_")[0][5:]))
print(f"{'layer':>5} {'lam(C_off)':>11} {'b1(K4d_i)':>11} {'b2':>9} {'b3(kappa3)':>11} {'b4':>9}   R^2   | ratios: mean(K4d)/mean(var), |k3|/|C_off|")
for f in files:
    z = np.load(f); l = int(os.path.basename(f).split("_")[0][5:])
    C = z["C"].astype(np.float64); K31 = z["K31"].astype(np.float64); K4d = z["K4d"]; Y21 = z["Y21"].astype(np.float64); m = z["mean"]
    n = C.shape[0]; var = np.diag(C).copy(); Coff = C - np.diag(var); off = ~np.eye(n, dtype=bool)
    ki = np.repeat(K4d[:, None], n, 1); mi = np.repeat(m[:, None], n, 1); vi = np.repeat(var[:, None], n, 1)
    F = np.stack([Coff[off], ki[off], (ki * Coff / vi)[off], Y21[off], (Y21 * mi / vi)[off]], 1)
    y = K31[off]
    c, *_ = np.linalg.lstsq(F, y, rcond=None)
    r2 = 1 - np.sum((y - F @ c) ** 2) / np.sum(y ** 2)
    print(f"L{l:>4} " + " ".join(f"{x:11.4e}" for x in c) + f"  {r2:.3f}  | {K4d.mean()/var.mean():.4e}  {np.sqrt((Y21[off]**2).mean())/np.sqrt((Coff[off]**2).mean()):.3f}")
