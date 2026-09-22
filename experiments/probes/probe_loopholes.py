"""Closes three loopholes from UNTESTED_IDEAS (B1, A4, B4) using the existing layer dumps.
 B1: metric-side siblings T2 = (A o A) diag(w2) P^T and (P o P) Q-projected: singular energy ranks (pooled old sources)
 A4: concentration of the closure coefficients w2 across neurons (does 10% of rows carry 90% of sum w2^2 ?)
 B4: best m deterministic 'chaos' directions for the old-source quadratic response: project U (n x r) on its own top-m
     LEFT singular directions across BOTH layers 10 and 14 (a fixed basis must serve every layer): error at m = 32/64/96
Usage: python probe_loopholes.py
"""
import numpy as np
def ranks(sv, qs=(0.9, 0.99, 0.999)):
    e = np.cumsum(sv**2) / np.sum(sv**2); return {q: int(np.searchsorted(e, q) + 1) for q in qs}
Us = {}
for L in (10, 14):
    d = np.load(f"data/dumps/query_dump_L{L}.npz"); Qc, _ = np.linalg.qr(d["Qc"]); n, r = Qc.shape; ka = int(d["ka"])
    A = [d[f"A{s}"].astype(np.float64) for s in range(ka)]; P = [d[f"P{s}"].astype(np.float64) for s in range(ka)]
    w2 = [d[f"w2_{s}"] for s in range(ka)]
    FA = [Qc.T @ a for a in A]; FP = [Qc.T @ p for p in P]
    U = sum(((A[s] * P[s]) * w2[s]) @ FA[s].T for s in range(ka))           # H-family, n x r
    T2 = sum(((A[s] * A[s]) * w2[s]) @ FP[s].T for s in range(ka))          # sibling with P on the right, n x r
    PP = sum(((P[s] * P[s]) * w2[s]) @ FA[s].T for s in range(ka))          # (P o P)-type sibling (M ~ A-type right factor as proxy)
    print(f"L{L}: B1 sibling energy ranks (of {r}):  U(H) {ranks(np.linalg.svd(U, compute_uv=False))}   T2 {ranks(np.linalg.svd(T2, compute_uv=False))}   PP {ranks(np.linalg.svd(PP, compute_uv=False))}")
    # A4: w2 concentration (per source at this layer, birth-index coefficients)
    for s in (0, ka - 1):
        w = np.sort(np.abs(w2[s]))[::-1]; c = np.cumsum(w**2) / np.sum(w**2)
        print(f"L{L}: A4 w2 concentration, source slot {s}: top 10% of neurons carry {c[int(0.1*n)-1]:.2f} of sum w2^2; top 50% carry {c[int(0.5*n)-1]:.2f}; max/median |w2| = {w[0]/np.median(w):.2f}")
    Us[L] = U
# B4: a fixed deterministic direction basis must serve both layers: fit on L10, test on L14 (and vice versa)
for fit, test in ((10, 14), (14, 10)):
    Uf, Ut = Us[fit], Us[test]
    Vl, sv, _ = np.linalg.svd(Uf, full_matrices=False)          # left singular directions (n-space) of the fit layer
    for m in (32, 64, 96, 128, 256):
        Pm = Vl[:, :m]; err = np.linalg.norm(Ut - Pm @ (Pm.T @ Ut)) / np.linalg.norm(Ut)
        err_in = np.linalg.norm(Uf - Pm @ (Pm.T @ Uf)) / np.linalg.norm(Uf)
        print(f"B4 basis from L{fit} applied to L{test}: m={m:3d}  in-sample err {err_in:.3f}  transfer err {err:.3f}")
