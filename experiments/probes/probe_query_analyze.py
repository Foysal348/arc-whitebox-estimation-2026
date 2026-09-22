"""Pilot C step 2: how compressible is the contracted old-source query U = sum_s (A_s o P_s) diag(w2_s) FA_s^T ?
Surrogates (all avoid forming A o P):
  trace   : U_ic ~ (|q_i|^2 / r) * tr(S_c),  S_c = sum_j w2_j f_cj a_j p_j^T   (isotropic rows)
  diag    : U ~ (Q o Q) D,  D = (FA o FP) diag(w2) FA^T                          (diagonal Hermite in Q coords)
  sketch_d: TensorSketch of degree 2: U ~ Phi Psi diag(w2) FA^T with Phi = CS(q_i (x) q_i) (n x d), Psi = CS(a_j (x) p_j) (d x n)
  svd_k   : best rank-k approximation of the exact U (ceiling for any linear compression)
Reports relative RMS error ||U_exact - U_approx||_F / ||U_exact||_F, aggregated over all old sources, and per-source.
Usage: python probe_query_analyze.py query_dump_L14.npz
"""
import sys, numpy as np
rng = np.random.default_rng(0)
f = sys.argv[1]
d = np.load(f)
Qc_raw = d["Qc"]; n, r = Qc_raw.shape; ka = int(d["ka"]); kb = int(d["kb"])
Qc, _R = np.linalg.qr(Qc_raw)                  # orthonormalize the (transported, non-orthonormal) basis; span unchanged
print(f"{f}: n={n} r={r} old sources={ka} (tier2 {kb})")

def exact_U(A, P, w2, FA):
    return (A * P) @ (w2[:, None] * FA.T)             # (n, r): (A o P) diag(w2) FA^T

def count_sketch(X, d, h, s):
    """Row-wise CountSketch of X (m x r): each column a -> bucket h[a] with sign s[a]."""
    out = np.zeros((X.shape[0], d))
    np.add.at(out.T, h, (X * s).T)
    return out
def tensor_sketch_rows(X, Y, dsk, seed):
    """TensorSketch of the rows x_i (x) y_i of X, Y (m x r) -> (m x dsk), via FFT convolution."""
    g = np.random.default_rng(seed)
    h1, h2 = g.integers(0, dsk, r), g.integers(0, dsk, r)
    s1, s2 = g.choice([-1.0, 1.0], r), g.choice([-1.0, 1.0], r)
    F1 = np.fft.rfft(count_sketch(X, dsk, h1, s1), axis=1)
    F2 = np.fft.rfft(count_sketch(Y, dsk, h2, s2), axis=1)
    return np.fft.irfft(F1 * F2, n=dsk, axis=1)

srcs = []
for s in range(ka):
    A = d[f"A{s}"].astype(np.float64); P = d[f"P{s}"].astype(np.float64); w2 = d[f"w2_{s}"]
    FA = Qc.T @ A; FP = Qc.T @ P                        # exact factors (legs lie in span Qc)
    rec = np.linalg.norm(Qc @ FA - A) / np.linalg.norm(A)
    srcs.append((A, P, w2, FA, FP, rec))
print("leg reconstruction error from Qc factors (should be ~1e-6):", ["%.1e" % x[5] for x in srcs])

U_ex = sum(exact_U(A, P, w2, FA) for A, P, w2, FA, FP, _ in srcs)
nrm = np.linalg.norm(U_ex)
per = [np.linalg.norm(exact_U(A, P, w2, FA)) / nrm for A, P, w2, FA, FP, _ in srcs]
print("||U_exact|| =", "%.3e" % nrm, " per-source share:", ["%.2f" % x for x in per])
def rel(Ua): return np.linalg.norm(U_ex - Ua) / nrm

# ceiling: best rank-k of U itself
sv = np.linalg.svd(U_ex, compute_uv=False)
e = np.cumsum(sv**2) / np.sum(sv**2)
print("U_exact energy ranks: r90 %d  r99 %d  r999 %d  (of %d)" % (tuple(int(np.searchsorted(e, q) + 1) for q in (0.9, 0.99, 0.999)) + (r,)))
for kk in (16, 32, 64, 128, 256):
    print(f"  svd_k={kk:3d}: rel err {np.sqrt(max(0.0, 1 - e[min(kk, r) - 1])):.4f}")

# trace surrogate (isotropic rows): U_ic ~ (|q_i|^2/r) tr(S_c);  tr(S_c) = sum_j w2_j f_cj (a_j . p_j)
lev = np.sum(Qc * Qc, axis=1)                          # |q_i|^2
U_tr = np.zeros_like(U_ex); U_dg = np.zeros_like(U_ex)
for A, P, w2, FA, FP, _ in srcs:
    ap = np.sum(FA * FP, axis=0)                        # a_j . p_j  (r-dim columns)
    U_tr += np.outer(lev / r, FA @ (w2 * ap))           # (n,) x (r,)
    Dm = (FA * FP) @ (w2[:, None] * FA.T)               # (r, r): D_ac = sum_j w2_j f_cj a_aj p_aj
    U_dg += (Qc * Qc) @ Dm
print(f"trace surrogate : rel err {rel(U_tr):.4f}")
print(f"diag-Hermite    : rel err {rel(U_dg):.4f}")

for dsk in (16, 32, 64, 128, 256, 512, 1024, 2048):
    errs = []
    for seed in range(5):
        U_sk = np.zeros_like(U_ex)
        Phi = tensor_sketch_rows(Qc, Qc, dsk, seed)                       # (n, d)
        for A, P, w2, FA, FP, _ in srcs:
            Psi = tensor_sketch_rows(FA.T, FP.T, dsk, seed)               # (n, d): sketch of a_j (x) p_j
            U_sk += Phi @ (Psi.T @ (w2[:, None] * FA.T))                  # (n,d)(d,n)(n,r)
        errs.append(rel(U_sk))
    print(f"tensor-sketch d={dsk:5d}: rel err median {np.median(errs):.4f}  (min {min(errs):.4f}, max {max(errs):.4f})")
