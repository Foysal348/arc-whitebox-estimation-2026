"""Round-4 falsifier (GPT-5.6 Sol GitHub report, finding #2): query-space row skeleton (ISDF/THC analogue).
Per old source s the contracted queries are U_H,s = ((A_s o P_s) w2_s) F_A,s^T and U_T,s = ((A_s o A_s) w2_s) F_P,s^T (n x r),
F = Qc^T (leg).  If a shared LEFT basis Q_U (n x m) reconstructs the aggregate query, only m interpolation rows per source
need forming (m r n instead of n^2 r).  The basis must be available WITHOUT computing the aggregate:
  (1) oracle: left SVD of the current aggregate (upper bound only)
  (2) anchor: left SVD of ONE source's query (the youngest old source, then the largest), applied to the others / the aggregate
  (3) previous-layer: left basis of the aggregate at layer l-1, transported by M_l = W_l D (the source transport) and re-orthonormalised
PASS gate (GPT): m <= 256 and every channel <= 2% relative RMS with a non-oracle basis.
Usage: python probe_isdf.py   (needs query_dump_L9/10/13/14.npz and the MLP weights -> rebuilt from local_engine's recipe)
"""
import numpy as np, os, sys
n = 1024
rng = np.random.default_rng(7); scale = (2.0 / n) ** 0.5
Wts = [(rng.standard_normal((n, n)) * scale).astype(np.float32).astype(np.float64) for _ in range(16)]   # local_engine.build_mlp(seed=7)

def queries(f):
    d = np.load(f); ka = int(d["ka"]); Qc, _ = np.linalg.qr(d["Qc"].astype(np.float64))
    out = []
    for s in range(ka):
        A = d[f"A{s}"].astype(np.float64); P = d[f"P{s}"].astype(np.float64); w2 = d[f"w2_{s}"].astype(np.float64)
        FA = Qc.T @ A; FP = Qc.T @ P
        out.append((((A * P) * w2[None, :]) @ FA.T, ((A * A) * w2[None, :]) @ FP.T))
    return out

def left_basis(X, m):
    U, s, _ = np.linalg.svd(X, full_matrices=False); return U[:, :m]
def err(Q, X):
    return np.linalg.norm(X - Q @ (Q.T @ X)) / np.linalg.norm(X)

for lp, l in ((9, 10), (13, 14)):
    prev = queries(f"data/dumps/query_dump_L{lp}.npz"); cur = queries(f"data/dumps/query_dump_L{l}.npz")
    UH = sum(u for u, _ in cur); UT = sum(t for _, t in cur)
    UHp = sum(u for u, _ in prev); UTp = sum(t for _, t in prev)
    ka = len(cur)
    # source transport between the dumps: legs4 at layer l = W_l (w1-scaled legs of l-1); the D scaling is a right/row scaling that
    # does not change the left subspace of a transported leg family -> transport the basis by W (the raw weight of layer l)
    W = Wts[l]     # estimator layer li uses mlp.weights[li]
    print(f"\nlayer {l} ({ka} old sources; queries n x {UH.shape[1]}): aggregate |U_H| {np.linalg.norm(UH):.3e} |U_T| {np.linalg.norm(UT):.3e}", flush=True)
    for m in (128, 192, 256, 320, 384):
        e_or = (err(left_basis(UH, m), UH), err(left_basis(UT, m), UT))
        # anchor = youngest old source (index ka-1) and largest-norm source
        big = int(np.argmax([np.linalg.norm(u) for u, _ in cur]))
        e_young = (err(left_basis(cur[ka - 1][0], m), UH), err(left_basis(cur[ka - 1][1], m), UT))
        e_big = (err(left_basis(cur[big][0], m), UH), err(left_basis(cur[big][1], m), UT))
        # joint basis of the two youngest old sources
        e_two = (err(left_basis(np.hstack([cur[ka - 1][0], cur[ka - 2][0]]), m), UH), err(left_basis(np.hstack([cur[ka - 1][1], cur[ka - 2][1]]), m), UT))
        # previous layer transported
        QpH = left_basis(UHp, m); QpT = left_basis(UTp, m)
        QtH, _ = np.linalg.qr(W @ QpH); QtT, _ = np.linalg.qr(W @ QpT)
        e_prev = (err(QtH, UH), err(QtT, UT))
        QrH, _ = np.linalg.qr(W.T @ QpH); QrT, _ = np.linalg.qr(W.T @ QpT)     # in case the convention is transposed
        e_prevT = (err(QrH, UH), err(QrT, UT))
        print(f"  m={m:3d}: oracle H/T {e_or[0]:.3f}/{e_or[1]:.3f} | anchor youngest {e_young[0]:.3f}/{e_young[1]:.3f} | anchor largest {e_big[0]:.3f}/{e_big[1]:.3f} | two youngest {e_two[0]:.3f}/{e_two[1]:.3f} | prev-layer W {e_prev[0]:.3f}/{e_prev[1]:.3f} | prev-layer W^T {e_prevT[0]:.3f}/{e_prevT[1]:.3f}", flush=True)
