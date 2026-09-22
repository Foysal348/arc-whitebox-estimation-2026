"""Column screening of young sources: H_s = (A o P o w2) A^T from the top-q fraction of birth columns j
ranked by column energy e_j = w2_j^2 |a_j|^2 |p_j|^2. Reports relative error of H, T2, D3 vs fraction kept."""
import numpy as np, sys
for L in (6, 10, 14):
    d = np.load(f"data/dumps/young_dump_L{L}.npz"); ka = int(d["ka"]); k = int(d["k"])
    print(f"layer {L}: slots 0..{k} (old < {ka}, young {ka}..{k-1}, newborn {k})")
    for s in range(ka, k + 1):
        A = d[f"A{s}"].astype(np.float64); P = d[f"P{s}"].astype(np.float64); w2 = d[f"w2_{s}"]
        e = (w2**2) * np.sum(A*A, axis=0) * np.sum(P*P, axis=0); order = np.argsort(e)[::-1]
        H = ((A*P)*w2) @ A.T; T2 = ((A*A)*w2) @ P.T; D3 = np.sum((A*A)*P*w2, axis=1)
        nH, nT, nD = np.linalg.norm(H), np.linalg.norm(T2), np.linalg.norm(D3)
        row = []
        for q in (0.5, 0.6, 0.7, 0.8, 0.9):
            keep = np.zeros(A.shape[1], bool); keep[order[:int(q*A.shape[1])]] = True
            w = w2 * keep
            Hq = ((A*P)*w) @ A.T; Tq = ((A*A)*w) @ P.T; Dq = np.sum((A*A)*P*w, axis=1)
            row.append(f"q={q:.1f}: H {np.linalg.norm(H-Hq)/nH:.3f} T2 {np.linalg.norm(T2-Tq)/nT:.3f} D3 {np.linalg.norm(D3-Dq)/nD:.3f}")
        ce = np.cumsum(np.sort(e)[::-1]) / np.sum(e)
        print(f"  slot {s:2d} (age {L - s}): energy in top 50%/70%/90% cols = {ce[511]:.3f}/{ce[716]:.3f}/{ce[921]:.3f} | " + " | ".join(row))
