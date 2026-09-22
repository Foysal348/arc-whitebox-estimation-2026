"""Astra GitHub-audit diagnostic: split each OLD source's D21 readout into the covariance-path family (AAP: 2H + T2) and the
repeated-index family (MPP: T3/3 + 2T4/3), following V29's own formulas
   H = (A o P o w2) A^T, T2 = (A o A o w2) P^T, T3 = (P o P) M^T, T4 = (M o P) P^T,  M = P diag(s) + 3 A diag(e) + Z L^T
Reports per layer: norms of the two families (aggregate over old sources), their cosine, each family's share of the total,
the same for the diagonal (D3-type: 3 sum_j w2 A^2 P + sum_j M P^2), and the fluctuation parts after the column-mean subtraction.
Usage: python probe_aap_mpp.py
"""
import numpy as np
def fl(X): return X - X.mean(axis=0)[None, :]
for L in (10, 14):
    d = np.load(f"data/dumps/query_dump_full_L{L}.npz"); ka = int(d["ka"]); n = d["A0"].shape[0]
    AAP = np.zeros((n, n)); MPP = np.zeros((n, n)); d3a = np.zeros(n); d3m = np.zeros(n); rows = []
    for s in range(ka):
        A = d[f"A{s}"].astype(np.float64); P = d[f"P{s}"].astype(np.float64); w2 = d[f"w2_{s}"]
        sv = d[f"s_{s}"]; ev = d[f"e_{s}"]; Z = d[f"Z{s}"].astype(np.float64); Lm = d[f"L{s}"].astype(np.float64)
        M = P * sv[None, :] + 3.0 * A * ev[None, :] + Z @ Lm.T
        H = (A * P * w2[None, :]) @ A.T; T2 = (A * A * w2[None, :]) @ P.T
        T3 = (P * P) @ M.T; T4 = (M * P) @ P.T
        aap = 2 * H + T2; mpp = T3 / 3 + 2 * T4 / 3
        AAP += aap; MPP += mpp
        d3a += 3 * np.sum(w2[None, :] * A * A * P, axis=1); d3m += np.sum(M * P * P, axis=1)
        rows.append((np.linalg.norm(aap), np.linalg.norm(mpp), np.sum(aap * mpp) / (np.linalg.norm(aap) * np.linalg.norm(mpp))))
    tot = AAP + MPP
    cos = np.sum(AAP * MPP) / (np.linalg.norm(AAP) * np.linalg.norm(MPP))
    print(f"L{L} ({ka} old sources): |AAP| {np.linalg.norm(AAP):.3e} |MPP| {np.linalg.norm(MPP):.3e} |total| {np.linalg.norm(tot):.3e}  cos(AAP,MPP) {cos:+.3f}")
    print(f"      share of the total D21 norm: AAP {np.linalg.norm(AAP)/np.linalg.norm(tot):.2f}, MPP {np.linalg.norm(MPP)/np.linalg.norm(tot):.2f}; dropping MPP -> rel err {np.linalg.norm(MPP)/np.linalg.norm(tot):.2f}, dropping AAP -> {np.linalg.norm(AAP)/np.linalg.norm(tot):.2f}")
    print(f"      fluctuation parts: |fl(AAP)| {np.linalg.norm(fl(AAP)):.3e} |fl(MPP)| {np.linalg.norm(fl(MPP)):.3e} cos {np.sum(fl(AAP)*fl(MPP))/(np.linalg.norm(fl(AAP))*np.linalg.norm(fl(MPP))):+.3f}")
    print(f"      diagonal (D3): |AAP part| {np.linalg.norm(d3a):.3e} |MPP part| {np.linalg.norm(d3m):.3e} cos {np.dot(d3a,d3m)/(np.linalg.norm(d3a)*np.linalg.norm(d3m)):+.3f}")
    print("      per source (|aap|, |mpp|, cos):", " ".join(f"({a:.1e},{m:.1e},{c:+.2f})" for a, m, c in rows))
