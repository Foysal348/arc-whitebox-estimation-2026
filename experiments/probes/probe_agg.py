"""Pilot D: three measurements requested by the round-2 consultation, from the layer dumps of probe_query.py.
 (1) cross-source response rank: SVD across old sources of their contracted queries U_s (GPT-5.6 high / Terra)
 (2) vector-metric ("aggregate response state") fit at the gate: T_s = sum_j w2_j a_j (x) a_j (x) p_j ~ sum_a G_a (x) v_a
     via the exact (ab)|c unfolding SVD computed through the n x n Gram  S = P diag(w2) [(A^T A) o (A^T A)] diag(w2) P^T
     (Gemini R2 / GPT-5.6 high); report Frobenius truncation and the errors of the three queries H, T2, D3 for k = 1..64
 (3) TT / Tucker ranks of the POOLED old-source core in the shared basis: T[a,b,c] = sum_s sum_j w2_j FA_aj FA_bj FP_cj (r^3, exact)
     (Terra / DeepSeek): singular spectra of the unfoldings (a)|(bc) and (ab)|(c)
Usage: python probe_agg.py query_dump_L14.npz
"""
import sys, time, numpy as np
f = sys.argv[1]; d = np.load(f)
Qc, _ = np.linalg.qr(d["Qc"]); n, r = Qc.shape; ka = int(d["ka"]); kb = int(d["kb"])
print(f"{f}: n={n} r={r} old sources={ka} (tier2 {kb})", flush=True)
srcs = []
for s in range(ka):
    A = d[f"A{s}"].astype(np.float64); P = d[f"P{s}"].astype(np.float64); w2 = d[f"w2_{s}"]
    srcs.append((A, P, w2, Qc.T @ A, Qc.T @ P))

def energy_ranks(sv, qs=(0.9, 0.99, 0.999, 0.9996)):
    e = np.cumsum(sv**2) / np.sum(sv**2)
    return {q: int(np.searchsorted(e, q) + 1) for q in qs}

# ---------- (1) cross-source response rank
t0 = time.time()
Us = [((A * P) @ (w2[:, None] * FA.T)) for A, P, w2, FA, FP in srcs]        # n x r each: H-type query
X = np.stack([u.ravel() / np.linalg.norm(u) for u in Us], axis=1)           # (n r) x ka, unit-normalized per source
sv = np.linalg.svd(X, compute_uv=False); e = np.cumsum(sv**2) / np.sum(sv**2)
print(f"(1) cross-source SVD of normalized U_s: energy after 1/2/3/4 modes = {e[0]:.4f} / {e[min(1,ka-1)]:.4f} / {e[min(2,ka-1)]:.4f} / {e[min(3,ka-1)]:.4f}  (target 0.9996 for 2%)  [{time.time()-t0:.0f}s]", flush=True)
Xraw = np.stack([u.ravel() for u in Us], axis=1); sv2 = np.linalg.svd(Xraw, compute_uv=False); e2 = np.cumsum(sv2**2)/np.sum(sv2**2)
print(f"    same, unnormalized (energy-weighted): {e2[0]:.4f} / {e2[min(1,ka-1)]:.4f} / {e2[min(2,ka-1)]:.4f} / {e2[min(3,ka-1)]:.4f}", flush=True)

# ---------- (2) vector-metric fit for single sources (youngest old, oldest old) and for the pooled tensor
def vm_fit(pairs, label):
    """pairs: list of (A, P, w2). Unfolding M = sum_s sum_j w2_j vec(a_j a_j^T) p_j^T ; Gram = M^T M (n x n)."""
    t0 = time.time()
    S = np.zeros((n, n))
    for (A1, P1, w1) in pairs:
        for (A2, P2, w2b) in pairs:
            G = A1.T @ A2                                   # n x n (birth j of s1, birth l of s2)
            S += (P1 * w1) @ (G * G) @ (P2 * w2b).T
    S = 0.5 * (S + S.T)
    lam, V = np.linalg.eigh(S); order = np.argsort(lam)[::-1]; lam = lam[order]; V = V[:, order]
    tot = np.sum(lam); fro_err = lambda k: np.sqrt(max(0.0, 1 - np.sum(lam[:k]) / tot))
    # exact queries (summed over pairs)
    H = sum(((A * P) * w2) @ A.T for A, P, w2 in pairs)          # H_ic = sum_j w2 A_ij P_ij A_cj = T_{i c i}
    T2 = sum(((A * A) * w2) @ P.T for A, P, w2 in pairs)         # T2_ic = sum_j w2 A_ij^2 P_cj = T_{i i c}
    D3 = sum(np.sum((A * A) * P * w2, axis=1) for A, P, w2 in pairs)   # T_{iii}
    nH, nT2, nD3 = np.linalg.norm(H), np.linalg.norm(T2), np.linalg.norm(D3)
    print(f"(2) {label}: Gram built [{time.time()-t0:.0f}s]; unfolding singular energy ranks (of {n}): " +
          str(energy_ranks(np.sqrt(np.maximum(lam, 0)))), flush=True)
    Hk = np.zeros((n, n)); T2k = np.zeros((n, n)); D3k = np.zeros(n); kprev = 0
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        for a in range(kprev, k):
            v = V[:, a]
            # G_a = M v_a reshaped = sum_s sum_j w2_j (p_j . v) a_j a_j^T
            Ga = sum(A @ ((w2 * (P.T @ v))[:, None] * A.T) for A, P, w2 in pairs)   # n x n symmetric
            Hk += Ga * v[:, None]                       # H_ic += (G_a)_ic (v_a)_i
            T2k += np.outer(np.diag(Ga), v)             # T2_ic += (G_a)_ii (v_a)_c
            D3k += np.diag(Ga) * v
        kprev = k
        print(f"    k={k:3d}: tensor-Frobenius err {fro_err(k):.4f} | query err H {np.linalg.norm(H-Hk)/nH:.4f}  T2 {np.linalg.norm(T2-T2k)/nT2:.4f}  D3 {np.linalg.norm(D3-D3k)/nD3:.4f}", flush=True)

A, P, w2, FA, FP = srcs[ka - 1]; vm_fit([(A, P, w2)], f"youngest old source (slot {ka-1})")
A, P, w2, FA, FP = srcs[0];      vm_fit([(A, P, w2)], "oldest old source (slot 0)")
vm_fit([(A, P, w2) for A, P, w2, FA, FP in srcs], f"POOLED {ka} old sources")

# ---------- (3) TT / Tucker ranks of the pooled core in the shared basis
t0 = time.time()
T = np.zeros((r * r, r))
for A, P, w2, FA, FP in srcs:
    KR = (FA[:, None, :] * FA[None, :, :]).reshape(r * r, n)      # (ab) x j  : FA_aj FA_bj
    T += KR @ (w2[:, None] * FP.T)                                   # (ab) x c
core = T.reshape(r, r, r)
sv_ab_c = np.linalg.svd(T, compute_uv=False)                         # (ab)|(c) unfolding
sv_a_bc = np.linalg.svd(core.reshape(r, r * r), compute_uv=False)    # (a)|(bc) unfolding
print(f"(3) pooled core in Qc basis built [{time.time()-t0:.0f}s]; ||T||={np.linalg.norm(T):.3e}")
print(f"    unfolding (ab)|(c) energy ranks (of {r}): {energy_ranks(sv_ab_c)}")
print(f"    unfolding (a)|(bc) energy ranks (of {r}): {energy_ranks(sv_a_bc)}")
for k in (16, 32, 64, 128, 256):
    e1 = np.sqrt(max(0, 1 - np.sum(sv_ab_c[:k]**2) / np.sum(sv_ab_c**2))); e2 = np.sqrt(max(0, 1 - np.sum(sv_a_bc[:k]**2) / np.sum(sv_a_bc**2)))
    print(f"    TT rank {k:3d}: Frobenius err (ab)|c {e1:.4f}   (a)|bc {e2:.4f}")
print("done")
