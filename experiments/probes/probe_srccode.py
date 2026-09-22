"""Round-3 falsifier (GPT-5.6 Sol high, 21 Sep): cubic SOURCE-INDEX CODING of the old tier.
Per old source s (dense legs A_s, P_s and birth weights w2_s from the layer dumps) the two D21-type contributions are
   H_s  = (A_s o P_s) diag(w2_s) A_s^T        T2_s = (A_s o A_s) diag(w2_s) P_s^T
both cubic in the legs. With i.i.d. codes z_s (E z = 0, E z^3 = 1) and pooled legs A^(q) = sum_s z_s A_s (same for P and for
the w2-scaled copies), E_z Phi(pooled) = sum_s Phi_s exactly; the estimate averages q channels.
Control variate (GPT's variance control): the rank-1 column-mean part 1 m^T of every Hadamard product is computed exactly
per source and only the centered fluctuation is coded.
Reports the relative Frobenius error of the coded estimate vs the exact old-tier sum, for q = 1..3 channels and for the
hybrid (p exact sources + q coded), with 32 independent channel draws. Kill criterion (GPT): > 2% at either layer.
Usage: python probe_srccode.py
"""
import numpy as np, time
PHI = (1 + 5 ** 0.5) / 2
def codes(rng, m):
    # two-point skew code: phi w.p. 0.276393, -1/phi otherwise: E z = 0, E z^2 = E z^3 = 1
    u = rng.random(m) < 0.2763932
    return np.where(u, PHI, -1 / PHI)

def hadamard_split(X, Y):
    """X o Y = 1 m^T + F ; returns (m, F)"""
    XY = X * Y; m = XY.mean(axis=0); return m, XY - m[None, :]

for L in (10, 14):
    d = np.load(f"data/dumps/query_dump_L{L}.npz"); ka = int(d["ka"])
    A = [d[f"A{s}"].astype(np.float64) for s in range(ka)]
    P = [d[f"P{s}"].astype(np.float64) for s in range(ka)]
    w2 = [d[f"w2_{s}"].astype(np.float64) for s in range(ka)]
    n = A[0].shape[0]
    Aw = [A[s] * w2[s][None, :] for s in range(ka)]   # A diag(w2)
    Pw = [P[s] * w2[s][None, :] for s in range(ka)]
    t0 = time.time()
    # exact totals and exact thin (rank-1) parts
    H_exact = np.zeros((n, n)); T2_exact = np.zeros((n, n)); H_thin = np.zeros((n, n)); T2_thin = np.zeros((n, n))
    Hs = []
    for s in range(ka):
        mH, FH = hadamard_split(A[s], P[s]); mT, FT = hadamard_split(A[s], A[s])
        h = (A[s] * P[s]) @ Aw[s].T; Hs.append(h)
        H_exact += h; T2_exact += (A[s] * A[s]) @ Pw[s].T
        H_thin += np.outer(np.ones(n), mH @ Aw[s].T); T2_thin += np.outer(np.ones(n), mT @ Pw[s].T)
    H_fluct = H_exact - H_thin; T2_fluct = T2_exact - T2_thin
    print(f"L{L}: {ka} old sources, |H| {np.linalg.norm(H_exact):.3e} |H_fluct| {np.linalg.norm(H_fluct):.3e} (thin part is {1-np.linalg.norm(H_fluct)/np.linalg.norm(H_exact):.1%} of the norm); "
          f"per-source |H_s| {[f'{np.linalg.norm(h):.2e}' for h in Hs]}  ({time.time()-t0:.0f}s)", flush=True)
    order = np.argsort([-np.linalg.norm(h) for h in Hs])   # for the hybrid: peel the largest sources
    rng = np.random.default_rng(0)
    NCH = 32
    for p in (0, 1, 2):
        exact_idx = list(order[:p]); coded_idx = [s for s in range(ka) if s not in exact_idx]
        if len(coded_idx) < 2: continue
        # exact part of the hybrid
        H_ex = sum(Hs[s] for s in exact_idx) if exact_idx else np.zeros((n, n))
        T2_ex = sum((A[s] * A[s]) @ Pw[s].T for s in exact_idx) if exact_idx else np.zeros((n, n))
        # thin parts of the coded sources (added back exactly)
        H_thin_c = np.zeros((n, n)); T2_thin_c = np.zeros((n, n))
        for s in coded_idx:
            mH, _ = hadamard_split(A[s], P[s]); mT, _ = hadamard_split(A[s], A[s])
            H_thin_c += np.outer(np.ones(n), mH @ Aw[s].T); T2_thin_c += np.outer(np.ones(n), mT @ Pw[s].T)
        chH = []; chT = []
        for c in range(NCH):
            z = codes(rng, len(coded_idx))
            Ap = sum(z[i] * A[s] for i, s in enumerate(coded_idx)); Pp = sum(z[i] * P[s] for i, s in enumerate(coded_idx))
            Awp = sum(z[i] * Aw[s] for i, s in enumerate(coded_idx)); Pwp = sum(z[i] * Pw[s] for i, s in enumerate(coded_idx))
            mH, FH = hadamard_split(Ap, Pp); mT, FT = hadamard_split(Ap, Ap)
            chH.append(FH @ Awp.T); chT.append(FT @ Pwp.T)      # coded centered fluctuation only
        chH = np.array(chH); chT = np.array(chT)
        for q in (1, 2, 3):
            errsH = []; errsT = []
            for k in range(0, NCH - q + 1, q):
                estH = H_ex + H_thin_c + chH[k:k + q].mean(axis=0); estT = T2_ex + T2_thin_c + chT[k:k + q].mean(axis=0)
                errsH.append(np.linalg.norm(estH - H_exact) / np.linalg.norm(H_exact))
                errsT.append(np.linalg.norm(estT - T2_exact) / np.linalg.norm(T2_exact))
            print(f"L{L}: p={p} exact + q={q} coded ({len(coded_idx)} sources pooled): rel err H {np.median(errsH):.3f} (min {min(errsH):.3f})   T2 {np.median(errsT):.3f} (min {min(errsT):.3f})   [budget p+q={p+q}: {2.04*(p+q):.1f} u/layer]", flush=True)
        # sanity: the coded estimator is unbiased -> error of the 32-channel mean should be ~1/sqrt(32) of q=1
        estH = H_ex + H_thin_c + chH.mean(axis=0)
        print(f"L{L}: p={p} 32-channel mean rel err H {np.linalg.norm(estH - H_exact)/np.linalg.norm(H_exact):.3f}  (unbiasedness check; expect ~q=1 error / 5.7)", flush=True)
