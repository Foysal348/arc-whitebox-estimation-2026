"""Round-3 falsifier (Astra 6 Max §5 / GPT-5.6 Sol §2, 21 Sep): COMPRESS THE BIRTH OPERATOR.
In V29 both legs of a source share the transport, so at every later layer A_s = P_s a_b exactly, where a_b is the birth
matrix (born as w1 * C_off). The proposal: a_b ~ diag(d) + U V^T with rank k <= 32-40, so that the transported A leg is
P diag(d) + (P U) V^T and the source costs ~1.03-1.19 u per layer instead of 2.05.
Test: recover a_b by least squares from the EARLIEST dump in which the source has dense (young) legs; fit d, U, V there
(Astra's recipe: d = diag(a), randomized rank-k SVD of a - diag(d) with oversampling 16 and one power iteration, then reset
d = diag(a - U V^T)); evaluate at layers 10 and 14 with the cached legs P_s, without refitting:
   H_s = (A_s o P_s) diag(w2_s) A_s^T ,  T2_s = (A_s o A_s) diag(w2_s) P_s^T   (exact vs. approximated A_s)
Reports per-source and aggregate relative Frobenius errors of the totals and of the centred fluctuation parts (same exact
column means m subtracted in both). Kill: > 2% aggregate at either layer for k <= 40.
Usage: python probe_birthcomp.py
"""
import numpy as np, time

def load(f):
    d = np.load(f); k = int(d["k"]) + 1
    return {s: (d[f"A{s}"].astype(np.float64), d[f"P{s}"].astype(np.float64), d[f"w2_{s}"].astype(np.float64)) for s in range(k) if f"A{s}" in d.files}

def rsvd(X, k, over=16, power=1, rng=np.random.default_rng(0)):
    G = rng.standard_normal((X.shape[1], k + over)); Y = X @ G
    for _ in range(power):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y); B = Q.T @ X
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub[:, :k]) * s[:k], Vt[:k].T

def fit(a, k):
    d = np.diag(a).copy()
    if k == 0:
        return d, None, None
    U, V = rsvd(a - np.diag(d), k)
    d = np.diag(a - U @ V.T).copy()
    return d, U, V

def fluct(X):
    return X - X.mean(axis=0)[None, :]

young6 = load("data/dumps/young_dump_L6.npz"); young10 = load("data/dumps/young_dump_L10.npz")
q10 = load("data/dumps/query_dump_L10.npz"); q14 = load("data/dumps/query_dump_L14.npz")
ka10, ka14 = 6, 10
# 1. TRUE birth matrices a_b = diag(w1) C_off dumped from the estimator (probe_birth_dump.py); slot s = birth at layer s
bd = np.load("data/dumps/birth_dump.npz")
births = {s: bd[f"a{s}"].astype(np.float64) for s in range(ka14)}
for L, q, ka in ((10, q10, ka10), (14, q14, ka14)):
    for s in range(ka):
        A, P, _ = q[s]
        print(f"L{L} slot {s}: consistency |A - P a_b| / |A| = {np.linalg.norm(P @ births[s] - A) / np.linalg.norm(A):.3e}", flush=True)
# 2. fit at birth for several ranks, 3. evaluate at layers 10 and 14
for k in (16, 32, 64, 128, 256, 384):
    fits = {s: fit(births[s], k) for s in range(ka14)}
    for L, q, ka in ((10, q10, ka10), (14, q14, ka14)):
        H_ex = np.zeros((1024, 1024)); H_ap = np.zeros_like(H_ex); T_ex = np.zeros_like(H_ex); T_ap = np.zeros_like(H_ex)
        per = []
        for s in range(ka):
            A, P, w2 = q[s]
            # exact consistency: the dumped (confined) A vs P a_b
            d, U, V = fits[s]
            Aap = P * d[None, :] + ((P @ U) @ V.T if U is not None else 0.0)
            Aex = P @ births[s]     # exact transported birth (through the same confined P)
            h_ex = (Aex * P) @ (Aex * w2[None, :]).T; h_ap = (Aap * P) @ (Aap * w2[None, :]).T
            t_ex = (Aex * Aex) @ (P * w2[None, :]).T; t_ap = (Aap * Aap) @ (P * w2[None, :]).T
            per.append(np.linalg.norm(h_ap - h_ex) / np.linalg.norm(h_ex))
            H_ex += h_ex; H_ap += h_ap; T_ex += t_ex; T_ap += t_ap
        eH = np.linalg.norm(H_ap - H_ex) / np.linalg.norm(H_ex); eT = np.linalg.norm(T_ap - T_ex) / np.linalg.norm(T_ex)
        eHf = np.linalg.norm(fluct(H_ap) - fluct(H_ex)) / np.linalg.norm(fluct(H_ex))
        eD = np.linalg.norm(np.diag(H_ap) - np.diag(H_ex)) / np.linalg.norm(np.diag(H_ex))
        print(f"k={k:3d} L{L}: aggregate rel err  H {eH:.3f}  H-fluct {eHf:.3f}  diag(H) {eD:.3f}  T2 {eT:.3f}   per-source H err {[f'{x:.2f}' for x in per]}", flush=True)
