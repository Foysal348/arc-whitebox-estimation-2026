"""Round-6 probe (23 Sep): measure old-tier (and young-tier) compressions in the DOWNSTREAM-QUERY metric, not Frobenius.

The final-mean response to a perturbation dD21 of the (2,1) slice at estimator layer li (post-activation layer p = li+1) is,
to first order in the Gaussian chain (validated in probe_jac_val / probe_adjoint_val at R^2 0.996-0.9999),
    Q(dD21) = G_16 o diag(K_{p->16} dC K^T) + sum_{p<l'<16} d_16 o K_{l'->16} [ G_l' o diag(K_{p->l'} dC K^T) ]
    dC = (1/2)(c_p o dD21 o d_p^T + transpose),   K_{p->l'} = W_l'^T D_{l'-1} ... D_{p+1} W_{p+1}^T
with c = density of z at 0 (E phi''), d = P(z>0) (E phi'), G = dE[relu]/dvar. This script
  1. rebuilds Q for the seed-7 network (weights from data/dumps/jac_val.npz) and validates it against the exact chain
     responses stored in jac_val.npz (random dC at li=10,14) and adjoint_val_preds.npz (old tier dropped at li=10,14);
  2. computes the PER-COLUMN query responses R_j of every source's contraction (D21_s = LA_s A_s^T + LP_s P_s^T is a sum
     over birth columns j), which gives the coherence ratio |sum_j R_j|^2 / sum_j |R_j|^2 and hence the EXACT expected
     squared error of any Horvitz-Thompson column sampler in the query metric (no sampling needed);
  3. evaluates the query-subspace restriction D21 -> D21 V V^T (V = top-rho right singular subspace of K_{p->16} diag(d_p),
     or its one-pass backward sketch) for old, young and all sources, against the deterministic baselines (shared-basis
     rank truncation 384->320/256, per-source rank truncation), all in the same metric, with FLOP counts.
Usage (repository root, system python):  python experiments/probes/probe_r6_query.py [stages=val,cols,sub,rank] [layers=6,10,14]
"""
import os, sys, time, json
import numpy as np

n, L = 1024, 16
STAGES = (sys.argv[1] if len(sys.argv) > 1 else "val,cols,sub,rank").split(",")
LAYERS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "6,10,14").split(",")]
SCR = os.environ.get("R6_SCRATCH", os.path.join(os.path.dirname(__file__), "..", "..", "results", "summaries"))
os.makedirs(SCR, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ---------------------------------------------------------------- 1. network + Gaussian-chain coefficients (MC, cached)
jv = np.load("data/dumps/jac_val.npz"); W = jv["W"].astype(np.float64)          # W[m] = 1-based W_{m+1}; forward z = h @ W
coef_f = os.path.join(SCR, "r6_coefs.npz")
if os.path.exists(coef_f):
    cf = np.load(coef_f); dd, cc, mu, sg, G = cf["dd"], cf["cc"], cf["mu"], cf["sg"], cf["G"]
else:
    N, eps = 200000, 0.05; h = np.random.default_rng(1).standard_normal((N, n)).astype(np.float32)
    dd, cc, mu, sg, G = [], [], [], [], []
    for l in range(L):
        z = h @ W[l].astype(np.float32)
        dd.append((z > 0).mean(axis=0).astype(np.float64)); cc.append(((np.abs(z) < eps).mean(axis=0) / (2 * eps)).astype(np.float64))
        m_ = z.mean(axis=0).astype(np.float64); s_ = z.std(axis=0).astype(np.float64); mu.append(m_); sg.append(s_)
        G.append(np.exp(-0.5 * (m_ / s_) ** 2) / np.sqrt(2 * np.pi) / (2 * s_)); h = np.maximum(z, 0)
    dd, cc, mu, sg, G = map(np.array, (dd, cc, mu, sg, G)); np.savez(coef_f, dd=dd, cc=cc, mu=mu, sg=sg, G=G)
log(f"coefficients ready: d16 mean {dd[15].mean():.3f}, c mean {cc[10].mean():.3f}, G16 mean {G[15].mean():.3f}")

def K_chain(p, lp, half=False):
    """K_{p->lp} = W_lp^T D_{lp-1} ... D_{p+1} W_{p+1}^T (1-based layers; D = diag(dd) or 1/2 if half)."""
    M = W[lp - 1].T.copy()
    for m in range(lp - 2, p - 1, -1):
        Dm = (0.5 * np.ones(n)) if half else dd[m]
        M = M @ (Dm[:, None] * W[m].T)
    return M
_K = {}
def K(p, lp, half=False):
    key = (p, lp, half)
    if key not in _K:
        _K[key] = K_chain(p, lp, half)
    return _K[key]

def Q(Delta, p):
    """final-mean response to dD21 = Delta at post-activation layer p (1-based)."""
    c, d = cc[p - 1], dd[p - 1]
    dC = 0.5 * (c[:, None] * Delta * d[None, :]); dC = dC + dC.T
    return Qcov(dC, p)
def Qcov(dC, p):
    resp = np.zeros(n)
    for lp in range(p + 1, L + 1):
        Kp = K(p, lp); dvar = np.einsum("ij,ij->i", Kp @ dC, Kp)
        if lp == L: resp += G[L - 1] * dvar
        else: resp += dd[L - 1] * (K(lp, L) @ (G[lp - 1] * dvar))
    return resp
def Qcols(Lf, Rf, p):
    """column-wise responses of D21 = Lf Rf^T: column j of the result = Q(Lf[:, j] Rf[:, j]^T)."""
    c, d = cc[p - 1], dd[p - 1]; R = np.zeros((n, n))
    for lp in range(p + 1, L + 1):
        Kp = K(p, lp); dvar = (Kp @ (c[:, None] * Lf)) * (Kp @ (d[:, None] * Rf))
        if lp == L: R += G[L - 1][:, None] * dvar
        else: R += dd[L - 1][:, None] * (K(lp, L) @ (G[lp - 1][:, None] * dvar))
    return R
def Qd3(D3, p):
    """direct mean path of the diagonal kappa3(i,i,i): dm_p = -(1/6) D3 mu c / sig^2, transported to the final mean."""
    dm = -(D3 / 6.0) * mu[p - 1] * cc[p - 1] / sg[p - 1] ** 2
    return dd[L - 1] * (K(p, L) @ dm) if p < L else dm
def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
def mse(v): return float(np.mean(v * v))

# ---------------------------------------------------------------- dumps
def load_full(li):
    d = np.load(f"data/dumps/query_dump_full_L{li}.npz"); ka = int(d["ka"]); out = []
    for s in range(ka):
        A = d[f"A{s}"].astype(np.float64); P = d[f"P{s}"].astype(np.float64); w2 = d[f"w2_{s}"]; sv = d[f"s_{s}"]; ev = d[f"e_{s}"]
        Z = d[f"Z{s}"].astype(np.float64); Lm = d[f"L{s}"].astype(np.float64)
        AP = A * P; PP = P * P; MP = (Z @ Lm.T) * P + PP * sv[None, :] + 3.0 * AP * ev[None, :]
        LA = 2.0 * AP * w2[None, :] + PP * ev[None, :]
        LP = A * A * w2[None, :] + PP * (sv / 3.0)[None, :] + (2.0 / 3.0) * MP
        D3 = 3.0 * np.sum(A * A * P * w2[None, :], axis=1) + np.sum(MP * P, axis=1)
        thin = ((PP @ Lm) @ Z.T) / 3.0
        out.append({"A": A, "P": P, "w2": w2, "LA": LA, "LP": LP, "D3": D3, "thin": thin})
    return out, d["Qc"].astype(np.float64), int(d["kb"])
def load_young(li):
    d = np.load(f"data/dumps/young_dump_L{li}.npz"); ka, k = int(d["ka"]), int(d["k"]); out = []
    for s in range(k):                                   # slots 0..k-1 are sources; slot k is the newborn/covariance slot
        A = d[f"A{s}"].astype(np.float64); P = d[f"P{s}"].astype(np.float64); w2 = d[f"w2_{s}"]
        out.append({"A": A, "P": P, "w2": w2, "LA": 2.0 * A * P * w2[None, :], "LP": A * A * w2[None, :], "old": s < ka})
    return out, d["Qc"].astype(np.float64), ka
def d21_of(srcs, key_pairs=(("LA", "A"), ("LP", "P"))):
    D = np.zeros((n, n))
    for s in srcs:
        for lk, rk in key_pairs: D += s[lk] @ s[rk].T
        if "thin" in s: D += s["thin"]
    return D

results = {}
# ---------------------------------------------------------------- 2. validation of Q against the exact chain
if "val" in STAGES:
    for li in (10, 14):
        p = li + 1; dC = jv[f"dC{li}"].astype(np.float64); dm = jv[f"m{li}"] - jv["m0"]
        est = Qcov(dC, p); a = float(np.dot(est, dm) / np.dot(est, est))
        log(f"VAL random dC at li={li}: R^2(exact, model) {r2(dm, est):.4f}  corr {np.corrcoef(dm, est)[0,1]:.4f}  scale {a:.3f} (1 = exact constants)")
        results[f"val_dC_L{li}"] = {"r2": r2(dm, est), "scale": a}
    av = np.load("data/dumps/adjoint_val_preds.npz")
    for li in (10, 14):
        p = li + 1; srcs, Qc, kb = load_full(li)
        D21 = d21_of(srcs); D3 = sum(s["D3"] for s in srcs)
        dm = av["full"][-1] - av[f"drop{li}"][-1]
        q1 = Q(D21, p); q3 = Qd3(D3, p)
        X = np.stack([q1, q3], axis=1); cf_, *_ = np.linalg.lstsq(X, dm, rcond=None); est = X @ cf_
        a1 = float(np.dot(q1, dm) / np.dot(q1, q1))
        log(f"VAL old tier dropped at li={li} ({len(srcs)} sources): exact drift MSE {mse(dm):.3e}; model D21-only: R^2 {r2(dm, a1*q1):.4f} scale {a1:.3f}, "
            f"MSE(Q(D21)) {mse(q1):.3e}; D21+D3 fit: R^2 {r2(dm, est):.4f} coefs {cf_[0]:.3f}, {cf_[1]:.3f}")
        results[f"val_drop_L{li}"] = {"mse_exact": mse(dm), "mse_model": mse(q1), "r2_d21": r2(dm, a1 * q1), "scale_d21": a1, "r2_fit": r2(dm, est)}

# ---------------------------------------------------------------- 3. per-column query responses -> coherence, HT variance
def ht_probs(e, q, alpha=0.5):
    w = np.power(np.maximum(e, 0.0), alpha); p = w / w.sum() * (q * n); p = np.clip(p, 0, 1)
    for _ in range(50):
        free = p < 1.0; deficit = q * n - p.sum()
        if abs(deficit) < 1e-9 or not free.any(): break
        p[free] = np.clip(p[free] * (1 + deficit / max(p[free].sum(), 1e-12)), 0, 1)
    return p
if "cols" in STAGES:
    for li in LAYERS:
        p = li + 1
        if li in (10, 14):
            srcs, Qc, kb = load_full(li); tag = "old(full 4 families)"
        else:
            ys, Qc, ka = load_young(li); srcs = [s for s in ys if s["old"]]; tag = "old(H+T2 only)"
        tot = np.zeros(n); sum_sq = 0.0; per_src = []; Rcols = []
        for s in srcs:
            Rs = Qcols(s["LA"], s["A"], p) + Qcols(s["LP"], s["P"], p)
            Rcols.append(Rs); rs = Rs.sum(axis=1); tot += rs; ssq = float(np.sum(Rs * Rs)); sum_sq += ssq
            per_src.append((mse(rs) * n, ssq))
        coh = float(np.dot(tot, tot) / sum_sq)
        log(f"COLS li={li} {tag}: |sum_j R_j|^2 / sum_j |R_j|^2 = {coh:.3f}  (1 = fully incoherent random walk over the {len(srcs)*n} (source,column) terms; "
            f"per source: {' '.join(f'{a/b:.2f}' for a, b in per_src)})")
        # exact expected relative squared error of HT column sampling (fresh draw), p_j ~ e_j^alpha
        row = {}
        for q in (0.9, 0.8, 0.7, 0.5):
            for alpha in (0.0, 0.5, 1.0):
                var = 0.0
                for s, Rs in zip(srcs, Rcols):
                    e = s["w2"] ** 2 * np.sum(s["A"] ** 2, axis=0) * np.sum(s["P"] ** 2, axis=0)
                    pj = ht_probs(e, q, alpha); var += float(np.sum((1.0 / np.maximum(pj, 1e-12) - 1.0) * np.sum(Rs * Rs, axis=0)))
                row[f"q{q}_a{alpha}"] = var / np.dot(tot, tot)
            log(f"      HT keep {q:.1f}: E|error|^2 / |response|^2 = " + " ".join(f"alpha={a}: {row[f'q{q}_a{a}']:.3f}" for a in (0.0, 0.5, 1.0)) + f"   (cost x{q:.1f}; the drift-MSE at this layer would be {row[f'q{q}_a0.5']:.2f} x the drop-old MSE)")
        # deterministic top-q screening in the same metric (coherent, per-layer view)
        for q in (0.9, 0.7):
            err = np.zeros(n)
            for s, Rs in zip(srcs, Rcols):
                e = s["w2"] ** 2 * np.sum(s["A"] ** 2, axis=0) * np.sum(s["P"] ** 2, axis=0)
                drop = np.argsort(e)[: n - int(round(q * n))]; err += Rs[:, drop].sum(axis=1)
            log(f"      deterministic top-{q:.1f} screening: |error|^2/|response|^2 = {np.dot(err, err)/np.dot(tot, tot):.3f}")
        results[f"cols_L{li}"] = {"coherence": coh, "ht": row, "n_src": len(srcs)}

# ---------------------------------------------------------------- 4. query-subspace restriction vs rank truncation
def top_right_subspace(M, rho):
    _, _, Vt = np.linalg.svd(M, full_matrices=False); return Vt[:rho].T
def sketch_subspace(M, rho, seed=0):
    Om = np.random.default_rng(seed).standard_normal((n, rho)); Y = M.T @ Om; Qs, _ = np.linalg.qr(Y); return Qs
if "sub" in STAGES:
    for li in LAYERS:
        p = li + 1; ys, Qc, ka = load_young(li)
        old = [s for s in ys if s["old"]]; young = [s for s in ys if not s["old"]]
        sets = {"old": old, "young": young, "all": ys}
        if li in (10, 14):
            fs, _, _ = load_full(li); sets["old4"] = fs
        Kd = K(p, L) * dd[p - 1][None, :]                      # K_{p->16} diag(d_p): the direct-path right factor
        Kh = K(p, L, half=True) * 0.5                          # same with D = 1/2 everywhere (no MC knowledge)
        sv = np.linalg.svd(Kd, compute_uv=False); en = np.cumsum(sv ** 2) / np.sum(sv ** 2)
        log(f"SUB li={li}: K_(p->16) diag(d) energy ranks 90/99/99.9/99.99%: {[int(np.searchsorted(en, t)) + 1 for t in (0.9, 0.99, 0.999, 0.9999)]}")
        ref = {nm: (d21_of(ss), Q(d21_of(ss), p)) for nm, ss in sets.items()}
        for nm in sets:
            log(f"      set {nm:5s} ({len(sets[nm])} sources): |Q(D21)|^2 (drift MSE if dropped) {mse(ref[nm][1]):.3e}")
        res = {}
        for rho in (32, 64, 128, 192, 256, 384, 512):
            V = top_right_subspace(Kd, rho); Vs = sketch_subspace(Kd, rho); Vh = top_right_subspace(Kh, rho)
            line = []
            for nm in sets:
                D21, q0 = ref[nm]
                for vn, VV in (("svd", V), ("sketch", Vs), ("D=1/2", Vh)):
                    err = D21 - (D21 @ VV) @ VV.T; qe = Q(err, p)
                    res[f"{nm}_{vn}_{rho}"] = mse(qe) / mse(q0)
                    if vn == "svd" or nm == "all":
                        line.append(f"{nm}/{vn}: {mse(qe)/mse(q0):.4f}")
            log(f"      rho={rho:3d}: relative residual response |Q(D21 - D21 V V^T)|^2/|Q(D21)|^2 -> " + "  ".join(line))
        results[f"sub_L{li}"] = res
        # two-sided restriction (left subspace from K diag(c)) for completeness, svd only, set all
        Kc = K(p, L) * cc[p - 1][None, :]
        for rho in (128, 256):
            V = top_right_subspace(Kd, rho); U = top_right_subspace(Kc, rho); D21, q0 = ref["all"]
            err = D21 - U @ (U.T @ D21 @ V) @ V.T; log(f"      two-sided rho={rho}: all sources residual {mse(Q(err, p))/mse(q0):.4f}")

if "rank" in STAGES:
    for li in LAYERS:
        if li not in (10, 14): continue
        p = li + 1; srcs, Qc, kb = load_full(li); ka = len(srcs)
        D21, q0 = d21_of(srcs), Q(d21_of(srcs), p)
        # shared-basis truncation: orthonormalise Qc, express legs, keep the top-r' directions of the pooled leg energy
        Qo, Rq = np.linalg.qr(Qc)
        F = np.concatenate([Qo.T @ s[k] for s in srcs for k in ("A", "P")], axis=1)   # (r, 2 ka n)
        resid = float(np.linalg.norm(np.concatenate([s[k] for s in srcs for k in ("A", "P")], axis=1) - Qo @ F) / np.linalg.norm(F))
        Uf, sf, _ = np.linalg.svd(F, full_matrices=False)
        log(f"RANK li={li}: legs outside span(Qc): {resid:.2e}; pooled leg energy ranks 99/99.9%: {int(np.searchsorted(np.cumsum(sf**2)/np.sum(sf**2), 0.99))+1}/{int(np.searchsorted(np.cumsum(sf**2)/np.sum(sf**2), 0.999))+1}")
        for rp in (320, 256, 192):
            B = Qo @ Uf[:, :rp]                                     # (n, r') truncated basis
            Dt = np.zeros((n, n))
            for s in srcs:
                A = B @ (B.T @ s["A"]); P = B @ (B.T @ s["P"]); w2 = s["w2"]
                # rebuild the families with truncated legs (M leg's thin part kept exact: it is O(n^2 r_thin))
                # note: s, e column scalings and Z L^T need the dump; recompute LA/LP from the truncated legs
                Dt += (2.0 * A * P * w2[None, :]) @ A.T + (A * A * w2[None, :]) @ P.T
            De = sum((2.0 * s["A"] * s["P"] * s["w2"][None, :]) @ s["A"].T + (s["A"] * s["A"] * s["w2"][None, :]) @ s["P"].T for s in srcs)
            qe = Q(Dt - De, p); qref = Q(De, p)
            log(f"      shared basis 384->{rp} (H+T2 families, cost x{rp/384:.3f}): residual response {mse(qe)/mse(qref):.4f}  (Frobenius rel err {np.linalg.norm(Dt-De)/np.linalg.norm(De):.3f})")
            results[f"rank_L{li}_shared_{rp}"] = mse(qe) / mse(qref)
        # per-source rank truncation (own SVD of [A P]) at equal relative residual
        for eps in (0.05, 0.1):
            Dt = np.zeros((n, n)); cost = 0
            for s in srcs:
                U_, s_, _ = np.linalg.svd(np.concatenate([s["A"], s["P"]], axis=1), full_matrices=False)
                en = np.cumsum(s_ ** 2) / np.sum(s_ ** 2); rs = int(np.searchsorted(en, 1 - eps ** 2)) + 1; cost += rs
                B = U_[:, :rs]; A = B @ (B.T @ s["A"]); P = B @ (B.T @ s["P"]); w2 = s["w2"]
                Dt += (2.0 * A * P * w2[None, :]) @ A.T + (A * A * w2[None, :]) @ P.T
            qe = Q(Dt - De, p)
            log(f"      per-source rank at leg residual {eps:.2f}: mean rank {cost/ka:.0f} (cost x{cost/ka/384:.3f}): residual response {mse(qe)/mse(qref):.4f}")
            results[f"rank_L{li}_persrc_{eps}"] = {"mean_rank": cost / ka, "resid": mse(qe) / mse(qref)}

if "asym" in STAGES:
    # contraction-only truncation: legs (hence Hadamards and D3) exact at r = 384, right factors A_s^T, P_s^T projected on the
    # top-r_c pooled leg directions. Cost per old source-layer: 2 n^2 r (legs) + 2 n^2 r_c (contractions) vs 4 n^2 r.
    for li in LAYERS:
        if li not in (10, 14): continue
        p = li + 1; srcs, Qc, kb = load_full(li)
        De = sum(s["LA"] @ s["A"].T + s["LP"] @ s["P"].T for s in srcs); qref = Q(De, p)
        Qo, _ = np.linalg.qr(Qc); F = np.concatenate([Qo.T @ s[k] for s in srcs for k in ("A", "P")], axis=1)
        Uf, sf, _ = np.linalg.svd(F, full_matrices=False)
        for rc in (320, 256, 192, 128, 64):
            B = Qo @ Uf[:, :rc]
            Dt = sum(s["LA"] @ (B @ (B.T @ s["A"])).T + s["LP"] @ (B @ (B.T @ s["P"])).T for s in srcs)
            qe = Q(Dt - De, p); cost = (2 * 384 + 2 * rc) / (4 * 384)
            log(f"ASYM li={li}: legs exact, contraction at r_c={rc} (old-tier cost x{cost:.3f}): residual response {mse(qe)/mse(qref):.4f} -> per-layer drift {mse(qe):.2e}")
            results[f"asym_L{li}_{rc}"] = {"cost": cost, "resid": mse(qe) / mse(qref), "drift": mse(qe)}
        # and the same truncation applied to the LEFT (Hadamard) side only, for comparison of the two error channels
        for rc in (256,):
            B = Qo @ Uf[:, :rc]
            Dt = np.zeros((n, n))
            for s in srcs:
                A = B @ (B.T @ s["A"]); P = B @ (B.T @ s["P"]); w2 = s["w2"]
                Dt += (2.0 * A * P * w2[None, :]) @ s["A"].T + (A * A * w2[None, :]) @ s["P"].T
            De2 = sum((2.0 * s["A"] * s["P"] * s["w2"][None, :]) @ s["A"].T + (s["A"] * s["A"] * s["w2"][None, :]) @ s["P"].T for s in srcs)
            log(f"ASYM li={li}: Hadamard side only at {rc} (H+T2): residual response {mse(Q(Dt - De2, p))/mse(Q(De2, p)):.4f}")

if "asymy" in STAGES:
    # young tier: dense legs exact; contraction LA_s A_s^T with A_s^T replaced by its own top-r_c projection (per-source SVD
    # of [A_s P_s]); cost per young source-layer: transports 2 n^3 (unchanged) + contractions 4 n^2 r_c instead of 2 n^3.
    for li in LAYERS:
        p = li + 1; ys, Qc, ka = load_young(li); young = [s for s in ys if not s["old"]]
        De = d21_of(young); qref = Q(De, p)
        bases = [np.linalg.svd(np.concatenate([s["A"], s["P"]], axis=1), full_matrices=False)[0] for s in young]
        for rc in (768, 640, 512, 384, 256):
            Dt = np.zeros((n, n))
            for s, U_ in zip(young, bases):
                B = U_[:, :rc]; Dt += s["LA"] @ (B @ (B.T @ s["A"])).T + s["LP"] @ (B @ (B.T @ s["P"])).T
            qe = Q(Dt - De, p); cost = (2 * n + 4 * rc) / (4 * n)
            log(f"ASYMY li={li}: young legs exact, contraction at r_c={rc} (young-tier cost x{cost:.3f}, contraction x{2*rc/n:.2f}): residual response {mse(qe)/mse(qref):.4f} -> per-layer drift {mse(qe):.2e}  (|Q(D21_young)|^2 {mse(qref):.2e})")
            results[f"asymy_L{li}_{rc}"] = {"cost": cost, "resid": mse(qe) / mse(qref), "drift": mse(qe)}

json.dump(results, open(os.path.join(SCR, "r6_query_results" + ("_" + "_".join(STAGES) if STAGES != ["val", "cols", "sub", "rank"] else "") + ".json"), "w"), indent=1)
log("done")
