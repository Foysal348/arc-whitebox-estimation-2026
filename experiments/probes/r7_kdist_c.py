"""Round 7 (Fable, 23 Sep), part 2: the TRANSPORT-CURVATURE term  K_ijc [C_ij w2_i w2_j w1_c + C_ic w2_i w1_j w2_c + C_jc w1_i w2_j w2_c]
(the O(C) correction to the linear transport w1 w1 w1 K_ijc of the all-distinct kappa3 through the ReLU, dropped by the
chain for every source at every layer) and the Gaussian triangle w2 w2 w2 C_ij C_jc C_ic, evaluated on the chain's own
sources at layer l and contracted to the next layer's (2,1) slice on a subset of columns a (both are n^4 exactly).

The chain's all-distinct tensor at the pre-activation of layer l is, per source s (legs indexed [neuron, hub r]):
    B1 = Sym(X1 x P x Y1), X1 = 3A + Xt, Y1 = A d(w2_b) + Yt, Xt = F1 R1^T, Yt = F2 R2^T   (F1|F2 = Zf, transported)
    B3 = Sym(M x P x P),   M = P d(s_b) + 3 A d(e_b) + Z L^T
    Bf = Sym(X3 x P x Y3), X3 = A d(c1) + P d(c2), Y3 = y 1^T with y = Z[:, rres+1] (transported)
with Sym = (1/6) sum over the 6 permutations.  Check: the linear transport of these blocks (+ the newborn hub of layer l)
must reproduce S_dist_chain(l+1) = D21_chain(l+1) - S_diag(K3) - S_21(K21) of part 1.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r7_kdist_c.py [n_cols=24]
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
NA = int(sys.argv[1]) if len(sys.argv) > 1 else 24
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]; n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
IW = {1: 4, 2: 6, 3: 8, 4: 12}
f64 = lambda x: np.asarray(x, np.float64); f32 = lambda x: np.asarray(x, np.float32)
rng = np.random.default_rng(7); COLS = np.sort(rng.choice(n, NA, replace=False))

def S_diag(W, k3): return ((W * W) * k3[:, None]).T @ W
def S_21(W, Y):
    YW = Y @ W; return (W * W).T @ YW + 2.0 * (W * YW).T @ W
def hub(W, g, L, M):
    LW = L @ W; MW = M @ W; Wg = W * g[:, None]
    return (Wg * LW).T @ MW + (Wg * MW).T @ LW + (LW * MW).T @ Wg
def stats(x, y):
    s = float(np.dot(x, y) / np.dot(x, x)); ss = float(np.sum(y * y))
    return dict(slope=s, r2_fit=float(1 - np.sum((y - s * x) ** 2) / ss), r2_1to1=float(1 - np.sum((y - x) ** 2) / ss), corr=float(np.corrcoef(x, y)[0, 1]))
def multi(X, y, names):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
    return dict(r2=r2, coef={nm: float(c) for nm, c in zip(names, coef)})

PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

def blocks_of_source(c, s, rres, rfb):
    """the three hub blocks of source s as leg triples (X, Y, Z) with Sym = (1/6) sum of perms; legs float32 (n, n)"""
    A = f32(c["A_st"][s]); P = f32(c["P_st"][s]); Z = f32(c["Z_st"][s]); L = f32(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f32(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f32(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f32(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f32(c["R2T_st"][s]).T
    else:
        Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = f32(Z[:, rres + 1])
    return [("B1", (X1, P, Y1)), ("B3", (M, P, P)), ("Bf", (X3, P, ("rank1", y)))]

def leg_mat(X):
    return X if not isinstance(X, tuple) else X[1][:, None] * np.ones((1, n), np.float32)

out = []
for l in (3, 7, 11, 14, 15):
    t0 = time.time()
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P_ = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz")); c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); cn = np.load(os.path.join(CAP, f"r7_cap_L{l+1}.npz"))
    W = f64(Wall[l]); Wf = f32(W); rres = int(c["rres"]); rfb = int(c["rfb"]); k = c["A_st"].shape[0]
    # ---- part-1 quantities restricted to the column subset ----
    k3_post_true = np.diag(f64(P_["Y21"])).copy(); Y21_post_true = f64(P_["Y21"]).copy(); np.fill_diagonal(Y21_post_true, 0.0)
    Dtrue = f64(Tn["Y21"]).copy(); np.fill_diagonal(Dtrue, 0.0); Dchain = f64(cn["D21"]).copy(); np.fill_diagonal(Dchain, 0.0)
    K3c = f64(c["K3"]); K21c = f64(c["K21"]).copy(); np.fill_diagonal(K21c, 0.0)
    Sx_t = Dtrue - S_diag(W, k3_post_true) - S_21(W, Y21_post_true); Sx_c = Dchain - S_diag(W, K3c) - S_21(W, K21c)
    e_x = (Sx_c - Sx_t)[COLS]; e_tot = (Dchain - Dtrue)[COLS]
    mask = np.ones((NA, n), bool); mask[np.arange(NA), COLS] = False
    # wicks at layer l: chain's own (the correction the chain would compute) and true
    Wa = f64(c["W_all"]); wc = {j: Wa[:, IW[j]] for j in (1, 2, 3)}
    mu_t, C_t = f64(T["mean"]), f64(T["C"]); a_ = mu_t / np.sqrt(np.diag(C_t)); s_ = np.sqrt(np.diag(C_t))
    wt = {1: Phi(a_), 2: pdf(a_) / s_, 3: -a_ * pdf(a_) / s_ ** 2}
    Co_c = f64(c["C_off"]); K_c = f64(c["D21"]).copy(); np.fill_diagonal(K_c, 0.0)
    Co_t = C_t.copy(); np.fill_diagonal(Co_t, 0.0); K_t = f64(T["Y21"]).copy(); np.fill_diagonal(K_t, 0.0)
    def fam(Co, K, w):
        Cw1 = Co * w[1][None, :]; Kw1 = K * w[1][None, :]; KTw2 = K.T * w[2][None, :]; C2w2 = Co * Co * w[2][None, :]
        return {"hub": hub(W, w[2], Cw1, Cw1), "famA": hub(W, w[3], Cw1, Kw1), "famB": hub(W, w[2], KTw2, Cw1), "dedge": hub(W, w[3], C2w2, Cw1)}
    Ft = {kk: v[COLS] for kk, v in fam(Co_t, K_t, wt).items()}; Fc = {kk: v[COLS] for kk, v in fam(Co_c, K_c, wc).items()}
    # ---- the chain's sources at layer l ----
    w1 = f32(wc[1]); w2 = f32(wc[2])
    Wb1 = Wf * w1[:, None]; Wb2 = Wf * w2[:, None]          # (n_c, n_b): W_cb w1_c, W_cb w2_c
    Cuse = f32(Co_c)          # the Mehler expansion is in the OFF-diagonal covariance: chain's C_off at layer l
    lin = np.zeros((NA, n)); kdc = np.zeros((NA, n))
    per_src = []
    for s in range(k):
        blocks = blocks_of_source(c, s, rres, rfb)
        # transported legs (linear): X' = W^T diag(w1) X  -> (n_a, n_r); precomputed for every leg; and X^T Wb1 == (X')^T
        legs = {}
        for name, (X, Y, Z) in blocks:
            for tag, Lg in zip("XYZ", (X, Y, Z)):
                key = (name, tag)
                if isinstance(Lg, tuple): legs[key] = ("rank1", Wb1.T @ Lg[1])         # (n_a,) : X'_{ar} = (W^T w1 y)_a for all r
                else: legs[key] = Wb1.T @ Lg                                            # (n_a, n_r)
        lin_s = np.zeros((NA, n)); kdc_s = np.zeros((NA, n))
        # linear transport check and K x C, column by column
        for ia, a in enumerate(COLS):
            u = Wf[:, a] * w2; v = Wf[:, a] * w1
            for name, (X, Y, Z) in blocks:
                legsXYZ = [X, Y, Z]
                # transported legs over all output neurons, (n_out, n_r) = Wb1^T leg (rank-1 legs kept as vectors)
                TL = [legs[(name, t)] for t in "XYZ"]
                def row(i):  # (n_r,) transported leg i at neuron a
                    Lg = TL[i]; return np.full(n, Lg[1][a], np.float32) if isinstance(Lg, tuple) else Lg[a]
                def allb(i):  # (n_r, n_b) transported leg i for all b (== leg^T Wb1)
                    Lg = TL[i]; return ("rank1", Lg[1]) if isinstance(Lg, tuple) else Lg.T
                # ---- linear: sum_perms (1/6) sum_r X'_ar Y'_ar Z'_br ----
                for (p, q, r_) in PERMS:
                    g = row(p) * row(q); Zb = allb(r_)
                    lin_s[ia] += (np.sum(g) * Zb[1]) if isinstance(Zb, tuple) else g @ Zb
                # ---- K x C: placement C_ij (i,j with a): (1/6) sum_perms G_r (Z')_rb, G_r = sum_i u_i X_ir (C diag(u) Y)_ir ----
                CuL = []
                for Lg in legsXYZ:
                    CuL.append(("rank1", Cuse @ (u * Lg[1])) if isinstance(Lg, tuple) else Cuse @ (u[:, None] * Lg))
                def Gpair(p, q):   # G_r = sum_i u_i X_p[i,r] (C u Y_q)[i,r]
                    Xp = legsXYZ[p]; Cq = CuL[q]
                    if isinstance(Xp, tuple) and isinstance(Cq, tuple): return np.full(n, np.dot(u * Xp[1], Cq[1]), np.float64)
                    if isinstance(Xp, tuple): return (u * Xp[1]) @ Cq
                    if isinstance(Cq, tuple): return (u * Cq[1]) @ Xp
                    return np.einsum("ir,ir->r", u[:, None] * Xp, Cq)
                for (p, q, r_) in PERMS:
                    G = Gpair(p, q); Zb = allb(r_)
                    kdc_s[ia] += (np.sum(G) * Zb[1]) if isinstance(Zb, tuple) else G @ Zb
                # ---- K x C: placements C_ic and C_jc (equal by symmetry): 2 x (1/6) sum_perms q^T Wb2, q = ((C u X) o Z) (Y^T v) ----
                for (p, q, r_) in PERMS:
                    Yq = legsXYZ[q]; Yv = (np.dot(v, Yq[1]) * np.ones(n, np.float32)) if isinstance(Yq, tuple) else (v @ Yq)   # (n_r,)
                    CuX = CuL[p]; Zr = legsXYZ[r_]
                    if isinstance(CuX, tuple) and isinstance(Zr, tuple): qv = CuX[1] * Zr[1] * np.sum(Yv)
                    elif isinstance(CuX, tuple): qv = CuX[1] * (Zr @ Yv)
                    elif isinstance(Zr, tuple): qv = Zr[1] * (CuX @ Yv)
                    else: qv = np.einsum("cr,cr,r->c", CuX, Zr, Yv)
                    kdc_s[ia] += 2.0 * (qv @ Wb2)
        lin += lin_s / 6.0; kdc += kdc_s / 6.0
        per_src.append({"src": s, "lin_rms": float(np.sqrt(np.mean(lin_s[mask] ** 2))) / 6.0, "kdc_rms": float(np.sqrt(np.mean(kdc_s[mask] ** 2))) / 6.0})
    # ---- newborn hub of layer l (born after this ReLU; part of the chain's D21(l+1)): X1 = 3 a_b + Xt, Y1 = a_b d(w2) + Yt, P = I, + feed ----
    a_b = w1[:, None] * f32(Co_c); D21c = f32(K_c)
    Xt = 1.5 * w2[:, None] * D21c; Yt = 0.5 * w1[:, None] * D21c.T * f32(wc[3])[None, :]
    X1n = 3.0 * a_b + Xt; Y1n = a_b * w2[None, :] + Yt
    X1W = Wb1.T @ X1n; PW = Wb1.T; Y1W = Wb1.T @ Y1n       # transported newborn legs (n_a, n_r), P = I -> P' = Wb1^T
    nb = np.zeros((NA, n))
    for ia, a in enumerate(COLS):
        TL = [X1W, PW, Y1W]
        for (p, q, r_) in PERMS: nb[ia] += (TL[p][a] * TL[q][a]) @ TL[r_].T
    nb /= 6.0
    # feed of the newborn: X3 = a_b d(c1) + I d(c2) with c1 = w1 lam, c2 = w1^2 dG ; Y3 = y 1^T, y = (m/4) w2 -- use the captured next-layer c1/c2/y of the NEXT capture if present
    recon = lin + nb
    # ---- triangle on the subset: t_c = diag(C u C u C), T1(a,b) = sum_c t_c W_cb w2_c ----
    tri = np.zeros((NA, n)); Ct = f32(Co_t)
    for ia, a in enumerate(COLS):
        u = Wf[:, a] * f32(wt[2]); CuC = (Ct * u[None, :]) @ Ct; t = np.einsum("ci,i,ic->c", CuC, u, Ct); tri[ia] = t @ Wb2
    # ---- regressions on the subset ----
    y = e_x[mask]; yt = e_tot[mask]
    rec = {"post_layer": l, "n_cols": NA, "n_sources": k,
           "recon_check": stats(recon[mask], Sx_c[COLS][mask]), "recon_rms": float(np.sqrt(np.mean(recon[mask] ** 2))), "Sx_chain_rms": float(np.sqrt(np.mean(Sx_c[COLS][mask] ** 2))),
           "nb_share_of_recon": float(np.sqrt(np.mean(nb[mask] ** 2)) / np.sqrt(np.mean(recon[mask] ** 2))),
           "rms": {"e_dist": float(np.sqrt(np.mean(y ** 2))), "kdc": float(np.sqrt(np.mean(kdc[mask] ** 2))), "tri": float(np.sqrt(np.mean(tri[mask] ** 2))), "famA": float(np.sqrt(np.mean(Ft["famA"][mask] ** 2)))},
           "single_on_e_dist": {"kdc": stats(kdc[mask], y), "tri": stats(tri[mask], y), "famA": stats(Ft["famA"][mask], y), "famB": stats(Ft["famB"][mask], y), "dedge": stats(Ft["dedge"][mask], y)},
           "per_source": per_src}
    names = ["kdc", "tri", "famA", "famB", "dedge"]
    X = np.stack([kdc[mask], tri[mask], Ft["famA"][mask], Ft["famB"][mask], Ft["dedge"][mask]], 1)
    rec["joint_on_e_dist"] = multi(X, y, names); rec["joint_on_e_tot"] = multi(X, yt, names)
    rec["joint_kdc_famAB_on_e_dist"] = multi(X[:, [0, 2, 3]], y, ["kdc", "famA", "famB"])
    # fixed-coefficient (theory) test: err = -kdc - (famA + famB)/2 - dedge
    pred = -kdc[mask] - 0.5 * (Ft["famA"][mask] + Ft["famB"][mask]) - Ft["dedge"][mask]
    rec["theory_1to1_r2_on_e_dist"] = float(1 - np.sum((y - pred) ** 2) / np.sum(y ** 2)); rec["theory_1to1_r2_on_e_tot"] = float(1 - np.sum((yt - pred) ** 2) / np.sum(yt ** 2))
    rec["theory_resid_rms"] = float(np.sqrt(np.mean((y - pred) ** 2)))
    # chain-state versions of the families (what the chain would compute)
    predc = -kdc[mask] - 0.5 * (Fc["famA"][mask] + Fc["famB"][mask]) - Fc["dedge"][mask]
    rec["theory_chainstate_1to1_r2_on_e_dist"] = float(1 - np.sum((y - predc) ** 2) / np.sum(y ** 2))
    out.append(rec)
    rc = rec["recon_check"]; so = rec["single_on_e_dist"]; j = rec["joint_on_e_dist"]
    print(f"post L{l} -> pre L{l+1} ({time.time()-t0:.0f}s, {k} sources, {NA} cols): recon of S_dist_chain: slope {rc['slope']:.3f} R2(1:1) {rc['r2_1to1']:.4f} (recon rms {rec['recon_rms']:.2e} vs {rec['Sx_chain_rms']:.2e}; newborn share {rec['nb_share_of_recon']:.2f})", flush=True)
    print(f"   rms: e_dist {rec['rms']['e_dist']:.2e} | kdc {rec['rms']['kdc']:.2e} tri {rec['rms']['tri']:.2e} famA {rec['rms']['famA']:.2e}", flush=True)
    print("   e_dist on each: " + ", ".join(f"{k_}: slope {v_['slope']:+.2f} R2 {v_['r2_fit']:.3f}" for k_, v_ in so.items()), flush=True)
    print(f"   joint: R2 {j['r2']:.3f} " + ", ".join(f"{k_} {v_:+.2f}" for k_, v_ in j["coef"].items()) + f" | kdc+famA+famB: R2 {rec['joint_kdc_famAB_on_e_dist']['r2']:.3f}"
          f" | THEORY (fixed coefs -1, -1/2, -1/2, -1): R2 e_dist {rec['theory_1to1_r2_on_e_dist']:.3f}, e_tot {rec['theory_1to1_r2_on_e_tot']:.3f}, resid rms {rec['theory_resid_rms']:.2e}; chain-state families: {rec['theory_chainstate_1to1_r2_on_e_dist']:.3f}", flush=True)
    print("   per source (lin rms / kdc rms): " + " ".join(f"s{p['src']}:{p['lin_rms']:.1e}/{p['kdc_rms']:.1e}" for p in per_src), flush=True)
json.dump(out, open("../results/summaries/r7_kdist_c.json", "w"), indent=1)
