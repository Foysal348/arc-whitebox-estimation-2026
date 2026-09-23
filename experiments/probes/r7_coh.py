"""Round 7 (Fable, 23 Sep), part 6: two attribution details on stored arrays.

(A) Coherent part of the transport curvature: per source s alive at layer l, the K_dist x C correction kdc_s (subset of
    columns, n^4 exact) regressed on the source's own linear-transport contribution alld_s (all-distinct).  slope = the
    systematic gain change of the true transport relative to the chain's W^T diag(w1) (per layer), corr = how coherent.
(B) Per-birth-layer regression: e_dist(l+1) on the transported famA of every birth layer l' <= l jointly, to read off
    how fast the born famA deficit actually survives in the chain (coefficient -1/2 = persists as the linear transport
    predicts, ~0 = washed out).

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r7_coh.py [n_cols=8]
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
NA = int(sys.argv[1]) if len(sys.argv) > 1 else 8
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]; n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64); f32 = lambda x: np.asarray(x, np.float32)
off = ~np.eye(n, dtype=bool); rng = np.random.default_rng(11); COLS = np.sort(rng.choice(n, NA, replace=False))
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

def S_diag(W, k3): return ((W * W) * k3[:, None]).T @ W
def S_21(W, Y):
    YW = Y @ W; return (W * W).T @ YW + 2.0 * (W * YW).T @ W
def stats(x, y):
    s = float(np.dot(x, y) / np.dot(x, x)); ss = float(np.sum(y * y))
    return dict(slope=s, r2_fit=float(1 - np.sum((y - s * x) ** 2) / ss), corr=float(np.corrcoef(x, y)[0, 1]))
def multi(X, y, names):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
    return dict(r2=r2, coef={nm: float(c) for nm, c in zip(names, coef)})
def blk_slices(X, P, Y):
    D21 = ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0; D3 = np.sum(X * P * Y, axis=1); return f64(D21), f64(D3)
def blocks_of_source(c, s, rres, rfb):
    A = f32(c["A_st"][s]); P = f32(c["P_st"][s]); Z = f32(c["Z_st"][s]); L = f32(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f32(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f32(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f32(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f32(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = f32(Z[:, rres + 1]); Y3 = np.repeat(y[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]

TRUE = {}
def true_state(l):
    if l not in TRUE:
        T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); C = f64(T["C"]); mu = f64(T["mean"]); s = np.sqrt(np.diag(C)); a = mu / s; ph = pdf(a)
        Co = C.copy(); np.fill_diagonal(Co, 0.0); K = f64(T["Y21"]).copy(); np.fill_diagonal(K, 0.0)
        TRUE[l] = dict(Co=f32(Co), K=f32(K), w1=f32(Phi(a)), w2=f32(ph / s), w3=f32(-a * ph / s ** 2))
    return TRUE[l]
def famA_born(l):
    t = true_state(l); Cw1 = t["Co"] * t["w1"][None, :]; Kw1 = t["K"] * t["w1"][None, :]
    X = 3.0 * (Cw1 * t["w3"][:, None]).T; Y = Kw1.T
    D21b, D3b = blk_slices(X, np.eye(n, dtype=np.float32), Y)     # the born tensor's own slice entries (removed: not part of the error)
    return X, Y, D21b, D3b
def transport(X, P, Y, l_from, l_to):
    W = f32(Wall[l_from]); X = W.T @ X; Y = W.T @ Y; P = W.T.copy() if P is None else W.T @ P
    for ll in range(l_from + 1, l_to + 1):
        w1 = true_state(ll)["w1"]; W = f32(Wall[ll]); X = W.T @ (w1[:, None] * X); Y = W.T @ (w1[:, None] * Y); P = W.T @ (w1[:, None] * P)
    return X, P, Y

out = []
for l in (7, 11, 15):
    t0 = time.time()
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); P_ = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz")); c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); cn = np.load(os.path.join(CAP, f"r7_cap_L{l+1}.npz"))
    W = f64(Wall[l]); Wf = f32(W); rres = int(c["rres"]); rfb = int(c["rfb"]); k = c["A_st"].shape[0]
    Wa = f64(c["W_all"]); w1 = f32(Wa[:, 4]); w2 = f32(Wa[:, 6])
    k3_post_true = np.diag(f64(P_["Y21"])).copy(); Y21_post_true = f64(P_["Y21"]).copy(); np.fill_diagonal(Y21_post_true, 0.0)
    Dtrue = f64(Tn["Y21"]).copy(); np.fill_diagonal(Dtrue, 0.0); Dchain = f64(cn["D21"]).copy(); np.fill_diagonal(Dchain, 0.0)
    K3c = f64(c["K3"]); K21c = f64(c["K21"]).copy(); np.fill_diagonal(K21c, 0.0)
    e_x = (Dchain - S_diag(W, K3c) - S_21(W, K21c)) - (Dtrue - S_diag(W, k3_post_true) - S_21(W, Y21_post_true))
    rec = {"post_layer": l, "n_sources": k}
    # ---- (A) per source: all-distinct linear contribution (full) and kdc (subset) ----
    WT1 = (Wf * w1[:, None]).T; Wb1 = Wf * w1[:, None]; Wb2 = Wf * w2[:, None]; Cuse = f32(f64(c["C_off"]))
    mask = np.ones((NA, n), bool); mask[np.arange(NA), COLS] = False
    per = []
    for s in range(k):
        blocks = blocks_of_source(c, s, rres, rfb)
        tot = np.zeros((n, n)); sl21 = np.zeros((n, n)); sl3 = np.zeros(n)
        for (X, P, Y) in blocks:
            D21b, D3b = blk_slices(X, P, Y); sl21 += D21b; sl3 += D3b
            Dn, _ = blk_slices(WT1 @ X, WT1 @ P, WT1 @ Y); tot += Dn
        np.fill_diagonal(sl21, 0.0); alld = tot - S_21(W, sl21 * (f64(w1) ** 2)[:, None] * f64(w1)[None, :]) - S_diag(W, sl3 * f64(w1) ** 3); np.fill_diagonal(alld, 0.0)
        kdc = np.zeros((NA, n))
        for ia, a in enumerate(COLS):
            u = Wf[:, a] * w2; v = Wf[:, a] * w1
            for (X, Y, Z) in blocks:
                legs = [X, Y, Z]; TL = [Wb1.T @ Lg for Lg in legs]; CuL = [Cuse @ (u[:, None] * Lg) for Lg in legs]
                for (p, q, r_) in PERMS:
                    G = np.einsum("ir,ir->r", u[:, None] * legs[p], CuL[q]); kdc[ia] += G @ TL[r_].T
                    Yv = v @ legs[q]; qv = np.einsum("cr,cr,r->c", CuL[p], legs[r_], Yv); kdc[ia] += 2.0 * (qv @ Wb2)
        kdc /= 6.0
        x = alld[COLS][mask]; yk = kdc[mask]
        st = stats(x, yk); per.append({"src": s, "age": k - s, "alld_rms": float(np.sqrt(np.mean(x ** 2))), "kdc_rms": float(np.sqrt(np.mean(yk ** 2))), "slope_kdc_on_alld": st["slope"], "corr": st["corr"],
                                       "kdc_on_e_dist": stats(yk, e_x[COLS][mask])})
    rec["per_source"] = per
    print(f"post L{l} ({time.time()-t0:.0f}s, {k} sources, {NA} cols) (A) kdc_s on alld_s: " + " ".join(f"age{p['age']}:{p['slope_kdc_on_alld']:+.4f}(c{p['corr']:+.2f})" for p in per), flush=True)
    print("   kdc_s alone on e_dist: slope/R2 " + " ".join(f"{p['kdc_on_e_dist']['slope']:+.2f}/{p['kdc_on_e_dist']['r2_fit']:.3f}" for p in per) + " | kdc/alld rms " + " ".join(f"{p['kdc_rms']/p['alld_rms']*100:.1f}%" for p in per), flush=True)
    # ---- (B) per-birth famA (all-distinct part only) jointly ----
    births = list(range(2, l + 1)); Ds = []
    for lp in births:
        X0, Y0, D21b, D3b = famA_born(lp)
        X, P, Y = transport(X0, None, Y0, lp, l); D, _ = blk_slices(X, P, Y)
        # remove the transported slice entries of the born tensor: M = 3 S^T with S = D21b (zero diag) and diag D3b, block Sym(M x I x I)
        np.fill_diagonal(D21b, 0.0); Mb = f32(3.0 * D21b.T + np.diag(D3b))
        Xm, Pm, Ym = transport(Mb, None, np.eye(n, dtype=np.float32), lp, l); Dm, _ = blk_slices(Xm, Pm, Ym)
        D = D - Dm; np.fill_diagonal(D, 0.0); Ds.append(D)
    y = e_x[off]; names = [f"born{lp}" for lp in births]
    rec["famA_per_birth_joint"] = multi(np.stack([D[off] for D in Ds], 1), y, names)
    rec["famA_per_birth_rms"] = {nm: float(np.sqrt(np.mean(D[off] ** 2))) for nm, D in zip(names, Ds)}
    acc = sum(Ds); rec["famA_acc_alldist_on_e_dist"] = stats(acc[off], y); rec["famA_fresh_alldist_on_e_dist"] = stats(Ds[-1][off], y)
    j = rec["famA_per_birth_joint"]
    print(f"   (B) famA per birth (joint R2 {j['r2']:.3f}): " + " ".join(f"{nm}:{v:+.2f}" for nm, v in j["coef"].items()), flush=True)
    print(f"       rms per birth: " + " ".join(f"{v:.1e}" for v in rec["famA_per_birth_rms"].values()) + f" | acc: slope {rec['famA_acc_alldist_on_e_dist']['slope']:+.3f} R2 {rec['famA_acc_alldist_on_e_dist']['r2_fit']:.3f} | fresh: slope {rec['famA_fresh_alldist_on_e_dist']['slope']:+.3f} R2 {rec['famA_fresh_alldist_on_e_dist']['r2_fit']:.3f}", flush=True)
    out.append(rec)
json.dump(out, open("../results/summaries/r7_coh.json", "w"), indent=1)
