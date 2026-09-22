"""Validation of the adjoint / linear-response formulation of the old tier (own idea, 21 Sep).
Step 1 (needs the uv env; run from whest-starterkit):  python -m uv run python ../experiments/probes/probe_adjoint_val.py run
   -> final-mean predictions of the full chain and of 'dropold at layer l' for l = 10, 14 (adjoint_val_preds.npz)
Step 2 (system python):  python probe_adjoint_val.py fit
   -> the exact effect dm_true(l) = m_full - m_dropold(l) vs the adjoint estimate
      dm_est(l) = 2 g o rowsum((J~_l D21_old^(l)) o J^_l),  J~ = J diag(c), J^ = J diag(d), J = J_{l->16} of the Gaussian chain,
      D21_old = a * H_tot + b * T2_tot (+ mean-path terms), a, b fitted; reports R^2 and correlation over the 1024 outputs.
"""
import os, sys, time, types
import numpy as np
mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
if mode == "run":
    sys.path.insert(0, os.path.abspath("."))
    from local_engine import build_mlp
    import flopscope as flops
    import flopscope.numpy as fnp
    SRC = "../submissions/v29m8/estimator.py"
    src = open(SRC, encoding="utf-8").read()
    anchor = "                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, "                _saved = _SENS_PRE(li, ka, k, A_st, P_st, Z_st, Zf_st)\n" + anchor)
    anchor2 = "                                        sb2=(fap24[:2 * kb] if kb > 0 else None), s_sb=s_sb)\n"
    assert src.count(anchor2) == 1
    src = src.replace(anchor2, anchor2 + "                _SENS_POST(_saved, Z_st, Zf_st)\n")
    CFG = {"drop": set()}
    def _zero(x): fnp.multiply(x, 0.0, out=x)
    def _SENS_PRE(li, ka, k, A_st, P_st, Z_st, Zf_st):
        if li not in CFG["drop"] or ka == 0: return None
        idx = list(range(ka))
        zs = np.array(Z_st)[idx].copy() if Z_st is not None else None
        zfs = np.array(Zf_st)[idx].copy() if Zf_st is not None else None
        for s in idx:
            _zero(A_st[s]); _zero(P_st[s])
            if Z_st is not None: _zero(Z_st[s])
            if Zf_st is not None: _zero(Zf_st[s])
        return (idx, zs, zfs)
    def _SENS_POST(saved, Z_st, Zf_st):
        if saved is None: return
        idx, zs, zfs = saved
        for j, s in enumerate(idx):
            if Z_st is not None: fnp.copyto(Z_st[s], zs[j])
            if Zf_st is not None: fnp.copyto(Zf_st[s], zfs[j])
    NOFEED = os.environ.get("NOFEED", "0") == "1"
    NOK4 = os.environ.get("NOK4", "0") == "1"
    if NOK4:
        os.environ["V17_NO_FEED"] = "1"; os.environ["V17_NO_WK431"] = "1"; os.environ["V17_NO_REGEN"] = "1"
    if NOFEED:
        os.environ["V18_NO_FB"] = "1"
        src = src.replace("    R_RES = 16 ", "    R_RES = 0 ")
    mod = types.ModuleType("est_sens"); mod.__dict__.update({"_SENS_PRE": _SENS_PRE, "_SENS_POST": _SENS_POST})
    exec(compile(src, SRC, "exec"), mod.__dict__)
    class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
    mlp = build_mlp(width=1024, depth=16, seed=7)
    out = {"W": np.stack([np.asarray(w, dtype=np.float32) for w in mlp.weights])}
    for tag, drop in ((("full", ()), ("drop14", (14,))) if NOK4 else (("full", ()), ("drop10", (10,)), ("drop14", (14,)))):
        CFG["drop"] = set(drop); est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
            pred = est.predict(mlp, 2**41)
        out[tag] = np.asarray(pred, dtype=np.float64); print(f"{tag}: {time.time()-t0:.0f}s", flush=True)
    np.savez("../data/dumps/adjoint_val_preds" + ("_nofeed" if NOFEED else "") + ("_nok4" if NOK4 else "") + ".npz", **out); print("saved")
    sys.exit(0)

# ---------------- fit ----------------
d = np.load("data/dumps/adjoint_val_preds" + ("_nofeed" if os.environ.get("NOFEED", "0") == "1" else "") + ("_nok4" if os.environ.get("NOK4", "0") == "1" else "") + ".npz")
LAY = ((14, "data/dumps/query_dump_L14.npz"),) if os.environ.get("NOK4", "0") == "1" else ((10, "data/dumps/query_dump_L10.npz"), (14, "data/dumps/query_dump_L14.npz")); W = d["W"].astype(np.float64); n, L = W.shape[1], W.shape[0]
m_full = d["full"][-1]
# Monte Carlo coefficients of the Gaussian chain: d_l = P(z>0), c_l = density of z at 0, final g = d m / d sigma^2
N = 200000; rng = np.random.default_rng(1); h = rng.standard_normal((N, n)).astype(np.float32)
dd = []; cc = []; eps = 0.05
for l in range(L):
    z = h @ W[l].astype(np.float32)
    dd.append((z > 0).mean(axis=0).astype(np.float64)); cc.append(((np.abs(z) < eps).mean(axis=0) / (2 * eps)).astype(np.float64))
    if l == L - 1:
        mu = z.mean(axis=0).astype(np.float64); sig = z.std(axis=0).astype(np.float64)
    h = np.maximum(z, 0)
g = np.exp(-0.5 * (mu / sig) ** 2) / np.sqrt(2 * np.pi) / (2 * sig)     # d E[relu(z)] / d sigma^2 for z ~ N(mu, sigma^2)
def J_to_end(l):
    """J_{l->16}: sensitivity of the final PRE-activation covariance to the post-activation covariance at layer l (1-based),
    column convention z_{l+1} = W_{l+1}^T h_l:  J = W_16^T D_15 W_15^T ... D_{l+1} W_{l+1}^T"""
    J = W[L - 1].T.copy()
    for m in range(L - 2, l - 1, -1):            # m = 14 .. l  (0-based index of W_{m+1})
        J = J @ (np.diag(dd[m]) @ W[m].T)
    return J
def load(f):
    q = np.load(f); ka = int(q["ka"])
    return [(q[f"A{s}"].astype(np.float64), q[f"P{s}"].astype(np.float64), q[f"w2_{s}"].astype(np.float64)) for s in range(ka)]
# per-layer Gaussian-chain quantities: mu_l, sig_l of the pre-activation z_l (1-based l = index+1), g_l = dE[relu]/dsigma^2
N2 = 200000; h = np.random.default_rng(2).standard_normal((N2, n)).astype(np.float32); G = []
for l in range(L):
    z = h @ W[l].astype(np.float32); mu_l = z.mean(axis=0).astype(np.float64); sg = z.std(axis=0).astype(np.float64)
    G.append(np.exp(-0.5 * (mu_l / sg) ** 2) / np.sqrt(2 * np.pi) / (2 * sg)); h = np.maximum(z, 0)
def K(p, lp):
    """K_{p->l'}: post-activation covariance at layer p (1-based) -> PRE-activation covariance at layer l' : W_l'^T D_{l'-1} ... D_{p+1} W_{p+1}^T"""
    M = W[lp - 1].T.copy()
    for m in range(lp - 2, p - 1, -1):
        M = M @ (np.diag(dd[m]) @ W[m].T)
    return M
for li, f in LAY:
    dm_true = m_full - d[f"drop{li}"][-1]
    srcs = load(f); p = li + 1                      # covariance correction of post-activation layer p
    slices = {"H=(AoP)W2A^T": sum((A * P) @ (A * w2[None, :]).T for A, P, w2 in srcs),
              "T2=(AoA)W2P^T": sum((A * A) @ (P * w2[None, :]).T for A, P, w2 in srcs),
              "(AoP)W2P^T": sum((A * P) @ (P * w2[None, :]).T for A, P, w2 in srcs),
              "(PoP)W2A^T": sum((P * P) @ (A * w2[None, :]).T for A, P, w2 in srcs)}
    c_p, d_p = cc[li], dd[li]
    names = []; feats = []
    for nm, X in slices.items():
        Xs = X * c_p[:, None] * d_p[None, :]; Xs = Xs + Xs.T                 # delta C_p (symmetrised, up to a constant)
        for lp in range(p + 1, L + 1):                                       # variance perturbation at pre-activation layer l'
            Kp = K(p, lp); dvar = np.sum((Kp @ Xs) * Kp, axis=1)
            if lp == L:
                feats.append(G[L - 1] * dvar); names.append(f"{nm} final-var")
            else:
                dm_lp = G[lp - 1] * dvar                                     # mean shift of post-activation layer l'
                feats.append(dd[L - 1] * (K(lp, L) @ dm_lp)); names.append(f"{nm} var@{lp}->mean")
    for nm, X in slices.items():                                             # direct mean path: kappa3_iii of z_p -> E[relu] at layer p -> final mean
        dm_p = np.diag(X)
        feats.append(dd[L - 1] * (K(p, L) @ dm_p) if p < L else dm_p); names.append(f"{nm} D3-mean")
    F = np.array(feats).T
    def fit(cols, label):
        X = F[:, cols]; coef, *_ = np.linalg.lstsq(X, dm_true, rcond=None); est = X @ coef
        r2 = 1 - np.sum((dm_true - est) ** 2) / np.sum((dm_true - dm_true.mean()) ** 2)
        print(f"L{li}: {label:40s} R^2 {r2:.4f}  corr {np.corrcoef(dm_true, est)[0,1]:.4f}", flush=True)
    fin = [i for i, nm in enumerate(names) if "final-var" in nm]
    fit(fin, "final-variance path only (4 slice types)")
    fit(list(range(len(names))), f"all first-order paths ({len(names)} features)")
    fit([i for i, nm in enumerate(names) if "D3-mean" in nm], "D3 mean path only (4 slice types)")
    fit([i for i, nm in enumerate(names) if "D3-mean" not in nm], "covariance paths only")
    for nm_sel in slices:
        fit([i for i, nm in enumerate(names) if nm.startswith(nm_sel)], f"all paths, slice {nm_sel}")
    # which layers l' carry it? single-feature correlations for H
    print("   corr per path (H):", " ".join(f"{nm.split()[1]}:{np.corrcoef(dm_true, F[:, i])[0,1]:+.2f}" for i, nm in enumerate(names) if nm.startswith("H=")))
