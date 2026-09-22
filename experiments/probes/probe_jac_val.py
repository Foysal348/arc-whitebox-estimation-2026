"""Validate the Gaussian-chain covariance response model used by the adjoint idea, INSIDE V29:
add a small random symmetric dC to the post-activation covariance at estimator layer li (=> post-activation layer p = li+1),
measure the exact change of the final mean, and compare with the first-order model
   dm = g_16 o diag(K_{p->16} dC K^T) + sum_{l'<16} d_16 o K_{l'->16}[ g_l' o diag(K_{p->l'} dC K^T) ]
Step 1 (uv env, from whest-starterkit): python -m uv run python ../experiments/probes/probe_jac_val.py run     -> jac_val.npz
Step 2 (system python):                 python probe_jac_val.py fit
"""
import os, sys, time, types, numpy as np
mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
LAYERS = (10, 14); EPS = 2e-4
if mode == "run":
    sys.path.insert(0, os.path.abspath("."))
    from local_engine import build_mlp
    import flopscope as flops
    import flopscope.numpy as fnp
    SRC = "../submissions/v29m8/estimator.py"; src = open(SRC, encoding="utf-8").read()
    anchor = "            fnp.fill_diagonal(C, K2v)\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + "            _PERTURB(li, C)\n")
    CFG = {"li": None, "dC": None}
    def _PERTURB(li, C):
        if li == CFG["li"]:
            fnp.add(C, fnp.asarray(CFG["dC"].astype(np.float32)), out=C)
    mod = types.ModuleType("est_jac"); mod.__dict__["_PERTURB"] = _PERTURB
    exec(compile(src, SRC, "exec"), mod.__dict__)
    class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
    mlp = build_mlp(width=1024, depth=16, seed=7); n = 1024
    out = {"W": np.stack([np.asarray(w, dtype=np.float32) for w in mlp.weights])}
    def run(li, dC):
        CFG["li"] = li; CFG["dC"] = dC; est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
            p = est.predict(mlp, 2**41)
        print(f"li={li} eps={0.0 if dC is None else float(np.abs(dC).max()):.1e}: {time.time()-t0:.0f}s", flush=True)
        return np.asarray(p, dtype=np.float64)[-1]
    out["m0"] = run(None, None)
    rng = np.random.default_rng(5)
    for li in LAYERS:
        R = rng.standard_normal((n, n)); dC = EPS * (R + R.T) / np.sqrt(2); np.fill_diagonal(dC, 0.0)   # off-diagonal perturbation
        out[f"dC{li}"] = dC.astype(np.float32); out[f"m{li}"] = run(li, dC); out[f"m{li}x2"] = run(li, 2 * dC)
    np.savez("../data/dumps/jac_val.npz", **out); print("saved"); sys.exit(0)

d = np.load("data/dumps/jac_val.npz"); W = d["W"].astype(np.float64); n, L = W.shape[1], W.shape[0]
N = 200000; h = np.random.default_rng(1).standard_normal((N, n)).astype(np.float32); dd = []; G = []
for l in range(L):
    z = h @ W[l].astype(np.float32); dd.append((z > 0).mean(axis=0).astype(np.float64))
    mu_l = z.mean(axis=0).astype(np.float64); sg = z.std(axis=0).astype(np.float64)
    G.append(np.exp(-0.5 * (mu_l / sg) ** 2) / np.sqrt(2 * np.pi) / (2 * sg)); h = np.maximum(z, 0)
def K(p, lp):
    M = W[lp - 1].T.copy()
    for m in range(lp - 2, p - 1, -1): M = M @ (np.diag(dd[m]) @ W[m].T)
    return M
for li in LAYERS:
    p = li + 1; dC = d[f"dC{li}"].astype(np.float64)
    dm = d[f"m{li}"] - d["m0"]; dm2 = d[f"m{li}x2"] - d["m0"]
    print(f"li={li}: |dm| rms {np.sqrt(np.mean(dm**2)):.3e}; linearity check corr(dm, dm_2eps/2) = {np.corrcoef(dm, dm2/2)[0,1]:.4f}, ratio {np.sqrt(np.mean(dm2**2))/np.sqrt(np.mean(dm**2)):.3f}")
    feats = []; names = []
    for lp in range(p + 1, L + 1):
        Kp = K(p, lp); dvar = np.sum((Kp @ dC) * Kp, axis=1)
        if lp == L: feats.append(G[L - 1] * dvar); names.append("final-var")
        else: feats.append(dd[L - 1] * (K(lp, L) @ (G[lp - 1] * dvar))); names.append(f"var@{lp}->mean")
    F = np.array(feats).T
    for cols, label in (([len(names) - 1], "final-variance only"), (list(range(len(names))), "all variance->mean paths")):
        X = F[:, cols]; coef, *_ = np.linalg.lstsq(X, dm, rcond=None); est = X @ coef
        r2 = 1 - np.sum((dm - est) ** 2) / np.sum((dm - dm.mean()) ** 2)
        print(f"   {label:28s} R^2 {r2:.4f} corr {np.corrcoef(dm, est)[0,1]:.4f} coef {np.array2string(coef, precision=2)}   (coef=1 means the model scale is exact)")
    print("   single-path corr:", " ".join(f"{nm}:{np.corrcoef(dm, F[:, i])[0,1]:+.2f}" for i, nm in enumerate(names)))
