"""Moonshot diagnostic (23 Sep): which term of the covariance closure makes the chain's variance error?

probe_errbudget.py showed that the fresh error the chain adds at every layer comes ~55% from the pre-activation
VARIANCE, ~30% from kappa3_iii and ~15% from kappa4_iiii, and that the mean closure itself is exact to 9e-11. The
variance of layer l+1 is diag(W^T C_post W) of the chain's post-ReLU covariance at layer l, so its error is fixed by
the error of C_post; and C_post is a sum of closure TERMS (the (1,1) group of the Wick program for the off-diagonal,
the (2,) second-moment group for the diagonal).

Because W_{l+1} is independent of everything at layer l, the error map is linear and known:

    dC_pre(l+1) = W^T dC_post(l) W ,

so each candidate cause X (a closure term, or the inherited error) can be tested by regressing the MEASURED
dC_pre(l+1) = C_teacher(l+1) - C_chain(l+1) on W^T X W over ~5e5 off-diagonal entries (and 1024 diagonal ones).
A coefficient a_t != 0 says term t is mis-scaled by (1 + a_t); the unexplained remainder is what the closure lacks.

Usage (from whest-starterkit):
    V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_coverr.py
"""
import os, sys, types, json, time, glob
import numpy as np
from math import erf, sqrt as _sqrt, pi as _pi
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

DIR = os.environ.get("TEACHER", "../data/kaggle_out/kaggle_out_k4pub/k4teach")
SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a2 = "            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]\n"
a3 = "            C = flops.as_symmetric(C, symmetry=(0, 1))\n"
for a in (a1, a2, a3): assert src.count(a) == 1, a
src = src.replace(a1, "            _CAPW(li, mu, var, None if trim else C_off, Phi=None)\n" + a1)
src = src.replace(a2, a2 + "            _CAPT(li, None if trim else (ABstack, WL2, WR2, fpm['IND2'], prog['AB2']), (B1stack, WL1, fpm['IND1'], prog['B1']), pk1v, alpha, sigma)\n")
src = src.replace(a3, a3 + "            _CAPP(li, C)\n")

S = {}
f32 = lambda x: None if x is None else np.asarray(x, dtype=np.float32).copy()
def _CAPW(li, mu, var, C_off, Phi=None):
    S.setdefault(li, {}).update(mu=f32(mu), var=f32(var), C_off=f32(C_off))
def _CAPT(li, two, one, pk1v, alpha, sigma):
    d = S.setdefault(li, {})
    d["pk1"] = f32(pk1v); d["alpha"] = f32(alpha); d["sigma"] = f32(sigma)
    if two is not None:
        AB, WL, WR, IND, names = two
        AB = np.asarray(AB, np.float32); WL = np.asarray(WL, np.float32); WR = np.asarray(WR, np.float32); IND = np.asarray(IND)
        # group 0 = (1,1): the off-diagonal post-ReLU covariance terms; keep each term's matrix (symmetrised, zero diag)
        terms = []
        for t in range(AB.shape[0]):
            if IND[t, 0] == 0: continue
            X = AB[t] * np.outer(WL[t, 0] if WL.ndim == 3 else WL[t], WR[t, 0] if WR.ndim == 3 else WR[t])
            X = 0.5 * (X + X.T); np.fill_diagonal(X, 0.0)
            terms.append((f"{names[t][0]}*{names[t][1]}", X.astype(np.float32)))
        d["terms11"] = terms
    B1, WL1, IND1, names1 = one
    B1 = np.asarray(B1, np.float32); WL1 = np.asarray(WL1, np.float32); IND1 = np.asarray(IND1)
    d["terms2"] = [(f"{names1[t][0]}*{names1[t][1]}", (B1[t] * WL1[t]).astype(np.float32)) for t in range(B1.shape[0]) if IND1[t, 1] != 0]
def _CAPP(li, C):
    S.setdefault(li, {})["Cpost"] = f32(C)

mod = types.ModuleType("est_ce"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPT": _CAPT, "_CAPP": _CAPP})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(DIR, "truth.npz"))
Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])],
          seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    pred = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
print(f"chain run {time.time()-t0:.0f}s", flush=True)

Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / _sqrt(2.0))))
iu = np.triu_indices(Wall.shape[1], 1)
out = []
for li in range(0, 15):
    l_next = li + 2                                         # 1-based teacher layer of the NEXT pre-activation
    f = os.path.join(DIR, f"layer{l_next}_seed0.npz")
    if li + 1 not in S or S[li + 1].get("C_off") is None or not os.path.exists(f) or "terms11" not in S[li]: continue
    Wn = Wall[li + 1].astype(np.float32)                    # z_next = h @ Wn  ->  C_pre_next = Wn^T C_post Wn
    Cc = S[li + 1]["C_off"].astype(np.float64); np.fill_diagonal(Cc, S[li + 1]["var"])
    # consistency: the chain's own C_pre_next equals Wn^T C_post Wn
    Crec = (Wn.T @ S[li]["Cpost"] @ Wn).astype(np.float64)
    cons = float(np.sqrt(np.mean((Crec - Cc) ** 2)) / np.sqrt(np.mean(Cc ** 2)))
    Ct = np.asarray(np.load(f)["C"], np.float64)
    dC = Ct - Cc
    tv = np.diag(Ct); noise_off = float(np.sqrt(np.mean(np.outer(tv, tv)[iu]) / 6e7))
    y_off = dC[iu]; y_dg = np.diag(dC)
    feats, names = [], []
    for nm, X in S[li]["terms11"]:
        P = Wn.T @ X @ Wn; feats.append(P); names.append("11:" + nm)
    # the diagonal of C_post: second-moment terms (group (2,)) minus 2 pk1 x (mean terms) is the post variance
    for nm, v in S[li]["terms2"]:
        P = (Wn.T * v[None, :]) @ Wn; feats.append(P); names.append("2:" + nm)
    # inherited error: the chain's C_pre error at THIS layer, carried by the linear closure term Phi_i Phi_j c_ij
    ft = os.path.join(DIR, f"layer{li + 1}_seed0.npz")
    if os.path.exists(ft) and S[li].get("C_off") is not None:
        Cc0 = S[li]["C_off"].astype(np.float64); np.fill_diagonal(Cc0, S[li]["var"])
        dC0 = np.asarray(np.load(ft)["C"], np.float64) - Cc0; np.fill_diagonal(dC0, 0.0)
        ph = Phi(S[li]["alpha"].astype(np.float64))
        Xin = (ph[:, None] * dC0 * ph[None, :]).astype(np.float32)
        feats.append(Wn.T @ Xin @ Wn); names.append("inherited(Phi dC Phi)")
    Fo = np.stack([F[iu] for F in feats], 1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(Fo, y_off, rcond=None)
    r2 = 1 - np.sum((y_off - Fo @ coef) ** 2) / np.sum(y_off ** 2)
    # the inherited feature alone, and the terms alone
    ci, *_ = np.linalg.lstsq(Fo[:, -1:], y_off, rcond=None); r2i = 1 - np.sum((y_off - Fo[:, -1:] @ ci) ** 2) / np.sum(y_off ** 2)
    Fd = np.stack([np.diag(F) for F in feats], 1).astype(np.float64)
    rec = {"layer_next": l_next, "consistency": cons, "dC_off_rms": float(np.sqrt(np.mean(y_off ** 2))), "noise_off": noise_off,
           "dvar_rms": float(np.sqrt(np.mean(y_dg ** 2))), "dvar_mean": float(np.mean(y_dg)),
           "r2_all": float(r2), "r2_inherited_only": float(r2i), "coef": dict(zip(names, map(float, coef)))}
    # how much of the DIAGONAL (variance) error do the off-diagonal-fitted coefficients explain?
    rec["r2_diag_from_offfit"] = float(1 - np.sum((y_dg - Fd @ coef) ** 2) / np.sum(y_dg ** 2))
    out.append(rec)
    top = sorted(zip(names, coef), key=lambda z: -abs(z[1]))[:6]
    print(f"-> layer {l_next}: cons {cons:.1e} | dC_off rms {rec['dC_off_rms']:.2e} (noise {noise_off:.1e}) | dvar rms {rec['dvar_rms']:.2e} "
          f"mean {rec['dvar_mean']:+.1e} | R2 all {r2:.3f} inherited-only {r2i:.3f} diag-from-offfit {rec['r2_diag_from_offfit']:.3f} | "
          + ", ".join(f"{n} {c:+.3f}" for n, c in top), flush=True)
json.dump(out, open("../results/summaries/coverr_v29plain.json", "w"), indent=1)
