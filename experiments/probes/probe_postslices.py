"""Moonshot probe (23 Sep): the chain's POST-ReLU slices against the post-ReLU teacher (kaggle/k4/run_post).

Two questions, both about the object the next layer is built from:
 1. kappa4 diagonal (worth -30%, K3ac). The next pre-activation diagonal is, exactly,
        k4(z_i) = (W^4)^T K4_post + 3 diag((W^2)^T K22_post W^2) + 4 diag((W^3)^T K31_post W) + [(2,1,1) and (1,1,1,1) terms]
    With the TRUE post slices: how much of the true diagonal do the pairwise terms explain? And how far is the chain's own
    K22_post / K4_post from the truth, entry by entry?
 2. variance (50% of the fresh error, K3ab): the chain's post-ReLU covariance error dC_post, measured directly: diagonal vs
    off-diagonal share of the implied variance error, and the regression of dC_post on the chain's own closure terms.

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_postslices.py
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

PRE = os.environ.get("TEACHER", "../data/kaggle_out/kaggle_out_k4pub/k4teach")
POST = os.environ.get("TEACHER_POST", "../data/kaggle_out/kaggle_out_k4post/k4post")
SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a2 = "            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]\n"
a3 = "            C = flops.as_symmetric(C, symmetry=(0, 1))\n"
a4 = "            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]\n"
for a in (a1, a2, a3, a4): assert src.count(a) == 1, a
src = src.replace(a1, "            _CAPW(li, mu, var, g4row)\n" + a1)
src = src.replace(a2, a2 + "            _CAPK(li, K22, K21, K4v, K2v, K3v)\n")
src = src.replace(a3, a3 + "            _CAPC(li, C)\n")
src = src.replace(a4, a4 + "            _CAPT(li, None if trim else (ABstack, WL2, WR2, fpm['IND2'], prog['AB2']))\n")
S = {}
f64 = lambda x: None if x is None else np.asarray(x, dtype=np.float64).copy()
def _CAPW(li, mu, var, g4row): S.setdefault(li, {}).update(mu=f64(mu), var=f64(var), g4=f64(g4row))
def _CAPK(li, K22, K21, K4v, K2v, K3v):
    S.setdefault(li, {}).update(K22=np.asarray(K22, np.float32).copy(), K21=np.asarray(K21, np.float32).copy(), K4v=f64(K4v), K2v=f64(K2v), K3v=f64(K3v))
def _CAPC(li, C): S.setdefault(li, {})["C"] = np.asarray(C, np.float32).copy()
def _CAPT(li, two):
    if two is None: return
    AB, WL, WR, IND, names = two
    AB = np.asarray(AB, np.float32); WL = np.asarray(WL, np.float32); WR = np.asarray(WR, np.float32); IND = np.asarray(IND)
    terms = []
    for t in range(AB.shape[0]):
        if IND[t, 0] == 0: continue                               # group 0 = (1,1): the off-diagonal covariance terms
        X = AB[t] * np.outer(WL[t], WR[t]); X = 0.5 * (X + X.T); np.fill_diagonal(X, 0.0)
        terms.append((f"{names[t][0]}*{names[t][1]}", X.astype(np.float32)))
    S.setdefault(li, {})["terms11"] = terms
mod = types.ModuleType("est_ps"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPK": _CAPK, "_CAPC": _CAPC, "_CAPT": _CAPT})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz"))
Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])],
          seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
print(f"chain run {time.time()-t0:.0f}s", flush=True)

pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
iu = np.triu_indices(Wall.shape[1], 1)
def cmp(a, b):
    """slope of truth on chain, R^2 of the chain as a predictor, relative rms error"""
    s = float(np.dot(a, b) / np.dot(a, a)); r2 = float(1 - np.sum((b - a) ** 2) / np.sum((b - b.mean()) ** 2))
    return {"slope": s, "r2": r2, "corr": float(np.corrcoef(a, b)[0, 1]), "rel_err": float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b * b)))}
out = []
for f in sorted(glob.glob(os.path.join(POST, "layer*_post_seed0.npz")), key=lambda p: int(os.path.basename(p).split("_")[0][5:])):
    l = int(os.path.basename(f).split("_")[0][5:]); li = l - 1; nx = li + 1
    P = np.load(f)
    tK22 = np.asarray(P["K22"], np.float64); tK31 = np.asarray(P["K31"], np.float64); tK4 = np.asarray(P["K4d"], np.float64)
    tC = np.asarray(P["C"], np.float64); tm = np.asarray(P["mean"], np.float64)
    c = S[li]; rec = {"post_layer": l}
    cK22 = c["K22"].astype(np.float64)
    rec["K22_post"] = cmp(cK22[iu], tK22[iu]); rec["K4_post"] = cmp(c["K4v"], tK4)
    cC = c["C"].astype(np.float64)
    rec["C_post_off"] = cmp(cC[iu], tC[iu]); rec["var_post"] = cmp(np.diag(cC), np.diag(tC))
    rec["dC_post_off_rms"] = float(np.sqrt(np.mean((cC - tC)[iu] ** 2))); rec["dvar_post_rms"] = float(np.sqrt(np.mean((np.diag(cC) - np.diag(tC)) ** 2)))
    rec["noise_C_off"] = float(np.sqrt(np.mean(np.outer(np.diag(tC), np.diag(tC))[iu]) / 6e7))
    # implied pre-activation variance error at the next layer: diagonal vs off-diagonal parts of dC_post
    W = Wall[nx].astype(np.float64); W2 = W * W
    dC = cC - tC; dCd = np.diag(np.diag(dC)); dCo = dC - dCd
    dv_d = np.einsum("ji,ji->i", W, dCd @ W); dv_o = np.einsum("ji,ji->i", W, dCo @ W)
    rec["dvar_next_from_diag_rms"] = float(np.sqrt(np.mean(dv_d ** 2))); rec["dvar_next_from_off_rms"] = float(np.sqrt(np.mean(dv_o ** 2)))
    # kappa4 diagonal of the next pre-activation from the TRUE post slices
    fpre = os.path.join(PRE, f"layer{l + 1}_seed0.npz")
    if os.path.exists(fpre) and nx in S and S[nx].get("g4") is not None:
        t4 = np.asarray(np.load(fpre)["K4d"], np.float64)
        K22o = tK22.copy(); np.fill_diagonal(K22o, 0.0); K31o = tK31.copy(); np.fill_diagonal(K31o, 0.0)
        d4 = (W2 * W2).T @ tK4; p22 = 3 * np.einsum("ji,ji->i", W2, K22o @ W2); p31 = 4 * np.einsum("ji,ji->i", W2 * W, K31o @ W)
        cK22o = cK22.copy(); np.fill_diagonal(cK22o, 0.0)
        pair_chain = (W2 * W2).T @ c["K4v"] + 3 * np.einsum("ji,ji->i", W2, cK22o @ W2)
        mu, var = S[nx]["mu"], S[nx]["var"]; s = np.sqrt(var); al = mu / s; w = ((al * al - 1) * pdf(al) / s ** 3) ** 2
        def ev(v):
            e = v - t4
            return {"corr": float(np.corrcoef(v, t4)[0, 1]), "slope": float(np.dot(v, t4) / np.dot(v, v)), "errG4": float(np.sqrt(np.sum(w * e * e) / np.sum(w * t4 * t4)))}
        rec["k4next"] = {"chain_regen": ev(S[nx]["g4"]), "pair_true(d4+22)": ev(d4 + p22), "pair_true(d4+22+31)": ev(d4 + p22 + p31),
                         "pair_chainslices": ev(pair_chain),
                         "rms": {"t4": float(np.sqrt(np.mean(t4 ** 2))), "d4": float(np.sqrt(np.mean(d4 ** 2))), "p22": float(np.sqrt(np.mean(p22 ** 2))), "p31": float(np.sqrt(np.mean(p31 ** 2)))}}
    # which (1,1) closure term explains the off-diagonal covariance error
    if "terms11" in c:
        y = -(dC[iu]); names = [nm for nm, _ in c["terms11"]]
        X = np.stack([X_[iu] for _, X_ in c["terms11"]], 1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum(y ** 2)
        rec["dCoff_on_terms"] = {"r2": float(r2), "coef": dict(zip(names, map(float, coef)))}
    out.append(rec)
    print(f"post L{l}: K22 slope {rec['K22_post']['slope']:.3f} R2 {rec['K22_post']['r2']:.3f} | K4 slope {rec['K4_post']['slope']:.3f} R2 {rec['K4_post']['r2']:.3f} "
          f"| C_off R2 {rec['C_post_off']['r2']:.4f} dC_off rms {rec['dC_post_off_rms']:.2e} (noise {rec['noise_C_off']:.1e}) | next dvar from diag {rec['dvar_next_from_diag_rms']:.2e} off {rec['dvar_next_from_off_rms']:.2e}", flush=True)
    if "k4next" in rec:
        k = rec["k4next"]; print("   k4 next: " + " | ".join(f"{nm}: corr {v['corr']:.3f} slope {v['slope']:.2f} errG4 {v['errG4']:.3f}" for nm, v in k.items() if nm != "rms") + f" | rms {k['rms']}", flush=True)
    if "dCoff_on_terms" in rec:
        top = sorted(rec["dCoff_on_terms"]["coef"].items(), key=lambda z: -abs(z[1]))[:6]
        print(f"   dC_off on the closure terms: R2 {rec['dCoff_on_terms']['r2']:.3f} | " + ", ".join(f"{n} {v:+.3f}" for n, v in top), flush=True)
json.dump(out, open("../results/summaries/postslices.json", "w"), indent=1)
