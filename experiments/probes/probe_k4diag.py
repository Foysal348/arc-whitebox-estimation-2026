"""Moonshot probe (23 Sep): can the kappa4 DIAGONAL be computed accurately at ~one dense product per layer?

The corrected oracle (k4_oracle_run2.py) found that replacing ONLY the pre-activation kappa4 diagonal by its true value
cuts the final MSE of public MLP #0 from 2.290e-8 to 1.609e-8 (-30%); the off-diagonal slices add nothing. The chain's
diagonal comes from a memoryless regeneration: dG = (W*W) g_prev + lam (var - (W*W) var_prev), with g_prev a harmonic
projection of the post-ReLU (K4 diagonal + K22 row sums). Exact algebra for z = h W (column w_i of W):

    k4(z_i) =  sum_j W_ji^4 K4_j                                   (all four indices equal)
            + 3 sum_{j != k} W_ji^2 W_ki^2 K22_jk                  (two pairs)   = 3 diag((W^2)^T K22 (W^2))
            + 4 sum_{j != k} W_ji^3 W_ki K31_jk                    (three + one)
            + 6 sum_j W_ji^2 sum_{k != m} W_ki W_mi K211_jkm + sum_{all distinct} ...

The first two use only the post-ReLU diagonal and (2,2) slice that the chain already computes; the quadratic form costs
one n^3 product ((W^2)^T K22) per layer. This probe measures how close each candidate is to the teacher's kappa4 diagonal
at the next layer, overall and on the neurons that decide the mean (G4^2-weighted).

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_k4diag.py
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

DIR = os.environ.get("TEACHER", "../data/kaggle_out/kaggle_out_k4pub/k4teach")
SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a2 = "            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]\n"
for a in (a1, a2): assert src.count(a) == 1, a
src = src.replace(a1, "            _CAPW(li, mu, var, g4row, lam_prev)\n" + a1)
src = src.replace(a2, a2 + "            _CAPK(li, K22, K4v, K2v)\n")
S = {}
f64 = lambda x: None if x is None else np.asarray(x, dtype=np.float64).copy()
def _CAPW(li, mu, var, g4row, lam):
    S.setdefault(li, {}).update(mu=f64(mu), var=f64(var), g4=f64(g4row), lam=float(lam) if lam is not None else 0.0)
def _CAPK(li, K22, K4v, K2v):
    S.setdefault(li, {}).update(K22=np.asarray(K22, np.float32).copy(), K4v=f64(K4v), K2v=f64(K2v))
mod = types.ModuleType("est_k4d"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPK": _CAPK})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(DIR, "truth.npz"))
Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])],
          seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
print(f"chain run {time.time()-t0:.0f}s", flush=True)

pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
out = []
for li in range(0, 15):
    nx = li + 1                                              # chain index of the next pre-activation
    f = os.path.join(DIR, f"layer{nx + 1}_seed0.npz")
    if nx not in S or "K22" not in S[li] or S[nx].get("g4") is None or not os.path.exists(f): continue
    t4 = np.asarray(np.load(f)["K4d"], np.float64)
    W = Wall[nx].astype(np.float64); W2 = W * W
    K22 = S[li]["K22"].astype(np.float64); np.fill_diagonal(K22, 0.0)
    diag4 = W2.T @ S[li]["K4v"] if False else (W2 * W2).T @ S[li]["K4v"]      # sum_j W_ji^4 K4_j
    pair = 3.0 * np.einsum("ji,ji->i", W2, K22 @ W2)                            # 3 diag(W2^T K22 W2)
    chain = S[nx]["g4"]
    mu, var = S[nx]["mu"], S[nx]["var"]; s = np.sqrt(var); a = mu / s
    G4 = (a * a - 1) * pdf(a) / s ** 3; w = G4 ** 2
    cands = {"chain": chain, "exact_pairwise": diag4 + pair, "pair_only": pair}
    # best linear combination of the chain and the pairwise form (what a cheap blend could reach)
    X = np.stack([chain, diag4 + pair], 1); cb, *_ = np.linalg.lstsq(X, t4, rcond=None); cands["blend_ls"] = X @ cb
    rec = {"layer": nx + 1, "blend_coef": list(map(float, cb))}
    line = f"L{nx + 1:2d}"
    for k, v in cands.items():
        e = v - t4
        rec[k] = {"corr": float(np.corrcoef(v, t4)[0, 1]), "slope": float(np.dot(v, t4) / np.dot(v, v)),
                  "rel_err": float(np.sqrt(np.mean(e * e) / np.mean(t4 * t4))),
                  "rel_err_G4w": float(np.sqrt(np.sum(w * e * e) / np.sum(w * t4 * t4)))}
        line += f" | {k}: corr {rec[k]['corr']:.3f} slope {rec[k]['slope']:.2f} err {rec[k]['rel_err']:.3f} errG4 {rec[k]['rel_err_G4w']:.3f}"
    out.append(rec); print(line + f" | blend {cb[0]:.2f},{cb[1]:.2f}", flush=True)
json.dump(out, open("../results/summaries/k4diag_probe.json", "w"), indent=1)
