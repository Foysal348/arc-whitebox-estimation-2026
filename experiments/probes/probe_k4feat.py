"""Moonshot probe (23 Sep): is the TRUE kappa4 diagonal predictable from quantities the chain already has?

The corrected oracle: knowing the pre-activation kappa4 diagonal exactly is worth -30% final MSE (MLP #0). The chain's
regenerated diagonal dG = t_g + lam t_v (t_g = (W*W) g_prev transports the post-ReLU core, t_v = var - (W*W) var_prev is
the off-diagonal part of the variance) has corr 0.70-0.83 with the truth. probe_k4diag.py showed the literal pairwise
transport of the chain's own post-ReLU slices is worse (corr 0.24-0.69).

This probe dumps, per layer and neuron, the chain's cheap candidate features and the teacher's diagonal, for both teacher
networks, so that a G4^2-weighted regression can be fitted on one network and tested on the other (analysis below).

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_k4feat.py
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a2 = "            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]\n"
a3 = "                        dG = t_g + t_v * lam_prev                 # var == diag(C_pre)\n"
for a in (a1, a2, a3): assert src.count(a) == 1, a
src = src.replace(a1, "            _CAPW(li, mu, var, g4row, D3, None if trim else D21)\n" + a1)
src = src.replace(a2, a2 + "            _CAPK(li, K22, K4v, K2v, K3v)\n")
src = src.replace(a3, a3 + "                        _CAPG(li, t_g, t_v, lam_prev)\n")
S = {}
f64 = lambda x: None if x is None else np.asarray(x, dtype=np.float64).copy()
def _CAPW(li, mu, var, g4row, D3, D21):
    d = S.setdefault(li, {}); d.update(mu=f64(mu), var=f64(var), g4=f64(g4row), D3=f64(D3))
    if D21 is not None:
        D = np.asarray(D21, np.float64); d.update(d21_rowsq=np.sum(D * D, 1), d21_colsq=np.sum(D * D, 0), d21_rowsum=D.sum(1))
def _CAPK(li, K22, K4v, K2v, K3v):
    K = np.asarray(K22, np.float64)
    S.setdefault(li, {}).update(K4v=f64(K4v), K2v=f64(K2v), K3v=f64(K3v), k22row=K.sum(1), k22rowsq=np.sum(K * K, 1))
def _CAPG(li, t_g, t_v, lam):
    S.setdefault(li, {}).update(t_g=f64(t_g), t_v=f64(t_v), lam=float(lam))
mod = types.ModuleType("est_k4f"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPK": _CAPK, "_CAPG": _CAPG})
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"

NETS = {"mlp0": "../data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "../data/kaggle_out/kaggle_out_k4/k4teach"}
SAVE = {}
for nk, DIR in NETS.items():
    S.clear()
    Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"]
    mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=0, name=nk)
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        est.predict(mlp, 2**41)
    print(f"{nk}: chain run {time.time()-t0:.0f}s", flush=True)
    for li in sorted(S):
        f = os.path.join(DIR, f"layer{li + 1}_seed0.npz")
        if not os.path.exists(f) or S[li].get("g4") is None or "t_g" not in S[li] or (li - 1) not in S: continue
        T = np.load(f)
        W = Wall[li].astype(np.float64); W2 = W * W; P = S[li - 1]
        feats = {
            "t4": np.asarray(T["K4d"], np.float64), "tk3": np.diag(np.asarray(T["Y21"], np.float64)).copy(),
            "tvar": np.diag(np.asarray(T["C"], np.float64)).copy(), "tmu": np.asarray(T["mean"], np.float64),
            "g4": S[li]["g4"], "t_g": S[li]["t_g"], "t_v": S[li]["t_v"], "lam": np.full(1024, S[li]["lam"]),
            "mu": S[li]["mu"], "var": S[li]["var"], "D3": S[li]["D3"],
            "d21_rowsq": S[li].get("d21_rowsq", np.zeros(1024)), "d21_colsq": S[li].get("d21_colsq", np.zeros(1024)),
            "d21_rowsum": S[li].get("d21_rowsum", np.zeros(1024)),
            # previous post-ReLU quantities transported by W*W (O(n^2) each)
            "ww_K4v": W2.T @ P["K4v"], "ww_k22row": W2.T @ P["k22row"], "ww_K2v2": W2.T @ (P["K2v"] ** 2),
            "ww_K3v2": W2.T @ (P["K3v"] ** 2 / np.maximum(P["K2v"], 1e-12)), "ww_k22rowsq": W2.T @ P["k22rowsq"],
            "w4_K4v": (W2 * W2).T @ P["K4v"],
        }
        for k, v in feats.items(): SAVE[f"{nk}_L{li + 1}_{k}"] = v
        print(f"   layer {li + 1}: saved", flush=True)
np.savez_compressed("../data/dumps/k4feat.npz", **SAVE)
print("saved ../data/dumps/k4feat.npz")
