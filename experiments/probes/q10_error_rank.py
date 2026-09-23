"""Queue Q10 (GPT 5.6 Terra R7 Q1.2): is the post-ReLU covariance error E low-rank as seen by the target quadratic forms?

E = C_chain_post - C_true_post at layers 7 and 14 (post-ReLU teacher). Target correction dv_i = w_i^T E w_i for the next
layer's neurons with |mu/sigma| < 0.75. Rank-k eigen-approximation dv_k = sum_a lam_a (w_i^T u_a)^2 for k = 4..128:
relative rms error on the targets. Also: the overlap of E's top eigenvectors with known directions (the post-ReLU
mean vector m, the all-ones vector, the top eigenvectors of C_true), and the share of E's Frobenius energy in its
top k eigenpairs (the teacher's noise is itself full rank, so a low-rank E would show as a steep spectrum).

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/q10_error_rank.py
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"
SRC = "../submissions/v31ang/estimator.py"
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a3 = "            C = flops.as_symmetric(C, symmetry=(0, 1))\n"
src = src.replace(a1, "            _CAPW(li, mu, var)\n" + a1).replace(a3, a3 + "            _CAPC(li, C)\n")
LAY = (7, 14); S = {}
def _CAPW(li, mu, var): S.setdefault(li, {}).update(mu=np.asarray(mu, np.float64).copy(), var=np.asarray(var, np.float64).copy())
def _CAPC(li, C):
    if li + 1 in LAY: S.setdefault(li, {})["C"] = np.asarray(C, np.float64).copy()
mod = types.ModuleType("est_q10"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPC": _CAPC})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx())
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
out = []
for l in LAY:
    P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz")); Ct = np.asarray(P["C"], np.float64); m = np.asarray(P["mean"], np.float64)
    E = S[l - 1]["C"] - Ct; E = 0.5 * (E + E.T)
    W = Wall[l].astype(np.float64); n_ = S[l]; a1 = n_["mu"] / np.sqrt(n_["var"]); T = np.abs(a1) < 0.75
    WT = W[:, T]; dv = np.sum(WT * (E @ WT), 0)
    lam, U = np.linalg.eigh(E); order = np.argsort(-np.abs(lam)); lam, U = lam[order], U[:, order]
    energy = np.cumsum(lam ** 2) / np.sum(lam ** 2)
    proj = (U.T @ WT) ** 2                                      # (k, m)
    rec = {"layer": l, "n_targets": int(T.sum()), "dv_rms": float(np.sqrt(np.mean(dv ** 2)))}
    line = f"post L{l}: targets {T.sum()}, dv rms {rec['dv_rms']:.2e} | "
    for k in (4, 8, 16, 32, 64, 128, 256):
        dvk = lam[:k] @ proj[:k]
        rec[f"k{k}"] = {"rel_err": float(np.sqrt(np.mean((dvk - dv) ** 2) / np.mean(dv ** 2))), "energy": float(energy[k - 1])}
        line += f"k={k}: err {rec[f'k{k}']['rel_err']:.2f} (energy {energy[k-1]:.2f})  "
    lt, Ut = np.linalg.eigh(Ct); Ut = Ut[:, np.argsort(-lt)]
    ov = lambda v: float(np.max(np.abs(U[:, :8].T @ (v / np.linalg.norm(v)))))
    rec["overlap_top8_with"] = {"mean_vector": ov(m), "ones": ov(np.ones(1024)), "C_top1": ov(Ut[:, 0]), "C_top2": ov(Ut[:, 1])}
    out.append(rec)
    print(line, flush=True)
    print(f"    max |overlap| of E's top-8 eigenvectors with: mean {rec['overlap_top8_with']['mean_vector']:.3f}, ones {rec['overlap_top8_with']['ones']:.3f}, "
          f"C top-1 {rec['overlap_top8_with']['C_top1']:.3f}, C top-2 {rec['overlap_top8_with']['C_top2']:.3f} (random ~{np.sqrt(2*np.log(8)/1024):.3f})", flush=True)
json.dump(out, open("../results/summaries/q10_error_rank.json", "w"), indent=1)
