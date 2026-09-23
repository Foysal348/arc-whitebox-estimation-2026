"""Queue Q5 (Astra R7 §1): teacher-forced bivariate closure - does the fresh post-ReLU covariance error come from the
closure itself or from its inputs?

At ONE layer l, the chain's closure inputs are replaced by the pre-activation teacher's values (a group at a time), the
chain computes its post-ReLU covariance C_post, and C_post is compared with the post-ReLU teacher at the same layer
(same input samples). Groups: gauss = (mu, var, C_off); k3 = (D3, D21); k4 = (kappa4 diagonal, (2,2) slice wk4m,
(3,1) slice wk431); all = everything. If 'all' leaves most of the error, the closure truncation is the cause (higher
bivariate orders); if one group removes it, that input is the cause. Scored as the off-diagonal rms error and the
next layer's variance error it implies, weighted by the mean-closure sensitivity (phi(a')/(2 s'))^2.

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_tf_closure.py
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
assert os.environ.get("V31_ANGULAR") == "0"
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"
SRC = "../submissions/v31ang/estimator.py"
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a3 = "            C = flops.as_symmetric(C, symmetry=(0, 1))\n"
for a in (a1, a3): assert src.count(a) == 1
src = src.replace(a1, "            if not trim and D21 is not None:\n"
                      "                mu, var, C_off, D3, D21, g4row, wk4m, wk431 = _TF(li, mu, var, C_off, D3, D21, g4row, wk4m, wk431)\n"
                      "            _CAPW(li, mu, var)\n" + a1)
src = src.replace(a3, a3 + "            _CAPC(li, C)\n")
CFG = {"layer": None, "set": "none"}; CAP = {}
def _TF(li, mu, var, C_off, D3, D21, g4row, wk4m, wk431):
    if CFG["set"] == "none" or li + 1 != CFG["layer"]: return mu, var, C_off, D3, D21, g4row, wk4m, wk431
    T = np.load(os.path.join(PRE, f"layer{li + 1}_seed0.npz")); f32 = lambda x: fnp.asarray(np.asarray(x, np.float32))
    st = CFG["set"]
    if st in ("all", "gauss"):
        C = np.asarray(T["C"], np.float64); mu = f32(T["mean"]); var = f32(np.diag(C)); Co = C.copy(); np.fill_diagonal(Co, 0.0); C_off = f32(Co)
    if st in ("all", "k3"):
        Y = np.asarray(T["Y21"], np.float64); D3 = f32(np.diag(Y)); Yo = Y.copy(); np.fill_diagonal(Yo, 0.0); D21 = f32(Yo)
    if st in ("all", "k4"):
        g4row = f32(T["K4d"])
        K22 = np.asarray(T["K22"], np.float64).copy(); np.fill_diagonal(K22, 0.0); wk4m = f32(K22)
        K31 = np.asarray(T["K31"], np.float64).copy(); np.fill_diagonal(K31, 0.0); wk431 = f32(K31)
    return mu, var, C_off, D3, D21, g4row, wk4m, wk431
def _CAPW(li, mu, var): CAP.setdefault(li, {}).update(mu=np.asarray(mu, np.float64).copy(), var=np.asarray(var, np.float64).copy())
def _CAPC(li, C):
    if li + 1 == CFG["layer"]: CAP.setdefault(li, {})["C"] = np.asarray(C, np.float64).copy()
mod = types.ModuleType("est_tf"); mod.__dict__.update({"_TF": _TF, "_CAPW": _CAPW, "_CAPC": _CAPC})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
off = ~np.eye(1024, dtype=bool); out = []
LAYERS = [int(x) for x in os.environ.get("TF_LAYERS", "7,14").split(",")]
for l in LAYERS:
    P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz")); Ct = np.asarray(P["C"], np.float64); v = np.diag(Ct)
    noise = float(np.sqrt(np.mean((np.outer(v, v) + Ct * Ct)[off] / 6e7)))
    W = Wall[l].astype(np.float64)                      # next pre-activation z_(l+1) = h_l @ W[l]
    base = None
    for st in ("none", "all", "gauss", "k3", "k4"):
        CFG.update(layer=l, set=st); CAP.clear()
        est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            est.predict(mlp, 2**41)
        E = CAP[l - 1]["C"] - Ct
        n_ = CAP[l]; s1 = np.sqrt(n_["var"]); a1 = n_["mu"] / s1; wsens = (pdf(a1) / (2 * s1)) ** 2
        Eo = E.copy(); np.fill_diagonal(Eo, 0.0); Ed = np.diag(np.diag(E))
        pv_o = np.sum(W * (Eo @ W), 0); pv_d = np.sum(W * (Ed @ W), 0)
        rec = {"layer": l, "set": st, "off_rms": float(np.sqrt(np.mean(E[off] ** 2))), "diag_rms": float(np.sqrt(np.mean(np.diag(E) ** 2))),
               "proj_off": float(np.sum(wsens * pv_o ** 2)), "proj_diag": float(np.sum(wsens * pv_d ** 2)), "noise_off": noise}
        if base is None: base = rec
        out.append(rec)
        print(f"L{l} TF[{st:5s}]: C_post off rms {rec['off_rms']:.2e} (x{rec['off_rms']/base['off_rms']:.3f}; noise {noise:.1e}) | diag rms {rec['diag_rms']:.2e} "
              f"| weighted next-var error from off {rec['proj_off']/base['proj_off']:.3f}, from diag {rec['proj_diag']/max(base['proj_diag'],1e-300):.3f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open("../results/summaries/tf_closure.json", "w"), indent=1)
