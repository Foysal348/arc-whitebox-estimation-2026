"""End-to-end test of Astra's quartic kappa4 kernel (23 Sep), in the mean closure only (the channel the oracle showed
carries the whole kappa4 gain). At every layer l >= 2 the chain's kappa4 diagonal in the MEAN term is replaced by

    k4_new = A * g4_chain + (1 - A) * (P_chain + S * (F0 - P_H))

where F0 is the kernel evaluated on the next layer's weight columns from the previous layer's C, mu, var, dG, lam, and
P_chain / P_H are the pair polynomials of the chain's post-ReLU slices / of the kernel (probe_astra7_k4.py). A = 0.3,
S = 1.25 were chosen as one global rule on the G4-weighted kappa4 error of both teacher networks (mean error x0.923,
every layer < 1). The kernel is computed in numpy here: this measures the ACCURACY gain; its cost (~18-26 u for all or
the decisive neurons) is billed separately.

Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/chain/a7_kernel_e2e.py "A=0.3,S=1.25" "A=1,S=0" ...
"""
import os, sys, types, json, time, glob
import numpy as np
from math import erf
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
assert os.environ.get("V31_ANGULAR") == "0"
SRC = "../submissions/v31ang/estimator.py"
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a2 = "            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]\n"
a4 = "            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]\n"
for a in (a1, a2, a4): assert src.count(a) == 1
src = src.replace(a1, "            _CAPW(li, mu, var, None if trim else C_off, g4row, lam_prev)\n" + a1)
src = src.replace(a2, a2 + "            _KERNEL(li, K22, K4v)\n")
src = src.replace(a4, a4 + "            pk1v = _MEANFIX(li, pk1v, mu, var, D3, g4row)\n")
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
def He(k, x):
    h0, h1 = np.ones_like(x), x
    if k == 0: return h0
    for j in range(1, k): h0, h1 = h1, x * h1 - j * h0
    return h1
def dG(k, mu, s):
    x = mu / s
    if k == 0: return s * pdf(x) + mu * Phi(x)
    if k == 1: return Phi(x)
    return (-1) ** k * He(k - 2, x) * pdf(x) / s ** (k - 1)
def edge(mu, var, k3, k4):
    s = np.sqrt(var)
    return dG(0, mu, s) + k3 / 6 * dG(3, mu, s) + k4 / 24 * dG(4, mu, s) + k3 ** 2 / 72 * dG(6, mu, s) + k3 * k4 / 144 * dG(7, mu, s)
METRIC = 2.0
CFG = {"A": 1.0, "S": 0.0, "W": None, "LMIN": 1, "DEC": 0.0, "M": []}
S = {}; NEXT = {}
def pairpoly(diag, coef22, W):
    W2 = W * W
    return (W2 * W2).T @ diag + 0.5 * np.einsum("ji,ji->i", W2, coef22 @ W2)
def _CAPW(li, mu, var, C_off, g4row, lam):
    S[li] = dict(mu=np.asarray(mu, np.float64).copy(), var=np.asarray(var, np.float64).copy(),
                 C_off=None if C_off is None else np.asarray(C_off, np.float64).copy(),
                 g4=None if g4row is None else np.asarray(g4row, np.float64).copy(), lam=float(lam))
def _KERNEL(li, K22, K4v):
    c = S.get(li)
    if CFG["A"] == 1.0 or c is None or c["C_off"] is None or c["g4"] is None or li + 1 >= len(CFG["W"]): return
    mu, var = c["mu"], c["var"]; s = np.sqrt(var); a = mu / s; ph = pdf(a)
    d1, d2, d3 = Phi(a), ph / s, -a * ph / s ** 2
    Co = c["C_off"]; C = Co.copy(); np.fill_diagonal(C, var); Cd = var
    g = c["g4"] / METRIC; lam = c["lam"]; W = CFG["W"][li + 1]
    U = d1[:, None] * W; B = d2[:, None] * W; Cm = d3[:, None] * W
    Q = C @ U; T = B * Q; R = C @ T; H = lam * Q + (g - lam * Cd)[:, None] * U
    F0 = METRIC * np.sum(U * U, 0) * np.sum(U * H, 0) + 4 * np.sum(Cm * Q ** 3, 0) + 12 * np.sum(T * R, 0)
    o = np.outer
    dH = METRIC * d1 ** 4 * g + 4 * d3 * Cd ** 3 * d1 ** 3 + 12 * Cd ** 3 * d2 ** 2 * d1 ** 2
    cH = (METRIC * o(d1 ** 2, d1 ** 2) * (g[:, None] + g[None, :]) + 12 * Co ** 2 * (o(d3 * Cd * d1, d1 ** 2) + o(d1 ** 2, d3 * Cd * d1))
          + 12 * Co ** 2 * (o(Cd * d2 ** 2, d1 ** 2) + o(d1 ** 2, Cd * d2 ** 2)) + 24 * Co * (o(Cd * d1 * d2, Cd * d1 * d2) + Co ** 2 * o(d1 * d2, d1 * d2)))
    np.fill_diagonal(cH, 0.0)
    K22c = np.asarray(K22, np.float64).copy(); np.fill_diagonal(K22c, 0.0)
    NEXT[li + 1] = pairpoly(np.asarray(K4v, np.float64), 6.0 * K22c, W) + CFG["S"] * (F0 - pairpoly(dH, cH, W))
def _MEANFIX(li, pk1v, mu, var, D3, g4row):
    if li not in NEXT or D3 is None or g4row is None: return pk1v
    m, v, k3, k4 = (np.asarray(x, np.float64) for x in (mu, var, D3, g4row))
    kern = NEXT.pop(li)
    if li + 1 < CFG["LMIN"]: return pk1v                        # 1-based layer of this pre-activation below the window
    k4n = CFG["A"] * k4 + (1 - CFG["A"]) * kern
    if CFG["DEC"] > 0:                                          # only the neurons the mean closure's kappa4 term looks at
        sel = np.abs(m / np.sqrt(v)) < CFG["DEC"]; k4n = np.where(sel, k4n, k4); CFG["M"].append(int(sel.sum()))
    else:
        CFG["M"].append(len(m))
    return pk1v + fnp.asarray((edge(m, v, k3, k4n) - edge(m, v, k3, k4)).astype(np.float32))
mod = types.ModuleType("est_a7e"); mod.__dict__.update({"_CAPW": _CAPW, "_KERNEL": _KERNEL, "_MEANFIX": _MEANFIX})
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
NETS = {"mlp0": "../data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "../data/kaggle_out/kaggle_out_k4/k4teach"}
res = {}
for nk, DIR in NETS.items():
    Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"]
    CFG["W"] = [w.astype(np.float64) for w in Wall]
    mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=0, name=nk)
    tf = os.path.join(DIR, "truth.npz")
    if os.path.exists(tf): lf, truth = 16, np.asarray(np.load(tf)["all_layer_means"][15], np.float64)
    else: lf, truth = 15, np.asarray(np.load(os.path.join(DIR, "layer15_seed0.npz"))["relu_mean"], np.float64)
    for cfg in sys.argv[1:]:
        kv = dict(x.split("=") for x in cfg.split(",")); CFG["A"] = float(kv["A"]); CFG["S"] = float(kv["S"])
        CFG["LMIN"] = int(kv.get("LMIN", 1)); CFG["DEC"] = float(kv.get("DEC", 0)); CFG["M"] = []; S.clear(); NEXT.clear()
        est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            p = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        mse = float(np.mean((p[lf - 1] - truth) ** 2))
        res.setdefault(nk, {})[cfg] = mse
        base = res[nk][sys.argv[1]]
        mm = CFG["M"]; cost = 3 * sum(mm) / 1024.0                # three n x n x m products per corrected layer, classical
        print(f"[{nk} {cfg}] layer-{lf} MSE {mse:.4e}  ratio {mse / base:.4f}  | corrected layers {len(mm)}, neurons {mm}, extra cost ~{cost:.1f} u  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open("../results/summaries/a7_kernel_e2e.json", "w"), indent=1)
