"""Round 8 (Fable, 23 Sep): END-TO-END gate of the kappa4 kernel WITH MEMORY, in the mean closure only.

Same harness as experiments/chain/a7_kernel_e2e.py (the kernel of probe_astra7_k4.py, computed in numpy from the chain's
own state at every layer, replaces the chain's kappa4 diagonal in the MEAN term only), plus the transported STAR
births of earlier layers (r8_k4_memory.py): at the pre-activation of layer l the correction is

    k4_new = A g4_chain + (1 - A) [ P_chain + S (F0 - P_H) + KC K3C + M sum_{ages 1..AGE} (6 beta P^2 At^2 + 4 gamma P^3 At + 4 w3 P At^3) ]

with the star legs (P = transport, At = transported a = diag(Phi) C_off) carried in numpy with exact dense transport (in a
deployment they are the kappa3 sources' own legs; the elementwise diagonal costs ~0.005 u per source-layer).  The kernel
and K3C are billed separately (~18-26 u for 3 products/layer; K3C 2 more products with the chain's D21).

Usage (from whest-starterkit):
  V31_ANGULAR=0 python -m uv run python ../experiments/probes/r8_k4_memory_e2e.py "A=1,S=0" "A=0,S=1,KC=1,M=1,AGE=4" ...
  optional keys: LMIN (first 1-based layer corrected), DEC (|mu/sigma| cut: only those neurons corrected), NETS=mlp0,synth
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
src = src.replace(a1, "            _CAPW(li, mu, var, None if trim else C_off, g4row, lam_prev, D3, None if trim else D21)\n" + a1)
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
CFG = {"A": 1.0, "S": 0.0, "KC": 0.0, "M": 0.0, "AGE": 4, "W": None, "LMIN": 1, "DEC": 0.0, "N": []}
S = {}; NEXT = {}; STARS = []          # STARS: list of dict(T, At, beta, gamma, w3, age) for the live births
def pairpoly(diag, coef22, W):
    W2 = W * W
    return (W2 * W2).T @ diag + 0.5 * np.einsum("ji,ji->i", W2, coef22 @ W2)
def _CAPW(li, mu, var, C_off, g4row, lam, D3, D21):
    S[li] = dict(mu=np.asarray(mu, np.float64).copy(), var=np.asarray(var, np.float64).copy(),
                 C_off=None if C_off is None else np.asarray(C_off, np.float64).copy(),
                 g4=None if g4row is None else np.asarray(g4row, np.float64).copy(), lam=float(lam),
                 D3=None if D3 is None else np.asarray(D3, np.float64).copy(), D21=None if D21 is None else np.asarray(D21, np.float64).copy())
def _KERNEL(li, K22, K4v):
    """called after the ReLU closure of the pre-activation with chain index li (1-based layer li+1); prepares the
    kappa4 diagonal of the NEXT pre-activation (li+1) and advances the star memory by one layer."""
    c = S.get(li)
    if c is None or li + 1 >= len(CFG["W"]): return
    W = CFG["W"][li + 1]
    mu, var = c["mu"], c["var"]; s = np.sqrt(var); a = mu / s; ph = pdf(a)
    d1, d2, d3 = Phi(a), ph / s, -a * ph / s ** 2
    Co = c["C_off"]
    if Co is None:                                       # layer 1: C_pre = W0^T W0 exactly
        W0 = CFG["W"][0]; Co = W0.T @ W0; np.fill_diagonal(Co, 0.0)
    # ---- star memory: transport the live births (T <- W^T diag(Phi) T; At likewise), then add this layer's birth ----
    for st in STARS:
        st["T"] = W.T @ (d1[:, None] * st["T"]); st["At"] = W.T @ (d1[:, None] * st["At"]); st["age"] += 1
    m_ = s * ph + mu * d1; Eh2 = (mu * mu + var) * d1 + mu * s * ph
    beta = 2 * d1 * (1 - d1) - 2 * m_ * ph / s; gamma = 3 * (1 - d1) * (Eh2 - 2 * m_ * m_)
    Tb = W.T.copy(); Atb = Tb @ (d1[:, None] * Co)
    STARS.append(dict(T=Tb, At=Atb, beta=beta, gamma=gamma, w3=d3, age=0))
    while STARS and STARS[0]["age"] > CFG["AGE"]: STARS.pop(0)
    if CFG["A"] == 1.0 or c["g4"] is None: return
    mem = np.zeros(len(mu))
    for st in STARS:
        if 1 <= st["age"] <= CFG["AGE"]:
            T, At = st["T"], st["At"]
            mem += 6 * ((T * T) * (At * At)) @ st["beta"] + 4 * ((T ** 3) * At) @ st["gamma"] + 4 * ((At ** 3) * T) @ st["w3"]
    # ---- Astra's fresh kernel (as in a7_kernel_e2e.py) ----
    C = Co.copy(); np.fill_diagonal(C, var); Cd = var
    g = c["g4"] / METRIC; lam = c["lam"]
    U = d1[:, None] * W; Bm = d2[:, None] * W; Cm = d3[:, None] * W
    Q = C @ U; T = Bm * Q; R = C @ T; H = lam * Q + (g - lam * Cd)[:, None] * U
    F0 = METRIC * np.sum(U * U, 0) * np.sum(U * H, 0) + 4 * np.sum(Cm * Q ** 3, 0) + 12 * np.sum(T * R, 0)
    o = np.outer
    dH = METRIC * d1 ** 4 * g + 4 * d3 * Cd ** 3 * d1 ** 3 + 12 * Cd ** 3 * d2 ** 2 * d1 ** 2
    cH = (METRIC * o(d1 ** 2, d1 ** 2) * (g[:, None] + g[None, :]) + 12 * Co ** 2 * (o(d3 * Cd * d1, d1 ** 2) + o(d1 ** 2, d3 * Cd * d1))
          + 12 * Co ** 2 * (o(Cd * d2 ** 2, d1 ** 2) + o(d1 ** 2, Cd * d2 ** 2)) + 24 * Co * (o(Cd * d1 * d2, Cd * d1 * d2) + Co ** 2 * o(d1 * d2, d1 * d2)))
    np.fill_diagonal(cH, 0.0)
    K22c = np.asarray(K22, np.float64).copy(); np.fill_diagonal(K22c, 0.0)
    kern = pairpoly(np.asarray(K4v, np.float64), 6.0 * K22c, W) + CFG["S"] * (F0 - pairpoly(dH, cH, W)) + CFG["M"] * mem
    if CFG["KC"] != 0.0 and c["D21"] is not None:        # GPT Sol's K3 x C '3+2' term with the slice approximation t_rep
        X = U; Rr = Co @ X; Ss = (d2[:, None] * W) * Rr
        D21c = c["D21"].copy(); np.fill_diagonal(D21c, 0.0)
        Trep = D21c.T @ (X * X) + 2 * X * (D21c @ X) - 2 * c["D3"][:, None] * (X * X)
        kern = kern + CFG["KC"] * 12 * np.sum(Ss * Trep, 0)
    NEXT[li + 1] = kern
def _MEANFIX(li, pk1v, mu, var, D3, g4row):
    if li not in NEXT or D3 is None or g4row is None: return pk1v
    m, v, k3, k4 = (np.asarray(x, np.float64) for x in (mu, var, D3, g4row))
    kern = NEXT.pop(li)
    if li + 1 < CFG["LMIN"]: return pk1v
    k4n = CFG["A"] * k4 + (1 - CFG["A"]) * kern
    if CFG["DEC"] > 0:
        sel = np.abs(m / np.sqrt(v)) < CFG["DEC"]; k4n = np.where(sel, k4n, k4); CFG["N"].append(int(sel.sum()))
    else: CFG["N"].append(len(m))
    return pk1v + fnp.asarray((edge(m, v, k3, k4n) - edge(m, v, k3, k4)).astype(np.float32))
mod = types.ModuleType("est_r8m"); mod.__dict__.update({"_CAPW": _CAPW, "_KERNEL": _KERNEL, "_MEANFIX": _MEANFIX})
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
NETS = {"mlp0": "../data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "../data/kaggle_out/kaggle_out_k4/k4teach"}
if os.environ.get("NETS"): NETS = {k: v for k, v in NETS.items() if k in os.environ["NETS"].split(",")}
OUT = f"../results/summaries/r8_k4_memory_e2e{os.environ.get('TAG', '')}.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
BASE = {"mlp0": 2.2903436065845552e-08, "synth": 1.9262352958825e-08}        # a7_kernel_e2e.json "A=1,S=0"
for nk, DIR in NETS.items():
    Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"]
    CFG["W"] = [w.astype(np.float64) for w in Wall]
    mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=0, name=nk)
    tf = os.path.join(DIR, "truth.npz")
    if os.path.exists(tf): lf, truth = 16, np.asarray(np.load(tf)["all_layer_means"][15], np.float64)
    else: lf, truth = 15, np.asarray(np.load(os.path.join(DIR, "layer15_seed0.npz"))["relu_mean"], np.float64)
    for cfg in sys.argv[1:]:
        kv = dict(x.split("=") for x in cfg.split(","))
        for k in ("A", "S", "KC", "M", "DEC"): CFG[k] = float(kv.get(k, {"A": 1.0, "S": 0.0, "KC": 0.0, "M": 0.0, "DEC": 0.0}[k]))
        CFG["AGE"] = int(kv.get("AGE", 4)); CFG["LMIN"] = int(kv.get("LMIN", 1)); CFG["N"] = []; S.clear(); NEXT.clear(); STARS.clear()
        est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            p = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        mse = float(np.mean((p[lf - 1] - truth) ** 2))
        res.setdefault(nk, {})[cfg] = mse
        nn_ = CFG["N"]; cost = (3 + (2 if CFG["KC"] else 0)) * sum(nn_) / 1024.0
        print(f"[{nk} {cfg}] layer-{lf} MSE {mse:.4e}  ratio to plain {mse / BASE[nk]:.4f}  | corrected layers {len(nn_)}, neurons/layer {nn_[:3]}.., kernel cost ~{cost:.1f} u (+ star memory ~0)  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
