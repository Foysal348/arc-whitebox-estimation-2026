"""Round 8 (Fable, 24 Sep): END-TO-END gate of the kappa4 kernel + the ALL-DISTINCT K3 x C '3+2' term from the chain's own
sources (r8_k4_1111.py), in the mean closure only.

At the pre-activation of chain index li the chain's kappa3 sources (the stacks passed to _dslices: ages >= 1) give, for
every column w of the next weight W[li+1] (x = d1 o w, s = (d2 o w) o (C_off x)):
    delta1111 = 12 sum_j s_j t_dist_j,   t_dist = sum_{sources, age <= AGE} [ t_source - t_rep(source's own slices) ]
with t_source = (1/3)[X (P^T x o Y^T x) + P (X^T x o Y^T x) + Y (X^T x o P^T x)] per hub block (X1,P,Y1), (M,P,P), (X3,P,Y3).
The next pre-activation's kappa4 diagonal in the MEAN term becomes
    k4_new = A g4_chain + (1 - A) [ P_chain + S (F0 - P_H) + KC K3C_rep + DIST delta1111 ]
(a7_kernel_e2e.py conventions; K3C_rep = GPT Sol's repeated-index term with the chain's D21, D3).  Computed in numpy
here (accuracy test); the deployable cost is 2 products (A^T S, P^T S) per source-layer at the query width m plus thin
work (Astra R7 section 5: ~54 u at m = 350 over the full source schedule; ~41 u for ages 1-4 only), on top of the
kernel's ~18-26 u.

Usage (from whest-starterkit):
  V31_ANGULAR=0 python -m uv run python ../experiments/probes/r8_k4_dist_e2e.py "A=0,S=1,KC=1,DIST=1,AGE=4" ...
  keys: A S KC DIST AGE LMIN (first corrected 1-based layer) DEC (|mu/sigma| cut); env NETS=mlp0,synth TAG=...
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
a5 = "                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,\n"
for a in (a1, a2, a4, a5): assert src.count(a) == 1, a[:50]
src = src.replace(a1, "            _CAPW(li, mu, var, None if trim else C_off, g4row, lam_prev, D3, None if trim else D21)\n" + a1)
src = src.replace(a2, a2 + "            _KERNEL(li, K22, K4v)\n")
src = src.replace(a4, a4 + "            pk1v = _MEANFIX(li, pk1v, mu, var, D3, g4row)\n")
src = src.replace(a5, "                _CAPS(li, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list, c1_list, c2_list, y_list, r, rfb, Zf_st, R1T_st, R2T_st)\n" + a5)
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
CFG = {"A": 0.0, "S": 1.0, "KC": 1.0, "DIST": 1.0, "AGE": 4, "W": None, "LMIN": 1, "DEC": 0.0, "N": [], "NSRC": []}
S = {}; NEXT = {}; STK = {}
f64 = lambda x: np.asarray(x, np.float64)
def pairpoly(diag, coef22, W):
    W2 = W * W
    return (W2 * W2).T @ diag + 0.5 * np.einsum("ji,ji->i", W2, coef22 @ W2)
def _CAPW(li, mu, var, C_off, g4row, lam, D3, D21):
    S[li] = dict(mu=f64(mu).copy(), var=f64(var).copy(), C_off=None if C_off is None else f64(C_off).copy(),
                 g4=None if g4row is None else f64(g4row).copy(), lam=float(lam),
                 D3=None if D3 is None else f64(D3).copy(), D21=None if D21 is None else f64(D21).copy())
def _CAPS(li, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list, c1_list, c2_list, y_list, rres, rfb, Zf_st, R1T_st, R2T_st):
    if CFG["A"] == 1.0 or CFG["DIST"] == 0.0: return
    k = A_st.shape[0]; s0 = max(0, k - int(CFG["AGE"]))
    STK[li] = dict(A=[f64(A_st[s]) for s in range(s0, k)], P=[f64(P_st[s]) for s in range(s0, k)], Z=[f64(Z_st[s]) for s in range(s0, k)], L=[f64(L_st[s]) for s in range(s0, k)],
                   w2b=[f64(w2b_list[s]) for s in range(s0, k)], sb=[f64(s_list[s]) for s in range(s0, k)], eb=[f64(e_list[s]) for s in range(s0, k)],
                   c1=[f64(c1_list[s]) for s in range(s0, k)], c2=[f64(c2_list[s]) for s in range(s0, k)], rres=int(rres), rfb=int(rfb),
                   Zf=None if Zf_st is None else [f64(Zf_st[s]) for s in range(s0, k)], R1=None if R1T_st is None else [f64(R1T_st[s]) for s in range(s0, k)],
                   R2=None if R2T_st is None else [f64(R2T_st[s]) for s in range(s0, k)], k=k)
def _t_of_block(X, P, Y, Xc, PX):
    XX = X.T @ Xc; YX = Y.T @ Xc
    t = (X @ (PX * YX) + P @ (XX * YX) + Y @ (XX * PX)) / 3.0
    sl21 = ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0; sl3 = np.sum(X * P * Y, axis=1)
    return t, sl21, sl3
def _dist_term(li, Xc, Ss):
    st = STK.pop(li, None)
    if st is None: return 0.0, 0
    n = Xc.shape[0]; tot = np.zeros_like(Xc); rres, rfb = st["rres"], st["rfb"]
    for i in range(len(st["A"])):
        A, P, Z, L = st["A"][i], st["P"][i], st["Z"][i], st["L"][i]
        M = P * st["sb"][i][None, :] + 3.0 * A * st["eb"][i][None, :] + Z @ L.T
        if rfb > 0 and st["Zf"] is not None:
            Zf = st["Zf"][i]; Xt = Zf[:, :rfb] @ st["R1"][i].T; Yt = Zf[:, rfb:] @ st["R2"][i].T
        else: Xt = Yt = 0.0
        X1 = 3.0 * A + Xt; Y1 = A * st["w2b"][i][None, :] + Yt
        X3 = A * st["c1"][i][None, :] + P * st["c2"][i][None, :]; y = Z[:, rres + 1]; Y3 = np.repeat(y[:, None], n, axis=1)
        PX = P.T @ Xc; ts = np.zeros_like(Xc); sl21 = np.zeros((n, n)); sl3 = np.zeros(n)
        for (X, Pb, Y) in ((X1, P, Y1), (M, P, P), (X3, P, Y3)):
            t, s21, s3 = _t_of_block(X, Pb, Y, Xc, PX); ts += t; sl21 += s21; sl3 += s3
        np.fill_diagonal(sl21, 0.0)
        trep = sl21.T @ (Xc * Xc) + 2 * Xc * (sl21 @ Xc) - 2 * sl3[:, None] * (Xc * Xc)
        tot += ts - trep
    return 12 * np.sum(Ss * tot, 0), len(st["A"])
def _KERNEL(li, K22, K4v):
    c = S.get(li)
    if c is None or CFG["A"] == 1.0 or c["C_off"] is None or c["g4"] is None or li + 1 >= len(CFG["W"]): STK.pop(li, None); return
    W = CFG["W"][li + 1]
    mu, var = c["mu"], c["var"]; s = np.sqrt(var); a = mu / s; ph = pdf(a)
    d1, d2, d3 = Phi(a), ph / s, -a * ph / s ** 2
    Co = c["C_off"]; C = Co.copy(); np.fill_diagonal(C, var); Cd = var
    g = c["g4"] / METRIC; lam = c["lam"]
    U = d1[:, None] * W; Bm = d2[:, None] * W; Cm = d3[:, None] * W
    Q = C @ U; T = Bm * Q; R = C @ T; H = lam * Q + (g - lam * Cd)[:, None] * U
    F0 = METRIC * np.sum(U * U, 0) * np.sum(U * H, 0) + 4 * np.sum(Cm * Q ** 3, 0) + 12 * np.sum(T * R, 0)
    o = np.outer
    dH = METRIC * d1 ** 4 * g + 4 * d3 * Cd ** 3 * d1 ** 3 + 12 * Cd ** 3 * d2 ** 2 * d1 ** 2
    cH = (METRIC * o(d1 ** 2, d1 ** 2) * (g[:, None] + g[None, :]) + 12 * Co ** 2 * (o(d3 * Cd * d1, d1 ** 2) + o(d1 ** 2, d3 * Cd * d1))
          + 12 * Co ** 2 * (o(Cd * d2 ** 2, d1 ** 2) + o(d1 ** 2, Cd * d2 ** 2)) + 24 * Co * (o(Cd * d1 * d2, Cd * d1 * d2) + Co ** 2 * o(d1 * d2, d1 * d2)))
    np.fill_diagonal(cH, 0.0)
    K22c = f64(K22).copy(); np.fill_diagonal(K22c, 0.0)
    kern = pairpoly(f64(K4v), 6.0 * K22c, W) + CFG["S"] * (F0 - pairpoly(dH, cH, W))
    Xc = U; Ss = Bm * (Co @ Xc)
    if CFG["KC"] != 0.0 and c["D21"] is not None:
        D21c = c["D21"].copy(); np.fill_diagonal(D21c, 0.0)
        Trep = D21c.T @ (Xc * Xc) + 2 * Xc * (D21c @ Xc) - 2 * c["D3"][:, None] * (Xc * Xc)
        kern = kern + CFG["KC"] * 12 * np.sum(Ss * Trep, 0)
    d = 0.0
    if CFG["DIST"] != 0.0:
        d, ns = _dist_term(li, Xc, Ss); kern = kern + CFG["DIST"] * d; CFG["NSRC"].append(ns)
    if os.environ.get("R8_DUMP"):
        DUMP[f"L{li+2}_pred"] = F0 - pairpoly(dH, cH, W); DUMP[f"L{li+2}_Pchain"] = pairpoly(f64(K4v), 6.0 * K22c, W)
        DUMP[f"L{li+2}_K3Crep"] = kern - DUMP[f"L{li+2}_Pchain"] - CFG["S"] * DUMP[f"L{li+2}_pred"] - CFG["DIST"] * np.asarray(d); DUMP[f"L{li+2}_dist"] = np.asarray(d) + np.zeros(len(mu)); DUMP[f"L{li+2}_g4next"] = None
    NEXT[li + 1] = kern
DUMP = {}
def _MEANFIX(li, pk1v, mu, var, D3, g4row):
    if li not in NEXT or D3 is None or g4row is None: return pk1v
    m, v, k3, k4 = (f64(x) for x in (mu, var, D3, g4row))
    if os.environ.get("R8_DUMP"): DUMP[f"L{li+1}_g4chain"] = k4.copy(); DUMP[f"L{li+1}_mu"] = m.copy(); DUMP[f"L{li+1}_var"] = v.copy()
    kern = NEXT.pop(li)
    if li + 1 < CFG["LMIN"]: return pk1v
    k4n = CFG["A"] * k4 + (1 - CFG["A"]) * kern
    if CFG["DEC"] > 0:
        sel = np.abs(m / np.sqrt(v)) < CFG["DEC"]; k4n = np.where(sel, k4n, k4); CFG["N"].append(int(sel.sum()))
    else: CFG["N"].append(len(m))
    return pk1v + fnp.asarray((edge(m, v, k3, k4n) - edge(m, v, k3, k4)).astype(np.float32))
mod = types.ModuleType("est_r8d"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPS": _CAPS, "_KERNEL": _KERNEL, "_MEANFIX": _MEANFIX})
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
NETS = {"mlp0": "../data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "../data/kaggle_out/kaggle_out_k4/k4teach"}
if os.environ.get("NETS"): NETS = {k: v for k, v in NETS.items() if k in os.environ["NETS"].split(",")}
OUT = f"../results/summaries/r8_k4_dist_e2e{os.environ.get('TAG', '')}.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
BASE = {"mlp0": 2.2903436065845552e-08, "synth": 1.9262352958825e-08}        # a7_kernel_e2e.json "A=1,S=0" (same harness, plain chain)
for nk, DIR in NETS.items():
    Wall = np.load(glob.glob(os.path.join(DIR, "weights_seed*.npz"))[0])["W"]
    CFG["W"] = [w.astype(np.float64) for w in Wall]
    mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=0, name=nk)
    tf = os.path.join(DIR, "truth.npz")
    if os.path.exists(tf): lf, truth = 16, np.asarray(np.load(tf)["all_layer_means"][15], np.float64)
    else: lf, truth = 15, np.asarray(np.load(os.path.join(DIR, "layer15_seed0.npz"))["relu_mean"], np.float64)
    for cfg in sys.argv[1:]:
        kv = dict(x.split("=") for x in cfg.split(","))
        for k, dflt in (("A", 0.0), ("S", 1.0), ("KC", 1.0), ("DIST", 1.0), ("DEC", 0.0)): CFG[k] = float(kv.get(k, dflt))
        CFG["AGE"] = int(kv.get("AGE", 4)); CFG["LMIN"] = int(kv.get("LMIN", 1)); CFG["N"] = []; CFG["NSRC"] = []; S.clear(); NEXT.clear(); STK.clear()
        est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            p = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        mse = float(np.mean((p[lf - 1] - truth) ** 2))
        res.setdefault(nk, {})[cfg] = {"mse": mse, "ratio": mse / BASE[nk], "neurons": CFG["N"], "nsrc": CFG["NSRC"]}
        nn_ = CFG["N"]; ns = CFG["NSRC"]
        cost_k = (3 + (2 if CFG["KC"] else 0)) * sum(nn_) / 1024.0; cost_d = 2 * sum(ns) * (np.mean(nn_) if nn_ else 0) / 1024.0 if CFG["DIST"] else 0.0
        print(f"[{nk} {cfg}] layer-{lf} MSE {mse:.4e}  ratio to plain {mse / BASE[nk]:.4f}  | corrected layers {len(nn_)} (neurons/layer {nn_[:2]}..), sources used/layer {ns[:6]}.. | est. cost: kernel ~{cost_k:.0f} u, dist ~{cost_d:.0f} u (classical, m = corrected neurons)  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        if os.environ.get("R8_DUMP"):
            np.savez_compressed(f"../data/dumps/r8_dist_e2e_{nk}_{cfg.replace(',', '_').replace('=', '')}.npz", **{k: v for k, v in DUMP.items() if v is not None}); DUMP.clear()
