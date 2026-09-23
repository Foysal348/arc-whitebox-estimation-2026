"""Round 7 (Astra) falsifier 2 (23 Sep): the quartic response kernel for the next kappa4 diagonal.

Astra's kernel (appendix C, verbatim): with u = d1 w, b = d2 w, c = d3 w, q = C u, t = b q, r = C t and the chain's
harmonic input kappa4 G = lam C + diag(g - lam diag C) (g = the chain's diagonal core dG, metric = 2):
    F0(w) = metric (sum u^2)(sum u h) + 4 sum c q^3 + 12 sum t r,     h = lam q + (g - lam diag C) u
(input kappa4 edge + kappa3-free Gaussian star + Gaussian path; the K3 x C term omitted in this first version).
Falsifier (Astra p.5): does the kernel's NON-PAIR part F0 - P_H predict the true non-pair part kappa4_next - P_true,
where P_true(w) = sum d_j w_j^4 + 3 sum_{j!=k} K22_jk w_j^2 w_k^2 uses the TRUE post-ReLU slices, and P_H is the same
pair polynomial of F0's own tensor (diagonal and (2,2) coefficients derived analytically below)?
Also the deployable version F = F0 - P_H + P_chain (pair content from the chain's own post slices) against the chain's
regenerated diagonal, G4^2-weighted on the next layer's neurons.

Pair coefficients of F0 (j != k; C includes its diagonal; u_j = d1_j w_j):
  f1:  diag  metric d1_j^4 g_j                         22-coef (of w_j^2 w_k^2)  metric d1_j^2 d1_k^2 (g_j + g_k)
  f2:  diag  4 d3_j C_jj^3 d1_j^3                      22-coef  12 C_jk^2 (d3_j C_jj d1_j d1_k^2 + d3_k C_kk d1_k d1_j^2)
  f3:  diag 12 C_jj^3 d2_j^2 d1_j^2                    22-coef  12 C_jk^2 (C_jj d2_j^2 d1_k^2 + C_kk d2_k^2 d1_j^2)
                                                                 + 24 C_jk (C_jj C_kk d1_j d2_j d1_k d2_k + C_jk^2 d1_j d1_k d2_j d2_k)
  (P(w) = sum diag_j w_j^4 + sum_{j<k} coef_jk w_j^2 w_k^2, i.e. H22_jk = coef_jk / 6.)
Usage (from whest-starterkit):  V31_ANGULAR=0 python -m uv run python ../experiments/probes/probe_astra7_k4.py
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
a2 = "            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]\n"
for a in (a1, a2): assert src.count(a) == 1
src = src.replace(a1, "            _CAPW(li, mu, var, None if trim else C_off, g4row, lam_prev)\n" + a1)
src = src.replace(a2, a2 + "            _CAPK(li, K22, K4v)\n")
POSTL = (3, 7, 11, 14, 15); S = {}
def _CAPW(li, mu, var, C_off, g4row, lam):
    d = S.setdefault(li, {}); d.update(mu=np.asarray(mu, np.float64).copy(), var=np.asarray(var, np.float64).copy(),
                                       g4=None if g4row is None else np.asarray(g4row, np.float64).copy(), lam=float(lam))
    if li + 1 in POSTL and C_off is not None: d["C_off"] = np.asarray(C_off, np.float64).copy()
def _CAPK(li, K22, K4v):
    if li + 1 in POSTL: S.setdefault(li, {}).update(K22=np.asarray(K22, np.float64).copy(), K4v=np.asarray(K4v, np.float64).copy())
mod = types.ModuleType("est_a7k"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPK": _CAPK})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
print(f"chain run {time.time()-t0:.0f}s", flush=True)
from math import erf
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
METRIC = 2.0
def pairpoly(diag, coef22, W):
    """P(w_i) = sum_j diag_j W_ji^4 + sum_{j<k} coef_jk W_ji^2 W_ki^2  (coef symmetric, zero diagonal)"""
    W2 = W * W
    return (W2 * W2).T @ diag + 0.5 * np.einsum("ji,ji->i", W2, coef22 @ W2)
out = []; SAVE = {}
for l in POSTL:
    li = l - 1; c = S[li]; n_ = S[li + 1]
    mu, var = c["mu"], c["var"]; s = np.sqrt(var); a = mu / s; ph = pdf(a)
    d1, d2, d3 = Phi(a), ph / s, -a * ph / s ** 2
    C = c["C_off"].copy(); np.fill_diagonal(C, var); Cd = var
    g = c["g4"] / METRIC; lam = c["lam"]                         # chain's diagonal core dG and lam at this layer
    W = Wall[li + 1].astype(np.float64)
    # ---- F0 for every next-layer neuron (w = column of W) ----
    U = d1[:, None] * W; B = d2[:, None] * W; Cm = d3[:, None] * W
    Q = C @ U; T = B * Q; R = C @ T
    H = lam * Q + (g - lam * Cd)[:, None] * U
    F0 = METRIC * np.sum(U * U, 0) * np.sum(U * H, 0) + 4 * np.sum(Cm * Q ** 3, 0) + 12 * np.sum(T * R, 0)
    parts = {"f1": METRIC * np.sum(U * U, 0) * np.sum(U * H, 0), "f2": 4 * np.sum(Cm * Q ** 3, 0), "f3": 12 * np.sum(T * R, 0)}
    # ---- F0's own pair polynomial P_H ----
    Co = c["C_off"]
    dH = METRIC * d1 ** 4 * g + 4 * d3 * Cd ** 3 * d1 ** 3 + 12 * Cd ** 3 * d2 ** 2 * d1 ** 2
    o = np.outer
    cH = (METRIC * o(d1 ** 2, d1 ** 2) * (g[:, None] + g[None, :])
          + 12 * Co ** 2 * (o(d3 * Cd * d1, d1 ** 2) + o(d1 ** 2, d3 * Cd * d1))
          + 12 * Co ** 2 * (o(Cd * d2 ** 2, d1 ** 2) + o(d1 ** 2, Cd * d2 ** 2))
          + 24 * Co * (o(Cd * d1 * d2, Cd * d1 * d2) + Co ** 2 * o(d1 * d2, d1 * d2)))
    np.fill_diagonal(cH, 0.0)
    PH = pairpoly(dH, cH, W)
    # ---- truth: next-layer kappa4 diagonal and its true pair part ----
    t4 = np.asarray(np.load(os.path.join(PRE, f"layer{l + 1}_seed0.npz"))["K4d"], np.float64)
    P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz"))
    K22t = np.asarray(P["K22"], np.float64).copy(); np.fill_diagonal(K22t, 0.0)
    Ptrue = pairpoly(np.asarray(P["K4d"], np.float64), 6.0 * K22t, W)
    K22c = c["K22"].copy(); np.fill_diagonal(K22c, 0.0)
    Pchain = pairpoly(c["K4v"], 6.0 * K22c, W)
    target = t4 - Ptrue; pred = F0 - PH
    s1 = np.sqrt(n_["var"]); a1 = n_["mu"] / s1; w = ((a1 * a1 - 1) * pdf(a1) / s1 ** 3) ** 2
    def ev(v, ref=t4):
        e = v - ref; return {"corr": float(np.corrcoef(v, ref)[0, 1]), "slope": float(np.dot(v, ref) / np.dot(v, v)),
                             "errG4": float(np.sqrt(np.sum(w * e * e) / np.sum(w * ref * ref)))}
    m = np.abs(a1) < 1.5
    rec = {"post_layer": l, "nonpair_pred_vs_target": ev(pred, target),
           "nonpair_corr_decisive": float(np.corrcoef(pred[m], target[m])[0, 1]),
           "chain_regen": ev(n_["g4"]), "F_deploy(P_chain+F0-P_H)": ev(Pchain + pred), "F_oracle_pairs(P_true+F0-P_H)": ev(Ptrue + pred),
           "rms": {"t4": float(np.sqrt(np.mean(t4 ** 2))), "target": float(np.sqrt(np.mean(target ** 2))), "pred": float(np.sqrt(np.mean(pred ** 2))),
                   **{k: float(np.sqrt(np.mean(v_ ** 2))) for k, v_ in parts.items()}, "PH": float(np.sqrt(np.mean(PH ** 2))), "Ptrue": float(np.sqrt(np.mean(Ptrue ** 2)))}}
    SAVE.update({f"L{l+1}_{k_}": v_ for k_, v_ in dict(t4=t4, Ptrue=Ptrue, Pchain=Pchain, pred=pred, g4=n_["g4"], w=w, a1=a1).items()})
    out.append(rec)
    npv = rec["nonpair_pred_vs_target"]
    print(f"post L{l} -> kappa4 at L{l+1}: NON-PAIR kernel vs truth: corr {npv['corr']:+.3f} (decisive neurons {rec['nonpair_corr_decisive']:+.3f}) slope {npv['slope']:.2f} | "
          f"errG4: chain {rec['chain_regen']['errG4']:.3f}, F deploy {rec['F_deploy(P_chain+F0-P_H)']['errG4']:.3f}, F w/ true pairs {rec['F_oracle_pairs(P_true+F0-P_H)']['errG4']:.3f} | "
          f"corr: chain {rec['chain_regen']['corr']:.3f}, F deploy {rec['F_deploy(P_chain+F0-P_H)']['corr']:.3f}", flush=True)
    print("    rms: " + ", ".join(f"{k} {v:.1e}" for k, v in rec["rms"].items()), flush=True)
json.dump(out, open("../results/summaries/astra7_k4.json", "w"), indent=1)
np.savez_compressed("../data/dumps/astra7_k4_mlp0.npz", **SAVE)
