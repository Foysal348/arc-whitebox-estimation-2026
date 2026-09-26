"""Phase A, candidate A4 (kappa4 riding the kappa3 source legs: Round-13 'carried pair innovations', queue Q22), the queue's
own cheap pre-check, run in its most generous form (26 Sep 2026).

Design: every kappa3 source s carries two extra per-hub-column coefficient vectors (gamma_s, eta_s) for a kappa4 memory
tensor sum_j gamma_sj Sym(p_j p_j^T x P_s P_s^T) + eta_sj p_j^(x4) built on the SAME transported P leg (p_j = column j of
P_s); the next-layer kappa4 diagonal is read without any dense product:  q = 6 (Z gamma) o rowsum(Z) + (Z o Z) eta,
Z = P o P.  Cost ~0.35 u (n^2 elementwise per source-layer).

Pre-check (queue Q22 'cheap pre-check: fit only per-source scalars gamma_s, eta_s (26 parameters); if even the best
26-parameter fit gives corr < 0.3, stop'). Generous form used here: the scalars are fitted DIRECTLY to the quantity the
mean closure consumes - the G4-weighted residual (teacher K4d - chain ck4) of the pre-activation kappa4 diagonal at the
readout layer, on the chain's own P legs captured there (r7_cap_L15.npz, 13 sources, and r7_cap_L14.npz), i.e. an
in-sample upper bound of the per-source-scalar family; plus a half/half neuron split for the out-of-sample number.
Also reported: the same with FREE per-source, per-age scaling of rowsum(Z)^2 vs rowsum(Z o Z) split by source age.
Gate (Q22): decisive pattern corr of (chain + q) >= 0.45 AND errG4 <= 0.85 x chain (out of sample); kill if the in-sample
best corr < 0.30 or errG4 > 0.93 x chain.

Usage (repo root):  python experiments/probes/archA_q22_precheck.py   -> results/summaries/archA_q22_precheck.json
"""
import numpy as np, os, json
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; CAP = "data/dumps"
pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
D = np.load("data/dumps/errbudget_v29plain.npz")
res = {}
for cap in ("r7_cap_L14.npz", "r7_cap_L15.npz"):
    c = np.load(os.path.join(CAP, cap)); L = int(c["li"]) + 1          # pre-activation layer of the legs (as in r13)
    Pst = c["P_st"]; k = Pst.shape[0]; n = Pst.shape[1]
    F = []
    for s in range(k):
        P = np.asarray(Pst[s], np.float64); Z = P * P
        F.append(6.0 * Z.sum(1) ** 2); F.append((Z * Z).sum(1))
    F = np.stack(F, 1)                                                   # (n, 2k)
    t = np.load(os.path.join(PRE, f"layer{L}_seed0.npz")); mu = np.asarray(t["mean"], np.float64); v = np.diag(np.asarray(t["C"], np.float64))
    s_ = np.sqrt(v); a = mu / s_; w = (a * a - 1) * pdf(a) / s_ ** 3 / 24; dec = np.abs(a) < 1.5
    tk4 = np.asarray(t["K4d"], np.float64); ck4 = np.asarray(D[f"L{L}_ck4"], np.float64)
    def met(k4, idx=slice(None)):
        m = np.zeros(n, bool); m[idx] = True
        return {"errG4": float(np.linalg.norm((w * (k4 - tk4))[m]) / np.linalg.norm((w * tk4)[m])),
                "dec_corr": float(np.corrcoef((k4 / v ** 2)[dec & m], (tk4 / v ** 2)[dec & m])[0, 1])}
    def fit(idx, cols):
        A = w[idx, None] * cols[idx]; y = w[idx] * (tk4 - ck4)[idx]; sc = np.sqrt(np.mean(A * A, 0)) + 1e-300
        cf = np.linalg.solve((A / sc).T @ (A / sc) + 1e-6 * len(y) * np.eye(A.shape[1]), (A / sc).T @ y) / sc
        return cf
    cols = np.c_[F, v ** 2]
    cf = fit(np.arange(n), cols)
    r = {"n_sources": k, "chain": met(ck4), "in_sample_26": met(ck4 + cols @ cf)}
    rng = np.random.default_rng(0); outs = []
    for rep in range(20):
        p = rng.permutation(n); A_, B_ = p[: n // 2], p[n // 2:]
        cfA = fit(A_, cols); outs.append((met(ck4 + cols @ cfA, B_), met(ck4, B_)))
    r["neuron_split_errG4_ratio"] = float(np.mean([o[0]["errG4"] / o[1]["errG4"] for o in outs]))
    r["neuron_split_dec_corr"] = float(np.mean([o[0]["dec_corr"] for o in outs]))
    r["neuron_split_dec_corr_chain"] = float(np.mean([o[1]["dec_corr"] for o in outs]))
    # the q feature alone (pattern check, best in-sample combination without the chain)
    cq = fit(np.arange(n), np.c_[F, v ** 2]); q = np.c_[F, v ** 2] @ cq
    r["q_alone_dec_corr_with_residual"] = float(np.corrcoef(((tk4 - ck4) / v ** 2)[dec], (q / v ** 2)[dec])[0, 1])
    res[f"L{L}"] = r
    print(f"legs at pre L{L} ({k} sources): chain errG4 {r['chain']['errG4']:.3f} corr {r['chain']['dec_corr']:.3f} | "
          f"26-scalar in-sample errG4 {r['in_sample_26']['errG4']:.3f} corr {r['in_sample_26']['dec_corr']:.3f} | "
          f"neuron-split errG4 ratio x{r['neuron_split_errG4_ratio']:.3f}, corr {r['neuron_split_dec_corr']:.3f} "
          f"(chain {r['neuron_split_dec_corr_chain']:.3f}) | q vs residual corr {r['q_alone_dec_corr_with_residual']:.3f}", flush=True)
best_in = max(res[k]["in_sample_26"]["dec_corr"] for k in res)
ratio_in = min(res[k]["in_sample_26"]["errG4"] / res[k]["chain"]["errG4"] for k in res)
res["gate"] = {"best_in_sample_corr": best_in, "best_in_sample_errG4_ratio": ratio_in,
               "verdict": "KILL" if (best_in < 0.30 or ratio_in > 0.93) else ("PASS" if min(res[k]["neuron_split_errG4_ratio"] for k in res if k.startswith("L")) <= 0.85 else "FAIL")}
print("GATE:", res["gate"])
json.dump(res, open("results/summaries/archA_q22_precheck.json", "w"), indent=1)
