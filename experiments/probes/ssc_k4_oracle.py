"""Puffi's SSC response closure for kappa4 (forum 18175, Phase 1 write-up #327725), teacher-forced one-step test (26 Sep).
SSC: kappa4 ~ sum_ab Lambda_ab Sym(B_a x B_b) from <= 4 live symmetric modes built from the post-ReLU state:
  B1 = C, B2 = Q / vbar, B3 = vbar I, B4_ij = (g_i + g_j)/2 C_ij,  g = standardised centred ReLU gate probabilities,
  Q = Poff[ c0 vbar C + c1 C o C + c2/2 (D_m K^T + K D_m) + c3 (t m^T + m t^T) + c4 v v^T + c5 vbar^2 1 1^T ]
  (K_ij = kappa3_iij, t = kappa3 diag, v = diag C); readout for next-layer neuron j: kappa4_j = 3 q_j^T Lambda q_j,
  q_aj = w_j^T B_a w_j.  Puffi fitted c per layer to kappa_iiij + kappa_jjji and Lambda to the next marginal kappa4.
Here: all nine basis quadratic forms (C, vbar I, gate-C, and the six Q features) are computed from the TEACHER post-ReLU
state at layers 3/7/11/14/15 (MLP #0) and the next pre-activation kappa4 diagonal is regressed on their pairwise
products (a linear superset of 3 q^T Lambda q, since B2 is itself linear in the features).
  model 'ssc4': 4 modes with B2 = Poff(C o C) only (10 coefficients); model 'ssc9': all nine features (45 coefficients).
Fit on one layer, TEST on the others (out-of-layer), on the excess kurtosis kappa4 / var^2 of the decisive neurons
(|mu/sigma| < 1.5, teacher). Baselines: chain 0.26-0.34 (L12-16), exact linear carry 0.77-0.80.

Usage:  python experiments/probes/ssc_k4_oracle.py
"""
import numpy as np, os, json, itertools
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
f = lambda x: np.nan_to_num(np.asarray(x, np.float64))
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5)))
D = np.load("data/dumps/errbudget_v29plain.npz")
n = 1024; data = {}
for l in (3, 7, 11, 14, 15):
    P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz")); T0 = np.load(os.path.join(PRE, f"layer{l}_seed0.npz"))
    C = f(P["C"]); m = f(P["mean"]); K = f(P["Y21"]).copy(); t = np.diag(K).copy(); np.fill_diagonal(K, 0.0)
    v = np.diag(C).copy(); vbar = v.mean()
    mu0 = f(T0["mean"]); s0 = np.sqrt(np.diag(f(T0["C"]))); g = Phi(mu0 / s0); g = (g - g.mean()) / g.std()
    Poff = lambda X: X - np.diag(np.diag(X))
    feats = {"C": C, "vI": vbar * np.eye(n), "gC": 0.5 * (g[:, None] + g[None, :]) * C,
             "Q_vC": Poff(vbar * C), "Q_CC": Poff(C * C), "Q_mK": Poff(0.5 * (m[:, None] * K.T + K * m[None, :])),
             "Q_tm": Poff(np.outer(t, m) + np.outer(m, t)), "Q_vv": Poff(np.outer(v, v)), "Q_11": Poff(vbar ** 2 * np.ones((n, n)))}
    Q = W[l]                                          # z_{l+1} = h_l @ W[l]; column j = w_j
    q = {k: np.sum(Q * (B @ Q), 0) for k, B in feats.items()}          # q_kj = w_j^T B_k w_j
    Tn = np.load(os.path.join(PRE, f"layer{l + 1}_seed0.npz")); mun = f(Tn["mean"]); vn = np.diag(f(Tn["C"]))
    dec = np.abs(mun / np.sqrt(vn)) < 1.5
    data[l] = {"q": q, "y": f(Tn["K4d"]) / vn ** 2, "dec": dec, "vn": vn, "chain": D[f"L{l + 1}_ck4"] / vn ** 2}
    print(f"layer {l} -> {l + 1}: features ready, decisive {dec.sum()}", flush=True)

def design(l, keys):
    q = data[l]["q"]; cols = [q[a] * q[b] for a, b in itertools.combinations_with_replacement(keys, 2)]
    return np.stack(cols, 1) / data[l]["vn"][:, None] ** 2
models = {"ssc4": ["C", "vI", "gC", "Q_CC"], "ssc9": list(data[3]["q"].keys())}
out = {}
for name, keys in models.items():
    out[name] = {}
    for lf in (7, 11, 14):
        X = design(lf, keys); d = data[lf]["dec"]; y = data[lf]["y"]
        Xs = X[d]; sc = Xs.std(0) + 1e-30; A = np.c_[Xs / sc, np.ones(d.sum())]
        lam = 1e-3 * len(y[d]); beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[d])
        row = {}
        for lt in (3, 7, 11, 14, 15):
            Xt = design(lt, keys); dt = data[lt]["dec"]; pred = np.c_[Xt[dt] / sc, np.ones(dt.sum())] @ beta
            row[f"L{lt + 1}"] = {"corr": float(np.corrcoef(pred, data[lt]["y"][dt])[0, 1]),
                                 "chain_corr": float(np.corrcoef(data[lt]["chain"][dt], data[lt]["y"][dt])[0, 1])}
        out[name][f"fit_L{lf + 1}"] = row
        print(f"{name} fit on L{lf + 1}: " + "  ".join(f"{k} {v['corr']:+.3f} (chain {v['chain_corr']:+.3f})" for k, v in row.items()), flush=True)
json.dump(out, open("results/summaries/ssc_k4_oracle.json", "w"), indent=1)
