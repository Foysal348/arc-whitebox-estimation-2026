"""Phase A, candidate A1 (P-SSC, Team Puffi's stack), falsifier F1b: the NONLINEAR NEIGHBOUR-FEATURE DRIFT CORRECTOR (26 Sep 2026).

Puffi's shipped stack (P1 #327725, App. D) adds m <- m + DriftCorrector(...) at every late layer: a 31 -> 32 -> 32 -> 1 tanh
MLP on 14 per-neuron features and 17 NEIGHBOUR features built from the pre-activation correlation matrix R (the top-8
|R_ij| neighbours j_it of neuron i: S_i(x) = sum_t r_it x_{j_it}, U_i(x) = sum_t |r_it| x_{j_it}, and kernel averages
P_gamma(x) = sum_j exp(gamma R_ij^2) x_j / sum_j exp(gamma R_ij^2)), trained on the teacher-forced one-step mean drift.
Q19 (our dead item) used 13 LINEAR per-neuron features only: in-sample R^2 0.02 (chance 0.013).

Data: the Q19 teacher-forced dump of public MLP #0 (data/q19_dumps_smoke/k4teach.npz: the chain is fed the TRUE post-ReLU
mean at every layer; target = truth / a1 - chain post-ReLU mean = the fresh one-step drift; 13 chain features) plus the
31 Puffi features computed from the 6e7-sample teacher's pre-activation state of the same layer (C, mean, kappa3
diagonal, kappa3 (2,1) slice) - a close proxy of the chain's own state (C error 0.1-0.7%).
No second network has all-layer truth locally (HF mini split incomplete; synthetic teacher only at even layers), so
held-out = (i) a random half of the NEURONS (same layers), (ii) held-out LAYERS (fit 9-12, test 13-16 and reverse).
Models: ridge (linear), and the 31(+13) -> 32 -> 32 -> 1 tanh MLP (numpy Adam, early stopping on an inner split).
Gate (pre-registered, audit F4): held-out R^2 >= 0.10; kill < 0.05.

Usage (repo root):  python experiments/probes/archA_drift_nbr.py      -> results/summaries/archA_drift_nbr.json
"""
import numpy as np, os, json, time
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
Q = np.load("data/q19_dumps_smoke/k4teach.npz")
F13, TGT = Q["feats"].astype(np.float64), Q["target"].astype(np.float64)       # (16, n, 13), (16, n); index li = pre layer li+1
n = 1024; t0 = time.time()
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)

def puffi31(l):
    t = np.load(os.path.join(PRE, f"layer{l}_seed0.npz"))
    C = np.asarray(t["C"], np.float64); C = 0.5 * (C + C.T); mu = np.asarray(t["mean"], np.float64)
    v = np.diag(C).copy(); s = np.sqrt(v); a = mu / s; g = Phi(a); ph = pdf(a)
    Y = np.asarray(t["Y21"], np.float64); k3 = np.diag(Y).copy(); K = Y.copy(); np.fill_diagonal(K, 0.0)
    eta = np.clip(k3 / s ** 3, -20, 20); ell = np.log(v / v.mean())
    R = C / np.outer(s, s); np.fill_diagonal(R, 0.0)
    # top-8 subspace sketch of C: squared row loading
    w_, V_ = np.linalg.eigh(C); pi = np.sum(V_[:, -8:] ** 2, 1)
    f0 = [a, g, eta, np.log(s / s.mean()), R.mean(1), (R * R).mean(1), ell, C @ a / n, C @ (C @ a) / n ** 2,
          R @ g / n, C @ g / n, K.sum(1) / n, (K * K).sum(1) / n, pi]
    idx = np.argsort(-np.abs(R), axis=1)[:, :8]; r = np.take_along_axis(R, idx, 1); u = np.abs(r)
    S = lambda x: np.sum(r * x[idx], 1); U = lambda x: np.sum(u * x[idx], 1)
    def P(gam, x):
        E = np.exp(gam * R * R); np.fill_diagonal(E, 0.0); return (E @ x) / E.sum(1)
    f1 = [S(a), S(g), S(ph), S(eta), U(ph), U(ell), np.sum(r * r, 1), u[:, 0], P(8, a), P(8, g), P(8, ph),
          P(32, a), P(32, g), P(32, ph), r[:, 0], a[idx[:, 0]], ph[idx[:, 0]]]
    return np.stack(f0 + f1, 1)

LAY = list(range(8, 16))                       # li = 8..15  (pre layers 9..16, Puffi's late range)
X31 = {li: puffi31(li + 1) for li in LAY}
print(f"features ready ({time.time()-t0:.0f}s); target rms per layer:", " ".join(f"{np.sqrt(np.mean(TGT[li]**2)):.1e}" for li in LAY), flush=True)

def design(li, which):
    parts = []
    if "31" in which: parts.append(X31[li])
    if "13" in which: parts.append(F13[li][:, 1:])          # drop the constant column
    return np.concatenate(parts, 1)

def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

def standardize(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-12; return (Xtr - m) / s, (Xte - m) / s

def ridge(Xtr, ytr, Xte, lam):
    Xs, Xt = standardize(Xtr, Xte); ys = ytr.std(); ym = ytr.mean()
    A = np.c_[Xs, np.ones(len(Xs))]; b = np.linalg.solve(A.T @ A + lam * len(ytr) * np.eye(A.shape[1]), A.T @ ((ytr - ym) / ys))
    return (np.c_[Xt, np.ones(len(Xt))] @ b) * ys + ym

def mlp(Xtr, ytr, Xte, seed=0, epochs=400, lr=3e-3, wd=1e-4):
    rng = np.random.default_rng(seed); Xs, Xt = standardize(Xtr, Xte); ym, ys = ytr.mean(), ytr.std(); y = (ytr - ym) / ys
    # inner early-stopping split
    p = rng.permutation(len(y)); va, tr = p[: len(y) // 5], p[len(y) // 5:]
    d = Xs.shape[1]; Ws = [rng.standard_normal((d, 32)) / np.sqrt(d), np.zeros(32), rng.standard_normal((32, 32)) / np.sqrt(32),
                          np.zeros(32), rng.standard_normal((32, 1)) / np.sqrt(32), np.zeros(1)]
    M = [np.zeros_like(w) for w in Ws]; V = [np.zeros_like(w) for w in Ws]; best = (1e30, [w.copy() for w in Ws]); step = 0
    def fwd(X, Ws):
        h1 = np.tanh(X @ Ws[0] + Ws[1]); h2 = np.tanh(h1 @ Ws[2] + Ws[3]); return h1, h2, (h2 @ Ws[4] + Ws[5])[:, 0]
    for ep in range(epochs):
        for bidx in np.array_split(rng.permutation(tr), 16):
            X = Xs[bidx]; yy = y[bidx]; h1, h2, out = fwd(X, Ws); g = 2 * (out - yy)[:, None] / len(bidx)
            gW4 = h2.T @ g; gb4 = g.sum(0); gh2 = g @ Ws[4].T * (1 - h2 ** 2)
            gW2 = h1.T @ gh2; gb2 = gh2.sum(0); gh1 = gh2 @ Ws[2].T * (1 - h1 ** 2)
            gW0 = X.T @ gh1; gb0 = gh1.sum(0)
            grads = [gW0 + wd * Ws[0], gb0, gW2 + wd * Ws[2], gb2, gW4 + wd * Ws[4], gb4]; step += 1
            for i in range(6):
                M[i] = 0.9 * M[i] + 0.1 * grads[i]; V[i] = 0.999 * V[i] + 0.001 * grads[i] ** 2
                Ws[i] -= lr * (M[i] / (1 - 0.9 ** step)) / (np.sqrt(V[i] / (1 - 0.999 ** step)) + 1e-8)
        vl = np.mean((fwd(Xs[va], Ws)[2] - y[va]) ** 2)
        if vl < best[0]: best = (vl, [w.copy() for w in Ws])
    return fwd(Xt, best[1])[2] * ys + ym

res = {}
rng = np.random.default_rng(1)
for which in ("13", "31", "31+13"):
    out = {}
    # (i) neuron split, all late layers pooled (per-layer centring of the target: layer means are not features)
    rs_r, rs_m = [], []
    for rep in range(3):
        perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
        Xtr = np.concatenate([design(li, which)[A] for li in LAY]); ytr = np.concatenate([TGT[li][A] - TGT[li][A].mean() for li in LAY])
        Xte = np.concatenate([design(li, which)[B] for li in LAY]); yte = np.concatenate([TGT[li][B] - TGT[li][A].mean() for li in LAY])
        rs_r.append(r2(yte, ridge(Xtr, ytr, Xte, 1e-3))); rs_m.append(r2(yte, mlp(Xtr, ytr, Xte, seed=rep)))
    out["neuron_split"] = {"ridge_R2": float(np.mean(rs_r)), "mlp_R2": float(np.mean(rs_m)), "mlp_R2_reps": rs_m}
    # (ii) held-out layers
    for nm, (tr_l, te_l) in {"fit9-12_test13-16": ([8, 9, 10, 11], [12, 13, 14, 15]), "fit13-16_test9-12": ([12, 13, 14, 15], [8, 9, 10, 11])}.items():
        Xtr = np.concatenate([design(li, which) for li in tr_l]); ytr = np.concatenate([TGT[li] - TGT[li].mean() for li in tr_l])
        Xte = np.concatenate([design(li, which) for li in te_l]); yte = np.concatenate([TGT[li] - TGT[li].mean() for li in te_l])
        out[nm] = {"ridge_R2": r2(yte, ridge(Xtr, ytr, Xte, 1e-3)), "mlp_R2": r2(yte, mlp(Xtr, ytr, Xte, seed=7)),
                   "ridge_R2_in_sample": r2(ytr, ridge(Xtr, ytr, Xtr, 1e-3))}
    res[which] = out
    print(f"[{which}] neuron-split R2 ridge {out['neuron_split']['ridge_R2']:+.3f} mlp {out['neuron_split']['mlp_R2']:+.3f} | "
          + " | ".join(f"{k}: ridge {v['ridge_R2']:+.3f} (in {v['ridge_R2_in_sample']:+.3f}) mlp {v['mlp_R2']:+.3f}"
                       for k, v in out.items() if k.startswith("fit")) + f"  ({time.time()-t0:.0f}s)", flush=True)
best = max(max(v["neuron_split"]["mlp_R2"], v["neuron_split"]["ridge_R2"]) for v in res.values())
res["gate"] = {"best_heldout_R2": best, "verdict": "PASS" if best >= 0.10 else ("KILL" if best < 0.05 else "FAIL")}
print("GATE:", res["gate"])
json.dump(res, open("results/summaries/archA_drift_nbr.json", "w"), indent=1)
