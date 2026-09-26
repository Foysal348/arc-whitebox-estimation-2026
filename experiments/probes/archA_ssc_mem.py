"""Phase A, candidate A1 (P-SSC: Team Puffi's published stack at 1024 x 16), falsifier F1a (26 Sep 2026).

Q23 (experiments/probes/ssc_k4_oracle.py) tested Puffi's SSC kappa4 closure as a stand-alone predictor of the next
kappa4 diagonal, fitted to the marginal kappa4 (Puffi's shipped target): decisive corr +0.05 over the chain. Three parts of
Puffi's stack were NOT in that test and are tested here:
  (a) the residual form: SSC added ON TOP of the chain's own regenerated kappa4 (our lambda C_off law is SSC's 1-mode case),
  (b) Puffi's own 'future work' target: Lambda fitted to the kappa4 term CONSUMED by the next mean closure (G4-weighted
      error, i.e. errG4) instead of the marginal kappa4 (Puffi reports -13.6% layer-32 MSE from this change alone),
  (c) the one-step signed response memory, A_mem = sqrt|beta| D_g (W A W^T) D_g of the previous layer's modes
      (post-ReLU teacher exists at layers 14 and 15, so (c) is testable for the L16 readout only).
Modes (post-ReLU teacher state at layers l = 3, 7, 11, 14, 15; readout at pre-activation l + 1, q_aj = w_j^T B_a w_j):
  B1 = C, B2 = Poff(C o C) / vbar (Puffi's Q restricted to its C o C part plus, as a variant, the full 6-feature Q with
  per-part free coefficients), B3 = vbar I, B4 = (g_i + g_j)/2 C_ij.  Features: 3 q_a q_b (all pairs).
Evaluation: residual target (teacher K4d - chain ck4) G4-weighted, (i) shared Lambda fitted on some layers, tested on
others; (ii) per-layer Lambda with a random half/half NEURON split (fit on one half, test on the other), repeated 20x.
Gate (pre-registered, same as A2): errG4 <= 0.85 x chain at >= 2 of L12/L15/L16 out of sample; kill if > 0.93 x at all.

Usage (repo root):  python experiments/probes/archA_ssc_mem.py        -> results/summaries/archA_ssc_mem.json
"""
import numpy as np, os, json, itertools, time
from math import erf
PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
f = lambda x: np.nan_to_num(np.asarray(x, np.float64))
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5))); pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
D = np.load("data/dumps/errbudget_v29plain.npz")
n = 1024; t0 = time.time()
Poff = lambda X: X - np.diag(np.diag(X))

def modes(l):
    P = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz")); T0 = np.load(os.path.join(PRE, f"layer{l}_seed0.npz"))
    C = f(P["C"]); C = 0.5 * (C + C.T); m = f(P["mean"]); K = f(P["Y21"]).copy(); t = np.diag(K).copy(); np.fill_diagonal(K, 0.0)
    v = np.diag(C).copy(); vbar = v.mean()
    mu0 = f(T0["mean"]); s0 = np.sqrt(np.diag(f(T0["C"]))); g = Phi(mu0 / s0); gs = (g - g.mean()) / g.std()
    B = {"C": C, "CC": Poff(C * C) / vbar, "vI": vbar * np.eye(n), "gC": 0.5 * (gs[:, None] + gs[None, :]) * C,
         # the remaining parts of Puffi's Q (C3), each with a free coefficient in the 'Qfull' model
         "Q_mK": Poff(0.5 * (m[:, None] * K.T + K * m[None, :])) / vbar, "Q_tm": Poff(np.outer(t, m) + np.outer(m, t)) / vbar,
         "Q_vv": Poff(np.outer(v, v)) / vbar, "Q_11": Poff(vbar ** 2 * np.ones((n, n))) / vbar}
    return B, g

data = {}; Bprev = None
for l in (3, 7, 11, 14, 15):
    B, g = modes(l)
    Q = W[l]
    q = {k: np.sum(Q * (M @ Q), 0) for k, M in B.items()}
    if l == 15:
        # one-step response memory from the layer-14 modes: A_mem = D_g (W A W^T) D_g, read at layer 16
        B14, _ = modes(14); W14 = W[14]
        for k in ("C", "CC", "gC"):
            Am = g[:, None] * (W14.T @ B14[k] @ W14) * g[None, :]
            q["mem_" + k] = np.sum(Q * (Am @ Q), 0)
    Tn = np.load(os.path.join(PRE, f"layer{l + 1}_seed0.npz")); mun = f(Tn["mean"]); vn = np.diag(f(Tn["C"])); sn = np.sqrt(vn); an = mun / sn
    data[l + 1] = {"q": q, "tk4": f(Tn["K4d"]), "ck4": np.asarray(D[f"L{l + 1}_ck4"], np.float64), "v": vn,
                   "G4": (an * an - 1) * pdf(an) / sn ** 3 / 24, "dec": np.abs(an) < 1.5}
    print(f"post {l} -> pre {l + 1} ready ({time.time()-t0:.0f}s)", flush=True)

def metrics(L, k4):
    d = data[L]; w = d["G4"]
    return {"errG4": float(np.linalg.norm(w * (k4 - d["tk4"])) / np.linalg.norm(w * d["tk4"])),
            "dec_corr": float(np.corrcoef((k4 / d["v"] ** 2)[d["dec"]], (d["tk4"] / d["v"] ** 2)[d["dec"]])[0, 1])}

def design(L, keys, pairs=True):
    q = data[L]["q"]
    cols = [3 * q[a] * q[b] for a, b in itertools.combinations_with_replacement(keys, 2)] if pairs else [q[a] for a in keys]
    return np.stack(cols + [data[L]["v"] ** 2], 1)

def fit(Ls, keys, idx=None, ridge=1e-6, pairs=True):
    A = np.concatenate([(data[L]["G4"][:, None] * design(L, keys, pairs))[idx if idx is not None else slice(None)] for L in Ls])
    y = np.concatenate([(data[L]["G4"] * (data[L]["tk4"] - data[L]["ck4"]))[idx if idx is not None else slice(None)] for L in Ls])
    sc = np.sqrt(np.mean(A * A, 0)) + 1e-300; As = A / sc
    cf = np.linalg.solve(As.T @ As + ridge * len(y) * np.eye(As.shape[1]), As.T @ y)
    return cf / sc

MODELS = {"ssc4": (["C", "CC", "vI", "gC"], True),
          "ssc4+Qfull": (["C", "CC", "vI", "gC", "Q_mK", "Q_tm", "Q_vv", "Q_11"], True),
          "ssc4_linear_q": (["C", "CC", "vI", "gC"], False)}
res = {"chain": {f"L{L}": metrics(L, data[L]["ck4"]) for L in data}, "shared": {}, "perlayer_split": {}}
print("chain errG4/dec:", {k: (round(v['errG4'], 3), round(v['dec_corr'], 2)) for k, v in res["chain"].items()})
SPL = {"fit8_12_test15_16": ([8, 12], [15, 16]), "fit15_16_test8_12": ([15, 16], [8, 12]), "fit4_8_test12_15_16": ([4, 8], [12, 15, 16])}
for nm, (keys, pairs) in MODELS.items():
    res["shared"][nm] = {}
    for sp, (fl, tl) in SPL.items():
        for ridge in (1e-6, 1e-3):
            cf = fit(fl, keys, ridge=ridge, pairs=pairs)
            row = {f"L{L}": metrics(L, data[L]["ck4"] + design(L, keys, pairs) @ cf) for L in tl}
            res["shared"][nm][f"{sp}_r{ridge:g}"] = row
            print(f"[shared {nm} {sp} ridge {ridge:g}] " + " ".join(f"{k} {v['errG4']:.3f}/{v['dec_corr']:.2f}" for k, v in row.items()), flush=True)
# per-layer tables, neuron split CV (Puffi's tables are per layer)
rng = np.random.default_rng(0)
for nm, (keys, pairs) in MODELS.items():
    res["perlayer_split"][nm] = {}
    for L in data:
        rat = []; cors = []
        for rep in range(20):
            perm = rng.permutation(n); a_, b_ = perm[: n // 2], perm[n // 2:]
            cf = fit([L], keys, idx=a_, ridge=1e-3, pairs=pairs)
            pred = data[L]["ck4"] + design(L, keys, pairs) @ cf
            w = data[L]["G4"]; tk = data[L]["tk4"]
            e_new = np.linalg.norm((w * (pred - tk))[b_]) / np.linalg.norm((w * tk)[b_])
            e_ch = np.linalg.norm((w * (data[L]["ck4"] - tk))[b_]) / np.linalg.norm((w * tk)[b_])
            rat.append(e_new / e_ch)
        res["perlayer_split"][nm][f"L{L}"] = {"errG4_ratio_mean": float(np.mean(rat)), "errG4_ratio_sd": float(np.std(rat))}
    print(f"[per-layer, neuron split] {nm}: " + " ".join(f"{k} x{v['errG4_ratio_mean']:.3f}" for k, v in res["perlayer_split"][nm].items()), flush=True)
# (c) response memory at L16 (only layer where the previous post-ReLU teacher exists): per-layer neuron-split CV
for nm, keys in (("ssc4+mem", ["C", "CC", "vI", "gC", "mem_C", "mem_CC", "mem_gC"]), ("ssc4", ["C", "CC", "vI", "gC"])):
    rat = []
    for rep in range(20):
        perm = rng.permutation(n); a_, b_ = perm[: n // 2], perm[n // 2:]
        cf = fit([16], keys, idx=a_, ridge=1e-3)
        pred = data[16]["ck4"] + design(16, keys) @ cf
        w = data[16]["G4"]; tk = data[16]["tk4"]
        rat.append((np.linalg.norm((w * (pred - tk))[b_]) / np.linalg.norm((w * tk)[b_])) /
                   (np.linalg.norm((w * (data[16]["ck4"] - tk))[b_]) / np.linalg.norm((w * tk)[b_])))
    res.setdefault("memory_L16", {})[nm] = {"errG4_ratio_mean": float(np.mean(rat)), "errG4_ratio_sd": float(np.std(rat))}
    print(f"[memory test L16, neuron split] {nm}: x{np.mean(rat):.3f} +- {np.std(rat):.3f}", flush=True)
json.dump(res, open("results/summaries/archA_ssc_mem.json", "w"), indent=1)
print(f"done ({time.time()-t0:.0f}s)")
