"""Phase A, candidate A2 (CPR: carried pair-response kappa4 mode), teacher-forced falsifier (26 Sep 2026).

Design being falsified: add to the chain ONE OR TWO carried symmetric n x n "response" matrices H (pre-activation index
space), updated every layer by the exact linear carry H <- U^T H U (U = diag(Phi_l) W[l], one block-symmetric sandwich)
plus a cheap birth, and read the kappa4 diagonal as the pair form  kappa4_jjjj ~ ck4_j + 3 sum_ab Lam_ab H_a,jj H_b,jj
(modes H_a in {carried H, C, I}).  Unlike Q23/SSC (modes regenerated from the live state, one-step memory) and D27
(one dense core born at layer 1, no later births) the modes here are CARRIED with a birth at every layer, so the pair
form has an all-distinct (1,1,1,1) sector with memory.

Generous oracle: births and transports use the TEACHER's per-layer pre-activation mean/covariance/kappa4 slices of public
MLP #0 (6e7-sample teacher, every layer 1..16), several birth types and decays; the blend coefficients are fitted
G4-weighted on the residual (teacher K4d - chain ck4) on some layers and TESTED on others (out-of-layer), exactly the
convention of opus12_k4_young_blend.py (chain = plain V29 dump errbudget_v29plain.npz).

Reference points (results/summaries/opus12_k4_carry.json): errG4 chain 0.142/0.193/0.199/0.210 (L8/12/15/16), exact
linear carry of the TRUE kappa4 (affine, in-sample) 0.129/0.126/0.128/0.136; decisive corr chain 0.37/0.34/0.26/0.32.
Gate (pre-registered): out-of-layer errG4 <= 0.85 x chain at >= 2 of L12/L15/L16 AND decisive corr >= 0.45 there.
Kill: best candidate out-of-layer errG4 > 0.93 x chain at all of L12/L15/L16.

Usage (repo root):  python experiments/probes/archA_cpr_oracle.py
Writes results/summaries/archA_cpr_oracle.json
"""
import numpy as np, os, json, time, itertools
from math import erf

PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)   # z_{l+1} = h_l @ W[l]
D = np.load("data/dumps/errbudget_v29plain.npz")
n = 1024
pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / 2 ** 0.5)))
t0 = time.time()

# ---- teacher per pre-activation layer l = 1..16 ----
T = {}
for l in range(1, 17):
    t = np.load(os.path.join(PRE, f"layer{l}_seed0.npz"))
    C = np.asarray(t["C"], np.float64); C = 0.5 * (C + C.T)
    mu = np.asarray(t["mean"], np.float64); v = np.diag(C).copy(); s = np.sqrt(v); a = mu / s
    K22 = np.asarray(t["K22"], np.float64); K22 = 0.5 * (K22 + K22.T)
    K31 = np.asarray(t["K31"], np.float64)
    T[l] = dict(C=C, mu=mu, v=v, a=a, w1=Phi(a), w2=pdf(a) / s, K22=K22, K31s=0.5 * (K31 + K31.T),
                tk4=np.asarray(t["K4d"], np.float64), G4=(a * a - 1) * pdf(a) / s ** 3 / 24, dec=np.abs(a) < 1.5)
    if l >= 2:
        T[l]["ck4"] = np.asarray(D[f"L{l}_ck4"], np.float64)
print(f"teacher loaded ({time.time()-t0:.0f}s)", flush=True)

# ---- births (pre-activation space of layer l, or post-ReLU of layer l then carried by W[l]) ----
def births(l):
    d = T[l]; C = d["C"]
    Coff = C - np.diag(d["v"])
    return {
        "C": C,                                                   # covariance history
        "CC": C * C,                                              # Sym(C x C)-type birth
        "K22": d["K22"],                                          # teacher (2,2) kappa4 slice
        "K31": d["K31s"],                                         # teacher (3,1) slice, symmetrised
        "Dw2": 0.5 * (d["w2"][:, None] * (Coff * Coff) * d["w2"][None, :]),   # post-ReLU Gaussian born covariance
    }
BIRTH_POST = {"Dw2"}          # born after the ReLU of layer l: carried to l+1 by W[l] only
DECAYS = (1.0, 0.7, 0.4)

H = {}      # (birth, decay) -> current carried matrix at pre-activation layer l
DIAG = {}   # (birth, decay) -> {l: diag(H(l))}
for l in range(1, 17):
    b = births(l)
    for key in [(bk, g) for bk in b for g in DECAYS]:
        bk, g = key
        if l == 1:
            Hn = np.zeros((n, n)) if bk in BIRTH_POST else b[bk].copy()
        else:
            U = T[l - 1]["w1"][:, None] * W[l - 1]               # pre(l-1) -> pre(l), linear part of the ReLU
            Hp = H[key]
            Hn = g * (U.T @ Hp @ U)
            if bk in BIRTH_POST:
                Bp = births(l - 1)[bk]
                Hn += W[l - 1].T @ Bp @ W[l - 1]
            else:
                Hn += b[bk]
        H[key] = 0.5 * (Hn + Hn.T)
        DIAG.setdefault(key, {})[l] = np.diag(H[key]).copy()
    print(f"  layer {l} carried ({time.time()-t0:.0f}s)", flush=True)

# ---- features at pre-activation layer L (per neuron j) ----
def feats(L):
    d = T[L]; v = d["v"]; F = {}
    for key, dd in DIAG.items():
        h = dd[L]; nm = f"{key[0]}_g{key[1]}"
        F[f"HxC[{nm}]"] = 3 * h * v                   # Sym(H x C)
        F[f"HxH[{nm}]"] = 3 * h * h                   # Sym(H x H)
        F[f"HxI[{nm}]"] = 3 * h                       # Sym(H x I)  (metric replaced)
        F[f"ck4*H/mean[{nm}]"] = d["ck4"] * h / np.mean(np.abs(h))   # chain core with a carried metric
    return F

def metrics(L, k4):
    d = T[L]; w, tk4, v, dec = d["G4"], d["tk4"], d["v"], d["dec"]
    return {"errG4": float(np.linalg.norm(w * (k4 - tk4)) / np.linalg.norm(w * tk4)),
            "dec_corr": float(np.corrcoef((k4 / v ** 2)[dec], (tk4 / v ** 2)[dec])[0, 1])}

LAYERS = list(range(4, 17))
FEATS = {L: feats(L) for L in LAYERS}
names = list(FEATS[LAYERS[0]].keys())

def fit(layers, cols):
    A = np.concatenate([np.stack([T[L]["G4"] * FEATS[L][c] if c != "v2" else T[L]["G4"] * T[L]["v"] ** 2 for c in cols], 1)
                        for L in layers])
    y = np.concatenate([T[L]["G4"] * (T[L]["tk4"] - T[L]["ck4"]) for L in layers])
    sc = np.sqrt(np.mean(A * A, 0)) + 1e-300
    return np.linalg.lstsq(A / sc, y, rcond=None)[0] / sc

def predict(L, cols, cf):
    return T[L]["ck4"] + sum(c * (FEATS[L][k] if k != "v2" else T[L]["v"] ** 2) for k, c in zip(cols, cf))

SPLITS = {"fit8_12_test15_16": ([8, 12], [15, 16]), "fit15_16_test8_12": ([15, 16], [8, 12]),
          "fit_even_test_odd": ([4, 6, 8, 10, 12, 14, 16], [5, 7, 9, 11, 13, 15])}
res = {"chain": {f"L{L}": metrics(L, T[L]["ck4"]) for L in LAYERS}, "single": {}, "pairs": {}}
print("chain:", " ".join(f"L{L} {res['chain'][f'L{L}']['errG4']:.3f}/{res['chain'][f'L{L}']['dec_corr']:.2f}" for L in (8, 12, 15, 16)))

# single carried feature + level term var^2 (as in the young-blend convention)
score = []
for nm in names:
    cols = [nm, "v2"]; row = {}
    for sp, (fl, tl) in SPLITS.items():
        cf = fit(fl, cols)
        row[sp] = {f"L{L}": metrics(L, predict(L, cols, cf)) for L in tl}
        row[sp]["coef"] = cf.tolist()
    res["single"][nm] = row
    # selection statistic: mean out-of-layer errG4 ratio on the 8/12 -> 15/16 and 15/16 -> 8/12 splits
    rat = [row[sp][f"L{L}"]["errG4"] / res["chain"][f"L{L}"]["errG4"] for sp in ("fit8_12_test15_16", "fit15_16_test8_12")
           for L in SPLITS[sp][1]]
    score.append((float(np.mean(rat)), nm))
score.sort()
print("best single carried features (mean out-of-layer errG4 ratio to chain on the two cross splits):")
for s, nm in score[:12]:
    r = res["single"][nm]
    print(f"  {s:.3f}  {nm:28s} | 8,12->15,16: " + " ".join(f"L{L} {r['fit8_12_test15_16'][f'L{L}']['errG4']:.3f}/{r['fit8_12_test15_16'][f'L{L}']['dec_corr']:.2f}" for L in (15, 16))
          + " | 15,16->8,12: " + " ".join(f"L{L} {r['fit15_16_test8_12'][f'L{L}']['errG4']:.3f}/{r['fit15_16_test8_12'][f'L{L}']['dec_corr']:.2f}" for L in (8, 12)), flush=True)
res["single_ranking"] = score[:40]

# pairs among the 10 best singles (2 carried modes + level)
top = [nm for _, nm in score[:10]]
pscore = []
for a_, b_ in itertools.combinations(top, 2):
    cols = [a_, b_, "v2"]; row = {}
    for sp, (fl, tl) in SPLITS.items():
        cf = fit(fl, cols)
        row[sp] = {f"L{L}": metrics(L, predict(L, cols, cf)) for L in tl}
    rat = [row[sp][f"L{L}"]["errG4"] / res["chain"][f"L{L}"]["errG4"] for sp in ("fit8_12_test15_16", "fit15_16_test8_12")
           for L in SPLITS[sp][1]]
    res["pairs"][f"{a_} + {b_}"] = row
    pscore.append((float(np.mean(rat)), f"{a_} + {b_}"))
pscore.sort()
print("best pairs:")
for s, nm in pscore[:6]:
    r = res["pairs"][nm]
    print(f"  {s:.3f}  {nm} | " + " ".join(f"L{L} {r['fit8_12_test15_16'][f'L{L}']['errG4']:.3f}/{r['fit8_12_test15_16'][f'L{L}']['dec_corr']:.2f}" for L in (15, 16))
          + " | " + " ".join(f"L{L} {r['fit15_16_test8_12'][f'L{L}']['errG4']:.3f}/{r['fit15_16_test8_12'][f'L{L}']['dec_corr']:.2f}" for L in (8, 12)), flush=True)
res["pair_ranking"] = pscore[:20]

# gate evaluation on the best single and best pair (out-of-layer, the 8/12 -> 15/16 split + even->odd for L15)
def gate(row):
    r1516 = row["fit8_12_test15_16"]; r12 = row["fit15_16_test8_12"]["L12"]
    tests = {"L12": r12, "L15": r1516["L15"], "L16": r1516["L16"]}
    passL = [L for L, m in tests.items() if m["errG4"] <= 0.85 * res["chain"][L]["errG4"] and m["dec_corr"] >= 0.45]
    killL = [L for L, m in tests.items() if m["errG4"] > 0.93 * res["chain"][L]["errG4"]]
    return {"tests": tests, "pass_layers": passL, "verdict": "PASS" if len(passL) >= 2 else ("KILL" if len(killL) == 3 else "FAIL")}
res["gate_best_single"] = {"name": score[0][1], **gate(res["single"][score[0][1]])}
res["gate_best_pair"] = {"name": pscore[0][1], **gate(res["pairs"][pscore[0][1]])}
print("GATE single:", res["gate_best_single"]["name"], res["gate_best_single"]["verdict"], res["gate_best_single"]["pass_layers"])
print("GATE pair:", res["gate_best_pair"]["name"], res["gate_best_pair"]["verdict"], res["gate_best_pair"]["pass_layers"])
json.dump(res, open("results/summaries/archA_cpr_oracle.json", "w"), indent=1)
print(f"done ({time.time()-t0:.0f}s)")
