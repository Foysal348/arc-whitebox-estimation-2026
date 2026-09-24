"""Opus R12, analysis 1: WHERE (which layer, which channel, which neurons) the final-layer MSE is made.

Linear-response map of the free-running plain chain (data/dumps/errbudget_v29plain.npz: chain and 6e7-sample teacher
mu / var / kappa3_iii / kappa4_iiii at every layer, baked 1e9 truth) on public MLP #0.

  fresh error of channel c at layer l :  f_c(l) = E(chain inputs) - E(chain inputs, channel c := teacher)
  propagation to layer 16            :  v <- dE/dmu(k) o (W[k-1]^T v),  k = l+1 .. 16   (d mu is exactly linear)
  value of fixing a set S of (c, l)   :  mean((e16 - sum_S v)^2) / mean(e16^2)

Checked against the measured channel oracles (meanfix_oracle.json: k4 0.699, var 0.533, k3 0.858).
Also reports the split by |mu/sigma| (decisive neurons) at the layer where the fix is applied.

Usage:  python experiments/probes/opus12_errmap.py
"""
import numpy as np, os, json
from math import erf
D = np.load("data/dumps/errbudget_v29plain.npz")
W = np.load("data/kaggle_out/kaggle_out_k4pub/k4teach/weights_seed0.npz")["W"].astype(np.float64)
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
L = 16
g = lambda k, l: D[f"L{l}_{k}"]
e = {l: g("cpred", l) - g("truth", l) for l in range(1, L + 1)}
dEdmu = {}
fresh = {}
for l in range(1, L + 1):
    m, v, k3, k4 = g("cmu", l), g("cvar", l), g("ck3", l), g("ck4", l)
    tv, tk3, tk4 = g("tvar", l), g("tk3", l), g("tk4", l)
    h = 1e-6 * np.sqrt(v)
    dEdmu[l] = (edge(m + h, v, k3, k4) - edge(m - h, v, k3, k4)) / (2 * h)
    E0 = edge(m, v, k3, k4)
    fresh[l] = {"var": E0 - edge(m, tv, k3, k4), "k3": E0 - edge(m, v, tk3, k4), "k4": E0 - edge(m, v, k3, tk4),
                "all3": E0 - edge(m, tv, tk3, tk4)}
    # the chain's prediction check
    fresh[l]["check"] = float(np.max(np.abs(E0 - g("cpred", l))))
def prop(vec, l):
    for k in range(l + 1, L + 1):
        vec = dEdmu[k] * (W[k - 1].T @ vec)
    return vec
V = {c: {l: prop(fresh[l][c], l) for l in range(2, L + 1)} for c in ("var", "k3", "k4", "all3")}
e16 = e[L]; base = float(np.mean(e16 ** 2))
out = {"base_mse": base, "closure_check_maxabs": max(fresh[l]["check"] for l in fresh)}
# consistency of the linear map: inherited + fresh total vs the actual error
tot_fresh = {l: e[l] - dEdmu[l] * (W[l - 1].T @ e[l - 1]) for l in range(2, L + 1)}
recon = prop(e[1], 1) + sum(prop(tot_fresh[l], l) for l in range(2, L + 1))
out["linear_recon_rel_err"] = float(np.linalg.norm(recon - e16) / np.linalg.norm(e16))
print(f"base final MSE {base:.4e}; closure check {out['closure_check_maxabs']:.2e}; linear reconstruction rel err {out['linear_recon_rel_err']:.3f}")
res = {}
for c in ("var", "k3", "k4", "all3"):
    full = float(np.mean((e16 - sum(V[c].values())) ** 2)) / base
    row = {"all_layers": full}
    for l0 in (4, 8, 10, 12, 13, 14, 15, 16):
        row[f"from_L{l0}"] = float(np.mean((e16 - sum(V[c][l] for l in range(l0, L + 1))) ** 2)) / base
    row["only_L16"] = float(np.mean((e16 - V[c][16]) ** 2)) / base
    row["only_L15_16"] = float(np.mean((e16 - V[c][16] - V[c][15]) ** 2)) / base
    # per-layer single-layer value
    row["single_layer"] = {l: float(np.mean((e16 - V[c][l]) ** 2)) / base for l in range(2, L + 1)}
    res[c] = row
    print(f"[{c:4s}] all layers {full:.3f} | from L4 {row['from_L4']:.3f} L8 {row['from_L8']:.3f} L12 {row['from_L12']:.3f} "
          f"L14 {row['from_L14']:.3f} L15 {row['from_L15']:.3f} | only L16 {row['only_L16']:.3f}")
    print("        single-layer ratios: " + " ".join(f"{l}:{row['single_layer'][l]:.3f}" for l in range(2, L + 1)))
out["ratios"] = res
# decisive neurons: fix the k4 channel only where |mu/sigma| < thr (at every layer)
dec = {}
for thr in (0.75, 1.0, 1.5, 2.0):
    for c in ("k4", "var", "k3"):
        acc = np.zeros(1024)
        for l in range(2, L + 1):
            a = np.abs(g("cmu", l) / np.sqrt(g("cvar", l)))
            acc += prop(np.where(a < thr, fresh[l][c], 0.0), l)
        dec[f"{c}_thr{thr}"] = float(np.mean((e16 - acc) ** 2)) / base
print("decisive-neuron restriction (fix only |mu/sigma| < thr):", {k: round(v, 3) for k, v in dec.items()})
out["decisive"] = dec
# kappa4 error structure on decisive neurons: relative error, G4-weighted
k4s = {}
for l in (4, 8, 12, 14, 15, 16):
    m, v, ck4, tk4 = g("cmu", l), g("cvar", l), g("ck4", l), g("tk4", l)
    wgt = dG(4, m, np.sqrt(v)) / 24
    errw = float(np.linalg.norm(wgt * (ck4 - tk4)) / np.linalg.norm(wgt * tk4))
    sl = float(np.sum(wgt ** 2 * ck4 * tk4) / np.sum(wgt ** 2 * tk4 ** 2))
    cw = float(np.corrcoef(wgt * ck4, wgt * tk4)[0, 1])
    k4s[l] = {"errG4": errw, "slopeG4": sl, "corrG4": cw, "exkurt_med_dec": float(np.median((tk4 / v ** 2)[np.abs(m / np.sqrt(v)) < 1.5]))}
    print(f"  L{l}: kappa4 G4-weighted err {errw:.3f} slope {sl:.3f} corr {cw:.3f}; median excess kurtosis (decisive) {k4s[l]['exkurt_med_dec']:.4f}")
out["k4_structure"] = k4s
# partial fixes: remove a fraction f of channel c's fresh error at layers >= l0 (linear in f)
part = {}
for c in ("k4", "var", "all3"):
    for l0 in (2, 8, 12):
        Vs = sum(V[c][l] for l in range(l0, L + 1))
        for fr in (0.25, 0.5):
            part[f"{c}_from_L{l0}_frac{fr}"] = float(np.mean((e16 - fr * Vs) ** 2)) / base
print("partial fixes:", {k: round(v, 3) for k, v in part.items()})
out["partial"] = part
json.dump(out, open("results/summaries/opus12_errmap.json", "w"), indent=1)
