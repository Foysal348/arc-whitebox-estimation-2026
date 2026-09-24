"""Round 11 reference (24 Sep): the chain's own per-channel error against the MC teacher, so a proposal's stage-1
channel check can be compared like for like.

Public MLP #0, teacher = 6e7-sample MC (data/kaggle_out/kaggle_out_k4pub/k4teach, pre-activation, every layer 1-16).
Diagonal channels (mean, variance, d3 = kappa3_iii, kappa4_iiii) from the free-running plain chain dump
data/dumps/errbudget_v29plain.npz (probe_errbudget.py; V31_ANGULAR=0, the chain comparable with a Gaussian teacher).
Off-diagonal channels (covariance C_ij, D21 = kappa3(z_i,z_i,z_j)) from the dense captures data/dumps/r7_cap_L{7,11,14,15}.npz.
Metric: relative Frobenius error ||chain - teacher|| / ||teacher||; for the mean, the error is also given in units of sigma.
Slope and correlation of chain on teacher are reported so a units mismatch would show.

Usage:  python experiments/probes/r11_chain_channel_ref.py
"""
import os, json, numpy as np
T = "data/kaggle_out/kaggle_out_k4pub/k4teach"
E = np.load("data/dumps/errbudget_v29plain.npz")
f64 = lambda x: np.asarray(x, np.float64)
rel = lambda c, t: float(np.linalg.norm(c - t) / np.linalg.norm(t))
slope = lambda c, t: float(np.dot(c, t) / np.dot(t, t))
corr = lambda c, t: float(np.corrcoef(c, t)[0, 1])
out = {"diag": {}, "offdiag": {}}
print("layer | mean err/sigma (rms) | var rel | d3 rel (slope, corr) | k4 rel (slope, corr)")
for L in (3, 7, 11, 14, 15, 16):
    cm, cv, c3, c4 = (f64(E[f"L{L}_c{k}"]) for k in ("mu", "var", "k3", "k4"))
    tm, tv, t3, t4 = (f64(E[f"L{L}_t{k}"]) for k in ("mu", "var", "k3", "k4"))
    r = {"mean_rms_sigma": float(np.sqrt(np.mean(((cm - tm) / np.sqrt(tv)) ** 2))), "var": rel(cv, tv),
         "d3": rel(c3, t3), "d3_slope": slope(c3, t3), "d3_corr": corr(c3, t3),
         "k4": rel(c4, t4), "k4_slope": slope(c4, t4), "k4_corr": corr(c4, t4)}
    out["diag"][L] = r
    print(f"L{L:2d} | {r['mean_rms_sigma']:.2e} | {r['var']:.4f} | {r['d3']:.4f} ({r['d3_slope']:.3f}, {r['d3_corr']:.4f}) | "
          f"{r['k4']:.4f} ({r['k4_slope']:.3f}, {r['k4_corr']:.4f})", flush=True)
print("layer | C offdiag rel (slope) | D21 offdiag rel (slope) | var (capture chain) rel")
for l in (7, 11, 14, 15):
    c = np.load(f"data/dumps/r7_cap_L{l}.npz"); li = int(c["li"]); t = np.load(os.path.join(T, f"layer{li + 1}_seed0.npz"))
    n = t["C"].shape[0]; off = ~np.eye(n, dtype=bool)
    Cc, Ct = f64(c["C_off"])[off], f64(t["C"])[off]
    Dc, Dt = f64(c["D21"])[off], f64(t["Y21"])[off]
    r = {"C_off": rel(Cc, Ct), "C_off_slope": slope(Cc, Ct), "D21_off": rel(Dc, Dt), "D21_slope": slope(Dc, Dt),
         "var_capture": rel(f64(c["var"]), np.diag(f64(t["C"])))}
    out["offdiag"][li + 1] = r
    print(f"L{li+1:2d} | {r['C_off']:.4f} ({r['C_off_slope']:.3f}) | {r['D21_off']:.4f} ({r['D21_slope']:.3f}) | {r['var_capture']:.4f}", flush=True)
json.dump(out, open("results/summaries/r11_chain_channel_ref.json", "w"), indent=1)
