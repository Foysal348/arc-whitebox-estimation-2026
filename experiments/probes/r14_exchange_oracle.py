"""Round 14 "Exchange Vertex Closure" falsifier (27-28 Sep), teacher-input component test, no fitting.
Exchange diagonal of the connected 4-point function: kappa4_iiii ~ 3 sum_ab Y_ia (C^-1)_ab Y_ib, Y_ia = kappa3(z_i, z_i, z_a)
(stored (2,1) slice with the kappa3 diagonal). Standardised and regularised as proposed:
  R = C / (s s^T), Z_ia = Y_ia / (v_i s_a), q_i = 3 Z_i (R + tau I)^-1 Z_i^T, tau = 1e-3.
Gates (as proposed): centred pattern correlation >= 0.7 with the teacher's standardised kappa4 diagonal on the decisive
neurons at layers 12-16; and (MLP #0, where the chain dump exists) the G4-weighted error of the level-preserving
prediction must be <= 0.5 x the chain's (errbudget_v29plain.npz L*_ck4).
Usage:  python experiments/probes/r14_exchange_oracle.py
"""
import numpy as np, os, json
from math import erf, sqrt, pi, exp

NETS = {"mlp0": "data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "data/kaggle_out/kaggle_out_k4/k4teach"}
D = np.load("data/dumps/errbudget_v29plain.npz")
TAU = float(os.environ.get("TAU", "1e-3"))
out = {}
for net, d in NETS.items():
    out[net] = {}
    for l in range(4, 17):
        f = os.path.join(d, f"layer{l}_seed0.npz")
        if not os.path.exists(f):
            continue
        T = np.load(f)
        C = np.nan_to_num(T["C"].astype(np.float64)); Y = np.nan_to_num(T["Y21"].astype(np.float64))
        mu = T["mean"].astype(np.float64); k4 = np.nan_to_num(T["K4d"].astype(np.float64))
        v = np.diag(C).copy(); s = np.sqrt(v)
        R = C / s[:, None] / s[None, :]
        Z = Y / v[:, None] / s[None, :]
        X = np.linalg.solve(R + TAU * np.eye(len(v)), Z.T)
        q = 3.0 * np.einsum("ia,ai->i", Z, X)                    # standardised exchange prediction
        t = k4 / v ** 2                                           # teacher standardised kappa4
        a = mu / s
        dec = np.abs(a) < 1.5
        c = lambda x, y: float(np.corrcoef(x[dec], y[dec])[0, 1])
        row = {"corr_exchange": c(q, t), "mean_q": float(q[dec].mean()), "mean_t": float(t[dec].mean())}
        ck = f"L{l}_ck4"
        if net == "mlp0" and ck in D.files:
            ch = np.nan_to_num(D[ck].astype(np.float64)) / v ** 2
            row["corr_chain"] = c(ch, t)
            # level-preserving prediction: q + mean offset to the chain's level (as proposed), G4-weighted error vs chain
            khat = q + (ch[dec] - q[dec]).mean()
            phi = np.exp(-a * a / 2) / sqrt(2 * pi)
            g4 = np.abs((a * a - 1) * phi) / 24.0 * v ** 2 / s ** 3   # weight per standardised unit
            e_hat = np.sum((g4 * (khat - t))[dec] ** 2); e_ch = np.sum((g4 * (ch - t))[dec] ** 2)
            row["err_ratio_vs_chain"] = float(e_hat / e_ch)
        out[net][f"L{l}"] = row
        print(f"{net} L{l:2d}: corr(exchange, teacher) {row['corr_exchange']:+.3f}"
              + (f" | chain corr {row['corr_chain']:+.3f} | G4-weighted err vs chain x{row['err_ratio_vs_chain']:.2f}" if "corr_chain" in row else "")
              + f" | mean level q {row['mean_q']:+.3f} vs teacher {row['mean_t']:+.3f}", flush=True)
json.dump(out, open(f"results/summaries/r14_exchange_oracle_tau{TAU:g}.json", "w"), indent=1)
