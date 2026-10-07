"""Which cheap per-neuron quantities track the PATTERN of the deep kappa4 diagonal? (7 Oct)
Teacher pre-activation arrays per layer (mean, C, Y21 = kappa3(z_i, z_i, z_j) incl. diagonal, K22, K4d). Target: standardised
kappa4 diagonal t = K4d / v^2 on the decisive neurons (|mu/sigma| < 1.5). Features (all from the SAME layer's teacher
quantities, i.e. the best case for a readout built on what the chain carries):
  s3sq   (kappa3_iii / v^1.5)^2
  y21    sum_j Y21_ij^2 / (v_i^2 v_j)          row energy of the standardised (2,1) slice
  y21T   sum_j Y21_ji^2 / (v_j^2 v_i)          column energy
  c2     sum_j C_ij^2 / (v_i v_j)              correlation-row energy
  k22    sum_j K22_ij / (v_i v_j)              (2,2)-slice row sum
  absa   |mu / sigma|
plus the chain's own kappa4 diagonal (errbudget_v29plain.npz, MLP #0) and an in-layer least-squares combination of all
features (in-sample ceiling) and its out-of-layer transfer (fit on the other layers).
Usage:  python experiments/probes/k4_pattern_features.py
"""
import numpy as np, os, json

NETS = {"mlp0": "data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "data/kaggle_out/kaggle_out_k4/k4teach"}
D = np.load("data/dumps/errbudget_v29plain.npz")
out = {}
for net, d in NETS.items():
    feats = {}
    for l in range(6, 17):
        f = os.path.join(d, f"layer{l}_seed0.npz")
        if not os.path.exists(f):
            continue
        T = np.load(f)
        C = np.nan_to_num(T["C"].astype(np.float64)); Y = np.nan_to_num(T["Y21"].astype(np.float64))
        K22 = np.nan_to_num(T["K22"].astype(np.float64)); k4 = np.nan_to_num(T["K4d"].astype(np.float64))
        mu = T["mean"].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v)
        dec = np.abs(mu / s) < 1.5
        F = {"s3sq": (np.diag(Y) / v ** 1.5) ** 2,
             "y21": ((Y / v[:, None] / s[None, :]) ** 2).sum(1),
             "y21T": ((Y / v[:, None] / s[None, :]) ** 2).sum(0),
             "c2": ((C / s[:, None] / s[None, :]) ** 2).sum(1),
             "k22": (np.where(np.eye(len(v), dtype=bool), 0.0, K22) / s[:, None] ** 2 / s[None, :] ** 2).sum(1),
             "absa": np.abs(mu / s)}
        if net == "mlp0" and f"L{l}_ck4" in D.files:
            F["chain"] = np.nan_to_num(D[f"L{l}_ck4"].astype(np.float64)) / v ** 2
        t = k4 / v ** 2
        feats[l] = (F, t, dec)
    out[net] = {}
    layers = sorted(feats)
    keys = [k for k in feats[layers[0]][0] if k not in os.environ.get("DROP", "chain").split(",")]
    for l in layers:
        F, t, dec = feats[l]
        row = {k: float(np.corrcoef(F[k][dec], t[dec])[0, 1]) for k in F}
        X = np.c_[[F[k][dec] for k in keys]].T; X = np.c_[X, np.ones(dec.sum())]
        beta = np.linalg.lstsq(X, t[dec], rcond=None)[0]
        row["lsq_in"] = float(np.corrcoef(X @ beta, t[dec])[0, 1])
        # out-of-layer: fit on all other layers (standardised features), apply here
        Xo, to = [], []
        for m in layers:
            if m == l: continue
            Fm, tm, dm = feats[m]
            Xo.append(np.c_[[Fm[k][dm] for k in keys]].T); to.append(tm[dm])
        Xo = np.vstack(Xo); Xo = np.c_[Xo, np.ones(len(Xo))]; bo = np.linalg.lstsq(Xo, np.concatenate(to), rcond=None)[0]
        row["lsq_out"] = float(np.corrcoef(X @ bo, t[dec])[0, 1])
        out[net][f"L{l}"] = row
        print(f"{net} L{l:2d}: " + " ".join(f"{k} {v:+.2f}" for k, v in row.items()), flush=True)
json.dump(out, open("results/summaries/k4_pattern_features_" + os.environ.get("DROP", "chain").replace(",", "_") + ".json", "w"), indent=1)
