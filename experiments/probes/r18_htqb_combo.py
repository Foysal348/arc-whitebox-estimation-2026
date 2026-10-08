"""Round 18 follow-up (8 Oct): is the HTQB Gaussian-tree pattern (star + path) complementary to the shipped V55 corrector's
per-layer features? For every target layer 6..16 on public MLP #0: HTQB star_h6 / path_h6 / path_h3 (teacher mu, C), the chain's own
kappa4 (errbudget_v29plain), the V55 features (cached k4c_feats, full layers) -> project-metric correlation of
  (a) V55 map alone, (b) star / star+path alone, (c) linear combinations fitted on pooled layers (leave-one-layer-out) of
  [V55 features, star/v^2, path/v^2]. Targets standardised by the teacher v (pattern = k4/v^2), decisive |mu/sigma| < 1.5.
Usage: python experiments/probes/r18_htqb_combo.py
"""
import importlib.util, json, numpy as np
spec = importlib.util.spec_from_file_location("htqb", "experiments/probes/r18_htqb/round18_htqb_falsifier.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
d = np.load("kaggle/r17/data/r17_data.npz")
W = d["W"].astype(np.float32); mean = d["mean"].astype(np.float64); C = d["C"].astype(np.float64); K4 = d["K4d"].astype(np.float64)
H.TARGETS = tuple(range(6, 17))
pred = H.carrier_predictions(W, mean, C)
S = np.load("data/dumps/k4c_feats.npz")
BF = np.array([1.46409 * 1.25, 0.0210715, -0.00621352, -0.00169367, 0.000620213, 0.760054, 0.0183707])   # shipped (s3sq x1.25)
rows = {}
for t in H.TARGETS:
    key = f"mlp0_L{t}_1"
    if key not in S.files: continue
    v = np.diag(C[t]); dec = np.abs(mean[t] / np.sqrt(v)) < 1.5
    X = S[key]                                  # (7, n): s3sq, y21, y21T, c2, absa, chain, 1
    y = K4[t] / v ** 2
    # path at 3 ages (recompute from per-age split is not stored: approximate with ship - star)
    path3 = pred[t]["ship_h6star_h3path"] - pred[t]["star_h6"]
    rows[t] = dict(dec=dec, y=y, v55=BF @ X, chain=X[5], X=X, star=pred[t]["star_h6"] / v ** 2, path=pred[t]["path_h6"] / v ** 2, path3=path3 / v ** 2)
c = lambda a, b, m: float(np.corrcoef(a[m], b[m])[0, 1])
out = {}
for t, r in rows.items():
    out[t] = {k: c(r[k] if k != "star+path" else r["star"] + r["path"], r["y"], r["dec"]) for k in ("chain", "v55", "star", "star+path")}
# leave-one-layer-out linear fits (standardised design), on decisive neurons
def design(r, kind):
    cols = {"v55": [r["v55"]], "v55+star": [r["v55"], r["star"]], "v55+star+path": [r["v55"], r["star"], r["path"]],
            "feat+star+path": list(r["X"][:6]) + [r["star"], r["path"]], "v55+star+path3": [r["v55"], r["star"], r["path3"]]}[kind]
    return np.stack(cols + [np.ones_like(r["y"])], 1)
for kind in ("v55+star", "v55+star+path", "v55+star+path3", "feat+star+path"):
    for t in rows:
        A = np.vstack([design(rows[s], kind)[rows[s]["dec"]] for s in rows if s != t and s >= 8])
        b = np.concatenate([rows[s]["y"][rows[s]["dec"]] for s in rows if s != t and s >= 8])
        sc = np.linalg.norm(A, axis=0); sc[sc == 0] = 1
        beta = np.linalg.lstsq(A / sc, b, rcond=None)[0] / sc
        out[t][kind + " (LOLO)"] = c(design(rows[t], kind) @ beta, rows[t]["y"], rows[t]["dec"])
for t in out:
    print(f"L{t:2d} " + " | ".join(f"{k} {v:.3f}" for k, v in out[t].items()))
json.dump(out, open("results/summaries/r18_htqb_combo.json", "w"), indent=1)
