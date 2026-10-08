"""V56 fit (8 Oct): chain-computed features incl. the HTQB cubic star (dumped by kaggle/k4s/data/k4s_dump.py on Kaggle) against the
6e7 teachers of MLP #0 (all layers) and the synthetic net (layers 2,4,..,14,15). Project metric (pattern k4/v^2, decisive
|mu/sigma| < 1.5). Compares: shipped map (V55 s3sq x1.25, 7 features), refit 7 features, refit 8 features (with star), star alone;
fits on mlp0 / synth / pooled (layers >= FIT_LMIN) and reports per-layer correlation on both nets.
Usage: python experiments/probes/k4s_fit.py data/kaggle_out/k4s/k4s_feats.npz
"""
import sys, os, json, numpy as np
F = np.load(sys.argv[1]); LMIN = int(os.environ.get("FIT_LMIN", "6"))
NETS = {"mlp0": "data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "data/kaggle_out/kaggle_out_k4/k4teach"}
BF55 = np.array([1.83011, 0.0210715, -0.00621352, -0.00169367, 0.000620213, 0.760054, 0.0183707])
data = {n: {} for n in NETS}
for k in F.files:
    if k.endswith("_pred"): continue
    net, L, full = k.split("_"); l = int(L[1:])
    if full != "1": continue
    p = os.path.join(NETS[net], f"layer{l}_seed0.npz")
    if not os.path.exists(p): continue
    T = np.load(p); v = np.diag(T["C"]).astype(np.float64); mu = T["mean"].astype(np.float64)
    data[net][l] = (F[k], T["K4d"].astype(np.float64) / v ** 2, np.abs(mu / np.sqrt(v)) < 1.5)
c = lambda a, b, m: float(np.corrcoef(a[m], b[m])[0, 1])
def X7(X): return np.delete(X, 6, axis=0)          # drop the star row
def fit(nets, feats):
    A = np.vstack([feats(X)[:, d].T for n in nets for l, (X, t, d) in data[n].items() if l >= LMIN])
    y = np.concatenate([t[d] for n in nets for l, (X, t, d) in data[n].items() if l >= LMIN])
    sc = np.linalg.norm(A, axis=0); sc[sc == 0] = 1
    return np.linalg.lstsq(A / sc, y, rcond=None)[0] / sc
res = {}
models = {"chain": None, "v55_shipped": None, "star_alone": None}
for fit_on in (["mlp0"], ["synth"], ["mlp0", "synth"]):
    models[f"refit7[{'+'.join(fit_on)}]"] = (X7, fit(fit_on, X7))
    models[f"refit8[{'+'.join(fit_on)}]"] = (lambda X: X, fit(fit_on, lambda X: X))
for name, m in models.items():
    res[name] = {}
    for net in data:
        for l, (X, t, d) in sorted(data[net].items()):
            if l < LMIN: continue
            if name == "chain": p = X[5]
            elif name == "star_alone": p = X[6]
            elif name == "v55_shipped": p = BF55 @ X7(X)
            else: p = m[1] @ m[0](X)
            res[name][f"{net}_L{l}"] = c(p, t, d)
deep = lambda name, net: np.mean([v for k, v in res[name].items() if k.startswith(net) and int(k.split("L")[1]) >= 8])
print(f"{'model':28s} {'mlp0 L8+':>9s} {'synth L8+':>9s}   per layer (mlp0 12/15 | synth 12/14/15)")
for name in res:
    r = res[name]
    print(f"{name:28s} {deep(name,'mlp0'):9.3f} {deep(name,'synth'):9.3f}   "
          f"{r.get('mlp0_L12',0):.3f}/{r.get('mlp0_L15',0):.3f} | {r.get('synth_L12',0):.3f}/{r.get('synth_L14',0):.3f}/{r.get('synth_L15',0):.3f}")
pooled = models["refit8[mlp0+synth]"][1]
print("refit8 pooled coefficients:", ":".join(f"{x:.6g}" for x in pooled))
res["coef_refit8_pooled"] = pooled.tolist()
res["coef_refit8_mlp0"] = models["refit8[mlp0]"][1].tolist(); res["coef_refit8_synth"] = models["refit8[synth]"][1].tolist()
json.dump(res, open("results/summaries/k4s_fit.json", "w"), indent=1)
