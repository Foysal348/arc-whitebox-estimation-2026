"""V55 refit study on cached chain features (dumped once by this script to data/dumps/k4c_feats.npz).
Fits the kappa4-pattern map on FIT_ON = mlp0 | synth | both (pooled layers >= FIT_LMIN, decisive neurons), with optional
extra features (EXT=1: squares and pairwise products of s3sq / chain / absa, log-chain); reports pattern corr per layer on both
nets and the chain's own corr.  Usage (from whest-starterkit): python -m uv run python ../experiments/probes/k4c_refit.py
"""
import os, sys, types, glob, json
import numpy as np
CACHE = "../data/dumps/k4c_feats.npz"
NETS = {"mlp0": "../data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "../data/kaggle_out/kaggle_out_k4/k4teach"}
if not os.path.exists(CACHE):
    sys.path.insert(0, os.path.abspath("."))
    os.environ.update({"V55_K4C": "1", "V55_K4C_BLEND": "0", "V55_K4C_DUMP": "1", "V55_K4C_LMIN": "2"})
    import flopscope as flops, flopscope.numpy as fnp
    from whestbench import MLP
    SRC = "../experiments/variants/v55_k4c.py"
    class C: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
    store = {}
    for net, d in NETS.items():
        mod = types.ModuleType("m"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
        W = np.load(glob.glob(d + "/weights_seed*.npz")[0])["W"].astype(np.float32)
        mlp = MLP(width=1024, depth=16, weights=[fnp.array(W[l]) for l in range(16)], seed=0, name=net)
        e = mod.Estimator(); e.setup(C())
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
            e.predict(mlp, 2**41)
        for layer, full, X in mod.K4C_DUMP:
            store[f"{net}_L{layer}_{int(full)}"] = np.asarray(X, np.float64)
        print("dumped", net, flush=True)
    np.savez(CACHE, **store)
S = np.load(CACHE)
data = {net: {} for net in NETS}
for k in S.files:
    net, L, full = k.split("_"); layer = int(L[1:])
    f = os.path.join(NETS[net], f"layer{layer}_seed0.npz")
    if not os.path.exists(f): continue
    T = np.load(f); v = np.diag(T["C"]).astype(np.float64); mu = T["mean"].astype(np.float64)
    data[net][layer] = (full == "1", S[k], T["K4d"].astype(np.float64) / v ** 2, np.abs(mu / np.sqrt(v)) < 1.5)
EXT = os.environ.get("EXT", "0") == "1"
def feats(X, full):
    if not EXT: return X
    s3, ab, ch = (X[0], X[4], X[5]) if full else (X[0], X[1], X[2])
    ex = [s3 * s3, ch * ch, s3 * ch, ab * ch, ab * s3, ab * ab, np.log(np.maximum(ch, 1e-6))]
    if full: ex += [X[1] * ch, X[2] * ch, X[3] * ch]
    return np.vstack([X] + ex)
LMIN = int(os.environ.get("FIT_LMIN", "6")); ON = os.environ.get("FIT_ON", "mlp0").split("+")
res = {}
for kind in ("full", "last"):
    A, y = [], []
    for net in ON:
        for layer, (full, X, t, dec) in data[net].items():
            if layer < LMIN or full != (kind == "full"): continue
            F = feats(X, full); A.append(F[:, dec].T); y.append(t[dec])
    if not A: continue
    beta = np.linalg.lstsq(np.vstack(A), np.concatenate(y), rcond=None)[0]
    res[kind] = beta.tolist()
    print(kind, "fit on", ON, "beta", ":".join(f"{b:.6g}" for b in beta))
    for net in data:
        out = []
        for layer, (full, X, t, dec) in sorted(data[net].items()):
            if layer < LMIN or full != (kind == "full"): continue
            p = beta @ feats(X, full)
            out.append(f"L{layer} {np.corrcoef(p[dec], t[dec])[0,1]:+.2f}/{np.corrcoef(X[5 if full else 2][dec], t[dec])[0,1]:+.2f}")
        print(" ", net, " ".join(out), flush=True)
json.dump(res, open(f"../results/summaries/k4c_refit_{'+'.join(ON)}_ext{int(EXT)}.json", "w"), indent=1)
