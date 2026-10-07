"""V55 kappa4-pattern corrector: dump the chain's own per-layer features (blend 0 = no effect), fit coefficients on
public MLP #0 against the teacher's standardised kappa4 diagonal (pooled layers >= LMIN), report pattern correlations on
MLP #0 (fit) and the synthetic net (held out).  Usage (from whest-starterkit): python -m uv run python ../experiments/probes/k4c_fit.py
"""
import os, sys, types, glob, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
os.environ.update({"V55_K4C": "1", "V55_K4C_BLEND": "0", "V55_K4C_DUMP": "1", "V55_K4C_LMIN": "2"})
import flopscope as flops, flopscope.numpy as fnp
from whestbench import MLP
SRC = "../experiments/variants/v55_k4c.py"
NETS = {"mlp0": "../data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "../data/kaggle_out/kaggle_out_k4/k4teach"}
class C: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
data = {}
for net, d in NETS.items():
    mod = types.ModuleType("m"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
    W = np.load(glob.glob(d + "/weights_seed*.npz")[0])["W"].astype(np.float32)
    mlp = MLP(width=1024, depth=16, weights=[fnp.array(W[l]) for l in range(16)], seed=0, name=net)
    e = mod.Estimator(); e.setup(C())
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
        e.predict(mlp, 2**41)
    rows = {}
    for layer, full, X in mod.K4C_DUMP:
        f = os.path.join(d, f"layer{layer}_seed0.npz")
        if not os.path.exists(f): continue
        T = np.load(f); v = np.diag(T["C"]).astype(np.float64); mu = T["mean"].astype(np.float64)
        rows[layer] = (full, np.asarray(X, np.float64), T["K4d"].astype(np.float64) / v ** 2, np.abs(mu / np.sqrt(v)) < 1.5)
    data[net] = rows
    print(net, "layers", sorted(rows), flush=True)
LMIN = int(os.environ.get("FIT_LMIN", "6"))
res = {}
for kind in ("full", "last"):
    Xs, ts = [], []
    for layer, (full, X, t, dec) in data["mlp0"].items():
        if layer < LMIN or full != (kind == "full"): continue
        Xs.append(X[:, dec].T); ts.append(t[dec])
    if not Xs: continue
    A = np.vstack(Xs); y = np.concatenate(ts)
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    res[kind] = beta.tolist()
    print(kind, "beta", ",".join(f"{b:.6g}" for b in beta))
    for net in data:
        out = []
        for layer, (full, X, t, dec) in sorted(data[net].items()):
            if layer < LMIN or full != (kind == "full"): continue
            p = beta @ X
            out.append(f"L{layer} corr {np.corrcoef(p[dec], t[dec])[0,1]:+.2f} (chain {np.corrcoef(X[-2 if kind=='full' else 2][dec], t[dec])[0,1]:+.2f})")
        print(" ", net, " | ".join(out))
json.dump(res, open("../results/summaries/k4c_fit.json", "w"), indent=1)
