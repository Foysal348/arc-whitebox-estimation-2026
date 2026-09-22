"""Pilot E (Gemini R1 H2): is the old-source D21 contribution at layer l+1 predictable from the previous layer's
contribution and odd covariance features?  Regress  H_old(l+1)  on  F1 = (B o B) H_old(l) B^T  (slice transport),
F2 = B H_old(l) B^T,  F3 = W C(l) (odd),  F4 = B H_old(l) (Gemini's literal one-sided term),  with B = W diag(w1).
Step 1: dump old legs, w2, w1_prev, W C at layers 10, 11, 13, 14 (one V29 run).  Step 2: least squares, report R^2.
Usage (from whest-starterkit): python -m uv run python ../experiments/probes/probe_gem.py dump   |   python ../experiments/probes/probe_gem.py fit
"""
import os, sys, types, json
import numpy as np
mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
LAYERS = [10, 11, 13, 14]

if mode == "dump":
    sys.path.insert(0, os.path.abspath("."))
    from local_engine import build_mlp
    import flopscope as flops
    SRC = "../submissions/v29m8/estimator.py"
    src = open(SRC, encoding="utf-8").read()
    anchor = ('                legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]\n'
              '                legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]\n'
              '            else:\n')
    assert src.count(anchor) == 1
    src = src.replace(anchor, '                _DUMP(li, legs4["AP1"], ka, kb, k, Qc, w2b_list, w1_prev)\n' + anchor)
    def _DUMP(li, legs4, ka, kb, k, Qc, w2b_list, w1_prev):
        if li not in LAYERS or ka == 0: return
        arr = np.asarray(legs4)
        out = {"ka": ka, "kb": kb, "k": k, "w1_prev": np.asarray(w1_prev, dtype=np.float64), "WC": arr[2 * k + 1, 0].astype(np.float64)}
        for s in range(ka):
            out[f"A{s}"] = arr[2 * s, 0].astype(np.float32); out[f"P{s}"] = arr[2 * s + 1, 0].astype(np.float32)
            out[f"w2_{s}"] = np.asarray(w2b_list[s], dtype=np.float64)
        np.savez_compressed(f"../data/dumps/gem_dump_L{li}.npz", **out); print(f"dumped layer {li}: ka={ka} k={k}", flush=True)
    mod = types.ModuleType("est_gem"); mod.__dict__["_DUMP"] = _DUMP
    exec(compile(src, SRC, "exec"), mod.__dict__)
    class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
    est = mod.Estimator(); est.setup(_Ctx())
    mlp = build_mlp(width=1024, depth=16, seed=7)
    np.savez_compressed("../data/dumps/gem_weights.npz", **{f"W{i}": np.asarray(w, dtype=np.float64) for i, w in enumerate(mlp.weights)})
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        est.predict(mlp, 2**41)
    print("done")
else:
    Wall = np.load("../data/dumps/gem_weights.npz")
    def H_old(dl, ka):
        return sum(((dl[f"A{s}"].astype(np.float64) * dl[f"P{s}"].astype(np.float64)) * dl[f"w2_{s}"]) @ dl[f"A{s}"].astype(np.float64).T for s in range(ka))
    for l0, l1 in ((10, 11), (13, 14)):
        d0 = np.load(f"../data/dumps/gem_dump_L{l0}.npz"); d1 = np.load(f"../data/dumps/gem_dump_L{l1}.npz")
        ka = int(d0["ka"])                                  # sources old at both layers (birth-ordered slots)
        H0 = H_old(d0, ka); H1 = H_old(d1, ka)
        W = Wall[f"W{l1}"]                                  # weights of layer l1 (0-based, as enumerate(mlp.weights))
        WT = W.T                                            # the chain transports with W^T @ x convention: z = W^T h  ->  B = W^T diag(w1)
        w1 = d1["w1_prev"]
        for name, B in (("B = W^T diag(w1)", WT * w1[None, :]), ("B = W diag(w1)", W * w1[None, :])):
            F1 = (B * B) @ H0 @ B.T; F2 = B @ H0 @ B.T; F3 = d1["WC"]; F4 = B @ H0
            feats = {"F1 (BoB)H B^T": F1, "F2 B H B^T": F2, "F3 W C": F3, "F4 B H": F4}
            y = H1.ravel(); ny = np.linalg.norm(y)
            def r2(cols):
                X = np.stack([feats[c].ravel() for c in cols], axis=1)
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                return 1 - np.linalg.norm(y - X @ coef) ** 2 / ny ** 2, coef
            print(f"layers {l0}->{l1} ({ka} old sources), {name}:")
            for cols in (["F1 (BoB)H B^T"], ["F2 B H B^T"], ["F3 W C"], ["F4 B H"], ["F1 (BoB)H B^T", "F3 W C"], ["F1 (BoB)H B^T", "F2 B H B^T", "F3 W C", "F4 B H"]):
                R, coef = r2(cols); print(f"   R^2 = {R:8.4f}   {cols}  coef {np.round(coef, 4)}", flush=True)
