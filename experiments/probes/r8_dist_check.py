"""Round 8 (Fable, 24 Sep): consistency check of the LIVE all-distinct 3+2 term (r8_k4_dist_e2e.py with R8_DUMP=1, shipped
chain configuration) against the OFFLINE term of r8_k4_1111.py (round-7 captures, every source dense) at pre-activation
layers 8 and 12, plus the G4-weighted kappa4 error of the live combination against the teacher.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r8_dist_check.py ../data/dumps/r8_dist_e2e_mlp0_<cfg>.npz
"""
import os, sys
import numpy as np
CAP = "../data/dumps"; PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"
live = np.load(sys.argv[1])
for Lt in (8, 12):
    f = os.path.join(CAP, f"r8_k4_1111_L{Lt}.npz")
    if not os.path.exists(f): continue
    off = np.load(f); t4 = off["t4"]; w = off["w"]; a1 = off["a1"]; m = np.abs(a1) < 1.5
    def st(x, y): return f"corr {np.corrcoef(x, y)[0, 1]:.3f} slope {np.dot(x, y) / np.dot(x, x):.3f} rms {np.sqrt(np.mean(x ** 2)):.2e} vs {np.sqrt(np.mean(y ** 2)):.2e}"
    ld, lp, lr, lpc = live[f"L{Lt}_dist"], live[f"L{Lt}_pred"], live[f"L{Lt}_K3Crep"], live[f"L{Lt}_Pchain"]
    print(f"L{Lt}: live dist (ages 1-4) vs offline dist_y4: {st(ld, off['K3C_dist_y4'])} | live pred vs offline pred: {st(lp, off['pred'])} | live Pchain vs offline: {st(lpc, off['Pchain'])}")
    if np.any(lr): print(f"     live K3C_rep vs offline: {st(lr, off['K3C_rep'])}")
    g4c = live.get(f"L{Lt}_g4chain")
    def ev(v): e = v - t4; return f"errG4 {np.sqrt(np.sum(w * e * e) / np.sum(w * t4 * t4)):.3f} corr {np.corrcoef(v, t4)[0, 1]:.3f} decisive non-pair corr {np.corrcoef((v - lpc)[m], (t4 - off['Pchain'])[m] if False else (off['target'])[m])[0, 1]:.3f}"
    for nm, v in (("live: Pchain + pred", lpc + lp), ("live: Pchain + 1.5 pred", lpc + 1.5 * lp), ("live: Pchain + pred + dist", lpc + lp + ld), ("live: Pchain + 1.5 pred + dist", lpc + 1.5 * lp + ld),
                  ("offline: Pchain + pred + rep + dist_y4", off["Pchain"] + off["pred"] + off["K3C_rep"] + off["K3C_dist_y4"])):
        print(f"     {nm:40s}: {ev(v)}")
    if g4c is not None:
        print(f"     chain g4 (live)                         : {ev(g4c)}")
        for A_ in (0.5,):
            print(f"     live blend A={A_}: chain + 1.5 pred + dist : {ev(A_ * g4c + (1 - A_) * (lpc + 1.5 * lp + ld))} | without dist: {ev(A_ * g4c + (1 - A_) * (lpc + 1.5 * lp))}")
