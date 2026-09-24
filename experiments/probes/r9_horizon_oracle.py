"""Round 9 follow-up (GPT 5.6 Sol's 'horizon oracle', 24 Sep): how long must the kappa3 memory be?

From Fable's dense source captures (data/dumps/r7_cap_L{l}.npz: every live source's blocks at the pre-activation of
layer l, with every source kept dense, so the chain's D21 = sum over sources of their own (2,1) slices), compute each
source's slice contribution by AGE and, for h = 1..8, the relative error of keeping only sources of age <= h:
    eps_h = || D21 - sum_{age <= h} D21_age ||_F / || D21 ||_F      (off-diagonal)
Also the same restricted to the rows of the neurons that decide the mean (|mu/sigma| < 1.5), and the error against the
TEACHER's true slice at the same layer (the chain's own full D21 is 4-7% from the truth).
Sol's gates: eps_2 or eps_3 < ~2% -> a short horizon state is viable; h ~ 4-6 -> age-pooled query memory; eps_8 > 2% ->
long memory in a radically cheaper coordinate system.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r9_horizon_oracle.py
"""
import os, json, numpy as np
CAP = "../data/dumps"; PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"
f64 = lambda x: np.asarray(x, np.float64)

def blocks_of_source(c, s, rres, rfb, n):          # identical to r8_k4_1111.py
    A = f64(c["A_st"][s]); P = f64(c["P_st"][s]); Z = f64(c["Z_st"][s]); L = f64(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f64(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f64(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f64(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f64(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = Z[:, rres + 1]; Y3 = np.repeat(y[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]

out = []
for l in (7, 11, 14, 15):
    c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); rres = int(c["rres"]); rfb = int(c["rfb"]); k = c["A_st"].shape[0]; n = c["A_st"].shape[1]
    D21 = f64(c["D21"]).copy(); np.fill_diagonal(D21, 0.0)
    mu, var = f64(c["mu"]), f64(c["var"]); dec = np.abs(mu / np.sqrt(var)) < 1.5
    li = int(c["li"]); T = np.load(os.path.join(PRE, f"layer{li + 1}_seed0.npz")); Yt = f64(T["Y21"]).copy(); np.fill_diagonal(Yt, 0.0)
    by_age = {}
    for s in range(k):
        age = k - s; sl = np.zeros((n, n))
        for (X, P, Y) in blocks_of_source(c, s, rres, rfb, n):
            sl += ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0
        np.fill_diagonal(sl, 0.0); by_age[age] = sl
    tot = sum(by_age.values())
    off = ~np.eye(n, dtype=bool); M = off & dec[:, None]
    rec = {"layer": li + 1, "n_sources": k, "sum_check": float(np.linalg.norm(tot - D21) / np.linalg.norm(D21)),
           "chain_vs_truth": float(np.linalg.norm((D21 - Yt)[off]) / np.linalg.norm(Yt[off])), "eps": {}, "eps_decisive": {}, "vs_truth": {}, "age_share": {}}
    for age in sorted(by_age): rec["age_share"][age] = float(np.linalg.norm(by_age[age][off]) / np.linalg.norm(D21[off]))
    for h in range(1, min(k, 10) + 1):
        part = sum(by_age[a] for a in by_age if a <= h)
        rec["eps"][h] = float(np.linalg.norm((D21 - part)[off]) / np.linalg.norm(D21[off]))
        rec["eps_decisive"][h] = float(np.linalg.norm((D21 - part)[M]) / np.linalg.norm(D21[M]))
        rec["vs_truth"][h] = float(np.linalg.norm((part - Yt)[off]) / np.linalg.norm(Yt[off]))
    out.append(rec)
    print(f"layer {li+1} ({k} sources; source sum vs chain D21 {rec['sum_check']:.1e}; chain vs truth {rec['chain_vs_truth']:.3f})", flush=True)
    print("   age share of ||D21||:  " + "  ".join(f"{a}:{v:.2f}" for a, v in rec["age_share"].items()), flush=True)
    print("   eps_h (keep ages <= h): " + "  ".join(f"h{h}:{v:.3f}" for h, v in rec["eps"].items()), flush=True)
    print("   same, decisive rows:    " + "  ".join(f"h{h}:{v:.3f}" for h, v in rec["eps_decisive"].items()), flush=True)
    print("   vs TRUE slice:          " + "  ".join(f"h{h}:{v:.3f}" for h, v in rec["vs_truth"].items()), flush=True)
json.dump(out, open("../results/summaries/r9_horizon_oracle.json", "w"), indent=1)
