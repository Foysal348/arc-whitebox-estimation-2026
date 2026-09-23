"""Round 7 (Fable, 23 Sep), part 3: the dropped hub-form terms ACCUMULATE.

The kappa3 content is transported through the ReLU + weights with gain ~1 per layer (each leg gets W^T diag(w1),
sqrt(2 E[w1^2]) ~ 1), so a term the newborn gets wrong at layer l' stays in the sources and shows up in every later
slice.  Part 1 regressed the slice error at l+1 on the FRESH dropped terms of layer l only (R^2 0.13-0.19).  Here every
earlier layer's dropped hub terms are born as Sym(X x I x Y) tensors from the TRUE pre-activation data of that layer,
transported linearly exactly as the chain transports its sources (W[l']^T, then W[l'']^T diag(w1(l'')) for l'' = l'+1..l),
and contracted to the (2,1) slice at l+1 with the chain's own formula (1/3)[(X*P)Y^T + (P*Y)X^T + (X*Y)P^T].

Families (coefficient the chain is missing in brackets):
    famA   T = (1/2) sum_hub w3_h [C_hx w1_x K_hhy w1_y + (x<->y)]        [-1/2]
    famB   T = (1/2) sum_hub w2_h [K_xxh w2_x C_hy w1_y + (x<->y)]        [-1/2]
    dedge  T = (1/2) sum_hub w3_h [C_hx^2 w2_x C_hy w1_y + (x<->y)]       [-1]
Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r7_accum.py
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"]; n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1.0 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64); f32 = lambda x: np.asarray(x, np.float32)
off = ~np.eye(n, dtype=bool)
TARGETS = [int(a) for a in sys.argv[1:]] or [3, 7, 11, 14, 15]

def S_diag(W, k3): return ((W * W) * k3[:, None]).T @ W
def S_21(W, Y):
    YW = Y @ W; return (W * W).T @ YW + 2.0 * (W * YW).T @ W
def stats(x, y):
    s = float(np.dot(x, y) / np.dot(x, x)); ss = float(np.sum(y * y))
    return dict(slope=s, r2_fit=float(1 - np.sum((y - s * x) ** 2) / ss), r2_1to1=float(1 - np.sum((y - x) ** 2) / ss), corr=float(np.corrcoef(x, y)[0, 1]))
def multi(X, y, names):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
    return dict(r2=r2, coef={nm: float(c) for nm, c in zip(names, coef)})

TRUE = {}
def true_state(l):
    """true pre-activation data of layer l: C_off, K (K[h,y] = kappa3(h,h,y), zero diag), wicks"""
    if l not in TRUE:
        T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); C = f64(T["C"]); mu = f64(T["mean"]); s = np.sqrt(np.diag(C)); a = mu / s; ph = pdf(a)
        Co = C.copy(); np.fill_diagonal(Co, 0.0); K = f64(T["Y21"]).copy(); np.fill_diagonal(K, 0.0)
        TRUE[l] = dict(Co=f32(Co), K=f32(K), w1=f32(Phi(a)), w2=f32(ph / s), w3=f32(-a * ph / s ** 2))
    return TRUE[l]

def born_legs(l):
    """post-ReLU hub tensors of layer l as Sym(X x I x Y): X[x,h] = 3 g_h L[h,x], Y[y,h] = M[h,y]  (float32)"""
    t = true_state(l); Co, K, w1, w2, w3 = t["Co"], t["K"], t["w1"], t["w2"], t["w3"]
    Cw1 = Co * w1[None, :]; Kw1 = K * w1[None, :]; KTw2 = K.T * w2[None, :]; C2w2 = Co * Co * w2[None, :]
    fam = {}
    if np.any(K != 0):
        fam["famA"] = (3.0 * (Cw1 * w3[:, None]).T, Kw1.T)     # g = w3, L = C w1, M = K w1
        fam["famB"] = (3.0 * (KTw2 * w2[:, None]).T, Cw1.T)    # g = w2, L = K^T w2, M = C w1
    fam["dedge"] = (3.0 * (C2w2 * w3[:, None]).T, Cw1.T)       # g = w3, L = C^2 w2, M = C w1
    return fam

def transport(X, P, Y, l_from, l_to):
    """legs born post-ReLU at l_from -> pre-activation legs at l_to + 1 (P starts as I)"""
    W = f32(Wall[l_from]); X = W.T @ X; Y = W.T @ Y; P = W.T.copy() if P is None else W.T @ P
    for ll in range(l_from + 1, l_to + 1):
        w1 = true_state(ll)["w1"]; W = f32(Wall[ll])
        X = W.T @ (w1[:, None] * X); Y = W.T @ (w1[:, None] * Y); P = W.T @ (w1[:, None] * P)
    return X, P, Y

def slice21(X, P, Y):
    D = (X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T
    D = f64(D) / 3.0; np.fill_diagonal(D, 0.0); return D

out = []
for l in TARGETS:
    t0 = time.time()
    P_ = np.load(os.path.join(POST, f"layer{l}_post_seed0.npz")); Tn = np.load(os.path.join(PRE, f"layer{l+1}_seed0.npz"))
    c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); cn = np.load(os.path.join(CAP, f"r7_cap_L{l+1}.npz")); W = f64(Wall[l])
    k3_post_true = np.diag(f64(P_["Y21"])).copy(); Y21_post_true = f64(P_["Y21"]).copy(); np.fill_diagonal(Y21_post_true, 0.0)
    Dtrue = f64(Tn["Y21"]).copy(); np.fill_diagonal(Dtrue, 0.0); Dchain = f64(cn["D21"]).copy(); np.fill_diagonal(Dchain, 0.0)
    K3c = f64(c["K3"]); K21c = f64(c["K21"]).copy(); np.fill_diagonal(K21c, 0.0)
    e_x = (Dchain - S_diag(W, K3c) - S_21(W, K21c)) - (Dtrue - S_diag(W, k3_post_true) - S_21(W, Y21_post_true)); e_tot = Dchain - Dtrue
    acc = {"famA": np.zeros((n, n)), "famB": np.zeros((n, n)), "dedge": np.zeros((n, n))}; fresh = {}; per_birth = {}
    for lp in range(1, l + 1):
        for name, (X0, Y0) in born_legs(lp).items():
            X, P, Y = transport(X0, None, Y0, lp, l); D = slice21(X, P, Y)
            acc[name] += D
            if lp == l: fresh[name] = D
            per_birth.setdefault(name, []).append(float(np.sqrt(np.mean(D[off] ** 2))))
    y = e_x[off]; yt = e_tot[off]
    rec = {"post_layer": l, "e_dist_rms": float(np.sqrt(np.mean(y ** 2))), "e_tot_rms": float(np.sqrt(np.mean(yt ** 2))),
           "rms_acc": {k_: float(np.sqrt(np.mean(v[off] ** 2))) for k_, v in acc.items()}, "rms_fresh": {k_: float(np.sqrt(np.mean(v[off] ** 2))) for k_, v in fresh.items()},
           "rms_per_birth_layer": per_birth,
           "single_acc": {k_: stats(v[off], y) for k_, v in acc.items()}, "single_fresh": {k_: stats(v[off], y) for k_, v in fresh.items()}}
    names = ["famA", "famB", "dedge"]
    Xa = np.stack([acc[k_][off] for k_ in names], 1); Xf = np.stack([fresh[k_][off] for k_ in names], 1)
    rec["joint_acc"] = multi(Xa, y, names); rec["joint_fresh"] = multi(Xf, y, names); rec["joint_acc_on_e_tot"] = multi(Xa, yt, names)
    theory = -0.5 * (acc["famA"] + acc["famB"]) - acc["dedge"]
    rec["theory_acc_1to1"] = {"r2_e_dist": float(1 - np.sum((y - theory[off]) ** 2) / np.sum(y ** 2)), "r2_e_tot": float(1 - np.sum((yt - theory[off]) ** 2) / np.sum(yt ** 2)),
                              "resid_rms": float(np.sqrt(np.mean((y - theory[off]) ** 2)))}
    ab = -0.5 * (acc["famA"] + acc["famB"])
    rec["theory_acc_AB_only_1to1_r2_e_dist"] = float(1 - np.sum((y - ab[off]) ** 2) / np.sum(y ** 2))
    rec["theory_acc_AB_only_stats"] = stats(ab[off], y)
    # what a corrected chain would leave: the residual slice error relative to the true slice
    rec["rel_slice_err_now"] = float(np.sqrt(np.mean(yt ** 2) / np.mean(Dtrue[off] ** 2)))
    rec["rel_slice_err_after_theory"] = float(np.sqrt(np.mean((yt - theory[off]) ** 2) / np.mean(Dtrue[off] ** 2)))
    out.append(rec)
    print(f"post L{l} -> pre L{l+1} ({time.time()-t0:.0f}s): e_dist rms {rec['e_dist_rms']:.2e} | acc rms famA {rec['rms_acc']['famA']:.2e} famB {rec['rms_acc']['famB']:.2e} dedge {rec['rms_acc']['dedge']:.2e} (fresh famA {rec['rms_fresh']['famA']:.2e})", flush=True)
    print("   per birth layer famA rms: " + " ".join(f"{v:.1e}" for v in per_birth["famA"]), flush=True)
    print("   e_dist on ACC: " + ", ".join(f"{k_}: slope {v['slope']:+.2f} R2 {v['r2_fit']:.3f}" for k_, v in rec["single_acc"].items()) + f" | joint R2 {rec['joint_acc']['r2']:.3f} " + ", ".join(f"{k_} {v:+.2f}" for k_, v in rec["joint_acc"]["coef"].items()), flush=True)
    print("   e_dist on FRESH: " + ", ".join(f"{k_}: slope {v['slope']:+.2f} R2 {v['r2_fit']:.3f}" for k_, v in rec["single_fresh"].items()) + f" | joint R2 {rec['joint_fresh']['r2']:.3f}", flush=True)
    th = rec["theory_acc_1to1"]; print(f"   THEORY accumulated (-1/2 A, -1/2 B, -1 dedge), 1:1: R2 e_dist {th['r2_e_dist']:.3f}, e_tot {th['r2_e_tot']:.3f}, resid rms {th['resid_rms']:.2e} | A+B only: R2 {rec['theory_acc_AB_only_1to1_r2_e_dist']:.3f} (slope {rec['theory_acc_AB_only_stats']['slope']:+.3f})"
          f" | relative slice error {rec['rel_slice_err_now']*100:.2f}% -> {rec['rel_slice_err_after_theory']*100:.2f}% after the correction", flush=True)
json.dump(out, open("../results/summaries/r7_accum.json", "w"), indent=1)
