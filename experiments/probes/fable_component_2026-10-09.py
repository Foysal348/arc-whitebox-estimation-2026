"""Fable component test (9 Oct 2026): the LKV readouts (tt, ct, star) built from the CHAIN'S OWN kappa3 legs instead of
teacher inputs (design: consultations/fable_2026-10-09_design.md; ceiling: experiments/probes/fable_ceiling_2026-10-09.py).

Uses the Round-7 captures data/dumps/r7_cap_L{l}.npz (plain chain, every source dense: V21_NO_CONFINE=1), which hold the
dense source stacks (A_st, P_st, Z_st, L_st, Zf_st, R1T/R2T + static per-source vectors) at layers 3, 7, 11, 14, 15 and
mu / var / D3 / D21 / C_off / W_all at every layer.  Legs are stored [neuron i, hub h] and the chain transports them by
leg <- (diag(w1_l) W_l)^T leg (w1 = W_all[:, 4]).  For a (vertex m, target t) pair with captures at both layers the chain's
kappa3 tensor at m is T_abc(m) = sum_s sum_blocks sum_h Sym(X[a,h] P[b,h] Y[c,h]) (blocks_of_source of r7_srcgain.py) and
the carrier's cross-layer object is
    Q_m(t)[a, i] = sum_bc T_abc(m) Ydot_bi Ydot_ci = (1/3) sum_s sum_blocks [X(m) (P(t) o Y(t))^T + P(m) (X(t) o Y(t))^T + Y(m) (X(t) o P(t))^T]
because the legs at t ARE the legs at m transported by Ydot = diag(w1_m) W_m ... diag(w1_{t-1}) W_{t-1} (slot s is the same
source at both layers; checked numerically for lag-1 pairs).  Then, with the chain's own mu/var at m (alpha, phi, Phi):
    g = phi/(2 sigma), c = -alpha phi/(6 var), Ybar = W_m diag(w1_{m+1}) ... W_{t-1} (hub leg, no w1 on the hub),
    X  = (C_off(m) + diag(var_m)) Ydot,   YL = (D21(m) + diag(D3(m))) Ydot,
    star = 24 sum_a c_a Ybar_ai X_ai^3,  ct = 24 sum_a g_a Ybar_ai X_ai Q_ai,  tt = 36 sum_a c_a Ybar_ai YL_ai Q_ai.
Scores: the project metric against the teacher K4d(t) (one age alone is expected to be low: set C gives 0.34 for a single
age even with the full K = 3 response), and, when the ceiling npz is given, the FIDELITY corr(chain quantity, sample-exact
quantity of the same (t, j)) on the decisive neurons, which is the component gate (design section 7).
Also reported: the rank-r sketch Q_r = Om Om^T Q with Om = top-r eigenvectors of the chain's C(m) (the deployable truncation).

Usage (from the project root; light: ~1-2 min per pair on the laptop, < 1 GB):
  python experiments/probes/fable_component_2026-10-09.py [--pairs 14:15,11:14,11:15,7:11] [--ceiling fable_ceiling.npz]
                                                           [--cap data/dumps] [--teacher data/kaggle_out/kaggle_out_k4pub/k4teach]
                                                           [--ranks 32,64,128] [--out fable_component]
Every printed number is MEASURED on the stored captures.
"""
import os, sys, json, time, argparse
import numpy as np
from math import erf, sqrt, pi

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="14:15,11:14,11:15,7:11")
ap.add_argument("--ceiling", default="")
ap.add_argument("--cap", default="data/dumps")
ap.add_argument("--teacher", default="data/kaggle_out/kaggle_out_k4pub/k4teach")
ap.add_argument("--ranks", default="32,64,128")
ap.add_argument("--out", default="fable_component")
ap.add_argument("--max-sources", type=int, default=0, help="debug: use only the first k sources of the stack (0 = all)")
args = ap.parse_args()
f32 = lambda x: np.asarray(x, np.float32); f64 = lambda x: np.asarray(x, np.float64)
Phi_np = lambda x: 0.5 * (1 + np.vectorize(erf)(x / sqrt(2))); pdf_np = lambda x: np.exp(-x * x / 2) / sqrt(2 * pi)
Wall = np.load(os.path.join(args.teacher, "weights_seed0.npz"))["W"]; n = Wall.shape[1]
RANKS = [int(x) for x in args.ranks.split(",") if x]
ceil = np.load(args.ceiling) if args.ceiling and os.path.exists(args.ceiling) else None
t0 = time.time()


def blocks_of_source(c, s, rres, rfb):
    """The chain's kappa3 blocks of source s (legs [neuron, hub]); copied from experiments/probes/r7_srcgain.py."""
    A = f32(c["A_st"][s]); P = f32(c["P_st"][s]); Z = f32(c["Z_st"][s]); L = f32(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f32(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f32(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f32(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f32(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = f32(Z[:, rres + 1]); Y3 = np.repeat(y[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]


class Cap:
    """Lazy capture with the big stacks loaded once."""
    def __init__(self, l):
        self.z = np.load(os.path.join(args.cap, f"r7_cap_L{l}.npz")); self.cache = {}
    def __getitem__(self, k):
        if k not in self.cache: self.cache[k] = self.z[k]
        return self.cache[k]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


res = {}
for pr in args.pairs.split(","):
    m, t = (int(x) for x in pr.split(":")); j = t - m
    cm, ct_ = Cap(m), Cap(t)
    rres, rfb = int(cm["rres"]), int(cm["rfb"]); k = cm["A_st"].shape[0]
    if args.max_sources: k = min(k, args.max_sources)
    assert ct_["A_st"].shape[0] >= k, "target capture has fewer sources than the vertex capture"
    # transports m -> t from the chain's own wick w1 at every layer in between
    w1 = {l: f32(np.load(os.path.join(args.cap, f"r7_cap_L{l}.npz"))["W_all"][:, 4]) for l in range(m, t)}
    # row convention z_{l+1} = h_l W[l]: Wall[l] maps h_l -> z_{l+1}, so the ReLU of layer m transports through Wall[m]
    Ydot = w1[m][:, None] * f32(Wall[m]); Ybar = f32(Wall[m]).copy()
    for l in range(m + 1, t):
        Ydot = Ydot @ (w1[l][:, None] * f32(Wall[l])); Ybar = Ybar @ (w1[l][:, None] * f32(Wall[l]))
    # slot check: for every source, leg(t) must equal Ydot^T leg(m)  (verifies the slot order and the transport convention)
    rel = []
    for s in (0, k - 1):
        Pm = f32(cm["P_st"][s]); Pt = f32(ct_["P_st"][s]); Pp = Ydot.T @ Pm
        rel.append(float(np.linalg.norm(Pp - Pt) / np.linalg.norm(Pt)))
    print(f"[{m}->{t}] slot/transport check rel err (first, last source): {rel[0]:.2e} {rel[1]:.2e}  k={k}  ({time.time() - t0:.0f}s)", flush=True)
    # Q_m(t) from the chain's blocks
    Q = np.zeros((n, n), np.float64)
    for s in range(k):
        bm = blocks_of_source(cm, s, rres, rfb); bt = blocks_of_source(ct_, s, rres, rfb)
        for (Xm, Pm, Ym), (Xt, Pt, Yt) in zip(bm, bt):
            Q += f64(Xm @ (Pt * Yt).T + Pm @ (Xt * Yt).T + Ym @ (Xt * Pt).T) / 3.0
        if s % 4 == 3: print(f"   sources {s + 1}/{k} ({time.time() - t0:.0f}s)", flush=True)
    # hub coefficients and legs from the chain's own state at m
    mu, var = f64(cm["mu"]), f64(cm["var"]); sd = np.sqrt(var); al = mu / sd; ph = pdf_np(al)
    g = ph / (2 * sd); cc = -al * ph / (6 * var)
    Cm = f64(cm["C_off"]) + np.diag(var); D21 = f64(cm["D21"]).copy(); np.fill_diagonal(D21, 0.0); D21 += np.diag(f64(cm["D3"]))
    Yd, Yb = f64(Ydot), f64(Ybar)
    X = Cm @ Yd; YL = D21 @ Yd
    out = {"star": 24.0 * (cc[:, None] * Yb * X ** 3).sum(0), "ct": 24.0 * (g[:, None] * Yb * X * Q).sum(0), "tt": 36.0 * (cc[:, None] * Yb * YL * Q).sum(0)}
    ev, V = np.linalg.eigh(Cm)
    for r in RANKS:
        Om = V[:, -r:]; Qr = Om @ (Om.T @ Q)
        out[f"ct_r{r}"] = 24.0 * (g[:, None] * Yb * X * Qr).sum(0); out[f"tt_r{r}"] = 36.0 * (cc[:, None] * Yb * YL * Qr).sum(0)
    out["ARCH"] = out["star"] + out["ct"] + out["tt"]
    # metric vs the teacher at t
    T = np.load(os.path.join(args.teacher, f"layer{t}_seed0.npz")); mu_t = f64(T["mean"]); v_t = np.diag(f64(T["C"])); K4 = f64(T["K4d"])
    dec = np.abs(mu_t / np.sqrt(v_t)) < 1.5
    rec = {"j": j, "n_sources": k, "transport_check": rel, "teacher_corr": {}, "fidelity_vs_exact": {}}
    for kq, v in out.items():
        rec["teacher_corr"][kq] = dict(corr=corr((v / v_t ** 2)[dec], (K4 / v_t ** 2)[dec]), level=float(v[dec].mean() / K4[dec].mean()))
        if ceil is not None and f"L{t}_j{j}_{kq}" in ceil.files:
            e = f64(ceil[f"L{t}_j{j}_{kq}"])
            rec["fidelity_vs_exact"][kq] = dict(corr=corr(v[dec], e[dec]), norm_ratio=float(np.linalg.norm(v[dec]) / max(np.linalg.norm(e[dec]), 1e-300)))
    res[f"{m}->{t}"] = rec
    fid = rec["fidelity_vs_exact"]
    print(f"[{m}->{t}] age {j}: teacher corr star {rec['teacher_corr']['star']['corr']:.3f} ct {rec['teacher_corr']['ct']['corr']:.3f} tt {rec['teacher_corr']['tt']['corr']:.3f} "
          f"ARCH {rec['teacher_corr']['ARCH']['corr']:.3f} r64 ct/tt {rec['teacher_corr']['ct_r64']['corr']:.3f}/{rec['teacher_corr']['tt_r64']['corr']:.3f}"
          + (f" | fidelity vs exact: star {fid['star']['corr']:.3f} ct {fid['ct']['corr']:.3f} tt {fid['tt']['corr']:.3f}" if "tt" in fid else " | (no ceiling npz: fidelity not computed)")
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(f"{args.out}_{m}to{t}.npz", **{k_: v for k_, v in out.items()})
    del cm, ct_, Q
json.dump(res, open(args.out + ".json", "w"), indent=1)
print(f"wrote {args.out}.json ({time.time() - t0:.0f}s)")
