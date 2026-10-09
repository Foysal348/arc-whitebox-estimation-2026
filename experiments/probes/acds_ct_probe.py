"""ACDS probe (9 Oct 2026): Age-Collapsed Dual Sketch for the Fable C.T readout, on the chain's own r7 captures.

Proposal (external, Architecture Analysis/fable_ct_agecollapsed_rank64.py), re-implemented in the conventions of
experiments/probes/fable_component_2026-10-09.py (w1 transport, blocks_of_source with rres/rfb, legs [neuron, hub]):
  per age m = t-h .. t-1:  G_m = g_m[:,None] * Ybar_{m->t} * (C_m Ydot_{m->t})          (n x n, the C.T hub functional)
  exact:  ct = 8 sum_m sum_s sum_blocks sum_roles rowsum_a( G_m o (A_s(m) H_s(t)^T) )   (24/3; roles X|PoY, P|XoY, Y|XoP)
  collapse: A_s(m) = Ydot_{a->m}^T A_s(a), a = first window layer holding source s  =>  sum_m A_s(m)^T G_m = A_s(a)^T Gamma_a,
            Gamma_a = sum_{m>=a} Ydot_{a->m} G_m  (exact identity; checked here against the per-age route)
  sketch: S_{t-h} = C_{t-h} Omega (or top-r eigvecs), S_{m+1} = Ydot_m^T S_m, Pi_a = sum_{m>=a} S_m^T G_m,
          A^T Gamma_a ~ (A^T S_a (S_a^T S_a)^-1) Pi_a  ->  ct_i ~ 8 sum_q (H B)_iq Pi_a[q, i]
Captures hold source stacks only at a few layers; legs elsewhere are transported from the latest stack layer, so the
test runs on the consistent source subset (sources born between stack layers are absent until the next stack).
Reported per target: corr of (age-collapsed exact, ACDS-r) vs the per-age chain C.T and vs the sample-exact ceiling sum, on
decisive neurons; teacher project-metric corr of affine(chain ck4, star, ct) for ct = ceiling / chain / ACDS.
Usage: python experiments/probes/acds_ct_probe.py [--targets 12,15,16] [--ranks 64,128] [--h 6]
"""
import os, json, time, argparse
import numpy as np
from math import erf, sqrt, pi

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="12,15,16")
ap.add_argument("--ranks", default="64,128")
ap.add_argument("--h", type=int, default=6)
ap.add_argument("--cap", default="data/dumps")
ap.add_argument("--teacher", default="data/kaggle_out/kaggle_out_k4pub/k4teach")
ap.add_argument("--ceiling", default="data/kaggle_out/fable_ceiling_all/fable_ceiling_all.npz")
ap.add_argument("--out", default="results/summaries/acds_ct_probe.json")
args = ap.parse_args()
f32 = lambda x: np.asarray(x, np.float32); f64 = lambda x: np.asarray(x, np.float64)
pdf = lambda x: np.exp(-x * x / 2) / sqrt(2 * pi)
Wall = np.load(os.path.join(args.teacher, "weights_seed0.npz"))["W"]; n = Wall.shape[1]
ceil = np.load(args.ceiling); E = np.load("data/dumps/errbudget_v29plain.npz")
t0 = time.time()


def blocks_of_source(c, s, rres, rfb):
    """c: source-state dict (stacked arrays, slot s)."""
    A = f32(c["A_st"][s]); P = f32(c["P_st"][s]); Z = f32(c["Z_st"][s]); L = f32(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f32(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f32(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f32(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f32(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = f32(Z[:, rres + 1]); Y3 = np.repeat(y[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]


def roles(bm, bt):
    """(anchor leg at the vertex layer, target Hadamard H) for the 3 Sym placements of every block."""
    out = []
    for (Xm, Pm, Ym), (Xt, Pt, Yt) in zip(bm, bt):
        out += [(Xm, Pt * Yt), (Pm, Xt * Yt), (Ym, Xt * Pt)]
    return out


class Cap:
    """npz with each array decompressed once (np.load re-reads an array on every access)."""
    def __init__(self, path):
        self.z = np.load(path); self.cache = {}
    def __getitem__(self, k):
        if k not in self.cache: self.cache[k] = self.z[k]
        return self.cache[k]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


res = {}
LAY = {l: Cap(os.path.join(args.cap, f"r7_cap_L{l}.npz")) for l in range(1, 17) if os.path.exists(os.path.join(args.cap, f"r7_cap_L{l}.npz"))}
STACKS = [l for l in sorted(LAY) if "A_st" in LAY[l].z.files]
TRK = ("A_st", "P_st", "Z_st", "Zf_st")                                   # neuron legs: transported, leg <- Ystep^T leg
STK = ("L_st", "R1T_st", "R2T_st", "w2b", "s_b", "e_b", "c1_b", "c2_b")   # hub side / per-source scalars: static
print("stack layers", STACKS, {l: LAY[l]["A_st"].shape[0] for l in STACKS}, flush=True)
for t in [int(x) for x in args.targets.split(",")]:
    m0 = max(1, t - args.h); win = list(range(m0, t))
    l0 = max(l for l in STACKS if l <= m0)
    rres, rfb = int(LAY[l0]["rres"]), int(LAY[l0]["rfb"])
    w1 = {l: f32(LAY[l]["W_all"][:, 4]) for l in range(l0, t)}
    Ystep = {l: w1[l][:, None] * f32(Wall[l]) for l in range(l0, t)}      # leg(l+1) = Ystep[l]^T leg(l)
    # source states: legs of every source known at the latest stack layer <= m, transported to m (consistent subset)
    st = {k: f32(LAY[l0][k]) for k in TRK + STK}
    for l in range(l0, m0):
        for k in TRK: st[k] = np.matmul(Ystep[l].T, st[k]).astype(np.float32)
    states = {}
    for m in win + [t]:
        if m in STACKS and m > l0:
            st = {k: f32(LAY[m][k]) for k in TRK + STK}
        states[m] = st
        if m < t:
            st = {k: (np.matmul(Ystep[m].T, v).astype(np.float32) if k in TRK else v) for k, v in st.items()}
    kk = {m: states[m]["A_st"].shape[0] for m in win + [t]}
    G, star = {}, np.zeros(n)
    for m in win:
        Yd = Ystep[m].copy(); Yb = f32(Wall[m]).copy()
        for l in range(m + 1, t):
            Yd = Yd @ Ystep[l]; Yb = Yb @ Ystep[l]
        c = LAY[m]; var = f64(c["var"]); sd = np.sqrt(var); al = f64(c["mu"]) / sd; ph = pdf(al)
        X = f32((f64(c["C_off"]) + np.diag(var)) @ f64(Yd))
        G[m] = f32(ph / (2 * sd))[:, None] * Yb * X
        star += 24.0 * ((-al * ph / (6 * var))[:, None] * f64(Yb) * f64(X) ** 3).sum(0)
    print(f"[L{t}] window {m0}..{t - 1} (from stack L{l0}), sources per layer {kk}  G built ({time.time() - t0:.0f}s)", flush=True)
    bt = {s: blocks_of_source(states[t], s, rres, rfb) for s in range(kk[t])}
    # 1) per-age chain C.T on the consistent subset (the component-test route, summed over ages)
    ct_age = np.zeros(n)
    for m in win:
        Q = np.zeros((n, n), np.float32)
        for s in range(kk[m]):
            for A, H in roles(blocks_of_source(states[m], s, rres, rfb), bt[s]):
                Q += A @ H.T
        ct_age += 8.0 * f64((G[m] * Q).sum(0))
    print(f"[L{t}] per-age chain C.T done ({time.time() - t0:.0f}s)", flush=True)
    # 2) age-collapsed exact: Gamma_a = G_a + Ystep_a Gamma_{a+1}
    Gam = {}; run = None
    for a in reversed(win):
        run = G[a].copy() if run is None else G[a] + Ystep[a] @ run
        Gam[a] = run
    anchor = {s: next(m for m in win if s < kk[m]) for s in range(kk[t]) if any(s < kk[m] for m in win)}
    ct_col = np.zeros(n); nterms = 0
    for s, a in anchor.items():
        for A, H in roles(blocks_of_source(states[a], s, rres, rfb), bt[s]):
            ct_col += 8.0 * f64(((A.T @ Gam[a]) * H.T).sum(0)); nterms += 1     # (A^T Gamma)[h, i] H[i, h]
    print(f"[L{t}] age-collapsed exact done, {nterms} source-role terms ({time.time() - t0:.0f}s)", flush=True)
    # 3) ACDS sketches
    sk = {}
    Cm0 = f64(LAY[m0]["C_off"]) + np.diag(f64(LAY[m0]["var"]))
    ev, V = np.linalg.eigh(Cm0)
    for r in [int(x) for x in args.ranks.split(",")]:
        rng = np.random.default_rng(1709 + t)
        for kind, S0 in (("rad", Cm0 @ (rng.choice([-1.0, 1.0], size=(n, r)) / np.sqrt(r))), ("eig", V[:, -r:])):
            S = {m0: f32(S0)}
            for m in win[:-1]:
                S[m + 1] = Ystep[m].T @ S[m]
            Pi = {}; run = None
            for a in reversed(win):
                x = S[a].T @ G[a]; run = x if run is None else run + x; Pi[a] = run
            Sti = {}
            for a in win:
                M = f64(S[a].T @ S[a]); Sti[a] = f32(f64(S[a]) @ np.linalg.inv(M + 1e-10 * np.trace(M) / r * np.eye(r)))
            ct = np.zeros(n)
            for s, a in anchor.items():
                for A, H in roles(blocks_of_source(states[a], s, rres, rfb), bt[s]):
                    B = A.T @ Sti[a]; ct += 8.0 * f64(((H @ B) * Pi[a].T).sum(1))
            sk[f"{kind}{r}"] = ct
        print(f"[L{t}] rank {r} done ({time.time() - t0:.0f}s)", flush=True)
    # scores
    T = np.load(os.path.join(args.teacher, f"layer{t}_seed0.npz")); v = np.diag(f64(T["C"])); K4 = f64(T["K4d"])
    dec = np.abs(f64(T["mean"]) / np.sqrt(v)) < 1.5
    exact = sum(f64(ceil[f"L{t}_j{j}_ct"]) for j in range(1, t - m0 + 1))
    exstar = sum(f64(ceil[f"L{t}_j{j}_star"]) for j in range(1, t - m0 + 1))
    ck = f64(E[f"L{t}_ck4"])

    def feat(cols):
        X = np.stack([c / v ** 2 for c in cols] + [np.ones(n)], 1); y = K4 / v ** 2
        b = np.linalg.lstsq(X, y, rcond=None)[0]; return corr((X @ b)[dec], y[dec])
    cands = {"ct_age": ct_age, "ct_collapsed": ct_col, **sk}
    row = {"identity_rel_err": float(np.linalg.norm(ct_col - ct_age) / np.linalg.norm(ct_age)),
           "star_fidelity": corr(star[dec], exstar[dec]),
           "fidelity_vs_ceiling": {k: corr(x[dec], exact[dec]) for k, x in cands.items()},
           "fidelity_vs_chain": {k: corr(x[dec], ct_age[dec]) for k, x in cands.items()},
           "teacher_affine": {"chain": feat([ck]), "ceil_star_ct": feat([ck, exstar + exact]),
                              **{k: feat([ck, star + x]) for k, x in cands.items()}}}
    res[t] = row
    print(f"[L{t}] identity err {row['identity_rel_err']:.2e} | fidelity vs ceiling " +
          " ".join(f"{k} {x:.3f}" for k, x in row["fidelity_vs_ceiling"].items()) + " | vs chain " +
          " ".join(f"{k} {x:.3f}" for k, x in row["fidelity_vs_chain"].items()) + " | teacher affine " +
          " ".join(f"{k} {x:.3f}" for k, x in row["teacher_affine"].items()), flush=True)
    del states, G, bt, Gam
json.dump(res, open(args.out, "w"), indent=1)
print(f"wrote {args.out} ({time.time() - t0:.0f}s)")
