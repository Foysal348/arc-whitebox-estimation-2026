# Kaggle wrapper for experiments/probes/acds_ct_probe.py (9 Oct 2026): sets up the data layout, then runs the probe verbatim.
import glob, os, sys
_src = os.path.dirname(glob.glob("/kaggle/input/**/r7_cap_L12.npz", recursive=True)[0])
os.chdir("/kaggle/working"); os.makedirs("data/dumps", exist_ok=True); os.makedirs("teach", exist_ok=True); os.makedirs("results/summaries", exist_ok=True)
for f in glob.glob(f"{_src}/*.npz"):
    b = os.path.basename(f)
    dst = f"teach/{b.split('__', 1)[1]}" if b.startswith("teach__") else f"data/dumps/{b}"
    if not os.path.exists(dst): os.symlink(f, dst)
for f in glob.glob("/kaggle/input/**/tv__layer*_seed0.npz", recursive=True):
    dst = f"teach/{os.path.basename(f).split('__', 1)[1]}"
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(f, dst)
sys.argv = ["acds2_ct_probe.py", "--targets", "7,8,9,10,11,12,13,14,15,16", "--ranks", "64,96,128", "--save", "results/summaries/acds2_feats.npz", "--teacher", "teach",
            "--ceiling", "data/dumps/fable_ceiling_all.npz", "--out", "results/summaries/acds2_ct_probe.json"]
"""ACDS round 2 (10 Oct 2026): Astra's grouped A/P evaluation and direct-Gamma sketches (gamA: per-anchor
orthonormal basis of Gamma_a Omega; gamAw: same with the target-aware column weights omega; gam1/gam1w: one basis at the window start,
transported like a leg; eig: top-r eigenvectors of C, the round-1 reference). Round-1 header follows.
ACDS probe (9 Oct 2026): Age-Collapsed Dual Sketch for the Fable C.T readout, on the chain's own r7 captures.

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
ap.add_argument("--ranks", default="64,80,96,128")
ap.add_argument("--h", type=int, default=6)
ap.add_argument("--cap", default="data/dumps")
ap.add_argument("--teacher", default="data/kaggle_out/kaggle_out_k4pub/k4teach")
ap.add_argument("--ceiling", default="data/kaggle_out/fable_ceiling_all/fable_ceiling_all.npz")
ap.add_argument("--out", default="results/summaries/acds2_ct_probe.json")
ap.add_argument("--save", default="", help="npz with star / ct variants per target (for E2E fits)")
ap.add_argument("--first-stack", type=int, default=3, help="window start is clipped to the first stack layer")
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


res = {}; SAVE = {}
LAY = {l: Cap(os.path.join(args.cap, f"r7_cap_L{l}.npz")) for l in range(1, 17) if os.path.exists(os.path.join(args.cap, f"r7_cap_L{l}.npz"))}
STACKS = [l for l in sorted(LAY) if "A_st" in LAY[l].z.files]
TRK = ("A_st", "P_st", "Z_st", "Zf_st")                                   # neuron legs: transported, leg <- Ystep^T leg
STK = ("L_st", "R1T_st", "R2T_st", "w2b", "s_b", "e_b", "c1_b", "c2_b")   # hub side / per-source scalars: static
print("stack layers", STACKS, {l: LAY[l]["A_st"].shape[0] for l in STACKS}, flush=True)
for t in [int(x) for x in args.targets.split(",")]:
    m0 = max(args.first_stack, t - args.h); win = list(range(m0, t))
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
    # 3) Astra round 2: grouped A/P evaluation (exact identity check) + direct-Gamma sketches
    parts = {}
    for s_, a in anchor.items():
        sa, stt = states[a], states[t]
        A = f32(sa["A_st"][s_]); P = f32(sa["P_st"][s_]); Z = f32(sa["Z_st"][s_]); Zf = f32(sa["Zf_st"][s_])
        L = f32(sa["L_st"][s_]); w2b, sb, eb, c1, c2 = (f32(sa[k][s_])[None, :] for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
        (X1t, Pt, Y1t), (Mt, _, _), (X3t, _, Y3t) = bt[s_]
        HA = 3 * (Pt * Y1t) + (X1t * Pt) * w2b + 3 * (Pt * Pt) * eb + (Pt * Y3t) * c1
        HP = X1t * Y1t + (Pt * Pt) * sb + 2 * (Mt * Pt) + (Pt * Y3t) * c2 + X3t * Y3t
        low = [(Z, (Pt * Pt) @ L), (Z[:, rres + 1:rres + 2], (X3t * Pt).sum(1, keepdims=True))]
        if rfb > 0:
            low += [(Zf[:, :rfb], (Pt * Y1t) @ f32(sa["R1T_st"][s_])), (Zf[:, rfb:], (X1t * Pt) @ f32(sa["R2T_st"][s_]))]
        lowc = sum(f64(((Za.T @ Gam[a]).T * HR).sum(1)) for Za, HR in low)      # exact: Z^T Gamma is thin
        parts[s_] = (a, A, P, HA, HP, lowc)
    ct_grp = np.zeros(n)
    for s_, (a, A, P, HA, HP, lowc) in parts.items():
        ct_grp += 8.0 * (f64(((A.T @ Gam[a]).T * HA).sum(1)) + f64(((P.T @ Gam[a]).T * HP).sum(1)) + lowc)
    print(f"[L{t}] grouped exact vs per-age: rel err {np.linalg.norm(ct_grp - ct_age) / np.linalg.norm(ct_age):.2e} ({time.time() - t0:.0f}s)", flush=True)
    # target-aware weights (chain's own mu / var at t): omega_i = dec_i sqrt(lambda_i) / v_i^2
    vt = f64(LAY[t]["var"]); dct = np.abs(f64(LAY[t]["mu"]) / np.sqrt(vt)) < 1.5
    lam = np.zeros(n)
    for s_, (a, A, P, HA, HP, lowc) in parts.items():
        lam += f64((HA ** 2) @ (A ** 2).sum(0)) + f64((HP ** 2) @ (P ** 2).sum(0))
    omg = f32(dct * np.sqrt(lam) / (vt ** 2 + 1e-6 * np.median(vt ** 2)))
    anchors = sorted(set(a for a, *_ in parts.values()))

    def evaluate(Sb):
        """Sb: {anchor: basis n x r}; Pi_a = S_a^T Gamma_a (= sum_m S_m^T G_m with S transported); S~ = S (S^T S)^-1."""
        ct = np.zeros(n)
        for a in anchors:
            S = Sb[a]; M = f64(S.T @ S); Sti = f32(f64(S) @ np.linalg.inv(M + 1e-10 * np.trace(M) / S.shape[1] * np.eye(S.shape[1])))
            Pi = S.T @ Gam[a]
            for s_, (aa, A, P, HA, HP, lowc) in parts.items():
                if aa != a: continue
                Zs = HA @ (A.T @ Sti) + HP @ (P.T @ Sti)
                ct += 8.0 * (f64((Zs * Pi.T).sum(1)) + lowc)
        return ct

    def transported(S0):
        S = {m0: S0}
        for m in win[:-1]:
            S[m + 1] = Ystep[m].T @ S[m]
        return S
    sk = {}; cost = {}
    nsrc = len(parts); u = lambda nprod, r: nprod * r / n
    Cm0 = f64(LAY[m0]["C_off"]) + np.diag(f64(LAY[m0]["var"])); ev, V = np.linalg.eigh(Cm0)
    for r in [int(x) for x in args.ranks.split(",")]:
        Om = f32(np.random.default_rng(1709 + t).choice([-1.0, 1.0], size=(n, r)))
        orth = lambda R: f32(np.linalg.qr(f64(R))[0])
        sk[f"gamA{r}"] = evaluate({a: orth(Gam[a] @ Om) for a in anchors})
        sk[f"gamAw{r}"] = evaluate({a: orth(Gam[a] @ (omg[:, None] * Om)) for a in anchors})
        sk[f"gamAp{r}"] = evaluate({a: orth(Gam[a] @ (Gam[a].T @ orth(Gam[a] @ Om))) for a in anchors})       # one power step
        sk[f"gamAwp{r}"] = evaluate({a: orth(Gam[a] @ (Gam[a].T @ orth(Gam[a] @ (omg[:, None] * Om)))) for a in anchors})
        sk[f"gam1{r}"] = evaluate(transported(orth(Gam[m0] @ Om)))
        sk[f"gam1w{r}"] = evaluate(transported(orth(Gam[m0] @ (omg[:, None] * Om))))
        sk[f"eig{r}"] = evaluate(transported(f32(V[:, -r:])))
        src = 4 * nsrc
        cost[r] = {"gamA": u(src + 11 + sum(2 * (t - a) - 1 for a in anchors), r), "gamAp": u(src + 33 + sum(2 * (t - a) - 1 for a in anchors), r), "gam1": u(src + 22, r), "eig": u(src + 11, r)}
        print(f"[L{t}] rank {r} done; per-target thin-GEMM cost u: {cost[r]} ({time.time() - t0:.0f}s)", flush=True)
    # scores
    T = np.load(os.path.join(args.teacher, f"layer{t}_seed0.npz")); v = f64(T["var"]) if "var" in T.files else np.diag(f64(T["C"])); K4 = f64(T["K4d"])
    dec = np.abs(f64(T["mean"]) / np.sqrt(v)) < 1.5
    exact = sum(f64(ceil[f"L{t}_j{j}_ct"]) for j in range(1, t - m0 + 1))
    exstar = sum(f64(ceil[f"L{t}_j{j}_star"]) for j in range(1, t - m0 + 1))
    ck = f64(E[f"L{t}_ck4"])

    def feat(cols):
        X = np.stack([c / v ** 2 for c in cols] + [np.ones(n)], 1); y = K4 / v ** 2
        b = np.linalg.lstsq(X, y, rcond=None)[0]; return corr((X @ b)[dec], y[dec])
    cands = {"ct_age": ct_age, "ct_grouped": ct_grp, **sk}
    row = {"identity_rel_err": float(np.linalg.norm(ct_col - ct_age) / np.linalg.norm(ct_age)), "grouped_rel_err": float(np.linalg.norm(ct_grp - ct_age) / np.linalg.norm(ct_age)),
           "cost_u_per_target": cost, "n_sources": nsrc, "anchors": anchors,
           "star_fidelity": corr(star[dec], exstar[dec]),
           "fidelity_vs_ceiling": {k: corr(x[dec], exact[dec]) for k, x in cands.items()},
           "fidelity_vs_chain": {k: corr(x[dec], ct_age[dec]) for k, x in cands.items()},
           "teacher_affine": {"chain": feat([ck]), "ceil_star_ct": feat([ck, exstar + exact]),
                              **{k: feat([ck, star + x]) for k, x in cands.items()}}}
    res[t] = row
    SAVE.update({f'L{t}_star': star, **{f'L{t}_{k}': x for k, x in cands.items()}})
    print(f"[L{t}] identity err {row['identity_rel_err']:.2e} | fidelity vs ceiling " +
          " ".join(f"{k} {x:.3f}" for k, x in row["fidelity_vs_ceiling"].items()) + " | vs chain " +
          " ".join(f"{k} {x:.3f}" for k, x in row["fidelity_vs_chain"].items()) + " | teacher affine " +
          " ".join(f"{k} {x:.3f}" for k, x in row["teacher_affine"].items()), flush=True)
    del states, G, bt, Gam, parts
json.dump(res, open(args.out, "w"), indent=1)
if args.save: np.savez(args.save, **SAVE)
print(f"wrote {args.out} ({time.time() - t0:.0f}s)")
