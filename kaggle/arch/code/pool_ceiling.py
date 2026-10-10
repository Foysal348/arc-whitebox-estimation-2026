"""Single-pool ceiling (10 Oct 2026): can ONE shared rank-r basis hold the WHOLE kappa3 (young + old sources)?

The chain's kappa3 at a stack layer l (r7 captures, every source dense) is T = sum_s sum_blocks Sym(X, P, Y) with legs
[neuron, hub]. Its (2,1) readout is D21 = sum_blocks [(X o P) Y^T + (X o Y) P^T + (P o Y) X^T] / 3 (diag = d3).
A fixed-size pool carrier keeps T in one basis U (n x r) for all three modes: T_r = T x1 UU^T x2 UU^T x3 UU^T, i.e. every leg
X -> UU^T X. Its readout error ||D21(T_r) - D21(T)|| / ||D21(T)|| (off-diagonal, and d3) is the ceiling of ANY pooled carrier
of rank r (no age split). U choices:
  oracle  top-r left singular vectors of the mode-1 unfolding of T (randomized range finder, Khatri-Rao test vectors, p = r + 128)
  cov     top-r eigenvectors of the chain's C_off + diag(var) at l (a cheap, data-free basis)
Also reported: the same with only sources older than `--young` layers pooled (young kept exact), i.e. today's split, for scale.
Pre-registered reading: a single pool is viable if some r <= 512 gives D21 error <= 2 % (the chain's 2 % tolerance) at L11-15.
Usage: python pool_ceiling.py [--layers 7,11,14,15] [--ranks 128,256,384,512,768] [--cap data/dumps]
"""
import os, json, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="7,11,14,15")
ap.add_argument("--ranks", default="128,256,384,512,768")
ap.add_argument("--cap", default="data/dumps")
ap.add_argument("--young", type=int, default=5)
ap.add_argument("--out", default="results/summaries/pool_ceiling.json")
args = ap.parse_args()
f32 = lambda x: np.asarray(x, np.float32)
t0 = time.time()


def blocks(c, s, rres, rfb, n):
    A = f32(c["A_st"][s]); P = f32(c["P_st"][s]); Z = f32(c["Z_st"][s]); L = f32(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f32(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f32(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f32(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f32(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; Y3 = np.repeat(f32(Z[:, rres + 1])[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]


def d21(bl):
    D = None
    for X, P, Y in bl:
        t = ((X * P) @ Y.T + (X * Y) @ P.T + (P * Y) @ X.T) / 3.0
        D = t if D is None else D + t
    return D


res = {}
for l in [int(x) for x in args.layers.split(",")]:
    c = np.load(os.path.join(args.cap, f"r7_cap_L{l}.npz")); k = c["A_st"].shape[0]; n = c["A_st"].shape[1]
    rres, rfb = int(c["rres"]), int(c["rfb"])
    allb = [blocks(c, s, rres, rfb, n) for s in range(k)]
    old_s = [s for s in range(k) if s < k - args.young]       # slot order = birth order; the last `young` slots are young
    D = sum(d21(b) for b in allb); off = ~np.eye(n, dtype=bool); nD, nd = np.linalg.norm(D[off]), np.linalg.norm(np.diag(D))
    print(f"[L{l}] {k} sources ({len(old_s)} old), |D21 off| {nD:.3e} ({time.time() - t0:.0f}s)", flush=True)
    # randomized range finder of the mode-1 unfolding: Y = sum_terms a (b.w1)(c.w2) over the 6 orderings of each Sym block
    pmax = max(int(x) for x in args.ranks.split(",")) + 128
    rng = np.random.default_rng(l)
    Om1 = f32(rng.standard_normal((n, pmax))); Om2 = f32(rng.standard_normal((n, pmax)))
    Yr = np.zeros((n, pmax), np.float64)
    for bl in allb:
        for X, P, Y in bl:
            for a, b, cc in ((X, P, Y), (X, Y, P), (P, X, Y), (P, Y, X), (Y, X, P), (Y, P, X)):
                Yr += a @ ((b.T @ Om1) * (cc.T @ Om2))
    Q = np.linalg.qr(Yr)[0]
    # one power-like refinement in the unfolding's column space: project T's legs and recompute (cheap proxy): skip, report as is
    Cm = c["C_off"].astype(np.float64) + np.diag(c["var"].astype(np.float64)); ev, V = np.linalg.eigh(Cm)
    row = {"n_sources": int(k), "n_old": len(old_s)}
    for r in [int(x) for x in args.ranks.split(",")]:
        Yk = Yr[:, :r + 128]; Uq = np.linalg.svd(Yk, full_matrices=False)[0][:, :r]
        for name, U in (("oracle", Uq), ("cov", V[:, -r:])):
            U = f32(U); PU = lambda Xm: U @ (U.T @ Xm)
            Dp = sum(d21([(PU(X), PU(P), PU(Y)) for X, P, Y in b]) for b in allb)
            Do = sum(d21([(PU(X), PU(P), PU(Y)) for X, P, Y in allb[s]]) for s in old_s) + sum(d21(allb[s]) for s in range(k) if s not in old_s)
            row[f"{name}_r{r}"] = dict(all_pooled_off=float(np.linalg.norm((Dp - D)[off]) / nD), all_pooled_d3=float(np.linalg.norm(np.diag(Dp - D)) / nd),
                                       old_pooled_off=float(np.linalg.norm((Do - D)[off]) / nD), old_pooled_d3=float(np.linalg.norm(np.diag(Do - D)) / nd))
        o, cv = row[f"oracle_r{r}"], row[f"cov_r{r}"]
        print(f"[L{l}] r={r}: ALL pooled D21 err oracle {o['all_pooled_off']:.3f} cov {cv['all_pooled_off']:.3f} | OLD-only pooled oracle {o['old_pooled_off']:.3f} "
              f"cov {cv['old_pooled_off']:.3f} | d3 all oracle {o['all_pooled_d3']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    res[l] = row
    del allb, c
json.dump(res, open(args.out, "w"), indent=1)
print("wrote", args.out)
