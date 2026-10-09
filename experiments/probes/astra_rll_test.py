"""Astra RLL test (10 Oct 2026): does the chain miss (half of) the cubic-response C x kappa3 term?

Hermite response through order 3 around the pre-activation (mu, sigma) of layer l:
    h_i - E h_i = l_i u_i + q_i (u_i^2 - v_i) + r_i (u_i^3 - 3 v_i u_i),  l = Phi(a), q = phi/(2 sigma), r = -a phi/(6 sigma^2).
One cubic vertex + two linear vertices, first order in T = kappa3(u) (Leonov-Shiryaev, connected partitions only):
    kappa(R_i, L_j, L_k) = r_i l_j l_k [3 C_ij T_iik + 3 C_ik T_iij]          (+ kappa5, dropped)
Mapped to the next pre-activation slice D21_ab = kappa3(z_a, z_a, z_b) with Q = diag(l) W, R = diag(r) W, B = C Q, E = D Q
(D[i,k] = T_iik = Y21 orientation):
    corrected:  dD21 = 6 (R o B)^T E + 6 (R o E)^T B + 6 (B o E)^T R
    astra:      dD21 = 6 (R o B)^T E + 6 (R o E)^T B + 2 diag(R^T d) (Q^T C Q) + (6 B o E + d (Q^T C Q)_pp)^T R
                (the C_jk T_iii term of the proposal is a disconnected partition; kept here only to test it)
Test (Astra's proposal): teacher inputs at layer l and W_l give dD21 at l+1; compare with the chain's residual
E_{l+1} = D21_teacher(l+1) - D21_chain(l+1) (plain chain, r7 captures). Off-diagonal and diagonal (d3) separately:
    beta = <E, dD> / |dD|^2, rho = corr(E, dD), r_half = |E - dD/2| / |E|.
Pre-registered reading: the mechanism is supported if beta > 0 and roughly 0.5, rho > 0 at late transitions, and adding dD/2
cuts the off-diagonal residual by >= 15 % at several late layers. KILL if beta ~ 0 or negative or rho < 0.1 at L>=9.
Light: one transition at a time, ~10 n^3 products, < 300 MB.
Usage: python experiments/probes/astra_rll_test.py [--layers 2-15]
"""
import json, argparse
import numpy as np
from scipy.special import ndtr

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="2-15")
ap.add_argument("--teacher", default="data/kaggle_out/kaggle_out_k4pub/k4teach")
ap.add_argument("--cap", default="data/dumps")
ap.add_argument("--out", default="results/summaries/astra_rll_test.json")
args = ap.parse_args()
lo, hi = (int(x) for x in args.layers.split("-"))
W = np.load(f"{args.teacher}/weights_seed0.npz")["W"]
f64 = lambda x: np.asarray(x, np.float64)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


res = {}
for l in range(lo, hi + 1):
    T = np.load(f"{args.teacher}/layer{l}_seed0.npz")
    C = f64(T["C"]); v = np.diag(C).copy(); s = np.sqrt(v); a = f64(T["mean"]) / s
    ph = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    ell, r = ndtr(a), -a * ph / (6 * v)
    D = f64(T["Y21"]); d = np.diag(D).copy()
    Wl = f64(W[l])
    Q = ell[:, None] * Wl; R = r[:, None] * Wl
    B = C @ Q; E = D @ Q
    base = 6 * (R * B).T @ E + 6 * (R * E).T @ B
    corrected = base + 6 * (B * E).T @ R
    G = Q.T @ B
    astra = base + 2 * (R.T @ d)[:, None] * G + (6 * B * E + d[:, None] * np.diag(G)[None, :]).T @ R
    Tn = np.load(f"{args.teacher}/layer{l + 1}_seed0.npz"); cn = np.load(f"{args.cap}/r7_cap_L{l + 1}.npz")
    Yt = f64(Tn["Y21"]); Dc = f64(cn["D21"]).copy(); np.fill_diagonal(Dc, f64(cn["D3"]))
    Res = Yt - Dc
    off = ~np.eye(len(v), dtype=bool)
    row = {"resid_rel_off": float(np.linalg.norm(Res[off]) / np.linalg.norm(Yt[off])),
           "resid_rel_diag": float(np.linalg.norm(np.diag(Res)) / np.linalg.norm(np.diag(Yt)))}
    for name, dD in (("corrected", corrected), ("astra", astra)):
        for part, sel in (("off", off), ("diag", np.eye(len(v), dtype=bool))):
            e, x = Res[sel], dD[sel]
            row[f"{name}_{part}"] = dict(beta=float(e @ x / (x @ x)), rho=corr(e, x),
                                         r_half=float(np.linalg.norm(e - x / 2) / np.linalg.norm(e)),
                                         size_vs_teacher=float(np.linalg.norm(x) / np.linalg.norm(Yt[sel])))
    res[l] = row
    c, aa = row["corrected_off"], row["astra_off"]
    print(f"L{l}->{l + 1}: resid off {row['resid_rel_off']:.3f} diag {row['resid_rel_diag']:.3f} | corrected off beta {c['beta']:+.3f} rho {c['rho']:+.3f} "
          f"r_half {c['r_half']:.3f} size {c['size_vs_teacher']:.3f} | diag beta {row['corrected_diag']['beta']:+.3f} rho {row['corrected_diag']['rho']:+.3f} "
          f"| astra off beta {aa['beta']:+.3f} rho {aa['rho']:+.3f} r_half {aa['r_half']:.3f}", flush=True)
    del C, D, B, E, base, corrected, astra, G, Yt, Dc, Res
json.dump(res, open(args.out, "w"), indent=1)
print("wrote", args.out)
