"""Round 8 (Fable, 24 Sep): the kappa4 kernel's missing (1,1,1,1) content is the K3 x C '3+2' term with the ALL-DISTINCT
pre-activation kappa3 - the chain's own sources (Astra R7 section 5 / queue Q8) - not Sol's repeated-index slice
approximation t_rep, which by construction lands in the (2,1,1)/(3,1) sectors.

Evidence chain: the kernel's non-pair prediction is 1.25x (layer 4) -> 1.9x (layer 16) too small; the (3,1) sector of the
true non-pair part is negligible (3e-5 of 1e-3); the exact leading (2,1,1) classes total 1.4e-4 (r8_k4_211.py);
transported old star births are anti-correlated (r8_k4_memory.py).  So the deficit is FRESH and (1,1,1,1).  For four
distinct outputs the only leading term not in the kernel is the hypertree K_jkl C_jm with all of j, k, l distinct:
    delta1111(w) = 12 sum_j s_j t_j,   s = (d2 o w) o (C_off x),  x = d1 o w,   t_j = sum_{k != l, k,l != j} K_jkl x_k x_l,
and the chain's sources give t exactly: for a hub block Sym(X x P x Y) (the chain's Sym = (1/6) sum of permutations),
    t = (1/3) [ X (P^T x o Y^T x) + P (X^T x o Y^T x) + Y (X^T x o P^T x) ],
summed over the three blocks (X1,P,Y1), (M,P,P), (X3,P,Y3) of every live source plus the newborn (P = I).  The repeated-
index part of the same tensor is t_rep = D21^T (x o x) + 2 x o (D21 x) - 2 D3 o (x o x) with the chain's own D21, D3
(the source slices reproduce D21 to R2 1.00000, r7_srcgain.py), so t_dist = t_full - t_rep.  Regressed on the kernel
residual (true non-pair - kernel - K3C_rep) at pre-activation layers 4, 8, 12, 15, 16; split by source age.
Cost of a deployment: P^T x and A^T x ARE the transported legs (free); the extra is the products with the left legs,
~3 (m/n) u per block and source-layer (Astra: ~54 u at m = 350 with the confined ranks).

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r8_k4_1111.py
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
n = Wall.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64)
D = np.load(os.path.join(CAP, "astra7_k4_mlp0.npz"))
TARGETS = [int(x) for x in os.environ.get("R8_LAYERS", "4,8,12,15,16").split(",")]

def t_of_block(X, P, Y, Xc, PXc=None):
    """t = sum_{kl} Sym(X x P x Y)_{jkl} x_k x_l for every column x of Xc (n x m)"""
    XX = X.T @ Xc; PX = P.T @ Xc if PXc is None else PXc; YX = Y.T @ Xc
    return (X @ (PX * YX) + P @ (XX * YX) + Y @ (XX * PX)) / 3.0, PX

def blocks_of_source(c, s, rres, rfb):
    A = f64(c["A_st"][s]); P = f64(c["P_st"][s]); Z = f64(c["Z_st"][s]); L = f64(c["L_st"][s])
    w2b, sb, eb, c1, c2 = (f64(c[k][s]) for k in ("w2b", "s_b", "e_b", "c1_b", "c2_b"))
    M = P * sb[None, :] + 3.0 * A * eb[None, :] + Z @ L.T
    if rfb > 0:
        Zf = f64(c["Zf_st"][s]); Xt = Zf[:, :rfb] @ f64(c["R1T_st"][s]).T; Yt = Zf[:, rfb:] @ f64(c["R2T_st"][s]).T
    else: Xt = Yt = 0.0
    X1 = 3.0 * A + Xt; Y1 = A * w2b[None, :] + Yt
    X3 = A * c1[None, :] + P * c2[None, :]; y = Z[:, rres + 1]; Y3 = np.repeat(y[:, None], n, axis=1)
    return [(X1, P, Y1), (M, P, P), (X3, P, Y3)]

def ufit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); return coef, float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
def wfit(X, y, w):
    sw = np.sqrt(w); coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None); return coef, float(1 - np.sum(w * (y - X @ coef) ** 2) / np.sum(w * y * y))
def ev(v, ref, w):
    e = v - ref; return {"corr": float(np.corrcoef(v, ref)[0, 1]), "slope": float(np.dot(v, ref) / np.dot(v, v)), "errG4": float(np.sqrt(np.sum(w * e * e) / np.sum(w * ref * ref)))}

out = []
for Lt in TARGETS:
    t0 = time.time(); l = Lt - 1
    t4, Ptrue, Pchain, pred, K3C, g4, w, a1 = (f64(D[f"L{Lt}_{k}"]) for k in ("t4", "Ptrue", "Pchain", "pred", "K3C", "g4", "w", "a1"))
    target = t4 - Ptrue; resid = target - pred - K3C; m = np.abs(a1) < 1.5
    W = Wall[l]; c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); rres = int(c["rres"]); rfb = int(c["rfb"]); k = c["A_st"].shape[0]
    mu, var = f64(c["mu"]), f64(c["var"]); s_ = np.sqrt(var); a_ = mu / s_; ph = pdf(a_)
    d1, d2, d3 = Phi(a_), ph / s_, -a_ * ph / s_ ** 2
    Co = f64(c["C_off"]); D21 = f64(c["D21"]).copy(); np.fill_diagonal(D21, 0.0); D3 = f64(c["D3"])
    Xc = d1[:, None] * W                                        # x = d1 o w for every next-layer column
    Ss = (d2[:, None] * W) * (Co @ Xc)                          # s = (d2 o w) o (C_off x)
    t_rep = D21.T @ (Xc * Xc) + 2 * Xc * (D21 @ Xc) - 2 * D3[:, None] * (Xc * Xc)
    K3C_rep = 12 * np.sum(Ss * t_rep, 0)
    rec = {"layer": Lt, "n_sources": k, "check_K3C_rep_vs_dump": float(np.max(np.abs(K3C_rep - K3C)) / np.max(np.abs(K3C)))}
    # ---- t from every source (by age) and from the newborn ----
    t_by_age = {}; trep_by_age = {}
    def trep_of(D21s, D3s):
        return D21s.T @ (Xc * Xc) + 2 * Xc * (D21s @ Xc) - 2 * D3s[:, None] * (Xc * Xc)
    for s in range(k):
        age = k - s; ts = np.zeros_like(Xc); PXc = None; sl21 = np.zeros((n, n)); sl3 = np.zeros(n)
        for (X, P, Y) in blocks_of_source(c, s, rres, rfb):
            tb, PXc = t_of_block(X, P, Y, Xc, PXc); ts += tb
            sl21 += ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0; sl3 += np.sum(X * P * Y, axis=1)   # the source's own (2,1) slice and diagonal
        np.fill_diagonal(sl21, 0.0)
        t_by_age[age] = ts; trep_by_age[age] = trep_of(sl21, sl3)
    w1c, w2c, w3c = (f64(c["W_all"])[:, i] for i in (4, 6, 8))
    a_b = w1c[:, None] * Co; Xt = 1.5 * w2c[:, None] * D21; Yt = 0.5 * w1c[:, None] * D21.T * w3c[None, :]
    X1 = 3.0 * a_b + Xt; Y1 = a_b * w2c[None, :] + Yt
    X1X = X1.T @ Xc; Y1X = Y1.T @ Xc
    t_by_age[0] = (X1 @ (Xc * Y1X) + (X1X * Y1X) + Y1 @ (X1X * Xc)) / 3.0          # P = I
    D21n = (np.diag(X1)[:, None] * Y1.T + X1 * Y1 + np.diag(Y1)[:, None] * X1.T) / 3.0; np.fill_diagonal(D21n, 0.0); D3n = np.diag(X1) * np.diag(Y1)
    trep_by_age[0] = trep_of(D21n, D3n)
    t_full = sum(t_by_age.values()); t_dist = t_full - t_rep
    trep_sum = sum(trep_by_age.values())
    rec["check_sum_source_trep_vs_chain_trep"] = float(np.sqrt(np.sum((trep_sum - t_rep) ** 2) / np.sum(t_rep ** 2)))
    tdist_by_age = {a: t_by_age[a] - trep_by_age[a] for a in t_by_age}
    K3Cd_age = {a: 12 * np.sum(Ss * t, 0) for a, t in tdist_by_age.items()}
    K3C_dist_old = sum(v for a, v in K3Cd_age.items() if a >= 1); K3C_dist_y4 = sum(v for a, v in K3Cd_age.items() if 1 <= a <= 4); K3C_dist_y2 = sum(v for a, v in K3Cd_age.items() if 1 <= a <= 2)
    # sanity: the repeated-index part of t_full must equal t_rep -> check on the diagonal-free structure via a random column: compare
    # t_full's (k = l) part reconstructed from D21 (the sources' own slices) - implicit in the source/chain D21 identity (r7_srcgain).
    K3C_dist = 12 * np.sum(Ss * t_dist, 0)
    K3C_age = {age: 12 * np.sum(Ss * t, 0) for age, t in t_by_age.items()}          # includes each age's repeated-index part
    young = sum(v for a, v in K3C_age.items() if a <= 4); old = sum(v for a, v in K3C_age.items() if a > 4)
    rec["rms"] = {"target": float(np.sqrt(np.mean(target ** 2))), "resid": float(np.sqrt(np.mean(resid ** 2))), "K3C_rep": float(np.sqrt(np.mean(K3C_rep ** 2))),
                  "K3C_dist": float(np.sqrt(np.mean(K3C_dist ** 2))), "K3C_full": float(np.sqrt(np.mean((K3C_dist + K3C_rep) ** 2))),
                  **{f"K3C_age{a}(with its rep part)": float(np.sqrt(np.mean(v ** 2))) for a, v in sorted(K3C_age.items())}}
    cu, r2u = ufit(K3C_dist[:, None], resid); cw, r2w = wfit(K3C_dist[:, None], resid, w)
    rec["resid_on_K3C_dist"] = {"coef": float(cu[0]), "r2": r2u, "coef_G4w": float(cw[0]), "r2_G4w": r2w, "corr_all": float(np.corrcoef(K3C_dist, resid)[0, 1]), "corr_decisive": float(np.corrcoef(K3C_dist[m], resid[m])[0, 1]),
                               "fixed1_r2": float(1 - np.sum((resid - K3C_dist) ** 2) / np.sum(resid ** 2)), "fixed1_r2_G4w": float(1 - np.sum(w * (resid - K3C_dist) ** 2) / np.sum(w * resid ** 2))}
    ck, r2k = ufit(np.stack([pred, K3C_rep], 1), target); cj, r2j = ufit(np.stack([pred, K3C_rep, K3C_dist], 1), target)
    ckw, r2kw = wfit(np.stack([pred, K3C_rep], 1), target, w); cjw, r2jw = wfit(np.stack([pred, K3C_rep, K3C_dist], 1), target, w)
    rec["target_on_kernel_K3Crep"] = {"r2": r2k, "coef": list(map(float, ck)), "r2_G4w": r2kw, "coef_G4w": list(map(float, ckw))}
    rec["target_on_kernel_K3Crep_K3Cdist"] = {"r2": r2j, "coef": list(map(float, cj)), "r2_G4w": r2jw, "coef_G4w": list(map(float, cjw))}
    # age split of the all-distinct part: regress the residual on young (ages 0-4, minus their share of t_rep is not separable; use full-age terms jointly)
    Xa = np.stack([K3C_age[a] for a in sorted(K3C_age)], 1); ca, r2a = ufit(Xa, resid + K3C_rep)       # target: residual + K3C_rep = true non-pair - kernel
    rec["nonpair_minus_kernel_on_ages"] = {"r2": r2a, "coef": {str(a): float(v) for a, v in zip(sorted(K3C_age), ca)}}
    Xd = np.stack([K3Cd_age[a] for a in sorted(K3Cd_age)], 1); cd_, r2d = ufit(Xd, resid); cdw, r2dw = wfit(Xd, resid, w)
    rec["resid_on_alldistinct_by_age"] = {"r2": r2d, "coef": {str(a): float(v) for a, v in zip(sorted(K3Cd_age), cd_)}, "r2_G4w": r2dw, "coef_G4w": {str(a): float(v) for a, v in zip(sorted(K3Cd_age), cdw)}}
    rec["alldistinct_rms_by_age"] = {str(a): float(np.sqrt(np.mean(v ** 2))) for a, v in sorted(K3Cd_age.items())}
    for nm, v in (("old(ages>=1)", K3C_dist_old), ("young(1-4)", K3C_dist_y4), ("young(1-2)", K3C_dist_y2)):
        cu_, r2u_ = ufit(v[:, None], resid); cw_, r2w_ = wfit(v[:, None], resid, w)
        rec[f"resid_on_dist_{nm}"] = {"rms": float(np.sqrt(np.mean(v ** 2))), "coef": float(cu_[0]), "r2": r2u_, "coef_G4w": float(cw_[0]), "r2_G4w": r2w_, "corr_decisive": float(np.corrcoef(v[m], resid[m])[0, 1]),
                                      "fixed1_r2_G4w": float(1 - np.sum(w * (resid - v) ** 2) / np.sum(w * resid ** 2))}
    combos = {"chain_regen": g4, "deploy_S1.25+K3Crep": Pchain + 1.25 * (pred + K3C_rep), "deploy_S1+K3Cfull": Pchain + pred + K3C_rep + K3C_dist,
              "deploy_S1.25+K3Cfull": Pchain + 1.25 * (pred + K3C_rep + K3C_dist), "deploy_S1+K3Crep+young_dist": Pchain + pred + K3C_rep + (young - 12 * np.sum(Ss * t_rep, 0) * 0),
              "deploy_fit3": Pchain + cj[0] * pred + cj[1] * K3C_rep + cj[2] * K3C_dist, "oracle_pairs+S1+K3Cfull": Ptrue + pred + K3C_rep + K3C_dist,
              "deploy_S1+K3Crep+dist_old": Pchain + pred + K3C_rep + K3C_dist_old, "deploy_S1+K3Crep+dist_y4": Pchain + pred + K3C_rep + K3C_dist_y4,
              "deploy_S1+K3Crep+dist_y2": Pchain + pred + K3C_rep + K3C_dist_y2, "deploy_S1.25+K3Crep+dist_y4": Pchain + 1.25 * (pred + K3C_rep) + K3C_dist_y4,
              "deploy_S1.25+K3Crep+1.25dist_y4": Pchain + 1.25 * (pred + K3C_rep + K3C_dist_y4)}
    combos.pop("deploy_S1+K3Crep+young_dist")
    rec["combos"] = {kk: {**ev(v, t4, w), "nonpair_corr": float(np.corrcoef(v - (Ptrue if kk.startswith("oracle") else Pchain), target)[0, 1]) if kk != "chain_regen" else None,
                          "nonpair_corr_decisive": float(np.corrcoef((v - (Ptrue if kk.startswith("oracle") else Pchain))[m], target[m])[0, 1]) if kk != "chain_regen" else None,
                          "nonpair_slope": float(np.dot(v - (Ptrue if kk.startswith("oracle") else Pchain), target) / np.dot(v - (Ptrue if kk.startswith("oracle") else Pchain), v - (Ptrue if kk.startswith("oracle") else Pchain))) if kk != "chain_regen" else None}
                     for kk, v in combos.items()}
    out.append(rec)
    if os.environ.get("R8_SAVE"):
        np.savez_compressed(os.path.join(CAP, f"r8_k4_1111_L{Lt}.npz"), K3C_rep=K3C_rep, K3C_dist_old=K3C_dist_old, K3C_dist_y4=K3C_dist_y4, K3C_dist_y2=K3C_dist_y2, pred=pred, target=target, t4=t4, Pchain=Pchain, w=w, a1=a1,
                            **{f"dist_age{a}": v for a, v in K3Cd_age.items()})
    r = rec["rms"]
    print(f"\nL{Lt} ({time.time()-t0:.0f}s, {k} sources; K3C_rep check {rec['check_K3C_rep_vs_dump']:.1e}): non-pair target {r['target']:.2e}, kernel residual {r['resid']:.2e} | K3C: rep {r['K3C_rep']:.2e}, ALL-DISTINCT {r['K3C_dist']:.2e}, full {r['K3C_full']:.2e}", flush=True)
    print("   K3C by source age (each with its own repeated part): " + ", ".join(f"{a}: {v:.1e}" for a, v in sorted(K3C_age.items()) if False) + ", ".join(f"age{a} {rec['rms'][f'K3C_age{a}(with its rep part)']:.1e}" for a in sorted(K3C_age)), flush=True)
    rd = rec["resid_on_K3C_dist"]
    print(f"   kernel residual on K3C_dist: coef {rd['coef']:+.2f} R2 {rd['r2']:.3f} (G4-w coef {rd['coef_G4w']:+.2f} R2 {rd['r2_G4w']:.3f}); fixed coef 1: R2 {rd['fixed1_r2']:.3f} (G4-w {rd['fixed1_r2_G4w']:.3f}); corr all {rd['corr_all']:.3f}, decisive {rd['corr_decisive']:.3f}", flush=True)
    tk, tj = rec["target_on_kernel_K3Crep"], rec["target_on_kernel_K3Crep_K3Cdist"]
    print(f"   target on [kernel, K3C_rep]: R2 {tk['r2']:.3f} coef {tk['coef'][0]:.2f},{tk['coef'][1]:.2f} -> + K3C_dist: R2 {tj['r2']:.3f} coef {tj['coef'][0]:.2f},{tj['coef'][1]:.2f},{tj['coef'][2]:.2f} | G4-w: R2 {tk['r2_G4w']:.3f} -> {tj['r2_G4w']:.3f} (coef {tj['coef_G4w'][0]:.2f},{tj['coef_G4w'][1]:.2f},{tj['coef_G4w'][2]:.2f})", flush=True)
    na = rec["nonpair_minus_kernel_on_ages"]; print(f"   (non-pair - kernel) on per-age K3C: R2 {na['r2']:.3f}, coefs " + ", ".join(f"age{a} {v:+.2f}" for a, v in na["coef"].items()), flush=True)
    print(f"   sum of per-source repeated parts vs chain t_rep: rel diff {rec['check_sum_source_trep_vs_chain_trep']:.1e}", flush=True)
    print("   ALL-DISTINCT K3C rms by age: " + ", ".join(f"age{a} {v:.1e}" for a, v in rec["alldistinct_rms_by_age"].items()), flush=True)
    rd2 = rec["resid_on_alldistinct_by_age"]; print(f"   residual on all-distinct per age: R2 {rd2['r2']:.3f} (G4-w {rd2['r2_G4w']:.3f}); coefs " + ", ".join(f"age{a} {v:+.2f}" for a, v in rd2["coef"].items()) + " | G4-w: " + ", ".join(f"{v:+.2f}" for v in rd2["coef_G4w"].values()), flush=True)
    for nm in ("old(ages>=1)", "young(1-4)", "young(1-2)"):
        q = rec[f"resid_on_dist_{nm}"]; print(f"   residual on dist {nm:12s}: rms {q['rms']:.2e} coef {q['coef']:+.2f} R2 {q['r2']:.3f} (G4-w coef {q['coef_G4w']:+.2f} R2 {q['r2_G4w']:.3f}; fixed-1 G4-w R2 {q['fixed1_r2_G4w']:.3f}); corr decisive {q['corr_decisive']:.3f}", flush=True)
    for kk, v in rec["combos"].items():
        extra = "" if v["nonpair_corr"] is None else f" | non-pair corr {v['nonpair_corr']:.3f} (decisive {v['nonpair_corr_decisive']:.3f}) slope {v['nonpair_slope']:.2f}"
        print(f"   {kk:28s}: errG4 {v['errG4']:.3f} corr {v['corr']:.3f} slope {v['slope']:.3f}{extra}", flush=True)
json.dump(out, open("../results/summaries/r8_k4_1111.json", "w"), indent=1, default=float)
