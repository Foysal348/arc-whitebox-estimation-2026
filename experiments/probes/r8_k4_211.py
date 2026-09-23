"""Round 8 (Fable, 24 Sep): the EXACT leading (2,1,1) birth law of the post-ReLU kappa4, contracted directionally, against
the quartic kernel's residual (queue Q12 / GPT Sol R7 Q2.3 / Astra R7 appendix A).

Facts so far: the kernel's non-pair part (star + path + input edge, + Sol's K3 x C) has corr 0.73-0.83 with the true
non-pair part of the next kappa4 diagonal but is 1.2x (layer 4) -> 1.9x (layer 16) too small; the (3,1) sector is
negligible (this round); transported old star births are anti-correlated with the residual (r8_k4_memory.py): the
deficit is FRESH and sits in the (2,1,1)/(1,1,1,1) sectors.  Astra: the truncated-Hermite polynomial "does not fully
resum repeated-index (2,1,1) terms".  Here the (2,1,1) law is derived exactly at leading order.

For r doubled and b, c distinct: kappa4(h_r,h_r,h_b,h_c) = kappa3(B_r,h_b,h_c) - 2 Cov(h_r,h_b) Cov(h_r,h_c), with
B_r = (h_r - m_r)^2 - Var_G(h_r); the joint cumulant of (B_r, h_b, h_c) is the connected-diagram expansion in the
trivariate cumulants of (z_r, z_b, z_c) with vertex weights v_k = E[B_r He_k(x_r)]/s_r^k at r (v1 = 2m(1-Phi),
v2 = 2Phi - 2m phi/s) and d_k = E[h^(k)] at b, c.  Classes kept (each a hub form in the doubled index r or in a single b):
  G1  star at r:            (v2_r - 2 d1_r^2) a_br a_cr                       a = d1 C           [= beta_r a a, K3ac]
  G2  path r-b-c (+mirror):  v1_r C_rb d2_b C_bc d1_c
  H3  double edge r=b (+m.): (v3_r/2 - d1_r d2_r) C_rb^2 d2_b C_rc d1_c        (Mehler C^2/2 and -2 Cov Cov)
  G4  double edge r=b, b-c:  (1/2) v2_r C_rb^2 d3_b C_bc d1_c (+mirror)
  G5  double edge b=c, r-b:  (1/2) v1_r C_rb d3_b C_bc^2 d2_c (+mirror)
  G7  triangle:              v2_r d2_b d2_c C_rb C_rc C_bc                     (non-hypertree; n^3 per neuron: subset)
  K1  K_rrb C_rc:            (v3_r/2 - d1_r d2_r) K[r,b] d1_b C_rc d1_c (+m.)  (incl. -2 Cov Cov)
  K2  K_rrb C_bc:            (1/2) v2_r K[r,b] d2_b C_bc d1_c (+m.)
  K3  K_bbr C_rc:            (v2_r/2 - d1_r^2) K[b,r] d2_b C_rc d1_c (+m.)     (incl. -2 Cov Cov)  [= beta_r/2 ...]
  K4  K_bbr C_bc:            (1/2) v1_r K[b,r] d3_b C_bc d1_c (+m.)
  K5  K_bbc C_rb:            (1/2) v1_r C_rb d3_b K[b,c] d1_c (+m.)
  K6  K_bbc C_rc:            (1/2) v1_r C_rc d2_c K[b,c] d2_b (+m.)
(K[x,y] = kappa3(z_x,z_x,z_y); the all-distinct K_rbc classes need the source legs and are left out.)  The next
diagonal receives 6 sum_{r,b,c distinct} w_r^2 w_b w_c kappa4[r,r,b,c]; hub-r forms give
6 sum_r w_r^2 g_r [2 (L^T w)_r (M^T w)_r - 2 ((L o M)^T w^2)_r], hub-b forms 12 sum_b g_b w_b [(L^T w^2)_b (M^T w)_b -
((L o M)^T w^3)_b]: n x m products on the pre-activation matrices only.  Regressed (free and fixed coefficients,
unweighted and G4^2-weighted) on the kernel residual true non-pair - kernel - K3C at pre-activation layers 4, 8, 12,
15, 16, from the TRUE layer-l cumulants (teacher) and from the chain's own state (deployable).

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r8_k4_211.py
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
NTRI = int(os.environ.get("R8_NTRI", "96"))
GH_x, GH_w = np.polynomial.hermite_e.hermegauss(120); GH_w = GH_w / GH_w.sum()

def He(k, x):
    h0, h1 = np.ones_like(x), x
    if k == 0: return h0
    for j in range(1, k): h0, h1 = h1, x * h1 - j * h0
    return h1

def vertex_weights(mu, var):
    """d_k = E[h^(k)] (k=1..3) and v_k = E[B He_k(x)]/s^k (k=1..3) for h = relu(mu + s x), B = (h - m)^2 - Var_G(h)"""
    s = np.sqrt(var); a = mu / s; ph = pdf(a); P = Phi(a)
    d1, d2, d3 = P, ph / s, -a * ph / s ** 2
    z = mu[:, None] + s[:, None] * GH_x[None, :]; h = np.maximum(z, 0.0)
    m = (h * GH_w).sum(1); vg = ((h - m[:, None]) ** 2 * GH_w).sum(1)
    Bq = (h - m[:, None]) ** 2 - vg[:, None]
    v = {k: (Bq * He(k, GH_x)[None, :] * GH_w).sum(1) / s ** k for k in (1, 2, 3)}
    chk = {"v1_vs_2m(1-Phi)": float(np.max(np.abs(v[1] - 2 * m * (1 - P)) / (np.abs(v[1]) + 1e-12))),
           "v2_vs_2Phi-2m_phi/s": float(np.max(np.abs(v[2] - (2 * P - 2 * m * ph / s)) / (np.abs(v[2]) + 1e-12)))}
    return dict(d1=d1, d2=d2, d3=d3, v1=v[1], v2=v[2], v3=v[3], m=m), chk

def classes(C, K, vw):
    """leg matrices and hub weights per class; hub-r forms: (g, L, M, sym) with kappa4[r,r,b,c] = g_r (L[b,r] M[c,r] + M[b,r] L[c,r])
    (sym: g_r L[b,r] L[c,r]); hub-b forms: (g, L, M) with kappa4[r,r,b,c] = g_b L[r,b] M[c,b] (+ mirror b<->c)"""
    d1, d2, d3, v1, v2, v3 = (vw[k] for k in ("d1", "d2", "d3", "v1", "v2", "v3"))
    C2 = C * C; KT = K.T
    col = lambda vec, M: vec[:, None] * M            # scale rows: (vec o M)[x, y] = vec_x M[x,y]
    hub_r = {"G1": (v2 - 2 * d1 ** 2, col(d1, C), None),
             "H3": (v3 / 2 - d1 * d2, col(d2, C2), col(d1, C)),
             "K1": (v3 / 2 - d1 * d2, col(d1, KT), col(d1, C)),          # L[b,r] = K[r,b] d1_b
             "K3": (v2 / 2 - d1 ** 2, col(d2, K), col(d1, C))}           # L[b,r] = K[b,r] d2_b
    hub_b = {"G2": (d2, col(v1, C), col(d1, C)),                         # L[r,b] = C_rb v1_r, M[c,b] = C_cb d1_c
             "G4": (0.5 * d3, col(v2, C2), col(d1, C)),                  # L[r,b] = C_rb^2 v2_r
             "G5": (0.5 * d3, col(v1, C), col(d2, C2)),                  # M[c,b] = C_cb^2 d2_c
             "K2": (0.5 * d2, col(v2, K), col(d1, C)),                   # L[r,b] = K[r,b] v2_r
             "K4": (0.5 * d3, col(v1, KT), col(d1, C)),                  # L[r,b] = K[b,r] v1_r
             "K5": (0.5 * d3, col(v1, C), col(d1, KT)),                  # M[c,b] = K[b,c] d1_c
             "K6": (0.5 * d2, col(v1, C), col(d2, K))}                   # M[c,b] = K[c,b] d2_c
    return hub_r, hub_b

def contract(hub_r, hub_b, W):
    W2 = W * W; W3 = W2 * W; out = {}
    for k, (g, L, M) in hub_r.items():
        LW = L.T @ W
        if M is None: out[k] = 6 * np.einsum("ri,ri->i", W2 * g[:, None], LW * LW - (L * L).T @ W2)
        else:
            MW = M.T @ W; out[k] = 6 * np.einsum("ri,ri->i", W2 * g[:, None], 2 * LW * MW - 2 * ((L * M).T @ W2))
    for k, (g, L, M) in hub_b.items():
        out[k] = 12 * np.einsum("bi,bi->i", W * g[:, None], (L.T @ W2) * (M.T @ W) - (L * M).T @ W3)
    return out

def triangle(C, vw, W, cols):
    """G7 on a subset of next-layer neurons: 6 sum_r w_r^2 v2_r sum_{b,c distinct, != r} (w_b d2_b C_rb) C_bc (w_c d2_c C_rc)"""
    d2, v2 = vw["d2"], vw["v2"]; res = np.zeros(len(cols))
    for t, i in enumerate(cols):
        w = W[:, i]; X = C * (w * d2)[:, None]                      # X[b, r] = w_b d2_b C_br
        Y = C @ X                                                   # Y[c, r] = sum_b C_cb X[b,r]
        full = np.einsum("cr,cr->r", X, Y)                          # sum_{b,c} X_br C_bc X_cr  (b = c term: C_bb X_br^2 -> subtract)
        full -= np.einsum("br,br->r", X * X, np.diag(C)[:, None] * np.ones_like(X))
        res[t] = 6 * np.sum(w * w * v2 * full)
    return res

def ufit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); return coef, float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y))
def wfit(X, y, w):
    sw = np.sqrt(w); coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None); return coef, float(1 - np.sum(w * (y - X @ coef) ** 2) / np.sum(w * y * y))
def ev(v, ref, w):
    e = v - ref; return {"corr": float(np.corrcoef(v, ref)[0, 1]), "slope": float(np.dot(v, ref) / np.dot(v, v)), "errG4": float(np.sqrt(np.sum(w * e * e) / np.sum(w * ref * ref)))}

out = []
for Lt in TARGETS:
    t0 = time.time(); l = Lt - 1                                    # post-ReLU layer l feeds W[l] -> pre-activation Lt
    t4, Ptrue, Pchain, pred, K3C, g4, w, a1 = (f64(D[f"L{Lt}_{k}"]) for k in ("t4", "Ptrue", "Pchain", "pred", "K3C", "g4", "w", "a1"))
    target = t4 - Ptrue; resid = target - pred - K3C; m = np.abs(a1) < 1.5
    W = Wall[l]
    T = np.load(os.path.join(PRE, f"layer{l}_seed0.npz")); c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz"))
    rec = {"layer": Lt, "rms": {"target": float(np.sqrt(np.mean(target ** 2))), "resid": float(np.sqrt(np.mean(resid ** 2)))}}
    for state in ("true", "chain"):
        if state == "true":
            Cf = f64(T["C"]); mu = f64(T["mean"]); var = np.diag(Cf).copy(); K = f64(T["Y21"]).copy()
        else:
            mu, var = f64(c["mu"]), f64(c["var"]); Cf = f64(c["C_off"]).copy(); np.fill_diagonal(Cf, var); K = f64(c["D21"]).copy()
        C = Cf.copy(); np.fill_diagonal(C, 0.0); np.fill_diagonal(K, 0.0)
        vw, chk = vertex_weights(mu, var)
        hr, hb = classes(C, K, vw); terms = contract(hr, hb, W)
        names = list(terms); X = np.stack([terms[k] for k in names], 1); tot = X.sum(1)
        cu, r2u = ufit(X, resid); cw, r2w = wfit(X, resid, w)
        c1, r1 = ufit(tot[:, None], resid); c1w, r1w = wfit(tot[:, None], resid, w)
        fixed_r2 = float(1 - np.sum((resid - tot) ** 2) / np.sum(resid ** 2)); fixed_r2w = float(1 - np.sum(w * (resid - tot) ** 2) / np.sum(w * resid ** 2))
        cj, r2j = ufit(np.stack([pred, K3C, tot], 1), target); cjw, r2jw = wfit(np.stack([pred, K3C, tot], 1), target, w)
        ck, r2k = ufit(np.stack([pred, K3C], 1), target); ckw, r2kw = wfit(np.stack([pred, K3C], 1), target, w)
        combos = {"chain_regen": g4, "deploy_S1.25+K3C": Pchain + 1.25 * (pred + K3C), "deploy_S1+K3C+211": Pchain + pred + K3C + tot,
                  "deploy_S1.25+K3C+211": Pchain + 1.25 * (pred + K3C) + tot, "deploy_fit3": Pchain + cj[0] * pred + cj[1] * K3C + cj[2] * tot}
        S = {"vertex_check": chk, "term_rms": {k: float(np.sqrt(np.mean(v ** 2))) for k, v in terms.items()}, "total_rms": float(np.sqrt(np.mean(tot ** 2))),
             "single": {k: {"corr": float(np.corrcoef(v, resid)[0, 1]), "slope": float(np.dot(v, resid) / np.dot(v, v))} for k, v in terms.items()},
             "joint_free": {"unweighted": {"r2": r2u, "coef": dict(zip(names, map(float, cu)))}, "G4weighted": {"r2": r2w, "coef": dict(zip(names, map(float, cw)))}},
             "total_one_coef": {"unweighted": {"coef": float(c1[0]), "r2": r1}, "G4weighted": {"coef": float(c1w[0]), "r2": r1w}, "corr_all": float(np.corrcoef(tot, resid)[0, 1]), "corr_decisive": float(np.corrcoef(tot[m], resid[m])[0, 1])},
             "total_fixed_coef1_r2": {"unweighted": fixed_r2, "G4weighted": fixed_r2w},
             "target_on_kernel_K3C": {"r2": r2k, "coef": list(map(float, ck)), "r2w": r2kw, "coefw": list(map(float, ckw))},
             "target_on_kernel_K3C_211": {"r2": r2j, "coef": list(map(float, cj)), "r2w": r2jw, "coefw": list(map(float, cjw))},
             "combos": {k: {**ev(v, t4, w), "nonpair_corr_decisive": float(np.corrcoef((v - Pchain)[m], target[m])[0, 1]) if k != "chain_regen" else None} for k, v in combos.items()}}
        if state == "true" and NTRI > 0:
            rng = np.random.default_rng(0); cols = np.sort(rng.choice(n, NTRI, replace=False))
            tri = triangle(C, vw, W, cols); r_sub = resid[cols]; t_sub = tot[cols]
            S["triangle_subset"] = {"rms": float(np.sqrt(np.mean(tri ** 2))), "corr_resid": float(np.corrcoef(tri, r_sub)[0, 1]), "slope": float(np.dot(tri, r_sub) / np.dot(tri, tri)),
                                    "corr_resid_minus_total": float(np.corrcoef(tri, r_sub - t_sub)[0, 1])}
        rec[state] = S
        print(f"\nL{Lt} [{state} state] ({time.time()-t0:.0f}s): target rms {rec['rms']['target']:.2e}, kernel residual rms {rec['rms']['resid']:.2e} | vertex check {chk}", flush=True)
        print("   class rms: " + ", ".join(f"{k} {v:.1e}" for k, v in S["term_rms"].items()) + f" | total {S['total_rms']:.2e}", flush=True)
        print("   single corr/slope on residual: " + ", ".join(f"{k} {v['corr']:+.2f}/{v['slope']:+.2f}" for k, v in S["single"].items()), flush=True)
        jf = S["joint_free"]; print(f"   joint free: R2 {jf['unweighted']['r2']:.3f} (G4-w {jf['G4weighted']['r2']:.3f}); coefs " + ", ".join(f"{k} {v:+.2f}" for k, v in jf["unweighted"]["coef"].items()), flush=True)
        tc = S["total_one_coef"]; print(f"   total (theory coef 1): one-coef fit {tc['unweighted']['coef']:+.2f} R2 {tc['unweighted']['r2']:.3f} (G4-w {tc['G4weighted']['coef']:+.2f} R2 {tc['G4weighted']['r2']:.3f}); fixed-1 R2 {S['total_fixed_coef1_r2']['unweighted']:.3f} (G4-w {S['total_fixed_coef1_r2']['G4weighted']:.3f}); corr all {tc['corr_all']:.3f} decisive {tc['corr_decisive']:.3f}", flush=True)
        tk, tj = S["target_on_kernel_K3C"], S["target_on_kernel_K3C_211"]
        print(f"   target on [kernel,K3C]: R2 {tk['r2']:.3f} coef {tk['coef'][0]:.2f},{tk['coef'][1]:.2f} -> +211: R2 {tj['r2']:.3f} coef {tj['coef'][0]:.2f},{tj['coef'][1]:.2f},{tj['coef'][2]:.2f} | G4-w R2 {tk['r2w']:.3f} -> {tj['r2w']:.3f} (coef {tj['coefw'][0]:.2f},{tj['coefw'][1]:.2f},{tj['coefw'][2]:.2f})", flush=True)
        for k, v in S["combos"].items():
            print(f"   {k:24s}: errG4 {v['errG4']:.3f} corr {v['corr']:.3f} slope {v['slope']:.3f}" + ("" if v["nonpair_corr_decisive"] is None else f" | non-pair corr decisive {v['nonpair_corr_decisive']:.3f}"), flush=True)
        if "triangle_subset" in S:
            ts = S["triangle_subset"]; print(f"   triangle (subset {NTRI}): rms {ts['rms']:.1e}, corr with residual {ts['corr_resid']:+.2f} slope {ts['slope']:+.2f}, with residual-minus-total {ts['corr_resid_minus_total']:+.2f}", flush=True)
    out.append(rec)
json.dump(out, open("../results/summaries/r8_k4_211.json", "w"), indent=1, default=float)
