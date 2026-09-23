"""Round 8 (Fable, 23 Sep): does the quartic kappa4 kernel's depth deficit (1.3x at layer 4 -> 2.2x at layer 16 too small)
come from MEMORY - the multi-index kappa4 content of OLDER births transported linearly - rather than from missing fresh
diagram classes?

Astra's kernel F0 (probe_astra7_k4.py) is the FRESH quartic response of one ReLU: input-kappa4 edge + Gaussian star +
Gaussian path (+ GPT Sol's K3 x C term).  Its non-pair part correlates 0.63-0.78 with the true non-pair part of the next
kappa4 diagonal but is too small by a factor that grows with depth.  A fresh-only birth misses what the kappa3 chain
keeps for kappa3: the births of earlier layers, transported.  The transport gain of a fourth-order tensor is
~ 2^2 E[Phi^2]^2 ~ 0.8 per layer, so the steady-state multi-index content is ~1.7x the fresh birth - the observed
deficit.  The star births (K3ac (d): k(h_r,h_r,h_k,h_m) = beta_r a_kr a_mr, k(h_r,h_r,h_r,h_k) = gamma_r a_kr, and the
all-distinct star w3_r a_ar a_br a_cr = the kernel's own f2 at age 0) share the legs a = diag(Phi) C_off and P = T of
the kappa3 sources, so their transported diagonal contribution at any later layer is ELEMENTWISE on legs the chain
already holds dense for the young tier: 6 beta (P^2 At^2) + 4 gamma (P^3 At) + 4 w3 (P At^3): O(n^2) per source-layer.

This probe transports the star births of every age with exact dense transport from the chain's own captured state
(data/dumps/r7_cap_L*.npz) and regresses the kernel's residual (true non-pair - kernel non-pair - K3C) at the next
pre-activation layers 4, 8, 12, 15, 16 on the OLD stars (ages >= 1), the FRESH (2,1,1)/(3,1) stars (age 0; the kernel
may or may not already contain them), split by sector; reports the G4^2-weighted error and the decisive-neuron
correlation of the deployable combinations.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r8_k4_memory.py
"""
import os, sys, json, time
import numpy as np
from math import erf
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "../data/kaggle_out/kaggle_out_k4post/k4post"; CAP = "../data/dumps"
Wall = np.load(os.path.join(PRE, "weights_seed0.npz"))["W"].astype(np.float64)
n = Wall.shape[1]; L = Wall.shape[0]
Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / 2 ** 0.5))); pdf = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
f64 = lambda x: np.asarray(x, np.float64)
D = np.load(os.path.join(CAP, "astra7_k4_mlp0.npz"))
TARGETS = [4, 8, 12, 15, 16]                              # 1-based pre-activation layers with a kernel record
MAXAGE = int(os.environ.get("R8_MAXAGE", "20"))

# ---- birth coefficients from the chain's own pre-activation state at every layer (r7_cap_L{l}: the state before the ReLU of layer l) ----
B = {}
for li in range(L - 1):                                   # chain index li = 1-based layer li+1
    c = np.load(os.path.join(CAP, f"r7_cap_L{li+1}.npz"))
    mu, var = f64(c["mu"]), f64(c["var"]); s = np.sqrt(var); a_ = mu / s; P = Phi(a_); ph = pdf(a_)
    m = s * ph + mu * P; Eh2 = (mu * mu + var) * P + mu * s * ph
    beta = 2 * P * (1 - P) - 2 * m * ph / s
    gamma = 3 * (1 - P) * (Eh2 - 2 * m * m)
    w3 = -a_ * ph / s ** 2
    if "C_off" in c.files: Coff = f64(c["C_off"])
    else:
        W0 = Wall[0]; Coff = W0.T @ W0; np.fill_diagonal(Coff, 0.0)
    B[li] = dict(beta=beta, gamma=gamma, w3=w3, a=P[:, None] * Coff, Phi=P)
# ---- transported star diagonals, per (birth, target, sector) ----
# target pre-activation layer lt+1 (1-based) receives births lb = 0..lt-1 (post-ReLU of layer lb+1); age = lt-1-lb
acc = {t: {} for t in TARGETS}
t0 = time.time()
for lb in range(L - 1):
    T = Wall[lb + 1].T.copy(); At = T @ B[lb]["a"]
    for lt in range(lb + 1, L):
        if lt > lb + 1:
            Dg = B[lt - 1]["Phi"]; T = Wall[lt].T @ (Dg[:, None] * T); At = Wall[lt].T @ (Dg[:, None] * At)
        if lt + 1 not in TARGETS: continue
        age = lt - 1 - lb
        if age > MAXAGE: continue
        b = B[lb]
        rec = {"s211": 6 * ((T * T) * (At * At)) @ b["beta"], "s31": 4 * ((T ** 3) * At) @ b["gamma"], "s1111": 4 * ((At ** 3) * T) @ b["w3"]}
        acc[lt + 1][age] = rec
    print(f"  birth {lb} transported ({time.time()-t0:.0f}s)", flush=True)

def ev(v, ref, w):
    e = v - ref
    return {"corr": float(np.corrcoef(v, ref)[0, 1]), "slope": float(np.dot(v, ref) / np.dot(v, v)), "errG4": float(np.sqrt(np.sum(w * e * e) / np.sum(w * ref * ref)))}
def wfit(X, y, w):
    sw = np.sqrt(w); coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r2 = float(1 - np.sum(w * (y - X @ coef) ** 2) / np.sum(w * y * y)); return coef, r2
def ufit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = float(1 - np.sum((y - X @ coef) ** 2) / np.sum(y * y)); return coef, r2

out = []
for Lt in TARGETS:
    t4, Ptrue, Pchain, pred, K3C, g4, w, a1 = (f64(D[f"L{Lt}_{k}"]) for k in ("t4", "Ptrue", "Pchain", "pred", "K3C", "g4", "w", "a1"))
    target = t4 - Ptrue                                   # true non-pair part: (3,1) + (2,1,1) + (1,1,1,1)
    ages = sorted(acc[Lt]); fresh = acc[Lt][0]
    old = {k: sum(acc[Lt][a][k] for a in ages if a >= 1) for k in ("s211", "s31", "s1111")}
    young = {k: sum(acc[Lt][a][k] for a in ages if 1 <= a <= 4) for k in ("s211", "s31", "s1111")}
    oldsum = old["s211"] + old["s31"] + old["s1111"]; youngsum = young["s211"] + young["s31"] + young["s1111"]
    m = np.abs(a1) < 1.5
    rec = {"layer": Lt, "n_ages": len(ages), "rms": {"target": float(np.sqrt(np.mean(target ** 2))), "pred": float(np.sqrt(np.mean(pred ** 2))), "K3C": float(np.sqrt(np.mean(K3C ** 2))),
           **{f"fresh_{k}": float(np.sqrt(np.mean(v ** 2))) for k, v in fresh.items()}, **{f"old_{k}": float(np.sqrt(np.mean(v ** 2))) for k, v in old.items()}}}
    # per-age rms of the total star (how fast memory decays)
    rec["age_rms"] = {a: float(np.sqrt(np.mean((acc[Lt][a]["s211"] + acc[Lt][a]["s31"] + acc[Lt][a]["s1111"]) ** 2))) for a in ages}
    # fresh all-distinct star == kernel's f2 (sanity)
    # regressions of the kernel residual
    resid = target - pred - K3C
    names = ["old_s1111", "old_s211", "old_s31", "fresh_s211", "fresh_s31"]
    X = np.stack([old["s1111"], old["s211"], old["s31"], fresh["s211"], fresh["s31"]], 1)
    cu, r2u = ufit(X, resid); cw, r2w = wfit(X, resid, w)
    rec["resid_on_stars"] = {"unweighted": {"r2": r2u, "coef": dict(zip(names, map(float, cu)))}, "G4weighted": {"r2": r2w, "coef": dict(zip(names, map(float, cw)))}}
    cu2, r2u2 = ufit(oldsum[:, None], resid); cw2, r2w2 = wfit(oldsum[:, None], resid, w)
    rec["resid_on_oldsum"] = {"unweighted": {"r2": r2u2, "coef": float(cu2[0])}, "G4weighted": {"r2": r2w2, "coef": float(cw2[0])}, "corr_all": float(np.corrcoef(oldsum, resid)[0, 1]), "corr_decisive": float(np.corrcoef(oldsum[m], resid[m])[0, 1])}
    # joint fit of the target on kernel + K3C + old stars (free coefficients) - the ceiling of this family
    Xj = np.stack([pred, K3C, oldsum], 1); cj, r2j = ufit(Xj, target); cjw, r2jw = wfit(Xj, target, w)
    rec["target_on_kernel_K3C_oldsum"] = {"unweighted": {"r2": r2j, "coef": list(map(float, cj))}, "G4weighted": {"r2": r2jw, "coef": list(map(float, cjw))}}
    Xk = np.stack([pred, K3C], 1); ck, r2k = ufit(Xk, target); ckw, r2kw = wfit(Xk, target, w)
    rec["target_on_kernel_K3C"] = {"unweighted": {"r2": r2k, "coef": list(map(float, ck))}, "G4weighted": {"r2": r2kw, "coef": list(map(float, ckw))}}
    # deployable combinations against the true diagonal, G4-weighted and on the decisive neurons
    combos = {"chain_regen": g4, "deploy_S1.25": Pchain + 1.25 * pred, "deploy_S1.25+K3C": Pchain + 1.25 * (pred + K3C),
              "deploy_S1+oldstars": Pchain + pred + oldsum, "deploy_S1+K3C+oldstars": Pchain + pred + K3C + oldsum,
              "deploy_S1+K3C+youngstars(age1-4)": Pchain + pred + K3C + youngsum,
              "deploy_S1.25+K3C+oldstars": Pchain + 1.25 * (pred + K3C) + oldsum,
              "oracle_pairs+kernel+K3C+oldstars": Ptrue + pred + K3C + oldsum}
    rec["combos"] = {k: {**ev(v, t4, w), "nonpair_corr": float(np.corrcoef(v - Pchain if not k.startswith("oracle") else v - Ptrue, target)[0, 1]) if k != "chain_regen" else None,
                         "nonpair_corr_decisive": float(np.corrcoef((v - Pchain)[m] if not k.startswith("oracle") else (v - Ptrue)[m], target[m])[0, 1]) if k != "chain_regen" else None,
                         "nonpair_slope": float(np.dot((v - Pchain) if not k.startswith("oracle") else (v - Ptrue), target) / np.dot((v - Pchain) if not k.startswith("oracle") else (v - Ptrue), (v - Pchain) if not k.startswith("oracle") else (v - Ptrue))) if k != "chain_regen" else None}
                     for k, v in combos.items()}
    out.append(rec)
    r = rec["rms"]
    print(f"\nL{Lt}: non-pair target rms {r['target']:.2e}; kernel {r['pred']:.2e}, K3C {r['K3C']:.2e} | fresh stars: 211 {r['fresh_s211']:.1e}, 31 {r['fresh_s31']:.1e}, 1111 {r['fresh_s1111']:.1e} | OLD stars (ages>=1): 211 {r['old_s211']:.1e}, 31 {r['old_s31']:.1e}, 1111 {r['old_s1111']:.1e}", flush=True)
    print("   star rms by age: " + ", ".join(f"{a}: {v:.1e}" for a, v in rec["age_rms"].items()), flush=True)
    ro = rec["resid_on_oldsum"]; rs = rec["resid_on_stars"]
    print(f"   kernel residual on OLD stars (sum): coef {ro['unweighted']['coef']:+.2f} R2 {ro['unweighted']['r2']:.3f} (G4-weighted coef {ro['G4weighted']['coef']:+.2f} R2 {ro['G4weighted']['r2']:.3f}); corr all {ro['corr_all']:.3f}, decisive {ro['corr_decisive']:.3f}", flush=True)
    print(f"   residual on 5 star regressors: R2 {rs['unweighted']['r2']:.3f} " + ", ".join(f"{k} {v:+.2f}" for k, v in rs["unweighted"]["coef"].items()) + f" | G4-weighted R2 {rs['G4weighted']['r2']:.3f} " + ", ".join(f"{k} {v:+.2f}" for k, v in rs["G4weighted"]["coef"].items()), flush=True)
    tk = rec["target_on_kernel_K3C"]; tj = rec["target_on_kernel_K3C_oldsum"]
    print(f"   target on [kernel, K3C]: R2 {tk['unweighted']['r2']:.3f} coef {tk['unweighted']['coef'][0]:.2f},{tk['unweighted']['coef'][1]:.2f} | + old stars: R2 {tj['unweighted']['r2']:.3f} coef {tj['unweighted']['coef'][0]:.2f},{tj['unweighted']['coef'][1]:.2f},{tj['unweighted']['coef'][2]:.2f} (G4-weighted R2 {tk['G4weighted']['r2']:.3f} -> {tj['G4weighted']['r2']:.3f})", flush=True)
    for k, v in rec["combos"].items():
        extra = "" if v["nonpair_corr"] is None else f" | non-pair corr {v['nonpair_corr']:.3f} (decisive {v['nonpair_corr_decisive']:.3f}) slope {v['nonpair_slope']:.2f}"
        print(f"   {k:36s}: errG4 {v['errG4']:.3f} corr {v['corr']:.3f} slope {v['slope']:.3f}{extra}", flush=True)
json.dump(out, open("../results/summaries/r8_k4_memory.json", "w"), indent=1, default=float)
np.savez_compressed(os.path.join(CAP, "r8_k4_memory.npz"), **{f"L{Lt}_age{a}_{k}": v for Lt in TARGETS for a in acc[Lt] for k, v in acc[Lt][a].items()})
