"""Fable ceiling test (9 Oct 2026): the "lagged kappa3-dressed vertex" (LKV) kappa4 carrier, exact and untruncated.

Design under test (consultations/fable_2026-10-09_design.md). At every ReLU vertex m the K = 3 Hermite response
    h - E h = Phi u + g (u^2 - v) + c (u^3 - 3 v u),  u = z_m - mu_m,  g = phi/(2 sigma),  c = -alpha phi/(6 sigma^2),
gives birth to kappa4 that is then carried LINEARLY to a target pre-activation z_t (age j = t - m) and read on the diagonal.
Row convention: z_{l+1} = h_l W[l].  Transport to t:  Ybar = W[m] diag(Phi_{m+1}) W[m+1] ... diag(Phi_{t-1}) W[t-1]  (hub leg,
no Phi on the hub), Ydot = diag(Phi_m) Ybar (linear-response legs), X = C_m Ydot (the C-leg).  The sample-exact objects
    S [a,i]  = E[u_a (u Ydot)_i^2]          = T(a; Ydot_i, Ydot_i)      (kappa3 of z_m, two legs at t, hub index at m)
    YL[a,i]  = E[u_a^2 (u Ydot)_i]          = (Y21(m) Ydot)[a,i]        (the (2,1)-slice leg)
    K2[a,i]  = kappa4(u_a, u_a, y_i, y_i)                              (for the C.K diagnostic only)
give the factorised readouts of the single-piece cubic/quadratic births (identities of ROUND18 section 3, verified <= 1e-14):
    star = 24 sum_a c_a Ybar_ai X_ai^3           (Gaussian cubic star, shipped as V56)
    ct   = 24 sum_a g_a Ybar_ai X_ai S_ai        (Q1 C.T: kappa3 converted at a QUADRATIC vertex)
    tt   = 36 sum_a c_a Ybar_ai YL_ai S_ai       (Cu1 T.T: kappa3 SQUARED at a CUBIC vertex, the new carrier)
    ck   = 36 sum_a c_a Ybar_ai X_ai K2_ai       (Cu1 C.K, diagnostic)
The ARCHITECTURE prediction is  ARCH_J = sum_{j<=J} (star + ct + tt)_j, and its deployable truncation projects the hub
index of S on the top-r eigenvectors of C_m (the chain's own C) and YL on the top-r_y left singular space of Y21(m).

In the same pass the FULL K = 3 one-vertex births are decomposed exactly from the four linear images
    yA = (Phi u) Ybar, yB = (g He2) Ybar, yC = (c He3) Ybar, yR = (relu(z_m) - yA-piece - yB-piece - yC-piece) Ybar
through the joint 4th-order cumulants of (yA, yB, yC, yR) per target neuron:  kappa4(yA) is the exact linear carry of
kappa4(z_m); kappa4(yA+yB+yC) - kappa4(yA) is the exact K = 3 birth at m with EXACT inputs (T, K, kappa5, kappa6 of z_m);
Q1 = 4 k(B,A,A,A), Cu1 = 4 k(C,A,A,A), Q2 = 6 k(B,B,A,A), QCu = 12 k(B,C,A,A), CuCu = 6 k(C,C,A,A); with yR the births are
exact and  sum_j births_exact_j + kappa4(yA at age J) == kappa4(z_t)  identically (telescoping self-check, printed).

Metric (project, section 6 of the brief): dec = |mu_t/sigma_t| < 1.5 (teacher);  corr(pred / v_t^2, K4d_t / v_t^2) on dec.

Usage:  python fable_ceiling_2026-10-09.py [N] [device] [--targets 12,15,16 | all] [--ages 6] [--data r17_data.npz]
                                            [--teacher-dir k4teach] [--out prefix] [--batch B] [--ranks 32,64,128] [--ry 16,64]
Env fallbacks: R17_DATA (packed npz: W (16,n,n), mean (17,n), C (17,n,n), K4d (17,n), index = layer), FABLE_OUT.
Outputs: <out>.json (all metrics), <out>.npz (per-(t, j) readout vectors, for the component test), and
         <out>_k4pred_<name>/layer{t}_seed0.npz with K4d = prediction (for the end-to-end injection through
         experiments/variants/r20_joint_oracle.py with OPUS_TEACHER=<that dir>, R20_K4=pattern).
Every printed number is MEASURED on the samples of this run.
"""
import os, sys, json, time, argparse, itertools
from math import erf, sqrt, pi, comb
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("N", nargs="?", type=float, default=20000)
ap.add_argument("device", nargs="?", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--targets", default="12,15,16")
ap.add_argument("--diag-targets", default="12,15,16", help="targets with the C.K diagnostic and the rank sketches")
ap.add_argument("--ages", type=int, default=6)
ap.add_argument("--data", default=os.environ.get("R17_DATA", "r17_data.npz"))
ap.add_argument("--teacher-dir", default=os.environ.get("FABLE_TEACHER_DIR", ""), help="k4teach dir with layer{l}_seed0.npz (Y21) for the thin-YL sketch")
ap.add_argument("--out", default=os.environ.get("FABLE_OUT", "fable_ceiling"))
ap.add_argument("--batch", type=int, default=0)
ap.add_argument("--ranks", default="32,64,128")
ap.add_argument("--ry", default="16,64")
ap.add_argument("--seed", type=int, default=20261009)
args = ap.parse_args()
N = int(args.N)
dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
B = args.batch or (20000 if dev.type == "cuda" else 1000)
B = min(B, N)
t0 = time.time()

D = np.load(args.data)
W = D["W"].astype(np.float32); MU = D["mean"].astype(np.float64); CC = D["C"].astype(np.float32); K4T = D["K4d"].astype(np.float64)
L, n = W.shape[0], W.shape[1]
SD = np.sqrt(np.stack([np.diag(CC[l]).astype(np.float64) if l > 0 else np.ones(n) for l in range(L + 1)]))
Phi_np = lambda x: 0.5 * (1 + np.vectorize(erf)(x / sqrt(2))); pdf_np = lambda x: np.exp(-x * x / 2) / sqrt(2 * pi)
TARGETS = list(range(2, L + 1)) if args.targets == "all" else [int(x) for x in args.targets.split(",")]
DIAG = set(int(x) for x in args.diag_targets.split(",")) & set(TARGETS)
RANKS = [int(x) for x in args.ranks.split(",") if x]; RY = [int(x) for x in args.ry.split(",") if x]
PAIRS = [(t, m) for t in TARGETS for m in range(max(1, t - args.ages), t)]   # (target, vertex)
VERTICES = sorted(set(m for _, m in PAIRS))
T32 = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
Wt = [T32(W[l]) for l in range(L)]
coef = {}
for m in VERTICES:
    a = MU[m] / SD[m]; ph = pdf_np(a); Ph = Phi_np(a)
    coef[m] = dict(mu=T32(MU[m]), v=T32(SD[m] ** 2), Phi=T32(Ph), g=T32(ph / (2 * SD[m])), c=T32(-a * ph / (6 * SD[m] ** 2)))
PhiT = {l: T32(Phi_np(MU[l] / SD[l])) for l in range(1, L + 1)}

# ---- transports and teacher legs per (t, m): Ybar (hub leg), Ydot = diag(Phi_m) Ybar, X = C_m Ydot, star ----
leg = {}
for t in TARGETS:
    acc = None
    for m in range(t - 1, max(0, t - args.ages - 1), -1):           # build Ybar_{m->t} backwards: Ybar_m = W[m] diag(Phi_{m+1}) Ybar_{m+1}
        acc = Wt[m].clone() if acc is None else Wt[m] @ (PhiT[m + 1][:, None] * acc)
        if m < 1: break
        Ybar = acc; Ydot = coef[m]["Phi"][:, None] * Ybar
        X = T32(CC[m]) @ Ydot
        star = 24.0 * (coef[m]["c"][:, None] * Ybar * X ** 3).sum(0).double()
        leg[(t, m)] = dict(Ybar=Ybar, Ydot=Ydot, X=X, star=star)
print(f"setup: {len(PAIRS)} (target, vertex) pairs, device {dev}, N {N}, batch {B}  ({time.time() - t0:.0f}s)", flush=True)

# ---- exponent tuples of (yA, yB, yC, yR) up to total degree 4 ----
EXPS = [e for e in itertools.product(range(5), repeat=4) if 1 <= sum(e) <= 4]
EIDX = {e: k for k, e in enumerate(EXPS)}
acc = {}
for (t, m) in PAIRS:
    acc[(t, m)] = dict(M=torch.zeros(len(EXPS), n, device=dev, dtype=torch.float64),
                       S1=torch.zeros(n, n, device=dev, dtype=torch.float64), S2=torch.zeros(n, n, device=dev, dtype=torch.float64))
    if t in DIAG:
        acc[(t, m)]["S0"] = torch.zeros(n, n, device=dev, dtype=torch.float64); acc[(t, m)]["S3"] = torch.zeros(n, n, device=dev, dtype=torch.float64)
ztrue = {t: torch.zeros(4, n, device=dev, dtype=torch.float64) for t in TARGETS}
gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
done = 0; nb = 0
while done < N:
    bsz = min(B, N - done)
    z = torch.randn(bsz, n, device=dev, generator=gen) @ Wt[0]
    zs = {1: z}
    for l in range(1, L):
        z = torch.relu(z) @ Wt[l]; zs[l + 1] = z
    for t in TARGETS:
        y = zs[t] - T32(MU[t]); y2 = y * y
        ztrue[t] += torch.stack([y.sum(0), y2.sum(0), (y2 * y).sum(0), (y2 * y2).sum(0)]).double()
    for (t, m) in PAIRS:
        cf = coef[m]; lg = leg[(t, m)]; A = acc[(t, m)]
        u = zs[m] - cf["mu"]; u2 = u * u
        pA = cf["Phi"] * u; pB = cf["g"] * (u2 - cf["v"]); pC = cf["c"] * (u2 * u - 3.0 * cf["v"] * u)
        pR = torch.relu(zs[m]) - pA - pB - pC
        yA = pA @ lg["Ybar"]; yB = pB @ lg["Ybar"]; yC = pC @ lg["Ybar"]; yR = pR @ lg["Ybar"]
        yA2 = yA * yA
        A["S1"] += (u.T @ yA2).double(); A["S2"] += (u2.T @ yA).double()
        if "S3" in A:
            A["S0"] += (u.T @ yA).double(); A["S3"] += (u2.T @ yA2).double()
        pw = [[None, yA, yA2, yA2 * yA, yA2 * yA2], [None, yB, yB * yB, None, None], [None, yC, yC * yC, None, None], [None, yR, yR * yR, None, None]]
        for k in range(1, 4):
            pw[k][3] = pw[k][2] * pw[k][1]; pw[k][4] = pw[k][2] * pw[k][2]
        for e, k in EIDX.items():
            prod = None
            for var in range(4):
                if e[var]:
                    prod = pw[var][e[var]] if prod is None else prod * pw[var][e[var]]
            A["M"][k] += prod.sum(0, dtype=torch.float64)
    done += bsz; nb += 1
    if nb % 10 == 0:
        print(f"  batch {nb}, {done}/{N} samples ({time.time() - t0:.0f}s)", flush=True)
print(f"sampling done ({time.time() - t0:.0f}s)", flush=True)


def cumulants(M):
    """M: raw power sums (len(EXPS), n) -> dict of central moments by exponent tuple and a kappa4 function over variable multisets."""
    raw = {e: (M[k] / N).cpu().numpy() for e, k in EIDX.items()}
    raw[(0, 0, 0, 0)] = np.ones(n)
    mean = [raw[tuple(1 if j == v else 0 for j in range(4))] for v in range(4)]
    cen = {}
    for e in EXPS:
        tot = np.zeros(n)
        for ep in itertools.product(*[range(ek + 1) for ek in e]):
            w = np.ones(n)
            for v in range(4):
                if e[v] - ep[v]:
                    w = w * comb(e[v], ep[v]) * (-mean[v]) ** (e[v] - ep[v])
            tot += w * raw[ep]
        cen[e] = tot
    def exp_of(vars_):
        e = [0, 0, 0, 0]
        for v in vars_: e[v] += 1
        return tuple(e)
    def k4(w1, w2, w3, w4):
        return (cen[exp_of((w1, w2, w3, w4))] - cen[exp_of((w1, w2))] * cen[exp_of((w3, w4))]
                - cen[exp_of((w1, w3))] * cen[exp_of((w2, w4))] - cen[exp_of((w1, w4))] * cen[exp_of((w2, w3))])
    def k4_sum(S):
        tot = np.zeros(n)
        for ws in itertools.product(S, repeat=4):
            tot += k4(*ws)
        return tot
    return k4, k4_sum


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


res = {"N": N, "device": str(dev), "ages": args.ages, "targets": TARGETS, "pairs": len(PAIRS)}
save = {}
preds = {}   # name -> {t: vector} for the end-to-end dirs
for t in TARGETS:
    vt = SD[t] ** 2; dec = np.abs(MU[t] / SD[t]) < 1.5; tk = K4T[t]
    met = lambda p: dict(corr=corr((p / vt ** 2)[dec], (tk / vt ** 2)[dec]), level=float(p[dec].mean() / tk[dec].mean()))
    m1, m2, m3, m4 = (ztrue[t][p].cpu().numpy() / N for p in range(4))
    vs = m2 - m1 ** 2; k4_true = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4 - 3 * vs ** 2
    rt = {"n_decisive": int(dec.sum()), "sample_k4_vs_teacher": met(k4_true), "per_age": {}, "cum": {}}
    per = {}
    for m in range(t - 1, max(0, t - args.ages - 1), -1):
        if (t, m) not in acc: continue
        j = t - m; A = acc[(t, m)]; lg = leg[(t, m)]; cf = coef[m]
        k4, k4_sum = cumulants(A["M"])
        kA = k4_sum([0]); kK3 = k4_sum([0, 1, 2]); kK2 = k4_sum([0, 1]); kK013 = k4_sum([0, 2]); kEx = k4_sum([0, 1, 2, 3])
        p = dict(L4=kA, births_K3=kK3 - kA, births_K2=kK2 - kA, births_K013=kK013 - kA, births_exact=kEx - kA,
                 Q1=4 * k4(1, 0, 0, 0), Cu1=4 * k4(2, 0, 0, 0), Q2=6 * k4(1, 1, 0, 0), QCu=12 * k4(1, 2, 0, 0), CuCu=6 * k4(2, 2, 0, 0))
        p["higher"] = p["births_K3"] - (p["Q1"] + p["Cu1"] + p["Q2"] + p["QCu"] + p["CuCu"])
        S = A["S1"] / N; YL = A["S2"] / N
        Ybar = lg["Ybar"].double(); X = lg["X"].double(); gd = cf["g"].double(); cd = cf["c"].double()
        p["star"] = lg["star"].cpu().numpy()
        p["ct"] = (24.0 * (gd[:, None] * Ybar * X * S).sum(0)).cpu().numpy()
        p["tt"] = (36.0 * (cd[:, None] * Ybar * YL * S).sum(0)).cpu().numpy()
        if "S3" in A:
            X_s = A["S0"] / N; Ey2 = torch.as_tensor(A["M"][EIDX[(2, 0, 0, 0)]] / N - (A["M"][EIDX[(1, 0, 0, 0)]] / N) ** 2, device=dev)
            K2 = A["S3"] / N - cf["v"].double()[:, None] * Ey2[None, :] - 2.0 * X_s * X_s
            p["ck"] = (36.0 * (cd[:, None] * Ybar * X * K2).sum(0)).cpu().numpy()
            # rank sketches of the hub index of S on the top-r eigenvectors of C_m (deployable: the chain's own C)
            ev, V = torch.linalg.eigh(T32(CC[m]).double())
            for r in RANKS:
                Om = V[:, -r:]; Sr = Om @ (Om.T @ S)
                p[f"ct_r{r}"] = (24.0 * (gd[:, None] * Ybar * X * Sr).sum(0)).cpu().numpy()
                p[f"tt_r{r}"] = (36.0 * (cd[:, None] * Ybar * YL * Sr).sum(0)).cpu().numpy()
            # thin YL: project on the top-r_y left singular space of the teacher Y21(m) if available, else of YL itself (oracle)
            Y21f = os.path.join(args.teacher_dir, f"layer{m}_seed0.npz") if args.teacher_dir else ""
            if Y21f and os.path.exists(Y21f):
                Y21 = torch.as_tensor(np.load(Y21f)["Y21"].astype(np.float64), device=dev); src = "teacherY21"
            else:
                Y21 = YL; src = "oracleYL"
            Uy, _, _ = torch.linalg.svd(Y21, full_matrices=False)
            for ry in RY:
                Oy = Uy[:, :ry]; YLr = Oy @ (Oy.T @ YL)
                for r in RANKS:
                    Om = V[:, -r:]; Sr = Om @ (Om.T @ S)
                    p[f"tt_r{r}_ry{ry}"] = (36.0 * (cd[:, None] * Ybar * YLr * Sr).sum(0)).cpu().numpy()
            p["ry_source"] = src
        per[j] = p
        for k_, v_ in p.items():
            if isinstance(v_, np.ndarray): save[f"L{t}_j{j}_{k_}"] = v_
        chk = {k_: met(v_) for k_, v_ in p.items() if isinstance(v_, np.ndarray)}
        s3 = p["star"] + p["tt"] + (p["ck"] if "ck" in p else 0.0)
        chk["identity_Cu1_vs_star+tt(+ck)"] = dict(corr=corr(p["Cu1"][dec], s3[dec]), norm_ratio=float(np.linalg.norm(s3[dec]) / max(np.linalg.norm(p["Cu1"][dec]), 1e-300)))
        chk["identity_Q1_vs_ct"] = dict(corr=corr(p["Q1"][dec], p["ct"][dec]), norm_ratio=float(np.linalg.norm(p["ct"][dec]) / max(np.linalg.norm(p["Q1"][dec]), 1e-300)))
        rt["per_age"][j] = chk
        print(f"L{t} age {j}: K3 births {chk['births_K3']['corr']:.3f} | Q1 {chk['Q1']['corr']:.3f} Cu1 {chk['Cu1']['corr']:.3f} QCu {chk['QCu']['corr']:.3f} "
              f"| star {chk['star']['corr']:.3f} ct {chk['ct']['corr']:.3f} tt {chk['tt']['corr']:.3f} | Cu1~star+tt+ck corr {chk['identity_Cu1_vs_star+tt(+ck)']['corr']:.3f} "
              f"ratio {chk['identity_Cu1_vs_star+tt(+ck)']['norm_ratio']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    # ---- cumulative over ages 1..J ----
    combos = {"star": ["star"], "ct": ["ct"], "tt": ["tt"], "star+tt": ["star", "tt"], "star+ct": ["star", "ct"], "ARCH=star+ct+tt": ["star", "ct", "tt"],
              "Q1": ["Q1"], "Cu1": ["Cu1"], "Q1+Cu1": ["Q1", "Cu1"], "Q1+Cu1+QCu": ["Q1", "Cu1", "QCu"], "births_K3": ["births_K3"],
              "births_K2": ["births_K2"], "births_K013": ["births_K013"], "births_exact": ["births_exact"]}
    if any("ck" in per[j] for j in per):
        combos["ARCH+ck"] = ["star", "ct", "tt", "ck"]
        for r in RANKS:
            combos[f"ARCH_r{r}"] = ["star", f"ct_r{r}", f"tt_r{r}"]
            for ry in RY: combos[f"ARCH_r{r}_ry{ry}"] = ["star", f"ct_r{r}", f"tt_r{r}_ry{ry}"]
    ages = sorted(per)
    for J in ages:
        row = {}
        for name, parts in combos.items():
            tot = np.zeros(n); ok = True
            for j in ages[:J]:
                for q in parts:
                    if q not in per[j]: ok = False; break
                    tot += per[j][q]
            if ok:
                row[name] = met(tot)
                if name in ("ARCH=star+ct+tt", "ARCH_r64", "ARCH_r64_ry16") and J in (4, args.ages):
                    preds.setdefault(f"{name.replace('=star+ct+tt', '')}_J{J}", {})[t] = tot
        # with the exact linear carry of kappa4(z_{t-J}) added: a "state J back" ceiling and the telescoping self-check
        row["births_K3+L4carry"] = met(sum(per[j]["births_K3"] for j in ages[:J]) + per[J]["L4"])
        ex = sum(per[j]["births_exact"] for j in ages[:J]) + per[J]["L4"]
        row["births_exact+L4carry"] = met(ex)
        row["telescoping_check_rel_err"] = float(np.linalg.norm(ex - k4_true) / np.linalg.norm(k4_true))
        row["L4carry_alone"] = met(per[J]["L4"])
        rt["cum"][J] = row
        print(f"L{t} ages 1..{J}: ARCH {row['ARCH=star+ct+tt']['corr']:.3f} (star {row['star']['corr']:.3f}, star+ct {row['star+ct']['corr']:.3f}, star+tt {row['star+tt']['corr']:.3f}) "
              f"| Q1+Cu1 {row['Q1+Cu1']['corr']:.3f} +QCu {row['Q1+Cu1+QCu']['corr']:.3f} | K3 births {row['births_K3']['corr']:.3f} K2 {row['births_K2']['corr']:.3f} K013 {row['births_K013']['corr']:.3f} "
              f"| K3+carry {row['births_K3+L4carry']['corr']:.3f} exact+carry {row['births_exact+L4carry']['corr']:.3f} (telescope err {row['telescoping_check_rel_err']:.1e}) carry alone {row['L4carry_alone']['corr']:.3f}"
              + (f" | r64 {row['ARCH_r64']['corr']:.3f}" if "ARCH_r64" in row else ""), flush=True)
    res[t] = rt

# ---- verdict on the pre-registered gates (design.md section 7) ----
J6 = args.ages
g = {}
for t in TARGETS:
    cum = res[t]["cum"]; Jmax = max(cum)
    g[t] = dict(arch6=cum[Jmax]["ARCH=star+ct+tt"]["corr"], star6=cum[Jmax]["star"]["corr"], starct6=cum[Jmax]["star+ct"]["corr"],
                arch4=cum[min(4, Jmax)]["ARCH=star+ct+tt"]["corr"], arch4_r64=cum[min(4, Jmax)].get("ARCH_r64", {}).get("corr"),
                k3births6=cum[Jmax]["births_K3"]["corr"])
res["gate_inputs"] = g
main = [t for t in (12, 15, 16) if t in g]
if main:
    pass_ = all(g[t]["arch6"] >= 0.70 and g[t]["arch6"] - g[t]["star6"] >= 0.12 for t in main)
    kill = all(g[t]["arch6"] < 0.60 for t in main) or all(g[t]["arch6"] - g[t]["starct6"] < 0.05 for t in main)
    res["verdict"] = "PASS" if pass_ else ("KILL" if kill else "INCONCLUSIVE")
    if N < 500000:
        res["verdict"] += " (NOT VALID: N < 5e5, sample-based terms are noise at this N; only the teacher-C star is meaningful)"
    print("gate inputs:", json.dumps({t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g[t].items()} for t in main}))
    print("VERDICT (pre-registered, design.md section 7):", res["verdict"])
np.savez(args.out + ".npz", **save)
json.dump(res, open(args.out + ".json", "w"), indent=1)
for name, dct in preds.items():
    d = f"{args.out}_k4pred_{name}"; os.makedirs(d, exist_ok=True)
    for t, vec in dct.items():
        np.savez(os.path.join(d, f"layer{t}_seed0.npz"), K4d=vec.astype(np.float64))
print(f"wrote {args.out}.json / .npz and {len(preds)} k4pred dirs ({time.time() - t0:.0f}s)")
