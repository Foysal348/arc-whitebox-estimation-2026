"""Fable kappa3 ceiling test (10 Oct 2026): the chain's kappa4 -> kappa3 FEED is the carried late-layer kappa3 error; replace it
by a carried cubic kappa4 (star + C.T) feed.  Exact, untruncated ceiling on Monte-Carlo samples of the true network.

DIAGNOSIS (derived; the chain's recursion is in submissions/v56As_fs/estimator.py, F68 block and _dslices).
  The all-distinct post-ReLU third cumulant kappa3(h_i, h_j, h_c) at a ReLU layer m, to FIRST order in the per-layer
  non-Gaussianity of z_m (K = 3 response h - E h = Phi u + g (u^2 - v) + c (u^3 - 3 v u),  u = z_m - mu), is
      Phi_i Phi_j Phi_c kappa3(z)_ijc                                             (linear transport: the chain's legs, exact)
    + g_c Phi_i Phi_j [kappa4(z)_ccij + 2 C_ci C_cj] + 2 perms                    (quadratic vertex: GAUSSIAN birth + KAPPA4 FEED)
    + (g g Phi, c Phi Phi) x C x kappa3 cross terms                               (K x C: the V18 feedback legs, FB 1.25, rank 4)
    + kappa4 x C terms at the (2,2,2), (3,2,1), (4,1,1) vertices                  (small; chain: wk431 / g22c model)
  The chain has the Gaussian birth exactly (A = w1 C_off), the K x C terms through the feedback legs (true D21 in them: x1.001,
  i.e. their INPUT is not the problem), and the linear transport exactly up to truncation (no confinement: -3.9 % only).
  The one first-order term whose INPUT the chain does not possess is the kappa4 feed  g_c Phi_i Phi_j kappa4(z_m)_ccij:
  the F68 block (X3 = d(w1) [lam C_off + diag(dG)] d(w1), Y3 = (m/4) w2 1^T, M_t1 = u v^T) is exactly the model
      kappa4(z_m)_ccij  ~  (m/6) [lam_m C_ij + delta_ij dG_i]      -- INDEPENDENT of the vertex neuron c (mean-field feed),
  i.e. a kappa4 whose only pattern is C itself (the parity theorem says no such statistic carries the deep pattern; the
  chain's own kappa4 diagonal pattern corr is 0.29-0.34 on the decisive neurons). The feed enters exactly the ALL-DISTINCT
  post-ReLU entries (hint 3), it is born at every vertex and then CARRIED by the source legs like any other block (hint 1:
  the error grows with depth), and its diagonal part shifts d3 systematically (hint 2). Everything else at first order is
  either exact or has been oracle-tested dead.

CEILING OBJECTS (all sample-exact; row convention z_{l+1} = h_l W[l]; transports as in fable_ceiling_2026-10-09.py):
  Ybar_{m->t} = W[m] diag(Phi_{m+1}) W[m+1] ... diag(Phi_{t-1}) W[t-1]   (hub leg),   Ydot = diag(Phi_m) Ybar,   X = C_m Ydot
  images at t:  yA = (Phi u) Ybar = u Ydot,   yB = g (u^2 - v) Ybar        (mu, v, C: SAMPLE statistics of this run, pass 1,
                                                                             so every image is exactly centred)
  (2,1) slice read at t (D21_pr = kappa3(z_t)_ppr, d3 = its diagonal), one vertex m:
      Q1_3(m->t)_pr  = 2 E[yB_p yA_p yA_r] + E[yA_p^2 yB_r]                 (exact quadratic-vertex kappa3 birth, carried)
      Gauss(m->t)_pr = 2 sum_c g_c [Ybar_cr X_cp^2 + 2 Ybar_cp X_cp X_cr]   (its Gaussian part = the chain's B1 birth)
      F4(m->t)       = Q1_3 - Gauss = sum_c g_c [Ybar_cr k4(c,c,y_p,y_p) + 2 Ybar_cp k4(c,c,y_p,y_r)],  y = u Ydot
                                                                            (the EXACT kappa4 feed of vertex m read at t)
      F4(t) = sum_{m<t} F4(m->t)   -> E2E file "exact"
  Carried-kappa4 design ceiling (any carrier whose kappa4 is born within the last J layers, exact births, exact linear carry):
      kappa4(z_m) -> kappa4(z_m) - kappa4(yA_{m-J->m}):   F4born_J = F4 - F4carry_J,  F4carry_J computed as F4 with u_m
      replaced by the linear image ell = yA_{m-J->m} (ell = 0 contribution when m - J <= 1: z_1 is Gaussian)   -> "bornJ{J}"
  Representable content of the design (what the chain's legs can carry as thin feed legs, see the design note):
      star-fed feed (Gaussian cubic birth at m'' in [m-J, m-1], closed form, no samples):
          k4star(c,c,p,q) = 6 sum_e c_e [2 Ybar''_ec X''_ec Xh_ep Xh_eq + X''_ec^2 (Yh_ep Xh_eq + Xh_ep Yh_eq)]
          (Ybar'', X'' = legs m''->m; Yh, Xh = legs m''->t); feed slice F_pr = sum_c g_c Ybar_cr K[c;p,p] + 2 sum_c g_c Ybar_cp K[c;p,r]
          = 12 [ (c o Xh^2)^T H1 + (c o Yh o Xh)^T H2 + 2 (c o H1 o Xh)^T Xh + (c o H2 o Yh)^T Xh + (c o H2 o Xh)^T Yh ],
          H1 = (Ybar'' o X'') diag(g) Ybar_{m->t},  H2 = (X'' o X'') diag(g) Ybar_{m->t}                         -> "star_J{J}"
      C.T-fed feed, (a,b)-at-vertex placement (the only placement that is hub-separable in the chain):
          k4ct(c,c,p,q) = sum_a G''_ac That_{a;pq},  G'' = diag(4 g'')(Ybar'' o X''),  That_{a;pq} = kappa3(u''_a, y_p, y_q)
          sample form:  w = u'' G'',  wt = (g_m o w) Ybar_{m->t},  ell = yA_{m''->t}:  F_pr = E[wt_r ell_p^2] + 2 E[wt_p ell_p ell_r]
                                                                                                             -> "starct_J{Jct}" (star + ct)
  Self-checks: --selftest (n = 6 brute-force index loops vs every closed form and sample estimator, exact to 1e-10);
  in the run, F4(1->t) must vanish to noise (z_1 is Gaussian; checks the Gaussian subtraction with all placements), the
  split-half reliability of every quantity is printed, and the sample Y21 is compared with the teacher where available.

PRE-REGISTERED GATES (E2E through experiments/variants/r20_feed_oracle.py, plain chain, MLP #0, window L9-16, the chain's own
feed removed at the readout and the file's F21 added; paired ratio to the plain chain; reference: teacher kappa3 slice + d3
in the same window ~ x0.69 derived from k3loc L9-12 x0.857 and L13-16 x0.806):
  Diagnosis gate (file "exact"):      PASS  ratio <= 0.80 (the feed is the dominant carried kappa3 error);
                                      PARTIAL 0.80 < ratio <= 0.92;  KILL ratio > 0.92 (the feed is not the lever).
  Design ceiling (file "bornJ6"):     PASS  ratio <= 0.85;  KILL ratio > 0.95.
  Representable content ("starct_J6"): PASS ratio <= 0.90; KILL ratio > 0.97 (then the non-separable C.T placements are needed).
  Validity sub-gate (reported): split-half reliability of F4(t) on the off-diagonal >= 0.90 at every window target, else rerun
  with a larger N before reading the E2E (the injected sampling noise would be charged to the design).
  Synthetic network: same window, reported; a PASS requires MLP #0 PASS and synthetic <= 0.90 (its teacher kappa3 gain is smaller).

USAGE
  python fable_k3feed_ceiling_2026-10-10.py [N] [device] [--targets 9-16] [--J 4,6] [--Jct 6] [--ct-targets same|none|list]
         [--data W.npz] [--teacher-dir k4teach] [--out prefix] [--batch B] [--seed S] [--no-files]
  python fable_k3feed_ceiling_2026-10-10.py --selftest
  python fable_k3feed_ceiling_2026-10-10.py --analyze <FEED_DUMP dir> --net mlp0 --out <prefix of a finished run> --teacher-dir k4teach
Outputs: <out>.json (every metric), <out>.npz (F4 variants per target, float32), <out>_feed_<name>/layer{t}_seed0.npz (key F21:
the (2,1) slice incl. diagonal, for FEED_DIR of the E2E variant).  Every printed number is MEASURED on the samples of this run.
"""
import os, sys, json, time, argparse
from math import erf, sqrt, pi
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("N", nargs="?", type=float, default=20000)
ap.add_argument("device", nargs="?", default="")
ap.add_argument("--targets", default="9-16")
ap.add_argument("--J", default="4,6", help="kappa4 birth-age horizons of the carried design (carry ceiling + star closed form)")
ap.add_argument("--Jct", type=int, default=6, help="birth-age horizon of the sampled C.T (a,b) feed")
ap.add_argument("--ct-targets", default="same", help="same | none | list; targets with the sampled C.T feed (the costly part)")
ap.add_argument("--data", default=os.environ.get("R17_DATA", "r17_data.npz"), help="npz with W (L, n, n) (packed r17 file or a k4teach weights file)")
ap.add_argument("--teacher-dir", default=os.environ.get("FABLE_TEACHER_DIR", ""), help="k4teach dir with layer{t}_seed0.npz (Y21) for noise references")
ap.add_argument("--out", default=os.environ.get("FABLE_OUT", "fable_k3feed"))
ap.add_argument("--batch", type=int, default=0)
ap.add_argument("--seed", type=int, default=20261010)
ap.add_argument("--no-files", action="store_true")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--analyze", default="", help="FEED_DUMP dir written by r20_feed_oracle.py: attribution of the chain's D21/d3 error")
ap.add_argument("--net", default="mlp0")
args = ap.parse_args()
t0 = time.time()


def parse_layers(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(tok))
    return sorted(set(out))


def corr(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = a - a.mean(); b = b - b.mean(); d = sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64).ravel() - np.asarray(b, np.float64).ravel()) / max(np.linalg.norm(np.asarray(b, np.float64).ravel()), 1e-300))


# =============================================================================================================================
# closed forms (numpy or torch tensors; all 2-D (n, n) unless stated).  Conventions: legs are [hub/vertex neuron, target neuron].
# =============================================================================================================================
def gauss_slice(g, Ybar, X):
    """Gaussian part of the quadratic-vertex kappa3 birth on the (2,1) slice: 2 sum_c g_c [Ybar_cr X_cp^2 + 2 Ybar_cp X_cp X_cr]."""
    gc = g[:, None]
    return 2.0 * ((gc * X * X).T @ Ybar) + 4.0 * ((gc * Ybar * X).T @ X)


def star_feed_slice(c, g, Ybar2, X2, Yh, Xh, Ybar):
    """Feed slice of the star-born kappa4 of vertex m'' (c = c_{m''}), consumed at vertex m (g = g_m), read at t.
    Ybar2, X2: legs m''->m; Yh, Xh: legs m''->t; Ybar: hub leg m->t."""
    gY = g[:, None] * Ybar
    H1 = (Ybar2 * X2) @ gY
    H2 = (X2 * X2) @ gY
    cc = c[:, None]
    return 12.0 * ((cc * Xh * Xh).T @ H1 + (cc * Yh * Xh).T @ H2 + 2.0 * (cc * H1 * Xh).T @ Xh + (cc * H2 * Yh).T @ Xh + (cc * H2 * Xh).T @ Yh)


def selftest():
    rng = np.random.default_rng(7); n = 6; N = 3000
    # (1) sample estimator of the exact kappa4 feed vs brute-force kappa4 contraction (identity exact on any sample set)
    u = rng.standard_normal((N, n)); u = u * rng.standard_normal((N, n)) ** 2 + 0.3 * u ** 2; u -= u.mean(0)
    C = u.T @ u / N; v = np.diag(C).copy()
    Phi = rng.uniform(0.2, 0.9, n); g = rng.standard_normal(n); Ybar = rng.standard_normal((n, n)); Ydot = Phi[:, None] * Ybar
    yA = (Phi * u) @ Ybar; yB = (g * (u * u - v)) @ Ybar
    P1 = (yB * yA).T @ yA / N; P2 = (yA * yA).T @ yB / N
    F = 2 * P1 + P2 - gauss_slice(g, Ybar, C @ Ydot)
    k4 = np.einsum("sc,sd,si,sj->cdij", u, u, u, u) / N - np.einsum("cd,ij->cdij", C, C) - np.einsum("ci,dj->cdij", C, C) - np.einsum("cj,di->cdij", C, C)
    K = np.einsum("ccij,ip,jq->cpq", k4, Ydot, Ydot)
    Fb = np.einsum("c,cr,cpp->pr", g, Ybar, K) + 2 * np.einsum("c,cp,cpr->pr", g, Ybar, K)
    e1 = rel(F, Fb)
    # (2) star closed form vs brute-force Sym4 tensor
    c = rng.standard_normal(n); Ybar2 = rng.standard_normal((n, n)); X2 = rng.standard_normal((n, n)); Yd = rng.standard_normal((n, n))
    Yh = Ybar2 @ Yd; Xh = X2 @ Yd
    k4s = 6 * (np.einsum("e,ep,eq,er,es->pqrs", c, Ybar2, X2, X2, X2) + np.einsum("e,eq,ep,er,es->pqrs", c, Ybar2, X2, X2, X2)
               + np.einsum("e,er,ep,eq,es->pqrs", c, Ybar2, X2, X2, X2) + np.einsum("e,es,ep,eq,er->pqrs", c, Ybar2, X2, X2, X2))
    K = np.einsum("ccij,ip,jq->cpq", k4s, Yd, Yd)
    Fb = np.einsum("c,cr,cpp->pr", g, Ybar, K) + 2 * np.einsum("c,cp,cpr->pr", g, Ybar, K)
    e2 = rel(star_feed_slice(c, g, Ybar2, X2, Yh, Xh, Ybar), Fb)
    # (3) C.T (a,b)-placement feed: sample estimator vs brute force with the sample kappa3
    g2 = rng.standard_normal(n); G2 = (4 * g2)[:, None] * Ybar2 * X2
    T3 = np.einsum("sa,sc,sd->acd", u, u, u) / N
    Yt = Yd  # legs m''->t
    That = np.einsum("acd,cp,dq->apq", T3, Yt, Yt)
    K = np.einsum("ac,apq->cpq", G2, That)
    Fb = np.einsum("c,cr,cpp->pr", g, Ybar, K) + 2 * np.einsum("c,cp,cpr->pr", g, Ybar, K)
    w = u @ G2; wt = (g * w) @ Ybar; ell = u @ Yt
    F = ((ell * ell).T @ wt + 2 * (wt * ell).T @ ell) / N
    e3 = rel(F, Fb)
    # (4) carry form: the same estimator on a linear image reproduces the kappa4 feed of that image
    A0 = rng.standard_normal((n, n)); ell0 = u @ A0; Cl = ell0.T @ ell0 / N; vl = np.diag(Cl)
    lA = (Phi * ell0) @ Ybar; lB = (g * (ell0 * ell0 - vl)) @ Ybar
    F = 2 * (lB * lA).T @ lA / N + (lA * lA).T @ lB / N - gauss_slice(g, Ybar, A0.T @ (C @ (A0 @ Ydot)))
    k4l = np.einsum("sc,sd,si,sj->cdij", ell0, ell0, ell0, ell0) / N - np.einsum("cd,ij->cdij", Cl, Cl) - np.einsum("ci,dj->cdij", Cl, Cl) - np.einsum("cj,di->cdij", Cl, Cl)
    K = np.einsum("ccij,ip,jq->cpq", k4l, Ydot, Ydot)
    Fb = np.einsum("c,cr,cpp->pr", g, Ybar, K) + 2 * np.einsum("c,cp,cpr->pr", g, Ybar, K)
    e4 = rel(F, Fb)
    print(f"selftest: exact-feed estimator {e1:.1e} | star closed form {e2:.1e} | C.T (a,b) estimator {e3:.1e} | carry form {e4:.1e}")
    ok = max(e1, e2, e3, e4) < 1e-9
    print("selftest", "PASS" if ok else "FAIL"); return ok


if args.selftest:
    sys.exit(0 if selftest() else 1)

# =============================================================================================================================
# analysis mode: attribution of the chain's D21 / d3 error to the feed (needs the dump of r20_feed_oracle.py and this run's npz)
# =============================================================================================================================
if args.analyze:
    Z = np.load(args.out + ".npz"); res = {}
    for t in parse_layers(args.targets):
        f = os.path.join(args.analyze, f"{args.net}_layer{t}.npz")
        tf = os.path.join(args.teacher_dir, f"layer{t}_seed0.npz")
        if not (os.path.exists(f) and os.path.exists(tf) and f"L{t}_exact" in Z.files): continue
        d = np.load(f); T = np.load(tf); Y = np.asarray(T["Y21"], np.float64); d3T = np.diag(Y).copy(); off = ~np.eye(Y.shape[0], dtype=bool)
        row = {}
        D3 = np.asarray(d["D3"], np.float64); D3n = np.asarray(d["D3_nofeed"], np.float64)
        e3 = D3 - d3T; f3c = D3 - D3n
        row["d3"] = {"chain_rel_err": rel(D3, d3T), "chain_level_ratio": float(D3.mean() / d3T.mean()), "chain_feed_rel": float(np.linalg.norm(f3c) / np.linalg.norm(d3T)),
                     "chain_feed_level": float(f3c.mean() / d3T.mean())}
        for name in [k[len(f"L{t}_"):] for k in Z.files if k.startswith(f"L{t}_") and not k.endswith("_h0") and not k.endswith("_h1")]:
            F = np.asarray(Z[f"L{t}_{name}"], np.float64); f4 = np.diag(F)
            d3fix = D3n + f4
            row["d3"][name] = {"feed_level": float(f4.mean() / d3T.mean()), "fixed_rel_err": rel(d3fix, d3T), "fixed_level_ratio": float(d3fix.mean() / d3T.mean()),
                               "corr_err_vs_delta": corr(e3, f4 - f3c), "residual_energy": float(np.sum((e3 - (f4 - f3c)) ** 2) / np.sum(e3 ** 2))}
        if d["D21"].size > 1:
            D21 = np.asarray(d["D21"], np.float64); D21n = np.asarray(d["D21_nofeed"], np.float64)
            E = (D21 - Y)[off]; fc = (D21 - D21n)[off]
            row["D21"] = {"chain_rel_err": float(np.linalg.norm(E) / np.linalg.norm(Y[off])), "chain_feed_rel": float(np.linalg.norm(fc) / np.linalg.norm(Y[off]))}
            for name in [k[len(f"L{t}_"):] for k in Z.files if k.startswith(f"L{t}_") and not k.endswith("_h0") and not k.endswith("_h1")]:
                F = np.asarray(Z[f"L{t}_{name}"], np.float64)[off]; dl = F - fc
                noise = None
                if f"L{t}_{name}_h0" in Z.files:
                    h0 = np.asarray(Z[f"L{t}_{name}_h0"], np.float64)[off]; h1 = np.asarray(Z[f"L{t}_{name}_h1"], np.float64)[off]
                    noise = float(np.sum((h0 - h1) ** 2) / 4.0)   # noise energy of the half-average
                r = {"feed_rel": float(np.linalg.norm(F) / np.linalg.norm(Y[off])), "corr_feed_vs_chainfeed": corr(F, fc),
                     "corr_err_vs_minus_delta": corr(E, -dl), "coef_err_on_delta": float(-(E @ dl) / (dl @ dl)),
                     "fixed_rel_err": float(np.linalg.norm(E - dl) / np.linalg.norm(Y[off])),
                     "residual_energy": float(np.sum((E - dl) ** 2) / np.sum(E ** 2))}
                if noise is not None:
                    r["residual_energy_noise_corrected"] = float((np.sum((E - dl) ** 2) - noise) / np.sum(E ** 2)); r["feed_noise_rel"] = float(sqrt(noise) / np.linalg.norm(Y[off]))
                row["D21"][name] = r
        res[t] = row
        print(f"[L{t}] " + json.dumps(row)[:1500], flush=True)
    json.dump(res, open(args.out + "_attribution.json", "w"), indent=1)
    print("wrote", args.out + "_attribution.json"); sys.exit(0)

# =============================================================================================================================
# main run
# =============================================================================================================================
import torch
N = int(args.N)
dev = torch.device(args.device if args.device and (args.device == "cpu" or torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu"))
B = args.batch or (8192 if dev.type == "cuda" else 500); B = min(B, N)
D = np.load(args.data); W = D["W"].astype(np.float32); L, n = W.shape[0], W.shape[1]
TARGETS = [t for t in parse_layers(args.targets) if 2 <= t <= L]
JS = [int(x) for x in args.J.split(",") if x]
CT_T = TARGETS if args.ct_targets == "same" else ([] if args.ct_targets == "none" else [t for t in parse_layers(args.ct_targets) if t in TARGETS])
Tmax = max(TARGETS)
T32 = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
T64 = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=dev)
Wt = [T32(W[l]) for l in range(Tmax)]
Phi_np = lambda x: 0.5 * (1 + np.vectorize(erf)(x / sqrt(2))); pdf_np = lambda x: np.exp(-x * x / 2) / sqrt(2 * pi)
print(f"device {dev}, N {N}, batch {B}, targets {TARGETS}, J {JS}, Jct {args.Jct} on {CT_T}", flush=True)


def forward(gen):
    z = torch.randn(B_cur, n, device=dev, generator=gen) @ Wt[0]; zs = {1: z}
    for l in range(1, Tmax):
        z = torch.relu(z) @ Wt[l]; zs[l + 1] = z
    return zs


# ---- pass 1: sample mean / covariance at every layer 1..Tmax (every later image is centred on these) ----
S1 = torch.zeros(Tmax + 1, n, device=dev, dtype=torch.float64); S2 = {l: torch.zeros(n, n, device=dev, dtype=torch.float64) for l in range(1, Tmax + 1)}
gen = torch.Generator(device=dev); gen.manual_seed(args.seed); done = 0
while done < N:
    B_cur = min(B, N - done); zs = forward(gen)
    for l in range(1, Tmax + 1):
        S1[l] += zs[l].sum(0, dtype=torch.float64); S2[l] += (zs[l].T @ zs[l]).double()
    done += B_cur
MU = (S1 / N); C = {l: S2[l] / N - MU[l][:, None] * MU[l][None, :] for l in range(1, Tmax + 1)}
V = {l: torch.diag(C[l]).clamp_min(1e-12) for l in C}; SD = {l: V[l].sqrt() for l in C}
cf = {}
for l in range(1, Tmax + 1):
    a = (MU[l] / SD[l]).cpu().numpy(); ph = pdf_np(a); Ph = Phi_np(a); sd = SD[l].cpu().numpy()
    cf[l] = dict(mu=T32(MU[l].cpu().numpy()), v=T32((sd ** 2)), Phi=T32(Ph), g=T32(ph / (2 * sd)), c=T32(-a * ph / (6 * sd ** 2)),
                 Phi64=T64(Ph), g64=T64(ph / (2 * sd)), c64=T64(-a * ph / (6 * sd ** 2)))
teach = {}
if args.teacher_dir:
    for t in range(1, Tmax + 1):
        f = os.path.join(args.teacher_dir, f"layer{t}_seed0.npz")
        if os.path.exists(f):
            T = np.load(f); teach[t] = dict(Y21=np.asarray(T["Y21"], np.float64), C=np.asarray(T["C"], np.float64), mean=np.asarray(T["mean"], np.float64))
res = {"N": N, "device": str(dev), "targets": TARGETS, "J": JS, "Jct": args.Jct, "ct_targets": CT_T, "seed": args.seed, "pass1_vs_teacher": {}}
for t in sorted(teach):
    off = ~np.eye(n, dtype=bool); Cs = C[t].cpu().numpy()
    res["pass1_vs_teacher"][t] = {"mean_rel": rel(MU[t].cpu().numpy(), teach[t]["mean"]), "C_off_rel": rel(Cs[off], teach[t]["C"][off]), "var_rel": rel(np.diag(Cs), np.diag(teach[t]["C"]))}
print(f"pass 1 done ({time.time() - t0:.0f}s); sample vs teacher: " + " ".join(f"L{t} Coff {r['C_off_rel']:.2e}" for t, r in res["pass1_vs_teacher"].items()), flush=True)

# ---- hub legs Ybar_{m->t} for all 1 <= m < t <= Tmax (float32), C.T hub functionals G''_{m''->m} ----
Ybar = {}
for t in range(2, Tmax + 1):
    acc = None
    for m in range(t - 1, 0, -1):
        acc = Wt[m].clone() if acc is None else Wt[m] @ (cf[m + 1]["Phi"][:, None] * acc)
        Ybar[(t, m)] = acc
Ydot = lambda t, m: cf[m]["Phi"][:, None] * Ybar[(t, m)]
Xleg = lambda t, m: C[m].float() @ Ydot(t, m)              # X_{m->t} = C_m Ydot (hub leg index at m, target index at t)
carry_pairs = {J: [(t, m) for t in TARGETS for m in range(2, t) if m - J >= 2] for J in JS}
VL = {}
for J in JS:
    for (t, m) in carry_pairs[J]:
        a = m - J
        if (m, a) not in VL:
            Yd = Ydot(m, a); VL[(m, a)] = (Yd * (C[a].float() @ Yd)).sum(0)          # E[ell^2], ell = u_a Ydot_{a->m}
ct_pairs = sorted(set((m, mpp) for t in CT_T for m in range(2, t) for mpp in range(max(1, m - args.Jct), m)))
Gcc = {}
for (m, mpp) in ct_pairs:
    Gcc[(m, mpp)] = (4.0 * cf[mpp]["g"])[:, None] * Ybar[(m, mpp)] * Xleg(m, mpp)
print(f"legs: {len(Ybar)} hub legs, {len(VL)} carry images, {len(Gcc)} C.T functionals ({time.time() - t0:.0f}s)", flush=True)

# ---- accumulators (float64, two independent halves for noise estimates) ----
z2 = lambda: [torch.zeros(n, n, device=dev, dtype=torch.float64) for _ in range(2)]
Y21acc = {t: z2() for t in TARGETS}
P1 = {(t, m): z2() for t in TARGETS for m in range(1, t)}; P2 = {(t, m): z2() for t in TARGETS for m in range(1, t)}
CP = {(t, J): z2() for J in JS for t in TARGETS}
CTP = {t: z2() for t in CT_T}
Nh = [0, 0]

# ---- pass 2: the same samples (same seed), third-moment accumulators ----
gen = torch.Generator(device=dev); gen.manual_seed(args.seed); done = 0; nb = 0
while done < N:
    B_cur = min(B, N - done); zs = forward(gen); h = nb % 2; Nh[h] += B_cur
    U = {l: zs[l] - cf[l]["mu"] for l in range(1, Tmax + 1)}
    pA = {l: cf[l]["Phi"] * U[l] for l in range(1, Tmax)}
    for t in TARGETS:
        y = U[t]; Y21acc[t][h] += ((y * y).T @ y).double()
        yAc = {}
        for m in range(1, t):
            yA = pA[m] @ Ybar[(t, m)]; yAc[m] = yA
            yB = (cf[m]["g"] * (U[m] * U[m] - cf[m]["v"])) @ Ybar[(t, m)]
            P1[(t, m)][h] += ((yB * yA).T @ yA).double(); P2[(t, m)][h] += ((yA * yA).T @ yB).double()
            for J in JS:
                a = m - J
                if a >= 2:
                    ell = pA[a] @ Ybar[(m, a)]
                    ellB = (cf[m]["g"] * (ell * ell - VL[(m, a)])) @ Ybar[(t, m)]; ellT = yAc[a]
                    CP[(t, J)][h] += 2.0 * ((ellB * ellT).T @ ellT).double() + ((ellT * ellT).T @ ellB).double()
            if t in CT_T and m >= 2:
                for mpp in range(max(1, m - args.Jct), m):
                    w = U[mpp] @ Gcc[(m, mpp)]; wt = (cf[m]["g"] * w) @ Ybar[(t, m)]; ell = yAc[mpp]
                    CTP[t][h] += 2.0 * ((wt * ell).T @ ell).double() + ((ell * ell).T @ wt).double()
        del yAc
    done += B_cur; nb += 1
    if nb % 10 == 0:
        print(f"  batch {nb}, {done}/{N} ({time.time() - t0:.0f}s)", flush=True)
print(f"sampling done ({time.time() - t0:.0f}s)", flush=True)

# ---- assembly per target ----
off = ~np.eye(n, dtype=bool)
SAVE = {}; FILES = {}
def halves_and_total(parts):
    """parts: list of 2 tensors (half sums) -> (h0, h1, total) numpy float64 arrays."""
    return parts[0].cpu().numpy(), parts[1].cpu().numpy(), ((parts[0] + parts[1])).cpu().numpy()


def summarize(name, h0, h1, tot, ref_off=None, ref_diag=None):
    d = {"norm_off": float(np.linalg.norm(tot[off])), "norm_diag": float(np.linalg.norm(np.diag(tot))),
         "reliability_off": corr(h0[off], h1[off]), "reliability_diag": corr(np.diag(h0), np.diag(h1)),
         "noise_rel_off": float(np.linalg.norm((h0 - h1)[off]) / 2.0 / max(np.linalg.norm(tot[off]), 1e-300))}
    if ref_off is not None:
        d["rel_to_Y21_off"] = float(np.linalg.norm(tot[off]) / np.linalg.norm(ref_off)); d["rel_to_d3"] = float(np.linalg.norm(np.diag(tot)) / np.linalg.norm(ref_diag))
        d["mean_diag_over_mean_d3"] = float(np.diag(tot).mean() / ref_diag.mean())
    return d


def xcorr(a, b):
    """cross-half correlation (independent samples in a[0] and b[1], a[1] and b[0]): unbiased by shared sampling noise."""
    return {"off": 0.5 * (corr(a[0][off], b[1][off]) + corr(a[1][off], b[0][off])), "diag": 0.5 * (corr(np.diag(a[0]), np.diag(b[1])) + corr(np.diag(a[1]), np.diag(b[0]))),
            "off_raw": corr(a[2][off], b[2][off]), "diag_raw": corr(np.diag(a[2]), np.diag(b[2]))}


for t in TARGETS:
    row = {}
    yh0, yh1, ytot = halves_and_total(Y21acc[t]); Y21s = [yh0 / Nh[0], yh1 / Nh[1], ytot / N]
    row["Y21_sample"] = {"noise_rel_off": float(np.linalg.norm((Y21s[0] - Y21s[1])[off]) / 2.0 / np.linalg.norm(Y21s[2][off])),
                         "noise_rel_diag": float(np.linalg.norm(np.diag(Y21s[0] - Y21s[1])) / 2.0 / np.linalg.norm(np.diag(Y21s[2])))}
    if t in teach:
        YT = teach[t]["Y21"]; row["Y21_sample"]["vs_teacher_off"] = rel(Y21s[2][off], YT[off]); row["Y21_sample"]["vs_teacher_diag"] = rel(np.diag(Y21s[2]), np.diag(YT))
    ref_off, ref_diag = Y21s[2][off], np.diag(Y21s[2])
    # exact feed per vertex and its sum
    F4 = {}; Fsum = [np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))]; prof = {}
    for m in range(1, t):
        Yb = Ybar[(t, m)].double(); X = Xleg(t, m).double(); Gs = gauss_slice(cf[m]["g64"], Yb, X).cpu().numpy()
        p1 = P1[(t, m)]; p2 = P2[(t, m)]
        fh = [((2.0 * p1[k] + p2[k]) / Nh[k]).cpu().numpy() - Gs for k in range(2)]
        ftot = ((2.0 * (p1[0] + p1[1]) + (p2[0] + p2[1])) / N).cpu().numpy() - Gs
        F4[m] = [fh[0], fh[1], ftot]
        for k in range(3): Fsum[k] += F4[m][k]
        prof[m] = {"norm_off": float(np.linalg.norm(ftot[off])), "norm_diag": float(np.linalg.norm(np.diag(ftot))), "reliability_off": corr(fh[0][off], fh[1][off]),
                   "gauss_norm_off": float(np.linalg.norm(Gs[off]))}
        if m == 1:
            prof[m]["gaussian_layer_check_off"] = float(np.linalg.norm(ftot[off]) / np.linalg.norm(Gs[off]))    # must be ~ noise: z_1 is Gaussian
            prof[m]["gaussian_layer_check_noise"] = float(np.linalg.norm((fh[0] - fh[1])[off]) / 2.0 / np.linalg.norm(Gs[off]))
    row["exact"] = summarize("exact", *Fsum, ref_off, ref_diag); row["per_vertex"] = prof
    cum = np.zeros((n, n)); row["cumulative_by_age_off"] = {}
    for j in range(1, t):
        cum += F4[t - j][2]; row["cumulative_by_age_off"][j] = float(np.linalg.norm(cum[off]) / np.linalg.norm(Fsum[2][off]))
    variants = {"exact": Fsum}
    # carried-kappa4 design ceiling: births within J layers
    for J in JS:
        Gl = np.zeros((n, n))
        for (tt, m) in carry_pairs[J]:
            if tt != t: continue
            a = m - J; Yd_am = Ydot(m, a).double(); Xl = Yd_am.T @ (C[a] @ Ydot(t, a).double())
            Gl += gauss_slice(cf[m]["g64"], Ybar[(t, m)].double(), Xl).cpu().numpy()
        ch = [(CP[(t, J)][k] / Nh[k]).cpu().numpy() - Gl for k in range(2)]; ctot = ((CP[(t, J)][0] + CP[(t, J)][1]) / N).cpu().numpy() - Gl
        born = [Fsum[k] - [ch[0], ch[1], ctot][k] for k in range(3)]
        variants[f"bornJ{J}"] = born
        row[f"bornJ{J}"] = summarize("born", *born, ref_off, ref_diag); row[f"bornJ{J}"]["carry_norm_off"] = float(np.linalg.norm(ctot[off]))
        row[f"bornJ{J}"]["vs_exact"] = xcorr(born, Fsum)
        # star closed form (no samples): births of ages <= J at every vertex m < t
        st = torch.zeros(n, n, device=dev, dtype=torch.float64)
        for m in range(2, t):
            for mpp in range(max(1, m - J), m):
                st += star_feed_slice(cf[mpp]["c64"], cf[m]["g64"], Ybar[(m, mpp)].double(), Xleg(m, mpp).double(), Ybar[(t, mpp)].double(), Xleg(t, mpp).double(), Ybar[(t, m)].double())
        st = st.cpu().numpy(); variants[f"star_J{J}"] = [st, st, st]
        row[f"star_J{J}"] = {"norm_off": float(np.linalg.norm(st[off])), "norm_diag": float(np.linalg.norm(np.diag(st))), "rel_to_Y21_off": float(np.linalg.norm(st[off]) / np.linalg.norm(ref_off)),
                             "vs_exact": xcorr([st, st, st], Fsum), "vs_born": xcorr([st, st, st], born)}
    if t in CT_T:
        ct = [(CTP[t][k] / Nh[k]).cpu().numpy() for k in range(2)] + [((CTP[t][0] + CTP[t][1]) / N).cpu().numpy()]
        row[f"ct_J{args.Jct}"] = summarize("ct", *ct, ref_off, ref_diag); row[f"ct_J{args.Jct}"]["vs_exact"] = xcorr(ct, Fsum)
        sc = [variants[f"star_J{args.Jct}"][k] + ct[k] for k in range(3)] if f"star_J{args.Jct}" in variants else None
        if sc is not None:
            variants[f"starct_J{args.Jct}"] = sc
            bk = f"bornJ{args.Jct}" if f"bornJ{args.Jct}" in variants else None
            row[f"starct_J{args.Jct}"] = summarize("starct", *sc, ref_off, ref_diag); row[f"starct_J{args.Jct}"]["vs_exact"] = xcorr(sc, Fsum)
            if bk: row[f"starct_J{args.Jct}"]["vs_born"] = xcorr(sc, variants[bk])
            # LS share of the representable content inside the born feed (cross-half, off-diagonal)
            if bk:
                b = variants[bk]; num = 0.5 * ((sc[0][off] * b[1][off]).sum() + (sc[1][off] * b[0][off]).sum()); den = (sc[0][off] * sc[1][off]).sum()
                row[f"starct_J{args.Jct}"]["ls_coef_in_born"] = float(num / den) if den != 0 else float("nan")
    res[t] = row
    variants["zero"] = [np.zeros((n, n))] * 3          # E2E control: the chain's own feed removed, nothing added
    for name, v in variants.items():
        SAVE[f"L{t}_{name}"] = v[2].astype(np.float32)
        if name in ("exact",) or name.startswith("bornJ") or name.startswith("starct") or name.startswith("ct"):
            SAVE[f"L{t}_{name}_h0"] = v[0].astype(np.float32); SAVE[f"L{t}_{name}_h1"] = v[1].astype(np.float32)
        FILES.setdefault(name, {})[t] = v[2].astype(np.float32)
    ex = row["exact"]
    msg = f"[L{t}] exact feed: off {ex['rel_to_Y21_off']:.4f} of |Y21|, diag {ex['rel_to_d3']:.4f} of |d3| (level {ex['mean_diag_over_mean_d3']:+.4f}), reliability off {ex['reliability_off']:.3f} diag {ex['reliability_diag']:.3f}"
    if 1 in prof: msg += f" | Gaussian-layer check {prof[1]['gaussian_layer_check_off']:.2e} (noise {prof[1]['gaussian_layer_check_noise']:.2e})"
    for J in JS:
        msg += f" | bornJ{J}: {row[f'bornJ{J}']['rel_to_Y21_off']:.4f}, corr vs exact {row[f'bornJ{J}']['vs_exact']['off']:.3f}/{row[f'bornJ{J}']['vs_exact']['diag']:.3f}; star_J{J} corr vs born {row[f'star_J{J}']['vs_born']['off']:.3f}"
    if f"starct_J{args.Jct}" in row:
        r_ = row[f"starct_J{args.Jct}"]; msg += f" | star+ct corr vs born {r_['vs_born']['off']:.3f}/{r_['vs_born']['diag']:.3f} (ls coef {r_.get('ls_coef_in_born', float('nan')):.3f})"
    print(msg + f" ({time.time() - t0:.0f}s)", flush=True)

# ---- verdict helper text (the E2E decides; this is the validity sub-gate) ----
res["validity"] = {t: {"reliability_off": res[t]["exact"]["reliability_off"], "ok": res[t]["exact"]["reliability_off"] >= 0.90} for t in TARGETS}
res["validity"]["all_ok"] = all(v["ok"] for k, v in res["validity"].items() if k != "all_ok")
print("validity sub-gate (split-half reliability of the exact feed >= 0.90 at every target):", "OK" if res["validity"]["all_ok"] else "NOT MET: increase N before the E2E")
json.dump(res, open(args.out + ".json", "w"), indent=1)
if not args.no_files:
    np.savez(args.out + ".npz", **SAVE)
    for name, d in FILES.items():
        od = f"{args.out}_feed_{name}"; os.makedirs(od, exist_ok=True)
        for t, F in d.items(): np.savez(os.path.join(od, f"layer{t}_seed0.npz"), F21=F)
    print("wrote", args.out + ".json", args.out + ".npz", "and E2E dirs:", ", ".join(f"{args.out}_feed_{k}" for k in FILES))
print(f"done ({time.time() - t0:.0f}s)")
