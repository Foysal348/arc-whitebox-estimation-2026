"""X12 table fit (light numpy). Same ridge recipe as kaggle/x11/fit_x11.py (= the N2 recipe: per layer, standardised
columns, intercept unpenalised, lambda per layer by grouped 5-fold CV over contiguous net blocks), on one or more dump dirs
(several dirs = the same nets dumped on different trajectories; their Gram sums are added per net, so CV groups stay nets).
Target y = truth - base (base = K1v + lead_dead of the dumped state: teacher forced -> one step; free running -> on policy).
Feature sets:
  T2 (101) = 87 e30_a features + h_k*st + h_k*sd (k = 0..6)          (N2 recipe)
  N3 (122) = T2 + h_k*st2 + h_k*s3st + h_k*sl                          (st2 = st^2, s3st = D3 st / var^1.5,
                                                                         sl = (g4ns - g4row) / var^2, corrector local part)
Usage: python fit_x12.py <out_dir> <tag> <dump_dir> [<dump_dir2> ...]   (env NETS=0-69, EST_SRC, N2_JSON)
Writes <out_dir>/<tag>_T2.json, <tag>_N3.json, <tag>_report.json and prints per layer CV SSE ratios."""
import os, sys, glob, json, re
import numpy as np

OUTD, TAG, DDS = sys.argv[1], sys.argv[2], sys.argv[3:]
NETS = os.environ.get("NETS", "0-69")
lo, hi = map(int, NETS.split("-"))
os.makedirs(OUTD, exist_ok=True)


def module_tables():
    src = open(os.environ["EST_SRC"], encoding="utf-8").read()
    ns = {}
    for name in ("CORR_FEATS", "CORR_BETA_NEW"):
        i = src.index("\n" + name + " = [") + 1
        j = src.index("\n]\n", i) + 2
        exec(src[i:j], ns)
    return ns["CORR_FEATS"], ns["CORR_BETA_NEW"]


FEATS87, BETA87 = module_tables()
assert len(FEATS87) == 87 and len(BETA87) == 16
I_D3 = FEATS87.index("D3")


def hermite(a, kmax=6):
    H = [np.ones_like(a), a]
    for k in range(2, kmax + 1):
        H.append(a * H[k - 1] - (k - 1) * H[k - 2])
    return H


MODS = ["st", "sd", "st2", "s3st", "sl"]
NAMES = FEATS87 + ["h%d*%s" % (k, m) for m in MODS for k in range(7)]
NF = len(NAMES)
COLS = {"T2": list(range(87 + 14)), "N3": list(range(NF))}
N2 = json.load(open(os.environ["N2_JSON"])) if os.environ.get("N2_JSON") else None
if N2 is not None:
    assert N2["feats"] == NAMES[:101], "N2 feature order differs"

per = {}
for dd in DDS:
    for f in sorted(glob.glob(os.path.join(dd, "**", "net_*.npz"), recursive=True)):
        i = int(re.search(r"net_(\d+)\.npz", f).group(1))
        if lo <= i <= hi:
            per.setdefault(i, []).append(f)
ids = sorted(per)
print("dumps", DDS, "nets", len(ids), "files", sum(len(v) for v in per.values()), flush=True)

G = np.zeros((16, len(ids), NF, NF)); B = np.zeros((16, len(ids), NF)); YY = np.zeros((16, len(ids)))
SX = np.zeros((16, len(ids), NF)); SY = np.zeros((16, len(ids))); NN = np.zeros((16, len(ids)))
E_orig = np.zeros((16, len(ids))); E_zero = np.zeros((16, len(ids))); E_n2 = np.zeros((16, len(ids)))
for c, i in enumerate(ids):
    for fpath in per[i]:
        z = np.load(fpath)
        F = z["F"].astype(np.float64); tr = z["truth"].astype(np.float64)
        base = np.concatenate([z["base_asm"], z["base"][15:16]]).astype(np.float64)
        st = z["st"].astype(np.float64); sp = z["sp"].astype(np.float64); al = z["alpha"].astype(np.float64)
        var = z["var"].astype(np.float64); g4 = z["g4"].astype(np.float64)
        g4l = z["g4l"].astype(np.float64) if "g4l" in z.files else g4
        sd = (z["g4c"].astype(np.float64) - g4) / var ** 2
        mods = {"st": st, "sd": sd, "st2": st * st, "s3st": F[:, :, I_D3] * st / var ** 1.5, "sl": (g4l - g4) / var ** 2}
        for l in range(16):
            H = hermite(al[l])
            X = np.concatenate([F[l]] + [np.stack([sp[l] * H[k] * mods[m][l] for k in range(7)], 1) for m in MODS], 1)
            y = tr[l] - base[l]
            G[l, c] += X.T @ X; B[l, c] += X.T @ y; YY[l, c] += y @ y; SX[l, c] += X.sum(0); SY[l, c] += y.sum(); NN[l, c] += len(y)
            E_orig[l, c] += np.sum((y - F[l] @ np.asarray(BETA87[l])) ** 2); E_zero[l, c] += y @ y
            if N2 is not None:
                E_n2[l, c] += np.sum((y - X[:, :101] @ np.asarray(N2["beta"][l])) ** 2)
        if c == 0:
            dchk = np.abs(F[8] @ np.asarray(BETA87[8]) - z["delta"][8]).max()
            print(fpath, "L8 87-table recompute vs dumped delta max|diff| %.2e (|delta| max %.2e); st nz layers %s; sl nz layers %s" % (
                dchk, np.abs(z["delta"][8]).max(), [l for l in range(16) if np.any(st[l] != 0)],
                [l for l in range(16) if np.any(mods["sl"][l] != 0)]), flush=True)
            if N2 is not None:
                H8 = hermite(al[8])
                X8 = np.concatenate([F[8]] + [np.stack([sp[8] * H8[k] * mods[m][8] for k in range(7)], 1) for m in MODS[:2]], 1)
                print("  N2 recompute vs dumped delta L8 max|diff| %.2e (meaningful only if the dump ran with N2)" % np.abs(
                    X8 @ np.asarray(N2["beta"][8]) - z["delta"][8]).max(), flush=True)

I1 = FEATS87.index("1")
LAMS = [0.0] + list(10.0 ** np.arange(-8, 1.01, 0.5))
folds = [ids[int(len(ids) * f / 5):int(len(ids) * (f + 1) / 5)] for f in range(5)]
fold_of = {i: f for f, blk in enumerate(folds) for i in blk}


def solve(idx, cols, lam):
    g = G[:, idx].sum(1); b = B[:, idx].sum(1); sx = SX[:, idx].sum(1); sy = SY[:, idx].sum(1); n = NN[:, idx].sum(1)
    out = []
    for l in range(16):
        c = [k for k in cols if k != I1]
        mx = sx[l, c] / n[l]; my = sy[l] / n[l]
        Gc = g[l][np.ix_(c, c)] - n[l] * np.outer(mx, mx)
        bc = b[l, c] - n[l] * mx * my
        sdv = np.sqrt(np.maximum(np.diag(Gc) / n[l], 0.0))
        keep = sdv > 1e-12 * max(1.0, sdv.max())
        beta = np.zeros(len(cols))
        if keep.any():
            ck = np.where(keep)[0]
            S = sdv[ck]
            A = Gc[np.ix_(ck, ck)] / np.outer(S, S) / n[l]
            r = bc[ck] / S / n[l]
            w = np.linalg.solve(A + lam * np.eye(len(ck)), r) if lam > 0 else np.linalg.lstsq(A, r, rcond=None)[0]
            bk = w / S
            pos = {k: j for j, k in enumerate(cols)}
            for j, kk in enumerate(ck):
                beta[pos[c[kk]]] = bk[j]
            icpt = my - mx[ck] @ bk
        else:
            icpt = my
        beta[cols.index(I1)] = icpt
        out.append(beta)
    return out


def sse(idx, cols, betas):
    e = np.zeros(16)
    for l in range(16):
        bb = np.zeros(NF); bb[cols] = betas[l]
        g = G[l, idx].sum(0); b = B[l, idx].sum(0); yy = YY[l, idx].sum()
        e[l] = yy - 2 * bb @ b + bb @ g @ bb
    return e


pos_all = np.arange(len(ids))
report = {"tag": TAG, "dumps": DDS, "nets": len(ids)}
eo = E_orig.sum(1); ez = E_zero.sum(1); en2 = E_n2.sum(1)
for T, cols in COLS.items():
    cv = np.zeros((len(LAMS), 16))
    for f in range(5):
        te = np.array([p for p, i in enumerate(ids) if fold_of[i] == f], dtype=int)
        trn = np.array([p for p, i in enumerate(ids) if fold_of[i] != f], dtype=int)
        for a, lam in enumerate(LAMS):
            cv[a] += sse(te, cols, solve(trn, cols, lam))
    best = cv.argmin(0)
    lam_l = [LAMS[a] for a in best]
    fits = {lam: solve(pos_all, cols, lam) for lam in set(lam_l)}
    beta = [list(map(float, fits[lam_l[l]][l])) for l in range(16)]
    cvbest = cv[best, np.arange(16)]
    report[T] = {"lam": lam_l, "cv_sse": cvbest.tolist(), "orig_sse": eo.tolist(), "zero_sse": ez.tolist(), "n2_sse": en2.tolist(),
                 "cv_over_orig_all": float(cvbest.sum() / eo.sum()), "cv_over_orig_last": float(cvbest[15] / eo[15]),
                 "cv_over_n2_all": float(cvbest.sum() / en2.sum()) if N2 else None,
                 "cv_over_n2_last": float(cvbest[15] / en2[15]) if N2 else None}
    json.dump({"feats": [NAMES[k] for k in cols], "beta": beta, "lam": lam_l, "nets": NETS, "dumps": DDS},
              open(os.path.join(OUTD, "%s_%s.json" % (TAG, T)), "w"))
    print("\n== %s %s (%d features): per layer CV one-step SSE / orig e30_a table, / N2 in-sample, / zero" % (TAG, T, len(cols)))
    for l in range(16):
        print("  L%2d lam %-8.1e  cv/orig %.4f  cv/N2 %.4f  cv/zero %.4f  N2/orig %.4f" % (
            l, lam_l[l], cvbest[l] / eo[l], cvbest[l] / max(en2[l], 1e-300), cvbest[l] / ez[l], en2[l] / eo[l]))
    print("  all-layer cv/orig %.4f cv/N2 %.4f   last layer cv/orig %.4f cv/N2 %.4f" % (
        cvbest.sum() / eo.sum(), cvbest.sum() / max(en2.sum(), 1e-300), cvbest[15] / eo[15], cvbest[15] / max(en2[15], 1e-300)), flush=True)
a, b = np.array(report["N3"]["cv_sse"]), np.array(report["T2"]["cv_sse"])
report["N3_over_T2_all"] = float(a.sum() / b.sum()); report["N3_over_T2_last"] = float(a[15] / b[15])
report["N3_over_T2_layers"] = (a / b).tolist()
print("\n%s: N3 CV / T2 CV  all-layer %.4f  last layer %.4f  per layer %s" % (TAG, report["N3_over_T2_all"], report["N3_over_T2_last"],
      " ".join("%.3f" % x for x in a / b)), flush=True)
json.dump(report, open(os.path.join(OUTD, "%s_report.json" % TAG), "w"), indent=1)
