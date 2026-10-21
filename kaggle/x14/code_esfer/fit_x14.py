"""X14 table fit (light numpy), N3 recipe of kaggle/x12/fit_x12.py (per layer: standardised columns, intercept
unpenalised, lambda per layer by grouped 5-fold CV over contiguous net blocks; target y = truth - base of the forced dump).
Feature sets (h_k = sigma*phi*He_k(alpha), k = 0..6; each modulator group = 7 columns):
  N3  (122) = 87 e30_a features + h_k * {st, sd, st2, s3st, sl}               (current best's recipe)
  N4a (164) = N3 + h_k * {zr, zc, rc2, ast, pdl, pst}                          (all six new groups)
  N4s       = N3 + per layer greedy forward selection of the new groups by CV (a group is kept if it cuts the layer's CV
              SSE by more than SEL_TH, default 0.1 %); groups unused in every layer are dropped from the table.
New modulators (raw sums dumped by e30k_x14, normalised here exactly as the estimator does with var^-E):
  zr = sum_j D21_ij^2/var_j / var_i^2, zc = sum_j D21_ji^2/var_j^2 / var_i, rc2 = sum_j C_ij^2/var_j / var_i,
  ast = |alpha| st, pdl = (W delta_prev) / sigma, pst = W st_prev.
Usage: python fit_x14.py <out_dir> <tag> <dump_dir>   (env NETS=0-69, EST_SRC, N3_JSON, SEL_TH)
Writes <out_dir>/<tag>_{N3,N4a,N4s}.json and <tag>_report.json."""
import os, sys, glob, json, re
import numpy as np

OUTD, TAG, DDS = sys.argv[1], sys.argv[2], sys.argv[3:]
NETS = os.environ.get("NETS", "0-69")
lo, hi = map(int, NETS.split("-"))
SEL_TH = float(os.environ.get("SEL_TH", "0.001"))
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


MODS3 = ["st", "sd", "st2", "s3st", "sl"]
NEWM = ["zr", "zc", "rc2", "ast", "pdl", "pst"]
MODS = MODS3 + NEWM
NAMES = FEATS87 + ["h%d*%s" % (k, m) for m in MODS for k in range(7)]
NF = len(NAMES)
N3C = list(range(87 + 7 * len(MODS3)))
GRP = {m: list(range(87 + 7 * (len(MODS3) + g), 87 + 7 * (len(MODS3) + g + 1))) for g, m in enumerate(NEWM)}
N3REF = json.load(open(os.environ["N3_JSON"])) if os.environ.get("N3_JSON") else None
if N3REF is not None:
    assert N3REF["feats"] == NAMES[:122], "N3 feature order differs"

per = {}
for dd in DDS:
    for f in sorted(glob.glob(os.path.join(dd, "**", "net_*.npz"), recursive=True)):
        i = int(re.search(r"net_(\d+)\.npz", f).group(1))
        if lo <= i <= hi:
            per.setdefault(i, []).append(f)
ids = sorted(per)
print("dumps", DDS, "nets", len(ids), "features", NF, flush=True)

G = np.zeros((16, len(ids), NF, NF)); B = np.zeros((16, len(ids), NF)); YY = np.zeros((16, len(ids)))
SX = np.zeros((16, len(ids), NF)); SY = np.zeros((16, len(ids))); NN = np.zeros((16, len(ids)))
E_n3 = np.zeros((16, len(ids)))
for c, i in enumerate(ids):
    for fpath in per[i]:
        z = np.load(fpath)
        F = z["F"].astype(np.float64); tr = z["truth"].astype(np.float64)
        base = np.concatenate([z["base_asm"], z["base"][15:16]]).astype(np.float64)
        st = z["st"].astype(np.float64); sp = z["sp"].astype(np.float64); al = z["alpha"].astype(np.float64)
        var = z["var"].astype(np.float64); g4 = z["g4"].astype(np.float64)
        g4l = z["g4l"].astype(np.float64) if "g4l" in z.files else g4
        g = lambda k: z[k].astype(np.float64)
        mods = {"st": st, "sd": (g("g4c") - g4) / var ** 2, "st2": st * st, "s3st": F[:, :, I_D3] * st / var ** 1.5,
                "sl": (g4l - g4) / var ** 2,
                "zr": g("zr") / var ** 2, "zc": g("zc") / var, "rc2": g("rc2") / var, "ast": np.abs(al) * st,
                "pdl": g("pdl") / np.sqrt(var), "pst": g("pst")}
        for l in range(16):
            H = hermite(al[l])
            X = np.concatenate([F[l]] + [np.stack([sp[l] * H[k] * mods[m][l] for k in range(7)], 1) for m in MODS], 1)
            y = tr[l] - base[l]
            G[l, c] += X.T @ X; B[l, c] += X.T @ y; YY[l, c] += y @ y; SX[l, c] += X.sum(0); SY[l, c] += y.sum(); NN[l, c] += len(y)
            if N3REF is not None:
                E_n3[l, c] += np.sum((y - X[:, :122] @ np.asarray(N3REF["beta"][l])) ** 2)
        if c == 0:
            if N3REF is not None:
                print(fpath, "N3 recompute vs dumped delta L8 max|diff| %.2e (|delta| max %.2e)" % (
                    np.abs(np.concatenate([F[8]] + [np.stack([sp[8] * hermite(al[8])[k] * mods[m][8] for k in range(7)], 1)
                                                    for m in MODS3], 1) @ np.asarray(N3REF["beta"][8]) - z["delta"][8]).max(),
                    np.abs(z["delta"][8]).max()), flush=True)
            for m in NEWM:
                print("  mod %-4s nonzero layers %s  |L8| rms %.3e" % (m, [l for l in range(16) if np.any(mods[m][l] != 0)],
                                                                      np.sqrt(np.mean(mods[m][8] ** 2))), flush=True)

I1 = FEATS87.index("1")
LAMS = [0.0] + list(10.0 ** np.arange(-8, 1.01, 0.5))
folds = [ids[int(len(ids) * f / 5):int(len(ids) * (f + 1) / 5)] for f in range(5)]
fold_of = {i: f for f, blk in enumerate(folds) for i in blk}
TE = [np.array([p for p, i in enumerate(ids) if fold_of[i] == f], dtype=int) for f in range(5)]
TR = [np.array([p for p, i in enumerate(ids) if fold_of[i] != f], dtype=int) for f in range(5)]
ALL = np.arange(len(ids))


def solve_l(l, idx, cols, lams):
    """beta (len(cols)) per lambda for layer l on nets idx."""
    g = G[l, idx].sum(0); b = B[l, idx].sum(0); sx = SX[l, idx].sum(0); sy = SY[l, idx].sum(); n = NN[l, idx].sum()
    c = [k for k in cols if k != I1]
    mx = sx[c] / n; my = sy / n
    Gc = g[np.ix_(c, c)] - n * np.outer(mx, mx)
    bc = b[c] - n * mx * my
    sdv = np.sqrt(np.maximum(np.diag(Gc) / n, 0.0))
    keep = sdv > 1e-12 * max(1.0, sdv.max())
    ck = np.where(keep)[0]
    pos = {k: j for j, k in enumerate(cols)}
    outs = []
    if len(ck):
        S = sdv[ck]
        A = Gc[np.ix_(ck, ck)] / np.outer(S, S) / n
        r = bc[ck] / S / n
        ev, U = np.linalg.eigh(A)
        ur = U.T @ r
    for lam in lams:
        beta = np.zeros(len(cols))
        if len(ck):
            if lam > 0:
                w = U @ (ur / (ev + lam))
            else:
                w = np.linalg.lstsq(A, r, rcond=None)[0]
            bk = w / S
            for j, kk in enumerate(ck):
                beta[pos[c[kk]]] = bk[j]
            icpt = my - mx[ck] @ bk
        else:
            icpt = my
        beta[cols.index(I1)] = icpt
        outs.append(beta)
    return outs


def sse_l(l, idx, cols, beta):
    bb = np.zeros(NF); bb[cols] = beta
    g = G[l, idx].sum(0); b = B[l, idx].sum(0); yy = YY[l, idx].sum()
    return yy - 2 * bb @ b + bb @ g @ bb


def cv_l(l, cols):
    cv = np.zeros(len(LAMS))
    for f in range(5):
        if len(TE[f]) == 0:
            continue
        for a, beta in enumerate(solve_l(l, TR[f], cols, LAMS)):
            cv[a] += sse_l(l, TE[f], cols, beta)
    a = int(cv.argmin())
    return cv[a], LAMS[a]


report = {"tag": TAG, "dumps": DDS, "nets": len(ids), "sel_th": SEL_TH, "layers": []}
tabs = {"N3": [], "N4a": [], "N4s": []}
cvs = {"N3": np.zeros(16), "N4a": np.zeros(16), "N4s": np.zeros(16)}
used = set()
en3 = E_n3.sum(1)
for l in range(16):
    allc = N3C + [k for m in NEWM for k in GRP[m]]
    c3, l3 = cv_l(l, N3C)
    ca, la = cv_l(l, allc)
    S, best, lb, chosen = list(N3C), c3, l3, []
    while True:
        trial = {m: cv_l(l, S + GRP[m]) for m in NEWM if m not in chosen}
        if not trial:
            break
        m = min(trial, key=lambda k: trial[k][0])
        if trial[m][0] < best * (1.0 - SEL_TH):
            S += GRP[m]; best, lb = trial[m]; chosen.append(m)
        else:
            break
    used |= set(chosen)
    for T, cols, lam, cvv in (("N3", N3C, l3, c3), ("N4a", allc, la, ca), ("N4s", S, lb, best)):
        bet = solve_l(l, ALL, cols, [lam])[0]
        full = np.zeros(NF); full[cols] = bet
        tabs[T].append((full, lam)); cvs[T][l] = cvv
    report["layers"].append({"l": l, "cv_n3": c3, "cv_n4a": ca, "cv_n4s": best, "lam_n3": l3, "lam_n4a": la, "lam_n4s": lb,
                             "chosen": chosen, "n3ref_insample": float(en3[l])})
    print("L%2d  N4a/N3 %.4f  N4s/N3 %.4f  chosen %s  (lam %.1e/%.1e/%.1e)  N3cv/N3ref-in-sample %.4f" % (
        l, ca / c3, best / c3, chosen, l3, la, lb, c3 / max(en3[l], 1e-300)), flush=True)
sel_cols = N3C + [k for m in NEWM if m in used for k in GRP[m]]
outcols = {"N3": N3C, "N4a": N3C + [k for m in NEWM for k in GRP[m]], "N4s": sel_cols}
for T in tabs:
    cols = outcols[T]
    json.dump({"feats": [NAMES[k] for k in cols], "beta": [list(map(float, tabs[T][l][0][cols])) for l in range(16)],
               "lam": [tabs[T][l][1] for l in range(16)], "nets": NETS, "dumps": DDS},
              open(os.path.join(OUTD, "%s_%s.json" % (TAG, T)), "w"))
for T in ("N4a", "N4s"):
    report[T + "_over_N3_all"] = float(cvs[T].sum() / cvs["N3"].sum())
    report[T + "_over_N3_last"] = float(cvs[T][15] / cvs["N3"][15])
report["used_groups"] = sorted(used)
report["N3_over_N3ref_insample_all"] = float(cvs["N3"].sum() / max(en3.sum(), 1e-300))
print("\n%s: CV all-layer  N4a/N3 %.4f  N4s/N3 %.4f | last layer N4a/N3 %.4f N4s/N3 %.4f | groups used %s | N3 cv / N3ref in-sample %.4f" % (
    TAG, report["N4a_over_N3_all"], report["N4s_over_N3_all"], report["N4a_over_N3_last"], report["N4s_over_N3_last"],
    sorted(used), report["N3_over_N3ref_insample_all"]), flush=True)
json.dump(report, open(os.path.join(OUTD, "%s_report.json" % TAG), "w"), indent=1)
