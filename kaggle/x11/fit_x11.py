"""X10 table refit (light numpy, local). Reads the teacher-forced dumps net_XXX.npz (kaggle/x10/dump_* outputs) and fits,
per layer, a ridge of y = truth - base (base = the chain's mean before the table, input mean forced to the truth) on
  T0: the 87 e30_a features (same names and order as CORR_FEATS)
  T1: T0 + h_k*st (k = 0..6; st = the corrector's k4star, zero where the corrector is off)
  T2: T1 + h_k*sd (sd = (g4mean - g4row) / var^2, the corrector's own kappa4 change)
Columns standardised (intercept '1' unpenalised), lambda per layer by grouped 5-fold CV over nets (blocks of nets).
Usage: python kaggle/x10/fit_x10.py <dump_dir> [out_dir=kaggle/x10/tables]
"""
import os, sys, glob, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DD = sys.argv[1]
OUTD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "tables")
NETS = os.environ.get("NETS", "0-69")
lo, hi = map(int, NETS.split("-"))
os.makedirs(OUTD, exist_ok=True)


def module_tables():
    """CORR_FEATS / CORR_BETA_NEW of the estimator file, parsed without importing it (no flopscope needed)."""
    src = open(os.environ.get("EST_SRC", os.path.join(ROOT, "experiments/variants/e30k_x10.py")), encoding="utf-8").read()
    ns = {}
    for name in ("CORR_FEATS", "CORR_BETA_NEW"):
        i = src.index("\n" + name + " = [") + 1
        j = src.index("\n]\n", i) + 2
        exec(src[i:j], ns)
    return ns["CORR_FEATS"], ns["CORR_BETA_NEW"]


FEATS87, BETA87 = module_tables()
assert len(FEATS87) == 87 and len(BETA87) == 16


def hermite(a, kmax=6):
    H = [np.ones_like(a), a]
    for k in range(2, kmax + 1):
        H.append(a * H[k - 1] - (k - 1) * H[k - 2])
    return H


files = sorted(glob.glob(os.path.join(DD, "**", "net_*.npz"), recursive=True))
nets = {}
for f in files:
    i = int(re.search(r"net_(\d+)\.npz", f).group(1))
    if lo <= i <= hi:
        nets[i] = f
ids = sorted(nets)
print("nets", len(ids), ids[:3], "...", ids[-3:])

# per layer, per net: Gram sums for the full design [87 | 7 st | 7 sd]
NF = 87 + 14
G = np.zeros((16, len(ids), NF, NF)); B = np.zeros((16, len(ids), NF)); YY = np.zeros((16, len(ids)))
SX = np.zeros((16, len(ids), NF)); SY = np.zeros((16, len(ids))); NN = np.zeros((16, len(ids)))
E_orig = np.zeros((16, len(ids))); E_zero = np.zeros((16, len(ids)))
for c, i in enumerate(ids):
    z = np.load(nets[i])
    F = z["F"].astype(np.float64); tr = z["truth"].astype(np.float64)
    base = np.concatenate([z["base_asm"], z["base"][15:16]]).astype(np.float64)
    st = z["st"].astype(np.float64); sp = z["sp"].astype(np.float64); al = z["alpha"].astype(np.float64)
    sd = (z["g4c"].astype(np.float64) - z["g4"].astype(np.float64)) / z["var"].astype(np.float64) ** 2
    for l in range(16):
        H = hermite(al[l])
        X = np.concatenate([F[l], np.stack([sp[l] * H[k] * st[l] for k in range(7)], 1),
                            np.stack([sp[l] * H[k] * sd[l] for k in range(7)], 1)], 1)
        y = tr[l] - base[l]
        G[l, c] = X.T @ X; B[l, c] = X.T @ y; YY[l, c] = y @ y; SX[l, c] = X.sum(0); SY[l, c] = y.sum(); NN[l, c] = len(y)
        E_orig[l, c] = np.sum((y - F[l] @ np.asarray(BETA87[l])) ** 2); E_zero[l, c] = y @ y
    if c == 0:
        dchk = np.abs(F[8] @ np.asarray(BETA87[8]) - z["delta"][8]).max()
        print("net", i, "check: table delta recompute max|diff| L8 = %.2e (|delta| max %.2e); st nonzero layers:" % (
            dchk, np.abs(z["delta"][8]).max()), [l for l in range(16) if np.any(st[l] != 0)])

COLS = {"T0": list(range(87)), "T1": list(range(94)), "T2": list(range(101))}
NAMES = FEATS87 + ["h%d*st" % k for k in range(7)] + ["h%d*sd" % k for k in range(7)]
I1 = FEATS87.index("1")
LAMS = [0.0] + list(10.0 ** np.arange(-8, 1.01, 0.5))
folds = [ids[j::1][int(len(ids) * f / 5):int(len(ids) * (f + 1) / 5)] for j in [0] for f in range(5)]   # contiguous net blocks
fold_of = {i: f for f, blk in enumerate(folds) for i in blk}


def solve(idx, cols, lam):
    """ridge on nets idx (positions), standardised columns, intercept column I1 unpenalised. Returns raw beta (len cols)."""
    g = G[:, idx].sum(1); b = B[:, idx].sum(1); sx = SX[:, idx].sum(1); sy = SY[:, idx].sum(1); n = NN[:, idx].sum(1)
    out = []
    for l in range(16):
        c = [k for k in cols if k != I1]
        mx = sx[l, c] / n[l]; my = sy[l] / n[l]
        Gc = g[l][np.ix_(c, c)] - n[l] * np.outer(mx, mx)
        bc = b[l, c] - n[l] * mx * my
        sd = np.sqrt(np.maximum(np.diag(Gc) / n[l], 0.0))
        keep = sd > 1e-12 * max(1.0, sd.max())
        beta = np.zeros(len(cols))
        if keep.any():
            ck = np.where(keep)[0]
            S = sd[ck]
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
report = {}
for T, cols in COLS.items():
    cv = np.zeros((len(LAMS), 16))
    for f in range(5):
        te = np.array([p for p, i in enumerate(ids) if fold_of[i] == f]); trn = np.array([p for p, i in enumerate(ids) if fold_of[i] != f])
        for a, lam in enumerate(LAMS):
            cv[a] += sse(te, cols, solve(trn, cols, lam))
    best = cv.argmin(0)
    lam_l = [LAMS[a] for a in best]
    # final fit on all nets with each layer's CV lambda
    fits = {lam: solve(pos_all, cols, lam) for lam in set(lam_l)}
    beta = [list(map(float, fits[lam_l[l]][l])) for l in range(16)]
    cvbest = cv[best, np.arange(16)]
    eo = E_orig.sum(1); ez = E_zero.sum(1)
    report[T] = {"lam": lam_l, "cv_sse": cvbest.tolist(), "orig_sse": eo.tolist(), "zero_sse": ez.tolist()}
    json.dump({"feats": [NAMES[k] for k in cols], "beta": beta, "lam": lam_l, "nets": NETS},
              open(os.path.join(OUTD, T + ".json"), "w"))
    print("\n== %s (%d features): per layer CV one-step SSE ratio vs the original table (and vs no table)" % (T, len(cols)))
    for l in range(16):
        print("  L%2d lam %-8.1e  cv/orig %.4f  cv/zero %.4f  orig/zero %.4f" % (l, lam_l[l], cvbest[l] / eo[l], cvbest[l] / ez[l], eo[l] / ez[l]))
    print("  all-layer cv/orig %.4f   last layer cv/orig %.4f" % (cvbest.sum() / eo.sum(), cvbest[15] / eo[15]))
json.dump(report, open(os.path.join(OUTD, "fit_report.json"), "w"), indent=1)
