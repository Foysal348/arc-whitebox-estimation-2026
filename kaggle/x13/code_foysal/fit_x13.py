"""X13 table refit on many nets (20 Oct). Same recipe as kaggle/x10/fit_x10.py (T2 = N2: 87 e30_a features + 7 h_k*st + 7
h_k*sd, standardised ridge, intercept unpenalised, per-layer lambda by grouped 5-fold CV over nets, final fit on all), but
fed by per-group Gram sums (stats_x13.py) so hundreds of nets fit in seconds.
Usage: python fit_x13.py <out_dir> <N2.json> <stats.npz> [<stats.npz> ...]     (env EST_SRC = estimator file for CORR_FEATS)
Fits:  big  = mini 0-69 + full nets (groups m*, u*)  -> out_dir/N2big.json   (the candidate)
       mini = mini 0-69 only (= the x11 C2R refit)   -> out_dir/N2mini.json  (diagnostic)
       full = full nets only                          -> out_dir/N2full.json  (diagnostic)
Report: CV one-step SSE per layer, and the one-step SSE on the HELD-OUT mini nets 70-99 (group ho, never fitted) of
zero / e30_a original / N2 / mini / full / big -> out_dir/fit_report.json + stdout.
"""
import os, sys, json
import numpy as np

OUTD = sys.argv[1]; N2F = sys.argv[2]; SFILES = sys.argv[3:]
os.makedirs(OUTD, exist_ok=True)
NF = 101


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
NAMES = FEATS87 + ["h%d*st" % k for k in range(7)] + ["h%d*sd" % k for k in range(7)]
I1 = FEATS87.index("1")
n2 = json.load(open(N2F))
assert n2["feats"] == NAMES, "N2 feature order differs"
BETA_N2 = [np.asarray(r, dtype=np.float64) for r in n2["beta"]]
BETA_ORIG = [np.concatenate([np.asarray(r, dtype=np.float64), np.zeros(14)]) for r in BETA87]
BETA_ZERO = [np.zeros(NF) for _ in range(16)]

# ---- merge groups
S = {}
seen = {}
for f in SFILES:
    z = np.load(f)
    groups = sorted({k.split("__")[0] for k in z.files if "__" in k})
    for g in groups:
        ids = [int(x) for x in z[g + "__ids"]]
        for i in ids:
            assert i not in seen, f"net {i} in {f} and {seen[i]}"
            seen[i] = f
        a = S.setdefault(g, {"G": 0, "B": 0, "YY": 0, "SX": 0, "SY": 0, "NN": 0, "ids": []})
        for k in ("G", "B", "YY", "SX", "SY", "NN"):
            a[k] = a[k] + z[g + "__" + k]
        a["ids"] += ids
for g in sorted(S):
    print("group", g, "nets", len(S[g]["ids"]), "ids", min(S[g]["ids"]), "-", max(S[g]["ids"]))
assert all(70 <= i <= 99 for i in S.get("ho", {"ids": []})["ids"])
assert not any(70 <= i <= 99 for g in S if g != "ho" for i in S[g]["ids"]), "held-out mini net in a fit group"


def tot(gs):
    gs = [g for g in gs if g in S]
    z = {"G": np.zeros((16, NF, NF)), "B": np.zeros((16, NF)), "YY": np.zeros(16), "SX": np.zeros((16, NF)), "SY": np.zeros(16), "NN": np.zeros(16)}
    return {k: sum((S[g][k] for g in gs), z[k]) for k in z}


def solve(T, lam):
    g, b, sx, sy, n = T["G"], T["B"], T["SX"], T["SY"], T["NN"]
    cols = list(range(NF)); out = []
    for l in range(16):
        c = [k for k in cols if k != I1]
        mx = sx[l, c] / n[l]; my = sy[l] / n[l]
        Gc = g[l][np.ix_(c, c)] - n[l] * np.outer(mx, mx)
        bc = b[l, c] - n[l] * mx * my
        sd = np.sqrt(np.maximum(np.diag(Gc) / n[l], 0.0))
        keep = sd > 1e-12 * max(1.0, sd.max())
        beta = np.zeros(NF)
        if keep.any():
            ck = np.where(keep)[0]
            Sd = sd[ck]
            A = Gc[np.ix_(ck, ck)] / np.outer(Sd, Sd) / n[l]
            r = bc[ck] / Sd / n[l]
            w = np.linalg.solve(A + lam * np.eye(len(ck)), r) if lam > 0 else np.linalg.lstsq(A, r, rcond=None)[0]
            bk = w / Sd
            for j, kk in enumerate(ck):
                beta[c[kk]] = bk[j]
            icpt = my - mx[ck] @ bk
        else:
            icpt = my
        beta[I1] = icpt
        out.append(beta)
    return out


def sse(T, betas):
    return np.array([T["YY"][l] - 2 * betas[l] @ T["B"][l] + betas[l] @ T["G"][l] @ betas[l] for l in range(16)])


LAMS = [0.0] + list(10.0 ** np.arange(-8, 1.01, 0.5))
FITS = {"big": ["m", "u"], "mini": ["m"], "full": ["u"]}
report = {"groups": {g: len(S[g]["ids"]) for g in S}, "fits": {}}
tables = {}
for name, srcs in FITS.items():
    gs_all = [s + str(f) for s in srcs for f in range(5)]
    if not any(g in S for g in gs_all):
        continue
    cv = np.zeros((len(LAMS), 16)); cv_n2 = np.zeros(16); cv_orig = np.zeros(16)
    for f in range(5):
        te = tot([s + str(f) for s in srcs]); tr = tot([s + str(h) for s in srcs for h in range(5) if h != f])
        for a, lam in enumerate(LAMS):
            cv[a] += sse(te, solve(tr, lam))
        cv_n2 += sse(te, BETA_N2); cv_orig += sse(te, BETA_ORIG)
    best = cv.argmin(0); lam_l = [LAMS[a] for a in best]
    Tall = tot(gs_all)
    fits = {lam: solve(Tall, lam) for lam in set(lam_l)}
    beta = [fits[lam_l[l]][l] for l in range(16)]
    tables[name] = beta
    nets = sorted(i for g in gs_all if g in S for i in S[g]["ids"])
    json.dump({"feats": NAMES, "beta": [list(map(float, b)) for b in beta], "lam": lam_l, "nets": "x13 %s: %d nets (%d mini 0-69, %d full)" % (
        name, len(nets), sum(i < 1000 for i in nets), sum(i >= 1000 for i in nets))}, open(os.path.join(OUTD, f"N2{name}.json"), "w"))
    cvb = cv[best, np.arange(16)]
    report["fits"][name] = {"nets": len(nets), "lam": lam_l, "cv_sse": cvb.tolist(), "cv_sse_N2": cv_n2.tolist(), "cv_sse_orig": cv_orig.tolist()}
    print("\n== fit %s: %d nets; per layer CV one-step SSE: refit/orig, refit/N2 (N2 is in-sample on mini 0-69)" % (name, len(nets)))
    for l in range(16):
        print("  L%2d lam %-8.1e cv/orig %.4f  cv/N2 %.4f" % (l, lam_l[l], cvb[l] / cv_orig[l], cvb[l] / cv_n2[l]))
    print("  all layers cv/orig %.4f cv/N2 %.4f | last layer cv/orig %.4f cv/N2 %.4f" % (
        cvb.sum() / cv_orig.sum(), cvb.sum() / cv_n2.sum(), cvb[15] / cv_orig[15], cvb[15] / cv_n2[15]))
    # CV restricted to the full-net folds only (out-of-sample for N2 too): fair N2 vs refit comparison
    if "u" in srcs and name == "big":
        cvu = np.zeros((len(LAMS), 16)); n2u = np.zeros(16)
        for f in range(5):
            te = tot(["u%d" % f]); tr = tot([s + str(h) for s in srcs for h in range(5) if h != f])
            for a, lam in enumerate(LAMS):
                if lam in lam_l:
                    cvu[a] += sse(te, solve(tr, lam))
            n2u += sse(te, BETA_N2)
        cu = cvu[best, np.arange(16)]
        report["fits"][name]["cv_full_folds_sse"] = cu.tolist(); report["fits"][name]["full_folds_sse_N2"] = n2u.tolist()
        print("  [full-net folds only, both out-of-sample] big/N2 all layers %.4f, last layer %.4f; per layer: %s" % (
            cu.sum() / n2u.sum(), cu[15] / n2u[15], " ".join("%.3f" % (cu[l] / n2u[l]) for l in range(16))))

if "ho" in S:
    H = S["ho"]; Th = tot(["ho"])
    cand = {"zero": BETA_ZERO, "orig": BETA_ORIG, "N2": BETA_N2, **tables}
    e = {k: sse(Th, v) for k, v in cand.items()}
    report["holdout"] = {"nets": sorted(H["ids"]), **{k: v.tolist() for k, v in e.items()}}
    print("\n== HELD-OUT mini 70-99 (%d nets, never fitted): one-step SSE ratio vs N2 per layer" % len(H["ids"]))
    print("  layer " + " ".join("%7s" % k for k in cand))
    for l in range(16):
        print("  L%2d   " % l + " ".join("%7.4f" % (e[k][l] / e["N2"][l]) for k in cand))
    print("  all   " + " ".join("%7.4f" % (e[k].sum() / e["N2"].sum()) for k in cand))
json.dump(report, open(os.path.join(OUTD, "fit_report.json"), "w"), indent=1)
print("tables written:", sorted(tables))
