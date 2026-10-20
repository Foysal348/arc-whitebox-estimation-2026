"""Merge X12 kernel outputs (KIND F / P / M) and print paired score ratios vs ref (= e30kc1n2t) plus the fit CV reports.
Usage: python kaggle/x12/summarize_x12.py data/kaggle_out/x12_f data/kaggle_out/x12_p data/kaggle_out/x12_m
Per net score = mse * max(0.1, util); 'score' = mean mse * max(0.1, mean util) over the common nets."""
import sys, os, glob, json
import numpy as np

dirs = sys.argv[1:]
res = {}
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, "eval", "**", "eval_*.json"), recursive=True)):
        for t, rows in json.load(open(f)).items():
            for i, r in rows.items():
                if "mse" not in r:
                    continue
                old = res.setdefault(t, {}).get(int(i))
                if old is not None and t == "ref" and abs(old["mse"] / r["mse"] - 1) > 1e-6:
                    print("WARNING ref differs between kernels on net", i, old["mse"], r["mse"])
                res[t][int(i)] = r
print({t: len(v) for t, v in res.items()})
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, "tables", "*_report.json"))):
        r = json.load(open(f))
        print("CV %s (%d nets): T2 cv/N2 all %.4f last %.4f | N3 cv/N2 all %.4f last %.4f | N3/T2 all %.4f last %.4f" % (
            r["tag"], r["nets"], r["T2"]["cv_over_n2_all"], r["T2"]["cv_over_n2_last"], r["N3"]["cv_over_n2_all"],
            r["N3"]["cv_over_n2_last"], r["N3_over_T2_all"], r["N3_over_T2_last"]))
arms = ["ref"] + sorted(t for t in res if t != "ref")
rng = np.random.default_rng(0)
out = {}
for label, sub in (("held-out 70-99", range(70, 100)), ("in-sample 0-69", range(0, 70))):
    out[label] = {}
    R = res["ref"]
    for t in arms:
        A = res[t]
        common = [i for i in sub if i in A and i in R]
        if not common:
            continue
        rm = np.mean([R[i]["mse"] for i in common]); ru = np.mean([R[i]["util"] for i in common]); rs = rm * max(0.1, ru)
        m = np.mean([A[i]["mse"] for i in common]); u = np.mean([A[i]["util"] for i in common]); s = m * max(0.1, u)
        ps = np.array([A[i]["mse"] * max(0.1, A[i]["util"]) / (R[i]["mse"] * max(0.1, R[i]["util"])) for i in common])
        lr = np.log(ps); boot = np.array([rng.choice(lr, len(lr)).mean() for _ in range(4000)])
        lo, hi = np.exp(np.quantile(boot, [.05, .95]))
        out[label][t] = dict(raw=m, util=u, score=s, ref_score=rs, score_ratio=s / rs, raw_ratio=m / rm, util_ratio=u / ru,
                             geo=float(np.exp(lr.mean())), geo90=[float(lo), float(hi)], wins=int((ps < 1).sum()), n=len(common))
        print(f"[{label}] {t:6s} n={len(common):2d} raw {m:.4e} util {u:.4f} score {s:.4e} | score x{s / rs:.4f} raw x{m / rm:.4f} "
              f"util x{u / ru:.4f} geo x{np.exp(lr.mean()):.4f} (90% {lo:.4f}-{hi:.4f}) wins {int((ps < 1).sum())}/{len(common)}")
json.dump({"dirs": dirs, "summary": out, "per_net": {t: {str(i): res[t][i] for i in sorted(res[t])} for t in arms}},
          open(os.path.join(dirs[0], "x12_merged.json"), "w"), indent=1)
