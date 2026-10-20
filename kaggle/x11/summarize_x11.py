"""Merge X11 eval outputs (pipeline kernel + eval-only rerun) for one candidate and print paired score ratios vs nbN2.
Usage: python kaggle/x11/summarize_x11.py C2 data/kaggle_out/x11_c2 data/kaggle_out/x11_c2e  [> summary]
Per net score = mse * max(0.1, util); 'score' = mean mse * max(0.1, mean util) (grader formula over the subset)."""
import sys, os, glob, json
import numpy as np

cand, dirs = sys.argv[1], sys.argv[2:]
res = {}
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, "eval", "**", "eval_*.json"), recursive=True)):
        for t, rows in json.load(open(f)).items():
            for i, r in rows.items():
                if "mse" in r:
                    res.setdefault(t, {})[int(i)] = r
arms = ["nbN2", cand + "N2", cand + "R"]
rng = np.random.default_rng(0)
out = {}
for label, sub in (("held-out 70-99", range(70, 100)), ("in-sample 0-69", range(0, 70))):
    common = [i for i in sub if all(i in res.get(t, {}) for t in arms)]
    R = res["nbN2"]
    rm = np.mean([R[i]["mse"] for i in common]); ru = np.mean([R[i]["util"] for i in common]); rs = rm * max(0.1, ru)
    print(f"[{label}] {len(common)} nets; nbN2 raw {rm:.4e} util {ru:.4f} score {rs:.4e}")
    out[label] = {"nets": common}
    for t in arms:
        A = res[t]
        m = np.mean([A[i]["mse"] for i in common]); u = np.mean([A[i]["util"] for i in common]); s = m * max(0.1, u)
        ps = np.array([A[i]["mse"] * max(0.1, A[i]["util"]) / (R[i]["mse"] * max(0.1, R[i]["util"])) for i in common])
        lr = np.log(ps); boot = np.array([rng.choice(lr, len(lr)).mean() for _ in range(4000)])
        lo, hi = np.exp(np.quantile(boot, [.05, .95]))
        out[label][t] = dict(raw=m, util=u, score=s, score_ratio=s / rs, raw_ratio=m / rm, util_ratio=u / ru,
                             geo=float(np.exp(lr.mean())), geo90=[float(lo), float(hi)], wins=int((ps < 1).sum()), n=len(common))
        print(f"  {t:6s} score x{s / rs:.4f}  raw x{m / rm:.4f}  util x{u / ru:.4f}  geo x{np.exp(lr.mean()):.4f} (90% {lo:.4f}-{hi:.4f})  wins {int((ps < 1).sum())}/{len(common)}")
    # refit vs no refit, same knob
    A, B = res[cand + "R"], res[cand + "N2"]
    pr = np.array([A[i]["mse"] / B[i]["mse"] for i in common])
    print(f"  {cand}R vs {cand}N2 raw: geo x{np.exp(np.log(pr).mean()):.4f}, wins {int((pr < 1).sum())}/{len(common)}")
    out[label]["refit_vs_norefit_geo"] = float(np.exp(np.log(pr).mean()))
json.dump({"cand": cand, "dirs": dirs, "summary": out, "per_net": {t: {str(i): res[t][i] for i in sorted(res[t])} for t in arms}},
          open(os.path.join(dirs[-1], f"x11_{cand}_merged.json"), "w"), indent=1)
