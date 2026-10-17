"""Experiment 7: paired 100-MLP comparison of the x7h arms against the e299xrk base (xr_k31_025 arm of xrk31-1).
Fetch first:  python -m kaggle kernels output esfersami50/whest-runner-x7-1 -p data/kaggle_out/x7h_1
              python -m kaggle kernels output esfersami50/whest-runner-xrk31-1 -p data/kaggle_out/xrk31_1
Usage: python kaggle/x7h/compare.py
Per-MLP score = final MSE x max(0.1, FLOPs / 2^41); ratios are geometric means over the paired MLPs.
"""
import json, math, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
base = json.load(open(os.path.join(ROOT, "data/kaggle_out/xrk31_1/summary.json")))["xr_k31_025"]
arms = json.load(open(os.path.join(ROOT, "data/kaggle_out/x7h_1/summary.json")))
b = {r[0]: r for r in base["per_mlp"]}
print(f"base xr_k31_025: score {base['adjusted_final_layer_score']:.4e} raw {base['final_layer_mse']:.4e} util {base['mean_compute_utilization']:.4f}")
for tag, a in arms.items():
    pr = [(r, b[r[0]]) for r in a["per_mlp"] if r[0] in b]
    gm = lambda f: math.exp(sum(math.log(f(x, y)) for x, y in pr) / len(pr))
    s = gm(lambda x, y: (x[4] * max(0.1, x[1])) / (y[4] * max(0.1, y[1])))
    m = gm(lambda x, y: x[4] / y[4]); u = gm(lambda x, y: x[1] / y[1])
    wins = sum(1 for x, y in pr if x[4] * max(0.1, x[1]) < y[4] * max(0.1, y[1]))
    print(f"{tag:12s} score {a['adjusted_final_layer_score']:.4e} raw {a['final_layer_mse']:.4e} util {a['mean_compute_utilization']:.4f} "
          f"failed {a['n_failed_mlps']} | paired n={len(pr)}: score x{s:.4f}, MSE x{m:.4f}, FLOPs x{u:.4f}, better on {wins}")
