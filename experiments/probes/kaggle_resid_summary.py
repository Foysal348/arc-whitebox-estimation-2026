"""Residual / wall-time distribution of Kaggle runner outputs (summary.json per run directory).
Usage: python experiments/probes/kaggle_resid_summary.py data/kaggle_out/<dir> [...]
"""
import json, sys, glob, os
import numpy as np

for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "summary.json")):
        S = json.load(open(f))
        for tag, r in S.items():
            per = r["per_mlp"]
            wall = np.array([p[2] for p in per]); res = np.array([p[3] for p in per])
            q = lambda a, x: float(np.quantile(a, x))
            print(f"{os.path.basename(d):18s} {tag:10s} adj {r['adjusted_final_layer_score']:.4e} raw {r['final_layer_mse']:.4e} "
                  f"util {r['mean_compute_utilization']:.4f} failed {r['n_failed_mlps']} | residual mean {res.mean():.3f} p50 {q(res, .5):.3f} "
                  f"p90 {q(res, .9):.3f} max {res.max():.3f} | wall mean {wall.mean():.0f} p90 {q(wall, .9):.0f} max {wall.max():.0f}")
