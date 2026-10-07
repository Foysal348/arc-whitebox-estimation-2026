"""Paired comparison of Kaggle runner outputs across directories (same public mini-split MLPs, matched by mlp_name).
Every tag in every summary.json is compared with a reference tag (default: base_v53rfb2): geometric-mean MSE ratio, wins,
a bootstrap 90% interval of the mean log-ratio, the score ratio (MSE x max(0.1, util)) and the residual-time profile.
Usage: python experiments/probes/kaggle_pair_compare.py [--ref TAG] data/kaggle_out/<dir> [...]
"""
import json, sys, glob, os
import numpy as np

args = sys.argv[1:]
ref = "base_v53rfb2"
if args and args[0] == "--ref":
    ref, args = args[1], args[2:]
runs = {}
for d in args:
    for f in glob.glob(os.path.join(d, "summary.json")):
        for tag, r in json.load(open(f)).items():
            runs[tag] = (os.path.basename(d), r)
if ref not in runs:
    sys.exit(f"reference {ref} not found; tags: {sorted(runs)}")


def per(r):
    return {p[0]: p for p in r["per_mlp"] if p[5] is None}


R = per(runs[ref][1]); rr = runs[ref][1]
ref_score = rr["final_layer_mse"] * max(0.1, rr["mean_compute_utilization"])
rng = np.random.default_rng(0)
print(f"reference {ref}: raw {rr['final_layer_mse']:.4e} util {rr['mean_compute_utilization']:.4f} score {ref_score:.4e}")
for tag, (d, r) in sorted(runs.items(), key=lambda kv: kv[1][1]["final_layer_mse"]):
    P = per(r)
    names = [n for n in P if n in R]
    lr = np.log(np.array([P[n][4] / R[n][4] for n in names]))
    boot = np.array([rng.choice(lr, len(lr)).mean() for _ in range(2000)])
    res = np.array([p[3] for p in r["per_mlp"]])
    score = r["final_layer_mse"] * max(0.1, r["mean_compute_utilization"])
    print(f"{tag:16s} [{d:22s}] n {len(names):3d} raw {r['final_layer_mse']:.4e} util {r['mean_compute_utilization']:.4f} "
          f"failed {r['n_failed_mlps']} | geo ratio {np.exp(lr.mean()):.4f} (90% {np.exp(np.quantile(boot, .05)):.4f}-"
          f"{np.exp(np.quantile(boot, .95)):.4f}) wins {int((lr < 0).sum())}/{len(lr)} | score ratio {score / ref_score:.4f} "
          f"| resid p50 {np.median(res):.3f} max {res.max():.3f}")
