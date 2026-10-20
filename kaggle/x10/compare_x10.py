"""Compare X10 eval arms (x10_driver.py MODE=eval outputs eval_*.json, recursively under the given dirs) per net index.
Optionally adds harness runs (summary.json per_mlp; index = position) as extra arms with --harness DIR:TAG:NEWTAG.
Reports, for all nets present and for nets 70-99 (out of sample for the refit tables): raw MSE, util, score
(mean MSE x max(0.1, mean util)), score ratio vs --ref, geo-mean MSE ratio with bootstrap 90 % interval, wins.
Usage: python kaggle/x10/compare_x10.py [--ref base] [--harness data/kaggle_out/e30k31_1:e30k40_k31_025:h_base] DIR [...]
"""
import json, sys, glob, os
import numpy as np

args = sys.argv[1:]
ref = "base"; harness = []
while args and args[0].startswith("--"):
    if args[0] == "--ref":
        ref = args[1]
    elif args[0] == "--harness":
        harness.append(args[1])
    args = args[2:]
arms = {}
for d in args:
    for f in sorted(glob.glob(os.path.join(d, "**", "eval_*.json"), recursive=True)):
        for tag, rows in json.load(open(f)).items():
            for i, r in rows.items():
                if "mse" in r:
                    arms.setdefault(tag, {})[int(i)] = r
for h in harness:
    d, tag, new = h.split(":")
    r = json.load(open(os.path.join(d, "summary.json")))[tag]
    arms[new] = {i: {"name": p[0], "mse": p[4], "util": p[1]} for i, p in enumerate(r["per_mlp"]) if p[5] is None}
# name consistency
for t, rows in arms.items():
    for i, r in rows.items():
        for t2, rows2 in arms.items():
            if i in rows2 and rows2[i]["name"] != r["name"]:
                sys.exit(f"name mismatch at {i}: {t}={r['name']} {t2}={rows2[i]['name']}")
if ref not in arms:
    sys.exit(f"ref {ref} missing; arms {sorted(arms)}")
rng = np.random.default_rng(0)


def table(sub, label):
    common = sorted(set.intersection(*[set(v) for v in arms.values()]) & sub)
    if not common:
        print(f"[{label}] no common nets"); return
    R = arms[ref]
    rm = np.mean([R[i]["mse"] for i in common]); ru = np.mean([R[i]["util"] for i in common]); rs = rm * max(0.1, ru)
    print(f"\n[{label}] {len(common)} nets common to all arms ({common[0]}..{common[-1]}); ref {ref}: raw {rm:.4e} util {ru:.4f} score {rs:.4e}")
    for t in sorted(arms, key=lambda t: np.mean([arms[t][i]["mse"] for i in common])):
        A = arms[t]
        m = np.mean([A[i]["mse"] for i in common]); u = np.mean([A[i]["util"] for i in common]); s = m * max(0.1, u)
        lr = np.log(np.array([A[i]["mse"] / R[i]["mse"] for i in common]))
        boot = np.array([rng.choice(lr, len(lr)).mean() for _ in range(2000)])
        print(f"  {t:14s} raw {m:.4e} util {u:.4f} score {s:.4e} | score ratio {s / rs:.4f} raw ratio {m / rm:.4f} | geo {np.exp(lr.mean()):.4f} "
              f"(90% {np.exp(np.quantile(boot, .05)):.4f}-{np.exp(np.quantile(boot, .95)):.4f}) wins {int((lr < 0).sum())}/{len(lr)}")


allidx = set().union(*[set(v) for v in arms.values()])
table(allidx, "all")
table(set(range(70, 100)), "70-99 (out of sample)")
table(set(range(0, 70)), "0-69 (in sample)")
