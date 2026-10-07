"""Deep analysis of the V55 kappa4-corrector Kaggle runs (7 Oct): 100 public mini MLPs, baked 1e9-sample truth.
MSE is deterministic per MLP, so variants compare across runners; residual time compares only within one runner session.
Sections: (1) paired vs the right base (B-class vs base_v53rfb2, A-class vs base_v52Ar352): geo ratio, bootstrap 90% CI,
wins, score ratio; (2) per-MLP gain distribution and worst MLPs; (3) does the gain depend on how hard the network is
(log-ratio vs log base MSE, vs base util); (4) head-to-head between corrector variants; (5) residual / wall per runner session
and the residual added by the corrector; (6) projected graded scores from the graded bases.
Usage: python experiments/probes/kaggle_deep_v55.py data/kaggle_out/v55_a data/kaggle_out/v55_b ...
"""
import json, sys, glob, os
import numpy as np

GRADED_BASE = {"base_v53rfb2": 4.0995e-9, "base_v52Ar352": 4.0784e-9}
GRADED = {"v55k4c": 3.8762e-9, "v55Ak4c": 3.8582e-9}
A_CLASS = {"base_v52Ar352", "v55Ak4c", "v55Alean"}
runs = {}
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "summary.json")):
        for tag, r in json.load(open(f)).items():
            runs[tag] = (os.path.basename(d.rstrip("/\\")), r)
print("tags:", {t: d for t, (d, _) in runs.items()})
rng = np.random.default_rng(0)


def table(tag):
    r = runs[tag][1]
    return {p[0]: dict(util=p[1], wall=p[2], resid=p[3], mse=p[4], err=p[5]) for p in r["per_mlp"]}


def score(tag):
    r = runs[tag][1]
    return r["final_layer_mse"] * max(0.1, r["mean_compute_utilization"])


def paired(a, b):
    A, B = table(a), table(b)
    names = sorted(n for n in A if n in B and A[n]["err"] is None and B[n]["err"] is None)
    lr = np.log(np.array([A[n]["mse"] / B[n]["mse"] for n in names]))
    boot = np.array([rng.choice(lr, len(lr)).mean() for _ in range(4000)])
    return names, lr, np.exp(lr.mean()), np.exp(np.quantile(boot, [0.05, 0.95])), int((lr < 0).sum())


print("\n(1) paired vs base")
for tag in sorted(runs):
    base = "base_v52Ar352" if tag in A_CLASS else "base_v53rfb2"
    if tag == base or base not in runs:
        continue
    names, lr, g, ci, w = paired(tag, base)
    sr = score(tag) / score(base)
    proj = GRADED_BASE[base] * sr
    print(f"  {tag:14s} vs {base:14s} n {len(names):3d} geo {g:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] wins {w:3d}/{len(names)} "
          f"| raw {runs[tag][1]['final_layer_mse']:.4e} util {runs[tag][1]['mean_compute_utilization']:.4f} failed "
          f"{runs[tag][1]['n_failed_mlps']} | score ratio {sr:.4f} -> projected graded {proj:.4e}"
          + (f" (actual {GRADED[tag]:.4e}, ratio {GRADED[tag] / GRADED_BASE[base]:.4f})" if tag in GRADED else ""))

print("\n(2)+(3) per-MLP gain distribution and dependence on network hardness")
for tag in sorted(runs):
    base = "base_v52Ar352" if tag in A_CLASS else "base_v53rfb2"
    if tag == base or base not in runs:
        continue
    names, lr, *_ = paired(tag, base)
    B = table(base)
    lb = np.log(np.array([B[n]["mse"] for n in names])); ub = np.array([B[n]["util"] for n in names])
    q = np.quantile(np.exp(lr), [0.05, 0.25, 0.5, 0.75, 0.95])
    worst = sorted(zip(lr, names))[-3:]
    print(f"  {tag:14s} ratio q05 {q[0]:.3f} q25 {q[1]:.3f} med {q[2]:.3f} q75 {q[3]:.3f} q95 {q[4]:.3f} | >1.02: "
          f"{int((lr > np.log(1.02)).sum())} MLPs | corr(log ratio, log base MSE) {np.corrcoef(lr, lb)[0, 1]:+.2f}, "
          f"vs base util {np.corrcoef(lr, ub)[0, 1]:+.2f} | worst {[(n, round(float(np.exp(x)), 3)) for x, n in worst]}")
    # hardness terciles
    t = np.quantile(lb, [1 / 3, 2 / 3])
    for lo, hi, lab in ((-np.inf, t[0], "easy"), (t[0], t[1], "mid"), (t[1], np.inf, "hard")):
        m = (lb > lo) & (lb <= hi)
        print(f"      {lab:4s} tercile: geo {np.exp(lr[m].mean()):.4f} (n {m.sum()})")

print("\n(4) head-to-head between corrector variants (same class)")
cor = [t for t in runs if not t.startswith("base_")]
for i, a in enumerate(sorted(cor)):
    for b in sorted(cor)[i + 1:]:
        if (a in A_CLASS) != (b in A_CLASS):
            continue
        names, lr, g, ci, w = paired(a, b)
        print(f"  {a:14s} vs {b:14s} geo {g:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] wins {w}/{len(names)}")

print("\n(5) residual / wall per runner session (seconds; Kaggle CPU, no residual cap)")
by_dir = {}
for tag, (d, r) in runs.items():
    by_dir.setdefault(d, []).append(tag)
for d, tags in sorted(by_dir.items()):
    for tag in tags:
        T = table(tag); res = np.array([v["resid"] for v in T.values()]); wall = np.array([v["wall"] for v in T.values()])
        print(f"  [{d:14s}] {tag:14s} resid p50 {np.median(res):.3f} p90 {np.quantile(res, .9):.3f} max {res.max():.3f} | "
              f"wall p50 {np.median(wall):.0f} max {wall.max():.0f} | elapsed {runs[tag][1]['elapsed_s'] / 3600:.2f} h")
    if len(tags) == 2:
        a, b = tags
        A, B = table(a), table(b)
        names = [n for n in A if n in B]
        rr = np.array([A[n]["resid"] / B[n]["resid"] for n in names])
        print(f"      same-session residual ratio {a}/{b}: median {np.median(rr):.3f} (q10 {np.quantile(rr, .1):.3f}, q90 {np.quantile(rr, .9):.3f})")
