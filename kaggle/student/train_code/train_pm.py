"""Train the Round-20 pair-memory student (PM364) against V56 teacher labels (11 Oct 2026). GPU if available.

Labels: net_<seed>.npz files from teacher_dump.py, in any directories under --data (searched recursively).
Weights: regenerated from the seed exactly as the teacher did (numpy default_rng(seed), N(0, 2/n), float32).
Splits: train = seeds 100000-199999, val = 200000-299999, test = 300000-399999.

Loss: free-running rollouts from layer 1 (no teacher forcing):
    L = 8 L_f + sum_l [ L_mu + 0.25 L_C + 2 w_l L_D + 2 w_l L_d + 0.25 L_4 ] + 1e-3 L_reg
with the normalisations of the plan:
- L_f: final-layer MSE to the teacher / 1e-8;
- L_mu: (mu − mu*)/σ*;
- L_C: sampled off-diagonal rows / (σ_i σ_j);
- L_D: sampled D21 rows / (σ_i² σ_j);
- L_d: d3 / σ³;
- L_4: g4row / σ⁴;
- w_l = 0.25 for l ≤ 8, 1 for l ≥ 9.
Usage:
    python train_pm.py --data DIR --est v56_dump.py --out OUTDIR [--epochs 16] [--seed 11] [--lr 2e-3] [--accum 8]
                       [--ablate none|h2|o|g] [--eval-only THETA.npy] [--max-train N]
"""
import os, sys, glob, json, time, argparse, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pm_student import PairMemoryStudent

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--est", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=16); ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--ablate", default="none"); ap.add_argument("--eval-only", default="")
ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--max-eval", type=int, default=0)
ap.add_argument("--val-from-train", type=int, default=0); ap.add_argument("--init", default=""); ap.add_argument("--patience", type=int, default=4)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
files = {}
for f in glob.glob(os.path.join(args.data, "**", "net_*.npz"), recursive=True):
    s = int(os.path.basename(f)[4:-4]); files[s] = f
split = {"train": sorted(s for s in files if 100000 <= s < 200000), "val": sorted(s for s in files if 200000 <= s < 300000),
         "test": sorted(s for s in files if 300000 <= s < 400000)}
if args.val_from_train:   # hold out the last K train seeds as validation (never trained on)
    split["val"] = split["train"][-args.val_from_train:]; split["train"] = split["train"][:-args.val_from_train]
if args.max_train: split["train"] = split["train"][:args.max_train]
if args.max_eval: split["val"] = split["val"][:args.max_eval]; split["test"] = split["test"][:args.max_eval]
print("device", dev, {k: len(v) for k, v in split.items()}, flush=True)
n, L = 1024, 16
model = PairMemoryStudent(args.est, device=dev, use_h2=args.ablate != "h2", use_o=args.ablate != "o", use_g=args.ablate != "g")
if args.init: model.theta.data.copy_(torch.tensor(np.load(args.init), device=dev))
if args.eval_only: model.theta.data.copy_(torch.tensor(np.load(args.eval_only), device=dev))


def weights(seed):
    rng = np.random.default_rng(seed)
    return [torch.tensor((rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32), device=dev) for _ in range(L)]


def labels(seed):
    d = np.load(files[seed]); g = lambda k: torch.tensor(d[k], device=dev) if k in d.files else None
    lab = {"final": g("final")}
    for l in range(1, L + 1):
        lab[l] = {k: g(f"L{l}_{k}") for k in ("mu_pre", "var", "d3", "g4row", "post_mean", "rows", "D21_rows", "C_rows")}
    return lab


def loss_fn(out, traj, lab, parts=False):
    eps = 1e-12
    Lf = ((out[-1] - lab["final"][-1]) ** 2).mean() / 1e-8
    tot = 8.0 * Lf; comp = {"final_mse": float(((out[-1] - lab["final"][-1]) ** 2).mean()), "D": [], "d": []}
    for l, rec in enumerate(traj, 1):
        t = lab[l]; sg = torch.sqrt(t["var"]); w = 0.25 if l <= 8 else 1.0
        Lmu = (((rec["mu_pre"] - t["mu_pre"]) / (sg + eps)) ** 2).mean()
        idx = t["rows"].long()
        offm = torch.ones(len(idx), n, device=dev); offm[torch.arange(len(idx)), idx] = 0
        term = Lmu
        if t["C_rows"] is not None and "C_pre" in rec:
            C_s = rec["C_pre"][idx]; Ct = t["C_rows"]
            term = term + 0.25 * ((((C_s - Ct) * offm) / (sg[idx][:, None] * sg[None, :] + eps)) ** 2).mean()
        if l > 1 and "D21" in rec and t["D21_rows"] is not None and t["d3"] is not None:
            Ds = rec["D21"][idx]; Dt = t["D21_rows"]
            sc = (sg[idx] ** 2)[:, None] * sg[None, :]
            LD = ((((Ds - Dt) * offm) / (sc + eps)) ** 2).mean()
            Ld = (((rec["d3"] - t["d3"]) / (sg ** 3 + eps)) ** 2).mean()
            L4 = (((rec["g4row"] - t["g4row"]) / (sg ** 4 + eps)) ** 2).mean()
            term = term + 2 * w * LD + 2 * w * Ld + 0.25 * L4
            if parts:
                comp["D"].append(float(torch.linalg.norm((Ds - Dt) * offm) / (torch.linalg.norm(Dt * offm) + 1e-30)))
                comp["d"].append(float(torch.linalg.norm(rec["d3"] - t["d3"]) / (torch.linalg.norm(t["d3"]) + 1e-30)))
        tot = tot + term
    rho, eta, o, g = model.heads(torch.zeros(1, 12, device=dev))  # regulariser on the head outputs at the feature origin
    tot = tot + 1e-3 * ((rho - 1) ** 2 + eta ** 2 + o ** 2 + g ** 2).sum()
    return (tot, comp) if parts else tot


def evaluate(seeds, tag):
    model.eval(); mses, Ds, ds, fails = [], [], [], 0
    with torch.no_grad():
        for s in seeds:
            out, traj = model(weights(s), return_traj=True)
            if not torch.isfinite(out).all():
                fails += 1; continue
            _, comp = loss_fn(out, traj, labels(s), parts=True)
            mses.append(comp["final_mse"]); Ds.append(comp["D"]); ds.append(comp["d"])
    Dm = np.sqrt(np.mean(np.square(np.array(Ds)), 0)) if Ds else []; dm = np.sqrt(np.mean(np.square(np.array(ds)), 0)) if ds else []
    res = {"tag": tag, "n": len(seeds), "fails": fails, "mse_to_teacher": float(np.mean(mses)) if mses else None,
           "max_net_mse": float(np.max(mses)) if mses else None,
           "D21_rel_by_layer(L2-16)": [round(float(x), 4) for x in Dm], "d3_rel_by_layer(L2-16)": [round(float(x), 4) for x in dm]}
    print(json.dumps(res), flush=True)
    model.train(); return res


if args.eval_only:
    r = {"val": evaluate(split["val"], "val"), "test": evaluate(split["test"], "test")}
    json.dump(r, open(os.path.join(args.out, "eval.json"), "w"), indent=1); sys.exit(0)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
steps_total = args.epochs * max(1, len(split["train"]) // args.accum); warm = min(500, max(1, steps_total // 10))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, k / max(1, steps_total))))))
hist = [evaluate(split["val"], "val@0")]
best = hist[0]["mse_to_teacher"] or 1e9; bad = 0
np.save(os.path.join(args.out, "theta_best.npy"), model.theta.detach().cpu().numpy())
rng = np.random.default_rng(args.seed)
for ep in range(args.epochs):
    t0 = time.time(); order = rng.permutation(split["train"]); opt.zero_grad(); run = 0.0; nb = 0
    for i, s in enumerate(order):
        out, traj = model(weights(int(s)), return_traj=True)
        loss = loss_fn(out, traj, labels(int(s))) / args.accum
        if not torch.isfinite(loss):
            opt.zero_grad(); continue
        loss.backward(); run += float(loss) * args.accum; nb += 1
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step(); sched.step(); opt.zero_grad()
    r = evaluate(split["val"], f"val@{ep + 1}"); r["train_loss"] = run / max(1, nb); r["epoch_s"] = time.time() - t0; hist.append(r)
    print(f"epoch {ep + 1}: train loss {run / max(1, nb):.4f}, val mse {r['mse_to_teacher']}, {time.time() - t0:.0f}s", flush=True)
    if r["mse_to_teacher"] is not None and r["mse_to_teacher"] < best:
        best = r["mse_to_teacher"]; bad = 0; np.save(os.path.join(args.out, "theta_best.npy"), model.theta.detach().cpu().numpy())
    else:
        bad += 1
        if bad >= args.patience: print("early stop", flush=True); break
    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
model.theta.data.copy_(torch.tensor(np.load(os.path.join(args.out, "theta_best.npy")), device=dev))
final = {"history": hist, "test": evaluate(split["test"], "test(best)")}
json.dump(final, open(os.path.join(args.out, "history.json"), "w"), indent=1)
