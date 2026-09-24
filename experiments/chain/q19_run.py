"""Q19 driver: refit the online mean correction of v40fb on its own teacher-forced trajectory.
Each net is a directory with weights_seed*.npz (W, shape (16, n, n), z = h @ W[l]) and truth.npz (all_layer_means (16, n)).

  dump  <out_dir> <net_dir>...   teacher-forced run per net; writes <out_dir>/<net>.npz (feats (16, n, 13), target (16, n))
  fit   <out_dir> <beta.npz> [lam] ridge fit per layer on every dump in out_dir -> beta (16, 13)
  eval  <beta.npz|none> <net_dir>...   free-running final-layer MSE per net (beta 'none' = shipped v40fb, NO_CORR)

Run from whest-starterkit:  python -m uv run python ../experiments/chain/q19_run.py dump ...
"""
import os, sys, types, glob, json, time
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC = "../experiments/variants/estimator_q19corr.py"
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"

def load_net(d):
    W = np.load(glob.glob(os.path.join(d, "weights_seed*.npz"))[0])["W"]
    t = np.load(os.path.join(d, "truth.npz"))
    mlp = MLP(width=W.shape[1], depth=W.shape[0], weights=[fnp.array(W[l]) for l in range(W.shape[0])],
              seed=int(t["mlp_seed"]) if "mlp_seed" in t else 0, name=os.path.basename(os.path.normpath(d)))
    return mlp, np.asarray(t["all_layer_means"], np.float64)

def module():
    mod = types.ModuleType("est_q19"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
    return mod

def run(mod, mlp):
    est = mod.Estimator(); est.setup(_Ctx())
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as b:
        p = np.asarray(est.predict(mlp, 2**41), np.float64)
    return p, b.flops_used

cmd = sys.argv[1]
if cmd == "dump":
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    est = module()
    for d in sys.argv[3:]:
        mlp, truth = load_net(d); name = os.path.basename(os.path.normpath(d)); t0 = time.time()
        os.environ.update(Q19_TRUTH=os.path.join(d, "truth.npz"), Q19_DUMP=os.path.join(out, name + ".npz"), Q19_FORCE="1")
        p, fl = run(est, mlp)
        for k in ("Q19_TRUTH", "Q19_DUMP", "Q19_FORCE"): os.environ.pop(k)
        z = np.load(os.path.join(out, name + ".npz"))
        print(f"dump {name}: feats {z['feats'].shape} target {z['target'].shape} a1 {float(z['a1']):.6f} "
              f"forced-run final MSE {np.mean((p[-1] - truth[-1]) ** 2):.3e} [{time.time()-t0:.0f}s]", flush=True)
elif cmd == "fit":
    out, bpath = sys.argv[2], sys.argv[3]; lam = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(out, "*.npz")))]
    L = Z[0]["feats"].shape[0]; beta = np.zeros((L, 13)); rep = []
    for l in range(L):
        X = np.concatenate([z["feats"][l] for z in Z]); y = np.concatenate([z["target"][l] for z in Z])
        s = X.std(0); s[s == 0] = 1.0; Xs = X / s
        b = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(13), Xs.T @ y) / s
        r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2); beta[l] = b; rep.append(float(r2))
    np.savez(bpath, beta=beta, r2=np.array(rep), lam=lam, n_nets=len(Z))
    print("fit on", len(Z), "nets, lam", lam, "| in-sample R2 per layer:", " ".join(f"{r:.2f}" for r in rep))
elif cmd == "eval":
    bpath = sys.argv[2]
    if bpath != "none": os.environ["Q19_BETA"] = bpath
    est = module(); res = {}
    for d in sys.argv[3:]:
        mlp, truth = load_net(d); name = os.path.basename(os.path.normpath(d)); t0 = time.time()
        p, fl = run(est, mlp)
        res[name] = {"final_mse": float(np.mean((p[-1] - truth[-1]) ** 2)), "util": fl / 2**41,
                     "per_layer": [float(np.mean((p[l] - truth[l]) ** 2)) for l in range(p.shape[0])]}
        print(f"eval[{os.path.basename(bpath)}] {name}: final MSE {res[name]['final_mse']:.4e} util {res[name]['util']:.4f} [{time.time()-t0:.0f}s]", flush=True)
    tag = "none" if bpath == "none" else os.path.splitext(os.path.basename(bpath))[0]
    json.dump(res, open(f"../results/summaries/q19_eval_{tag}_{len(res)}nets.json", "w"), indent=1)
