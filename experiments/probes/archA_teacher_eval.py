"""Phase A copy of experiments/chain/run_variant_teacher.py (26 Sep 2026): identical, but writes results/summaries/archA_<TAG>.json and saves after every configuration."""
"""Paired local evaluation of an estimator variant on the two networks that have a Monte-Carlo teacher:
public MLP #0 (logan-fitzgerald, baked 1e9-sample truth for every layer) and the synthetic He network (6e7-sample
teacher ReLU means at layers 2, 4, ..., 14, 15 only; its final layer is not available, so layer 15 is reported).

Each configuration is a set of environment variables; the module is re-executed for each one (the knobs are read at
import). The comparison is PAIRED: every variant is scored against the same truth vector as the base, so the truth's
own noise cancels to first order in the difference.

Usage (from whest-starterkit):
    python -m uv run python ../experiments/chain/run_variant_teacher.py ../experiments/variants/estimator_v39k4m.py \
        "V39_K4MULT=0" "V39_K4MULT=1,V39_K4MODE=mean" ...
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SRC = sys.argv[1]
NETS = {"mlp0": "../data/kaggle_out/kaggle_out_k4pub/k4teach", "synth": "../data/kaggle_out/kaggle_out_k4/k4teach"}
if os.environ.get("NETS"): NETS = {k: v for k, v in NETS.items() if k in os.environ["NETS"].split(",")}
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"

def load(dirn):
    W = np.load(glob.glob(os.path.join(dirn, "weights_seed*.npz"))[0])["W"]
    tf = os.path.join(dirn, "truth.npz")
    if os.path.exists(tf):
        t = np.load(tf); truth = {l: np.asarray(t["all_layer_means"][l - 1], np.float64) for l in range(1, 17)}
        seed, name = int(t["mlp_seed"]), str(t["mlp_name"])
    else:
        truth = {}
        for f in glob.glob(os.path.join(dirn, "layer*_seed*.npz")):
            l = int(os.path.basename(f).split("_")[0][5:]); truth[l] = np.asarray(np.load(f)["relu_mean"], np.float64)
        seed, name = 0, "synthetic"
    mlp = MLP(width=W.shape[1], depth=W.shape[0], weights=[fnp.array(W[l]) for l in range(W.shape[0])], seed=seed, name=name)
    return mlp, truth

nets = {k: load(v) for k, v in NETS.items()}
res = {}
for cfg in sys.argv[2:]:
    env = dict(kv.split("=", 1) for kv in cfg.split(",") if kv)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    mod = types.ModuleType("est_var"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
    for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    for nk, (mlp, truth) in nets.items():
        est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            p = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        per = {l: float(np.mean((p[l - 1] - truth[l]) ** 2)) for l in sorted(truth)}
        lf = max(per)
        res.setdefault(cfg, {})[nk] = {"final_layer": lf, "mse": per[lf], "per_layer": per, "util": ctx.flops_used / 2**41}
        print(f"[{cfg}] {nk}: layer {lf} MSE {per[lf]:.4e}  util {ctx.flops_used/2**41:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(f"../results/summaries/archA_{os.environ.get('TAG', 'teacher_eval')}.json", "w"), indent=1)
base = sys.argv[2]
print("\npaired ratios to the first configuration:")
for cfg in sys.argv[2:]:
    print(f"  {cfg:40s} " + "  ".join(f"{nk} {res[cfg][nk]['mse']/res[base][nk]['mse']:.4f}" for nk in nets))
tag = os.environ.get("TAG", os.path.basename(SRC).replace(".py", ""))
json.dump(res, open(f"../results/summaries/archA_{tag}.json", "w"), indent=1)
