"""V56 feature dump (8 Oct): run the chain (v56_star.py, K4C blend 0 = no effect on the output, LMIN 2) on MLP #0 and the synthetic
net and save every layer's K4C feature matrix X (full layers: s3sq, y21, y21T, c2, absa, chain, STAR, 1). Fitting happens locally
against the stored teachers. Usage: python k4s_dump.py <dir with v56_star.py, w_mlp0.npz, w_synth.npz> <out.npz>
"""
import os, sys, types, time
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]
os.environ.update({"V55_K4C": "1", "V55_K4C_BLEND": "0", "V55_K4C_DUMP": "1", "V55_K4C_LMIN": "2", "V56_K4S": "1"})
import flopscope as flops, flopscope.numpy as fnp
from whestbench import MLP
SRC = os.path.join(D, "v56_star.py")
class C: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
store = {}
for net in ("mlp0", "synth"):
    t0 = time.time()
    mod = types.ModuleType("m"); exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), mod.__dict__)
    W = np.load(os.path.join(D, f"w_{net}.npz"))["W"].astype(np.float32)
    n = W.shape[1]; C.width = n
    mlp = MLP(width=n, depth=16, weights=[fnp.array(W[l]) for l in range(16)], seed=0, name=net)
    e = mod.Estimator(); e.setup(C())
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
        out = e.predict(mlp, 2**41)
    for layer, full, X in mod.K4C_DUMP:
        store[f"{net}_L{layer}_{int(full)}"] = np.asarray(X, np.float64)
    store[f"{net}_pred"] = np.asarray(out, np.float64)
    print("dumped", net, len(mod.K4C_DUMP), "layers", f"{time.time() - t0:.0f}s", flush=True)
np.savez(OUT, **store)
print("saved", OUT)
