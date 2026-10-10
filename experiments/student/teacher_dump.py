"""Teacher-label generation for the Round-20 pair-memory student (11 Oct 2026).

For each network seed: build the MLP exactly as whestbench.sample_mlp does (W ~ N(0, 2/n), float32) from
numpy.random.default_rng(seed), run the frozen V56 chain (experiments/student/v56_dump.py = v56As_fs + a dump hook) once,
and save per layer:
- pre-activation mean, var, d3, g4row, g4mean (the V56 kappa4 input to the mean closure), lam;
- sampled rows of D21 / C_pre / K21 / K22;
- post-ReLU closure outputs (mean, var, d3, k4);
- the final prediction.
One compressed npz per network: <out>/net_<seed>.npz.

Usage: python teacher_dump.py <estimator.py> <out_dir> <seed_start> <seed_end_exclusive> [rows=32] [full_seeds=a,b]
Env: OMP/MKL threads are the caller's choice.
"""
import os, sys, time, types
import numpy as np

EST, OUT, s0, s1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ROWS = int(sys.argv[5]) if len(sys.argv) > 5 else 32
FULL = set(int(x) for x in sys.argv[6].split(",") if x) if len(sys.argv) > 6 else set()
os.environ["STUDENT_DUMP"] = "1"; os.environ["STUDENT_ROWS"] = str(ROWS)
os.makedirs(OUT, exist_ok=True)
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

mod = types.ModuleType("est_dump"); mod.__file__ = os.path.abspath(EST)
exec(compile(open(EST, encoding="utf-8").read(), os.path.abspath(EST), "exec"), mod.__dict__)


class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


est = mod.Estimator(); est.setup(_Ctx())
n, L = 1024, 16
for seed in range(s0, s1):
    path = os.path.join(OUT, f"net_{seed}.npz")
    if os.path.exists(path):
        continue
    t0 = time.time()
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
    mlp = MLP(width=n, depth=L, weights=[fnp.array(w) for w in Ws], seed=int(seed), name=f"net{seed}")
    os.environ["STUDENT_SEED"] = str(seed); os.environ["STUDENT_FULL"] = "1" if seed in FULL else "0"
    mod.DUMP_STORE.clear()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        out = est.predict(mlp, 2**41)
    save = {"final": np.asarray(out, np.float32), "flops_u": np.float64(ctx.flops_used / 2.0**31)}
    for layer, d in mod.DUMP_STORE.items():
        for k, v in d.items():
            save[f"L{layer}_{k}"] = v
    np.savez_compressed(path, **save)
    print(f"seed {seed}: {time.time() - t0:.0f}s, {len(mod.DUMP_STORE)} layers, final shape {save['final'].shape}, "
          f"{os.path.getsize(path) / 1e6:.1f} MB", flush=True)
