"""X12 driver (= kaggle/x10/code/x10_driver.py + the g4l dump key of e30k_x12). X10 driver (20 Oct). Two modes on a slice [START, END) of the public mini split (v2-phase2):

MODE=dump : run EST (e30k_x10.py) with X10_DUMP=1 X_TAB_FORCE=1 (mean teacher-forced to the dataset truth, one step per
            layer) and save per net the 87 table features, the corrector star st, g4 (raw / corrected), var, alpha,
            sigma*phi, the table delta, the base (pred before the table) and the truth -> OUT/net_XXX.npz.
MODE=eval : for each arm in ARMS (JSON {tag: {env}}), import EST fresh with that env and record per net the final-layer
            MSE and the flopscope FLOPs -> OUT/eval_START_END.json (all arms on one net before moving to the next).
"""
import os, sys, time, json, importlib.util, traceback
import numpy as np

EST = os.environ["EST"]
MODE = os.environ.get("MODE", "dump")
OUT = os.environ.get("OUT", "/kaggle/working/out")
START = int(os.environ.get("START", "0"))
END = int(os.environ.get("END", "1"))
os.makedirs(OUT, exist_ok=True)
import flopscope as flops
import flopscope.numpy as fnp
import whestbench

ds = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")


class _Ctx:
    seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


def load(tag, env):
    for k in [k for k in os.environ if k.startswith(("X10_", "X5_", "X8_", "X2_")) or k in ("X_TAB_FORCE",)]:
        del os.environ[k]
    os.environ.update({k: str(v) for k, v in env.items()})
    sp = importlib.util.spec_from_file_location("est_" + tag, EST)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    e = m.Estimator()
    e.setup(_Ctx())
    return m, e


if MODE == "dump":
    m, est = load("dump", {"X10_DUMP": "1", "X_TAB_FORCE": "1", **json.loads(os.environ.get("DUMP_ENV", "{}"))})
    print("dump module: X10_DUMP", m.X10_DUMP, "TAB_FORCE", m.TAB_FORCE, "K4C scale", m.X5_K4C_SCALE, "K31 mix", m.X2_K31_MIX,
          "feats", len(m.CORR_FEATS), flush=True)
    for i, mlp in enumerate(whestbench.iter_mlps(ds)):
        if i < START:
            continue
        if i >= END:
            break
        t0 = time.time()
        truth = np.asarray(ds[i]["all_layer_means"], dtype=np.float64)
        assert truth.shape == (16, 1024), truth.shape
        m._X10_REC = {}
        m._TAB_TRUTH = [fnp.asarray(truth[l].astype(np.float32)) for l in range(16)]
        try:
            with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
                pred = est.predict(mlp, 2**41)
        except Exception:
            traceback.print_exc()
            continue
        pred = np.asarray(pred, dtype=np.float64)
        rec = m._X10_REC
        st = {k: np.stack([np.asarray(rec[l][k], dtype=np.float32) for l in range(16)])
              for k in ("st", "g4c", "g4", "g4l", "var", "alpha", "sp", "delta", "base")}
        F = np.stack([np.asarray(rec[l]["F"], dtype=np.float32) for l in range(16)])
        base_asm = np.stack([np.asarray(rec[l]["base_asm"], dtype=np.float32) for l in range(15)])
        mu_free = np.stack([np.asarray(rec[l]["mu_free"], dtype=np.float32) for l in range(15)])
        forced_ok = bool(np.array_equal(pred[:15].astype(np.float32), truth[:15].astype(np.float32)))
        one = [float(np.mean((mu_free[l] - truth[l]) ** 2)) for l in range(15)]
        one.append(float(np.mean((pred[15] - truth[15]) ** 2)))
        np.savez_compressed(os.path.join(OUT, "net_%03d.npz" % i), F=F, base_asm=base_asm, mu_free=mu_free,
                            truth=truth.astype(np.float32), name=mlp.name, onestep_mse=np.asarray(one), **st)
        print("%d %s forced_ok=%s onestep L0 %.3e L8 %.3e L15 %.3e st_nz(L8)=%d  %.0fs" % (
            i, mlp.name, forced_ok, one[0], one[8], one[15], int(np.count_nonzero(st["st"][8])), time.time() - t0), flush=True)

elif MODE == "eval":
    # nets outer, arms inner (partial results always cover every arm). NETS="70-84,0-34" (inclusive ranges, in this
    # order) overrides START/END.
    import gc
    ARMS = json.loads(os.environ["ARMS"])
    order = []
    for part in os.environ.get("NETS", "%d-%d" % (START, END - 1)).split(","):
        a, _, b = part.partition("-")
        order += list(range(int(a), int(b or a) + 1))
    res = {t: {} for t in ARMS}
    fn = os.path.join(OUT, "eval_%s.json" % os.environ.get("NETS", "%d_%d" % (START, END)).replace(",", "_"))
    try:
        from whestbench.dataset import resolve_seed_context
        from whestbench.domain import MLP as _MLP
        _pv, _salt = resolve_seed_context(ds)
        get_mlp = lambda i: _MLP.from_row(ds[i], seed_protocol_version=_pv, seed_salt=_salt)
        m0 = get_mlp(0); m0b = next(whestbench.iter_mlps(ds)); assert m0.name == m0b.name; del m0, m0b
        print("random access ok", flush=True)
    except Exception:
        traceback.print_exc()
        _all = None
        def get_mlp(i):
            for j, mm in enumerate(whestbench.iter_mlps(ds)):
                if j == i:
                    return mm
    mods = {}
    for t, env in ARMS.items():
        m, e = load(t, env)
        mods[t] = (m, e)
        print("arm", t, "feats", len(m.CORR_FEATS), "mods", (m._FZ or {}).get("mods"), "K4C scale", m.X5_K4C_SCALE,
              "K31 mix", m.X2_K31_MIX, "D21S", m.X8_D21S, "SLOC", m.X8_SLOC, "SSTAR", m.X8_SSTAR, flush=True)
    for i in order:
        mlp = get_mlp(i)
        truth = np.asarray(ds[i]["all_layer_means"], dtype=np.float64)
        for t in ARMS:
            m, e = mods[t]
            t0 = time.time()
            try:
                with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
                    pred = e.predict(mlp, 2**41)
                pred = np.asarray(pred, dtype=np.float64)
                mse = float(np.mean((pred[-1] - truth[-1]) ** 2))
                allm = float(np.mean((pred - truth) ** 2))
                res[t][str(i)] = {"name": mlp.name, "mse": mse, "all_mse": allm, "util": ctx.flops_used / 2**41,
                                  "secs": time.time() - t0}
                print(t, i, mlp.name, "mse %.4e util %.4f %.0fs" % (mse, ctx.flops_used / 2**41, time.time() - t0), flush=True)
            except Exception:
                traceback.print_exc()
                res[t][str(i)] = {"name": mlp.name, "error": traceback.format_exc()[-500:]}
            json.dump(res, open(fn, "w"), indent=1)
            gc.collect()
        del mlp
