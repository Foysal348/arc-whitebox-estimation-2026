"""X13 driver (20 Oct) = kaggle/x10/code/x10_driver.py with two net sources:
  net id < 1000  : public MINI split row id (v2-phase2), loaded with whestbench.load_dataset (only if such ids are asked)
  net id >= 1000 : public FULL split mlp_id = id - 1000, prepared by prep_full.py into FULL_DIR (weights .npy + truth .npy)
MODE=dump : EST with X10_DUMP=1 X_TAB_FORCE=1 (teacher-forced) + DUMP_ENV; per net OUT/net_<id>.npz (same content as x10)
MODE=eval : per net, per arm in ARMS ({tag: env}) final-layer MSE + flopscope util -> OUT/eval_<NETS>.json
NETS = "70-77,1640-1647" (inclusive ranges, processed in this order).
"""
import os, sys, time, json, importlib.util, traceback
import numpy as np

EST = os.environ["EST"]
MODE = os.environ.get("MODE", "dump")
OUT = os.environ.get("OUT", "/kaggle/working/out")
FULL_DIR = os.environ.get("FULL_DIR", "/tmp/x13_full")
os.makedirs(OUT, exist_ok=True)
import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from whestbench.domain import MLP as _MLP

order = []
for part in os.environ["NETS"].split(","):
    a, _, b = part.partition("-")
    order += list(range(int(a), int(b or a) + 1))

_ds = None


def mini_ds():
    global _ds
    if _ds is None:
        _ds = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")
    return _ds


def get(i):
    """(mlp, truth float64 (16,1024))"""
    if i < 1000:
        ds = mini_ds()
        try:
            from whestbench.dataset import resolve_seed_context
            pv, salt = resolve_seed_context(ds)
            mlp = _MLP.from_row(ds[i], seed_protocol_version=pv, seed_salt=salt)
        except Exception:
            traceback.print_exc()
            mlp = None
            for j, mm in enumerate(whestbench.iter_mlps(ds)):
                if j == i:
                    mlp = mm; break
        return mlp, np.asarray(ds[i]["all_layer_means"], dtype=np.float64)
    b = os.path.join(FULL_DIR, "f%04d" % (i - 1000))
    meta = json.load(open(b + ".json"))
    W = np.load(b + "_w.npy", mmap_mode="r")
    row = {"weights": W, "mlp_seed": meta["mlp_seed"], "mlp_name": meta["name"]}
    ctx = json.load(open(os.path.join(FULL_DIR, "seed_ctx.json")))
    mlp = None
    if ctx.get("version"):
        try:
            salt = bytes.fromhex(ctx["salt_hex"]) if ctx.get("salt_hex") else None
            mlp = _MLP.from_row(row, seed_protocol_version=ctx["version"], seed_salt=salt)
        except Exception as e:
            print("from_row with seed ctx failed:", repr(e)[:200], flush=True)
    if mlp is None:
        mlp = _MLP.from_row(row, seed_protocol_version="2.0")   # estimator seed unused by e30k (setup uses ctx.seed)
    return mlp, np.load(b + "_t.npy").astype(np.float64)


class _Ctx:
    seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


def load(tag, env):
    for k in [k for k in os.environ if k.startswith(("X10_", "X5_", "X8_", "X2_")) or k in ("X_TAB_FORCE", "X_DEAD_TAU_B", "V21_R_OLD")]:
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
          "D21S", m.X8_D21S, "SLOC", m.X8_SLOC, "SSTAR", m.X8_SSTAR, "DEAD_TAU_B", m.DEAD_TAU_B, "feats", len(m.CORR_FEATS), flush=True)
    for i in order:
        if os.path.exists(os.path.join(OUT, "net_%04d.npz" % i)):
            continue
        t0 = time.time()
        try:
            mlp, truth = get(i)
        except Exception:
            traceback.print_exc(); continue
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
              for k in ("st", "g4c", "g4", "var", "alpha", "sp", "delta", "base")}
        F = np.stack([np.asarray(rec[l]["F"], dtype=np.float32) for l in range(16)])
        base_asm = np.stack([np.asarray(rec[l]["base_asm"], dtype=np.float32) for l in range(15)])
        mu_free = np.stack([np.asarray(rec[l]["mu_free"], dtype=np.float32) for l in range(15)])
        forced_ok = bool(np.array_equal(pred[:15].astype(np.float32), truth[:15].astype(np.float32)))
        one = [float(np.mean((mu_free[l] - truth[l]) ** 2)) for l in range(15)]
        one.append(float(np.mean((pred[15] - truth[15]) ** 2)))
        np.savez_compressed(os.path.join(OUT, "net_%04d.npz" % i), F=F, base_asm=base_asm, mu_free=mu_free,
                            truth=truth.astype(np.float32), name=mlp.name, onestep_mse=np.asarray(one), **st)
        print("%d %s forced_ok=%s onestep L0 %.3e L8 %.3e L15 %.3e st_nz(L8)=%d  %.0fs" % (
            i, mlp.name, forced_ok, one[0], one[8], one[15], int(np.count_nonzero(st["st"][8])), time.time() - t0), flush=True)
        del mlp

elif MODE == "eval":
    import gc
    ARMS = json.loads(os.environ["ARMS"])
    res = {t: {} for t in ARMS}
    fn = os.path.join(OUT, "eval_%s.json" % os.environ["NETS"].replace(",", "_"))
    mods = {}
    for t, env in ARMS.items():
        m, e = load(t, env)
        mods[t] = (m, e)
        print("arm", t, "feats", len(m.CORR_FEATS), "beta0[15][:3]", [round(x, 6) for x in m.CORR_BETA_NEW[15][:3]], "K4C scale",
              m.X5_K4C_SCALE, "K31 mix", m.X2_K31_MIX, "D21S", m.X8_D21S, "SLOC", m.X8_SLOC, "SSTAR", m.X8_SSTAR,
              "DEAD_TAU_B", m.DEAD_TAU_B, flush=True)
    for i in order:
        mlp, truth = get(i)
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
