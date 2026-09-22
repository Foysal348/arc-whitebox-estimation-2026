"""Run the whest-kprop-reference notebook cells locally (CPU torch, tiny width) BEFORE pushing to Kaggle.
Rewrites the Kaggle-only lines (git clone, /kaggle/working paths) and runs every cell in order.
Usage: python local_check.py            (from kaggle_kprop; system python with torch)
Env: WIDTH (default 32), N_MLPS (1), MC_SAMPLES (1e6)
"""
import json, os, sys, time, types, traceback
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
os.environ.setdefault("WIDTH", "32"); os.environ.setdefault("N_MLPS", "1"); os.environ.setdefault("MC_SAMPLES", "1e6")
work = os.path.join(here, "_local_work"); os.makedirs(work, exist_ok=True)
nb = json.load(open(os.path.join(here, "whest-kprop-reference.ipynb"), encoding="utf-8"))
g = {"__name__": "__main__"}
t0 = time.time()
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    src = "\n".join(("print('skipped clone/pip')" if l.startswith('print(subprocess.run("git clone') else l) for l in src.splitlines())
    src = src.replace("/kaggle/working/mcp/src", os.path.join(root, "mlp_cumulant_propagation", "src").replace("\\", "/"))
    src = src.replace("/kaggle/working/", work.replace("\\", "/") + "/")
    print(f"=== cell {i} ===", flush=True)
    try:
        exec(compile(src, f"cell{i}", "exec"), g)
    except Exception:
        traceback.print_exc(); print(f"LOCAL CHECK FAILED in cell {i}"); sys.exit(1)
res = json.load(open(os.path.join(work, "kprop_results.json")))
bad = [k for k, v in res.items() if "error" in v]
print(f"local check done in {time.time()-t0:.0f}s: {len(res)} configs, {len(bad)} errors: {bad}")
sys.exit(1 if bad else 0)
