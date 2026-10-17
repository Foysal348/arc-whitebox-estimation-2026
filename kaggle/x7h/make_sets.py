"""Experiment 7 (17 Oct): e299_x7 (e299xrk + e30_a dead-neuron compactions, LordProcciuto MIT), 100 mini MLPs,
flopscope 0.12.3. Two arms per kernel; the paired base comes from xrk31-1 (see SETS comment), not from this harness.
Usage: python kaggle/x7h/make_sets.py   (edit SETS below; set k -> esfersami50 if k <= 2 else foysalemonshanto)
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
# MLP #0 (kernel esfersami50/whest-x7): yrow x1.063, cpre x0.9930, both x1.040 -> only CPRE goes to 100 MLPs. The paired base is
# the xr_k31_025 arm of esfersami50/whest-runner-xrk31-1 (e299xr_k31.py with X2_K31TWO=1, X2_K31_MIX=0.25 == e299xrk == base_x7).
SETS = {1: [("cpre", {"X7_CPRE": "1"}), ("cpre_arriba", {"X7_CPRE": "1", "X7_DEAD_ROUND": "arriba"})]}
nb0 = json.load(open(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), encoding="utf-8"))
for k, pair in SETS.items():
    user = "esfersami50" if k <= 2 else "foysalemonshanto"
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}"); os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/variants/e299_x7.py"), os.path.join(sd, "e299_x7.py"))
    json.dump({t: {"file": "e299_x7.py", "env": e} for t, e in pair}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-x7-{k}", "id": f"{user}/whest-estimators-x7-{k}", "licenses": [{"name": "MIT"}]},
              open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    nb = json.loads(json.dumps(nb0)); src = "".join(nb["cells"][0]["source"]); a = '"flopscope>=0.12.1,<0.13","whestbench>=0.16.1,<0.17"'; assert a in src
    nb["cells"][0]["source"] = src.replace(a, NEW)
    json.dump(nb, open(os.path.join(rd, "whest-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/whest-runner-x7-{k}", "title": f"whest-runner-x7-{k}", "code_file": "whest-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-estimators-x7-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, user, [t for t, _ in pair])
