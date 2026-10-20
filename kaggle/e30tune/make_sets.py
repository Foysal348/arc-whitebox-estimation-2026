"""Experiment 5 (17 Oct): E299's kappa4 corrector ported into e30_a (experiments/variants/e30a_k4c.py), 100 mini MLPs,
flopscope 0.12.3. Arms from the porting agent; pairs of arms per kernel. Kernels 1-2 esfersami50, 3-5 foysalemonshanto.
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
ARMS = [("base_e30k", {}), ("tilt_m02", {"X_LAM_TILT": "-0.2"}), ("tilt_00", {"X_LAM_TILT": "0.0"}), ("tau_260", {"X_DEAD_TAU": "2.6"}), ("tau_310", {"X_DEAD_TAU": "3.1"}), ("hot_25", {"X_VB_HOT": "2.5"}), ("hot_35", {"X_VB_HOT": "3.5"}), ("mix_040", {"X2_K31_MIX": "0.4"}), ("scale_035", {"X5_K4C_SCALE": "0.35"}), ("mix_015", {"X2_K31_MIX": "0.15"})]
nb0 = json.load(open(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), encoding="utf-8"))
for k in range(1, 6):
    user = "esfersami50" if k <= 2 else "foysalemonshanto"
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}"); os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "submissions/e30ak4c40k31/estimator.py"), os.path.join(sd, "e30k.py"))
    pair = ARMS[2 * (k - 1): 2 * k]
    json.dump({t: {"file": "e30k.py", "env": e} for t, e in pair}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-e30tune-{k}", "id": f"{user}/whest-estimators-e30tune-{k}", "licenses": [{"name": "MIT"}]}, open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    nb = json.loads(json.dumps(nb0)); src = "".join(nb["cells"][0]["source"]); a = '"flopscope>=0.12.1,<0.13","whestbench>=0.16.1,<0.17"'; assert a in src
    nb["cells"][0]["source"] = src.replace(a, NEW)
    json.dump(nb, open(os.path.join(rd, "whest-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/whest-runner-e30tune-{k}", "title": f"whest-runner-e30tune-{k}", "code_file": "whest-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-estimators-e30tune-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, user, [t for t, _ in pair])
