"""Experiment 5 (17 Oct): E299's kappa4 corrector ported into e30_a (experiments/variants/e30a_k4c.py), 100 mini MLPs,
flopscope 0.12.3. Arms from the porting agent; pairs of arms per kernel. Kernels 1-2 esfersami50, 3-5 foysalemonshanto.
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
ARMS = [("base_e30a", {}), ("k4c_s025", {"X5_K4C": "1", "X5_K4C_SCALE": "0.25"}),
        ("k4c_s030", {"X5_K4C": "1", "X5_K4C_SCALE": "0.3"}), ("k4c_s040", {"X5_K4C": "1", "X5_K4C_SCALE": "0.4"}),
        ("k4c_s030_lvl", {"X5_K4C": "1", "X5_K4C_SCALE": "0.3", "X5_K4LVL": "1"}), ("lvl_only", {"X5_K4LVL": "1"}),
        ("k4c_s020", {"X5_K4C": "1", "X5_K4C_SCALE": "0.2"}), ("k4c_s050", {"X5_K4C": "1", "X5_K4C_SCALE": "0.5"}),
        ("k4c_s030_nostar", {"X5_K4C": "1", "X5_K4C_SCALE": "0.3", "X5_K4S": "0"}), ("k4c_s030_rider", {"X5_K4C": "1", "X5_K4C_SCALE": "0.3", "X5_RIDER": "1"})]
nb0 = json.load(open(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), encoding="utf-8"))
for k in range(1, 6):
    user = "esfersami50" if k <= 2 else "foysalemonshanto"
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}"); os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/variants/e30a_k4c.py"), os.path.join(sd, "e30a_k4c.py"))
    pair = ARMS[2 * (k - 1): 2 * k]
    json.dump({t: {"file": "e30a_k4c.py", "env": e} for t, e in pair}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-exp5-{k}", "id": f"{user}/whest-estimators-exp5-{k}", "licenses": [{"name": "MIT"}]}, open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    nb = json.loads(json.dumps(nb0)); src = "".join(nb["cells"][0]["source"]); a = '"flopscope>=0.12.1,<0.13","whestbench>=0.16.1,<0.17"'; assert a in src
    nb["cells"][0]["source"] = src.replace(a, NEW)
    json.dump(nb, open(os.path.join(rd, "whest-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/whest-runner-exp5-{k}", "title": f"whest-runner-exp5-{k}", "code_file": "whest-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-estimators-exp5-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, user, [t for t, _ in pair])
