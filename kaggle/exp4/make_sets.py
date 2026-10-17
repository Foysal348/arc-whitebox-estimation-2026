"""Experiment 4 (17 Oct): e30_a fixed-basis old level and hot-row rule in E299 (experiments/variants/e299_fija.py), 100 mini MLPs,
flopscope 0.12.3. Arms from the porting agent; pairs of arms per kernel. Kernels 1-2 esfersami50, 3-5 foysalemonshanto.
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
ARMS = [("base_fija", {}), ("fija", {"X4_FIJA": "1"}), ("hot3", {"X4_HOT": "3.0"}), ("fija_hot3", {"X4_FIJA": "1", "X4_HOT": "3.0"})]
nb0 = json.load(open(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), encoding="utf-8"))
for k in range(1, 3):
    user = "foysalemonshanto"
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}"); os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/variants/e299_fija.py"), os.path.join(sd, "e299_fija.py"))
    pair = ARMS[2 * (k - 1): 2 * k]
    json.dump({t: {"file": "e299_fija.py", "env": e} for t, e in pair}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-exp4-{k}", "id": f"{user}/whest-estimators-exp4-{k}", "licenses": [{"name": "MIT"}]}, open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    nb = json.loads(json.dumps(nb0)); src = "".join(nb["cells"][0]["source"]); a = '"flopscope>=0.12.1,<0.13","whestbench>=0.16.1,<0.17"'; assert a in src
    nb["cells"][0]["source"] = src.replace(a, NEW)
    json.dump(nb, open(os.path.join(rd, "whest-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/whest-runner-exp4-{k}", "title": f"whest-runner-exp4-{k}", "code_file": "whest-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-estimators-exp4-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, user, [t for t, _ in pair])
