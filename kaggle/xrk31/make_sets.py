"""Experiment 5 (17 Oct): E299's kappa4 corrector ported into e30_a (experiments/variants/e299xr_k31.py), 100 mini MLPs,
flopscope 0.12.3. Arms from the porting agent; pairs of arms per kernel. Kernels 1-2 esfersami50, 3-5 foysalemonshanto.
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
ARMS = [("base_xr", {}), ("xr_k31_025", {"X2_K31TWO": "1", "X2_K31_MIX": "0.25"}), ("xr_k31_015", {"X2_K31TWO": "1", "X2_K31_MIX": "0.15"}), ("xr_k31_035", {"X2_K31TWO": "1", "X2_K31_MIX": "0.35"})]
nb0 = json.load(open(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), encoding="utf-8"))
for k in range(1, 3):
    user = "esfersami50"
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}"); os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/variants/e299xr_k31.py"), os.path.join(sd, "e299xr_k31.py"))
    pair = ARMS[2 * (k - 1): 2 * k]
    json.dump({t: {"file": "e299xr_k31.py", "env": e} for t, e in pair}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-xrk31-{k}", "id": f"{user}/whest-estimators-xrk31-{k}", "licenses": [{"name": "MIT"}]}, open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    nb = json.loads(json.dumps(nb0)); src = "".join(nb["cells"][0]["source"]); a = '"flopscope>=0.12.1,<0.13","whestbench>=0.16.1,<0.17"'; assert a in src
    nb["cells"][0]["source"] = src.replace(a, NEW)
    json.dump(nb, open(os.path.join(rd, "whest-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/whest-runner-xrk31-{k}", "title": f"whest-runner-xrk31-{k}", "code_file": "whest-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-estimators-xrk31-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, user, [t for t, _ in pair])
