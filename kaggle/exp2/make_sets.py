"""Experiment 2 (17 Oct): E299 + two-term (3,1) closure on 100 mini MLPs, flopscope 0.12.3. Reference: base_E299_fs123
(kaggle/fs123 set 2). Kernels on foysalemonshanto.
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
SETS = {1: {"k31": {"X2_K31TWO": "1"}, "k31_mix05": {"X2_K31TWO": "1", "X2_K31_MIX": "0.5"}},
        2: {"k31_b2_13": {"X2_K31TWO": "1", "X2_K31_B2": "1.3"}, "k31_mix025": {"X2_K31TWO": "1", "X2_K31_MIX": "0.25"}}}
nb0 = json.load(open(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), encoding="utf-8"))
for k, vs in SETS.items():
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}"); os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/variants/e299_k31.py"), os.path.join(sd, "e299_k31.py"))
    json.dump({t: {"file": "e299_k31.py", "env": e} for t, e in vs.items()}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-exp2-{k}", "id": f"foysalemonshanto/whest-estimators-exp2-{k}", "licenses": [{"name": "MIT"}]}, open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    nb = json.loads(json.dumps(nb0)); src = "".join(nb["cells"][0]["source"]); a = '"flopscope>=0.12.1,<0.13","whestbench>=0.16.1,<0.17"'; assert a in src
    nb["cells"][0]["source"] = src.replace(a, NEW)
    json.dump(nb, open(os.path.join(rd, "whest-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"foysalemonshanto/whest-runner-exp2-{k}", "title": f"whest-runner-exp2-{k}", "code_file": "whest-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"foysalemonshanto/whest-estimators-exp2-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, vs)
