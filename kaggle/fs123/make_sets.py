"""Experiment 1 (17 Oct): flopscope 0.12.1 vs 0.12.3 billing on E299 / E299-B (100 mini MLPs), plus E295 and e30_a under 0.12.3.
Also gives the 100-MLP base of E299 under the new grader version for every later experiment. All kernels on esfersami50.
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OLD = '"flopscope==0.12.1","whestbench==0.16.1"'; NEW = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
SRC = {"E299.py": "kaggle/e299b/E299_estimator.py", "E299B.py": "submissions/e299b/estimator.py",
       "E295.py": "kaggle/e299b/E295_estimator.py", "e30a.py": "external/e30_a_estimator.py"}
SETS = {1: (OLD, {"E299_fs121": "E299.py", "E299B_fs121": "E299B.py"}),
        2: (NEW, {"base_E299_fs123": "E299.py", "E299B_fs123": "E299B.py"}),
        3: (NEW, {"E295_fs123": "E295.py", "e30a_fs123": "e30a.py"})}
nb0 = json.load(open(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), encoding="utf-8"))
for k, (pins, vs) in SETS.items():
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}"); os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    for f in set(vs.values()): shutil.copy(os.path.join(ROOT, SRC[f]), os.path.join(sd, f))
    json.dump({t: {"file": f, "env": {}} for t, f in vs.items()}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-fs123-{k}", "id": f"esfersami50/whest-estimators-fs123-{k}", "licenses": [{"name": "MIT"}]}, open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    nb = json.loads(json.dumps(nb0)); src = "".join(nb["cells"][0]["source"])
    a = '"flopscope>=0.12.1,<0.13","whestbench>=0.16.1,<0.17"'; assert a in src
    nb["cells"][0]["source"] = src.replace(a, pins)
    json.dump(nb, open(os.path.join(rd, "whest-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"esfersami50/whest-runner-fs123-{k}", "title": f"whest-runner-fs123-{k}", "code_file": "whest-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"esfersami50/whest-estimators-fs123-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, pins, vs)
