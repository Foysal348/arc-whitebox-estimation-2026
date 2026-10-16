"""Build the 11 Oct knob-sweep Kaggle sets: 10 dataset + runner pairs, 2 variants each, on v56As_fs (#334870).
Kernels 1-5 go to esfersami50, kernels 6-10 to foysalemonshanto. The runner notebook is the agegrid runner (100 mini MLPs).
Reference for the paired comparison: base_fbg in data/kaggle_out/fbg_1 (same estimator, same 100 MLPs).
Usage: python kaggle/tune/make_sets.py
"""
import os, json, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
V = json.load(open(os.path.join(HERE, "variants_all.json")))
tags = list(V)
assert len(tags) == 20
for k in range(1, 11):
    user = "esfersami50" if k <= 5 else "foysalemonshanto"
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}")
    os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "submissions/v56As_fs/estimator.py"), os.path.join(sd, "v56As_fs.py"))
    pair = tags[2 * (k - 1): 2 * k]
    json.dump({t: {"file": "v56As_fs.py", "env": V[t]} for t in pair}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-tune-{k}", "id": f"{user}/whest-estimators-tune-{k}", "licenses": [{"name": "MIT"}]},
              open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    shutil.copy(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), os.path.join(rd, "whest-runner.ipynb"))
    json.dump({"id": f"{user}/whest-runner-tune-{k}", "title": f"whest-runner-tune-{k}", "code_file": "whest-runner.ipynb",
               "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-estimators-tune-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, user, pair)
