"""Experiment 5 (17 Oct): E299's kappa4 corrector ported into e30_a (experiments/variants/e30a_k4c.py), 100 mini MLPs,
flopscope 0.12.3. Arms from the porting agent; pairs of arms per kernel. Kernels 1-2 esfersami50, 3-5 foysalemonshanto.
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NEW = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
ARMS = [("base_x8", {}), ("c1", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7"}), ("c1_tauB22", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "X_DEAD_TAU_B": "2.2"}), ("c1_tauB24", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "X_DEAD_TAU_B": "2.4"}), ("c1_tauB26", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "X_DEAD_TAU_B": "2.6"}), ("c1_stv", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "X_STV": "1"}), ("c1_l6b7", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "X_L6": "1", "X_L6_BMIN": "7"}), ("c1_b2_13", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "X2_K31_B2": "1.3"}), ("c1_d21s096", {"X8_D21S": "0.96", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7"}), ("c1_d21s097", {"X8_D21S": "0.97", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7"}), ("c1_hot35", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "X_VB_HOT": "3.5"}), ("c1_d21L5", {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "X8_D21L": "5"})]
nb0 = json.load(open(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), encoding="utf-8"))
for k in range(1, 7):
    user = "esfersami50" if k <= 3 else "foysalemonshanto"
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}"); os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/variants/e30k_x8.py"), os.path.join(sd, "e30k_x8.py"))
    pair = ARMS[2 * (k - 1): 2 * k]
    json.dump({t: {"file": "e30k_x8.py", "env": e} for t, e in pair}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-x8c-{k}", "id": f"{user}/whest-estimators-x8c-{k}", "licenses": [{"name": "MIT"}]}, open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    nb = json.loads(json.dumps(nb0)); src = "".join(nb["cells"][0]["source"]); a = '"flopscope>=0.12.1,<0.13","whestbench>=0.16.1,<0.17"'; assert a in src
    nb["cells"][0]["source"] = src.replace(a, NEW)
    json.dump(nb, open(os.path.join(rd, "whest-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/whest-runner-x8c-{k}", "title": f"whest-runner-x8c-{k}", "code_file": "whest-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-estimators-x8c-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, user, [t for t, _ in pair])
