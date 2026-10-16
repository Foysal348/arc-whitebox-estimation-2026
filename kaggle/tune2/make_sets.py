"""Round 2 of the knob sweep (16 Oct): combinations on top of K4C blend 1.1 (submitted as #336382 A / #336383 B).
A-class file v56As_fsb11.py (blend 1.1 default), B-class file v56s_fsb11.py; base_B = v56s_fs.py (blend 1.0) for the B reference.
A reference: base_fbg in data/kaggle_out/fbg_1 (A class, blend 1.0). Kernels 1-5 esfersami50, 6-10 foysalemonshanto.
Usage: python kaggle/tune2/make_sets.py   then   bash kaggle/tune/launch.sh-style launch (kaggle/tune2/launch.sh)
"""
import os, json, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STAR115 = {"V56_K4S_BF": "1.28907:0.033064:-0.00299685:-0.000235647:-6.43616e-05:0.602528:0.704062:0.0258636"}
SMALL4 = dict(STAR115, Q25_LAMB="1:1:0.95:1", V17_LAM_SCALE="0.93", V25_BETA="1.3")
SMALL5 = dict(SMALL4, V55_K4C_LMIN="3")
A, B = "v56As_fsb11.py", "v56s_fsb11.py"
V = {
    "A_b120": (A, {"V55_K4C_BLEND": "1.2"}), "A_b130": (A, {"V55_K4C_BLEND": "1.3"}),
    "A_b11_star115": (A, STAR115), "A_b11_lamM095": (A, {"Q25_LAMB": "1:1:0.95:1"}),
    "A_b11_lam093": (A, {"V17_LAM_SCALE": "0.93"}), "A_b11_beta130": (A, {"V25_BETA": "1.3"}),
    "A_b11_lmin3": (A, {"V55_K4C_LMIN": "3"}), "A_b11_star_lamM": (A, dict(STAR115, Q25_LAMB="1:1:0.95:1")),
    "A_b11_small4": (A, SMALL4), "A_b11_small5": (A, SMALL5),
    "A_b12_small4": (A, dict(SMALL4, V55_K4C_BLEND="1.2")), "A_b11_d21thr020": (A, {"V50_D21_THR": "0.02"}),
    "base_B": ("v56s_fs.py", {}), "B_b11": (B, {}),
    "B_b11_small4": (B, SMALL4), "B_b11_d21thr020": (B, {"V50_D21_THR": "0.02"}),
    "A_b115": (A, {"V55_K4C_BLEND": "1.15"}), "A_b12_small5": (A, dict(SMALL5, V55_K4C_BLEND="1.2")),
    "B_b12": (B, {"V55_K4C_BLEND": "1.2"}), "B_b12_small4": (B, dict(SMALL4, V55_K4C_BLEND="1.2")),
}
json.dump({t: {"file": f, "env": e} for t, (f, e) in V.items()}, open(os.path.join(HERE, "variants_all.json"), "w"), indent=1)
tags = list(V)
assert len(tags) == 20
for k in range(1, 11):
    user = "esfersami50" if k <= 5 else "foysalemonshanto"
    sd = os.path.join(HERE, f"set_{k}"); rd = os.path.join(HERE, f"runner_{k}")
    os.makedirs(sd, exist_ok=True); os.makedirs(rd, exist_ok=True)
    pair = tags[2 * (k - 1): 2 * k]
    for t in pair:
        f = V[t][0]; src = {"v56As_fsb11.py": "v56As_fsb11", "v56s_fsb11.py": "v56s_fsb11", "v56s_fs.py": "v56s_fs"}[f]
        shutil.copy(os.path.join(ROOT, f"submissions/{src}/estimator.py"), os.path.join(sd, f))
    json.dump({t: {"file": V[t][0], "env": V[t][1]} for t in pair}, open(os.path.join(sd, "variants.json"), "w"), indent=1)
    json.dump({"title": f"whest-estimators-tune2-{k}", "id": f"{user}/whest-estimators-tune2-{k}", "licenses": [{"name": "MIT"}]},
              open(os.path.join(sd, "dataset-metadata.json"), "w"), indent=1)
    shutil.copy(os.path.join(ROOT, "kaggle/agegrid/runner_1/whest-runner.ipynb"), os.path.join(rd, "whest-runner.ipynb"))
    json.dump({"id": f"{user}/whest-runner-tune2-{k}", "title": f"whest-runner-tune2-{k}", "code_file": "whest-runner.ipynb",
               "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-estimators-tune2-{k}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(k, user, pair)
