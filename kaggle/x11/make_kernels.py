"""X11 (20 Oct): cost cut + table refit, fully on Kaggle. Same pipeline pieces as kaggle/x10 (driver
kaggle/x10/code/x10_driver.py, estimator experiments/variants/e30k_x10.py, fit recipe kaggle/x10/fit_x10.py -> T2 = the N2
recipe), on cheaper settings of the nb base:
  C1 = nb + V21_R_OLD=352 (e30_a default 416)       C2 = nb + X_DEAD_TAU_B=2.4 (near-dead source-column cut)
Usage: python kaggle/x11/make_kernels.py pipe  -> kaggle/x11/{code_esfer,code_foysal,c1_1,c2_3}: one self-contained kernel per
       candidate running x11_pipeline.py (dump nets 0-69 -> fit -> eval nets 70-99 + 15 in-sample -> summary), 4 processes.
Kernel c1_1 on esfersami50, c2_3 on foysalemonshanto.  NOTE: x10_driver.load() only clears X10_/X5_/X8_/X2_ env keys, so
every eval arm (x11_pipeline.py) sets V21_R_OLD and X_DEAD_TAU_B explicitly (no leak between arms loaded in one process).
"""
import os, sys, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PIN = "'flopscope>=0.12.3,<0.13','whestbench>=0.16.2,<0.17'"
USERS = {1: "esfersami50", 3: "foysalemonshanto"}
TAG = {"esfersami50": "esfer", "foysalemonshanto": "foysal"}
NB = {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7"}
C1 = dict(NB, V21_R_OLD="352", X_DEAD_TAU_B="0")
C2 = dict(NB, V21_R_OLD="416", X_DEAD_TAU_B="2.4")
EST = os.path.join(ROOT, "experiments/variants/e30k_x10.py")
DRV = os.path.join(ROOT, "kaggle/x10/code/x10_driver.py")
CODE = [EST, DRV, os.path.join(HERE, "fit_x11.py"), os.path.join(HERE, "x11_pipeline.py"), os.path.join(ROOT, "kaggle/x10/tables/N2.json")]

CELL_RUN = r"""env = dict(os.environ); env.update(__ENV__)
env.update({'INPUT': INPUT, 'WORK': '/kaggle/working', 'PYTHONWARNINGS': 'ignore', 'PYTHONUTF8': '1'})
t0 = time.time()
p = subprocess.Popen([sys.executable, INPUT + '/x11_pipeline.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if 'Warning' in line or 'warnings.warn' in line: continue
    print(line, end='', flush=True)
p.wait(); print('exit', p.returncode, 'elapsed', time.time() - t0)
print(subprocess.run(['du', '-sh', '/kaggle/working'], capture_output=True, text=True).stdout)"""


def nb_pipe(cand, cenv, extra=None):
    """self-contained notebook: install pins, download the mini split, run x11_pipeline.py (dump -> fit -> eval -> summary)."""
    cells = [
        "import os, subprocess, sys, json, time, glob\nos.environ['PYTHONUTF8']='1'\n"
        "print(subprocess.run([sys.executable,'-m','pip','install','-q'," + PIN + "],capture_output=True,text=True).stdout[-500:])\n"
        "from importlib.metadata import version; print('flopscope', version('flopscope'), 'whestbench', version('whestbench'))\n"
        "print('cpus', os.cpu_count()); print(subprocess.run(['free','-g'],capture_output=True,text=True).stdout)",
        "hits = glob.glob('/kaggle/input/**/x11_pipeline.py', recursive=True); print('pipeline at', hits)\nINPUT = os.path.dirname(hits[0])\n"
        "print(sorted(os.listdir(INPUT)))",
        "os.environ['HF_XET_HIGH_PERFORMANCE']='1'\nt0=time.time()\n"
        "r=subprocess.run(['whest','dataset','download','aicrowd/arc-whestbench-public-2026','--revision','v2-phase2','--split','mini'],capture_output=True,text=True)\n"
        "print(r.stdout[-800:], r.stderr[-800:]); print('download s', time.time()-t0)",
        CELL_RUN.replace("__ENV__", json.dumps(dict({"CAND": cand, "CAND_ENV": json.dumps(cenv), "NPROC": "4"}, **(extra or {})))),
    ]
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


if sys.argv[1] == "pipe":
    for user in sorted(set(USERS.values())):
        cd = os.path.join(HERE, f"code_{TAG[user]}"); os.makedirs(cd, exist_ok=True)
        for f in CODE:
            shutil.copy(f, os.path.join(cd, os.path.basename(f)))
        json.dump({"title": f"whest-x11-code-{TAG[user]}", "id": f"{user}/whest-x11-code-{TAG[user]}",
                   "licenses": [{"name": "MIT"}]}, open(os.path.join(cd, "dataset-metadata.json"), "w"), indent=1)
    for cand, cenv, k in (("C1", C1, 1), ("C2", C2, 3)):
        user = USERS[k]; kp = cand.lower()
        rd = os.path.join(HERE, f"{kp}_{k}"); os.makedirs(rd, exist_ok=True)
        json.dump(nb_pipe(cand, cenv), open(os.path.join(rd, "x10-runner.ipynb"), "w", encoding="utf-8"), indent=1)
        json.dump({"id": f"{user}/whest-x11-{kp}-{k}", "title": f"whest-x11-{kp}-{k}", "code_file": "x10-runner.ipynb",
                   "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
                   "dataset_sources": [f"{user}/whest-x11-code-{TAG[user]}"], "competition_sources": [], "kernel_sources": []},
                  open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
        print(cand, k, user, cenv)

if sys.argv[1] == "evalonly":
    # eval-only rerun (20 Oct): python make_kernels.py evalonly C1 1 "70-77,0-3;78-84,20-23;85-92,40-43"
    # needs <CAND>R.json (the kernel's refit table) copied into code_<tag>/ before the dataset version
    cand, k, nets = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    cenv = {"C1": C1, "C2": C2}[cand]; user = USERS[k]; kp = cand.lower() + "e"
    for u in sorted(set(USERS.values())):
        for f in CODE:
            shutil.copy(f, os.path.join(HERE, f"code_{TAG[u]}", os.path.basename(f)))
    rd = os.path.join(HERE, f"{kp}_{k}"); os.makedirs(rd, exist_ok=True)
    json.dump(nb_pipe(cand, cenv, {"STAGES": "eval,sum", "EVAL_NETS": nets}), open(os.path.join(rd, "x10-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/whest-x11-{kp}-{k}", "title": f"whest-x11-{kp}-{k}", "code_file": "x10-runner.ipynb",
               "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-x11-code-{TAG[user]}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(kp, k, user, nets)
