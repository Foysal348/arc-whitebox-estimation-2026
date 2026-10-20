"""X12 (20 Oct, experiment 12): table iteration on the current best e30kc1n2t (N2 + X_DEAD_TAU_B 2.4), fully on Kaggle.
Self-contained kernels running x12_pipeline.py (install pins -> download mini split -> dump -> fit -> eval -> summary):
  f_3 (foysalemonshanto) KIND=F  teacher-forced dump -> N3 (richer modulators)            arms ref, F_N3
  m_4 (foysalemonshanto) KIND=M  forced + free dumps -> DAgger aggregate                   arms ref, M_T2, M_N3
  p_1 (esfersami50)      KIND=P  free-running dump with N2 on (on-policy refit, DAgger 1)  arms ref, P_T2, P_N3
  s_2 (esfersami50)      smoke: KIND=M on nets 0-1, eval net 70 only (code-path check of every stage)
Usage: python kaggle/x12/make_kernels.py   then   bash kaggle/x12/launch.sh ds ; bash kaggle/x12/launch.sh push s 2 ; ...
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PIN = "'flopscope>=0.12.3,<0.13','whestbench>=0.16.2,<0.17'"
TAG = {"esfersami50": "esfer", "foysalemonshanto": "foysal"}
CODE = [os.path.join(ROOT, "experiments/variants/e30k_x12.py"), os.path.join(HERE, "x12_driver.py"), os.path.join(HERE, "fit_x12.py"),
        os.path.join(HERE, "x12_pipeline.py"), os.path.join(ROOT, "kaggle/x10/tables/N2.json")]
KERNELS = {  # name: (user, env)
    "f_3": ("foysalemonshanto", {"KIND": "F"}),
    "m_4": ("foysalemonshanto", {"KIND": "M"}),
    "p_1": ("esfersami50", {"KIND": "P"}),
    "s_2": ("esfersami50", {"KIND": "M", "DUMP_NETS": "0-1", "EVAL_NETS": "70", "STALL": "900"}),
}

CELL_RUN = r"""env = dict(os.environ); env.update(__ENV__)
env.update({'INPUT': INPUT, 'WORK': '/kaggle/working', 'PYTHONWARNINGS': 'ignore', 'PYTHONUTF8': '1', 'NPROC': '4'})
t0 = time.time()
p = subprocess.Popen([sys.executable, INPUT + '/x12_pipeline.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if 'Warning' in line or 'warnings.warn' in line: continue
    print(line, end='', flush=True)
p.wait(); print('exit', p.returncode, 'elapsed', time.time() - t0)
print(subprocess.run(['du', '-sh', '/kaggle/working'], capture_output=True, text=True).stdout)"""


def nb(env):
    cells = [
        "import os, subprocess, sys, json, time, glob\nos.environ['PYTHONUTF8']='1'\n"
        "print(subprocess.run([sys.executable,'-m','pip','install','-q'," + PIN + "],capture_output=True,text=True).stdout[-500:])\n"
        "from importlib.metadata import version; print('flopscope', version('flopscope'), 'whestbench', version('whestbench'))\n"
        "print('cpus', os.cpu_count()); print(subprocess.run(['free','-g'],capture_output=True,text=True).stdout)",
        "hits = glob.glob('/kaggle/input/**/x12_pipeline.py', recursive=True); print('pipeline at', hits)\nINPUT = os.path.dirname(hits[0])\n"
        "print(sorted(os.listdir(INPUT)))",
        "os.environ['HF_XET_HIGH_PERFORMANCE']='1'\nt0=time.time()\n"
        "r=subprocess.run(['whest','dataset','download','aicrowd/arc-whestbench-public-2026','--revision','v2-phase2','--split','mini'],capture_output=True,text=True)\n"
        "print(r.stdout[-800:], r.stderr[-800:]); print('download s', time.time()-t0)",
        CELL_RUN.replace("__ENV__", json.dumps(env)),
    ]
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


for user, tg in TAG.items():
    cd = os.path.join(HERE, f"code_{tg}"); os.makedirs(cd, exist_ok=True)
    for f in CODE:
        shutil.copy(f, os.path.join(cd, os.path.basename(f)))
    json.dump({"title": f"whest-x12-code-{tg}", "id": f"{user}/whest-x12-code-{tg}", "licenses": [{"name": "MIT"}]},
              open(os.path.join(cd, "dataset-metadata.json"), "w"), indent=1)
for name, (user, env) in KERNELS.items():
    rd = os.path.join(HERE, name); os.makedirs(rd, exist_ok=True)
    slug = "whest-x12-" + name.replace("_", "-")
    json.dump(nb(env), open(os.path.join(rd, "x12-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/{slug}", "title": slug, "code_file": "x12-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-x12-code-{TAG[user]}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(name, user, slug, env)
