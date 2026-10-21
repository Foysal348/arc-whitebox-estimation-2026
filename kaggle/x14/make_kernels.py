"""X14 (21 Oct, experiment 14): N4 table modulators + T21 orientation fix on the current best e30kc1n3 (N3), fully on Kaggle.
Self-contained kernels running x14_pipeline.py (install pins -> download mini split -> dump -> fit -> eval -> summary):
  s_1 (esfersami50)      smoke: KIND G (dump with T21) on nets 0-1, eval net 70, every arm kind
  e_2 (esfersami50)      eval only: ref vs T21 (N3 table fitted without T21)
  f_3 (foysalemonshanto) KIND F forced dump (T21 off) -> fit N3 / N4a / N4s -> arms ref, F_N4s, F_N4a, F_N4s_T21
  g_4 (foysalemonshanto) KIND G forced dump with T21 on -> refit N3 / N4s on the T21 trajectory -> arms ref, G_N3_T21, G_N4s_T21
Usage: python kaggle/x14/make_kernels.py   then   bash kaggle/x14/launch.sh ds code ; bash kaggle/x14/launch.sh push s 1 ; ...
"""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PIN = "'flopscope>=0.12.3,<0.13','whestbench>=0.16.2,<0.17'"
TAG = {"esfersami50": "esfer", "foysalemonshanto": "foysal"}
CODE = [os.path.join(ROOT, "experiments/variants/e30k_x14.py"), os.path.join(HERE, "x14_driver.py"), os.path.join(HERE, "fit_x14.py"),
        os.path.join(HERE, "x14_pipeline.py"), (os.path.join(ROOT, "data/kaggle_out/x12_f/F_N3.json"), "N3.json")]
A_ALL = {"ref": {}, "T21": {"X14_T21": "1"}, "N3re": {"X10_TAB": "@N3"}, "N4s": {"X10_TAB": "@N4s"}, "N4a": {"X10_TAB": "@N4a"},
         "N4s_T21": {"X10_TAB": "@N4s", "X14_T21": "1"}}
KERNELS = {  # name: (user, env)
    "s_1": ("esfersami50", {"KIND": "G", "DUMP_T21": "1", "DUMP_NETS": "0-1", "EVAL_NETS": "70", "STALL": "900", "ARMS": json.dumps(A_ALL)}),
    "e_2": ("esfersami50", {"KIND": "E", "STAGES": "eval,sum", "ARMS": json.dumps({"ref": {}, "T21": {"X14_T21": "1"}})}),
    "f_3": ("foysalemonshanto", {"KIND": "F", "DUMP_T21": "0", "ARMS": json.dumps(
        {"ref": {}, "F_N4s": {"X10_TAB": "@N4s"}, "F_N4a": {"X10_TAB": "@N4a"}, "F_N4s_T21": {"X10_TAB": "@N4s", "X14_T21": "1"}})}),
    "g_4": ("foysalemonshanto", {"KIND": "G", "DUMP_T21": "1", "ARMS": json.dumps(
        {"ref": {}, "G_N3_T21": {"X10_TAB": "@N3", "X14_T21": "1"}, "G_N4s_T21": {"X10_TAB": "@N4s", "X14_T21": "1"}})}),
}

CELL_RUN = r"""env = dict(os.environ); env.update(__ENV__)
env.update({'INPUT': INPUT, 'WORK': '/kaggle/working', 'PYTHONWARNINGS': 'ignore', 'PYTHONUTF8': '1', 'NPROC': '4'})
t0 = time.time()
p = subprocess.Popen([sys.executable, INPUT + '/x14_pipeline.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
        "hits = glob.glob('/kaggle/input/**/x14_pipeline.py', recursive=True); print('pipeline at', hits)\nINPUT = os.path.dirname(hits[0])\n"
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
        src, dst = f if isinstance(f, tuple) else (f, os.path.basename(f))
        shutil.copy(src, os.path.join(cd, dst))
    json.dump({"title": f"whest-x14-code-{tg}", "id": f"{user}/whest-x14-code-{tg}", "licenses": [{"name": "MIT"}]},
              open(os.path.join(cd, "dataset-metadata.json"), "w"), indent=1)
for name, (user, env) in KERNELS.items():
    rd = os.path.join(HERE, name); os.makedirs(rd, exist_ok=True)
    slug = "whest-x14-" + name.replace("_", "-")
    json.dump(nb(env), open(os.path.join(rd, "x14-runner.ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": f"{user}/{slug}", "title": slug, "code_file": "x14-runner.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
               "dataset_sources": [f"{user}/whest-x14-code-{TAG[user]}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
    print(name, user, slug, env)
