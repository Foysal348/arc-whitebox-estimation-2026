"""X13 (20 Oct): N2 table refit on ~270 public FULL-split nets + mini 0-69, all on Kaggle.
  python kaggle/x13/make_kernels.py code            -> kaggle/x13/code_{esfer,foysal} (code datasets; stats_*.npz in
                                                       kaggle/x13/stats/ are added when present)
  python kaggle/x13/make_kernels.py dump            -> kernels d1_1, d2_2 (foysalemonshanto), d3_3, d4_4 (esfersami50)
  python kaggle/x13/make_kernels.py eval            -> kernels e1_1 (foysalemonshanto, mini nets), e2_3 (esfersami50, full held-out)
D kernels: prep full shards -> teacher-forced dump (C2 knobs = e30kc1n2t) -> stats_<TAG>.npz (Gram sums per group).
D1 also dumps mini 70-99 (held-out one-step diagnostic only, never fitted) and reads the x11 C2 dumps of mini 0-69
(kernel source foysalemonshanto/whest-x11-c2-3, same C2 knobs).
E kernels: fit (N2big etc.) from all stats -> eval N2 vs N2big -> summary.
"""
import os, sys, json, shutil, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PIN = "'flopscope>=0.12.3,<0.13','whestbench>=0.16.2,<0.17'"
TAG = {"esfersami50": "esfer", "foysalemonshanto": "foysal"}
CODE = [os.path.join(ROOT, "experiments/variants/e30k_x10.py"), os.path.join(ROOT, "kaggle/x10/tables/N2.json")] + \
       [os.path.join(HERE, f) for f in ("x13_driver.py", "x13_pipeline.py", "prep_full.py", "stats_x13.py", "fit_x13.py")]
KERNELS = {
    "d1_1": ("foysalemonshanto", {"KIND": "D", "TAG": "D1", "SHARDS": "0-1", "MINI_NETS": "70-99"}, ["foysalemonshanto/whest-x11-c2-3"]),
    "d2_2": ("foysalemonshanto", {"KIND": "D", "TAG": "D2", "SHARDS": "2-6"}, []),
    "d3_3": ("esfersami50", {"KIND": "D", "TAG": "D3", "SHARDS": "7-11"}, []),
    "d4_4": ("esfersami50", {"KIND": "D", "TAG": "D4", "SHARDS": "12-16"}, []),
    "e1_1": ("foysalemonshanto", {"KIND": "E", "TAG": "E1", "EVAL_NETS": "70-77;78-84;85-92;93-99;0-4;5-9"}, []),
    "e2_3": ("esfersami50", {"KIND": "E", "TAG": "E2", "SHARDS": "40-41",
                             "EVAL_NETS": "1640-1647;1648-1655;1656-1663;1664-1671"}, []),
}

CELL_RUN = r"""env = dict(os.environ); env.update(__ENV__)
env.update({'INPUT': INPUT, 'WORK': '/kaggle/working', 'PYTHONWARNINGS': 'ignore', 'PYTHONUTF8': '1', 'NPROC': '4'})
t0 = time.time()
p = subprocess.Popen([sys.executable, INPUT + '/x13_pipeline.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
        "print('cpus', os.cpu_count()); print(subprocess.run(['free','-g'],capture_output=True,text=True).stdout)\n"
        "print(subprocess.run(['df','-h','/tmp','/kaggle/working'],capture_output=True,text=True).stdout)",
        "hits = glob.glob('/kaggle/input/**/x13_pipeline.py', recursive=True); print('pipeline at', hits)\nINPUT = os.path.dirname(hits[0])\n"
        "print(sorted(os.listdir(INPUT))); print('kernel-source npz:', len(glob.glob('/kaggle/input/**/net_*.npz', recursive=True)))",
        CELL_RUN.replace("__ENV__", json.dumps(env)),
    ]
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


mode = sys.argv[1]
if mode == "code":
    stats = sorted(glob.glob(os.path.join(HERE, "stats", "stats_*.npz")))
    for user, tg in TAG.items():
        cd = os.path.join(HERE, f"code_{tg}"); os.makedirs(cd, exist_ok=True)
        for f in CODE + stats:
            shutil.copy(f, os.path.join(cd, os.path.basename(f)))
        json.dump({"title": f"whest-x13-code-{tg}", "id": f"{user}/whest-x13-code-{tg}", "licenses": [{"name": "MIT"}]},
                  open(os.path.join(cd, "dataset-metadata.json"), "w"), indent=1)
        print(cd, sorted(os.listdir(cd)))
if mode in ("dump", "eval"):
    for kd, (user, env, ksrc) in KERNELS.items():
        if (mode == "dump") != kd.startswith("d"):
            continue
        rd = os.path.join(HERE, kd); os.makedirs(rd, exist_ok=True)
        json.dump(nb(env), open(os.path.join(rd, "x13-runner.ipynb"), "w", encoding="utf-8"), indent=1)
        name = "whest-x13-" + kd.replace("_", "-")
        json.dump({"id": f"{user}/{name}", "title": name, "code_file": "x13-runner.ipynb", "language": "python",
                   "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
                   "dataset_sources": [f"{user}/whest-x13-code-{TAG[user]}"], "competition_sources": [], "kernel_sources": ksrc},
                  open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
        print(kd, user, env, ksrc)
