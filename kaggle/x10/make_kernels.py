"""X10 (20 Oct): Kaggle CPU kernels for the e30_a mean-table refit on our trajectory.
Usage: python kaggle/x10/make_kernels.py dump      -> kaggle/x10/{code_esfer,code_foysal,dump_1..4}
       python kaggle/x10/make_kernels.py eval      -> kaggle/x10/{evcode_esfer,evcode_foysal,eval_1..4} (needs tables/*.json)
Kernels 1-2 esfersami50, 3-4 foysalemonshanto (max 2 per account). flopscope 0.12.3 / whestbench 0.16.2 pins (as x8).
"""
import os, sys, json, shutil, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PIN = '"flopscope>=0.12.3,<0.13","whestbench>=0.16.2,<0.17"'
USERS = {1: "esfersami50", 2: "esfersami50", 3: "foysalemonshanto", 4: "foysalemonshanto"}
TAG = {"esfersami50": "esfer", "foysalemonshanto": "foysal"}


def nb(cmd_env, outdir):
    cells = [
        "import os, subprocess, sys, json, time, glob\nos.environ['PYTHONUTF8']='1'\n"
        "print(subprocess.run([sys.executable,'-m','pip','install','-q'," + PIN.replace('"', "'") + "],capture_output=True,text=True).stdout[-500:])\n"
        "from importlib.metadata import version; print('flopscope', version('flopscope'), 'whestbench', version('whestbench'))\n"
        "print('cpus', os.cpu_count()); print(subprocess.run(['free','-g'],capture_output=True,text=True).stdout)",
        "hits = glob.glob('/kaggle/input/**/x10_driver.py', recursive=True); print('driver at', hits)\nINPUT = os.path.dirname(hits[0])\n"
        "print(sorted(os.listdir(INPUT)))",
        "os.environ['HF_XET_HIGH_PERFORMANCE']='1'\nt0=time.time()\n"
        "r=subprocess.run(['whest','dataset','download','aicrowd/arc-whestbench-public-2026','--revision','v2-phase2','--split','mini'],capture_output=True,text=True)\n"
        "print(r.stdout[-800:], r.stderr[-800:]); print('download s', time.time()-t0)",
        "env = dict(os.environ); env.update(" + json.dumps(cmd_env) + ")\n"
        "env.update({'EST': INPUT + '/e30k_x10.py', 'OUT': '" + outdir + "', 'PYTHONWARNINGS': 'ignore', 'PYTHONUTF8': '1'})\n"
        "if 'ARMS' in env:\n"
        "    arms = json.loads(env['ARMS'])\n"
        "    for t, e in arms.items():\n"
        "        if 'X10_TAB' in e and not e['X10_TAB'].startswith('/'): e['X10_TAB'] = INPUT + '/' + e['X10_TAB']\n"
        "    env['ARMS'] = json.dumps(arms)\n"
        "t0 = time.time()\n"
        "p = subprocess.Popen([sys.executable, INPUT + '/x10_driver.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
        "for line in p.stdout:\n"
        "    if 'Warning' in line or 'warnings.warn' in line: continue\n"
        "    print(line, end='', flush=True)\n"
        "p.wait(); print('exit', p.returncode, 'elapsed', time.time() - t0)\n"
        "print(subprocess.run(['du', '-sh', '" + outdir + "'], capture_output=True, text=True).stdout)",
    ]
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def write(kind, slices, envs, code_files, ds_prefix, k_prefix, outdir):
    for user in sorted(set(USERS.values())):
        cd = os.path.join(HERE, f"{ds_prefix}_{TAG[user]}"); os.makedirs(cd, exist_ok=True)
        for f in code_files:
            shutil.copy(f, os.path.join(cd, os.path.basename(f)))
        json.dump({"title": f"whest-x10-{ds_prefix}-{TAG[user]}", "id": f"{user}/whest-x10-{ds_prefix}-{TAG[user]}",
                   "licenses": [{"name": "MIT"}]}, open(os.path.join(cd, "dataset-metadata.json"), "w"), indent=1)
    for k, (a, b) in slices.items():
        user = USERS[k]
        rd = os.path.join(HERE, f"{k_prefix}_{k}"); os.makedirs(rd, exist_ok=True)
        env = dict(envs, START=str(a), END=str(b))
        if isinstance(envs.get("NETS"), dict):
            env["NETS"] = envs["NETS"][k]
        json.dump(nb(env, outdir), open(os.path.join(rd, "x10-runner.ipynb"), "w", encoding="utf-8"), indent=1)
        json.dump({"id": f"{user}/whest-x10-{k_prefix}-{k}", "title": f"whest-x10-{k_prefix}-{k}", "code_file": "x10-runner.ipynb",
                   "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
                   "dataset_sources": [f"{user}/whest-x10-{ds_prefix}-{TAG[user]}"], "competition_sources": [], "kernel_sources": []},
                  open(os.path.join(rd, "kernel-metadata.json"), "w"), indent=1)
        print(k, user, a, b)


EST = os.path.join(ROOT, "experiments/variants/e30k_x10.py")
DRV = os.path.join(HERE, "code/x10_driver.py")
if sys.argv[1] == "dump":
    write("dump", {1: (0, 18), 2: (18, 36), 3: (36, 53), 4: (53, 70)}, {"MODE": "dump"}, [EST, DRV], "code", "dump",
          "/kaggle/working/x10dump")
elif sys.argv[1] == "dumpnb":
    # teacher-forced dump on the newest base (x8c arm c1: D21S 0.98, K31 mix 0.4, SLOC 0.2, SSTAR 0.7), nets 0-69,
    # one kernel per account (kernels dnb_1 esfersami50 nets 0-35, dnb_3 foysalemonshanto nets 35-70)
    NB = {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7"}
    write("dump", {1: (0, 35), 3: (35, 70)}, {"MODE": "dump", "DUMP_ENV": json.dumps(NB)}, [EST, DRV], "code", "dnb",
          "/kaggle/working/x10dump")
elif sys.argv[1] == "eval2":
    # python make_kernels.py eval2 <arms.json> <ds_prefix> <k_prefix>: two kernels (1 esfersami50 nets 70-84 + 0-34,
    # 3 foysalemonshanto nets 85-99 + 35-69), all arms of <arms.json>; tables/*.json shipped next to the estimator
    arms = json.load(open(sys.argv[2]))
    # nets 70-99 (out of sample for tables fitted on 0-69) first, so partial results answer the in-sample guard early
    write("eval", {1: (0, 50), 3: (50, 100)}, {"MODE": "eval", "ARMS": json.dumps(arms),
                                               "NETS": {1: "70-84,0-34", 3: "85-99,35-69"}},
          [EST, DRV] + sorted(glob.glob(os.path.join(HERE, "tables", "*.json"))), sys.argv[3], sys.argv[4], "/kaggle/working/x10eval")
elif sys.argv[1] == "eval":
    # ARMS from kaggle/x10/arms.json ({tag: env}); tables from kaggle/x10/tables/*.json shipped next to the estimator
    arms = json.load(open(os.path.join(HERE, "arms.json")))
    lo, hi = int(sys.argv[2]), int(sys.argv[3])
    n = hi - lo; cuts = [lo + round(n * j / 4) for j in range(5)]
    write("eval", {k: (cuts[k - 1], cuts[k]) for k in range(1, 5)}, {"MODE": "eval", "ARMS": json.dumps(arms)},
          [EST, DRV] + sorted(glob.glob(os.path.join(HERE, "tables", "*.json"))), sys.argv[4], sys.argv[5], "/kaggle/working/x10eval")
