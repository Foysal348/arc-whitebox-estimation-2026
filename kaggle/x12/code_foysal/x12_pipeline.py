"""X12 end-to-end pipeline (20 Oct), runs entirely inside one Kaggle CPU kernel (no laptop step). Reuses the x11
machinery (kaggle/x11/x11_pipeline.py: one arm per eval process, STALL watchdog with requeue).
Reference = e30kc1n2t (submission #337336): nb knobs + V21_R_OLD 416 + X_DEAD_TAU_B 2.4 + N2 table.
Env: INPUT (dataset dir: e30k_x12.py, x12_driver.py, fit_x12.py, N2.json), KIND, WORK, NPROC (4), STAGES, EVAL_NETS.
KIND selects dumps / fits / arms:
  F : teacher-forced dump (X_TAB_FORCE=1)                    -> fit F (T2 = N2b, N3)  -> arms ref, F_N3
  P : free-running dump with N2 active (on-policy, DAgger 1)  -> fit P (T2, N3)        -> arms ref, P_T2, P_N3
  M : both dumps, Gram sums added per net (DAgger aggregate)  -> fit M (T2, N3)        -> arms ref, M_T2, M_N3
NOTE: with X_TAB_FORCE=1 the table only moves mu, which is then overwritten by the truth, so a forced dump is identical
with or without N2: the F fit of T2 is exactly the C2R refit of x11 (measured x1.000 vs N2); kept for the CV report.
"""
import os, sys, json, subprocess, time, glob, shutil
import numpy as np

INPUT = os.environ["INPUT"]; KIND = os.environ["KIND"]
WORK = os.environ.get("WORK", "/kaggle/working"); NPROC = int(os.environ.get("NPROC", "4"))
REF = {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "V21_R_OLD": "416", "X_DEAD_TAU_B": "2.4"}
N2 = os.path.join(INPUT, "N2.json")
DUMP = os.path.join(WORK, "dump"); TAB = os.path.join(WORK, "tables"); EV = os.path.join(WORK, "eval"); LOG = os.path.join(WORK, "logs")
for d in (DUMP, TAB, EV, LOG):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
STAGES = os.environ.get("STAGES", "dump,fit,eval,sum").split(",")
BASE_ENV = dict(os.environ, EST=os.path.join(INPUT, "e30k_x12.py"), PYTHONWARNINGS="ignore", PYTHONUTF8="1",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
DUMPS = {"F": ["F"], "P": ["P"], "M": ["F", "P"]}[KIND]
DUMP_ENVS = {"F": dict(REF, X10_TAB=N2, X_TAB_FORCE="1"), "P": dict(REF, X10_TAB=N2, X_TAB_FORCE="0")}
DUMP_NETS = os.environ.get("DUMP_NETS", "0-69")
FIT_NETS = os.environ.get("FIT_NETS", DUMP_NETS)


def say(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


STALL = int(os.environ.get("STALL", "1500"))   # s without log growth -> the job is treated as hung, killed and requeued


def rss(pid):
    try:
        return [l.split()[1] for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS")][0] + "k"
    except Exception:
        return "?"


def run_parallel(stage, jobs, maxp=None):
    """(x11 copy) jobs: list of (name, env[, nets]). Runs x12_driver.py per job, at most maxp at once, logs to
    LOG/<stage>_<name>.log. Watchdog: a job whose log has not grown for STALL s is killed; an eval job is requeued (max 2x)
    on its not yet finished nets."""
    maxp = maxp or NPROC
    queue = [tuple(j) + ((None,) * (3 - len(j))) + (0,) for j in jobs]   # (name, env, nets, tries)
    run = []
    last = 0
    while queue or run:
        while queue and len(run) < maxp:
            name, env, nets, tries = queue.pop(0)
            lf = open(os.path.join(LOG, f"{stage}_{name}.log"), "a")
            p = subprocess.Popen([sys.executable, os.path.join(INPUT, "x12_driver.py")], env=dict(BASE_ENV, **env),
                                 stdout=lf, stderr=subprocess.STDOUT, text=True)
            run.append([name, env, nets, tries, p, lf, -1, time.time()])
            say(stage, name, "started pid", p.pid, "try", tries)
        time.sleep(20)
        for r in list(run):
            name, env, nets, tries, p, lf, sz, t_ch = r
            if p.poll() is not None:
                lf.close(); run.remove(r); say(stage, name, "exit", p.returncode); continue
            lf.flush(); cur = os.path.getsize(lf.name)
            if cur != sz:
                r[6], r[7] = cur, time.time()
            elif time.time() - t_ch > STALL:
                say(stage, name, "STALLED", int(time.time() - t_ch), "s, rss", rss(p.pid), "-> kill")
                p.kill(); p.wait(); lf.close(); run.remove(r)
                if nets is not None and tries < 2:
                    fn = os.path.join(env["OUT"], "eval_%s.json" % env["NETS"].replace(",", "_"))
                    done = set()
                    try:
                        done = {int(i) for rows in json.load(open(fn)).values() for i, x in rows.items() if "mse" in x}
                    except Exception:
                        pass
                    left = [i for i in nets if i not in done]
                    if left:
                        e2 = dict(env, NETS=",".join(map(str, left)))
                        queue.append((f"{name}_r{tries + 1}", e2, left, tries + 1))
                        say(stage, name, "requeued nets", left)
        if time.time() - last > 600:
            last = time.time()
            n = len(glob.glob(os.path.join(DUMP, "**", "net_*.npz"), recursive=True)) if stage == "dump" else None
            tails = []
            for name, env, nets, tries, p, lf, sz, t_ch in run:
                try:
                    ln = [x for x in open(lf.name, errors="replace").read().splitlines() if x.strip() and "Warning" not in x and "warn" not in x]
                    tails.append(f"{name} rss {rss(p.pid)}: {ln[-1][:100] if ln else ''}")
                except Exception as e:
                    tails.append(f"{name}: ? {e}")
            say(stage, "progress", f"npz={n}" if n is not None else "", f"queued={len(queue)}", "|", " || ".join(tails))
            os.system("free -g | head -2")


def expand(nets):
    out = []
    for part in nets.split(","):
        a_, _, b_ = part.partition("-")
        out += list(range(int(a_), int(b_ or a_) + 1))
    return out


# ---- 1. dumps (all slices of all dump kinds in one queue, NPROC at a time)
if "dump" in STAGES:
    d0, d1 = map(int, DUMP_NETS.split("-"))
    nn = d1 + 1 - d0
    cuts = [d0 + round(nn * j / NPROC) for j in range(NPROC + 1)]
    jobs = []
    for dk in DUMPS:
        od = os.path.join(DUMP, dk); os.makedirs(od, exist_ok=True)
        jobs += [(f"{dk}_{cuts[j]}_{cuts[j + 1]}", {"MODE": "dump", "OUT": od, "START": str(cuts[j]), "END": str(cuts[j + 1]),
                                                     "DUMP_ENV": json.dumps(DUMP_ENVS[dk])}) for j in range(NPROC) if cuts[j + 1] > cuts[j]]
    run_parallel("dump", jobs)
for dk in DUMPS:
    say("dump", dk, len(glob.glob(os.path.join(DUMP, dk, "net_*.npz"))), "nets")
for f in sorted(glob.glob(os.path.join(LOG, "dump_*.log"))):
    print(f, "".join([l for l in open(f, errors="replace") if "forced_ok" in l or "dump module" in l][:2]), flush=True)

# ---- 2. fit
if "fit" in STAGES:
    r = subprocess.run([sys.executable, os.path.join(INPUT, "fit_x12.py"), TAB, KIND] + [os.path.join(DUMP, dk) for dk in DUMPS],
                       env=dict(BASE_ENV, NETS=FIT_NETS, EST_SRC=os.path.join(INPUT, "e30k_x12.py"), N2_JSON=N2,
                                OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4"), capture_output=True, text=True)
    open(os.path.join(TAB, "fit_log_%s.txt" % KIND), "w").write(r.stdout + r.stderr)
    print(r.stdout[-9000:], r.stderr[-3000:], flush=True)
    assert r.returncode == 0, "fit failed"
    for T in ("T2", "N3"):
        shutil.copy(os.path.join(TAB, "%s_%s.json" % (KIND, T)), os.path.join(WORK, "%s_%s.json" % (KIND, T)))
    say("fit done")
elif os.path.exists(os.path.join(INPUT, "%s_T2.json" % KIND)):
    for T in ("T2", "N3"):
        shutil.copy(os.path.join(INPUT, "%s_%s.json" % (KIND, T)), os.path.join(WORK, "%s_%s.json" % (KIND, T)))

# ---- 3. eval: one arm per process
ARMS = {"ref": dict(REF, X10_TAB=N2)}
for T in {"F": ["N3"], "P": ["T2", "N3"], "M": ["T2", "N3"]}[KIND]:
    ARMS["%s_%s" % (KIND, T)] = dict(REF, X10_TAB=os.path.join(WORK, "%s_%s.json" % (KIND, T)))
if os.environ.get("ARMS_ONLY"):
    ARMS = {t: e for t, e in ARMS.items() if t in os.environ["ARMS_ONLY"].split(",")}
json.dump(ARMS, open(os.path.join(WORK, f"x12_{KIND}_arms.json"), "w"), indent=1)
NETS = os.environ.get("EVAL_NETS", "70-77,0-3;78-84,20-23;85-92,40-43;93-99,60-62").split(";")
if "eval" in STAGES:
    jobs = []
    for n in NETS:
        for t, e in ARMS.items():
            od = os.path.join(EV, t); os.makedirs(od, exist_ok=True)
            jobs.append((f"{t}_{n.replace(',', '_')}", {"MODE": "eval", "OUT": od, "ARMS": json.dumps({t: e}), "NETS": n}, expand(n)))
    run_parallel("eval", jobs)

# ---- 4. collect + summary (paired per-net score ratios vs ref; score = mse * max(0.1, util))
res = {t: {} for t in ARMS}
for f in glob.glob(os.path.join(EV, "**", "eval_*.json"), recursive=True):
    for t, rows in json.load(open(f)).items():
        for i, x in rows.items():
            if "mse" in x:
                res.setdefault(t, {})[i] = x
json.dump(res, open(os.path.join(WORK, f"x12_{KIND}_eval.json"), "w"), indent=1)
summ = {}
lines = [f"X12 {KIND} ref env {json.dumps(REF)} + N2; elapsed {time.time() - T0:.0f}s"]
rng = np.random.default_rng(0)
for label, sub in (("held-out 70-99", range(70, 100)), ("in-sample 0-69", range(0, 70)), ("all", range(0, 100))):
    common = sorted(i for i in sub if all(str(i) in res.get(t, {}) for t in ARMS))
    if not common:
        lines.append(f"[{label}] no common nets"); continue
    R = res["ref"]
    rm = np.mean([R[str(i)]["mse"] for i in common]); ru = np.mean([R[str(i)]["util"] for i in common]); rs = rm * max(0.1, ru)
    lines.append(f"[{label}] {len(common)} nets; ref raw {rm:.4e} util {ru:.4f} score {rs:.4e}")
    summ[label] = {"nets": common}
    for t in ARMS:
        A = res[t]
        m = np.mean([A[str(i)]["mse"] for i in common]); u = np.mean([A[str(i)]["util"] for i in common]); s = m * max(0.1, u)
        ps = np.array([(A[str(i)]["mse"] * max(0.1, A[str(i)]["util"])) / (R[str(i)]["mse"] * max(0.1, R[str(i)]["util"])) for i in common])
        lr = np.log(ps); boot = np.array([rng.choice(lr, len(lr)).mean() for _ in range(2000)])
        d = {"raw": m, "util": u, "score": s, "score_ratio": s / rs, "raw_ratio": m / rm, "util_ratio": u / ru,
             "geo_score_ratio": float(np.exp(lr.mean())), "geo90": [float(np.exp(np.quantile(boot, .05))), float(np.exp(np.quantile(boot, .95)))],
             "wins": int((ps < 1).sum()), "n": len(common)}
        summ[label][t] = d
        lines.append(f"  {t:6s} raw {m:.4e} util {u:.4f} score {s:.4e} | score x{s / rs:.4f} raw x{m / rm:.4f} util x{u / ru:.4f} | "
                     f"geo x{d['geo_score_ratio']:.4f} (90% {d['geo90'][0]:.4f}-{d['geo90'][1]:.4f}) wins {d['wins']}/{len(common)}")
txt = "\n".join(lines)
print(txt, flush=True)
open(os.path.join(WORK, f"x12_{KIND}_summary.txt"), "w").write(txt + "\n")
json.dump(summ, open(os.path.join(WORK, f"x12_{KIND}_summary.json"), "w"), indent=1)
say("ALL DONE")
