"""X13 pipeline (20 Oct): refit the N2 mean-correction table on many public FULL-split nets. Runs inside one Kaggle CPU kernel.
Reference = e30kc1n2t (#337336, 2.8256e-9) = C2 knobs (nb + V21_R_OLD 416 + X_DEAD_TAU_B 2.4) + N2 table.
KIND=D (dump kernel): prep full shards SHARDS (prep_full.py; mini-overlapping nets excluded) [+ mini MINI_NETS] ->
        teacher-forced dump with the C2 knobs (x13_driver MODE=dump, NPROC processes) -> WORK/stats_<TAG>.npz
        (stats_x13.py: per-group Gram sums; also over any net_*.npz found under /kaggle/input = x11 C2 dumps of mini 0-69)
KIND=E (fit + eval kernel): stats_*.npz in INPUT -> fit_x13.py (N2big, N2mini, N2full + held-out one-step report) ->
        prep eval shards SHARDS -> eval arms N2 (reference) and N2big on EVAL_NETS (";"-separated groups, one arm per
        process) -> WORK/x13_<TAG>_summary.txt/.json (paired score ratios, wins)
"""
import os, sys, json, subprocess, time, glob, shutil
import numpy as np

INPUT = os.environ["INPUT"]; KIND = os.environ["KIND"]; TAG = os.environ.get("TAG", KIND)
WORK = os.environ.get("WORK", "/kaggle/working"); NPROC = int(os.environ.get("NPROC", "4"))
FULL_DIR = os.environ.get("FULL_DIR", "/tmp/x13_full")
C2 = {"X8_D21S": "0.98", "X2_K31_MIX": "0.4", "X8_SLOC": "0.2", "X8_SSTAR": "0.7", "V21_R_OLD": "416", "X_DEAD_TAU_B": "2.4"}
EST = os.path.join(INPUT, "e30k_x10.py")
DUMP = os.path.join(WORK, "dump"); TAB = os.path.join(WORK, "tables"); EV = os.path.join(WORK, "eval"); LOG = os.path.join(WORK, "logs")
for d in (DUMP, TAB, EV, LOG):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
BASE_ENV = dict(os.environ, EST=EST, FULL_DIR=FULL_DIR, PYTHONWARNINGS="ignore", PYTHONUTF8="1",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
STALL = int(os.environ.get("STALL", "1500"))


def say(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


def expand(nets):
    out = []
    for part in nets.split(","):
        a_, _, b_ = part.partition("-")
        out += list(range(int(a_), int(b_ or a_) + 1))
    return out


def rss(pid):
    try:
        return [l.split()[1] for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS")][0] + "k"
    except Exception:
        return "?"


def done_ids(stage, env):
    if stage == "dump":
        return {i for i in expand(env["NETS"]) if os.path.exists(os.path.join(env["OUT"], "net_%04d.npz" % i))}
    fn = os.path.join(env["OUT"], "eval_%s.json" % env["NETS"].replace(",", "_"))
    try:
        return {int(i) for rows in json.load(open(fn)).values() for i, x in rows.items() if "mse" in x}
    except Exception:
        return set()


def run_parallel(stage, jobs, maxp=None):
    """(x11 run_parallel) jobs: (name, env). At most maxp x13_driver.py processes; a job whose log has not grown for
    STALL s is killed and requeued (max 2x) on its unfinished nets."""
    maxp = maxp or NPROC
    queue = [(n, e, 0) for n, e in jobs]
    run = []; last = 0
    while queue or run:
        while queue and len(run) < maxp:
            name, env, tries = queue.pop(0)
            lf = open(os.path.join(LOG, f"{stage}_{name}.log"), "a")
            p = subprocess.Popen([sys.executable, os.path.join(INPUT, "x13_driver.py")], env=dict(BASE_ENV, **env),
                                 stdout=lf, stderr=subprocess.STDOUT, text=True)
            run.append([name, env, tries, p, lf, -1, time.time()])
            say(stage, name, "started pid", p.pid, "try", tries, "nets", env["NETS"][:60])
        time.sleep(20)
        for r in list(run):
            name, env, tries, p, lf, sz, t_ch = r
            if p.poll() is not None:
                lf.close(); run.remove(r); say(stage, name, "exit", p.returncode); continue
            lf.flush(); cur = os.path.getsize(lf.name)
            if cur != sz:
                r[5], r[6] = cur, time.time()
            elif time.time() - t_ch > STALL:
                say(stage, name, "STALLED", int(time.time() - t_ch), "s, rss", rss(p.pid), "-> kill")
                p.kill(); p.wait(); lf.close(); run.remove(r)
                if tries < 2:
                    left = [i for i in expand(env["NETS"]) if i not in done_ids(stage, env)]
                    if left:
                        queue.append((f"{name}_r{tries + 1}", dict(env, NETS=",".join(map(str, left))), tries + 1))
                        say(stage, name, "requeued nets", left)
        if time.time() - last > 600:
            last = time.time()
            tails = []
            for name, env, tries, p, lf, sz, t_ch in run:
                try:
                    ln = [x for x in open(lf.name, errors="replace").read().splitlines() if x.strip() and "arn" not in x]
                    tails.append(f"{name} rss {rss(p.pid)}: {ln[-1][:90] if ln else ''}")
                except Exception as e:
                    tails.append(f"{name}: ? {e}")
            n = len(glob.glob(os.path.join(DUMP, "net_*.npz")))
            say(stage, "progress npz", n, "queued", len(queue), "|", " || ".join(tails))
            os.system("free -g | head -2; df -h /tmp | tail -1")


def prep(shards):
    if not shards:
        return []
    r = subprocess.run([sys.executable, os.path.join(INPUT, "prep_full.py"), shards, FULL_DIR], env=BASE_ENV,
                       capture_output=True, text=True)
    open(os.path.join(LOG, "prep.log"), "w").write(r.stdout + r.stderr)
    print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
    assert r.returncode == 0, "prep failed"
    shutil.copy(os.path.join(FULL_DIR, "prep_report.json"), os.path.join(WORK, f"prep_report_{TAG}.json"))
    return [1000 + i for i in json.load(open(os.path.join(FULL_DIR, "prep_report.json")))["kept"]]


def mini_download():
    t0 = time.time()
    r = subprocess.run(["whest", "dataset", "download", "aicrowd/arc-whestbench-public-2026", "--revision", "v2-phase2",
                        "--split", "mini"], capture_output=True, text=True, env=dict(BASE_ENV, HF_XET_HIGH_PERFORMANCE="1"))
    say("mini download rc", r.returncode, "%.0fs" % (time.time() - t0), r.stdout[-300:], r.stderr[-300:])


if KIND == "D":
    full_ids = prep(os.environ.get("SHARDS", ""))
    mini_ids = expand(os.environ["MINI_NETS"]) if os.environ.get("MINI_NETS") else []
    assert not any(i < 70 for i in mini_ids)
    say("prep done:", len(full_ids), "full nets", full_ids[:3], "...", full_ids[-3:], "| mini", mini_ids[:1], mini_ids[-1:])
    if mini_ids:
        mini_download()
    ids = mini_ids + full_ids
    jobs = [(f"d{j}", {"MODE": "dump", "OUT": DUMP, "NETS": ",".join(map(str, ids[j::NPROC])), "DUMP_ENV": json.dumps(C2)})
            for j in range(NPROC) if ids[j::NPROC]]
    run_parallel("dump", jobs)
    shutil.rmtree(FULL_DIR, ignore_errors=True)
    nd = sorted(glob.glob(os.path.join(DUMP, "net_*.npz")))
    say("dump done:", len(nd), "of", len(ids), "nets")
    for f in sorted(glob.glob(os.path.join(LOG, "dump_*.log"))):
        ls = open(f, errors="replace").read().splitlines()
        print(f, [l for l in ls if "dump module" in l][:1], "forced_ok False:", [l[:60] for l in ls if "forced_ok=False" in l][:3], flush=True)
    srcs = [DUMP] + sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/net_*.npz", recursive=True)})
    say("stats over", srcs)
    r = subprocess.run([sys.executable, os.path.join(INPUT, "stats_x13.py"), os.path.join(WORK, f"stats_{TAG}.npz")] + srcs,
                       env=dict(BASE_ENV, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"), capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    assert r.returncode == 0
    say("ALL DONE")

if KIND == "E":
    STAGES = os.environ.get("STAGES", "fit,eval,sum").split(",")
    if "fit" in STAGES:
        sf = sorted(glob.glob(os.path.join(INPUT, "stats_*.npz")))
        r = subprocess.run([sys.executable, os.path.join(INPUT, "fit_x13.py"), TAB, os.path.join(INPUT, "N2.json")] + sf,
                           env=dict(BASE_ENV, EST_SRC=EST, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"), capture_output=True, text=True)
        open(os.path.join(TAB, "fit_log.txt"), "w").write(r.stdout + r.stderr)
        print(r.stdout[-12000:], r.stderr[-3000:], flush=True)
        assert r.returncode == 0, "fit failed"
        import hashlib
        say("fit done; N2big sha1", hashlib.sha1(open(os.path.join(TAB, "N2big.json"), "rb").read()).hexdigest())
    ARMS = {"N2": dict(C2, X10_TAB=os.path.join(INPUT, "N2.json")), "N2big": dict(C2, X10_TAB=os.path.join(TAB, "N2big.json"))}
    for extra in [a for a in os.environ.get("EXTRA_ARMS", "").split(",") if a]:
        ARMS[extra] = dict(C2, X10_TAB=os.path.join(TAB, extra + ".json"))
    json.dump(ARMS, open(os.path.join(WORK, f"x13_{TAG}_arms.json"), "w"), indent=1)
    if "eval" in STAGES:
        full_ok = set(prep(os.environ.get("SHARDS", "")))
        groups = []
        for g in os.environ["EVAL_NETS"].split(";"):
            ids = [i for i in expand(g) if i < 1000 or i in full_ok]
            if ids:
                groups.append(",".join(map(str, ids)))
        say("eval groups", groups)
        if any(i < 1000 for g in groups for i in expand(g)):
            mini_download()
        jobs = []
        for n in groups:
            for t, e in ARMS.items():
                od = os.path.join(EV, t); os.makedirs(od, exist_ok=True)
                jobs.append((f"{t}_{n.split(',')[0]}", {"MODE": "eval", "OUT": od, "ARMS": json.dumps({t: e}), "NETS": n}))
        run_parallel("eval", jobs)
    # ---- summary
    res = {t: {} for t in ARMS}
    for f in glob.glob(os.path.join(EV, "**", "eval_*.json"), recursive=True):
        for t, rows in json.load(open(f)).items():
            for i, x in rows.items():
                if "mse" in x:
                    res.setdefault(t, {})[int(i)] = x
    json.dump({t: {str(i): v for i, v in sorted(r.items())} for t, r in res.items()}, open(os.path.join(WORK, f"x13_{TAG}_eval.json"), "w"), indent=1)
    rng = np.random.default_rng(0); summ = {}
    lines = [f"X13 {TAG} ref N2 (C2 knobs {json.dumps(C2)}) elapsed {time.time() - T0:.0f}s"]
    for label, cond in (("held-out mini 70-99", lambda i: 70 <= i <= 99), ("held-out full", lambda i: i >= 1000),
                        ("in-sample mini 0-69", lambda i: i < 70)):
        common = sorted(i for i in res["N2"] if cond(i) and all(i in res[t] for t in ARMS))
        if not common:
            continue
        R = res["N2"]
        rm = np.mean([R[i]["mse"] for i in common]); ru = np.mean([R[i]["util"] for i in common]); rs = rm * max(0.1, ru)
        lines.append(f"[{label}] {len(common)} nets; ref N2 raw {rm:.4e} util {ru:.4f} score {rs:.4e}")
        summ[label] = {"nets": common}
        for t in ARMS:
            A = res[t]
            m = np.mean([A[i]["mse"] for i in common]); u = np.mean([A[i]["util"] for i in common]); s = m * max(0.1, u)
            ps = np.array([(A[i]["mse"] * max(0.1, A[i]["util"])) / (R[i]["mse"] * max(0.1, R[i]["util"])) for i in common])
            lr = np.log(ps); boot = np.array([rng.choice(lr, len(lr)).mean() for _ in range(4000)])
            lo, hi = np.exp(np.quantile(boot, [.05, .95]))
            d = {"raw": m, "util": u, "score": s, "score_ratio": s / rs, "raw_ratio": m / rm, "util_ratio": u / ru,
                 "geo": float(np.exp(lr.mean())), "geo90": [float(lo), float(hi)], "wins": int((ps < 1).sum()), "n": len(common)}
            summ[label][t] = d
            lines.append(f"  {t:7s} raw {m:.4e} util {u:.4f} score {s:.4e} | score x{s / rs:.4f} raw x{m / rm:.4f} util x{u / ru:.4f} | "
                         f"geo x{d['geo']:.4f} (90% {lo:.4f}-{hi:.4f}) wins {d['wins']}/{len(common)}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    open(os.path.join(WORK, f"x13_{TAG}_summary.txt"), "w").write(txt + "\n")
    json.dump(summ, open(os.path.join(WORK, f"x13_{TAG}_summary.json"), "w"), indent=1)
    say("ALL DONE")
