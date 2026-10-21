"""X13: per-group Gram sums of the N2 table design (kaggle/x10/fit_x10.py recipe) from teacher-forced dumps, so a fit over
hundreds of nets needs only a few MB per kernel. Usage: python stats_x13.py <out.npz> <dump_dir> [<dump_dir> ...]
Design per layer l (identical to fit_x10.py / fit_x11.py): X = [87 e30_a features | sp*H_k(alpha)*st (k=0..6) |
sp*H_k(alpha)*sd (k=0..6)], sd = (g4c - g4)/var^2, target y = truth - base (base = chain mean before the table).
Groups: mini 0-69 -> 'm<id%5>' (fit, CV fold id%5); mini 70-99 -> 'ho' (held out, never fitted); full (id >= 1000)
-> 'u<id%5>'. Saved per group: G (16,101,101), B (16,101), YY, SX, SY, NN, ids, and per net onestep last-layer MSE.
"""
import os, sys, glob, re, json
import numpy as np

OUTF = sys.argv[1]; DIRS = sys.argv[2:]
NF = 101


def hermite(a, kmax=6):
    H = [np.ones_like(a), a]
    for k in range(2, kmax + 1):
        H.append(a * H[k - 1] - (k - 1) * H[k - 2])
    return H


def group(i):
    if 0 <= i <= 69:
        return "m%d" % (i % 5)
    if 70 <= i <= 99:
        return "ho"
    if i >= 1000:
        return "u%d" % (i % 5)
    raise ValueError(i)


files = {}
for d in DIRS:
    for f in glob.glob(os.path.join(d, "**", "net_*.npz"), recursive=True):
        i = int(re.search(r"net_(\d+)\.npz", os.path.basename(f)).group(1))
        if i in files:
            print("duplicate id", i, files[i], f, "-> keeping the first")
            continue
        files[i] = f
acc = {}
names = {}
bad = []
for i in sorted(files):
    try:
        z = np.load(files[i])
        F = z["F"].astype(np.float64); tr = z["truth"].astype(np.float64)
        base = np.concatenate([z["base_asm"], z["base"][15:16]]).astype(np.float64)
        st = z["st"].astype(np.float64); sp = z["sp"].astype(np.float64); al = z["alpha"].astype(np.float64)
        sd = (z["g4c"].astype(np.float64) - z["g4"].astype(np.float64)) / z["var"].astype(np.float64) ** 2
        nm = str(z["name"])
    except Exception as e:
        bad.append((i, repr(e)[:200])); continue
    g = group(i)
    a = acc.setdefault(g, {"G": np.zeros((16, NF, NF)), "B": np.zeros((16, NF)), "YY": np.zeros(16), "SX": np.zeros((16, NF)),
                           "SY": np.zeros(16), "NN": np.zeros(16), "ids": []})
    for l in range(16):
        H = hermite(al[l])
        X = np.concatenate([F[l], np.stack([sp[l] * H[k] * st[l] for k in range(7)], 1),
                            np.stack([sp[l] * H[k] * sd[l] for k in range(7)], 1)], 1)
        y = tr[l] - base[l]
        a["G"][l] += X.T @ X; a["B"][l] += X.T @ y; a["YY"][l] += y @ y; a["SX"][l] += X.sum(0); a["SY"][l] += y.sum(); a["NN"][l] += len(y)
    a["ids"].append(i); names[i] = nm
out = {}
for g, a in acc.items():
    for k, v in a.items():
        out[f"{g}__{k}"] = np.asarray(v)
out["names_json"] = np.asarray(json.dumps({str(k): v for k, v in names.items()}))
np.savez(OUTF, **out)
print("stats:", {g: len(a["ids"]) for g, a in sorted(acc.items())}, "bad", bad)
