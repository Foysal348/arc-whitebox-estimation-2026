"""Apply the V55 kappa4 pattern corrector (block + knobs from submissions/v55k4c/estimator.py) to another estimator.
Usage: python experiments/probes/apply_k4c.py <src_estimator.py> <dst_estimator.py> [reference estimator, default submissions/v55k4c]
"""
import sys, re
ref = open(sys.argv[3] if len(sys.argv) > 3 else "submissions/v55k4c/estimator.py", encoding="utf-8").read().split("\n")
src = open(sys.argv[1], encoding="utf-8").read().split("\n")
knobs = [l for l in ref if re.match(r"\s*K4C(_LMIN|_BLEND|_BF|_BL|_BLN|_LEAN|_DUMP)? = ", l)]
i0 = next(i for i, l in enumerate(ref) if l.strip() == "g4mean = g4row")
i1 = next(i for i, l in enumerate(ref) if "g4mean = g4" in l and "K4C_BLEND" in l)
block = ref[i0:i1 + 1]
ind_ref = re.match(r"\s*", ref[i0]).group(0)
out = []
done = [0, 0, 0]
for l in src:
    if re.match(r"\s*TQR = ", l) or (not done[0] and re.match(r"\s*D21COL = ", l)):
        out.append(l); out += knobs; done[0] = 1; continue
    if l.strip().startswith('b2 = {"ones2"') and not done[1]:
        ind = re.match(r"\s*", l).group(0)
        out += [ind + b[len(ind_ref):] if b.startswith(ind_ref) else b for b in block]; done[1] = 1
    if 'b1.update({"d3": D3, "g4": g4row})' in l:
        l = l.replace('"g4": g4row', '"g4": g4mean'); done[2] = 1
    out.append(l)
assert all(done), done
open(sys.argv[2], "w", encoding="utf-8").write("# V55 kappa4 pattern corrector applied to " + sys.argv[1] + "\n" + "\n".join(out))
print("ok", done)
