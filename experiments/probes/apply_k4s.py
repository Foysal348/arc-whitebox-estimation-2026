"""Apply the V56 HTQB-star feature (knobs, birth recording, star row in the K4C full-layer map) to a V55-style estimator
(lean or non-lean K4C block). Usage: python experiments/probes/apply_k4s.py <src.py> <dst.py> <K4S_BF colon list> [header]"""
import sys
src, dst, bf = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70]); s = s.replace(old, new)
rep("K4C_DUMP = []", f'''K4S = _os.environ.get("V56_K4S", "1") == "1"            # V56: Round 18 HTQB cubic star from the young dense legs
K4S_AGES = int(_os.environ.get("V56_K4S_AGES", "6"))
K4S_BF = [float(x) for x in _os.environ.get("V56_K4S_BF", "{bf}").split(":")]   # s3sq, y21, y21T, c2, absa, chain, STAR, 1 (fit pooled MLP #0 + synth, k4s_fit.py)
K4C_DUMP = []''')
rep("        s_list = []\n        e_list = []\n", "        s_list = []\n        e_list = []\n        k4s_c = []       # V56: per-source star coefficient c = -alpha phi / (6 var) at birth\n        k4s_dv = []      # V56: per-source diagonal piece w1 * var at birth (X = A + P diag(w1 var))\n")
rep("            a_b = fnp.multiply(w1col, C_off, out=legs[\"AP0\"][k_b, 0])\n",
    "            a_b = fnp.multiply(w1col, C_off, out=legs[\"AP0\"][k_b, 0])\n            if K4S:\n                k4s_c.append(alpha * phi * (-1.0 / 6.0) / var)\n                k4s_dv.append(w1 * var)\n")
ind = "                    "
star = (f"{ind}if K4S:\n"
        f"{ind}    # V56: HTQB cubic star 24 sum_j c_j P_ij X_ij^3 over the young dense slots, X = A + P diag(w1 var) at birth\n"
        f"{ind}    kk = A_st.shape[0]\n{ind}    j0 = max(ka, kk - K4S_AGES)\n"
        f"{ind}    if kk > j0:\n{ind}        Py = P_st[j0:kk]\n"
        f"{ind}        Xs = A_st[j0:kk] + Py * fnp.stack(k4s_dv[j0:kk], axis=0)[:, None, :]\n"
        f"{ind}        k4star = fnp.sum(Xs * Xs * Xs * Py * fnp.stack(k4s_c[j0:kk], axis=0)[:, None, :], axis=(0, 2)) * 24.0 / v2\n"
        f"{ind}    else:\n{ind}        k4star = ones_n * 0.0\n"
        f"{ind}    X = fnp.stack([s3sq, fnp.sum(Z2, axis=1), fnp.sum(Z2, axis=0), fnp.sum(rc * rc, axis=1), absa, ch, k4star, ones_n], axis=0)\n"
        f"{ind}    bt = K4S_BF\n{ind}else:\n")
old = f"{ind}X = fnp.stack([s3sq, fnp.sum(Z2, axis=1), fnp.sum(Z2, axis=0), fnp.sum(rc * rc, axis=1), absa, ch, ones_n], axis=0)\n{ind}bt = K4C_BF\n"
rep(old, star + f"{ind}    X = fnp.stack([s3sq, fnp.sum(Z2, axis=1), fnp.sum(Z2, axis=0), fnp.sum(rc * rc, axis=1), absa, ch, ones_n], axis=0)\n{ind}    bt = K4C_BF\n")
if len(sys.argv) > 4:
    s = sys.argv[4] + "\n" + s
open(dst, "w", encoding="utf-8").write(s)
print("ok", dst)
