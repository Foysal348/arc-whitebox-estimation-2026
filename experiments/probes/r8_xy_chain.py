"""Round 8 (Fable, 23 Sep): END-TO-END test of the K x C families at the Edgeworth coefficient WITHOUT doubling the K x K
cross term.

The newborn hub Sym(X1 x I x Y1), X1 = 3 a_b + Xt, Y1 = a_b w2 + Yt (Xt = 1.5 d(w2) D21, Yt = 0.5 d(w1) D21^T d(w3)) carries
families A and B (K x C) at half their Edgeworth coefficient through a_b x Yt and Xt x a_b, and type R (K x K, see
r8_kk_terms.py) at half its coefficient through Xt x Yt.  Round 7's knob R7_FB_SCALE = S multiplies Xt and Yt, so it
scales A, B by S and R by S^2: S = 2 makes A, B exact but doubles R.  This probe separates the two: S scales Xt and Yt
(A, B -> S/2 of exact), and XY scales the Xt x Yt cross tensor on top (R -> XY S^2 / 2 of exact), consistently at
birth (D21_new, D3_new) and in every later contraction of the source (_dslices: the (1/3) Xt*Yt block, the (1/3) Xt*P
thin right factor of Yt, the P*Yt thin right factor of Xt).  S = 2, XY = 0.5 is the full Edgeworth value of A, B and R;
S = 1, XY = 1 is the shipped chain; S = 2, XY = 1 is round 7's S = 2 (+3.4%).  Zero extra FLOPs.

Reports per configuration: final MSE against the baked truth (public MLP #0), per-layer MSE, the (2,1)-slice error
against the pre-activation teacher at layers 4, 8, 12, 15, 16.

Usage (from whest-starterkit):  PYTHONUTF8=1 V31_ANGULAR=0 V19_FULL_LAST=1 V26_STRASSEN=0 python -m uv run python ../experiments/probes/r8_xy_chain.py "S=2,XY=0.5" "S=1,XY=2" ...
"""
import os, sys, types, json, time, glob
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
assert os.environ.get("V31_ANGULAR") == "0"
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"
SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
LAY = {4, 8, 12, 15, 16}
src = open(SRC, encoding="utf-8").read()
# ---- knobs ----
p1 = "                R1T_b = Bf.T * 1.5                              # (n, rfb) = R1^T\n"
p2 = "                R2T_b = Qf * (w3 * 0.5)[:, None]     # (n, rfb) = R2^T\n"
# birth dslices of the newborn: scale the Xt x Yt cross content by XY
p3 = ("                D21_new = fnp.multiply((xd)[:, None], Y1_b.T, out=NN(\"d21new\"))\n"
      "                fnp.multiply(X1_b, Y1_b, out=T1)\n"
      "                fnp.add(D21_new, T1, out=D21_new)\n"
      "                fnp.multiply((yd)[:, None], X1_b.T, out=T1)\n"
      "                fnp.add(D21_new, T1, out=D21_new)\n"
      "                fnp.multiply(D21_new, 1.0 / 3.0, out=D21_new)\n"
      "                D3_new = xd * yd\n")
r3 = ("                _c = 1.0 - _XY\n"
      "                D21_new = ((xd)[:, None] * (Y1_b - _c * Yt_b).T + (X1_b * Y1_b - _c * Xt_b * Yt_b) + (yd)[:, None] * (X1_b - _c * Xt_b).T) * (1.0 / 3.0)\n"
      "                D3_new = xd * yd * _XY\n")
# _dslices: the three places where the transported Xt x Yt cross tensor enters
p4 = ("            fnp.multiply(Xt, Yt, out=U)\n"
      "            fnp.multiply(U, 1.0 / 3.0, out=U)\n")
r4 = ("            fnp.multiply(Xt, Yt, out=U)\n"
      "            fnp.multiply(U, _XY / 3.0, out=U)\n")
p5 = ("                fnp.multiply(Xt, P_st, out=T)\n"
      "                fnp.multiply(T, 1.0 / 3.0, out=T)\n"
      "                fnp.add(T, AP, out=T)                 # GY\n")
r5 = ("                fnp.multiply(Xt, P_st, out=T)\n"
      "                fnp.multiply(T, _XY / 3.0, out=T)\n"
      "                fnp.add(T, AP, out=T)                 # GY\n")
p6 = ("                fnp.multiply(AP, W2B, out=T)\n"
      "                fnp.add(T, U, out=T)                  # AP*w2 + P*Yt   (U holds P*Yt)\n")
r6 = ("                fnp.multiply(AP, W2B, out=T)\n"
      "                fnp.add(T, U * _XY, out=T)            # AP*w2 + XY P*Yt   (U holds P*Yt)\n")
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
for p in (p1, p2, p3, p4, p5, p6, a1): assert src.count(p) == 1, p[:60]
src = (src.replace(p1, p1.replace("* 1.5", "* (1.5 * _S)")).replace(p2, p2.replace("w3 * 0.5", "w3 * (0.5 * _S)"))
          .replace(p3, r3).replace(p4, r4).replace(p5, r5).replace(p6, r6).replace(a1, "            _CAP(li, var, D21)\n" + a1))
S = {}
def _CAP(li, var, D21):
    if li + 1 in LAY and D21 is not None: S[li + 1] = np.asarray(D21, np.float32).copy()
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
ALM = np.asarray(truth["all_layer_means"], np.float64); off = ~np.eye(1024, dtype=bool)
TRUE = {l: np.asarray(np.load(os.path.join(PRE, f"layer{l}_seed0.npz"))["Y21"], np.float64) for l in LAY}
OUT = f"../results/summaries/r8_xy_chain{os.environ.get('TAG', '')}.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for cfg in sys.argv[1:]:
    kv = dict(x.split("=") for x in cfg.split(",")); s_, xy = float(kv.get("S", 1.0)), float(kv.get("XY", 1.0)); S.clear()
    mod = types.ModuleType(f"est_r8_{cfg}"); mod.__dict__.update({"_CAP": _CAP, "_S": s_, "_XY": xy})
    exec(compile(src, SRC, "exec"), mod.__dict__)
    est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    per = [float(np.mean((pred[k] - ALM[k]) ** 2)) for k in range(16)]
    d21 = {}
    for l in sorted(S):
        D = S[l]; Y = TRUE[l]; e = (D.astype(np.float64) - Y)[off]
        d21[l] = {"rel_err": float(np.sqrt(np.mean(e ** 2) / np.mean(Y[off] ** 2))), "err_rms": float(np.sqrt(np.mean(e ** 2)))}
    res[cfg] = {"S": s_, "XY": xy, "final": per[-1], "per_layer": per, "d21": d21, "util": ctx.flops_used / 2**41,
                "coef": {"famAB_over_exact": s_ / 2, "typeR_over_exact": xy * s_ * s_ / 2}}
    print(f"[{cfg}] A,B at {s_/2:.2f} x exact, R at {xy*s_*s_/2:.2f} x exact: final MSE {per[-1]:.4e} (ratio to shipped 2.2930e-8: {per[-1]/2.2930e-8:.4f}; L8 {per[7]:.3e}, L12 {per[11]:.3e}) "
          f"util {ctx.flops_used/2**41:.4f} ({time.time()-t0:.0f}s) | D21 rel err: " + ", ".join(f"L{l} {d21[l]['rel_err']*100:.2f}%" for l in sorted(d21)), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    del est, mod
