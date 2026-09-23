"""Round 7 (Fable, 23 Sep): ONE plain-Gaussian chain run that dumps everything the slice-error analysis needs.

Per layer l = li + 1 (pre-activation stage, right before the wick matrix): mu, var, D3, D21, C_off, g4row, wk4m, and the
wick table W_all (n x 21).  After the closure: the post-ReLU K1 (mean), K2 (var), K3 (kappa3 diag), K4 (kappa4 diag),
K21 (kappa3(h_i,h_i,h_j)), K22, C_post.  At the post layers of the post-ReLU teacher (l in 3, 7, 11, 14, 15) also the
dense source stacks seen by _dslices (A, P, Z, L, Zf, R1T, R2T + the static per-source vectors), so the chain's own
all-distinct kappa3 tensor can be reconstructed offline.

Run with every source dense (V21_NO_CONFINE=1), the D21 of the last layer computed (V19_FULL_LAST=1) and no Strassen
pools (V26_STRASSEN=0, less RAM; values unchanged to 4 digits).  Writes data/dumps/r7_cap_L{l}.npz (one file per layer).

Usage (from whest-starterkit):
  PYTHONUTF8=1 V31_ANGULAR=0 V21_NO_CONFINE=1 V19_FULL_LAST=1 V26_STRASSEN=0 python -m uv run python ../experiments/probes/r7_capture.py
"""
import os, sys, types, time, glob, json
import numpy as np
sys.path.insert(0, os.path.abspath("."))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
assert os.environ.get("V31_ANGULAR") == "0" and os.environ.get("V21_NO_CONFINE") == "1" and os.environ.get("V19_FULL_LAST") == "1"
PRE = "../data/kaggle_out/kaggle_out_k4pub/k4teach"
OUT = "../data/dumps"
SRC = os.environ.get("EST", "../submissions/v31ang/estimator.py")
SRC_LAYERS = {3, 7, 11, 14, 15}           # layers whose source stacks are dumped (post-ReLU teacher layers)
src = open(SRC, encoding="utf-8").read()
a1 = "            # ---- wick matrix ----\n            sigma = fnp.sqrt(var)\n"
a2 = "            WT = W_all.T\n"
a3 = "            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]\n"
a4 = "            C = flops.as_symmetric(C, symmetry=(0, 1))\n"
a5 = "        k = len(w2b_list)\n        W2B = fnp.stack(w2b_list, axis=0)[:, None, :]\n"
a6 = "            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]\n"
for a in (a1, a2, a3, a4, a5, a6): assert src.count(a) == 1, a
src = src.replace(a1, "            _CAPW(li, mu, var, D3, D21, C_off, g4row, wk4m)\n" + a1)
src = src.replace(a2, a2 + "            _CAPWK(li, W_all)\n")
src = src.replace(a3, a3 + "            _CAPK(li, K1v, K2v, K3v, K4v, K21, K22)\n")
src = src.replace(a4, a4 + "            _CAPC(li, C)\n")
src = src.replace(a5, "        _CAPS(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list, c1_list, c2_list, y_list, rres, rfb, Zf_st, R1T_st, R2T_st, ka, kb)\n" + a5)
src = src.replace(a6, a6 + "            _CAPP(li, pk11, pk21)\n")

CUR = {}
f64 = lambda x: None if x is None else np.asarray(x, dtype=np.float64).copy()
f32 = lambda x: None if x is None else np.asarray(x, dtype=np.float32).copy()
NDS = [0]   # number of _dslices calls so far (called once per layer from li = 1)

def _flush(li):
    if not CUR: return
    l = li + 1
    d = {k: v for k, v in CUR.items() if v is not None}
    np.savez(os.path.join(OUT, f"r7_cap_L{l}.npz"), **d)
    print(f"  wrote L{l}: {sorted(d.keys())}", flush=True)
    CUR.clear()

def _CAPW(li, mu, var, D3, D21, C_off, g4row, wk4m):
    CUR.update(li=np.int64(li), mu=f64(mu), var=f64(var), D3=f64(D3), D21=f32(D21), C_off=f32(C_off), g4row=f64(g4row), wk4m=f32(wk4m))
def _CAPWK(li, W_all): CUR["W_all"] = f64(W_all)
def _CAPK(li, K1v, K2v, K3v, K4v, K21, K22): CUR.update(K1=f64(K1v), K2=f64(K2v), K3=f64(K3v), K4=f64(K4v), K21=f32(K21), K22=f32(K22))
def _CAPP(li, pk11, pk21): CUR.update(pk11=f32(pk11), pk21=f32(pk21))
def _CAPC(li, C):
    CUR["C_post"] = f32(C)
    _flush(li)          # the last capture of the layer body
def _CAPS(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list, c1_list, c2_list, y_list, rres, rfb, Zf_st, R1T_st, R2T_st, ka, kb):
    NDS[0] += 1
    li = NDS[0]         # _dslices is called at li = 1, 2, ... in order
    if li + 1 not in SRC_LAYERS: return
    assert ka == 0 and kb == 0, (ka, kb)
    CUR.update(A_st=f32(A_st), P_st=f32(P_st), Z_st=f32(Z_st), L_st=f32(L_st),
               w2b=np.stack([f64(x) for x in w2b_list]), s_b=np.stack([f64(x) for x in s_list]), e_b=np.stack([f64(x) for x in e_list]),
               c1_b=np.stack([f64(x) for x in c1_list]), c2_b=np.stack([f64(x) for x in c2_list]), y_b=np.stack([f64(x) for x in y_list]),
               rres=np.int64(rres), rfb=np.int64(rfb), Zf_st=f32(Zf_st), R1T_st=f32(R1T_st), R2T_st=f32(R2T_st))

mod = types.ModuleType("est_r7"); mod.__dict__.update({"_CAPW": _CAPW, "_CAPWK": _CAPWK, "_CAPK": _CAPK, "_CAPC": _CAPC, "_CAPS": _CAPS, "_CAPP": _CAPP})
exec(compile(src, SRC, "exec"), mod.__dict__)
truth = np.load(os.path.join(PRE, "truth.npz")); Wall = np.load(glob.glob(os.path.join(PRE, "weights_seed*.npz"))[0])["W"]
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
mlp = MLP(width=Wall.shape[1], depth=Wall.shape[0], weights=[fnp.array(Wall[l]) for l in range(Wall.shape[0])], seed=int(truth["mlp_seed"]), name=str(truth["mlp_name"]))
est = mod.Estimator(); est.setup(_Ctx()); t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    pred = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
ALM = np.asarray(truth["all_layer_means"], np.float64)
# the last layer flushes at _CAPP (no C is assembled there)
if CUR: np.savez(os.path.join(OUT, "r7_cap_L16.npz"), **{k: v for k, v in CUR.items() if v is not None}); print("  wrote L16 (partial)", flush=True)
per = [float(np.mean((pred[k] - ALM[k]) ** 2)) for k in range(16)]
print(f"chain run {time.time()-t0:.0f}s  util {ctx.flops_used/2**41:.4f}  final MSE {per[-1]:.4e}  (L8 {per[7]:.3e}, L12 {per[11]:.3e})", flush=True)
json.dump({"per_layer": per, "util": ctx.flops_used / 2**41, "env": {k: os.environ.get(k) for k in ("V31_ANGULAR", "V21_NO_CONFINE", "V19_FULL_LAST", "V26_STRASSEN")}},
          open("../results/summaries/r7_capture_run.json", "w"), indent=1)
