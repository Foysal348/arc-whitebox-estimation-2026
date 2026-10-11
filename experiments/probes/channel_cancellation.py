"""Direct measurement of channel-error cancellation in the mean closure (11 Oct 2026; paper §7.5 [OPEN]).

Question: do the chain's errors in its four mean-closure inputs (μ, var, d3 = κ₃ diagonal, k4 = κ₄ diagonal) partly
cancel? If so, correcting one channel alone uncovers the errors it was offsetting.

Method (one step, per layer, MLP #0, teacher 6·10⁷ samples in data/kaggle_out/kaggle_out_k4pub/k4teach):
- Closure M(μ, v, κ₃, κ₄) = μΦ(a) + σφ(a) − κ₃ aφ(a)/(6σ²) + κ₄ (a²−1)φ(a)/(24σ³), with a = μ/σ (Edgeworth to
  order 4). It is checked first: on the teacher's own inputs it must reproduce the teacher's ReLU mean.
- Chain point: the frozen V56 chain (dump variant) gives μ, var, d3 and its actual post-ReLU mean. M is linear in κ₃
  and κ₄. The κ₄ the chain effectively used is k4_eff = (post_mean − M(μ, v, d3, 0)) / ((a²−1)φ/(24σ³)), on neurons with
  |a² − 1| > 0.1 (elsewhere κ₄ has no effect on the mean); the reconstruction is checked against the chain's post mean.
- Channel errors: e_ch = M(chain) − M(chain with channel ch set to the teacher's), for ch in {μ, var, d3, k4}; the total
  error is e = M(chain) − M(teacher). Reported per layer:
  - ‖e_ch‖ / ‖e‖;
  - the cosine matrix of the e_ch;
  - the cancellation ratio ‖Σ e_ch‖² / Σ ‖e_ch‖² (1 means no cancellation, < 1 means cancellation);
  - the fix-one ratios ‖e − e_ch‖ / ‖e‖ (> 1: correcting channel ch alone makes the step error larger);
  - the fix-pair ratios for (d3, k4) and (var, k4).
- Teacher noise reference: ‖M(teacher inputs) − teacher ReLU mean‖, the closure-plus-noise floor.

Usage: python experiments/probes/channel_cancellation.py
Laptop-light: one chain pass at width 1024 plus vector arithmetic.
"""
import os, json, types
import numpy as np
from math import sqrt, pi
from scipy.special import ndtr

TD = "data/kaggle_out/kaggle_out_k4pub/k4teach"
pdf = lambda x: np.exp(-0.5 * x * x) / sqrt(2 * pi)


def closure(mu, v, k3, k4):
    s = np.sqrt(v); a = mu / s; ph = pdf(a)
    return mu * ndtr(a) + s * ph - k3 * a * ph / (6 * v) + k4 * (a * a - 1) * ph / (24 * v * s)


def c4(mu, v):
    s = np.sqrt(v); a = mu / s
    return (a * a - 1) * pdf(a) / (24 * v * s)


T = {}
for l in range(1, 17):
    z = np.load(f"{TD}/layer{l}_seed0.npz")
    T[l] = dict(mu=z["mean"].astype(np.float64), v=np.diag(z["C"]).astype(np.float64), d3=np.diag(z["Y21"]).astype(np.float64),
                k4=z["K4d"].astype(np.float64), relu=z["relu_mean"].astype(np.float64))

# ---- chain pass ----
os.environ["STUDENT_DUMP"] = "1"; os.environ["STUDENT_ROWS"] = "8"; os.environ["STUDENT_FULL"] = "0"; os.environ["STUDENT_SEED"] = "0"
import flopscope as flops, flopscope.numpy as fnp
from whestbench import MLP
EST = "experiments/student/v56_dump.py"
mod = types.ModuleType("est_cc"); mod.__file__ = os.path.abspath(EST)
exec(compile(open(EST, encoding="utf-8").read(), os.path.abspath(EST), "exec"), mod.__dict__)
W = np.load(f"{TD}/weights_seed0.npz")["W"]
n, L = W.shape[1], W.shape[0]


class _Ctx: seed = 0; width = n; depth = L; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


est = mod.Estimator(); est.setup(_Ctx()); mod.DUMP_STORE.clear()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
    pred = est.predict(MLP(width=n, depth=L, weights=[fnp.array(w) for w in W], seed=0, name="mlp0"), 2**41)
D = {l: {k: np.asarray(v, np.float64) for k, v in d.items() if np.ndim(v) == 1 and np.size(v) == n} for l, d in mod.DUMP_STORE.items()}
print("chain layers dumped:", sorted(D), flush=True)

nrm = np.linalg.norm
cos = lambda x, y: float(x @ y / max(nrm(x) * nrm(y), 1e-300))
CH = ("mu", "v", "d3", "k4")
out = {"closure_check": {}, "layers": {}}
for l in range(2, 17):
    t = T[l]
    # closure check on the teacher's own inputs
    Mt = closure(t["mu"], t["v"], t["d3"], t["k4"])
    chk = dict(rel_err_vs_teacher_relu=float(nrm(Mt - t["relu"]) / nrm(t["relu"])),
               rel_err_gauss_only=float(nrm(closure(t["mu"], t["v"], 0, 0) - t["relu"]) / nrm(t["relu"])))
    out["closure_check"][l] = chk
    if l not in D: continue
    d = D[l]
    c = dict(mu=d["mu_pre"], v=d["var"], d3=d["d3"])
    cc4 = c4(c["mu"], c["v"]); ok = np.abs(cc4) * 24 * c["v"] * np.sqrt(c["v"]) / np.maximum(pdf(c["mu"] / np.sqrt(c["v"])), 1e-300) > 0.1
    base = closure(c["mu"], c["v"], c["d3"], 0.0)
    k4 = np.where(ok, (d["post_mean"] - base) / np.where(ok, cc4, 1.0), t["k4"])   # neurons with |a^2-1| <= 0.1: k4 inert, set to teacher
    c["k4"] = k4
    Mc = closure(c["mu"], c["v"], c["d3"], c["k4"])
    recon = float(nrm(Mc - d["post_mean"]) / nrm(d["post_mean"]))
    Mtt = closure(t["mu"], t["v"], t["d3"], t["k4"])
    e = Mc - Mtt
    E = {}
    for ch in CH:
        x = dict(c); x[ch] = t[ch]
        E[ch] = Mc - closure(x["mu"], x["v"], x["d3"], x["k4"])
    S = sum(E.values())
    pair = lambda a, b: float(nrm(e - E[a] - E[b]) / nrm(e))
    row = dict(recon_rel=recon, k4_inert_frac=float((~ok).mean()), err_rel_to_relu=float(nrm(e) / nrm(t["relu"])),
               err_vs_true_relu_rel=float(nrm(Mc - t["relu"]) / nrm(t["relu"])),
               share={ch: float(nrm(E[ch]) / nrm(e)) for ch in CH},
               cos={f"{a}-{b}": cos(E[a], E[b]) for i, a in enumerate(CH) for b in CH[i + 1:]},
               cancel_ratio=float(nrm(S) ** 2 / sum(nrm(E[ch]) ** 2 for ch in CH)),
               interaction_rel=float(nrm(e - S) / nrm(e)),
               fix_one={ch: float(nrm(e - E[ch]) / nrm(e)) for ch in CH},
               fix_pair={"d3+k4": pair("d3", "k4"), "v+k4": pair("v", "k4"), "v+d3": pair("v", "d3"), "mu+v": pair("mu", "v")})
    # fresh channels only (var, d3, k4: the quantities the chain computes at this step; mu is the carried mean error)
    x = dict(t); x["mu"] = c["mu"]; ef = Mc - closure(x["mu"], x["v"], x["d3"], x["k4"])     # error with the true v, d3, k4 at the chain's mu
    F = {}
    for ch in ("v", "d3", "k4"):
        y = dict(c); y[ch] = t[ch]; F[ch] = Mc - closure(y["mu"], y["v"], y["d3"], y["k4"])
    SF = sum(F.values())
    row["fresh"] = dict(err_rel_total=float(nrm(ef) / nrm(e)), cancel_ratio=float(nrm(SF) ** 2 / sum(nrm(F[ch]) ** 2 for ch in F)),
                        sum_of_norms_over_norm=float(sum(nrm(F[ch]) for ch in F) / nrm(ef)),
                        fix_one={ch: float(nrm(ef - F[ch]) / nrm(ef)) for ch in F},
                        fix_pair={f"{p_}+{q_}": float(nrm(ef - F[p_] - F[q_]) / nrm(ef)) for p_, q_ in (("v", "d3"), ("v", "k4"), ("d3", "k4"))})
    out["layers"][l] = row
    print(f"L{l:2d} recon {recon:.1e} | |e|/|relu| {row['err_rel_to_relu']:.2e} | share mu {row['share']['mu']:.2f} v {row['share']['v']:.2f} "
          f"d3 {row['share']['d3']:.2f} k4 {row['share']['k4']:.2f} | cancel {row['cancel_ratio']:.2f} | "
          f"fix-one mu {row['fix_one']['mu']:.2f} v {row['fix_one']['v']:.2f} d3 {row['fix_one']['d3']:.2f} k4 {row['fix_one']['k4']:.2f} | "
          f"cos v-d3 {row['cos']['v-d3']:+.2f} v-k4 {row['cos']['v-k4']:+.2f} d3-k4 {row['cos']['d3-k4']:+.2f} mu-v {row['cos']['mu-v']:+.2f} | "
          f"teacher closure err {chk['rel_err_vs_teacher_relu']:.1e}", flush=True)
    fr = row["fresh"]
    print(f"     fresh (v, d3, k4): |e_f|/|e| {fr['err_rel_total']:.2f} | cancel {fr['cancel_ratio']:.2f} | sum|F|/|e_f| {fr['sum_of_norms_over_norm']:.2f} | "
          f"fix-one v {fr['fix_one']['v']:.2f} d3 {fr['fix_one']['d3']:.2f} k4 {fr['fix_one']['k4']:.2f} | "
          f"fix-pair v+d3 {fr['fix_pair']['v+d3']:.2f} v+k4 {fr['fix_pair']['v+k4']:.2f} d3+k4 {fr['fix_pair']['d3+k4']:.2f}", flush=True)
json.dump(out, open("results/summaries/channel_cancellation.json", "w"), indent=1)
print("wrote results/summaries/channel_cancellation.json")
