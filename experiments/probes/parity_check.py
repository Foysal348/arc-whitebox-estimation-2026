"""Numerical verification of the parity theorem (paper §2.2) and of the sqrt(6/(r+8)) floor (11 Oct 2026).

Part A: the conditional-expectation floor (Monte Carlo; exact algebra checked by sampling).
- Setup: g, h ~ N(0, I_r) independent; T a symmetric 3-tensor; y = T(g, g, h).
- Claim 1: E[y²] = |tau|² + 2|T|² with tau_c = sum_a T_aac, and E[y | h] = tau · h,
  so the residual variance is 2|T|².
- Claim 2: |tau|² <= (r+2)|T|²/3, with equality at T = sym(I ⊗ v).
  Hence relative RMS error >= sqrt(6/(r+8)).
- Checked for random T, for the extremal T, and for random r.

Part B: parity on the real chain (frozen V56 transport, deterministic).
- Run the frozen V56 chain (dump variant) on a network and on the same network with the sign of one weight matrix W[l]
  flipped.
- Claim: at the pre-activation of that same layer (W[l] already applied, before its ReLU), everything linear in κ₃
  (the D21 slice, d3) and the mean flip sign exactly; the even statistics (covariance C, variances) are unchanged.
  Every source alive there was born before W[l], so this is the 'old source' statement of §2.2.
- Layers before l are identical. So the average of (W, −W) of any κ₃-linear quantity, conditioned on the even
  statistics (which are equal), is exactly 0.
- Width 128 (laptop-light).

Usage: python experiments/probes/parity_check.py
"""
import os, sys, json, types
import numpy as np

out = {}
rng = np.random.default_rng(0)


def sym3(T):
    return (T + T.transpose(0, 2, 1) + T.transpose(1, 0, 2) + T.transpose(1, 2, 0) + T.transpose(2, 0, 1) + T.transpose(2, 1, 0)) / 6.0


# ---- Part A ----
A = []
for r, kind in ((8, "random"), (16, "random"), (16, "extremal"), (32, "extremal"), (24, "random")):
    if kind == "random":
        T = sym3(rng.standard_normal((r, r, r)))
    else:
        v = rng.standard_normal(r); I = np.eye(r)
        T = sym3(np.einsum("ab,c->abc", I, v))
    tau = np.einsum("aac->c", T); nT2 = float((T * T).sum()); tau2 = float(tau @ tau)
    N = 400000
    g = rng.standard_normal((N, r)); h = rng.standard_normal((N, r))
    y = np.einsum("abc,na,nb,nc->n", T, g, g, h, optimize=True)
    Ey2_mc = float((y * y).mean()); cond = h @ tau
    resid_mc = float(((y - cond) ** 2).mean())
    rel_err = np.sqrt(2 * nT2 / (tau2 + 2 * nT2)); floor = np.sqrt(6.0 / (r + 8))
    row = dict(r=r, kind=kind, Ey2_mc=Ey2_mc, Ey2_formula=tau2 + 2 * nT2, resid_mc=resid_mc, resid_formula=2 * nT2,
               tau2_over_bound=tau2 / ((r + 2) * nT2 / 3), rel_err=float(rel_err), floor=float(floor))
    A.append(row)
    print(f"[A] r={r:2d} {kind:8s}: E[y^2] MC {Ey2_mc:.4g} vs formula {tau2 + 2 * nT2:.4g} | resid MC {resid_mc:.4g} vs 2|T|^2 {2 * nT2:.4g} | "
          f"|tau|^2/bound {tau2 / ((r + 2) * nT2 / 3):.4f} | rel err {rel_err:.4f} >= floor {floor:.4f}", flush=True)
out["floor_checks"] = A
# exact sharpness of |tau|^2 <= (r+2)|T|^2/3 over ALL symmetric T: top eigenvalue of P M^T M P (P = symmetriser, M = trace map)
sharp = []
for r in (2, 3, 4, 6, 8):
    E = np.eye(r ** 3).reshape(r ** 3, r, r, r)
    P = np.stack([sym3(e) for e in E]).reshape(r ** 3, r ** 3)            # orthogonal projector onto Sym3
    M = np.einsum("kaac->kc", E.reshape(r ** 3, r, r, r)).T                # (r, r^3): vec(T) -> tau
    top = float(np.linalg.eigvalsh(P @ M.T @ M @ P)[-1])
    sharp.append(dict(r=r, max_tau2_over_T2=top, bound=(r + 2) / 3.0))
    print(f"[A] r={r}: max over Sym3 of |tau|^2/|T|^2 = {top:.6f}  vs (r+2)/3 = {(r + 2) / 3:.6f}", flush=True)
out["bound_sharpness_exact"] = sharp
out["floor_values"] = {r: float(np.sqrt(6.0 / (r + 8))) for r in (100, 224, 384, 1024)}
print("[A] floor values:", {k: round(v, 4) for k, v in out["floor_values"].items()})

# ---- Part B ----
os.environ["STUDENT_DUMP"] = "1"; os.environ["STUDENT_ROWS"] = "128"; os.environ["STUDENT_FULL"] = "1"; os.environ["STUDENT_SEED"] = "7"
import flopscope as flops, flopscope.numpy as fnp
from whestbench import MLP
EST = "experiments/student/v56_dump.py"
mod = types.ModuleType("est_par"); mod.__file__ = os.path.abspath(EST)
exec(compile(open(EST, encoding="utf-8").read(), os.path.abspath(EST), "exec"), mod.__dict__)


class _Ctx: seed = 0; width = 128; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"


n, L = 128, 16
rng2 = np.random.default_rng(7)
Ws = [(rng2.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]


def run(ws):
    est = mod.Estimator(); est.setup(_Ctx())
    mod.DUMP_STORE.clear()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True):
        est.predict(MLP(width=n, depth=L, weights=[fnp.array(w) for w in ws], seed=0, name="p"), 2**41)
    return {l: {k: np.array(v) for k, v in d.items()} for l, d in mod.DUMP_STORE.items()}


base = run(Ws)
B = []
for flip_li in (4, 9):                     # dump layer li+1 = pre-activation of W[li]
    Wf = [w.copy() for w in Ws]; Wf[flip_li] = -Wf[flip_li]
    fl = run(Wf); lay = flip_li + 1
    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    prev_same = max(rel(fl[l]["mu_pre"], base[l]["mu_pre"]) for l in range(1, lay))
    r = dict(flipped_W_index=flip_li, checked_layer=lay,
             earlier_layers_mu_rel=prev_same,
             C_rel=rel(fl[lay]["C"], base[lay]["C"]), var_rel=rel(fl[lay]["var"], base[lay]["var"]),
             mu_pre_flip_rel=rel(fl[lay]["mu_pre"], -base[lay]["mu_pre"]),
             d3_flip_rel=rel(fl[lay]["d3"], -base[lay]["d3"]), D21_flip_rel=rel(fl[lay]["D21"], -base[lay]["D21"]),
             d3_pair_mean_rel=float(np.linalg.norm(0.5 * (fl[lay]["d3"] + base[lay]["d3"])) / np.linalg.norm(base[lay]["d3"])),
             D21_pair_mean_rel=float(np.linalg.norm(0.5 * (fl[lay]["D21"] + base[lay]["D21"])) / np.linalg.norm(base[lay]["D21"])))
    B.append(r)
    print(f"[B] flip W[{flip_li}] -> layer {lay}: earlier mu identical {prev_same:.1e} | C unchanged {r['C_rel']:.1e}, var {r['var_rel']:.1e} | "
          f"mu flips {r['mu_pre_flip_rel']:.1e} | d3 flips {r['d3_flip_rel']:.1e}, D21 flips {r['D21_flip_rel']:.1e} | "
          f"|(W,-W) mean| / |.|: d3 {r['d3_pair_mean_rel']:.1e}, D21 {r['D21_pair_mean_rel']:.1e}", flush=True)
out["chain_parity"] = B
json.dump(out, open("results/summaries/parity_check.json", "w"), indent=1)
print("wrote results/summaries/parity_check.json")
