"""Architecture analysis (9 Oct): which Hermite order of the ReLU response does each CHANNEL need, one step at a time?

Exact samples of z_l through public MLP #0 (Gaussian input). At layer l the ReLU is replaced by its Hermite expansion of order K
(K = 1, 2, 3, 4) around the TEACHER's per-layer mu / sigma, or kept exact, on the SAME samples (common random numbers), and
z_{l+1} = act(z_l) W[l]. For every variant the next-layer statistics are accumulated in float64:
  mean, covariance C (off-diagonal and variance), the kappa3 (2,1) slice k3(i,i,j) incl. its diagonal, the kappa4 diagonal.
Reported, per source layer l, relative to the EXACT-ReLU step on the same samples:
  rel. Frobenius error of C_off, of var, of the D21 off-diagonal slice, of the k3 diagonal, of the mean;
  kappa4 decisive-neuron pattern corr (k4 / v^2, |mu/sigma| < 1.5) vs the exact step.
Interpretation: a chain whose births stop at order K makes at least this per-step error in each channel. Compare with the chain's
own fresh per-step errors (paper section 4.3-4.5: C_off fresh ~4.2e-5 RMS/entry, D21 4.6-6.6% accumulated, all-distinct k3 9-16%).
Usage: python r20_onestep_channels.py [N] [device]   env R17_DATA (packed teacher npz), R20_LAYERS (default 4,8,12,15), R20_OUT
"""
import os, sys, json, time
import numpy as np
import torch
from math import erf, sqrt, pi, factorial

DATA = os.environ.get("R17_DATA", "r17_data.npz")
OUT = os.environ.get("R20_OUT", "r20_onestep_channels")
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 20000
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 and (sys.argv[2] == "cpu" or torch.cuda.is_available()) else "cpu")
B = 20000 if dev.type == "cuda" else 4000
LAYERS = [int(x) for x in os.environ.get("R20_LAYERS", "4,8,12,15").split(",")]
D = np.load(DATA)
W = D["W"].astype(np.float32); MU = D["mean"].astype(np.float64); CC = D["C"].astype(np.float64)
L, n = W.shape[0], W.shape[1]
SD = np.sqrt(np.stack([np.diag(CC[l]) if l > 0 else np.ones(n) for l in range(L + 1)]))
Phi = lambda x: 0.5 * (1 + np.vectorize(erf)(x / sqrt(2))); pdf = lambda x: np.exp(-x * x / 2) / sqrt(2 * pi)


def He(k, x):
    h0, h1 = np.ones_like(x), x
    if k == 0: return h0
    for j in range(1, k):
        h0, h1 = h1, x * h1 - j * h0
    return h1


T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), device=dev, dtype=dt)
Wt = [T(W[l]) for l in range(L)]
mut = [T(MU[l]) for l in range(L + 1)]; sdt = [T(SD[l]) for l in range(L + 1)]
cf = {}
for l in range(1, L + 1):
    a = MU[l] / SD[l]
    cf[l] = T(np.array([a * Phi(a) + pdf(a), Phi(a)] + [pdf(a) * He(k - 2, -a) / factorial(k) for k in (2, 3, 4)]))


def act(z, l, K):
    if K == 0:
        return torch.relu(z)
    x = (z - mut[l]) / sdt[l]; x2 = x * x
    hs = [torch.ones_like(x), x, x2 - 1, x2 * x - 3 * x, x2 * x2 - 6 * x2 + 3]
    return sdt[l] * sum(cf[l][k] * hs[k] for k in range(K + 1))


VARIANTS = {"exact": 0, "K1": 1, "K2": 2, "K3": 3, "K4": 4}
res = {"N": N, "device": str(dev)}; t0 = time.time()
for l in LAYERS:
    acc = {v: dict(s1=torch.zeros(n, device=dev, dtype=torch.float64), s11=torch.zeros(n, n, device=dev, dtype=torch.float64),
                   s21=torch.zeros(n, n, device=dev, dtype=torch.float64), s3=torch.zeros(n, device=dev, dtype=torch.float64),
                   s4=torch.zeros(n, device=dev, dtype=torch.float64)) for v in VARIANTS}
    gen = torch.Generator(device=dev); gen.manual_seed(2020 + l)
    done = 0
    while done < N:
        m = min(B, N - done)
        h = torch.randn(m, n, device=dev, generator=gen)
        z = h @ Wt[0]
        for k in range(1, l):
            z = torch.relu(z) @ Wt[k]
        for v, K in VARIANTS.items():
            y = act(z, l, K) @ Wt[l]
            y2 = y * y
            a = acc[v]
            a["s1"] += y.sum(0, dtype=torch.float64)
            a["s11"] += (y.T @ y).double()
            a["s21"] += (y2.T @ y).double()
            yd = y.double(); a["s3"] += (yd ** 3).sum(0); a["s4"] += (yd ** 4).sum(0)
        done += m
    st = {}
    for v in VARIANTS:
        a = {k: x.cpu().numpy() / N for k, x in acc[v].items()}
        m1 = a["s1"]; E11 = a["s11"]; E21 = a["s21"]; E2d = np.diag(E11)
        C = E11 - np.outer(m1, m1); var = np.diag(C).copy()
        mi = m1[:, None]; mj = m1[None, :]
        Y21 = E21 - 2 * mi * E11 - mj * E2d[:, None] + 2 * mi ** 2 * mj          # k3(i,i,j)
        m3 = a["s3"] - 3 * m1 * E2d + 2 * m1 ** 3
        m4 = a["s4"] - 4 * m1 * a["s3"] + 6 * m1 ** 2 * E2d - 3 * m1 ** 4
        k4 = m4 - 3 * var ** 2
        st[v] = dict(m1=m1, C=C, var=var, Y21=Y21, k3=np.diag(Y21).copy(), k4=k4)
    ex = st["exact"]; off = ~np.eye(n, dtype=bool)
    mu_n, v_n = MU[l + 1], SD[l + 1] ** 2; dec = np.abs(mu_n / np.sqrt(v_n)) < 1.5
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
    row = {}
    for v in VARIANTS:
        if v == "exact": continue
        s = st[v]
        row[v] = dict(C_off_rel=rel(s["C"][off], ex["C"][off]), C_off_rms=float(np.sqrt(np.mean((s["C"] - ex["C"])[off] ** 2))),
                      var_rel=rel(s["var"], ex["var"]), D21_off_rel=rel(s["Y21"][off], ex["Y21"][off]), k3diag_rel=rel(s["k3"], ex["k3"]),
                      mean_rel=rel(s["m1"], ex["m1"]),
                      k4_pattern_corr=float(np.corrcoef((s["k4"] / v_n ** 2)[dec], (ex["k4"] / v_n ** 2)[dec])[0, 1]),
                      k4_level=float(s["k4"][dec].mean() / ex["k4"][dec].mean()))
    # sampling-noise reference: exact step vs the 6e7 teacher at layer l+1 (independent noise)
    teacher_C = CC[l + 1]
    row["exact_vs_teacher"] = dict(C_off_rel=rel(ex["C"][off], teacher_C[off]), var_rel=rel(ex["var"], np.diag(teacher_C)),
                                   mean_rel=rel(ex["m1"], MU[l + 1]))
    res[l] = row
    print(f"L{l}->{l + 1} ({time.time() - t0:.0f}s): " + " | ".join(
        f"{v}: Coff {r['C_off_rel']:.2e} var {r['var_rel']:.2e} D21 {r['D21_off_rel']:.3f} k3d {r['k3diag_rel']:.3f} k4corr {r['k4_pattern_corr']:.2f}"
        for v, r in row.items() if v != "exact_vs_teacher") + f" | noise Coff {row['exact_vs_teacher']['C_off_rel']:.2e}", flush=True)
json.dump(res, open(OUT + ".json", "w"), indent=1)
