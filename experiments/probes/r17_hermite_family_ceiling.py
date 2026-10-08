"""Round 17 family ceiling (8 Oct): can a carrier built from second-order (Section B) curvature births, with a 6-layer memory,
reach pattern corr 0.7 on the deep kappa4 diagonal?

Monte Carlo through public MLP #0 (z_1 = x W[0], z_{l+1} = act_l(z_l) W[l]). act is either the true ReLU or its Hermite
expansion around the teacher's per-layer pre-activation mean / std (oracle mu, sigma):
    h = sigma * sum_{k<=K} c_k He_k((z - mu)/sigma),  c_0 = a Phi + phi, c_1 = Phi, c_k = phi(a) He_{k-2}(-a) / k!  (k >= 2).
K = 2 is the Section B model (L4 pass-through + Q1..Q4 births). Compounding K = 2 over all 16 layers diverges (smoke test,
level x1e12), so the family is tested over a finite horizon h before the target layer T:
  relu            true ReLU from the input (sanity vs the 6e7-sample teacher)
  gauss_h6_relu   z_{T-6} ~ N(mu, C) of the teacher (non-Gaussian history erased), true ReLU for the last 6 layers
                  (the memory-horizon reference, quoted 0.77)
  gauss_h6_K2/K3  same restart, Hermite-K activations for the last 6 layers (= best 6-age carrier with K-th order births)
  true_h{1,3,6}_K2  true ReLU up to z_{T-h}, then Hermite-2 for the last h layers (h = 1 reproduces the quoted QUAD 0.82-0.86)
Score: corr(k4 / v_t^2, K4d_t / v_t^2) on decisive neurons |mu/sigma| < 1.5 at T = 12, 15, 16; also the level ratio.
Gate for the Section B family: gauss_h6_K2 (or true_h6_K2) >= 0.70 at all of L12/15/16.

Usage: python r17_hermite_family_ceiling.py [N] [device]
Env: R17_DATA = npz with W (16,n,n), mean (17,n), C (17,n,n), K4d (17,n) [index = layer]; R17_OUT = output prefix.
"""
import os, sys, json, time
import numpy as np
import torch
from math import erf, sqrt, pi, factorial

DATA = os.environ.get("R17_DATA", "r17_data.npz")
OUT = os.environ.get("R17_OUT", "r17_hermite_family_ceiling")
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 20000
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 and (sys.argv[2] == "cpu" or torch.cuda.is_available()) else "cpu")
B = 25000 if dev.type == "cuda" else 5000
D = np.load(DATA)
W = D["W"].astype(np.float32); MU = D["mean"].astype(np.float64); CC = D["C"].astype(np.float64); K4T = D["K4d"].astype(np.float64)
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
chol = {}


def act(z, l, K):
    """K = 0: true ReLU; K = int: Hermite orders 0..K; K = tuple: exactly those Hermite orders."""
    if K == 0:
        return torch.relu(z)
    orders = range(K + 1) if isinstance(K, int) else K
    x = (z - mut[l]) / sdt[l]
    x2 = x * x
    hs = [torch.ones_like(x), x, x2 - 1, x2 * x - 3 * x, x2 * x2 - 6 * x2 + 3]
    return sdt[l] * sum(cf[l][k] * hs[k] for k in orders)


def run(target, mode, gen):
    """Returns power sums (4, n) of z_target over N samples."""
    S = torch.zeros(4, n, device=dev, dtype=torch.float64); done = 0
    kind, h, K = mode
    start = target - h
    while done < N:
        m = min(B, N - done)
        if kind == "gauss":
            if start not in chol:
                ev, V = np.linalg.eigh(CC[start])                     # PSD square root (C is rank-deficient: dead neurons upstream)
                chol[start] = T(V * np.sqrt(np.clip(ev, 0, None)))
            z = mut[start] + torch.randn(m, n, device=dev, generator=gen) @ chol[start].T
            l0 = start
        else:
            z = torch.randn(m, n, device=dev, generator=gen) @ Wt[0]; l0 = 1
        for l in range(l0, target):
            Kl = K if (kind == "gauss" or l >= start) else 0
            z = act(z, l, Kl) @ Wt[l]
        zz = z.double()
        for p in range(4):
            S[p] += (zz ** (p + 1)).sum(0)
        done += m
    return S.cpu().numpy()


MODES_A = {"relu": ("true", 0, 0), "gauss_h6_relu": ("gauss", 6, 0), "gauss_h6_K2": ("gauss", 6, 2), "gauss_h6_K3": ("gauss", 6, 3),
           "true_h1_K2": ("true", 1, 2), "true_h3_K2": ("true", 3, 2), "true_h6_K2": ("true", 6, 2), "true_h6_K3": ("true", 6, 3)}
# set B (8 Oct, after set A): which Hermite orders and which horizon does the passing K = 3 family need?
MODES_B = {"gauss_h6_K013": ("gauss", 6, (0, 1, 3)), "gauss_h6_K4": ("gauss", 6, 4), "gauss_h4_K3": ("gauss", 4, 3),
           "gauss_h3_K3": ("gauss", 3, 3), "gauss_h4_relu": ("gauss", 4, 0), "gauss_h3_relu": ("gauss", 3, 0),
           "gauss_h2_K3": ("gauss", 2, 3), "gauss_h2_relu": ("gauss", 2, 0)}
MODES = MODES_B if os.environ.get("R17_SET", "A") == "B" else MODES_A
TARGETS = [int(x) for x in os.environ.get("R17_TARGETS", "12,15,16").split(",")]
res = {"N": N, "device": str(dev)}; save = {}
t0 = time.time()
for name, mode in MODES.items():
    res[name] = {}
    for tl in TARGETS:
        gen = torch.Generator(device=dev); gen.manual_seed(1717 + tl)
        S = run(tl, mode, gen)
        m1, m2, m3, m4 = (S[p] / N for p in range(4))
        v = m2 - m1 ** 2
        k4 = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4 - 3 * v ** 2
        vt = SD[tl] ** 2; dec = np.abs(MU[tl] / SD[tl]) < 1.5
        ok = np.isfinite(k4).all()
        res[name][tl] = dict(pattern_corr=float(np.corrcoef((k4 / vt ** 2)[dec], (K4T[tl] / vt ** 2)[dec])[0, 1]) if ok else None,
                             level_ratio=float(k4[dec].mean() / K4T[tl][dec].mean()) if ok else None,
                             var_ratio=float(v[dec].mean() / vt[dec].mean()) if ok else None)
        save[f"{name}_L{tl}_k4"] = k4; save[f"{name}_L{tl}_var"] = v
    print(f"{name:14s} " + " ".join(f"L{tl} corr {r['pattern_corr']} level {r['level_ratio']}" for tl, r in res[name].items())
          + f"  ({time.time() - t0:.0f}s)", flush=True)
np.savez(OUT + ".npz", **save)
json.dump(res, open(OUT + ".json", "w"), indent=1)
