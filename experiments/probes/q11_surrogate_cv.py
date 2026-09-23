"""Queue Q11 (+ Gemini R7 Q3: QUADRATIC surrogate with SURR=quad) (Astra R7 §6, GPT Terra R7 Q1.4, Grok R7 1c): can a COUPLED linear surrogate make Monte-Carlo estimates of
the per-neuron corrections cheap?

Surrogate: G_l(x) = m_l + J_l x with J_l = W_l^T diag(Phi_(l-1)) J_(l-1) (the tangent / mean-field linearisation), fed the
SAME input x as the network, so its samples are coupled to the true pre-activations z_l(x). Its moments are known exactly
(Gaussian, covariance J J^T), so it is a legitimate control variate. For the per-neuron statistics
    q=2: (z - mu)^2  (variance),  q=3: (z - mu)^3,  q=4: (z - mu)^4 - 3 var^2
the optimal CV reduces the estimator variance by the factor 1 / (1 - rho^2), rho = corr(stat(z), stat(G)).
Astra: a 100x reduction needs rho >= 0.9975 / 0.9983 / 0.9987. This measures rho on public MLP #0 with 20k samples.
Phi uses the teacher's exact pre-activation means/variances (the best possible linearisation).

Usage:  python experiments/probes/q11_surrogate_cv.py
"""
import numpy as np, os, json, time
from math import erf
DIR = "data/kaggle_out/kaggle_out_k4pub/k4teach"
W = np.load(os.path.join(DIR, "weights_seed0.npz"))["W"].astype(np.float64)       # z_l = h_(l-1) @ W[l-1]
L, n = W.shape[0], W.shape[1]
Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / 2 ** 0.5)))
mu, var = [], []
for l in range(1, L + 1):
    T = np.load(os.path.join(DIR, f"layer{l}_seed0.npz")); mu.append(np.asarray(T["mean"], np.float64)); var.append(np.diag(np.asarray(T["C"], np.float64)).copy())
rng = np.random.default_rng(11); N = int(os.environ.get("N", "20000")); B = 2000
SURR = os.environ.get("SURR", "lin")
phi_ = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
LAY = [2, 4, 8, 12, 16]
acc = {l: {q: [] for q in (2, 3, 4)} for l in LAY}
t0 = time.time()
for b0 in range(0, N, B):
    x = rng.standard_normal((B, n))
    h = x; g = x.copy()                                          # g: the surrogate's "input" to layer l, in row form
    Jx = x                                                        # J_0 x = x
    for l in range(1, L + 1):
        z = h @ W[l - 1]                                          # true pre-activation
        Gz = (mu[l - 1] + (Jx @ W[l - 1]) if l > 1 else z.copy())  # surrogate (exact at layer 1)
        if l in LAY:
            zc, gc = z - mu[l - 1], Gz - mu[l - 1]
            for q in (2, 3, 4):
                acc[l][q].append((zc ** q, gc ** q))
        s_ = np.sqrt(var[l - 1]); ph = Phi(mu[l - 1] / s_)
        h = np.maximum(z, 0.0)
        if SURR == "quad":
            # Gemini's surrogate: h_CV = Phi o dz + (1/2) (phi/s) o (dz^2 - var), dz = z_CV - mu, propagated with the SAME x
            dz = Gz - mu[l - 1]; d2 = phi_(mu[l - 1] / s_) / s_
            Jx = dz * ph[None, :] + 0.5 * d2[None, :] * (dz * dz - var[l - 1][None, :])
            Gz_next_input = Jx
        else:
            Jx = (Jx @ W[l - 1]) * ph[None, :] if l > 1 else z * ph[None, :]   # d h ~ Phi d z  (row form: J_l x = (J_(l-1) x) W D)
    print(f"  samples {b0 + B}  ({time.time()-t0:.0f}s)", flush=True)
out = {}
for l in LAY:
    row = {}
    for q in (2, 3, 4):
        A = np.concatenate([a for a, _ in acc[l][q]]); G = np.concatenate([g_ for _, g_ in acc[l][q]])
        Ac = A - A.mean(0); Gc = G - G.mean(0)
        rho = np.sum(Ac * Gc, 0) / np.sqrt(np.sum(Ac * Ac, 0) * np.sum(Gc * Gc, 0))
        red = 1.0 / np.maximum(1 - rho ** 2, 1e-12)
        row[q] = {"rho_median": float(np.median(rho)), "rho_p90": float(np.percentile(rho, 90)), "reduction_median": float(np.median(red))}
    out[l] = row
    print(f"layer {l:2d}: " + " | ".join(f"q={q}: corr median {row[q]['rho_median']:.3f} (p90 {row[q]['rho_p90']:.3f}) -> variance reduction x{row[q]['reduction_median']:.1f}" for q in (2, 3, 4)), flush=True)
json.dump(out, open(f"results/summaries/q11_surrogate_cv_{SURR}.json", "w"), indent=1)
