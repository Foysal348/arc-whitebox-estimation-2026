"""One-step closure floors and the copula oracle (GPT R5 gates 2/4), teacher-forced, on one He-init ReLU MLP.
At selected layers l: draw N samples of the true post-activation h_{l-1} (forward MC), form z = W_l^T h (column convention:
h_l = relu(h_{l-1} @ W_l) in the row convention used by the challenge), and compare the exact next-layer mean E[relu(z)] (MC) with
  (A) Gaussian closure of h: z ~ N(W^T m_h, W^T C_h W) -> closed form   [the K2 one-step]
  (B) Gaussian copula + EXACT marginals of the previous pre-activation z_{l-1}: latent Gaussian with corr(z_{l-1}), marginals =
      empirical quantiles of z_{l-1}; h = relu(F^-1(Phi(g))); z = W^T h; mean by MC   [GPT family 9 oracle]
  (C) Gaussian z_{l-1} (mean, cov of z_{l-1}) pushed through ReLU exactly per sample, then W: i.e. the K2 chain's own one-step
      from the PRE-activation (isolates the non-Gaussian dependence of z_{l-1} from that of h)
Reports per layer the RMS one-step error of A, B, C (vs the chain's final RMS error ~1.5e-4) and the MC standard error.
Env: WIDTH (1024), DEPTH (16), N (samples), LAYERS ("4,8,12,15"), SEED, DEVICE (cuda/cpu).  Output: onestep_results.json
"""
import os, json, time, numpy as np, torch
n = int(os.environ.get("WIDTH", "1024")); L = int(os.environ.get("DEPTH", "16")); N = int(float(os.environ.get("N", "2e6")))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "4,8,12,15").split(",")]; seed = int(os.environ.get("SEED", "0"))
dev = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")); dt = torch.float32
rng = np.random.default_rng(1000 + seed)
W = [torch.as_tensor((rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32), device=dev) for _ in range(L)]
g = torch.Generator(device=dev); g.manual_seed(seed)
CH = 1 << 15
def chol_robust(Cm):
    """Cholesky factor of a covariance that may be numerically indefinite (dead neurons at small width): eigen-clip fallback."""
    Cm = 0.5 * (Cm + Cm.T); jit = 1e-7 * float(torch.diag(Cm).mean())
    try:
        return torch.linalg.cholesky(Cm + jit * torch.eye(Cm.shape[0], device=Cm.device, dtype=Cm.dtype))
    except Exception:
        w, V = torch.linalg.eigh(Cm); w = w.clamp_min(jit)
        return V * torch.sqrt(w)[None, :]          # L with L L^T = V diag(w) V^T
def relu_mean_gauss(mu, var):
    s = torch.sqrt(var.clamp_min(1e-30)); a = mu / s
    return s * (torch.exp(-0.5 * a * a) / np.sqrt(2 * np.pi) + a * 0.5 * (1 + torch.erf(a / np.sqrt(2))))
res = {}
for l in LAYERS:          # l = 1-based layer whose pre-activation z_l = W_l^T h_{l-1} we study; h_0 = x
    t0 = time.time()
    # pass 1: accumulate statistics of h_{l-1} and z_{l-1}, and the truth E[relu(z_l)]
    m_h = torch.zeros(n, device=dev, dtype=torch.float64); C_h = torch.zeros(n, n, device=dev, dtype=torch.float64)
    m_z = torch.zeros(n, device=dev, dtype=torch.float64); C_z = torch.zeros(n, n, device=dev, dtype=torch.float64)
    truth = torch.zeros(n, device=dev, dtype=torch.float64); truth2 = torch.zeros(n, device=dev, dtype=torch.float64)
    zprev_samples = []   # a subset of z_{l-1} samples for empirical quantiles (copula marginals)
    done = 0
    while done < N:
        m = min(CH, N - done); h = torch.randn(m, n, device=dev, generator=g); zprev = None
        for k in range(l - 1):
            zprev = h @ W[k]; h = torch.relu(zprev)
        if zprev is None: zprev = h            # l = 1: the "previous pre-activation" is the input itself
        z = h @ W[l - 1]; y = torch.relu(z)
        m_h += h.sum(0, dtype=torch.float64); C_h += (h.T @ h).double(); m_z += zprev.sum(0, dtype=torch.float64); C_z += (zprev.T @ zprev).double()   # fp32 GEMM, fp64 accumulation
        truth += y.sum(0, dtype=torch.float64); truth2 += (y * y).sum(0, dtype=torch.float64)
        if len(zprev_samples) * CH < 262_144: zprev_samples.append(zprev.cpu())
        done += m
        if done % (CH * 512) == 0: print(f"  layer {l}: pass 1 {done/N:.0%} ({time.time()-t0:.0f}s)", flush=True)
    m_h /= N; C_h = C_h / N - torch.outer(m_h, m_h); m_z /= N; C_z = C_z / N - torch.outer(m_z, m_z)
    truth /= N; se = torch.sqrt((truth2 / N - truth ** 2) / N)
    # (A) Gaussian closure of h
    Wd = W[l - 1].double(); mu = Wd.T @ m_h; var = torch.einsum("ij,ik,jk->i", Wd, Wd, C_h) if False else ((Wd.T @ C_h) * Wd.T).sum(1)
    A = relu_mean_gauss(mu, var)
    # (C) Gaussian z_{l-1} pushed through ReLU exactly (sampled), then W  [= the K2 chain's own one-step from the pre-activation]
    Lz = chol_robust(C_z).float(); mzf = m_z.float()
    accC = torch.zeros(n, device=dev, dtype=torch.float64); accB = torch.zeros(n, device=dev, dtype=torch.float64)
    Zs_np = np.sort(torch.cat(zprev_samples, 0).numpy(), axis=0)                 # sort on the CPU (a GPU sort needs ~4x the data in index buffers)
    Zs_sorted = torch.as_tensor(Zs_np, device=dev); S = Zs_sorted.shape[0]; del Zs_np, zprev_samples
    Cz_corr = C_z / torch.sqrt(torch.outer(torch.diag(C_z), torch.diag(C_z)))
    Lc = chol_robust(Cz_corr).float()
    NB = N // 2; doneB = 0
    while doneB < NB:
        m = min(CH, NB - doneB); xi = torch.randn(m, n, device=dev, generator=g)
        zC = mzf + xi @ Lz.T; hC = torch.relu(zC); accC += torch.relu(hC @ W[l - 1]).sum(0, dtype=torch.float64)
        gl = xi @ Lc.T                                                            # latent Gaussian with corr(z_{l-1})
        u = 0.5 * (1 + torch.erf(gl / np.sqrt(2)))                                # uniforms
        idx = (u * (S - 1)).round().long().clamp(0, S - 1)                        # empirical quantile per neuron
        zB = torch.gather(Zs_sorted, 0, idx); hB = torch.relu(zB); accB += torch.relu(hB @ W[l - 1]).sum(0, dtype=torch.float64)
        doneB += m
    C = accC / NB; B = accB / NB
    r = lambda x: float(torch.sqrt(torch.mean((x - truth) ** 2)))
    res[l] = {"rms_A_gauss_h": r(A), "rms_B_copula_exact_marginals": r(B), "rms_C_gauss_zprev": r(C), "mc_se_rms": float(torch.sqrt(torch.mean(se ** 2))),
              "mean_level": float(truth.mean()), "secs": time.time() - t0}
    json.dump(res, open(os.environ.get("OUT", "onestep_results.json"), "w"), indent=1)
    print(f"layer {l:2d}: one-step RMS error  A (Gaussian h) {res[l]['rms_A_gauss_h']:.3e}   B (copula + exact marginals) {res[l]['rms_B_copula_exact_marginals']:.3e}   "
          f"C (Gaussian z_prev -> ReLU -> W) {res[l]['rms_C_gauss_zprev']:.3e}   [MC se {res[l]['mc_se_rms']:.1e}, mean level {res[l]['mean_level']:.3f}]  {time.time()-t0:.0f}s", flush=True)
json.dump(res, open(os.environ.get("OUT", "onestep_results.json"), "w"), indent=1)
