"""TRUE fourth-cumulant teacher per layer (the one axis still open: kappa4 accuracy at K3 cost).
For one He-init ReLU MLP, at each layer l, Monte-Carlo the pre-activation z_l = h_{l-1} W_l and accumulate the exact
cumulant slices that V29's memoryless kappa4 channel models:
   C[i,j]    = cov(z_i, z_j)
   S22[i,j]  = kappa4(z_i, z_i, z_j, z_j) = E[y_i^2 y_j^2] - E[y_i^2]E[y_j^2] - 2 E[y_i y_j]^2        (y = z - mean)
   S31[i,j]  = kappa4(z_i, z_i, z_i, z_j) = E[y_i^3 y_j] - 3 E[y_i^2] E[y_i y_j]
   S4[i]     = kappa4(z_i x4) = E[y_i^4] - 3 E[y_i^2]^2
Three GEMMs per chunk: (y^2)^T(y^2), (y^3)^T y, y^T y. Everything accumulated in float64.
V29's model: the kappa4 matrix core is G = diag(g) + lambda_l * C_off with ONE scalar per layer (F68). This script saves the
true slices so the law can be re-fitted and richer (still n^2-cheap) laws can be tested offline.
Env: WIDTH (1024), DEPTH (16), N (samples), SEED, OUT (dir), LAYERS ("all" or a list), CHUNK.
"""
import os, json, time, numpy as np, torch
n = int(os.environ.get("WIDTH", "1024")); L = int(os.environ.get("DEPTH", "16")); N = int(float(os.environ.get("N", "1e8")))
seed = int(os.environ.get("SEED", "0")); OUT = os.environ.get("OUT", "./k4teach"); CH = int(os.environ.get("CHUNK", str(1 << 14)))
LAYERS = list(range(1, L + 1)) if os.environ.get("LAYERS", "all") == "all" else [int(x) for x in os.environ["LAYERS"].split(",")]
dev = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1000 + seed)
Wnp = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
W = [torch.as_tensor(w, device=dev) for w in Wnp]
np.savez_compressed(os.path.join(OUT, f"weights_seed{seed}.npz"), W=np.stack(Wnp))
g = torch.Generator(device=dev); g.manual_seed(seed + 7)
for l in LAYERS:
    t0 = time.time()
    # pass 1: mean of z_l (and the post-ReLU mean of layer l, the estimator's target)
    m1 = torch.zeros(n, device=dev, dtype=torch.float64); mrelu = torch.zeros(n, device=dev, dtype=torch.float64)
    done = 0
    while done < N:
        m = min(CH, N - done); h = torch.randn(m, n, device=dev, generator=g)
        for k in range(l - 1): h = torch.relu(h @ W[k])
        z = h @ W[l - 1]
        m1 += z.sum(0, dtype=torch.float64); mrelu += torch.relu(z).sum(0, dtype=torch.float64); done += m
    m1 /= N; mrelu /= N; mu = m1.float()
    # pass 2: central moments (same stream, regenerator reset for reproducibility)
    g2 = torch.Generator(device=dev); g2.manual_seed(seed + 7)
    S11 = torch.zeros(n, n, device=dev, dtype=torch.float64); S22 = torch.zeros(n, n, device=dev, dtype=torch.float64)
    S31 = torch.zeros(n, n, device=dev, dtype=torch.float64); M4 = torch.zeros(n, device=dev, dtype=torch.float64)
    done = 0
    while done < N:
        m = min(CH, N - done); h = torch.randn(m, n, device=dev, generator=g2)
        for k in range(l - 1): h = torch.relu(h @ W[k])
        y = h @ W[l - 1] - mu
        y2 = y * y
        S11 += (y.T @ y).double(); S22 += (y2.T @ y2).double(); S31 += ((y2 * y).T @ y).double(); M4 += (y2 * y2).sum(0, dtype=torch.float64)
        done += m
    S11 /= N; S22 /= N; S31 /= N; M4 /= N
    var = torch.diag(S11)
    K22 = S22 - torch.outer(var, var) - 2 * S11 * S11        # kappa4(i,i,j,j)
    K31 = S31 - 3 * var[:, None] * S11                       # kappa4(i,i,i,j)
    K4d = M4 - 3 * var * var                                 # kappa4(i,i,i,i)
    np.savez_compressed(os.path.join(OUT, f"layer{l}_seed{seed}.npz"),
                        mean=m1.cpu().numpy(), relu_mean=mrelu.cpu().numpy(), C=S11.cpu().numpy().astype(np.float32),
                        K22=K22.cpu().numpy().astype(np.float32), K31=K31.cpu().numpy().astype(np.float32), K4d=K4d.cpu().numpy(), N=N)
    Coff = S11 - torch.diag(var); K22off = K22 - torch.diag(torch.diag(K22))
    lam = float((K22off * Coff).sum() / (Coff * Coff).sum())
    r2 = float(1 - ((K22off - lam * Coff) ** 2).sum() / (K22off ** 2).sum())
    print(f"layer {l:2d}: var mean {float(var.mean()):.4f}  kappa4 diag mean {float(K4d.mean()):.3e}  |K22off| {float(K22off.norm()):.3e}  "
          f"LS lambda (K22off ~ lam C_off) {lam:.4e}  R^2 {r2:.3f}   [{time.time()-t0:.0f}s]", flush=True)
