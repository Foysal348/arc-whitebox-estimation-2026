"""Single-pass version of k4teacher.py: raw moments -> central cumulant slices (half the GPU time; one forward stream).
Accumulates per layer: R1 = sum z, R11 = z^T z, R21 = (z^2)^T z, R22 = (z^2)^T (z^2), R31 = (z^3)^T z, and diagonal sums of z^3, z^4,
then converts to kappa4(i,i,j,j), kappa4(i,i,i,j), kappa4(i^4) and the covariance. Validated against the two-pass version.
Env: WIDTH DEPTH N SEED OUT LAYERS CHUNK DEVICE
"""
import os, time, numpy as np, torch
n = int(os.environ.get("WIDTH", "1024")); L = int(os.environ.get("DEPTH", "16")); N = int(float(os.environ.get("N", "1e8")))
seed = int(os.environ.get("SEED", "0")); OUT = os.environ.get("OUT", "./k4teach"); CH = int(os.environ.get("CHUNK", str(1 << 14)))
LAYERS = list(range(1, L + 1)) if os.environ.get("LAYERS", "all") == "all" else [int(x) for x in os.environ["LAYERS"].split(",")]
dev = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
os.makedirs(OUT, exist_ok=True)

MLP_IDX = os.environ.get("MLP_IDX", "")      # if set: use the public mini-split MLP of that index (its 1e9-sample truth exists)
if MLP_IDX != "":
    from datasets import load_dataset
    ds = load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini").with_format("numpy")
    row = ds[int(MLP_IDX)]; Wnp = [np.ascontiguousarray(row["weights"][k]) for k in range(L)]
    n = Wnp[0].shape[0]
    np.savez_compressed(os.path.join(OUT, "truth.npz"), all_layer_means=row["all_layer_means"], final_means=row["final_means"],
                        mlp_name=str(row["mlp_name"]), mlp_seed=int(row["mlp_seed"]))
    print("public MLP", row["mlp_name"], n, L, flush=True)
else:
    rng = np.random.default_rng(1000 + seed)
    Wnp = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
W = [torch.as_tensor(w, device=dev) for w in Wnp]
np.savez_compressed(os.path.join(OUT, f"weights_seed{seed}.npz"), W=np.stack(Wnp))
for l in LAYERS:
    t0 = time.time(); g = torch.Generator(device=dev); g.manual_seed(seed + 7)
    Z = lambda: torch.zeros(n, n, device=dev, dtype=torch.float64)
    R1 = torch.zeros(n, device=dev, dtype=torch.float64); R11 = Z(); R21 = Z(); R22 = Z(); R31 = Z()
    R3d = torch.zeros(n, device=dev, dtype=torch.float64); R4d = torch.zeros(n, device=dev, dtype=torch.float64)
    Rrelu = torch.zeros(n, device=dev, dtype=torch.float64); done = 0
    while done < N:
        m = min(CH, N - done); h = torch.randn(m, n, device=dev, generator=g)
        for k in range(l - 1): h = torch.relu(h @ W[k])
        z = h @ W[l - 1]; z2 = z * z; z3 = z2 * z
        R1 += z.sum(0, dtype=torch.float64); Rrelu += torch.relu(z).sum(0, dtype=torch.float64)
        R11 += (z.T @ z).double(); R21 += (z2.T @ z).double(); R22 += (z2.T @ z2).double(); R31 += (z3.T @ z).double()
        R3d += z3.sum(0, dtype=torch.float64); R4d += (z2 * z2).sum(0, dtype=torch.float64)
        done += m
        if done % (CH * 1024) == 0: print(f"  layer {l}: {done/N:.0%} ({time.time()-t0:.0f}s)", flush=True)
    m1 = R1 / N; E11 = R11 / N; E21 = R21 / N; E22 = R22 / N; E31 = R31 / N; E3d = R3d / N; E4d = R4d / N
    mi = m1[:, None]; mj = m1[None, :]; E2d = torch.diag(E11)
    C = E11 - torch.outer(m1, m1)                                   # cov
    v = torch.diag(C)
    Y21 = E21 - 2 * mi * E11 - mj * E2d[:, None] + 2 * mi ** 2 * mj                     # E[y_i^2 y_j]
    Y22 = (E22 - 2 * mi * E21.T - 2 * mj * E21 + mi ** 2 * E2d[None, :] + mj ** 2 * E2d[:, None]
           + 4 * mi * mj * E11 - 3 * mi ** 2 * mj ** 2)                                  # E[y_i^2 y_j^2]
    Y31 = (E31 - 3 * mi * E21 - mj * E3d[:, None] + 3 * mi ** 2 * E11 + 3 * mi * mj * E2d[:, None] - 3 * mi ** 3 * mj)  # E[y_i^3 y_j]
    Y4 = E4d - 4 * m1 * E3d + 6 * m1 ** 2 * E2d - 3 * m1 ** 4                            # E[y_i^4]
    K22 = Y22 - torch.outer(v, v) - 2 * C * C
    K31 = Y31 - 3 * v[:, None] * C
    K4d = Y4 - 3 * v * v
    np.savez_compressed(os.path.join(OUT, f"layer{l}_seed{seed}.npz"), mean=m1.cpu().numpy(), relu_mean=(Rrelu / N).cpu().numpy(),
                        C=C.cpu().numpy().astype(np.float32), K22=K22.cpu().numpy().astype(np.float32), K31=K31.cpu().numpy().astype(np.float32),
                        K4d=K4d.cpu().numpy(), Y21=Y21.cpu().numpy().astype(np.float32), N=N)
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    print(f"layer {l:2d}: var {float(v.mean()):.4f}  k4diag {float(K4d.mean()):.3e}  |K22off| rms {float((K22[off]**2).mean()**0.5):.3e}  "
          f"|K31off| rms {float((K31[off]**2).mean()**0.5):.3e}  [{time.time()-t0:.0f}s]", flush=True)
