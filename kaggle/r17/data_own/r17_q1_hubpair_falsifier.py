#!/usr/bin/env python3
"""
Round 17 falsifier: hub-pair (curvature-hub x kappa3-hub) kappa4 carrier.
[Received from Claude (outside consultation) on 8 Oct 2026. Our edits are marked "OURS":
 --teacher-npz (one packed file with mean / C / K4d per layer), and the project's own metric added next to the
 author's: decisive = |mu_t/sigma_t| < 1.5 from the teacher, pattern = corr(pred / v_t^2, K4d_t / v_t^2).
 The gate is evaluated on the project metric.]

Computes, from exact Monte-Carlo samples of the TRUE bias-free ReLU MLP, the
Q1 (C.T) curvature birth of the post-ReLU fourth cumulant, carried by
U_m = diag(Phi_m) W[m], read out as the pre-activation kappa4 diagonal, and
reports decisive-neuron Pearson correlation with the teacher kappa4 diagonal.

Conventions (row vectors): z_1 = x W[0], h_l = relu(z_l), z_{l+1} = h_l W[l].
Birth at ReLU layer l (cumulants of u = z_l - mu_l), target pre-activation z_t, age = t - l.
  Ybar_{l,t} = W[l] U_{l+1} ... U_{t-1}           (n x n)
  Ydot_{l,t} = diag(Phi_l) Ybar_{l,t}
  Q1  : D_i  = 24 * sum_a g_a Ybar_ai (C_l Ydot)_ai * E[u_a (u . Ydot_i)^2]
  path: D_i  = 48 * v_i^T C_l v_i,  v_a = g_a Ybar_ai (C_l Ydot)_ai       (Gaussian part of Q2)
  g_a = phi(alpha_a)/(2 sigma_a),  alpha = mu/sigma.
T(a;x,x) = E[u_a (u.x)^2] is the exact third-cumulant contraction (u centred), so no
n^3 tensor is formed. Truncations (age H', hub rank r = top-r by g) are applied post hoc.

Dtypes: float32 forward, float64 accumulation. Marks: every printed number is MEASURED.
Gate (pre-registered): PASS if corr >= 0.70 at L12, L15 and L16 for (H=6, r=n, Q1);
KILL if < 0.55 at all three; otherwise INCONCLUSIVE.
"""
import argparse, json, os, time
import numpy as np
import torch


def load_weights(path):
    d = np.load(path)
    key = "W" if "W" in d.files else d.files[0]
    W = d[key].astype(np.float32)
    assert W.ndim == 3 and W.shape[1] == W.shape[2], W.shape
    return W


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--teacher-npz", required=True, help="OURS: npz with mean (17,n), C (17,n,n), K4d (17,n)")
    ap.add_argument("--N", type=int, default=2_000_000)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--targets", type=int, nargs="+", default=[12, 15, 16])
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--decisive-k", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="r17_q1_hubpair.json")
    a = ap.parse_args()

    dev = torch.device(a.device)
    W_np = load_weights(a.weights)
    L, n, _ = W_np.shape
    W = torch.from_numpy(W_np).to(dev)
    TD = np.load(a.teacher_npz)                                        # OURS
    maxl = max(a.targets)
    pairs = sorted({(t - age, t) for t in a.targets for age in range(1, a.H + 1) if t - age >= 1})
    layers = list(range(min(l for l, _ in pairs), maxl + 1))
    N = (a.N // a.chunk) * a.chunk
    nch = N // a.chunk
    t0 = time.time()

    def forward(gen):
        h = torch.randn(a.chunk, n, generator=gen, device=dev, dtype=torch.float32)
        zs = {}
        for l in range(1, maxl + 1):
            z = h @ W[l - 1]
            zs[l] = z
            h = torch.relu(z)
        return zs

    # ---------------- pass 1: mean and covariance of z_l ----------------
    s1 = {l: torch.zeros(n, dtype=torch.float64, device=dev) for l in layers}
    s2 = {l: torch.zeros(n, n, dtype=torch.float64, device=dev) for l in layers}
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    for c in range(nch):
        zs = forward(gen)
        for l in layers:
            s1[l] += zs[l].double().sum(0)
            s2[l] += (zs[l].T @ zs[l]).double()
    mu = {l: s1[l] / N for l in layers}
    C = {l: s2[l] / N - torch.outer(mu[l], mu[l]) for l in layers}
    sig = {l: C[l].diag().clamp_min(1e-30).sqrt() for l in layers}
    alpha = {l: mu[l] / sig[l] for l in layers}
    Phi = {l: torch.special.ndtr(alpha[l]) for l in layers}
    phi = {l: torch.exp(-0.5 * alpha[l] ** 2) / np.sqrt(2 * np.pi) for l in layers}
    g = {l: phi[l] / (2 * sig[l]) for l in layers}
    print(f"[pass1] {time.time()-t0:.0f}s", flush=True)

    # ---------------- transport matrices ----------------
    Ybar = {}
    for l in sorted({l for l, _ in pairs}):
        Y = W[l].double()                       # h_l -> z_{l+1}
        for t in range(l + 1, maxl + 1):
            if (l, t) in pairs:
                Ybar[(l, t)] = Y.clone()
            if t < maxl:
                Y = Y @ (Phi[t].unsqueeze(1) * W[t].double())   # U_t = diag(Phi_t) W[t]
    Ydot = {k: Phi[k[0]].unsqueeze(1) * Y for k, Y in Ybar.items()}
    Yd32 = {k: v.float() for k, v in Ydot.items()}
    mu32 = {l: mu[l].float() for l in layers}

    # ---------------- pass 2: E[u_a (u.Ydot_i)^2] and MC kappa4 diagonal ----------------
    G = {k: torch.zeros(n, n, dtype=torch.float64, device=dev) for k in pairs}
    m4 = {t: torch.zeros(n, dtype=torch.float64, device=dev) for t in a.targets}
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    for c in range(nch):
        zs = forward(gen)
        for (l, t) in pairs:
            u = zs[l] - mu32[l]
            s = u @ Yd32[(l, t)]
            G[(l, t)] += (u.T @ (s * s)).double()
        for t in a.targets:
            m4[t] += ((zs[t].double() - mu[t]) ** 4).sum(0)
        if c % max(1, nch // 10) == 0:
            print(f"[pass2] chunk {c}/{nch} {time.time()-t0:.0f}s", flush=True)

    # ---------------- readout and metrics ----------------
    cont, path = {}, {}
    for (l, t) in pairs:
        Yb, Yd = Ybar[(l, t)], Ydot[(l, t)]
        CY = C[l] @ Yd
        Mv = g[l].unsqueeze(1) * Yb * CY
        cont[(l, t)] = (24.0 * Mv * (G[(l, t)] / N)).cpu().numpy()   # [a, i]
        path[(l, t)] = Mv.cpu().numpy()
    Cn = {l: C[l].cpu().numpy() for l in layers}
    gn = {l: g[l].cpu().numpy() for l in layers}

    def predict(t, Hp, r, with_path):
        pred = np.zeros(n)
        for age in range(1, Hp + 1):
            l = t - age
            if (l, t) not in cont:
                continue
            mask = np.zeros(n, bool)
            mask[np.argsort(-gn[l])[:r]] = True
            pred += cont[(l, t)][mask].sum(0)
            if with_path:
                Mv = path[(l, t)] * mask[:, None]
                pred += 48.0 * ((Cn[l] @ Mv) * Mv).sum(0)
        return pred

    configs = [("ceiling Q1 (H=6, r=n)", a.H, n, False),
               ("Q1+path (H=6, r=n)", a.H, n, True),
               ("Q1 (H=4, r=n)", min(4, a.H), n, False),
               ("Q1 (H=2, r=n)", min(2, a.H), n, False),
               ("Q1 (H=6, r=64)", a.H, min(64, n), False),
               ("Q1 (H=4, r=32)", min(4, a.H), min(32, n), False)]
    res = {"n": n, "N": N, "pairs": len(pairs), "configs": {}, "teacher_source": "stored 6e7 teacher (packed npz)"}
    for name, Hp, r, wp in configs:
        res["configs"][name] = {}
    for t in a.targets:
        tk = TD["K4d"][t].astype(np.float64)
        vt = np.diag(TD["C"][t]).astype(np.float64); mut = TD["mean"][t].astype(np.float64)       # OURS
        dec_proj = np.abs(mut / np.sqrt(vt)) < 1.5                                                  # OURS: project decisive set
        score = phi[t].cpu().numpy() * np.abs(tk) / sig[t].cpu().numpy() ** 3                      # author's proxy
        dec_auth = np.argsort(-score)[:a.decisive_k]
        mc_k4 = (m4[t] / N - 3 * C[t].diag() ** 2).cpu().numpy()
        for name, Hp, r, wp in configs:
            p = predict(t, Hp, r, wp)
            res["configs"][name][f"L{t}"] = {
                "corr_project": pearson((p / vt ** 2)[dec_proj], (tk / vt ** 2)[dec_proj]),               # OURS (gate)
                "corr_author_raw_top128": pearson(p[dec_auth], tk[dec_auth]),
                "level_ratio_project": float(p[dec_proj].mean() / tk[dec_proj].mean())}
        res.setdefault("mc_sanity", {})[f"L{t}"] = pearson((mc_k4 / vt ** 2)[dec_proj], (tk / vt ** 2)[dec_proj])
    cl = [res["configs"]["ceiling Q1 (H=6, r=n)"][f"L{t}"]["corr_project"] for t in a.targets]
    res["gate"] = "PASS" if all(c >= 0.70 for c in cl) else ("KILL" if all(c < 0.55 for c in cl) else "INCONCLUSIVE")
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
