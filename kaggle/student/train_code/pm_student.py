"""Round-20 pair-memory student (PM364, adapted), PyTorch (11 Oct 2026).

The student carries (mu, C, H1, H2) with no per-ReLU sources. Per layer:

1. Pre-activation transport.
   - mu = W mu_post and C = W C_post Wᵀ (W = w32ᵀ, row convention z = h w32).
   - The pair memories are transported by the restricted cubic operator P_k = (w32∘w32)ᵀ H_k w32.
2. Readout.
   - D21 = offdiag(P1) + O, with O_ij = o_i P2_ij − o_j P2_ji (an explicitly odd correction).
   - d3 is the exact restricted diagonal (1 + g_i)·[3 (P1_ii − c_i) + c_i], with c = (w32∘w32∘w32)ᵀ diag(H1)
     (the coherent placements that the off-diagonal operator drops).
   - The chain's D21COL rule zeroes the columns of dead neurons (Phi < D21_THR).
3. κ4 inputs.
   - Exactly V56's memory-carried regeneration (dG, adaptive λ, wk4m, wk431, V31 layer-0 core).
   - The V55 pattern map (7 features, no star) for the mean closure.
4. Closure.
   - V56's K3-simple closure, ported exactly: wick matrix, TERM_SPECS programs, PK2K. It returns the post-ReLU mean,
     variance, d3, k4, K11 (covariance), K21 (the (2,1) slice) and K22.
5. Memory update (post-ReLU).
   - Pfull = K21 + diag(K3v), the closure's own local post slice, which already contains births, the κ4→κ3 feed and
     the linear dressing.
   - H1h = Pfull + (ρ − 1)⊙L(H1) + η⊙L(H2)
   - H2h = (Pfull − Pfullᵀ)/2 + η⊙L(H1) + ρ⊙L(H2)
   - L(H)_ij = w1_i² w1_j H_ij on the transported (pre-activation) memories.

The four heads (ρ, η, o, g) are quadratic in 12 bounded per-neuron features: 4 × 91 = 364 parameters, initialised at
zero, so the zero model is the pure slice recursion.

Teacher forcing (for parity and warm-up): `force[layer] = dict(D21=..., d3=..., g4mean=..., mu=..., C=...)` replaces the
corresponding student quantities at that (1-based) layer.
"""
import math, types, os
import torch

_TABLES = None


def load_tables(est_path):
    """Exec the frozen estimator once to read its tables and helpers (needs flopscope importable)."""
    global _TABLES
    if _TABLES is None:
        mod = types.ModuleType("est_tables"); mod.__file__ = os.path.abspath(est_path)
        exec(compile(open(est_path, encoding="utf-8").read(), os.path.abspath(est_path), "exec"), mod.__dict__)
        C1, C2, SELC = mod._build_wick_consts()
        _TABLES = dict(C1=C1, C2=C2, SELC=SELC, progs={m: mod._term_prog(m) for m in (0, 1, 2, 3)},
                       WICK_PAIRS=mod.WICK_PAIRS, LAM=list(mod.LAM), REF_R=list(mod.REF_R), BETA=mod.BETA,
                       METRIC_C=mod.METRIC_C, statics=mod._statics(1024), K4C_BF=list(mod.K4C_BF), K4C_BL=list(mod.K4C_BL),
                       K4C_LMIN=mod.K4C_LMIN, ROWC_THR=mod.ROWC_THR, D21_THR=mod.D21_THR, ANGULAR=mod.ANGULAR, ANG_A1=mod.ANG_A1,
                       D2_IPS=mod.D2_IPS, D1_IPS=mod.D1_IPS)
    return _TABLES


def _zd(M):
    return M - torch.diag_embed(torch.diagonal(M))


class PairMemoryStudent(torch.nn.Module):
    NF = 12

    def __init__(self, est_path, device="cpu", dtype=torch.float32, use_h2=True, use_o=True, use_g=True):
        super().__init__()
        T = load_tables(est_path)
        self.T = T; self.dev = device; self.dt = dtype
        self.use_h2, self.use_o, self.use_g = use_h2, use_o, use_g
        self.theta = torch.nn.Parameter(torch.zeros(4, 91, device=device, dtype=dtype))     # rho, eta, o, g
        t = lambda x: torch.tensor(x, device=device, dtype=dtype)
        self.C1, self.C2, self.SELC = t(T["C1"]), t(T["C2"]), t(T["SELC"])
        wp = T["WICK_PAIRS"]; self.i11, self.i21 = wp.index((1, 1)), wp.index((2, 1))
        self.progs = T["progs"]
        iu = torch.triu_indices(self.NF, self.NF)
        self.register_buffer("iu0", iu[0].to(device)); self.register_buffer("iu1", iu[1].to(device))

    # ---- helpers -------------------------------------------------------------------------------------------------
    def _psi(self, X):                       # X: (n, 12) -> (n, 91) = [1, x, upper-triangular x x^T]
        q = X[:, self.iu0] * X[:, self.iu1]
        return torch.cat([torch.ones_like(X[:, :1]), X, q], dim=1)

    def heads(self, X):
        Q = self._psi(X) @ self.theta.T     # (n, 4)
        rho = torch.exp(0.25 * torch.tanh(Q[:, 0])); eta = 0.25 * torch.tanh(Q[:, 1])
        o = 0.5 * torch.tanh(Q[:, 2]); g = 0.2 * torch.tanh(Q[:, 3])
        if not self.use_h2: eta = torch.zeros_like(eta)
        if not self.use_o: o = torch.zeros_like(o)
        if not self.use_g: g = torch.zeros_like(g)
        return rho, eta, o, g

    def wick(self, mu, var):
        sigma = torch.sqrt(var); alpha = mu / sigma
        phi = torch.exp(-0.5 * alpha * alpha) / math.sqrt(2 * math.pi); Phi = 0.5 * (1 + torch.erf(alpha / math.sqrt(2)))
        APOW = torch.stack([alpha ** k for k in range(6)], 1)
        inv = 1.0 / sigma
        SB = torch.stack([inv ** 6, inv ** 5, inv ** 4, inv ** 3, inv ** 2, inv, torch.ones_like(sigma), sigma, sigma ** 2, sigma ** 3, sigma ** 4], 1)
        W_all = ((APOW @ self.C1) * phi[:, None] + (APOW @ self.C2) * Phi[:, None]) * (SB @ self.SELC)
        return W_all, sigma, alpha, phi, Phi

    def closure(self, mu, var, C_off, b2, b1, pmode, last):
        W_all, sigma, alpha, phi, Phi = self.wick(mu, var)
        prog = self.progs[pmode]
        n = mu.shape[0]
        PK1 = [torch.zeros(n, device=self.dev, dtype=self.dt) for _ in range(len(self.T["D1_IPS"]))]
        for t, (a, b) in enumerate(prog["B1"]):
            val = b1[b] if a in ("ones1", "ones2") else (b1[a] if b in ("ones1", "ones2") else b1[a] * b1[b])
            wl = torch.tensor(prog["SL1"][t], device=self.dev, dtype=self.dt) @ W_all.T
            g = prog["IND1"][t].index(1.0)
            PK1[g] = PK1[g] + val * wl
        out = dict(W_all=W_all, sigma=sigma, alpha=alpha, phi=phi, Phi=Phi, pk1=PK1[0])
        if last:
            return out
        PK2 = [torch.zeros(n, n, device=self.dev, dtype=self.dt) for _ in range(len(self.T["D2_IPS"]))]
        for t, (a, b) in enumerate(prog["AB2"]):
            val = b2[b] if a in ("ones1", "ones2") else (b2[a] if b in ("ones1", "ones2") else b2[a] * b2[b])
            wl = torch.tensor(prog["SL2"][t], device=self.dev, dtype=self.dt) @ W_all.T
            wr = torch.tensor(prog["SR2"][t], device=self.dev, dtype=self.dt) @ W_all.T
            g = prog["IND2"][t].index(1.0)
            PK2[g] = PK2[g] + val * wl[:, None] * wr[None, :]
        pk1, pk2, pk3, pk4 = PK1
        pk11 = _zd(0.5 * (PK2[0] + PK2[0].T)); pk21 = _zd(PK2[1]); pk22 = _zd(0.5 * (PK2[2] + PK2[2].T))
        K2 = pk2 - pk1 * pk1
        K3 = 2 * pk1 ** 3 - 3 * pk1 * pk2 + pk3
        K4 = -6 * pk1 ** 4 + 12 * pk1 ** 2 * pk2 - 4 * pk1 * pk3 - 3 * pk2 ** 2 + pk4
        c, r = pk1[:, None], pk1[None, :]
        K21 = _zd(-2.0 * c * pk11 + pk21)
        K22 = _zd(4.0 * r * c * pk11 - 2.0 * r * pk21 - 2.0 * c * pk21.T - 2.0 * pk11 * pk11 + pk22)
        Cpost = 0.5 * (pk11 + pk11.T) + torch.diag_embed(K2)
        out.update(K1=pk1, K2=K2, K3=K3, K4=K4, K21=K21, K22=K22, C=Cpost)
        return out

    def features(self, alpha, sigma, w1, phi, R, H1, H2, g4row, Bdiag=None, Brow=None):
        s3 = sigma ** 3
        sc = (sigma ** 2)[:, None] * sigma[None, :]
        sbar = sc.mean()
        rms = lambda M: torch.sqrt((M * M).mean(1) + 1e-30)
        x4 = torch.tanh(torch.log(sigma) - torch.log(sigma).mean())
        zero = torch.zeros_like(alpha)
        cols = [torch.tanh(alpha / 2), 2 * w1 - 1, phi / (1 / math.sqrt(2 * math.pi)), x4, torch.tanh(2 * rms(R)),
                torch.tanh(Bdiag / s3) if Bdiag is not None else zero,
                torch.tanh(2 * Brow) if Brow is not None else zero,
                torch.tanh(torch.diagonal(H1) / s3), torch.tanh(rms(H1 / (sc + 1e-12 * sbar))),
                torch.tanh(torch.diagonal(H2) / s3), torch.tanh(rms(H2 / (sc + 1e-12 * sbar))),
                torch.tanh(g4row / (6 * sigma ** 4 + 1e-12))]
        return torch.stack(cols, 1)

    # ---- one full rollout -----------------------------------------------------------------------------------------
    def forward(self, Ws, force=None, return_traj=False, checkpoint=False):
        T = self.T; L = len(Ws); n = Ws[0].shape[0]; dev, dt = self.dev, self.dt
        st = T["statics"]; MC = T["METRIC_C"]
        mu_post = torch.zeros(n, device=dev, dtype=dt); C_post = None
        H1h = H2h = None
        g_prev = var_prev = None; lam_prev = 0.0
        rows, traj = [], []
        for li, w32 in enumerate(Ws):
            last = li == L - 1; layer = li + 1
            f = (force or {}).get(layer, {})
            W = w32.T
            mu = W @ mu_post
            C_pre = (w32.T @ w32) if li == 0 else (W @ C_post @ W.T)
            if "C" in f: C_pre = f["C"]
            if "mu" in f: mu = f["mu"]
            var = torch.clamp(torch.diagonal(C_pre), min=1e-10)
            C_off = _zd(C_pre)
            sig = torch.sqrt(var); alpha0 = mu / sig
            Phi0 = 0.5 * (1 + torch.erf(alpha0 / math.sqrt(2)))
            mode = 0 if li == 0 else 1
            rec = {}
            if mode == 1:
                w2c = w32 * w32
                P1 = w2c.T @ H1h @ w32
                cdiag = (w2c * w32).T @ torch.diagonal(H1h)
                d3 = 3.0 * (torch.diagonal(P1) - cdiag) + cdiag
                D21 = _zd(P1)
                if self.use_h2:
                    P2 = w2c.T @ H2h @ w32
                else:
                    P2 = torch.zeros_like(P1)
                # readout heads on pre-activation features
                W_all0, _, _, phi0, _ = self.wick(mu, var)
                w1_0 = torch.where(Phi0 < T["ROWC_THR"], torch.zeros_like(Phi0), W_all0[:, self.i11])
                R = C_pre / (sig[:, None] * sig[None, :] + 1e-12 * var.mean())
                X0 = self.features(alpha0, sig, w1_0, phi0, R, P1, P2, torch.zeros_like(var))
                _, _, o, gh = self.heads(X0)
                D21 = D21 + _zd(o[:, None] * P2 - (o[:, None] * P2).T) if self.use_o else D21
                d3 = (1.0 + gh) * d3
                # V50 D21COL: dead columns (Phi < D21_THR) are zero in the chain
                D21 = D21 * (Phi0 >= T["D21_THR"]).to(dt)[None, :]
                if "D21" in f: D21 = f["D21"]
                if "d3" in f: d3 = f["d3"]
                # kappa4 inputs: V56 regeneration with adaptive lambda
                WW = W * W
                t_g = WW @ g_prev; t_v = var - WW @ var_prev
                dG0 = t_g + t_v * lam_prev
                rr = dG0.mean() / var.mean()
                ref = float(T["REF_R"][min(li - 1, len(T["REF_R"]) - 1)])
                lam = lam_prev * torch.clamp(rr / ref, 0.5, 2.0) ** T["BETA"]
                dG = t_g + t_v * lam
                g4row = dG * MC
                g22c = (dG * (MC / 6.0))[:, None]
                wk4m = _zd(g22c + g22c.T)
                wk431 = None if last else C_off * (0.5 * MC * lam)
                # V55 pattern map (no star) for the mean closure
                g4mean = g4row
                if layer >= T["K4C_LMIN"]:
                    v2 = var * var; ch = g4row / v2; s3sq = (d3 / (var * sig)) ** 2; absa = torch.abs(alpha0)
                    if not last:
                        Zs = D21 / (var[:, None] * sig[None, :]); Z2 = Zs * Zs
                        rc = C_pre / (sig[:, None] * sig[None, :])
                        X = torch.stack([s3sq, Z2.sum(1), Z2.sum(0), (rc * rc).sum(1), absa, ch, torch.ones_like(var)], 0)
                        bt = torch.tensor(T["K4C_BF"], device=dev, dtype=dt)
                    else:
                        X = torch.stack([s3sq, absa, ch, torch.ones_like(var)], 0)
                        bt = torch.tensor(T["K4C_BL"], device=dev, dtype=dt)
                    g4new = (bt @ X) * v2
                    g4new = g4new * (g4row.mean() / g4new.mean())
                    g4mean = g4new
                if "g4mean" in f: g4mean = f["g4mean"]
                pmode = 1
                ones = torch.ones(n, device=dev, dtype=dt)
                b2 = {"ones2": ones[:, None] * ones[None, :], "c_off": C_off}
                if not last:
                    b2.update({"d21": D21, "d21T": D21.T, "wk4m": wk4m, "d3col": d3[:, None] * ones[None, :],
                               "d3row": (d3[:, None] * ones[None, :]).T, "g4col": g4row[:, None] * ones[None, :],
                               "g4row": (g4row[:, None] * ones[None, :]).T, "wk431": wk431})
                b1 = {"ones1": ones, "d3": d3, "g4": g4mean}
                rec.update(D21=D21, d3=d3, g4row=g4row, g4mean=g4mean, lam=lam)
                lam_used = lam
            else:
                ang_g = -6.0 / (n + 2.0)
                dG = var * ang_g
                g4row = dG * MC
                g22c = (dG * (MC / 6.0))[:, None]
                wk4m = _zd(g22c + g22c.T)
                wk431 = C_off * (0.5 * MC * ang_g)
                ones = torch.ones(n, device=dev, dtype=dt)
                pmode = 3
                b2 = {"ones2": ones[:, None] * ones[None, :], "c_off": C_off, "wk4m": wk4m, "g4col": g4row[:, None] * ones[None, :],
                      "g4row": (g4row[:, None] * ones[None, :]).T, "wk431": wk431}
                b1 = {"ones1": ones, "g4": g4row}
                lam_used = 0.0
            rec.update(mu_pre=mu, var=var, C_pre=C_pre)
            cl = self.closure(mu, var, C_off, b2, b1, pmode, last)
            if last:
                rows.append(cl["pk1"]); rec.update(post_mean=cl["pk1"]); traj.append(rec)
                break
            rows.append(cl["K1"])
            rec.update(post_mean=cl["K1"], K21=cl["K21"], post_d3=cl["K3"])
            traj.append(rec)
            # ---- memory update (post-ReLU) ----
            W_all = cl["W_all"]
            w1 = torch.where(cl["Phi"] < T["ROWC_THR"], torch.zeros_like(cl["Phi"]), W_all[:, self.i11])
            Pfull = cl["K21"] + torch.diag_embed(cl["K3"])
            if mode == 1:
                L1 = (w1 * w1)[:, None] * P1 * w1[None, :]
                L2 = (w1 * w1)[:, None] * P2 * w1[None, :]
                Rr = C_pre / (sig[:, None] * sig[None, :] + 1e-12 * var.mean())
                sc = (sig ** 2)[:, None] * sig[None, :]
                Brow = torch.sqrt(((Pfull / (sc + 1e-12 * sc.mean())) ** 2).mean(1) + 1e-30)
                X = self.features(alpha0, sig, w1, cl["phi"], Rr, P1, P2, g4row, Bdiag=cl["K3"], Brow=Brow)
                rho, eta, _, _ = self.heads(X)
                H1h = Pfull + (rho - 1.0)[:, None] * L1 + eta[:, None] * L2
                H2h = 0.5 * (Pfull - Pfull.T) + eta[:, None] * L1 + rho[:, None] * L2
            else:
                H1h = Pfull
                H2h = 0.5 * (Pfull - Pfull.T)
            if not self.use_h2:
                H2h = torch.zeros_like(H2h)
            mu_post, C_post = cl["K1"], cl["C"]
            k22row = cl["K22"].sum(1)
            g_prev = (cl["K4"] + k22row) * float(st["cA"]) + (cl["K4"].sum() + k22row.sum()) * float(st["cI"])
            var_prev = cl["K2"]
            lam_prev = float(T["LAM"][min(li, len(T["LAM"]) - 1)])
        out = torch.stack(rows, 0)
        if T["ANGULAR"] and T["ANG_A1"]:
            a1 = math.exp(math.lgamma((n + 1) / 2.0) - math.lgamma(n / 2.0)) * math.sqrt(2.0 / n)
            out = out * a1
        return (out, traj) if return_traj else out
