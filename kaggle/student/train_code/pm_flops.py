"""Round-20 pair-memory student, flopscope implementation (11 Oct 2026): the export path and the resource falsifier.

This is the same forward pass as experiments/student/pm_student.py (PyTorch), written with flopscope ops.
- The tables (wick constants, TERM_SPECS programs, LAM/REF_R, V55 map) are read from the frozen V56 estimator at setup.
- The 364 head parameters come from PM_THETA (an .npy file of shape (4, 91)); default zeros = pure slice recursion.
- Classical matmuls throughout (no Strassen); 2-D ops only.
Usage as an estimator: whest run --estimator pm_flops.py (with PM_EST pointing at the V56 file for the tables).
"""
import math, os, types
import numpy as _np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_EST = os.environ.get("PM_EST", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v56_dump.py"))
_THETA = os.environ.get("PM_THETA", "")


def _load_tables():
    mod = types.ModuleType("pm_tables"); mod.__file__ = os.path.abspath(_EST)
    exec(compile(open(_EST, encoding="utf-8").read(), os.path.abspath(_EST), "exec"), mod.__dict__)
    C1, C2, SELC = mod._build_wick_consts()
    return dict(C1=C1, C2=C2, SELC=SELC, progs={m: mod._term_prog(m) for m in (1, 3)}, WICK_PAIRS=mod.WICK_PAIRS,
                LAM=list(mod.LAM), REF_R=list(mod.REF_R), BETA=mod.BETA, MC=mod.METRIC_C, st=mod._statics(1024),
                K4C_BF=list(mod.K4C_BF), K4C_BL=list(mod.K4C_BL), K4C_LMIN=mod.K4C_LMIN, ROWC_THR=mod.ROWC_THR,
                D21_THR=mod.D21_THR, n1=len(mod.D1_IPS), n2=len(mod.D2_IPS))


def _zd(M):
    fnp.fill_diagonal(M, 0.0)
    return M


class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        self.T = _load_tables()
        self.theta = _np.load(_THETA).astype(_np.float32) if _THETA else _np.zeros((4, 91), _np.float32)
        wp = self.T["WICK_PAIRS"]; self.i11 = wp.index((1, 1))
        iu = _np.triu_indices(12)
        self.iu0, self.iu1 = iu[0], iu[1]
        # per-term static data of the programs (numpy, outside the meter)
        self.terms = {}
        for m, prog in self.T["progs"].items():
            t1 = [(a, b, int(_np.argmax(_np.abs(prog["SL1"][t]))), float(sum(prog["SL1"][t])),
                   prog["IND1"][t].index(1.0)) for t, (a, b) in enumerate(prog["B1"])]
            t2 = [(a, b, int(_np.argmax(_np.abs(prog["SL2"][t]))), float(sum(prog["SL2"][t])), int(_np.argmax(prog["SR2"][t])),
                   prog["IND2"][t].index(1.0)) for t, (a, b) in enumerate(prog["AB2"])]
            self.terms[m] = (t1, t2)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return self._forward(mlp)

    # ------------------------------------------------------------------------------------------------------------
    def _heads(self, X):                 # X: (n, 12) fnp
        n = X.shape[0]
        Q2 = X[:, self.iu0] * X[:, self.iu1]
        psi = fnp.concatenate([fnp.ones((n, 1), dtype=fnp.float32), X, Q2], axis=1)   # (n, 91)
        Q = psi @ fnp.asarray(self.theta.T)
        rho = fnp.exp(0.25 * fnp.tanh(Q[:, 0])); eta = 0.25 * fnp.tanh(Q[:, 1])
        o = 0.5 * fnp.tanh(Q[:, 2]); g = 0.2 * fnp.tanh(Q[:, 3])
        return rho, eta, o, g

    def _features(self, alpha, sigma, w1, phi, R, H1, H2, g4row, Bdiag=None, Brow=None):
        s3 = sigma ** 3
        sc = (sigma * sigma)[:, None] * sigma[None, :]
        sbar = fnp.mean(sc)
        rms = lambda M: fnp.sqrt(fnp.mean(M * M, axis=1) + 1e-30)
        ls = fnp.log(sigma)
        zero = alpha * 0.0
        cols = [fnp.tanh(alpha / 2), 2 * w1 - 1, phi * math.sqrt(2 * math.pi), fnp.tanh(ls - fnp.mean(ls)), fnp.tanh(2 * rms(R)),
                fnp.tanh(Bdiag / s3) if Bdiag is not None else zero, fnp.tanh(2 * Brow) if Brow is not None else zero,
                fnp.tanh(fnp.diag(H1) / s3), fnp.tanh(rms(H1 / (sc + 1e-12 * sbar))),
                fnp.tanh(fnp.diag(H2) / s3), fnp.tanh(rms(H2 / (sc + 1e-12 * sbar))), fnp.tanh(g4row / (6 * sigma ** 4 + 1e-12))]
        return fnp.stack(cols, axis=1)

    def _wick(self, mu, var):
        T = self.T
        sigma = fnp.sqrt(var); alpha = mu / sigma
        phi = flops.stats.norm.pdf(alpha).astype(fnp.float32); Phi = flops.stats.norm.cdf(alpha).astype(fnp.float32)
        ones = alpha * 0.0 + 1.0
        a2 = alpha * alpha
        APOW = fnp.stack([ones, alpha, a2, a2 * alpha, a2 * a2, a2 * a2 * alpha], axis=1)
        inv = 1.0 / sigma; i2 = inv * inv; i3 = i2 * inv
        s2 = sigma * sigma
        SB = fnp.stack([i3 * i3, i3 * i2, i2 * i2, i3, i2, inv, ones, sigma, s2, s2 * sigma, s2 * s2], axis=1)
        W_all = ((APOW @ fnp.asarray(_np.asarray(T["C1"], _np.float32))) * phi[:, None]
                 + (APOW @ fnp.asarray(_np.asarray(T["C2"], _np.float32))) * Phi[:, None]) * (SB @ fnp.asarray(_np.asarray(T["SELC"], _np.float32)))
        return W_all, sigma, alpha, phi, Phi

    def _closure(self, mu, var, b2, b1, pmode, last):
        W_all, sigma, alpha, phi, Phi = self._wick(mu, var)
        t1, t2 = self.terms[pmode]
        n = mu.shape[0]
        PK1 = [None] * self.T["n1"]
        for a, b, wl, coef, g in t1:
            val = b1[b] if a == "ones1" else (b1[a] if b == "ones1" else b1[a] * b1[b])
            c = val * (W_all[:, wl] * coef)
            PK1[g] = c if PK1[g] is None else PK1[g] + c
        out = dict(W_all=W_all, phi=phi, Phi=Phi, pk1=PK1[0])
        if last:
            return out
        PK2 = [None] * self.T["n2"]
        for a, b, wl, coef, wr, g in t2:
            val = b2[b] if a == "ones2" else (b2[a] if b == "ones2" else b2[a] * b2[b])
            c = val * (W_all[:, wl] * coef)[:, None] * W_all[:, wr][None, :]
            PK2[g] = c if PK2[g] is None else PK2[g] + c
        pk1, pk2, pk3, pk4 = PK1
        pk11 = _zd(0.5 * (PK2[0] + PK2[0].T)); pk21 = _zd(PK2[1] * 1.0); pk22 = _zd(0.5 * (PK2[2] + PK2[2].T))
        K2 = pk2 - pk1 * pk1
        K3 = 2 * pk1 ** 3 - 3 * pk1 * pk2 + pk3
        K4 = -6 * pk1 ** 4 + 12 * pk1 ** 2 * pk2 - 4 * pk1 * pk3 - 3 * pk2 ** 2 + pk4
        c, r = pk1[:, None], pk1[None, :]
        K21 = _zd(-2.0 * c * pk11 + pk21)
        K22 = _zd(4.0 * r * c * pk11 - 2.0 * r * pk21 - 2.0 * c * pk21.T - 2.0 * pk11 * pk11 + pk22)
        Cpost = pk11 * 1.0
        fnp.fill_diagonal(Cpost, K2)
        out.update(K1=pk1, K2=K2, K3=K3, K4=K4, K21=K21, K22=K22, C=Cpost)
        return out

    def _forward(self, mlp):
        T = self.T; MC = T["MC"]; st = T["st"]
        n = mlp.width; L = len(mlp.weights)
        f32 = fnp.float32
        ones = fnp.ones(n, dtype=f32); ones2 = ones[:, None] * ones[None, :]
        mu_post = fnp.zeros(n, dtype=f32); C_post = None
        H1h = H2h = None; g_prev = var_prev = None; lam_prev = 0.0
        rows = []
        for li, w in enumerate(mlp.weights):
            last = li == L - 1; layer = li + 1
            w32 = w if w.dtype == f32 else w.astype(f32)
            mu = mu_post @ w32
            if li == 0:
                C_pre = fnp.einsum("ia,ib->ab", w32, w32)
            else:
                C_pre = w32.T @ (C_post @ w32)
            var = fnp.maximum(fnp.diag(C_pre), 1e-10)
            C_off = _zd(C_pre * 1.0)
            sig = fnp.sqrt(var); alpha0 = mu / sig
            Phi0 = flops.stats.norm.cdf(alpha0).astype(f32)
            if li > 0:
                w2c = w32 * w32
                P1 = w2c.T @ (H1h @ w32)
                cdiag = (w2c * w32).T @ fnp.diag(H1h)
                d3 = 3.0 * (fnp.diag(P1) - cdiag) + cdiag
                P2 = w2c.T @ (H2h @ w32)
                W_all0, _, _, phi0, _ = self._wick(mu, var)
                w1_0 = fnp.where(Phi0 < T["ROWC_THR"], 0.0, W_all0[:, self.i11]).astype(f32)
                R = C_pre / (sig[:, None] * sig[None, :] + 1e-12 * fnp.mean(var))
                X0 = self._features(alpha0, sig, w1_0, phi0, R, P1, P2, var * 0.0)
                _, _, o, gh = self._heads(X0)
                OP = o[:, None] * P2
                D21 = _zd(P1 + OP - OP.T)
                d3 = (1.0 + gh) * d3
                D21 = D21 * fnp.where(Phi0 >= T["D21_THR"], 1.0, 0.0).astype(f32)[None, :]
                WW = w32.T * w32.T
                t_g = WW @ g_prev; t_v = var - WW @ var_prev
                dG0 = t_g + t_v * lam_prev
                rr = fnp.mean(dG0) / fnp.mean(var)
                ref = float(T["REF_R"][min(li - 1, len(T["REF_R"]) - 1)])
                lam = lam_prev * fnp.power(fnp.clip(rr / ref, 0.5, 2.0), T["BETA"])
                dG = t_g + t_v * lam
                g4row = dG * MC
                g22c = (dG * (MC / 6.0))[:, None]
                wk4m = _zd(g22c + g22c.T)
                g4mean = g4row
                if layer >= T["K4C_LMIN"]:
                    v2 = var * var; ch = g4row / v2; s3sq = (d3 / (var * sig)) ** 2; absa = fnp.abs(alpha0)
                    if not last:
                        Zs = D21 / (var[:, None] * sig[None, :]); Z2 = Zs * Zs
                        rc = C_pre / (sig[:, None] * sig[None, :])
                        X = fnp.stack([s3sq, fnp.sum(Z2, axis=1), fnp.sum(Z2, axis=0), fnp.sum(rc * rc, axis=1), absa, ch, ones], axis=0)
                        bt = fnp.asarray(_np.asarray(T["K4C_BF"], _np.float32))
                    else:
                        X = fnp.stack([s3sq, absa, ch, ones], axis=0)
                        bt = fnp.asarray(_np.asarray(T["K4C_BL"], _np.float32))
                    g4new = (bt @ X) * v2
                    g4mean = g4new * (fnp.mean(g4row) / fnp.mean(g4new))
                pmode = 1
                b2 = {"ones2": ones2, "c_off": C_off}
                if not last:
                    d3c = d3[:, None] * ones[None, :]; g4c = g4row[:, None] * ones[None, :]
                    b2.update({"d21": D21, "d21T": D21.T, "wk4m": wk4m, "d3col": d3c, "d3row": d3c.T, "g4col": g4c, "g4row": g4c.T,
                               "wk431": C_off * (0.5 * MC * lam)})
                b1 = {"ones1": ones, "d3": d3, "g4": g4mean}
            else:
                ang_g = -6.0 / (n + 2.0)
                dG = var * ang_g; g4row = dG * MC
                g22c = (dG * (MC / 6.0))[:, None]
                wk4m = _zd(g22c + g22c.T)
                g4c = g4row[:, None] * ones[None, :]
                b2 = {"ones2": ones2, "c_off": C_off, "wk4m": wk4m, "g4col": g4c, "g4row": g4c.T, "wk431": C_off * (0.5 * MC * ang_g)}
                b1 = {"ones1": ones, "g4": g4row}
                pmode = 3
            cl = self._closure(mu, var, b2, b1, pmode, last)
            if last:
                rows.append(cl["pk1"]); break
            rows.append(cl["K1"])
            W_all = cl["W_all"]
            w1 = fnp.where(cl["Phi"] < T["ROWC_THR"], 0.0, W_all[:, self.i11]).astype(f32)
            Pfull = cl["K21"] * 1.0
            fnp.fill_diagonal(Pfull, cl["K3"])
            if li > 0:
                w1s = (w1 * w1)[:, None]; w1r = w1[None, :]
                L1 = w1s * P1 * w1r; L2 = w1s * P2 * w1r
                sc = (sig * sig)[:, None] * sig[None, :]
                Brow = fnp.sqrt(fnp.mean((Pfull / (sc + 1e-12 * fnp.mean(sc))) ** 2, axis=1) + 1e-30)
                Rr = C_pre / (sig[:, None] * sig[None, :] + 1e-12 * fnp.mean(var))
                X = self._features(alpha0, sig, w1, cl["phi"], Rr, P1, P2, g4row, Bdiag=cl["K3"], Brow=Brow)
                rho, eta, _, _ = self._heads(X)
                H1h = Pfull + (rho - 1.0)[:, None] * L1 + eta[:, None] * L2
                H2h = 0.5 * (Pfull - Pfull.T) + eta[:, None] * L1 + rho[:, None] * L2
            else:
                H1h = Pfull
                H2h = 0.5 * (Pfull - Pfull.T)
            mu_post, C_post = cl["K1"], cl["C"]
            k22row = fnp.sum(cl["K22"], axis=1)
            g_prev = (cl["K4"] + k22row) * float(st["cA"]) + (fnp.sum(cl["K4"]) + fnp.sum(k22row)) * float(st["cI"])
            var_prev = cl["K2"]
            lam_prev = float(T["LAM"][min(li, len(T["LAM"]) - 1)])
        out = fnp.stack(rows, axis=0)
        a1 = math.exp(math.lgamma((n + 1) / 2.0) - math.lgamma(n / 2.0)) * math.sqrt(2.0 / n)
        return out * float(a1)
