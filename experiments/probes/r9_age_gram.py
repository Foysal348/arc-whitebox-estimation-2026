"""Round 9 follow-up (GPT 5.6 Sol's Age-Gram test, 24 Sep): is the long kappa3 memory low-rank ACROSS AGES?

Correction that motivated it: the horizon oracle's age shares s_a = ||D_a|| / ||sum D_a|| have sum s_a^2 = 0.24-0.50,
not ~1, so ~50-76% of the total energy comes from POSITIVE cross-age inner products: the ages are coherent.

Per stored layer, with D_a the (2,1)-slice contribution of the source of age a (off-diagonal, dense n x n):
  H_ab = <D_a, D_b>_F, G_ab = H_ab / (||D_a|| ||D_b||); eigen-spectrum of both;
  first-mode model  D_a = alpha_a B + R_a  (B = unit-norm first principal matrix of the age axis) and the reconstruction
  error of the TOTAL after projecting the age axis on its first r modes: || sum_a D_a - sum_a P_r(D_a) || / || sum_a D_a ||.
This is not spatial compression: every mode is a dense full-rank n x n matrix.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r9_age_gram.py
"""
import os, json, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("ho", "../experiments/probes/r9_horizon_oracle.py")
CAP = "../data/dumps"; f64 = lambda x: np.asarray(x, np.float64)
# reuse the block builder without running the horizon script
src = open("../experiments/probes/r9_horizon_oracle.py", encoding="utf-8").read()
ns = {}; exec(src.split("out = []")[0], ns); blocks_of_source = ns["blocks_of_source"]
out = []
for l in (7, 11, 14, 15):
    c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); rres = int(c["rres"]); rfb = int(c["rfb"]); k = c["A_st"].shape[0]; n = c["A_st"].shape[1]
    off = ~np.eye(n, dtype=bool)
    Ds = []
    for s in range(k):
        sl = np.zeros((n, n))
        for (X, P, Y) in blocks_of_source(c, s, rres, rfb, n):
            sl += ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0
        Ds.append(sl[off])                                    # age = k - s
    Dm = np.stack(Ds[::-1])                                    # rows ordered by age 1..k
    tot = Dm.sum(0); nt = np.linalg.norm(tot)
    H = Dm @ Dm.T; nrm = np.sqrt(np.diag(H)); G = H / np.outer(nrm, nrm)
    shares = nrm / nt
    lamH, VH = np.linalg.eigh(H); lamH, VH = lamH[::-1], VH[:, ::-1]
    lamG = np.linalg.eigvalsh(G)[::-1]
    # projection of the age axis on the first r modes of H: P_r(D_a) = sum_m <D_a, B_m> B_m with B_m = V_m^T Dm / sqrt(lam_m)
    rec = {"layer": int(c["li"]) + 1, "n_ages": k, "sum_share_sq": float(np.sum(shares ** 2)),
           "G_offdiag_mean": float((G.sum() - k) / (k * (k - 1))), "H_energy_top": [float(x) for x in (np.cumsum(lamH) / lamH.sum())[:6]],
           "G_eigs": [float(x) for x in lamG[:6]], "err_total_rank": {}, "err_each_rank": {}}
    for r in (1, 2, 3, 4, 6):
        V = VH[:, :r]; B = (V.T @ Dm) / np.sqrt(lamH[:r])[:, None]            # (r, N) orthonormal modes
        coef = Dm @ B.T; Dhat = coef @ B
        rec["err_total_rank"][r] = float(np.linalg.norm(tot - Dhat.sum(0)) / nt)
        rec["err_each_rank"][r] = float(np.linalg.norm(Dm - Dhat) / np.linalg.norm(Dm))
    V1 = VH[:, 0]; alpha = Dm @ ((V1 @ Dm) / np.sqrt(lamH[0]))
    rec["alpha_first_mode"] = [float(x) for x in alpha / np.abs(alpha).max()]
    out.append(rec)
    print(f"layer {rec['layer']} ({k} ages): sum s^2 = {rec['sum_share_sq']:.3f}; mean pairwise cosine {rec['G_offdiag_mean']:.3f}; "
          f"H energy in top 1..4 modes {[round(x, 3) for x in rec['H_energy_top'][:4]]}", flush=True)
    print(f"   error of the TOTAL after keeping r age-modes: " + "  ".join(f"r{r}:{v:.3f}" for r, v in rec["err_total_rank"].items())
          + " | per-age error: " + "  ".join(f"r{r}:{v:.2f}" for r, v in rec["err_each_rank"].items()), flush=True)
    print("   first-mode loadings by age: " + " ".join(f"{x:+.2f}" for x in rec["alpha_first_mode"]), flush=True)
json.dump(out, open("../results/summaries/r9_age_gram.json", "w"), indent=1)
