"""Round 17 check (8 Oct): the SciMaster/Bohrium "Factorised Curvature-Birth kappa4 Carrier" PDF claims its falsifier
"confirms >= 0.70 at L12/15/16". The PDF contains no measured numbers; its script's Monte Carlo part is never used in the
comparison. This re-implements the script's deterministic CARRIER PREDICTION (transport_diagonal summed over births) and
scores it with (i) the script's own metric (raw kappa4, all neurons) and (ii) the project metric
(decisive |mu/sigma| < 1.5, pattern kappa4 / v^2).
Variants:
  as_written : the script's algebra, including its index conventions
               (U_chain[l-1] = diag(Phi_l) W[l-1]; (2,2) term diag(U2 K22 U2^T); pre-activation teacher arrays used as
               the "post" birth where no post-ReLU teacher exists; births at every layer 1..L-1 summed)
  fixed      : correct orientation (U_l = diag(Phi_l) W[l] maps z_l -> z_{l+1}, row convention; readout sum_ab U2_ai K22_ab U2_bi),
               post-ReLU slices only where they exist (L3/7/11/14/15), one-step births only (target = birth + 1 .. birth + 6)
Usage: python experiments/probes/r17_scimaster_check.py
"""
import numpy as np, json, os
from scipy.special import ndtr

PRE = "data/kaggle_out/kaggle_out_k4pub/k4teach"; POST = "data/kaggle_out/kaggle_out_k4post/k4post"
W = np.load(f"{PRE}/weights_seed0.npz")["W"].astype(np.float64)
n = W.shape[1]
T = {l: np.load(f"{PRE}/layer{l}_seed0.npz") for l in range(1, 17)}
mu = {l: T[l]["mean"].astype(np.float64) for l in T}; var = {l: np.diag(T[l]["C"]).astype(np.float64) for l in T}
d = {l: ndtr(mu[l] / np.sqrt(var[l])) for l in T}
phi = {l: np.exp(-0.5 * (mu[l] ** 2 / var[l])) / np.sqrt(2 * np.pi) for l in T}
curv = {l: phi[l] / np.sqrt(var[l]) for l in T}


def post(l):
    p = f"{POST}/layer{l}_post_seed0.npz"
    return np.load(p) if os.path.exists(p) else None


def contrib(Ut, k4d, K22, K31, C, cv, wrong):
    """The script's transport_diagonal algebra for one birth; Ut maps birth-side index (rows) to target index (cols)."""
    U4 = Ut ** 4; U2 = Ut ** 2
    d4 = (U4.T @ k4d) if not wrong else (U4 @ k4d)
    d22 = (np.einsum("ai,ab,bi->i", U2, K22, U2, optimize=True) if not wrong else np.diag(U2 @ K22 @ U2.T))
    UK31 = K31 @ Ut.T if wrong else None
    if wrong:   # the script's per-a loop, vectorised exactly as written
        d31 = (Ut ** 4 * (cv * np.diag(C))[None, :]).sum(1) + ((Ut ** 3) * cv[None, :] * UK31.T).sum(1)
    else:       # the same two terms with the birth index on the rows
        d31 = ((Ut ** 4) * (cv * np.diag(C))[:, None]).sum(0) + ((Ut ** 3) * cv[:, None] * (K31 @ Ut)).sum(0)
    return d4 + d22 + d31


def score(pred, L):
    k4t = T[L]["K4d"].astype(np.float64); v = var[L]; dec = np.abs(mu[L] / np.sqrt(v)) < 1.5
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    return dict(script_metric_raw_all=c(pred, k4t), project_metric=c((pred / v ** 2)[dec], (k4t / v ** 2)[dec]))


out = {}
Uw = {l: d[l][:, None] * W[l - 1] for l in range(1, 17)}            # as written: diag(Phi_l) W[l-1]
Uf = {l: d[l][:, None] * W[l] for l in range(1, 16)}                 # fixed: z_l -> z_{l+1}
for L in (12, 15, 16):
    pw = np.zeros(n)
    for b in range(1, L + 1):                                       # as written: births 1..L, U_tot = prod U_chain[b..L-1]
        Ut = np.eye(n)
        for l in range(b, L):
            Ut = Ut @ Uw[l + 1] if l + 1 <= 16 else Ut
        P = post(b); src = P if P is not None else T[b]
        pw += contrib(Ut, src["K4d"].astype(np.float64), src["K22"].astype(np.float64), src["K31"].astype(np.float64),
                      src["C"].astype(np.float64), curv[b], wrong=True)
    pf = np.zeros(n)
    for b in (3, 7, 11, 14, 15):                                     # fixed: true post-ReLU births within 6 ages
        if not (1 <= L - b <= 6):
            continue
        Ut = W[b].copy()                                             # h_b -> z_{b+1}
        for l in range(b + 1, L):
            Ut = Ut @ Uf[l]
        P = post(b)
        pf += contrib(Ut, P["K4d"].astype(np.float64), P["K22"].astype(np.float64) * (1 - np.eye(n)),
                      P["K31"].astype(np.float64) * (1 - np.eye(n)), P["C"].astype(np.float64), curv[b], wrong=False)
    out[L] = {"as_written": score(pw, L), "fixed_post_births": score(pf, L) if pf.any() else None}
    print(L, json.dumps(out[L]), flush=True)
json.dump(out, open("results/summaries/r17_scimaster_check.json", "w"), indent=1)
