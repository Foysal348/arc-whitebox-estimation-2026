"""kappa3 error profile and kappa4->kappa3 feed attribution (10 Oct 2026). Light; laptop.

Part 1 (writes results/summaries/chain_k3_error_profile.json and teacher_k3_noise_estimate.json):
  plain chain (r7 captures, D21 / D3) vs the 6e7-sample teacher (MLP #0) per layer: D21 off-diagonal and d3 relative errors,
  d3 decisive-neuron corr and norm ratio; teacher sampling noise from a Gaussian approximation
  Var[z_i^2 z_j] ~ 3 v_i^2 v_j (1 + 4 rho_ij^2), Var[z_i^3] ~ 15 v_i^3, N = 6e7.
Part 2 (writes results/summaries/k3feed/feed_attribution.json; needs the FEED_DUMP of the plain chain from
  kaggle/feedor/k1 in data/kaggle_out/feedor_1/feeddump and the assembled feeds in kaggle/feedor/feeds):
  chain feed = D21 - D21_nofeed; its corr / size vs the sampled exact feed; corr of (feed - chain feed) with the chain's slice
  residual; slice residual after replacing the chain feed; best scalar rescaling of the chain feed (beta) and of the sign-flipped
  Gaussian star.
Usage: python experiments/probes/k3_error_profile.py
"""
import os, json
import numpy as np

TEACH = "data/kaggle_out/kaggle_out_k4pub/k4teach"; N = 6e7
off = ~np.eye(1024, dtype=bool)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
prof, noise = {}, {}
for l in range(2, 17):
    c = np.load(f"data/dumps/r7_cap_L{l}.npz"); T = np.load(f"{TEACH}/layer{l}_seed0.npz")
    Y = T["Y21"].astype(np.float64); td3 = np.diag(Y).copy(); C = T["C"].astype(np.float64); v = np.diag(C)
    D = c["D21"].astype(np.float64); d3 = c["D3"].astype(np.float64); dec = np.abs(T["mean"] / np.sqrt(v)) < 1.5
    prof[l] = dict(D21_rel=rel(D[off], Y[off]), D21T_rel=rel(D.T[off], Y[off]), d3_rel=rel(d3, td3), d3_corr_dec=corr(d3[dec], td3[dec]),
                   d3_norm_ratio=float(np.linalg.norm(d3) / np.linalg.norm(td3)))
    var = 3 * np.outer(v ** 2, v) * (1 + 4 * C ** 2 / np.outer(v, v)) / N
    noise[l] = dict(teacher_noise_rel_D21=float(np.sqrt(var[off].sum()) / np.linalg.norm(Y[off])),
                    teacher_noise_rel_d3=float(np.sqrt((15 * v ** 3 / N).sum()) / np.linalg.norm(td3)))
    print(l, {k: round(x, 4) for k, x in {**prof[l], **noise[l]}.items()}, flush=True)
json.dump(prof, open("results/summaries/chain_k3_error_profile.json", "w"), indent=1)
json.dump(noise, open("results/summaries/teacher_k3_noise_estimate.json", "w"), indent=1)

DUMP, FEEDS = "data/kaggle_out/feedor_1/feeddump", "kaggle/feedor/feeds"
if os.path.isdir(DUMP) and os.path.isdir(FEEDS):
    att = {}
    for t in range(9, 16):
        d = np.load(f"{DUMP}/mlp0_layer{t}.npz"); D21 = d["D21"].astype(np.float64); ch = (D21 - d["D21_nofeed"].astype(np.float64))[off]
        Y = np.load(f"{TEACH}/layer{t}_seed0.npz")["Y21"].astype(np.float64)[off]; r = Y - D21[off]; ny = np.linalg.norm(Y)
        beta = float(r @ ch / (ch @ ch))
        row = dict(chain_feed_rel_Y21=float(np.linalg.norm(ch) / ny), slice_resid_rel=float(np.linalg.norm(r) / ny),
                   chain_scale_corr=corr(ch, r), chain_scale_beta=beta, chain_scale_resid=float(np.linalg.norm(r - beta * ch) / ny))
        for nm in ("exact", "bornJ6"):
            F = np.load(f"{FEEDS}/{nm}__layer{t}_seed0.npz")["F21"].astype(np.float64)[off]
            row[nm] = dict(corr_chain_feed=corr(ch, F), size_vs_chain_feed=float(np.linalg.norm(F) / np.linalg.norm(ch)),
                           corr_correction_resid=corr(F - ch, r), resid_after_replace=float(np.linalg.norm(r - (F - ch)) / ny))
        S = -np.load(f"{FEEDS}/star_J6__layer{t}_seed0.npz")["F21"].astype(np.float64)[off]; bs = float(r @ S / (S @ S))
        row["neg_star"] = dict(corr_chain_feed=corr(ch, S), corr_resid=corr(S, r), beta=bs, resid=float(np.linalg.norm(r - bs * S) / ny))
        att[t] = row
        print(t, {k: (round(x, 4) if isinstance(x, float) else {kk: round(xx, 4) for kk, xx in x.items()}) for k, x in row.items()}, flush=True)
    os.makedirs("results/summaries/k3feed", exist_ok=True)
    json.dump(att, open("results/summaries/k3feed/feed_attribution.json", "w"), indent=1)
