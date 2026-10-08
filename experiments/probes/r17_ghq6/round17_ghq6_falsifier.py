#!/usr/bin/env python3
"""
Round 17 ceiling falsifier for GHQ-6 (Gaussian Hypertree Quartic Bank).

Primary test:
  * Build the exact, UNTRUNCATED GHQ carrier (all births) from the stored 6e7
    teacher means/covariances and the true weights.
  * Build the cheap age<=6 version from the same carrier.
  * Estimate the true pre-activation kappa4 diagonal by an independent Monte
    Carlo run of the true MLP (float32 forward, float64 accumulation), N=2e6
    by default; also compare against stored teacher K4d when present.
  * Print decisive-neuron Pearson correlations at L12, L15, L16.

Pre-registered gate:
  PASS ceiling: all-history correlation >= 0.70 at L12/L15/L16.
  PASS cheap:   age<=6 correlation >= 0.70 at L12/L15/L16.
  KILL:         all-history correlation < 0.55 at all three layers.

Run from project root, e.g.:
  python experiments/probes/round17_ghq6_falsifier.py
  python experiments/probes/round17_ghq6_falsifier.py --device gpu
  python experiments/probes/round17_ghq6_falsifier.py --skip-mc   # quick, uses stored 6e7 teacher only

The script is self-contained and does not import estimator code.
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
import numpy as np

try:
    from scipy.special import ndtr
except Exception:
    ndtr = None

N = 1024
TARGETS = (12, 15, 16)
PRE = Path("data/kaggle_out/kaggle_out_k4pub/k4teach")


def _phi_cdf(x):
    if ndtr is not None:
        return ndtr(x)
    # Slow fallback, used only for 16 x 1024 values.
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in np.ravel(x)],
                    dtype=np.float64).reshape(np.shape(x))


def _phi_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _get(d, names):
    for k in names:
        if k in d.files:
            return np.asarray(d[k])
    raise KeyError(f"none of {names} in {d.files}")


def _maybe_get(d, names):
    for k in names:
        if k in d.files:
            return np.asarray(d[k])
    return None


def load_weights():
    candidates = [
        PRE / "weights_seed0.npz",
        Path("data/kaggle_out/kaggle_out_k4pub/k4teach/weights_seed0.npz"),
        Path("weights_seed0.npz"),
        Path("data/weights_seed0.npz"),
    ]
    for p in candidates:
        if p.exists():
            d = np.load(p, allow_pickle=False)
            W = _get(d, ("W", "weights"))
            if W.shape[0] < 16:
                raise ValueError(f"weights have shape {W.shape}")
            return np.asarray(W[:16], dtype=np.float32), p
    raise FileNotFoundError("weights_seed0.npz not found in expected locations")


def load_teachers():
    teachers = {}
    for l in range(1, 17):
        p = PRE / f"layer{l}_seed0.npz"
        d = _load_npz(p)
        mean = _get(d, ("mean", "mu", "K1")).astype(np.float64)
        C = _get(d, ("C", "cov", "covariance")).astype(np.float64)
        k4 = _maybe_get(d, ("K4d", "K4", "k4d", "kappa4", "K4_diag"))
        if k4 is not None:
            k4 = np.asarray(k4, dtype=np.float64).reshape(-1)
        teachers[l] = dict(mean=mean, C=C, k4=k4, path=p)
    return teachers


def responses(mean, C):
    var = np.clip(np.diag(C).astype(np.float64), 1e-12, None)
    sig = np.sqrt(var)
    alpha = mean / sig
    pdf = _phi_pdf(alpha)
    r1 = _phi_cdf(alpha)                  # E[ReLU'(Z)]
    r2 = pdf / sig                        # E[ReLU''(Z)] distributionally
    r3 = -alpha * pdf / var               # E[ReLU'''(Z)] distributionally
    return r1, r2, r3, alpha


def pearson(a, b, mask):
    a = np.asarray(a, np.float64)[mask]
    b = np.asarray(b, np.float64)[mask]
    a = a - a.mean(); b = b - b.mean()
    da = float(np.dot(a, a)); db = float(np.dot(b, b))
    if da <= 0 or db <= 0:
        return float("nan")
    return float(np.dot(a, b) / math.sqrt(da * db))


class Source:
    __slots__ = ("birth", "P", "A", "c", "H")
    def __init__(self, birth, P, A, c, H):
        self.birth = int(birth)
        self.P = P          # current POST-ReLU coordinates, float32
        self.A = A
        self.c = c          # fixed birth-coordinate star coefficient, float32
        self.H = H          # fixed symmetric path core, float32


def birth_source(layer, teacher):
    """Leading connected Gaussian-tree K4 birth at post-ReLU layer `layer`.

    Define response coefficients r1=Phi(alpha), r2=phi(alpha)/sigma,
    r3=-alpha*phi(alpha)/sigma^2 and C0=C with zero diagonal.

    A[j,r] = r1[j] C0[j,r]
    c[r]   = r3[r]
    H[r,t] = r2[r] C0[r,t] r2[t]

    The represented birth tensor is
      4 Sym sum_r c_r e_r tensor A_r^3
    +12 Sym sum_rt H_rt e_r tensor A_r tensor e_t tensor A_t.
    """
    mean, C = teacher["mean"], teacher["C"]
    r1, r2, r3, _ = responses(mean, C)
    C0 = C.copy()
    np.fill_diagonal(C0, 0.0)
    A = (r1[:, None] * C0).astype(np.float32)
    H = ((r2[:, None] * C0) * r2[None, :]).astype(np.float32)
    # Force exact symmetry against tiny teacher numerical asymmetry.
    H = (0.5 * (H + H.T)).astype(np.float32)
    P = np.eye(C.shape[0], dtype=np.float32)
    c = r3.astype(np.float32)
    return Source(layer, P, A, c, H)


def source_readout(src, Pz, Az):
    """Exact diagonal of this factorised source in the current preactivation coords."""
    # Star: 4 * sum_r c_r P_ir A_ir^3
    # The factor 4 is the number of labelled star trees.
    star = 4.0 * np.sum(
        Pz.astype(np.float64) * (Az.astype(np.float64) ** 3) * src.c.astype(np.float64)[None, :],
        axis=1,
    )
    # Path: 12 * x_i^T H x_i, x=P o A.
    # The factor 12 is the number of labelled path trees.
    X = (Pz * Az).astype(np.float32)
    # Use a dense product in the ceiling script. The submitted estimator can batch
    # the six active sources and use the same Strassen family as the chain.
    Y = X @ src.H
    path = 12.0 * np.sum(Y.astype(np.float64) * X.astype(np.float64), axis=1)
    return star + path


def build_carrier_predictions(W, teachers):
    """Run the exact factorised carrier from layer 1 and return all-history/age6 diagonals."""
    sources = []
    pred_all, pred6 = {}, {}
    for l in range(1, 17):
        # Existing sources are in post-ReLU coordinates of layer l-1.
        if l >= 2:
            WT = W[l-1].T  # W[l-1] maps h_{l-1} -> z_l (0-based W)
            q_all = np.zeros(N, dtype=np.float64)
            q_6 = np.zeros(N, dtype=np.float64)
            transported = []
            for s in sources:
                Pz = WT @ s.P
                Az = WT @ s.A
                if l in TARGETS:
                    q = source_readout(s, Pz, Az)
                    q_all += q
                    if (l - s.birth) <= 6:
                        q_6 += q
                transported.append((s, Pz, Az))
            if l in TARGETS:
                pred_all[l] = q_all
                pred6[l] = q_6
            # Linearised ReLU dressing to post h_l for continued memory.
            if l <= 15:
                r1, _, _, _ = responses(teachers[l]["mean"], teachers[l]["C"])
                d = r1.astype(np.float32)[:, None]
                for s, Pz, Az in transported:
                    s.P = d * Pz
                    s.A = d * Az
        # Birth at post-ReLU h_l; no need after final layer.
        if l <= 15:
            sources.append(birth_source(l, teachers[l]))
    return pred_all, pred6


def mc_k4(W, teachers, Nmc, batch, seed, use_gpu=False):
    """Independent true-MLP MC K4 diagonal at TARGETS, float32 forward / float64 sums."""
    xp = np
    cp = None
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
        except Exception as e:
            raise RuntimeError("--device gpu requested but CuPy is unavailable") from e
    Wd = xp.asarray(W, dtype=xp.float32)
    sums = {l: [np.zeros(N, np.float64) for _ in range(4)] for l in TARGETS}
    rng = np.random.default_rng(seed)
    done = 0
    t0 = time.time()
    while done < Nmc:
        b = min(batch, Nmc - done)
        x_np = rng.standard_normal((b, N), dtype=np.float32)
        x = xp.asarray(x_np) if use_gpu else x_np
        h = x
        for li in range(16):
            z = h @ Wd[li]
            layer = li + 1
            if layer in TARGETS:
                z_np = cp.asnumpy(z) if use_gpu else np.asarray(z)
                z64 = z_np.astype(np.float64)
                sums[layer][0] += z64.sum(axis=0)
                sums[layer][1] += (z64*z64).sum(axis=0)
                sums[layer][2] += (z64*z64*z64).sum(axis=0)
                sums[layer][3] += ((z64*z64)*(z64*z64)).sum(axis=0)
            h = xp.maximum(z, xp.float32(0.0))
        done += b
        if done == b or done == Nmc or done % max(batch, Nmc//10) == 0:
            print(f"MC {done}/{Nmc} ({time.time()-t0:.1f}s)", flush=True)
    out = {}
    for l in TARGETS:
        s1,s2,s3,s4 = [x / float(Nmc) for x in sums[l]]
        mu = s1
        var = s2 - mu*mu
        m4 = s4 - 4*mu*s3 + 6*(mu*mu)*s2 - 3*(mu**4)
        out[l] = m4 - 3*var*var
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2_000_000, help="MC samples (default 2e6)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=17017)
    ap.add_argument("--device", choices=("cpu","gpu"), default="cpu")
    ap.add_argument("--skip-mc", action="store_true", help="use stored 6e7 teacher only")
    args = ap.parse_args()

    W, wpath = load_weights()
    teachers = load_teachers()
    print("weights:", wpath, W.shape)
    print("Building exact all-history and age<=6 GHQ carrier ...", flush=True)
    t0 = time.time()
    pred_all, pred6 = build_carrier_predictions(W, teachers)
    print(f"carrier built in {time.time()-t0:.1f}s")

    stored = {}
    for l in TARGETS:
        if teachers[l]["k4"] is None:
            raise KeyError(f"No K4 diagonal key found in {teachers[l]['path']}")
        stored[l] = teachers[l]["k4"]

    mc = None
    if not args.skip_mc:
        mc = mc_k4(W, teachers, args.N, args.batch, args.seed, args.device=="gpu")

    print("\n=== Round 17 GHQ-6 ceiling gate ===")
    all_corrs, six_corrs = [], []
    for l in TARGETS:
        var = np.clip(np.diag(teachers[l]["C"]), 1e-12, None)
        decisive = np.abs(teachers[l]["mean"] / np.sqrt(var)) < 1.5
        # The stored 6e7 teacher is the primary gate (higher S/N); the fresh 2e6 MC
        # is an independent sanity check requested by the prompt.
        truth = stored[l]
        ca = pearson(pred_all[l], truth, decisive)
        c6 = pearson(pred6[l], truth, decisive)
        all_corrs.append(ca); six_corrs.append(c6)
        print(f"L{l}: n_dec={int(decisive.sum())}  all-history_vs_6e7={ca:+.4f}  age<=6_vs_6e7={c6:+.4f}", end="")
        if mc is not None:
            cmca = pearson(pred_all[l], mc[l], decisive)
            cmc6 = pearson(pred6[l], mc[l], decisive)
            cs = pearson(stored[l], mc[l], decisive)
            print(f"  all-vs-MC2e6={cmca:+.4f}  age6-vs-MC2e6={cmc6:+.4f}  6e7-vs-MC2e6={cs:+.4f}")
        else:
            print("  MC skipped")

    pass_ceiling = all(c >= 0.70 for c in all_corrs)
    pass_cheap = all(c >= 0.70 for c in six_corrs)
    kill = all(c < 0.55 for c in all_corrs)
    print("\nPRE-REGISTERED VERDICT:")
    if pass_ceiling and pass_cheap:
        print("PASS: exact ceiling and age<=6 cheap carrier both clear 0.70 at all targets.")
    elif pass_ceiling:
        print("CEILING PASS / CHEAP FAIL: representation works, age truncation does not.")
    elif kill:
        print("KILL: all-history carrier is <0.55 at L12/L15/L16.")
    else:
        print("MIXED: ceiling neither passes all 0.70 gates nor hits the all-three <0.55 kill rule.")
    print("Raw all-history:", [round(x,4) for x in all_corrs])
    print("Raw age<=6:   ", [round(x,4) for x in six_corrs])


if __name__ == "__main__":
    main()
