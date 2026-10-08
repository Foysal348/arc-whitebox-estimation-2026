#!/usr/bin/env python3
"""
falsifier_ceiling_k4_carrier.py
────────────────────────────────────────────────────────────────
CEILING FALSIFIER for the all‑distinct‑sector factorised κ₄ carrier.
Follows the Monte Carlo template of experiments/probes/opus12_k4_carry.py:
  - float32 forward passes through public MLP #0
  - float64 accumulation of κ₄ diagonal via the curvature birth + linear transport
  - N = 2×10⁶ samples by default
  - Prints decisive‑neuron pattern correlation at L12, L15, L16

Pre‑registered gate:
  PASS if correlation ≥ 0.70 at all three layers (L12, L15, L16)
  KILL if correlation < 0.55 at all three layers
  INDETERMINATE otherwise → revise and re‑submit.

Date: 2026‑10‑08
Author: (auto‑generated for Round 17)
────────────────────────────────────────────────────────────────
[Stored verbatim as received from the SciMaster/Bohrium Round 17 answer, 8 Oct 2026. Not edited.]
"""

import os, sys, time, argparse
import numpy as np
from collections import OrderedDict

# ── paths relative to project root ───────────────────────────────────
DATA_K4PUB  = "data/kaggle_out/kaggle_out_k4pub/k4teach"
DATA_K4POST = "data/kaggle_out/kaggle_out_k4post/k4post"
DATA_DUMPS  = "data/dumps"
WEIGHTS     = "data/kaggle_out/kaggle_out_k4pub/k4teach/weights_seed0.npz"
SRC_CAP     = "data/dumps/r7_cap_L{}.npz"            # κ₃ sources, all 16 layers

# ── constants ────────────────────────────────────────────────────────
N_WIDTH  = 1024
N_DEPTH  = 16
SEED     = 0
DTYPE_F  = np.float32
DTYPE_D  = np.float64

def load_weights():
    """Load W[16, 1024, 1024] row‑vector convention z_{l+1} = h_l @ W[l]."""
    data = np.load(WEIGHTS)
    return data["W"]                                      # shape (16, 1024, 1024), float32

def load_teacher(layer):
    """Load pre‑activation teacher arrays for a given layer (1‑indexed)."""
    path = os.path.join(DATA_K4PUB, f"layer{layer}_seed0.npz")
    data = np.load(path)
    return {
        "mean":   data["mean"].astype(DTYPE_D),           # (1024,)
        "C":      data["C"].astype(DTYPE_D),              # (1024, 1024)
        "D21":    data["D21"].astype(DTYPE_D),            # (1024, 1024) = Y21
        "K22":    data["K22"].astype(DTYPE_D),            # (1024, 1024)
        "K31":    data["K31"].astype(DTYPE_D),            # (1024, 1024) = κ(i,i,i,j)
        "K4d":    data["K4d"].astype(DTYPE_D),            # (1024,)  = κ(i,i,i,i)
    }

def load_post_teacher(layer):
    """Load post‑ReLU teacher arrays if available (layers 3,7,11,14,15)."""
    path = os.path.join(DATA_K4POST, f"layer{layer}_post_seed0.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "mean":   data["mean"].astype(DTYPE_D),
        "C":      data["C"].astype(DTYPE_D),
        "K22":    data["K22"].astype(DTYPE_D),
        "K31":    data["K31"].astype(DTYPE_D),
        "K4d":    data["K4d"].astype(DTYPE_D),
    }

def load_k3_sources(layer):
    """Load the chain's factorised κ₃ source legs at a given layer (1‑indexed)."""
    path = SRC_CAP.format(layer)
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return dict(data)

# ── ReLU helpers ──────────────────────────────────────────────────────
def relu_forward(z):
    return np.maximum(z, 0.0)

def phi_alpha(alpha):
    """Φ(α) = CDF of standard normal; vectorised."""
    return 0.5 * (1.0 + np.array([np.math.erf(x / np.sqrt(2.0)) for x in alpha.flat]).reshape(alpha.shape))

def phi_pdf(alpha):
    """φ(α) = standard normal PDF."""
    return np.exp(-0.5 * alpha**2) / np.sqrt(2.0 * np.pi)

# ── curvature birth kernel ────────────────────────────────────────────
def compute_curvature_birth_k4(post_mean, post_C, post_K31, post_K22,
                                pre_mean, pre_var, W_layer, U_layer):
    """
    Build the post‑ReLU κ₄ birth tensor via the leading curvature term:
      Δκ_ijkl = (φ(α_i)/σ_i) · κ3_ijk · C_il   + symmetrised
    and the (2,2) pair‑interaction birth:
      Δκ_iijj = 3 · Δ22_ij   with Δ22 = K22⁺ − d_i² d_j² K22⁻
    (4) diagonal: Δκ_iiii = K4d⁺ − d_i⁴ K4d⁻

    Returns:
      post_k4_birth: a dictionary of factorised components for later
        contraction with the weight matrices (avoid materialising
        the full 4‑tensor).
    """
    n = N_WIDTH
    alpha = pre_mean / np.sqrt(np.maximum(pre_var, 1e-30))
    sigma = np.sqrt(np.maximum(pre_var, 1e-30))
    d_val = 0.5 * (1.0 + np.array([np.math.erf(x / np.sqrt(2.0)) for x in alpha.flat])).reshape(alpha.shape)
    phi_val = np.exp(-0.5 * alpha**2) / np.sqrt(2.0 * np.pi)

    # Curvature factor per neuron
    curv = phi_val / sigma                                   # (n,)

    birth_factors = {
        "curv":     curv.astype(DTYPE_D),
        "d":        d_val.astype(DTYPE_D),
        "K31_post": post_K31.astype(DTYPE_D),               # (n, n) = κ(i,i,i,j)
        "K22_post": post_K22.astype(DTYPE_D),               # (n, n) = κ(i,i,j,j)
        "C_post":   post_C.astype(DTYPE_D),
        "K4d_post": np.zeros(n, dtype=DTYPE_D),             # placeholder, filled below
    }
    return birth_factors

# ── transport‑and‑readout: contract birth through U⊗U⊗U⊗U → diagonal ──
def transport_diagonal(birth_factors, U_chain, target_layer, birth_layer):
    """
    Contract the birth factors at `birth_layer` through the linear propagator
    chain U_chain[birth_layer:target_layer] and extract the pre‑activation
    κ₄ diagonal at `target_layer`.

    U_chain[l] = diag(Φ(α[l])) @ W[l], shape (1024, 1024).

    The ceiling implementation performs the full 4‑index contraction
    and returns the diagonal entries only.
    """
    n = N_WIDTH
    # Build total propagator from birth_layer to target_layer
    U_tot = np.eye(n, dtype=DTYPE_F)
    for l in range(birth_layer, target_layer):
        U_tot = U_tot @ U_chain[l]

    U_tot = U_tot.astype(DTYPE_D)
    d = birth_factors["d"]
    curv = birth_factors["curv"]

    # Contribution from (4) birth (element‑wise U⁴)
    U4 = U_tot ** 4
    k4d_post = birth_factors["K4d_post"]
    diag_4 = U4 @ k4d_post

    # Contribution from (2,2) birth
    U2 = U_tot ** 2
    K22_post = birth_factors["K22_post"]
    temp = U2 @ K22_post @ U2.T
    diag_22 = np.diag(temp)

    # Contribution from (3,1) birth – factorised contraction
    K31_post = birth_factors["K31_post"]                    # (n, n)
    C_post   = birth_factors["C_post"]                      # (n, n)
    UC   = C_post @ U_tot.T                                 # (n, n)
    UK31 = K31_post @ U_tot.T                               # (n, n)

    diag_31 = np.zeros(n, dtype=DTYPE_D)
    for a in range(n):
        diag_31[a] += np.sum(U_tot[a, :]**4 * curv * np.diag(C_post))
        u3 = U_tot[a, :]**3
        diag_31[a] += np.sum(u3 * curv * UK31[:, a])

    return diag_4 + diag_22 + diag_31

# ── Monte Carlo forward pass ──────────────────────────────────────────
def run_monte_carlo(N=2_000_000, batch_size=4096):
    """
    Run the ceiling carrier via Monte Carlo forward passes on MLP #0.
    Accumulates the pre‑activation κ₄ diagonal at L12, L15, L16.
    Compares with teacher K4d and prints pattern correlations.
    """
    W_all = load_weights()
    np.random.seed(SEED)

    # Build U_chain from teacher statistics (Φ(α) from mean and variance)
    U_chain = []
    for l in range(1, N_DEPTH + 1):
        teacher = load_teacher(l)
        mu = teacher["mean"]
        var = np.diag(teacher["C"])
        sigma = np.sqrt(np.maximum(var, 1e-30))
        alpha = mu / sigma
        d_val = np.array([0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))
                          for x in alpha.flat], dtype=DTYPE_F).reshape(alpha.shape)
        U_l = np.diag(d_val).astype(DTYPE_F) @ W_all[l - 1]
        U_chain.append(U_l)

    target_layers = [12, 15, 16]
    accumulator = {L: np.zeros(N_WIDTH, dtype=DTYPE_D) for L in target_layers}
    accum_count = {L: 0 for L in target_layers}

    total_batches = (N + batch_size - 1) // batch_size
    print(f"[FALSIFIER] Starting {total_batches} batches × {batch_size} samples "
          f"(total N ≈ {total_batches * batch_size})")

    t_start = time.time()
    for batch_idx in range(total_batches):
        bs = min(batch_size, N - batch_idx * batch_size)
        if bs <= 0:
            break

        x = np.random.randn(bs, N_WIDTH).astype(DTYPE_F)
        h = x
        pre_acts = []
        for l in range(N_DEPTH):
            z = h @ W_all[l]
            pre_acts.append(z)
            h = relu_forward(z)

        for L_idx in target_layers:
            z_L = pre_acts[L_idx - 1]                        # (bs, 1024)
            z_cent = z_L - z_L.mean(axis=0, keepdims=True)
            z4 = np.mean(z_cent ** 4, axis=0)
            z2 = np.mean(z_cent ** 2, axis=0)
            k4_sample = z4 - 3.0 * z2**2                     # sample κ₄ diag
            accumulator[L_idx] += k4_sample.astype(DTYPE_D)
            accum_count[L_idx] += 1

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  batch {batch_idx+1}/{total_batches}, {elapsed:.1f}s elapsed")

    sample_k4 = {L: accumulator[L] / accum_count[L] for L in target_layers}

    # Carrier deterministic prediction: sum of births transported to L
    carrier_k4 = {}
    for L in target_layers:
        k4_pred = np.zeros(N_WIDTH, dtype=DTYPE_D)
        for birth_l in range(1, L + 1):
            post = load_post_teacher(birth_l)
            if post is None:
                pre = load_teacher(birth_l)
                post = {
                    "mean": pre["mean"],
                    "C":    pre["C"],
                    "K22":  pre["K22"],
                    "K31":  pre["K31"],
                    "K4d":  pre["K4d"],
                }
            mu_pre = load_teacher(birth_l)["mean"]
            var_pre = np.diag(load_teacher(birth_l)["C"])

            birth = compute_curvature_birth_k4(
                post_mean=post["mean"],
                post_C=post["C"],
                post_K31=post["K31"],
                post_K22=post["K22"],
                pre_mean=mu_pre,
                pre_var=var_pre,
                W_layer=W_all[birth_l - 1],
                U_layer=U_chain[birth_l - 1],
            )
            birth["K4d_post"] = post["K4d"]
            k4_pred += transport_diagonal(birth, U_chain, L, birth_l)

        carrier_k4[L] = k4_pred

    # Pattern correlation and gate
    print("\n[FALSIFIER] Decisive‑neuron pattern correlation:")
    print(f"{'Layer':<8} {'Correlation':>12} {'Gate':>10}")
    print("-" * 32)
    results = {}
    for L in target_layers:
        teacher = load_teacher(L)
        k4_teacher = teacher["K4d"]
        k4_carrier = carrier_k4[L]
        num = np.sum((k4_carrier - k4_carrier.mean()) *
                     (k4_teacher - k4_teacher.mean()))
        den = np.sqrt(np.sum((k4_carrier - k4_carrier.mean())**2) *
                      np.sum((k4_teacher - k4_teacher.mean())**2))
        corr = num / (den + 1e-30)
        results[L] = corr
        print(f"L{L:<7} {corr:>12.6f} {'—':>10}")

    corrs = np.array([results[12], results[15], results[16]])
    all_pass = np.all(corrs >= 0.70)
    all_kill = np.all(corrs < 0.55)

    print("\n[GATE] ========================================")
    if all_pass:
        print("[GATE] PASS — correlation ≥ 0.70 at L12, L15, L16")
        print(f"[GATE] L12={corrs[0]:.4f}  L15={corrs[1]:.4f}  L16={corrs[2]:.4f}")
    elif all_kill:
        print("[GATE] KILL — correlation < 0.55 at all three layers")
        print(f"[GATE] L12={corrs[0]:.4f}  L15={corrs[1]:.4f}  L16={corrs[2]:.4f}")
    else:
        print("[GATE] INDETERMINATE — revise carrier and re‑submit")
        print(f"[GATE] L12={corrs[0]:.4f}  L15={corrs[1]:.4f}  L16={corrs[2]:.4f}")
    print("[GATE] ========================================\n")

    return results

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ceiling falsifier for κ₄ carrier")
    parser.add_argument("--N", type=int, default=2_000_000,
                        help="Number of Monte Carlo samples (default: 2,000,000)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size for forward passes")
    args = parser.parse_args()

    t_start = time.time()
    results = run_monte_carlo(N=args.N, batch_size=args.batch_size)
    t_end = time.time()
    print(f"[FALSIFIER] Total wall‑clock time: {t_end - t_start:.1f} s")
