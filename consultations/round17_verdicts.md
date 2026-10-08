# Round 17 verdicts (8 Oct 2026)

Both runs below used Kaggle GPU (account foysalemonshanto) on public MLP #0, scored with the project metric:
- decisive neurons |μ/σ| < 1.5 from the 6e7-sample teacher;
- pattern = corr(κ4 / v_t², K4d_t / v_t²).

## 1. Claude's hub-pair Q1 carrier: KILL (pre-registered gate)

Design: the Q1 C·T curvature birth at each ReLU, carried linearly by U for up to 6 ages and read out as the pre-activation diagonal. The κ3 contraction is computed exactly from samples, so this is the generous version.

Script: `experiments/probes/r17_claude/r17_q1_hubpair_falsifier.py`, the author's version with our metric added. Results: `results/summaries/r17_q1_hubpair.json`, N = 2e6. MC sanity: the sampled κ4 vs the teacher is 0.936 / 0.960 / 0.964.

| config | L12 | L15 | L16 | level ratio |
|---|---|---|---|---|
| ceiling Q1 (H = 6, r = n) | 0.442 | 0.432 | 0.431 | 0.84-0.92 |
| Q1 + Gaussian path | 0.401 | 0.386 | 0.428 | 1.8-1.95 |
| Q1, H = 4 | 0.473 | 0.447 | 0.454 | 0.62-0.69 |
| Q1, H = 2 | 0.420 | 0.392 | 0.439 | 0.32-0.41 |
| Q1, H = 6, r = 64 | 0.255 | 0.203 | 0.361 | ~0 |
| Q1, H = 4, r = 32 | 0.096 | 0.126 | 0.255 | ~0 |

The author's own metric (raw κ4, top-128 neurons by φ|κ4|/σ³) gives 0.80-0.87. It is not comparable: the raw diagonal contains v², which the chain already gets right.

## 2. Family ceiling: which Hermite order of the ReLU must the births carry? (our own test)

Method:
- Monte Carlo with the ReLU replaced by its Hermite expansion around the teacher's per-layer μ, σ.
- "gauss_h6": the state 6 layers before the target is a Gaussian restart N(μ, C) of the teacher, so all older non-Gaussian history is erased; this is the best a 6-age carrier could do.
- "true_hX": the true state X layers back.

Script: `experiments/probes/r17_hermite_family_ceiling.py`. Results: `results/summaries/r17_hermite_family_ceiling_A.json`, N = 4e6.

| mode | L12 | L15 | L16 |
|---|---|---|---|
| true ReLU (sanity) | 0.967 | 0.973 | 0.980 |
| Gaussian restart 6 back, true ReLU | 0.859 | 0.820 | 0.818 |
| Gaussian restart 6 back, Hermite K = 2 | 0.548 | 0.449 | 0.530 |
| **Gaussian restart 6 back, Hermite K = 3** | **0.848** | **0.799** | **0.817** |
| true state 1 back, K = 2 (the quoted QUAD) | 0.850 | 0.874 | 0.881 |
| true state 3 back, K = 2 | 0.687 | 0.688 | 0.738 |
| true state 6 back, K = 2 | 0.527 | 0.477 | 0.578 |
| true state 6 back, K = 3 | 0.915 | 0.927 | 0.949 |

Two notes:
- Compounding K = 2 over all 16 layers diverges (level ×1e12); K = 2 over h layers degrades with h.
- The memory-horizon reference (Gaussian restart 6 back, true ReLU) is 0.82-0.86 here, slightly above the 0.77 quoted earlier.

## Verdict

- **The second-order (Section B) birth family is dead at the ceiling.** Even with exact cumulants and unlimited sectors it reaches only 0.45-0.55 over a 6-layer horizon. This is consistent with Q1 at 0.43-0.47.
- **Third-order births pass with a wide margin.** Adding the CUBIC Hermite term c3 He3, with c3 = φ(α) He1(−α)/3! = −α φ(α)/6, reaches 0.80-0.85. That equals the true ReLU at the same horizon.
- The next design target is a factorised carrier whose births come from the cubic response of each ReLU.
- Set B (running) asks two further questions:
  - is the quadratic term needed at all (orders 0, 1, 3)?
  - how short can the horizon be (h = 4, 3, 2)?

## 3. Family ceiling, set B: which orders and which horizon? (N = 4e6)

| mode (Gaussian restart h back) | L12 | L15 | L16 |
|---|---|---|---|
| h = 6, orders 0, 1, 3 (no quadratic) | 0.511 | 0.498 | 0.469 (level negative) |
| h = 6, K = 4 | 0.865 | 0.816 | 0.822 |
| h = 4, K = 3 | 0.749 | 0.687 | 0.711 |
| h = 4, true ReLU | 0.753 | 0.702 | 0.710 |
| h = 3, K = 3 | 0.652 | 0.588 | 0.643 |
| h = 3, true ReLU | 0.663 | 0.600 | 0.647 |
| h = 2, K = 3 | 0.517 | 0.436 | 0.485 |
| h = 2, true ReLU | 0.525 | 0.434 | 0.494 |

Findings:
- Both the quadratic and the cubic Hermite pieces are needed.
- K = 3 is sufficient: K = 4 adds ≤ 0.02.
- K = 3 tracks the true ReLU at every horizon.
- The 0.70 gate needs a horizon of about 5-6 layers: 4 is borderline (L15 0.687) and 3 fails.

## 4. Cubic-piece identities (verified exactly, `experiments/probes/r17_curvature_identities.py`)

    κ(u_a³, u_b, u_c, u_d) = κ6_aaabcd + 3 v_a K_abcd + 3 (C_ab K_aacd + C_ac K_aabd + C_ad K_aabc)
                             + 3 (T_abc T_aad + T_abd T_aac + T_acd T_aab) + 6 C_ab C_ac C_ad
    κ(u_a³ − 3 v_a u_a, u_b, u_c, u_d) = same without the 3 v_a K_abcd term

| check | max relative error |
|---|---|
| κ(u_a³, …) identity | 8.1e-16 |
| He3 form | 6.3e-16 |
| Gaussian-star readout 24 Σ_a c_a W_ai ((C U)_ai)³ | 9.0e-15 |

## 5. SciMaster/Bohrium "Factorised Curvature-Birth κ4 Carrier" PDF (25 pages, "AI Generated"): claims unsupported, KILL

Source: `Architecture Analysis/Round 17 A Factorised Curvature-Birth κ₄ Carrier ... ≤ 30 u Overhead.pdf`.

**Claims.** The script "confirms ≥ 0.70 at L12/15/16 and passes the gate", at a total cost of ~0.023 u (61 ops). The PDF contains no measured correlations, no run log and no attachments.

**Design problems**
- The births are second-order only: the PDF itself writes h − E h = Φ z̃ + (φ/σ)(z̃² − σ²)/2 + O(z̃³). By the family ceiling (§2-3) any such carrier is capped near 0.5.
- Term A, (φ/σ)_a ⊗ u_s ⊗ C_post, does not have the derived Q1 structure g_a C_ab T_acd: the curvature index is decoupled from the κ3 tensor.
- Term C has a coefficient η fitted to the teacher.
- The 0.023 u cost ignores the dense n × n post-ReLU slices (K22, K31, C) its own birth uses.

**Script problems**
- The Monte Carlo κ4 it accumulates is never used.
- The "carrier" is the TEACHER's slices transported. Where no post-ReLU teacher exists, it uses PRE-activation arrays as the post-ReLU birth.
- It sums births at every layer 1..L, which double counts.
- The (2,2) contraction is diag(U2 K22 U2ᵀ), i.e. the wrong orientation.
- U_chain[l−1] = diag(Φ_l) W[l−1] puts Φ on the wrong side.
- It scores raw κ4 on all neurons.

**Measured** (`experiments/probes/r17_scimaster_check.py`, `results/summaries/r17_scimaster_check.json`):

| | L12 | L15 | L16 |
|---|---|---|---|
| script as written, its own metric | −0.257 | −0.220 | −0.316 |
| script as written, project metric | 0.182 | 0.061 | 0.250 |
| orientation fixed, true post-ReLU births only, project metric | 0.392 | 0.321 | 0.397 |

**Verdict.** KILL. The fixed version sits at the known "true slices carried" level of 0.30-0.45.

Addendum (8 Oct): we stored the authors' full script verbatim at `experiments/probes/r17_scimaster/falsifier_ceiling_k4_carrier.py`.
- As written it does not run: `KeyError: 'D21'`, because the teacher key is Y21, as the prompt stated; `np.math` is also gone in NumPy 2. So it was never executed by its authors.
- With only those two names fixed (`falsifier_ceiling_k4_carrier_minfix.py`), the script's OWN output is L12 −0.2572, L15 −0.2202, L16 −0.3156, `[GATE] KILL` (`results/summaries/r17_scimaster_minfix_run.txt`). This reproduces our re-implementation exactly.

## 6. GHQ-6, Gaussian Hypertree Quartic Bank (Round 17)

Source: `Architecture Analysis/Round 17 - Gaussian Hypertree Quartic Bank.pdf` and `round17_ghq6_falsifier.py`, run unmodified via `experiments/probes/r17_ghq6/run_ghq6_project_metric.py`; results in `results/summaries/r17_ghq6.json`.

**Design.** The same Gaussian-tree family as HTQB:
- star 4 Σ c_r P A³;
- path 12 xᵀ H x, with H = diag(r2) C0 diag(r2);
- legs P and A carried by the linearised map, births at every layer.

The author's metric is the raw κ4 correlation on decisive neurons.

| | L12 | L15 | L16 |
|---|---|---|---|
| author metric, all history | −0.119 | −0.193 | −0.031 |
| author metric, age ≤ 6 | −0.113 | −0.191 | −0.050 |
| project metric, all history | 0.559 | 0.471 | 0.499 |
| project metric, age ≤ 6 | 0.561 | 0.467 | 0.481 |

**Verdict.** KILL by its own pre-registered gate (author metric < 0.55 at all three). On the project metric it is MIXED at 0.47-0.56, and dominated by HTQB (0.56-0.64), which is the refined version of the same trees.
