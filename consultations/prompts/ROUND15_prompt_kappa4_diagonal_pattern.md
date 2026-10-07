# Round 15 — ONE question: a cheap estimator of the PATTERN of the deep κ₄ diagonal

*(Answer in English, at most 4 pages. Give ONE concrete estimator (at most two), its exact cost at the real shapes, and ONE
falsifier on our stored data. Mark every number as derived, estimated or quoted. Do not re-propose anything in the dead list
unless you show why its kill was too narrow.)*

## Setting (short)

Bias-free ReLU MLP, width n = 1024, depth L = 16, He weights, input x ~ N(0, I); estimate the final-layer post-ReLU means.
Score = final MSE × FLOPs / 2⁴¹ (1 u = 2³¹ FLOPs ≈ one 1024³ product; a level-5 Strassen product ≈ 0.51 u). Our estimator is the
public factorised third-cumulant chain (mean, dense covariance C, κ₃ as a sum over per-ReLU "sources" with dense legs for young
sources and a shared low-rank basis for old ones). Its κ₄ is memoryless: at each layer a core diag(g) + λ_l C_pre,off is
regenerated and only its diagonal g is carried by (W∘W). The κ₄ diagonal enters ONLY the univariate Edgeworth mean closure.
Graded now: raw MSE 2.20e-8 at utilisation 0.187 (score 4.1e-9). Grader limit: ≤ ~18,000 metered operations per network.

## What a perfect κ₄ diagonal is worth (measured: coupled rollouts from layer 1, public MLP #0, teacher = 6×10⁷-sample MC)

| teacher κ₄ diagonal injected into the mean closure | final raw MSE | share of the full gain |
|---|---|---|
| none (the chain) | 2.290e-8 | 0% |
| all layers | 1.609e-8 | 100% |
| layers ≥ 8 only | 1.707e-8 | 86% |
| layers ≥ 12 only | 1.813e-8 | 70% |
| layers ≤ 11 only | 2.064e-8 | 33% |
| 50% blend (chain + teacher)/2, all layers | 1.789e-8 | 74% |
| 75% teacher + 25% chain | 1.658e-8 | 93% |
| **teacher PATTERN rescaled to the chain's per-layer mean level** | **1.551e-8** | **108%** |
| chain pattern rescaled to the teacher's level | 2.783e-8 | −72% (worse) |
| teacher (2,2) slice only / (3,1) slice only / all slices | 2.796e-8 / 2.353e-8 / 1.738e-8 | — |

So: the whole gain is the per-neuron PATTERN of the diagonal (the chain's level is fine); half-way accuracy already gives
three quarters of the gain; layers ≥ 8 carry 86%. The full gain would put us at the leaders' raw MSE (1.5-1.6e-8).

## Measured facts about that pattern

- The chain's diagonal pattern correlates only 0.26-0.37 with the teacher at layers 7-16 (decisive neurons |μ/σ| < 1.5).
- Exact LINEAR carry of the true full κ₄ tensor of the previous layer, y = (Φ(α) ∘ (z − μ)) W, gives pattern corr 0.77-0.80
  (MLP #0) at layers 12-16: the pattern is carried memory. Fresh ReLU physics from a Gaussian input is 1-4% of the deep level.
- Memory horizon: erasing non-Gaussian history h layers back recovers pattern corr 0.41 (h = 2), 0.59 (h = 3), 0.77 (h = 6).
- The next diagonal is ~90% the (2,2) pair term 3 Σ_{a≠b} W_ai² W_bi² K22(h)_ab by norm, yet transporting the TRUE post-ReLU
  (4) and (2,2) slices one step gives the right level but pattern corr only 0.24-0.59: the pattern lives in the
  all-distinct (2,1,1) and (1,1,1,1) sectors.
- Parity: the pattern is not predictable from statistics even in the weights (per-neuron features: corr ≤ 0.04).

## Measured on 7 Oct: what the chain's OWN per-layer features already give (our baseline for your answer)

A linear map from per-neuron features the chain already has at the layer, namely
- s3sq = (κ3_ii/v^1.5)²,
- the row and column energy of the standardised D21,
- the correlation-row energy Σ_j C_pre,ij²/(v_i v_j),
- |μ/σ|,
- the chain's own κ4/v², and
- 1,

fitted on pooled layers ≥ 6 (decisive neurons), rescaled to the chain's layer-mean level and fed to the mean closure only, gives:

- Pattern corr 0.41-0.56 at L6-15. The chain's own pattern corr is 0.24-0.41. The fit transfers across networks: fitted on the synthetic net, it gives 0.43-0.56 on MLP #0.
- End to end, with zero FLOPs and +310 metered ops:
  - MLP #0: raw 2.196e-8 → 2.038e-8 (−7.2%).
  - Held-out synthetic net: 1.886e-8 → 1.834e-8 (−2.7%).
  - Submitted to the grader.
- These additions do NOT help (≤ +0.03 corr in sample, 0 or negative held out):
  - squares, pairwise products and the log of these features;
  - the previous layer's same features carried by (W∘W)ᵀ, unweighted or weighted by Φ(α), Φ², Φ⁴.
- So per-layer chain features saturate at corr ~0.5. Going to ≥ 0.7 needs NEW carried information (a state that is not one
  of these per-neuron summaries). Your estimator must beat this 0.5 baseline on the same decisive neurons.

## Measured dead (do not propose)

Scale-mixture / Sym(C⊗C) κ₄ predictors (corr ≤ 0.09); learned per-neuron or 13-feature regressions across networks;
fresh-physics κ₄ kernels; a young-source κ₄ carrier over 2-3 layers (out of sample 0.210 → 0.203 error); κ₄ riding the κ₃ source
legs (out of sample ×1.05); a carried pair-response mode (×0.98-1.00); SSC response closure (×1.006-1.023); the exchange term
3 Y C⁻¹ Yᵀ with Y the κ₃ (2,1) slice (pattern corr −0.07 … −0.23 at L12-16, its level 10× the truth); role-split or per-layer
refits of λ; cutting κ₃ cost and recovering with κ₄ (κ₄ helps less the cheaper κ₃ gets).

## The question

**Design an estimator of the deep κ₄ diagonal's per-neuron pattern (layers ≥ 8) that reaches pattern correlation ≥ ~0.7
with the truth (a 50-75% blend-equivalent) for ≤ ~20-30 u over the whole network and ≤ ~1,000 extra metered operations**,
using quantities our chain already carries (μ, C, the κ₃ sources' dense legs and shared-basis factors, the (2,1) slice D21
and κ₃ diagonal D3 at each layer, the regenerated core). The level stays the chain's (rescale per layer).

Deliverable: (1) the state / readout with shapes and exact equations; (2) cost per layer in dense products and metered ops;
(3) why it should carry the ≥ 6-layer memory and the all-distinct sectors; (4) ONE falsifier on our stored data (per layer: the
teacher's mean, C, D21 slice, K22, K31 and κ₄ diagonal on all 16 layers of MLP #0; post-ReLU teachers at layers 3/7/11/14/15;
dense captures of the chain's κ₃ sources at layers 7/11/14/15; the weights) with gate: pattern corr ≥ 0.7 at L12-16, or
≥ 0.5-blend-equivalent G4-weighted error.
