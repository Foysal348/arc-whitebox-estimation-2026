# Round 7 (Astra) — the whole error is three numbers per neuron. How do we compute them better, cheaply?

*(Paste this whole file. Answer in English. Prefer algebra, explicit FLOP counts in units of n³ and falsifiers we can
run on stored numpy arrays in under an hour. Please return your answer as a PDF.)*

## Setting

Bias-free ReLU MLP, width n = 1024, depth 16, weights i.i.d. N(0, 2/n), input x ~ N(0, I). Predict the final-layer
post-ReLU means. Score = final-layer MSE × max(0.1, FLOPs / 2⁴¹); 1 unit = 2³¹ FLOPs = one dense n³ product (a level-5
Strassen product costs 0.51 u). Our estimator is the public factorised K = 3 cumulant chain (504aldo's V29): per layer
it propagates the pre-activation mean μ, covariance C, the third cumulant as a **sum over sources** (a source born at
ReLU layer b is Sym(Σ_r e_r ⊗ e_r ⊗ a_r), a_b = diag(Φ)·C_off, transported **linearly** by T = Wᵀ diag(Φ) ⋯), and a
memoryless κ₄ regeneration (G_pre_off = λ_l C_pre_off, diagonal dG = (W∘W) g_prev + λ(var − (W∘W) var_prev)). The ReLU
closure is a Wick/Edgeworth program on the bivariate cumulants of each pair. Cost 260 u → graded raw MSE 2.1e-8 at
0.244 × budget. Leaders: raw 1.64–1.9e-8 at 0.13–0.16 × budget.

## What we measured this week (6e7-sample Monte-Carlo teacher of one public network, pre- and post-ReLU statistics at every layer; 1e9-sample truth)

1. **The mean closure is exact.** Given the true marginal cumulants (μ, var, κ₃ᵢᵢᵢ, κ₄ᵢᵢᵢᵢ) of each pre-activation,
   the Edgeworth mean E[relu(zᵢ)] has error 9e-11. The final error is accumulated through the mean (gain 0.65–0.96 per
   layer); the fresh error per layer is ~50 % variance, ~25 % κ₃ᵢᵢᵢ, ~25 % κ₄ᵢᵢᵢᵢ (joint fit R² 0.93–0.97, coefficients −1).
2. **Channel oracles** — true value injected only into the mean closure, at every layer:

   | exact input | final MSE | ratio |
   |---|---|---|
   | none | 2.290e-8 | 1.000 |
   | variance | 1.220e-8 | **0.533** |
   | κ₄ diagonal | 1.600e-8 | 0.699 |
   | κ₃ diagonal | 1.965e-8 | 0.858 |
   | all three | 0.130e-8 | 0.057 (truth noise) |

   The chain's whole error lives in these three per-neuron numbers.
3. **Variance.** Relative error only 0.15 %, but ~95 % of it is a random projection dvarᵢ = wᵢᵀ dC_post wᵢ of the
   previous layer's **off-diagonal post-ReLU covariance error**. That error is part inherited (Φᵢ dC_pre,ij Φⱼ, R² 0.52 →
   0.79 with depth) and part **fresh: ~4.2e-5 rms per entry at every layer**, which the chain's own closure terms explain
   only 3–8 % of. The chain's post-ReLU C, K22 and κ₄ diagonal are all accurate (R² 0.9–1.0).
4. **κ₃ slice.** The chain's κ₃(i,i,j) slice has slope 1.00, R² 0.996–0.998, but a **random relative error of 4 % → 7 %**
   with depth. It is not rank truncation (all sources exact: unchanged; D21-feedback rank 16 → 128: unchanged).
5. **κ₄ diagonal.** The chain's regenerated diagonal has corr 0.75–0.83 with the truth; on the neurons that matter
   (|μ/σ| < 1.5) its relative error is ~20 %. The pairwise transport of the **true** post-ReLU slices,
   (W⁴)ᵀK4_post + 3 diag((W²)ᵀ K22_post W²), gets the level exactly (slope 1.00) but the per-neuron pattern badly (corr
   0.24–0.59): the pattern is in the (2,1,1) and (1,1,1,1) parts of the post-ReLU κ₄ tensor. No cheap combination of
   chain quantities (13 features, cross-validated across two networks) fixes it; |κ₃ᵢᵢᵢ| correlates with it only on
   neurons the mean closure ignores. A Mehler "star" birth law κ(h_r,h_r,h_k,h_m) = β_r a_kr a_mr is right at layer 2
   but, transported linearly, is only ~10 % of the deep diagonal: the deep κ₄ is regenerated at every ReLU.
6. Dead, measured: scalar or depth multipliers on κ₄ (V39: 16–26 % worse; the κ₄ bias sits in neurons the closure
   ignores), exact old tier (−3.9 % MSE for +19 % cost), everything listed in earlier rounds (rank/Tucker/TT compression,
   column sampling, age gates, aggregates, learned corrections, copulas, per-family Strassen).

(Another model is working on the **κ₃ transport** question: which second-order / curvature terms of κ₃(h) the linear
source transport drops. Please take the other two.)

## Questions

**Q1 — the variance channel (worth −47 %).** The pre-activation variance of the next layer is the quadratic form
wᵢᵀ C_post wᵢ. We only need it on neurons with |μ/σ| < 0.75, to ~0.1 % relative accuracy. Is there a cheaper or more
exact way to get **those** 300–400 numbers than propagating the full n×n covariance through the closure? For example:
an adjoint / per-row recursion for diag(Wᵀ C_post W) that uses the exact bivariate law on the (i, j) pairs that
dominate wᵢᵀ C_post wᵢ; a control-variate or stratified Monte-Carlo estimate of the variance *correction* only (the
chain's value is known, so only the error must be sampled — how many samples, at what cost?); or a closure term we are
missing in the off-diagonal covariance (what produces a random ~4e-5 per entry, ~2 % of the entry's typical size?).

**Q2 — the κ₄ diagonal (worth −30 %).** κ₄(zᵢ') of the next layer = Σ W⁴ K4 + 3·(2,2) + 4·(3,1) + 6·(2,1,1) + (1,1,1,1)
contractions of the post-ReLU κ₄ tensor. The first two set the level; the per-neuron pattern is in the last two.
Derive the leading (2,1,1)/(1,1,1,1) content of the post-ReLU κ₄ at depth (born at each ReLU from the pre-activation
κ₃ ⊗ C and κ₄ content, not transported from old births) and a way to evaluate its contraction wᵢ-by-wᵢ without the
full tensor. Give the FLOP count.

**Q3 — sampling the three numbers.** Given the chain's own estimates, is there any estimator of the per-neuron
*corrections* (δvar, δκ₃ᵢᵢᵢ, δκ₄ᵢᵢᵢᵢ) — e.g. antithetic or conditional Monte-Carlo, Rao–Blackwellisation over the last
weight layer, quasi-random designs on the sphere (a 1-homogeneous network: f(x) = |x| f(x/|x|)) — whose variance at a
budget of ~10⁴–10⁵ forward samples (1 sample through 16 layers ≈ 0.016 u) is small enough to help? We estimate naïve
MC needs ~10⁶ samples for the variance and ~10⁵ for κ₄, far too many; can a control variate built from the chain's own
Gaussian surrogate cut that by 100×?

**Q4 — which one first.** Given the costs you derive, rank the three channels by expected score gain per unit of
compute and say which experiment we should run first.

For every proposal: (a) the state kept and its size, (b) FLOPs in units of n³ per layer, (c) the falsifier on our stored
arrays (per layer we have the true pre- and post-ReLU mean, C, κ₃(i,i,j), κ₄(i,i,j,j), κ₄(i,i,i,j), κ₄ diagonal, and we
can capture any chain quantity), (d) the end-to-end test.
