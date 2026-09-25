# Round 13 — ONE question: a cheap carried fourth-cumulant object with an all-distinct sector

*(Answer in English, at most 4 pages. Give ONE concrete mechanism, its exact FLOP count at the real shapes, and ONE
falsifier we can run on our stored data. Mark every number as derived, estimated or quoted; never present an estimate as
a measurement. Do not re-propose anything in the "measured dead" list.)*

## Setting

Fixed bias-free ReLU MLP, width n = 1024, depth L = 16, He-initialised weights W_l (N(0, 2/n)), input x ~ N(0, I).
Task: estimate the final-layer post-ReLU means. Score = final-layer MSE × max(0.1, FLOPs / 2⁴¹); 1 u = 2³¹ FLOPs ≈ one dense
1024³ product (a level-5 Strassen square product ≈ 0.51 u). Pre-activation of layer l+1: z_{l+1} = h_l W_l, h_l = ReLU(z_l).

Our estimator (public, MIT, 504aldo's factorised K = 3 chain, plus our changes): per layer it carries the mean, the dense
covariance C, the third cumulant as a sum over "sources" (one born at each ReLU, carried by dense n × n legs for ages ≤ 4
and in a shared low-rank basis afterwards), and a **memoryless** fourth cumulant: at each layer the κ₄ core is regenerated
as G = diag(g) + λ_l C_off (λ_l fitted per layer), carried one layer by (W∘W) on its diagonal (λ_l also adapted online per network). The mean closure is a
univariate Edgeworth formula in (μ_i, var_i, κ₃ᵢᵢᵢ, κ₄ᵢᵢᵢᵢ); the covariance closure is bivariate Wick/Edgeworth.
Current graded result: raw MSE 2.18e-8 at utilisation 0.227 (≈ 232 u). Leaders: raw 1.56–1.8e-8 at 0.13–0.20.

## Measured facts (all against Monte-Carlo teachers of 6×10⁷ samples per layer; MLP #0 = a public network; "synthetic" = a second He network)

1. **The closures are exact given true inputs.** Teacher-forced, the post-ReLU covariance error falls to the teacher's noise.
2. **Channel oracles** (teacher value injected at every layer; final-MSE ratio; MLP #0): κ₄ diagonal ×0.70; variance ×0.53;
   κ₃ diagonal ×0.87; all three ×0.057. A perfect κ₃ input (the (2,1) slice + diagonal): ×0.61 (MLP #0), ×0.71 (synthetic).
3. **The chain's κ₄ diagonal is its weakest channel:** 32–34% relative error at layers 14–16. Its mean level is right
   (ratio ≈ 1.0) but its per-neuron **pattern** is wrong: correlation with the teacher 0.26–0.34 on the neurons the mean closure
   is sensitive to (|μ/σ| < 1.5).
4. **The deep κ₄ pattern is carried memory, not fresh physics.**
   - A Gaussian input with the true mean and covariance gives the next layer only 1–4% of the true κ₄ level from layer 8 on.
   - Carrying the TRUE pre-activation κ₄ of layer l forward linearly, y = (Φ(α_l) ∘ (z_l − μ_l)) W_l with Φ the ReLU slope,
     recovers 83–90% of the level at layers 12–16, with pattern correlation 0.77–0.80 (MLP #0); 0.54 / 0.74 / 0.78 / 0.82 at
     L8/12/14/15 on the synthetic net. (Correlations are noise-attenuated; noise-corrected 0.73–0.88.)
   - **Memory horizon:** erasing all non-Gaussian history h layers back and running the true network forward recovers
     2–6% of the deep κ₄ level at h = 2 and 20–35% at h = 6; pattern corr 0.41 at h = 2 (0.52 noise-corrected), 0.77 at h = 6.
     The level needs ≥ 6 layers of history; the pattern somewhat fewer.
5. **Structure of the next κ₄ diagonal.** κ₄(z_{l+1})ᵢᵢᵢᵢ = Σ_{abcd} W_ai W_bi W_ci W_di κ₄(h_l)_abcd. About 90% of its norm is the
   (2,2) pair term 3 Σ_{a≠b} W_ai² W_bi² K22(h_l)_ab; the (4) and (3,1) terms are 1–8%. Even with the EXACT post-ReLU (4), (3,1)
   and (2,2) slices, the (2,1,1) + (1,1,1,1) all-distinct remainder leaves 14 / 22 / 33 / 43 / 46% error at L4 / 8 / 12 / 15 / 16.
   The chain's own (2,2) model (from the regenerated core) matches the teacher's K22 off-diagonal to 15–28% after scaling,
   pattern corr 0.65–0.73. The chain's κ₄ representation has **no (1,1,1,1) sector by construction** (Sym(G ⊗ m I)).
6. **Parity.** The old sources' contributions are odd in the later weights; statistics that are even in the weights
   (per-neuron variances, kurtosis, ensemble averages) have ~zero correlation with the missing content (measured ≤ 0.04).
7. **Cost anatomy (≈ 232 u):** young κ₃ sources ~105 u (193 dense square products, the floor for that tensor network),
   join into the shared basis ~25–30 u after our change (it was 41–46 u; we now carry the previous basis instead of re-running a range finder), old tier ~41 u, thin legs ~15 u, covariance + closure ~8 u.

## Leaderboard evidence (public per-submission results, Phase 2)

- Independent teams reached our chain's accuracy at much lower cost early: raw 2.02e-8 at utilisation 0.148 (6 Sep),
  raw 1.97e-8 at 0.150 (6 Sep), raw 1.89e-8 at 0.134 (13 Sep) — while the public chain we build on sits at ~0.24.
- One team gained accuracy in single steps at FIXED cost: raw 1.79e-8 → 1.69e-8 → 1.655e-8 in three submissions on one day
  (17 Sep) at utilisation 0.135, and is now at 1.59e-8. Another reached 1.81e-8 as early as 2 Sep.
- So an accuracy mechanism worth ~8–15% exists that adds little or no cost. We suspect the κ₄ channel (facts 2–5), but we have
  not identified it.

## Measured dead (do not propose)

Low-rank / Tucker / TT / CP / sketch compression of source legs or their elementwise products (products need ~700 of 1024
directions); any linear compressed state closed under the layer maps (Sym³/Sym⁴ irreducible); short horizons, age modes;
diagonal-only κ₃/κ₄ transport (the next diagonal is fed 90–99% by off-diagonal content); low-rank-plus-diagonal covariance;
rank-1 κ₃; slice-state recursions (carry the (2,1) slice + a response matrix: 36% error even in sample); deterministic
moment-matched particles; Monte-Carlo / control variates at this budget; learned per-neuron corrections (R² ≤ 0.02 at depth);
scale-mixture or Sym(C⊗C) κ₄ predictors (corr ≤ 0.09 with the true pattern); fresh-physics κ₄ kernels of any order (capped at
1–4% of the deep level); a young-source κ₄ carrier over the last 2–3 layers (out of sample the κ₄ error only moves
0.210 → 0.203); refitting λ_l; FB/λ knobs.

## The question

**Design a carried fourth-cumulant object — with a genuine all-distinct (1,1,1,1) sector — whose per-layer update costs
≤ ~1.5 dense square products (≈ 20–25 u over the network) and whose diagonal readout κ₄(z_{l+1})ᵢᵢᵢᵢ reaches pattern
correlation ≥ 0.7 with the teacher at layers 12–16** (the chain today: 0.26–0.34; the exact linear carry: 0.77–0.80).

We think the right object may be a small set of carried vectors or matrices whose quartic form Σ W_ai W_bi W_ci W_di (·)_abcd is
cheap to evaluate for all i (like the chain's d3 readout of κ₃, which costs O(n²) per source), and that is born at each ReLU from
quantities the chain already has (μ, C, the κ₃ slice D21 = κ₃(zᵢ,zᵢ,zⱼ), the regenerated core). Take or reject this with a reason.

Deliverable:
1. The object (state per layer, with shapes), its birth rule at a ReLU and its transport through W_l (exact equations).
2. The readout for the diagonal κ₄(z_{l+1})ᵢᵢᵢᵢ, and how it enters the existing mean closure (in place of, or added to, the
   regenerated κ₄ diagonal).
3. The exact FLOP count per layer at n = 1024.
4. Why its pattern should correlate ≥ 0.7 with the truth given facts 4–6 (what memory does it carry, over how many layers?).
5. ONE falsifier on our stored data: per layer we have the teacher's pre-activation mean, C, the (2,1) slice, K22 and K31
   (κ₄ (2,2) and (3,1) slices), the κ₄ diagonal (layers 1–16, MLP #0; even layers + 15 on the synthetic net), post-ReLU
   teachers at layers 3/7/11/14/15, the weights, and the chain's own per-layer channels. The gate: pattern corr ≥ 0.7 at
   L12–16 on both networks when the object is run free from layer 1 (or teacher-forced one step, stated clearly).
