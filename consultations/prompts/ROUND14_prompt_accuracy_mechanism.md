# Round 14 — ONE question: which estimator design explains a ~25% accuracy lead at lower cost, and how do we grow it?

*(Answer in English, at most 5 pages. Give ONE concrete design (at most two), its cost at the real shapes, a staged
evaluation plan, and ONE cheap first falsifier on our stored data. Mark every number as derived, estimated or quoted; never
present an estimate as a measurement. Do not re-propose anything in the "measured dead" list unless you say exactly why the
kill was too narrow.)*

## Important: how the design will be judged

We will NOT require your design's first prototype to beat our current, heavily tuned estimator. Our chain is the product of
hundreds of experiments; a new design needs time to mature. Judge and present your design by:

1. **Ceiling:** how much of the final error could a perfect version of it remove (tie this to the oracle numbers below)?
2. **Component accuracy:** does its key quantity (e.g. a κ₃ slice, a κ₄ diagonal, the covariance) beat the SAME component
   of the plain public chain (V29, below) on our teacher data, one step teacher-forced or run free?
3. **Maturity projection:** a realistic score after ~2 weeks of engineering (submissions close 18 Oct 2026), with the
   cost tricks that transfer to any design (Strassen products, compaction of inactive neurons, see "Engineering facts").
4. **Development path:** milestones, each with a pass/fail gate, compared against **V29** (same maturity), not against us.

## Setting

Fixed bias-free ReLU MLP, width n = 1024, depth L = 16, He-initialised weights (N(0, 2/n)), input x ~ N(0, I). Task:
estimate the final-layer post-ReLU means. Score = final-layer MSE × max(0.1, FLOPs / 2⁴¹); 1 u = 2³¹ FLOPs ≈ one dense
1024³ product (a level-5 Strassen square product ≈ 0.51 u). z_{l+1} = h_l W_l, h_l = ReLU(z_l). All computation runs inside
a metered numpy-like library (flopscope); only its operations are allowed.

**Two hard grader caps per network:** wall time 120 s and **0.4 s of "residual" time (Python outside the metered ops)**.
A capped network gets zero predictions. We measured that every metered operation costs ~0.02 ms of residual on the grader
(0.07 ms on our laptop), so a design must run in **≲ 18,000 metered operations per network** (batched ops, no per-neuron or
per-term Python loops). Our current chain issues ~17,700.

## Baseline and our estimator

- **V29 (public, MIT, team 504aldo):** factorised K = 3 cumulant chain. Carries mean, dense covariance C, the third cumulant as
  a sum over "sources" (one born at each ReLU; dense n × n legs for ages ≤ 4, a shared rank-384 basis (nested rank-224)
  afterwards) and a memoryless fourth cumulant (core regenerated each layer as diag(g) + λ_l C_off). Univariate Edgeworth mean
  closure, bivariate Wick/Edgeworth covariance closure. Graded: raw MSE 2.13e-8 at utilisation 0.254 (score 5.4e-9).
- **Ours (same formulation, engineered):** raw 2.16e-8 at utilisation 0.189 (score 4.09e-9; a residual-safe version
  4.12e-9). Our lead over V29 is cost, not accuracy.
- **Leaders (public per-submission results):** raw 1.54–1.61e-8 at utilisation 0.125–0.139 (score 2.0–2.1e-9). In the last two
  days several teams improved raw MSE by 4–8% AND cost by 10–20% at once. Teams on the public chain family converge to
  raw ~2.18e-8 at utilisation ~0.178 (3.9e-9). So a mechanism worth ~25% raw accuracy at ~30% lower cost exists outside the
  public chain family.

## Measured facts (Monte-Carlo teachers, 6×10⁷ samples per layer; MLP #0 = public network; "synthetic" = a second He network)

1. **Closures are exact given true inputs** (teacher-forced post-ReLU covariance error falls to the teacher's noise).
2. **Channel oracles** (teacher value injected at every layer, final-MSE ratio, MLP #0): variance ×0.53; κ₄ diagonal ×0.70;
   κ₃ diagonal ×0.87; all three ×0.057. Perfect κ₃ input (the (2,1) slice κ₃(zᵢ,zᵢ,zⱼ) + diagonal): ×0.61 (MLP #0), ×0.71
   (synthetic). Forcing the true mean at every layer removes ~80% of the final error.
3. **The input error is carried memory.** κ₃ and κ₄ "all-distinct" content is carried over ≥ 6 layers: keeping only the κ₃
   sources of the last 10 layers still leaves 16% of the (2,1) slice at layer 15; erasing all non-Gaussian history 2 layers
   back recovers only 2–6% of the deep κ₄ level (20–35% at 6 layers). The chain's κ₄ diagonal has the right level but the wrong
   per-neuron pattern (corr 0.26–0.34 on the neurons the mean closure is sensitive to); an exact linear carry of the true κ₄
   gives 0.77–0.80.
4. **Parity:** old sources' contributions are odd in the later weights; even statistics (per-neuron variance, kurtosis,
   ensemble averages) have ~zero correlation (≤ 0.04) with the missing content.
5. **Inactive neurons:** 5–25% of each layer from layer 5 on have Φ(μ/σ) < 10⁻². Their linear pass-through can be dropped
   (MSE ×0.99) and so can their (2,1)-slice columns (×1.00–1.01), but their own skew/κ₃ is essential: removing them as hubs of
   the newborn κ₃ source costs ×6.4 (still ×1.5–1.8 at Φ < 10⁻⁵); always-on neurons (Φ > 1 − 10⁻³) as hubs ×3.7; their own κ₃
   diagonal at intermediate layers ×1.6–1.8 (free only below Φ = 10⁻⁵).

## Engineering facts (transfer to any design)

- Cost of our chain (≈ 193 u): young κ₃ sources (dense legs, 4 square products per source per layer) ~100 u; the old
  shared-basis tier ~36 u; covariance ~6 u; the rest small. Level-5 Strassen on every n × n × n and n × 384 × n family;
  inactive-neuron rows dropped from the contraction dimension; Gram products billed at half via aliased einsums.
- Each Strassen call costs ~110–240 metered ops, so the op budget limits how many separate large products a design may issue
  per layer (~6–8 Strassen families per layer in total).

## Measured dead (do not propose unless you show the kill was too narrow)

Low-rank / Tucker / TT / CP / sketch compression of κ₃ source legs or their elementwise products (~700 of 1024 directions
needed); any linear compressed state closed under the layer maps (Sym³/Sym⁴ irreducible); short memory horizons, age modes;
diagonal-only κ₃/κ₄ transport; low-rank-plus-diagonal covariance; rank-1 κ₃; slice-state recursions (36% error in sample);
deterministic moment-matched particles; Monte-Carlo / control variates at this budget; learned per-neuron corrections
(R² ≤ 0.02 at depth); scale-mixture or Sym(C⊗C) κ₄ predictors; fresh-physics κ₄ kernels; a young-source κ₄ carrier;
Team Puffi's published stack re-implemented (SSC κ₄ response closure + drift corrector: ×1.006–1.023, held-out R² ≤ 0.004);
a carried pair-response κ₄ mode (×0.98–1.00); κ₄ riding the κ₃ source legs (out of sample ×1.05); role-split κ₄ constants
(×0.97 / ×1.00); per-layer / per-block refits of the closure constants; Strassen level 6+ (memory and wall time).

## The question

**What estimator design — representation, per-layer update and closure — could plausibly explain the leaders' ~25% raw
accuracy lead over the public chain family at ~30% lower cost than our engineered version of it (half the cost of V29), and how
would you grow it from a first prototype to a competitive estimator in ~2 weeks?** It may change what is represented (not just how the same sum-over-sources tensor is factorised), and it may drop
parts of the chain entirely if the accuracy comes back elsewhere.

Deliverable:
1. The design: state per layer (with shapes), birth and transport rules, the mean and covariance closures it feeds (exact
   equations where they differ from the chain).
2. Cost per layer at n = 1024 in dense square products and an estimate of metered operations per network (≤ 18k).
3. Its ceiling and why its key quantity should beat the V29 component (tie to facts 2–5).
4. A staged plan: (a) a ceiling or component test on our stored data; (b) a first prototype gate against V29 (not against
   us); (c) the engineering steps to maturity with a projected score.
5. ONE cheap first falsifier on our stored data. Per layer we have: the teacher's pre-activation mean, C, the (2,1) slice,
   K22 and K31 (κ₄ (2,2) and (3,1) slices) and the κ₄ diagonal (all 16 layers on MLP #0; even layers + 15 on the synthetic
   net), post-ReLU teachers at layers 3/7/11/14/15, the weights, dense captures of the chain's κ₃ sources at layers
   7/11/14/15, and the chain's own per-layer channels.
