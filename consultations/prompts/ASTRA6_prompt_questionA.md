You are helping with a research problem in mechanistic estimation of neural networks. Please read everything, then answer the questions at the end. Be concrete and quantitative; every proposal must carry a FLOP estimate in the cost unit defined below and a first falsifying experiment. Do not re-propose anything in the "already dead" list unless you can say precisely why the recorded measurement does not apply.

## 1. The problem

A random ReLU MLP: width n = 1024, depth L = 16, weights W^(l) i.i.d. N(0, 2/n) (He init), no biases. Input x ~ N(0, I_n); h^(l) = ReLU(W^(l) h^(l-1)). Estimate the vector of expected post-ReLU activations E[h^(16)] (1024 numbers) to minimal mean-squared error, under a hard compute budget B = 2^41 FLOPs per network (about 65,000 forward passes). Every arithmetic operation is metered analytically (matmul (M,K)@(K,N) = M·N·(2K-1) FLOPs; elementwise ops 1/element; exp/log 16/element; float64 costs 2x float32). Score = MSE x max(0.1, FLOPs/B): below 10% of the budget only accuracy counts; above it, cost multiplies the error. Ground truth is a 1e9-sample Monte Carlo mean.

Cost unit used below: 1 unit = one dense (1024 x 1024 x 1024) matmul = 2^31 FLOPs. B = 1024 units.

Reference points (all measured on the public leaderboard, 100 MLPs):
- Monte Carlo sampling saturates at adjusted ~1.2e-6 (its MSE ~ 1/samples exactly cancels the cost multiplier).
- Mean propagation (per-neuron Gaussian moments): raw MSE 2.2e-4.
- Gain-covariance propagation (mean + full covariance, Gaussian closure at each ReLU): raw 4.05e-6, 24 units.
- Factorized K=3 cumulant propagation with a memoryless 4th-cumulant regeneration (public MIT code, "V29"): raw 2.13e-8 at 260 units (0.254 B). This is the method we run.
- The leaders: raw 1.64e-8 to 1.9e-8 at 0.13-0.16 B (133-165 units). Nobody has disclosed how.

## 2. The method we run (factorized K=3 chain), in enough detail to reason about cost

Per layer the chain carries the pre-activation mean, the full covariance C (n x n), and the third cumulant kappa3 (n^3, never materialised). kappa3 is represented as a SUM OF SOURCES: each ReLU layer injects non-Gaussianity ("a source is born"), and each source is transported by later layers. A source s is stored in factored form:
- a propagator leg P_s (n x n): identity at birth, then P_s <- W diag(w1) P_s at every later layer (w1 = per-neuron first Hermite/Wick coefficient of the ReLU at that layer),
- a birth leg A_s = P_s a_b (n x n), where a_b is the covariance at birth,
- a few thin legs (n x 16).
With 16 layers up to 15 sources are alive.

The ReLU at the next layer needs only two slices of kappa3: the diagonal D3_i = kappa3_iii (skewness, n numbers) and the (2,1) slice D21_ic = kappa3_iic (n x n). Their contributions from a source are contractions of ELEMENTWISE (Hadamard) products of its legs, e.g.
  H = (A ∘ P ∘ w2) A^T,   T2 = (A ∘ A ∘ w2) P^T,   T3 = (P ∘ P) M^T,   T4 = (M ∘ P) P^T,
  D3_i = 3 sum_j w2_j A_ij^2 P_ij + sum_j M_ij P_ij^2,
(M is a third leg expressible through A, P and a thin residual). Empirical error law: extra final MSE ~ 4.2e-6 x eps^2 where eps is the relative RMS error of D21, so D21 must be right to ~2%.

Cost anatomy of V29 (260 units per network): "young" sources (age <= 4 layers, dense legs) 116 units, flat 8.7 units/layer from layer 5; "old" sources 107 units; thin/elementwise 25; covariance 7; ReLU closure 6.
Old sources (age > 4) are confined to ONE shared orthonormal basis Qc (n x 384; re-fit every layer by a randomized range finder on the weighted leg Gram): A_s = Qc FA_s, P_s = Qc FP_s with static (384 x n) factors; sources older than 7 layers move to a nested rank-224 sub-basis. Measured rank law: the rank that holds accuracy is ~3n/8 at age 4, n/4 at age 6, 7n/32 at age 7; one notch lower is a cliff (+22% / +15% error).
BUT the shared basis is only a 2x lever per source, because at every layer each old source must be RE-FORMED as dense n x n legs (Qc FA_s: n^2 r FLOPs each) and RE-CONTRACTED (n^2 r each), because the Hadamard products A ∘ P etc. cannot be done in factor space: for A = Qc FA, P = Qc FP the product has Khatri-Rao rank r^2 >= n. By layer 14 there are 10 old + 4 young sources and the two tiers cost the same (~14 units/layer).

Key identity: the leaders' compute (133-165 units) equals V29's bill WITHOUT the old tier (153 units) to three digits. So whatever carries the old (deep) third-cumulant content in the leading methods costs ~nothing, while in V29 it costs 107 units for a 6x accuracy gain (dropping old sources: raw 1.0e-6; keeping only 4-5 young sources: 1.35-2.9e-7).

## 3. A measurement we made today (the reason for this question)

We took the actual dense legs A_s, P_s of live sources at layers 6, 10, 14 of a real network and computed singular spectra:
- The legs themselves are low-rank and get lower with age: number of directions holding 99% of the energy (of 1024): old sources 213-261 at layer 6, 133-163 at layer 10 (age >= 7), 96-116 at layer 14. Young dense legs 230-500.
- Their elementwise products A∘P, A∘A, P∘P are NEAR FULL RANK with a flat, noise-like spectrum: 99% of energy needs 680-780 of 1024 directions at every layer and age; the top singular value is only ~1/10-1/30 of the Frobenius norm.
So: no low-rank sketch of the products is possible; the Tucker route Sum_s (Qc ⊗ Qc) T_s Qc^T with T_s = Sum_j FA_aj FP_bj FA_cj costs n r^3 (56 units at r = 384, 10.7 at r = 224, and accuracy cliffs below r ~ 150). The wall is numerical, not only algebraic.


## 3b. Two further results obtained after section 3 (please take them as given)

(i) Parity (derived, exact within the Hermite closure): for a source born at layer b and any later layer l, the legs are P = W^(l) D P', A = P a_b with P', a_b, w2^(b) independent of W^(l). Hence every quantity linear in the source's kappa3 (H, D21, D3 contributions) is an ODD function of W^(l), while the covariance, the leg Grams and all per-neuron variances are EVEN. The weight ensemble is sign-symmetric, so E[ H_s^(l) | C^(l), Gram(A_s,P_s), all state at layers < l ] = 0 exactly. An old source has no mean-field part: the noise-like Hadamard spectrum of section 3 IS the whole signal. (Corollary: kappa4 terms built from two kappa3 contributions are even, so their conditional expectation is nonzero and, by rotational invariance, must be built from C (x) C with layer scalars: this explains the regeneration law G_off = lambda_l C_off of item 7.)

(ii) Sensitivity map (measured on one network, final-layer MSE of the variant against the full V29 prediction; V29's own error vs truth is 2.1e-8):
- keep the old sources' contributions only in the last 1 / 2 / 3 layers (transport them, re-form dense only late): 4.9e-7 / 4.2e-7 / 3.2e-7. Late re-formation is dead.
- zero the old sources' contributions at ONE layer only: L5 3.4e-9, L6 1.1e-8, L7 1.5e-8, L8 2.6e-8, L9 2.7e-8, L10 5.2e-8, L11 5.2e-8, L12 7.1e-8, L13 6.7e-8, L14 8.2e-8, L15 6.4e-8. Every layer from 8 on matters at the scale of the whole error budget; effects are roughly additive (sum ~5e-7).
- drop one birth entirely (slots 0..10, oldest first): 1.3e-7, 2.7e-7, 1.8e-7, 1.9e-7, 1.6e-7, 2.1e-7, 1.9e-7, 1.7e-7, 1.7e-7, 1.8e-7, 1.7e-7. Flat: no dispensable source.
So within the sum-of-sources representation the old tier can be neither compressed (section 3), nor averaged (parity), nor omitted anywhere (this map). Question Q1(b) and the "late re-formation" reading of Q2 are therefore closed; please do not spend words on them.

The question is now sharper: the leaders' bill (133-165 units) equals the young tier alone, yet the deep content is needed at every layer from every birth. What FORMULATION of the deep third cumulant could carry it at ~zero marginal cost per old source-layer? Candidates we have not tested and would like your judgement on: a recursion on the needed slices D21/D3 themselves (D21^(l+1) from D21^(l), D3^(l), C^(l) and W^(l+1) through the exact transport law plus a closure for the part of kappa3 outside the slices - what does the j != k remainder look like and is it small for OLD sources specifically?); an orthogonal-polynomial (Hermite) expansion of the whole layer map in which sources are never separated; a "shared propagator" trick where all old sources are transported by ONE common operator with per-source scalars; or anything from statistical physics of deep random networks that keeps quenched (per-network) odd-order structure.

## 4. Already dead (measured, do not re-propose without a new mechanism)

1. Drop old sources (window of youngest W): W=4 -> raw 1.0e-6 (every source matters at the final layer).
2. Memoryless closure replacing old sources by a tensor supported on the (2,1)/(3,) slices: 9.1e-7; the lost part is 18-23% of D21 energy per layer, high-rank, and NOT predictable from live-state features (+0.02 R^2 with linear features).
3. Gaussian scale-mixture closure of the fluctuation part: 2.3e-6.
4. CP-rank cap on the merged tensor ("hub columns"): needs 8n columns for 2% D21 error, only 4n affordable; column importance is flat.
5. Symmetric Tucker confinement of all legs: 14-31x worse (P is born as the identity: full rank at birth).
6. Per-source rank ladders / shared basis below the rank law: cliffs.
7. Per-source explicit kappa4 legs: >= 0.95 B. Instead a regeneration law G_off = lambda_l C_off (one scalar per layer, R^2 0.66-0.97) recovers 90% of the value free; why the kappa4 core aligns with the covariance is unexplained.
8. Learned output corrections (ridge on live features; per-layer mean corrections): final-layer features correlate < 0.09 with the error.
9. Marginal Edgeworth / Gram-Charlier / exact Mehler off-diagonal on top of covariance propagation: capped at 4-5e-6 (order-2 + marginals cannot see the cross-neuron third cumulant).
10. Monte Carlo control-variate hybrids on the analytic chain: would need 33-300x variance reduction; realized 1.5x.
11. Symmetric-contraction FLOP tricks: no dense symmetric order-3 object is ever formed.
12. Adjoint/backward sensitivity of the final mean w.r.t. D21: no cost drop (still one dense contraction per source-layer).
13. Tucker core in the shared basis, multiplicatively closed bases, hub-side merges: closed by arithmetic (see section 3).
14. Deeper Strassen: level 6 exceeds the 8 GB memory cap; the 18 block additions cancel the savings at level 7.
15. Literature (finite-width corrections, tensor programs, uncertainty propagation, cumulant tensors, ~25 papers): annealed-over-weights results have kappa3 = 0 by construction; quenched results stop at covariance.
16. Sampling variants (QMC, antithetic, importance, cubature): ~1.2e-6 adjusted at this shape.

## 5. What we ask you

Q1. Propose representations of the DEEP (old) third-cumulant content that never require elementwise products of dense legs, or that make those products cheap. For each: the mathematical object carried, the per-layer update rule, the cost in units per layer (with the counting rule above), what it loses relative to exact transport, and the first experiment that would falsify it on one network in under an hour of CPU. Consider, at least: (a) exploiting that old legs have ~100 effective directions while their products do not — e.g. moving the ReLU-side nonlinearity into the basis (Hermite expansion of the closure in the Qc coordinates), (b) [closed by parity, see 3b — skip], (c) representing kappa3 of deep sources through the covariance and the D3/D21 slices of EARLIER layers via the exact transport law (what is the minimal state that reproduces D21 at layer l to 2%?), (d) any exact identity that turns (A∘P)A^T for A = Qc FA into an operation of cost O(n r^2) or O(n^2 r) WITHOUT forming A or P.

Q2. Independently of Q1: the leaders spend exactly the young-tier cost. Give your best three hypotheses for what they carry instead of old sources, ranked, each with the observable that would distinguish it on the public leaderboard (cost/accuracy signature) or in a local experiment.

Q3. If nothing in Q1/Q2 is credible, say so, and instead propose the single most informative measurement we should make next (the per-layer and per-birth maps of 3b are already done).

Answer in English, with equations where needed, in at most ~2000 words. Mark clearly which claims are derived, which are conjectures, and which you would want us to measure.
