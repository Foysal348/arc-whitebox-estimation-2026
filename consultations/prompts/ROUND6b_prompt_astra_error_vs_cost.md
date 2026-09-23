# Round 6b (Astra) — why does accuracy always fall faster than cost here, and is any family exempt?

*(Companion to the round-6 prompt sent to another model, which asks "find a mechanism". This one asks the complementary
question: derive the law, then say which family can break it. Answer in English. Prefer algebra and explicit FLOP
ledgers over narrative; where you propose a diagnostic, give code we can run on stored dumps.)*

## The setting, in one paragraph

Width n = 1024, depth L = 16, bias-free ReLU MLP, weights i.i.d. N(0, 2/n), x ~ N(0, I); predict the final post-ReLU
mean to MSE ≈ 1.6e-8 within 2⁴¹ FLOPs (1 unit = 2³¹ FLOPs = one dense n³ product; level-5 Strassen = 0.51 u; budget
1024 u). Our baseline is the public factorised K = 3 cumulant chain (V29): mean, covariance, and a third cumulant kept
as a **sum over births**. A source born at layer b carries two legs, `T_b = M_l ⋯ M_{b+1}` and `Ã_b = T_b a_b` with
`a_b = diag(w1_b)·C_off^(b)` (verified to 1e-5), and the next layer consumes

    D21^(l)_{ik} = Σ_b [ (T_b∘T_b) Ã_bᵀ + 2 (T_b∘Ã_b) T_bᵀ ]_{ik} / 3 ,   plus its diagonal.

Cost: 260 u = young tier 116 + **old tier 107** + thin legs 25 + covariance 7 + closure 6; four dense products per
source-layer (two transports, two slice contractions) = 2.05 u. Graded: raw 2.108e-8 at C/B = 0.2437, score 5.137e-9.
Leaders: raw 1.64–1.9e-8 at 0.13–0.16, score 2.1–3.0e-9.

## The empirical law we want explained

The score is `MSE × max(0.1, C/B)`, so a cut of 35 % in C that costs 20 % in MSE is a 22 % better score. **Every
compression we have measured is on the wrong side of that line**, and the failure mode is quantitatively consistent:

| compression | ΔC | Δ(raw MSE) | implied exponent p in ΔMSE ∝ (ΔC)^p |
|---|---|---|---|
| shared basis 384 → 320 | −5.9 % | +7.5 % | ≈ 1.3 |
| shared basis 384 → 256 | −12 % | +30 % | ≈ 2.2 |
| nested rank 224 → 160 | −3 % | +5.6 % | ≈ 1.9 |
| nested rank 224 → 128 | −4 % | +14 % | ≈ 3.5 |
| birth columns, deep births at 0.9 / 0.8 / 0.7 | −1 / −2 / −5.6 % | +4 / +14 / +25 % | ≈ 4 / 7 / 4.5 |
| layer-dependent age gate (gate 3 to layer 8) | −0.9 % | +7.6 % | ≈ 8 |
| old tier removed | −41 % | ×35 | — |

Two facts that should constrain any explanation: (i) the per-source error is small where the end-to-end error is large —
deterministic column screening at 70 % gives ≤ 1.2 % error in a single source's D21 but 25–32 % in the final mean, so the
error **amplifies by a factor ~25 along the chain**; (ii) the amplification is not generic noise growth, because the
chain's own float32 rounding (drift MSE 2.9e-13 against a float64 run) does not amplify at all.

**Question 1.** Derive the amplification. Is it (a) coherent accumulation of a *bias* over 14 layers and ~10 sources,
(b) a condition number of the layer map acting on the deep content, or (c) a feedback loop through the newborn source
(the chain feeds D21 back into the next birth matrix)? Give the scaling you predict for each, and a diagnostic that
separates them on our dumps. We can zero any part of the chain at any layer and re-run.

**Question 2.** Given the mechanism, is any compression family *exempt*, in the sense that its error is **linear** in the
compute removed rather than super-linear? Two candidates we care about:

* **Unbiased (Horvitz–Thompson) column sampling.** Every slice term is cubic in a source's birth columns, so keeping
  column j with probability p_j and rescaling its columns by p_j^(−1/3) is unbiased for every term. If the amplification
  in Question 1 is coherent-bias accumulation, an unbiased scheme should accumulate like a random walk (√14 rather than
  14) and the trade could flip. Give the variance of this estimator for the cubic functional (T∘T)diag(w2)Ãᵀ with
  T and Ã correlated through Ã = T a, the optimal p_j, and the resulting error as a function of the keep fraction.
  We are running the experiment; we want your prediction and the optimal sampling law.
* **Query-orthogonal truncation.** Every truncation we tried discards a part that is *small*; is there one where the
  discarded part is provably *orthogonal to the downstream query* — i.e. where the error enters the final mean only at
  second order? The query at layer l is the Gaussian chain's sensitivity of the final mean to the covariance at that
  layer, which we can compute (we have a validated first-order response model: injecting a random δC at layer l and
  predicting the final mean change gives R² = 0.9999 at l = 14 and 0.9957 at l = 10, with unit coefficients).

**Question 3.** Antithetic or cancelling truncations. If the amplification is coherent, the natural fix is to make
successive errors anticorrelated: alternate the dropped set between layers, use sign-symmetrised truncations, or pair
each source with a complementary truncation. For a cubic functional of two correlated legs, what is the correct
antithetic construction, and what cancellation order does it buy?

**Question 4.** Check our arithmetic on the objective. We claim that at fixed accuracy no further cost cut is available,
but that the score permits losing accuracy for cost at a rate we have never achieved. Is there a cost-accuracy frontier
argument (an inequality relating the two) that would say the trade is impossible for *any* estimator of this class,
rather than merely for the ones we tried? If such an inequality exists, we would rather know it than keep searching.

## Already measured dead (do not re-propose)

Pooled Tucker/TT cores (multilinear rank ≈ 250 needed for 2 % query error, costing n·r³/2 + n·r³ per layer ≈ the current
107 u); vector–metric aggregate states (87–89 % error at k ≤ 2); isotropic-trace, diagonal-Hermite and degree-2
TensorSketch surrogates (98 %, 95 %, 790 %); cubic source-index coding with skew codes (unbiased, but cross-source
interference gives 250–530 % error at 1–3 channels); birth-operator compression a_b ≈ diag + rank-k (57–73 % at k = 32);
slice recursion (R² = 0.03–0.05); adjoint contraction (the low-rank part of the sensitivity carries 14 % of the
response); learned per-neuron corrections (100 networks, cross-validated: ≤ 0.7 % recovered); Richardson extrapolation in
rank; TT-cross in input coordinates; copula with exact marginals; every per-family Strassen level.

Three obstructions we consider established: **parity** (a source's contribution is odd in the current layer's weights, so
its conditional expectation given any even statistic is zero); the **Khatri–Rao wall** (legs have 99 % of their energy in
100–260 of 1024 directions, but their Hadamard products are full rank with a flat spectrum); the **sensitivity map**
(zeroing the old content at any single layer ≥ 8, or dropping any single birth, each costs more than the whole error
budget).

## What we can run for you

One 1024×16 network with stored layer dumps at layers 6, 9, 10, 13 and 14 (dense legs, shared basis, birth matrices,
thin legs, per-source birth coefficients), the full chain with any part zeroed or replaced at any layer, a 100-network
scoring harness against 10⁹-sample ground truth, and GPU Monte-Carlo teachers for any statistic you want measured.
Anything runnable in under an hour with numpy on the dumps we will run the same day.
