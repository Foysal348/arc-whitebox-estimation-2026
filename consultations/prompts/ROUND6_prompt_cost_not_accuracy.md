# Round 6 — the question has changed: we need CHEAPER, not more accurate

*(Paste this whole file. Answer in English. Be quantitative. Every claim should come with either algebra or a falsifier we can run in under an hour on one 1024×16 network with numpy.)*

## What changed since round 5

Two things are now settled that were open before, and they invert the question.

**1. The accuracy side is capped and the cap is public.** The author of the public chain (504aldo) published the end-to-end numbers for his κ₄ channel on 22 September: replacing the κ₄ matrix core by the cheap scalar law gives raw MSE 2.23e-8, the exact dense channel gives 2.04e-8, and dropping the channel gives 4.20e-8. So a *perfect* fourth-cumulant treatment is worth about 9 % of raw MSE and costs three extra dense products per source-layer — far more than the budget can pay. We independently measured a Monte-Carlo κ₄ teacher on two networks and confirmed the scalar law is an accurate fit *of the object the chain actually propagates* (the r = 1 matrix core), and that "improving" it against the raw cumulant slices makes the final answer monotonically worse. **Accuracy work on this chain is over.**

**2. The scoring rule means we do not need accuracy — we need cost.** The score is `final_layer_MSE × max(0.1, C/B)`. We sit at raw 2.108e-8 × 0.2437 = 5.137e-9. The leaders sit at 1.64–1.9e-8 × 0.13–0.16 ≈ 2.1–3.0e-9. Therefore:

* cutting cost by 35 % while **losing 20 % of accuracy** is a win (2.53e-8 × 0.158 = 4.0e-9, a 22 % better score);
* cutting cost by 45 % while losing 40 % accuracy is also a win (2.95e-8 × 0.134 = 3.95e-9);
* improving accuracy by 9 % at any cost increase above 9 % is a loss.

**Every trade we have measured is on the wrong side of this line**, and that is the puzzle we want attacked. Examples, all measured end-to-end on real networks:

| cost cut | accuracy cost | score |
|---|---|---|
| shared-basis rank 384 → 320 (−5.9 % C) | +7.5 % raw | ×1.01 (worse) |
| shared-basis rank 384 → 256 (−12 % C) | +30 % raw | ×1.19 (worse) |
| nested rank 224 → 128 (−4 % C) | +14 % raw | ×1.09 (worse) |
| birth-column screening, deep births at 70 % (−5.6 % C) | +25 % raw | ×1.18 (worse) |
| layer-dependent age gate, gate 3 up to layer 8 (−0.9 % C) | +7.6 % raw | ×1.07 (worse) |
| drop the old-source tier entirely (−41 % C) | ×35 raw | ×21 (much worse) |

The pattern is always the same: **accuracy degrades faster than cost falls.** We want a mechanism where it does not.

## The object and where the cost is

Width n = 1024, depth 16, no biases, weights i.i.d. N(0, 2/n), x ~ N(0, I). Predict the final post-ReLU mean to MSE ≈ 1.6e-8 within 2⁴¹ FLOPs (1 unit = 2³¹ FLOPs = one dense n³ product; a level-5 Strassen product costs 0.51 u; budget = 1024 u).

The public chain (V29, our baseline, 260 u) propagates mean, covariance and a factorised third cumulant represented as a **sum over births**. A source born at layer b has, at every later layer, two legs: the propagator `T_b = M_l ⋯ M_{b+1}` and `Ã_b = T_b a_b`, where `a_b = diag(w1_b)·C_off^(b)` is the birth matrix (exactly the row-scaled off-diagonal post-ReLU covariance of the birth layer — we verified this to 1e-5). What the next layer needs is

    D21^(l)_{ik} = Σ_b [ (T_b∘T_b) Ã_bᵀ + 2 (T_b∘Ã_b) T_bᵀ ]_{ik} / 3      (∘ = Hadamard)

plus its diagonal. Cost anatomy of the 260 u: **young tier 116, old tier 107**, thin legs 25, covariance 7, closure 6. Per source-layer it is four dense products (two transports, two slice contractions) = 2.05 u with Strassen. The leaders' whole bill (133–165 u) is close to the young tier alone, which is why the old tier is the target.

## What is already measured dead (do not re-propose)

Rank reduction of the shared basis (six settings); pooled Tucker/TT cores (need multilinear rank ≈ 250 for 2 % query error, which costs n·r³/2 + n·r³ per layer ≈ the same 107 u); aggregate "vector–metric" states (87–89 % error at k ≤ 2); isotropic-trace, diagonal-Hermite and degree-2 TensorSketch surrogates (98 %, 95 %, 790 % error); deterministic birth-column screening; cubic source-index coding with skew codes (unbiased, but the cross-source interference variance is ~m× the signal: 250–530 % error with 1–3 channels); birth-operator compression a_b ≈ diag + rank-k (57–73 % error at k = 32, 16–22 % at k = 384); slice recursion (R² = 0.03–0.05: 95 % of the next layer's contribution comes from off-slice entries); adjoint/linear-response contraction (the low-rank part of the final sensitivity carries 14 % of the response); learned per-neuron corrections (cross-validated over 100 networks: ridge recovers 0.3 %, gradient boosting 0.5 %, every single-feature correlation ≤ 0.04); Richardson extrapolation in rank (the truncation error is not a smooth power law — different ranks pick different subspaces); TT-cross of the network function in input coordinates (one ReLU layer at width 32 already gives 60–500 % integral error); copula with exact marginals (worse than a plain Gaussian closure at every layer); per-family Strassen levels (capped in code or memory-bound at 7.84 GB against an 8 GB cap).

Three structural obstructions explain all of it: **parity** (a source's contribution is odd in the current layer's weights, so its conditional expectation given any even statistic is exactly zero); the **Khatri–Rao wall** (the legs are low-rank, 100–260 of 1024, but their Hadamard products are full rank with a flat spectrum); and the **sensitivity map** (zeroing the old contribution at any single layer ≥ 8, or dropping any single birth, each costs more than the entire error budget).

## The question

Find a mechanism whose **cost falls faster than its accuracy**, in this problem. Concretely, any of:

1. **A cheaper unbiased estimator of the slice sum.** All slice terms are *cubic* in a source's birth columns, so Horvitz–Thompson importance sampling of the birth index is available (keep column j with probability p_j, rescale by p_j^(−1/3)); the deterministic version of the same truncation died because its error is a coherent bias that accumulates along the chain, while an unbiased version should accumulate like a random walk. We are running this now. What other unbiased randomisations of this contraction exist, and what is the variance of each? Be careful: naive randomised matrix multiplication has relative error √(n/s) here because the factors are noise-like, which is useless.

2. **A representation where the deep content degrades gracefully.** Every compression we tried has an error that grows faster than linearly in the saving. Is there a parameterisation of the old tier in which the error is *linear* in the compute removed — for instance one where the discarded part is provably orthogonal to the downstream query rather than merely small?

3. **A different allocation of the same budget.** We spend 2.05 u on every source at every layer regardless of age or of the layer's measured influence. Our attempt to exploit that (confining early-born sources sooner) failed because confinement is a *permanent* state change whose error is carried into the sensitive layers. Is there a re-allocation that is *reversible* — cheap treatment at one layer that does not degrade the source's state for later layers?

4. **A way to make the old tier's error cancel rather than add.** The measured amplification (1 % per-source error → 32 % final error) says the errors add coherently. Antithetic pairing, sign-symmetrised truncations, or a control variate built from the chain's own quantities could make them cancel. What is the right pairing for a cubic functional of two correlated legs?

For each proposal give: (a) the state kept and its size; (b) the FLOP count in units of n³, being careful that a Khatri–Rao product of an n×r with itself is n×r², not n×r; (c) why the error is unbiased or provably orthogonal to the query; (d) a falsifier runnable in under an hour with numpy on stored layer dumps (we have the legs, the shared basis, the birth matrices and the thin legs at layers 6, 9, 10, 13 and 14 of a real network).

If your answer is that no such mechanism exists and the leaders must be doing something outside this formulation, then state precisely which of the three obstructions above their method must evade, and what the cheapest object is that evades it.
