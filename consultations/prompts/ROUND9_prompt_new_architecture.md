# Round 9 — design a NEW estimator architecture (not a patch) from a complete error map

*(Paste this whole file. Answer in English, as a PDF. We want architecture-level proposals: a different state, a
different propagation rule or a different split of the work — each with an explicit FLOP ledger and a falsifier we can
run on stored arrays. Patches to the existing chain are out of scope unless they are part of a new design.)*

## Problem

Bias-free ReLU MLP, width n = 1024, depth 16, weights i.i.d. N(0, 2/n), FIXED (only the input is random), input
x ~ N(0, I). Predict the 1024 final-layer post-ReLU means. Score = final-layer MSE × max(0.1, FLOPs / 2⁴¹);
1 u = 2³¹ FLOPs = one dense n³ product (a level-5 Strassen product counts 0.51 u); budget 1024 u; hard caps 8 GB RAM,
120 s wall on the grader's CPU.

Our estimator (public, 504aldo's V29 family): a factorised K = 3 cumulant chain — mean, covariance, third cumulant as a
sum over sources (each born at a ReLU as a Gaussian-closure star Sym(Σ_r e_r ⊗ e_r ⊗ a_r), a = diag(Φ)·C_off, transported
linearly by T = Wᵀ diag(Φ) …, old sources in a shared low-rank basis), a memoryless κ₄ regeneration, and a bivariate
Wick/Edgeworth closure at every ReLU. 260 u → raw MSE 2.1e-8, score 5.1e-9. Leaders: raw 1.64–1.9e-8 at 133–165 u
(score 2.1–3.0e-9). We need both: ~1.3× more accurate AND ~half the cost.

## The complete error map (measured against pre- AND post-ReLU Monte-Carlo teachers, 6e7 samples, two networks; 100-network harness is deterministic with paired SE 0.01–0.15 %)

1. **The mean closure is exact** given the true marginal (μ, var, κ₃ᵢᵢᵢ, κ₄ᵢᵢᵢᵢ) of each pre-activation (9e-11).
2. **Channel oracles** (true value injected only into the mean closure at every layer): variance −47 %, κ₄ diagonal −30 %,
   κ₃ diagonal −14 %, all three −94 % (= truth noise). The whole error is these three per-neuron inputs.
3. **The bivariate closure is exact too**: with all its inputs teacher-forced at one layer the post-ReLU covariance error
   falls to the teacher noise and the next layer's weighted variance error by 87–91 %. The Gaussian inputs alone
   (μ, var, C) remove 62–81 %, κ₃ inputs 15–28 %, κ₄ ~0.
4. **So the error is FRESH per-layer input error that ACCUMULATES**: through the mean (gain 0.65–0.96 per layer; per-layer
   MSE grows 0.7e-9 → 22e-9 smoothly over 16 layers) and through the covariance (inherited share 52 % → 79 % with depth).
5. **Where the fresh error is born**:
   - κ₃: 91–96 % of the (2,1)-slice error comes from the **all-distinct** entries κ₃(hᵢ,hⱼ,h_k) of the post-ReLU tensor,
     which the chain gets 9–16 % wrong. First-order-in-κ₃ Edgeworth terms explain 35–45 % of that; the port carries
     exactly half of two K×C families; second order in κ₃ (12 K×K terms) is negligible; **~55 % stays unexplained** within
     the cumulant expansion. Rank truncations are not the cause (exact untruncated sources: unchanged).
   - κ₄ diagonal: the pairwise transport of the true post slices gives the level but not the per-neuron pattern; the
     pattern is regenerated at every ReLU in the (1,1,1,1)/(2,1,1) sectors. A fresh quartic response kernel (input-κ₄
     edge + Gaussian star + Gaussian path) plus the all-distinct K₃×C "3+2" term evaluated on the chain's own κ₃ sources
     explains it (coefficient 1.0 ± 0.1): **−8.8 % end to end, but at ~140 u** (two extra products per source-layer).
6. **An unexplained paradox**: restoring the missing half of the K×C Edgeworth families (the theoretically correct
   coefficient) improves the κ₃ slice at layers 4–10 but makes the final MSE **worse** (+3.4 %); 1.25× of the shipped value
   is best (−0.4…−0.8 %). Something else compensates.
7. **Cost anatomy** of the 260 u: young κ₃ sources (ages ≤ 4, dense n×n legs) 116 u, old sources (shared basis rank
   384/224) 107 u, thin legs 25 u, covariance 7 u, closure 6 u. Every knob is at a score optimum (accuracy is bought
   one-for-one with cost).

## Measured dead (do not re-propose)
Rank/Tucker/TT/CP compressions of the κ₃ state; column sampling; aggregate/vector-metric states; low-rank or sparse
covariance corrections; learned per-neuron corrections; copulas; every Monte-Carlo or control-variate route (the best
coupled surrogate — linear or quadratic Taylor network on the same x — correlates 0.19 / 0.00 with the needed statistics
at layer 16; 100× would need 0.997); scalar/depth multipliers on κ₄; memoryless κ₄ laws; exact old tier; Strassen
variants; age gates; spherical/QMC designs; random-weight (mixture) arguments — the weights are fixed.

## The question

Propose up to three **new architectures** that, by construction, avoid the floor above. For each: (a) the state and its
size; (b) the propagation rule per layer; (c) the FLOP ledger in u; (d) why its per-layer fresh error in the three
channels is smaller than the chain's (be specific about the κ₃ all-distinct content and the κ₄ (1,1,1,1) content);
(e) a falsifier we can run on stored arrays — per layer we have the true pre- and post-ReLU mean, covariance, κ₃(i,i,j),
κ₄(i,i,j,j), κ₄(i,i,i,j), κ₄ diagonal at layers 3, 7, 11, 14, 15 (post) and 1–16 (pre), and we can inject any per-layer
per-neuron (var, κ₃ᵢᵢᵢ, κ₄ᵢᵢᵢᵢ) prediction into the mean closure and read the final MSE within minutes.

Directions we consider open, to take or reject with reasons: (i) a representation of the post-ReLU κ₃ in which the
all-distinct content is born and transported exactly to the order that matters, instead of stars + linear transport;
(ii) a κ₄-aware chain whose extra contractions ride inside the existing transport products (the "x-legs" of the 3+2 term
ARE the transported legs) so the −8.8 % comes at ≤ 20 u; (iii) a cheaper replacement for the young dense sources (116 u)
that keeps their slice accuracy; (iv) an explanation of the paradox in item 6 and the partner error it implies.
