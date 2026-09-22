# Round 3 — one precise question about the deep third cumulant of a random ReLU MLP

*(Paste this whole file. Answer in English. Be quantitative. If you cannot beat the bound, say so and explain why.)*

## The object

A ReLU MLP with no biases, width n = 1024, depth 16, weights i.i.d. N(0, 2/n), input x ~ N(0, I). Wanted: the mean of the final post-ReLU layer, to a final-layer MSE of about 1.6e-8 (means are O(0.4)), with a FLOP budget of 2^41 (one dense n×n×n product costs 2^31 FLOPs = "1 u"; the budget is 1024 u; a level-5 Strassen–Winograd product costs 0.51 u).

The state of the art we can reproduce (public MIT code, "V29", 260 u, raw MSE 2.13e-8) propagates mean, covariance and a *factorized third cumulant* layer by layer. The third cumulant born at layer b by the nonlinearity has the exact structure (ARC's reference code, `FactoredTensor.from_dstensor`)

    K_b  =  Sym( Σ_r  e_r ⊗ e_r ⊗ a_r ),     a_r = κ3_{rr·} (a column of the born (2,1) slice, plus the diagonal)

i.e. it is a sum over n "hub" neurons r of terms with two legs equal to the unit vector e_r and one leg a_r. Each later linear layer multiplies every leg by M_l = W_l D_l (D_l = diagonal of Wick coefficients), so at layer l > b the source is

    K_b^(l) = Sym( Σ_r  t_r ⊗ t_r ⊗ ã_r ),     T_b = M_l ⋯ M_{b+1} = [t_1 … t_n],   Ã_b = T_b A_b .

**Both "P" legs of every source are the propagator T_b itself; the "A" leg is the propagator applied to the birth matrix A_b (≈ diag(κ3_iii) + 3·(born (2,1) slice)ᵀ, full rank, "identity + O(1/√n) noise").**
What the next layer needs from all sources is only the (2,1) slice of the pre-activation third cumulant,

    D21^(l)_{ik} = Σ_b  [ (T_b ∘ T_b) Ã_bᵀ + 2 (T_b ∘ Ã_b) T_bᵀ ]_{ik} / 3        (∘ = Hadamard product)

(the covariance and the mean get pointwise corrections from D21 and from its diagonal; to first order the final mean depends on D21^(l) linearly through the Gaussian chain's adjoint), and the D21 of every layer must be accurate to about 2 % (extra MSE ≈ 4.2e-6·ε² — measured).

## What is measured (all on real 1024×16 networks unless stated; do not re-propose these)

1. Cost anatomy of V29: transports of T_b and Ã_b (2 products per source per layer) + the two slice products = 4 dense products per source-layer; with Strassen 2.05 u; 120 source-layers in a 16-layer net. Sources older than 4 layers are kept in a shared rank-384 basis (rank 224 after age 7): that tier costs 107 u of the 260. ARC's own reference chain (no age gating) costs 1523 u for the same accuracy class; its FLOP counter attributes exactly half to leg transport and half to slice formation.
2. The leaderboard leaders spend 133–165 u and reach raw 1.64–1.9e-8: **1.1–1.4 u per source-layer if they keep all sources, below the 2.05 u floor of this formulation, and 10× below ARC's reference.**
3. Old-source legs T_b are low-rank (99 % of energy in 100–260 directions, decreasing with age) but the Hadamard products T_b∘T_b and T_b∘Ã_b are full rank (680–780 of 1024) with a flat spectrum. This is the Khatri–Rao wall.
4. Parity: every source's contribution is odd under W_l → −W_l, so E[D21_b^(l) | covariance, Grams, all even statistics] = 0. Mean-field / even closures of the old tier lose the whole contribution (measured 9e-7 to 2e-6). The cheap rank-1 "partial-trace" part of (T_b∘T_b), 1·m_bᵀ, is already in V29 (its "thin legs"); it is the fluctuation part F_b = T_b∘T_b − 1 m_bᵀ, of relative size 1/√n, that costs the 107 u and is needed to 2 %.
5. Sensitivity: zeroing the old sources' D21 at any single layer ≥ 8 costs ≥ the whole error budget; dropping any single birth costs 1.3–2.7e-7 (flat over births); keeping old content only in the last 1–3 layers costs 15–25× the budget. Everything is needed, from every birth, at every layer.
6. Surrogates for the contracted query U = (T∘Ã) D Fᵀ that enters D21: isotropic trace 98 % error, diagonal Hermite 95 %, degree-2 TensorSketch 790 % (at width 2048); JL sketches of the hub index cannot preserve a trilinear sum.
7. Aggregate / pooled states: a pooled Tucker/TT core of the old sources needs multilinear rank ≈ 250 for 2 % D21 error (rank 128: 7.8 %); with a core of rank r the per-layer slice cost is n·r³/2 (13 u at 384, 4 u at 256, 1 u at 160) plus one n·r³/2 join per new source and 3r⁴ per basis rotation — never below V29's 10 u/layer at any rank that keeps 2 %. Vector–metric states Σ sym₃(G_a ⊗ v_a) with k ≤ 2 components: 87–89 % error. Cross-source response energy: 2 modes carry 26 %.
8. Slice recursion: the next layer's old D21 is *not* predictable from the current slices — regressing it on the exact slice transport gives R² = 0.03–0.05 (95 % of the next contribution comes from off-slice entries of the tensor).
9. Column (hub) screening at birth: drift grows quadratically in the dropped fraction, savings only linearly; net negative at every schedule.
10. Compensating closure with the old tier removed (retuning the κ4 regeneration, young amplitude, age gate): measured negative on 10 networks with a 1e9-sample truth: without the old tier the raw MSE is 7.4e-7 (32×) and no knob moves it (κ4 scale 0.5–2.0: 7.45–7.56e-7; young amplitude 0.85/1.15/1.3: 1.4e-6/2.9e-6/4.5e-5); keeping one more age gives 4.9e-7, one fewer 1.1e-6 — the omitted content decays only ×0.66 per age.
11. Accuracy ladder (ARC reference code, 256×16 networks, 1e9-sample truth): K3-simple 1.87e-6, K3 augmented (dense, keeps all Edgeworth terms) 1.25e-6, V29 on the same nets 1.99e-6; augmentation buys 1.5× for 8× the cost. K=4 rung (synthetic 128×16, 2e9-sample truth): K2 7.79e-5, K3-simple 1.23e-5, K3-augment-dense 9.97e-6, K4-simple 7.60e-6 (516 u already at width 128: n⁴-class, impossible at 1024). The leaders' 1.3× accuracy margin over V29 is exactly one rung of this ladder, so they also carry one more cumulant order (or the dropped augment terms) cheaply — a second thing we would like you to price.
12. Engineering: Strassen level 6 exceeds the 8 GB address-space cap; level 7 self-cancels; float16 is billed as float32; no vendored libraries, no threads, only numpy-like ops through the metering layer.

## The question

Given the explicit structure — **the same propagator T_b is two of the three legs of every source, and the sum over sources is a sum over birth layers of (T_b∘T_b)Ã_bᵀ + 2(T_b∘Ã_b)T_bᵀ with T_b = T_c T_{b→c} for b < c** — is there any way to obtain Σ_b over all births at a cost of ≤ 1.2 u per source-layer (equivalently ≤ 8 u per layer for all old sources together, at 2 % accuracy), or a different state whose per-layer update is O(few n³) and which reproduces the fluctuation part F_b Ã_bᵀ to 2 %?

Concretely, for each proposal give: (a) the state kept per layer and its size; (b) the update, with the FLOP count in units of n³ (be careful with Khatri–Rao products: n × r² × n is *not* n²r); (c) why it reproduces the odd, full-rank fluctuation part rather than its (zero) conditional mean; (d) a falsifier we can run on one 1024×16 network in under an hour with numpy.

Directions we have **not** measured and would like you to assess first: (i) exploiting T_b = T_c T_{b→c} to share the *squared* propagator across births (is (T_c X)∘(T_c X) expressible through (T_c∘T_c) and X-dependent objects of size n²?); (ii) Hermite/Wick expansion of (T_b∘T_b) in the *columns* of M_l (rows of W_l are i.i.d. Gaussian: is the fluctuation part a low-order chaos in W_l whose coefficients are cheap?); (iii) treating the old tier as a *linear response* problem and transporting adjoint sensitivities backwards through fewer layers than the sources travel forwards; (iv) any representation from turbulence closure / kinetic theory / tensor-network literature for "sum over births of odd, full-rank, self-similar contributions".

If your answer is "the floor is 2 u per source-layer and the leaders must be dropping or approximating sources", say exactly what approximation could lose < 1.3× in MSE while halving the bill, given items 5 and 9.
