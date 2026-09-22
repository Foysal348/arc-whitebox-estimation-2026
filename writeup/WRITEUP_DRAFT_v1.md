# Where the deep third cumulant of a random ReLU MLP lives — and why a sum-of-sources chain cannot carry it below ~0.24 B

**Phase 2 algorithmic-contribution write-up · submission #331686 (adjusted 5.220e-9, raw final-layer MSE 2.132e-8, 0.245 B, 0 failed MLPs) · participant foyshal_sk (Foysal Emon Shanto)**

*Draft v1, 21 September 2026. Sections marked [TODO] are to be completed before the 24 October deadline.*

---

## Abstract

We start from the public factorized K=3 cumulant-propagation chain released under MIT by team 504aldo ("V29": raw 2.13e-8 at 0.254 B) and ask a single question: **can the "old-source tier" — the 107 of 260 cost units spent re-forming and re-contracting third-cumulant sources older than four layers — be made cheap without losing accuracy?** The leaders' bills (133–165 units) equal V29 *without* that tier to three digits, so this is the question the leaderboard poses. We answer it negatively, from six independent directions, with one theorem and five measurements on real 1024×16 networks:

1. **Parity.** A source's contribution to the ReLU slices at layer *l* is an odd function of the layer-*l* weights; its conditional expectation given any even statistic (covariance, Grams) is exactly zero. There is no mean-field part to keep. (Corollary: the empirical κ4 law G_off ≈ λ_l C_off follows from evenness and isotropy.)
2. **Spectra.** Old-source legs are low-rank (99% of energy in 100–260 of 1024 directions, decreasing with age) but their Hadamard products are near full rank with a flat, noise-like spectrum (680–780 directions) — the Khatri-Rao wall is numerical, not only algebraic.
3. **Sensitivity map.** Zeroing the old contributions at any single layer ≥ 8 costs as much as V29's whole error budget; dropping any single birth costs 6–13× that budget; keeping old content only in the last 1–3 layers costs 15–25×. The content is needed at every layer from every birth.
4. **Query surrogates.** The contracted query U = (A∘P)·D·F_Aᵀ that actually enters D21 cannot be approximated without forming the dense legs: isotropic trace 98%, diagonal Hermite 95%, degree-2 TensorSketch 790% error at width 2048.
5. **Aggregate states.** The one representation eight consulted models converged on — an aggregate "vector–metric" state Σ_a sym₃(G_a ⊗ v_a), or a pooled TT/Tucker core — needs k ≈ 250–300 components (rank ≈ 250) for 2% query error; budget-compatible k ≤ 2 gives 87–89% error. The cross-source response rank confirms it: two modes carry 26% of the sources' query energy, not 99.96%.
6. **Engineering floor.** Strassen recursion below 32×32 leaves (−3.9% cost at prediction drift 5e-13; graded 5.418e-9 → 5.220e-9) is the last free trim; level 6 breaks the 8 GB cap, level 7 self-cancels, and the author's fitted constants are at their optimum (Kaggle sweep, 10 baked-truth MLPs).

The conclusion is a quantitative one: within the sum-of-sources representation the old tier is irreducible, so the leading methods must use a different formulation of the deep third cumulant — one that eight models could not name. We state precisely what such a formulation must satisfy.

## 1. Setting

[TODO: 1 paragraph — problem, score, budget, the V29 chain in five lines (sources, legs, slices, shared basis, cost anatomy 116/107/25/7/6 units). Cite arXiv:2605.05179 §4.3 and the 504aldo release.]

## 2. What the question is

[TODO: the cost identity "leaders = V29 − old tier"; tolerance law extra MSE ≈ 4.2e-6·ε² (D21 to 2%); definitions of H, T2, T3, T4, D3; why re-forming dense legs is forced by A∘P.]

## 3. Theorem: no mean-field part (parity)

**Statement.** Let a source be born at layer *b*. For any later layer *l*, its legs are P = W^(l) D P′, A = P a_b with P′, a_b and the birth coefficients independent of W^(l). Every quantity linear in the source's κ₃ (its contributions H, T2, T3, T4, D3) is therefore odd under W^(l) → −W^(l), while the covariance C^(l), the leg Grams and all per-neuron variances are even. Since the weight ensemble is sign-symmetric,
E[ H_s^(l) | C^(l), Gram(A_s, P_s), state at layers < l ] = 0.

**Consequences.** (i) Any closure that replaces old sources by a function of even statistics — mean-field, scale mixtures, "self-averaging" — discards the entire contribution; the measured failures of such closures (9.1e-7, 2.3e-6) are not tuning failures. (ii) Expectation-only replacement conditional on one extra fresh row has relative error ≥ √(6/(r+8)) (12% at r=384, 16% at r=224), an order of magnitude above tolerance. (iii) κ₄ terms built from two κ₃ contributions are even; by rotational invariance the only available invariant is C⊗C, which explains V29's regeneration law G_off = λ_l C_off (R² 0.66–0.97) as an isotropy consequence rather than a fit.

[TODO: 1-page derivation; credit: the argument was obtained with LLM assistance (Fable 5) and checked by hand; the floor bound was derived independently by two other models and agrees.]

## 4. Measurement A — legs versus their products

[TODO: table (layer 6/10/14 × old/young × A, P, A∘P, A∘A, P∘P; r90/r99/r999); method: probe_rank.py; interpretation: r99(A∘P) ≈ 700 while r99(A) ≈ 100–260; top singular value ≈ 1/10–1/30 of Frobenius; Khatri-Rao rank min(n, r(r+1)/2) = n for r ≥ 45.]

## 5. Measurement B — where the old content matters

[TODO: three tables: keeplast_k; dropold per layer (L5 3.4e-9 … L14 8.2e-8); dropslot per birth (1.3e-7 … 2.7e-7). Method: sens_probe.py, zeroing dense + thin legs before the slice contractions, thin legs restored after; metric = MSE against the full chain on the same network; additivity check (Σ_layers ≈ keeplast).]

## 6. Measurement C — the contracted query cannot be surrogated

[TODO: definition of U and its siblings; table of surrogate errors at layers 10/14; the SVD ceiling (r99 ≈ 100, 2% at ≈ 230); why the TensorSketch variance scales as 1/√d (approximate matrix multiplication of two noise-like factors); method: probe_query.py / probe_query_analyze.py.]

## 7. Measurement D — aggregate states

[TODO: the vector–metric family and its exact transport; the Gram trick for the (ab)|c unfolding SVD; k-sweep table (pooled: k=1 89% … k=128 20%; single source k=128 4.2%); TT/Tucker unfolding ranks of the pooled core (rank 256 for 1.6%); cross-source response energy (1/2/4 modes: 0.16/0.26/0.46); cost arithmetic 2k units/layer vs the 0–22 unit window; method: probe_agg.py.]

## 8. Engineering results on the public chain

[TODO: STRASSEN_MIN 32→8 mechanism (rank-384/224 families gain levels; additions vs savings per level: 77 vs 138 MFLOP at level 6, 135 vs 120 at level 7); memory 5.56 / 5.97 / 8.13 GB; graded results; knob sweep table; residual-time facts.]

## 9. What the leaders' formulation must satisfy

From §3–§7, any method that reaches raw ≤ 2e-8 at ≤ 0.16 B must carry a **realized, odd, full-rank** object whose slices are needed at every layer from every birth, with a per-layer cost ≤ 2 dense matmuls for *all* deep content. It cannot be a sum over sources, a low-rank or sketched version of the products, an even closure, a k ≤ 2 aggregate state, or a pooled core of rank < 250. Candidates that survive on paper (untested): [TODO — list the two low-prior directions and the "different closure with compensating errors" hypothesis; state the falsifiers].

## 10. Negative results and things we did not do

**Slice recursion (measured, pilot E).** Regressing the next layer's old H-contribution on its exact slice transport (B∘B)·H·Bᵀ recovers the theoretical coefficient 1 but only R² = 0.03–0.05: about 95% of the old contribution at layer l+1 comes from off-slice (a≠b) entries of the tensor at layer l. The odd covariance feature W·C has R² = 0.0000. Any recursion on (C, D21, D3) is therefore hopeless, more so than the author's 18–23% figure suggested.

[TODO: L6/L7 Strassen; AGE_OLD 5 (raw −2%, cost +3%); ideas rejected on paper (JL sketch of the birth index; bilinear birth Grams; even closure of the odd remainder; adjoint; shadowing).]

## 11. Reproducibility

All measurements are on one locally generated network (build_mlp seed 7, width 1024, depth 16) unless stated; graded numbers are from the public leaderboard (submissions #331681, #331686); the Kaggle sweep used 10 MLPs of the mini split with the baked 1e9 truth. Scripts: probe_rank.py, sens_probe.py, probe_query.py, probe_query_analyze.py, probe_agg.py, probe_gem.py, knob_test.py, and the Kaggle runner (kernel foysalemonshanto/whest-runner). [TODO: publish under MIT with the write-up; pin versions whestbench 0.16.1 / flopscope 0.12.1.]

## 12. Disclosure

The code changes to the estimator are limited to one constant (STRASSEN_MIN) on 504aldo's MIT-licensed V29, with attribution retained. Experiments, derivations and this text were produced with LLM assistance (Claude Opus 5, directed by the participant, who ran every experiment and read every number); the parity argument came from a consultation with Fable 5 and was checked by hand; round-1/2 consultations with GPT-5.6 Sol, GPT-5.6 Terra, Astra 6 Ultra, Gemini 3.1 Pro, Grok, DeepSeek and Copilot supplied the candidate ideas that §7 and §10 test. Claims are labelled derived / measured / conjecture throughout. [TODO: per-section labels.]
