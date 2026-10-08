# Round 17 (complete prompt, with clarifications) — ONE question: the cheapest FACTORISED κ₄ carrier that keeps the all-distinct post-ReLU sectors

*(Answer in English, at most 4 pages. Give ONE concrete carrier, its exact cost at the real shapes, and ONE falsifier as a
SELF-CONTAINED PYTHON SCRIPT. Run it on our data yourself if you can reach it (as you did in Round 16b) and embed the script
and its JSON in the PDF; otherwise we run it here. Mark every number as derived, estimated, quoted or measured. Do not
re-propose anything in the dead list unless you show why its kill was too narrow.)*

This prompt merges the original Round 17 question with our answers to your two clarifying questions: where the
representation should live, and the exact curvature birth terms (verified numerically).

## Setting (short)

The network is a bias-free ReLU MLP of width n = 1024 and depth L = 16, with He weights and input x ~ N(0, I). The task is to estimate the final-layer post-ReLU means.

Score = final MSE × FLOPs / 2⁴¹. One u = 2³¹ FLOPs ≈ one 1024³ product; a level-5 Strassen product ≈ 0.51 u.

Our estimator is the public factorised third-cumulant chain at ≈ 192 u:
- it carries the mean, a dense covariance C, and κ₃ as a sum over per-ReLU "sources";
- young sources have dense n × n legs; old sources share a rank-352 basis;
- every source leg is transported each layer by U = diag(Φ(α)) W.

Its κ₄ is memoryless: at each layer a core diag(g) + λ C_pre,off is regenerated, and only the diagonal is carried, by (W∘W). The κ₄ diagonal enters only the univariate Edgeworth mean closure.

Graded now: raw 2.06e-8 at utilisation 0.187, score 3.86e-9. The grader allows ≤ ~18,000 metered operations per network.

Leaderboard reference:
- J2W: raw 1.47e-8 at utilisation 0.102.
- Masato: raw 1.46e-8 at 0.151.

## Established (public MLP #0; teacher = 6×10⁷-sample Monte Carlo)

1. **The accuracy gap is the PATTERN of the deep κ₄ diagonal.** The teacher's diagonal pattern, at the chain's level, fed into the mean closure takes raw 2.29e-8 → 1.55e-8 (108% of the full gain). A 50% blend gives 74%; layers ≥ 8 give 86%.
2. **The pattern is carried memory.**
   - The exact LINEAR carry of the full true pre-activation κ₄ tensor one layer, y = (Φ(α) ∘ (z − μ)) W, gives decisive-neuron pattern correlation 0.77-0.80 at L12-16 (Monte Carlo: `experiments/probes/opus12_k4_carry.py`, LIN).
   - Erasing the non-Gaussian history h layers back leaves correlation 0.41 (h = 2), 0.59 (h = 3), 0.77 (h = 6).
   - Fresh ReLU physics from a Gaussian input is only 1-4% of the deep level.
   - Adding the leading ReLU curvature to that one-step carry (the "QUAD" response, section A below) raises the correlation with the teacher from 0.77 / 0.79 / 0.80 to 0.82 / 0.86 / 0.86 at L12 / L15 / L16. The curvature births matter.
3. **NEW (Round 16, measured): the missing pattern lives in the post-ReLU entries κ(a,a,b,c) and κ(a,b,c,d).** One step from the TRUE post-ReLU teacher state:

   | state carried one step | pattern corr at L8 / L12 / L15 / L16 |
   |---|---|
   | (4), (3,1) and (2,2) slices, carried exactly | 0.45 / 0.39 / 0.30 / 0.38 |
   | full tensor | 0.77-0.80 |

   So about 0.4 of correlation comes only from entries with three or four distinct indices. Every slice-based or matrix-based state is capped near 0.4.
4. **Per-layer chain features saturate at ≈ 0.5 (V55, shipped):** −5.4% on the grader.
5. **The κ₄ is odd in the later weights** (parity), so even statistics carry nothing.

## Measured dead (do not propose)

**Matrix or slice states (capped by fact 3)**
- Round 16 SBQM (Sym(H ⊗ C), H carried by Wᵀ H W, diagonal refresh). Teacher-forced one-step ceiling 0.32-0.39 at L12/15/16, the same as carrying the diagonal alone. The K31/K22 slices cannot both be fitted by Sym(H ⊗ C).
- CPR, one or two carried response matrices H ← Uᵀ H U with births and a pair readout (×0.98-1.00).
- Scale-mixture / Sym(C ⊗ C) predictors (correlation ≤ 0.09).
- A carried pair-response mode; SSC response closure.

**Carriers whose births hold only (4) + (2,2) content (Round 16b, Astra ran it, we reproduced it)**
- A dense pair-innovation core per ReLU, carried by exact source transport:
  - diagonal Δ4, off-diagonal 3 Δ22 of post- vs linearised pre-ReLU slices;
  - readout Σ_b ((P∘P) G_b ∘ (P∘P)) row sums.

  Its ceiling is 0.37-0.44 at L12-16, for a six-age window and for all births alike, even though transport gives it all-distinct entries.
- So the BIRTH must contain the newborn (3,1), (2,1,1) and (1,1,1,1) content. Long memory of a (4)+(2,2) birth is not enough.

**Readouts of cubic history**
- Round 15 source-resolved skew energy: +0.004 correlation at a per-layer oracle. The signed cubic history does not predict κ₄.
- κ₄ riding the κ₃ source legs (×1.05 out of sample).

**Short horizons and closed forms**
- A young-source κ₄ carrier over only 2-3 layers (0.210 → 0.203 error). Its horizon is below the measured 6 layers: you may argue it was too narrow.
- The exchange term 3 Y C⁻¹ Yᵀ; per-layer refits of λ; learned per-neuron regressions; cutting κ₃ cost and recovering it with κ₄.

## A. Where the representation should live (answer to your first clarifying question)

Model the BIRTH directly in post-ReLU space, then map it to pre-activation space through the weights. Do not keep the whole
representation in pre-activation space with post-ReLU information entering only through C and the κ₃ source legs.

- The missing pattern lives in the post-ReLU entries κ(a,a,b,c) and κ(a,b,c,d) (fact 3).
- The "pre-activation only, injected through C and the κ₃ legs" route is already dead three times:
  - κ₄ riding the κ₃ source legs (×1.05 out of sample);
  - Round 15 skew energy (+0.004 correlation);
  - per-layer chain features (saturate at ≈ 0.5).
- The ReLU curvature births measurably add correlation (fact 2: 0.77-0.80 → 0.82-0.86).

What we expect in practice:
- At each ReLU, a factorised post-ReLU quartic birth is formed. Its all-distinct entries come from:
  - the Φ-weighted pass-through of the carried pre-activation κ₄;
  - the curvature births (section B).
- The birth is mapped once by W into pre-activation space, carried by U = diag(Φ) W like our κ₃ sources, and read out as the
  pre-activation diagonal.
- You may use any post-ReLU quantity the chain already has: Φ, φ, the post-ReLU C, the κ₃ source legs and hub weights, D21
  and D3.
- The form must be factorised (hub / source): no dense n⁴ or n³ objects.

## B. The curvature birth terms, exact at second order (answer to your second clarifying question)

### Notation

All quantities are at one layer l; "pre" means the pre-activation z, "post" means h = relu(z).

- u = z − μ, the centred pre-activation.
- Cumulants of u: C (covariance), T_abc = κ₃(u), K_abcd = κ₄(u), κ₅, κ₆. The chain does not carry κ₅ or κ₆.
- α = μ/σ, Φ_a = Φ(α_a), and g_a = φ(α_a)/(2σ_a).

### Model: second-order Hermite expansion of ReLU around the mean

    h_a − E h_a ≈ Φ_a u_a + g_a (u_a² − v_a)

This is the "QUAD" response of `experiments/probes/opus12_k4_carry.py`. It was measured one step, with exact Monte Carlo samples of u (so every cumulant order of u is included), as pattern correlation with the teacher:

| response | L12 | L15 | L16 |
|---|---|---|---|
| linear only, Φ u | 0.77 | 0.79 | 0.80 |
| with the g-term | 0.82 | 0.86 | 0.86 |

### B1. Post-ReLU fourth cumulant, any indices (a, b, c, d), including all-distinct

By multilinearity, κ(h_a, h_b, h_c, h_d) is the sum over which arguments take the quadratic piece. The constant −v_a never contributes to cumulants of order ≥ 2.

**(L4) No quadratic piece: the pass-through, i.e. the carried memory of the incoming pre-activation κ₄ in all sectors**

    Φ_a Φ_b Φ_c Φ_d K_abcd

**(Q1) One quadratic piece, on a (sum over the 4 positions)**

    g_a Φ_b Φ_c Φ_d κ(u_a², u_b, u_c, u_d),
    κ(u_a², u_b, u_c, u_d) = κ₅_aabcd + 2 [ C_ab T_acd + C_ac T_abd + C_ad T_abc ]

The C·T part is the leading NEW birth of all-distinct entries. One curvature hub a is joined to one index through C and to the other two through the κ₃ tensor.

**(Q2) Two quadratic pieces, on a and b (sum over the 6 position pairs)**

    g_a g_b Φ_c Φ_d κ(u_a², u_b², u_c, u_d),
    κ(u_a², u_b², u_c, u_d) = κ₆_aabbcd
        + 2 C_ac K_abbd + 2 C_ad K_abbc + 2 C_bc K_aabd + 2 C_bd K_aabc + 4 C_ab K_abcd     [C·K]
        + 4 T_abc T_abd + 2 T_aab T_bcd + 2 T_abb T_acd                                     [T·T]
        + 4 C_ab ( C_ac C_bd + C_ad C_bc )                                                   [Gaussian path]

**(Q3, Q4) Three or four quadratic pieces**

These are higher order in g. Their Gaussian parts are, for example, the 4-cycle 16 C_ab C_bc C_cd C_da + perms. Gaussian-input fresh physics measures only 1-4% of the deep κ₄ level, so the Gaussian-only parts are minor. The information is in the C·T term (Q1) and the C·K and T·T terms (Q2).

### B2. Readout through the next weights

Row convention z′ = h W, and U = diag(Φ) W. The next pre-activation diagonal is

    κ₄(z′)_i = Σ_abcd W_ai W_bi W_ci W_di κ(h_a, h_b, h_c, h_d)

**C·T curvature birth (Q1)**

    D_i^CT = 24 Σ_a g_a W_ai (C U)_ai · T(a; U_:i, U_:i),   where T(a; x, x) = Σ_cd T_acd x_c x_d

- With the chain's κ₃ factorised by sources (hub-j legs, as in `blocks_of_source`: X1 P Y1 + M P² + X3 P y), T(a; U_:i, U_:i) uses the SAME transported legs the chain already forms for its next-layer κ₃ diagonal D3.
- Hence D_i^CT = 24 Σ_s Σ_j (Mᵀ X_s)_ij (A′_s ∘ B′_s)_ij, with M_ai = g_a W_ai (C U)_ai and A′, B′ the transported legs.
- Cost: one n × n product per source block, plus elementwise work.

**Gaussian path (Q2, Gaussian part)**

    D_i^path = 48 v_iᵀ C v_i,   v_a = g_a W_ai (C U)_ai

- Cost: one n³ product (C V).
- It is minor, by the fresh-physics measurement.

### B3. Numerical verification (done here, 8 Oct)

Script: `experiments/probes/r17_curvature_identities.py`; results: `results/summaries/r17_curvature_identities.json`.

Method:
- An exact empirical measure on 40 weighted points in n = 6, built non-Gaussian, skewed and correlated.
- Every joint cumulant is computed exactly from the partition formula, so no Monte Carlo noise enters.
- The readout factors are checked against the full 4-tensor contraction with random W, Φ and g.

| check | max relative error |
|---|---|
| Q1 identity (all-distinct and coinciding index tuples) | 5.1e-16 |
| Q2 identity (same tuples) | 2.4e-16 |
| readout factor 24 for the C·T birth | 3.5e-15 |
| readout factor 48 for the Gaussian path | 4.1e-15 |

All formulas above are therefore exact identities, to rounding. The only approximation is the quadratic Hermite truncation of ReLU, whose accuracy is the measured 0.82-0.86.

### B4. What the carrier must keep

The born term from Q1 has a two-hub factorised form, with curvature hub x and κ₃ hub j:

    Σ_x Σ_j g_x · X_xj · [ e_x ⊗ C_x: ⊗ A_:j ⊗ B_:j ]   (symmetrised)

- Its four legs (e_x, the C row of x, and the two κ₃ legs of hub j) are each transported by U, exactly like κ₃ source legs.
- The readout at any later layer is the diagonal contraction of the four transported legs.

**Design question for you:** the cheapest factorisation of this two-hub object, and of the C·K and T·T terms, that keeps it alive for ≥ 6 layers. Options include:
- reuse the κ₃ source legs and the C U product the chain already computes;
- low rank in the curvature hub x (only decisive neurons have large g_x = φ/(2σ));
- age truncation.

The L4 pass-through must also carry the previous births. Round 16b showed that a birth with only (4) + (2,2) content is capped at ~0.43. So the Q1 C·T birth, and probably the Q2 C·K and T·T terms, are what must be added.

## The question

**Design the CHEAPEST factorised representation of the carried pre-activation κ₄ tensor that keeps its all-distinct sectors.**
- Its diagonal must reach decisive-neuron pattern correlation ≥ 0.7 at L12-16.
- The representation must be a sum over births (ReLUs / layers) of structured terms, like our κ₃ sources, carried for ≥ 6 layers.
- Build the birth from section B. At least the Q1 C·T term, and probably the Q2 C·K and T·T terms, plus the L4 pass-through of earlier births. State which terms you keep and why.
- State how each term is transported by U = diag(Φ) W and how the diagonal is read out.
- State what can be truncated (age, rank, sector) while keeping correlation ≥ 0.7.

**Cost envelope (derived break-even)**

A carrier worth 74% of the oracle gain multiplies raw MSE by ≈ 0.78. At +30 u on ≈ 192 u the score improves ≈ 10%; at +60 u it is already ≈ 2% worse. So it must cost ≤ ~30 u, ≤ ~1,000 extra metered operations, and fit 8 GB of memory.

If nothing meets the cost, give the cheapest carrier that meets the CEILING (exact, untruncated) and its cost.

**Deliverable**
1. The equations: birth, transport and readout, with shapes.
2. The cost per layer: dense products, FLOPs and metered operations.
3. Why it captures the κ(a,a,b,c) and κ(a,b,c,d) sectors with ≥ 6-layer memory, and why it is not a dead item.
4. ONE self-contained falsifier script, ceiling first; run it yourself on our data if you can, and embed the script and JSON. It must run the exact (untruncated, Monte-Carlo-backed if needed) version of your carrier on public MLP #0 and print decisive-neuron pattern correlation at L12, L15 and L16.
   - Follow the Monte Carlo template of `experiments/probes/opus12_k4_carry.py`: exact samples through the true MLP, float32 forward, float64 accumulation, N = 2×10⁶ by default.
   - It must run on one CPU or GPU in ≤ ~2 hours. We will run it on Kaggle if needed.
   - Pre-register the gate: PASS ≥ 0.70 at all of L12/L15/L16; KILL if < 0.55 at all three.

## Stored data (paths relative to the project root)

- **Pre-activation teachers, MLP #0, all 16 layers:** mean, C, D21 = Y21, K22, K31 and K4d, in `data/kaggle_out/kaggle_out_k4pub/k4teach/layer{l}_seed0.npz`, plus `weights_seed0.npz` (W stacked 16 × 1024 × 1024, row-vector convention z_{l+1} = h_l @ W[l]).
- **Post-ReLU teachers, layers 3/7/11/14/15:** the same arrays, in `data/kaggle_out/kaggle_out_k4post/k4post/layer{l}_post_seed0.npz`.
- **Slice conventions:** K31_ij = κ(i,i,i,j); K22_ij = κ(i,i,j,j).
- **Synthetic second network:** pre-activation teachers at layers 2, 4, 6, 8, 10, 12, 14 and 15, in `data/kaggle_out/kaggle_out_k4/k4teach`.
- **Dense captures of the chain's κ₃ sources at all layers:** `data/dumps/r7_cap_L{1..16}.npz`. Factor reconstruction is `blocks_of_source` in `experiments/probes/r7_srcgain.py`.
- **The chain's own κ₄ diagonal:** `data/dumps/errbudget_v29plain.npz` (`L{l}_ck4`).
- **The graded estimator:** `submissions/v55k4c/estimator.py`.

## Time budget

Today is 8 Oct 2026; the competition closes 18 Oct (Bangladesh time). We need the ceiling verdict by 10 Oct and a prototype by 12 Oct. A carrier whose ceiling passes but costs more than 30 u is still worth reporting.
