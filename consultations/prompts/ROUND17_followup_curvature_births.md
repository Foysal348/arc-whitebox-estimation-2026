# Round 17 follow-up — the curvature birth terms of the post-ReLU κ₄ (exact at second order, numerically verified)

Answer to: "Could you provide the precise analytic expression for the curvature birth term that populates the post-ReLU κ₄'s (2,1,1) and (1,1,1,1) entries?"

## Notation

All quantities are at one layer l; "pre" means the pre-activation z, "post" means h = relu(z).

- u = z − μ, the centred pre-activation.
- Cumulants of u: C (covariance), T_abc = κ₃(u), K_abcd = κ₄(u), κ₅, κ₆. The chain does not carry κ₅ or κ₆.
- α = μ/σ, Φ_a = Φ(α_a), and g_a = φ(α_a)/(2σ_a).

## Model: second-order Hermite expansion of ReLU around the mean

    h_a − E h_a ≈ Φ_a u_a + g_a (u_a² − v_a)

This is the "QUAD" response of `experiments/probes/opus12_k4_carry.py`. It was measured one step, with exact Monte Carlo samples of u (so every cumulant order of u is included), as pattern correlation with the teacher:

| response | L12 | L15 | L16 |
|---|---|---|---|
| linear only, Φ u | 0.77 | 0.79 | 0.80 |
| with the g-term | 0.82 | 0.86 | 0.86 |

## 1. Post-ReLU fourth cumulant, any indices (a, b, c, d), including all-distinct

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

## 2. Readout through the next weights

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

## 3. Numerical verification (done here, 8 Oct)

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

## 4. What the carrier must keep

The born term from Q1 has a two-hub factorised form, with curvature hub x and κ₃ hub j:

    Σ_x Σ_j g_x · X_xj · [ e_x ⊗ C_x: ⊗ A_:j ⊗ B_:j ]   (symmetrised)

- Its four legs (e_x, the C row of x, and the two κ₃ legs of hub j) are each transported by U, exactly like κ₃ source legs.
- The readout at any later layer is the diagonal contraction of the four transported legs.

**Design question for you:** the cheapest factorisation of this two-hub object, and of the C·K and T·T terms, that keeps it alive for ≥ 6 layers. Options include:
- reuse the κ₃ source legs and the C U product the chain already computes;
- low rank in the curvature hub x (only decisive neurons have large g_x = φ/(2σ));
- age truncation.

The L4 pass-through must also carry the previous births. Round 16b showed that a birth with only (4) + (2,2) content is capped at ~0.43. So the Q1 C·T birth, and probably the Q2 C·K and T·T terms, are what must be added.

## Gate and deliverable (unchanged)

- Ceiling first: exact, untruncated, Monte-Carlo-backed if needed.
- PASS: ≥ 0.70 at L12, L15 and L16. KILL: < 0.55 at all three.
- Cost: ≤ ~30 u and ≤ ~1,000 extra ops.
- Run the script on our data yourself and embed the script and its JSON, as in Round 16b.
