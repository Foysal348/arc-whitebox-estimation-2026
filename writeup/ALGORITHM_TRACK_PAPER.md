# Where the Error of the Factorised Cumulant Chain Lives: Exact Identities, Cost Engineering, and a Cubic-Birth κ₄-Pattern Programme for Deep ReLU MLPs

**WhestBench Phase 2, algorithmic-contribution write-up. Team Hack2Publish (participant foyshal_sk).**
**Best graded submission: #334870 (v56As_fs), adjusted score 3.6521e-9, raw final-layer MSE 1.950e-8. Best nomination-safe submission: #334869 (v56s_fs), 3.6723e-9.**
*Draft of 11 October 2026. Every number is either a graded leaderboard result (submission id given) or a measurement whose result file is named in §8. Items still open are marked **[OPEN]**. Unmeasured quantities are marked **(estimate)** or **(untested)**. Claims are labelled **derived** (algebra), **measured** (run by us) or **graded** (official grader).*

---

## Abstract

We start from the public, MIT-licensed factorised third-cumulant chain of team 504aldo ("V29", raw 2.13e-8 at 0.254 of the budget) and pursue two questions: where its error is made, and what an estimator must contain to remove that error at an affordable cost.

**Contributions.**
1. **Two exact identities that we ship.** A spherical-input identity turns the Gaussian input into a uniform-sphere input plus one exact scalar, a₁ = 0.99975589. It is free and graded −1.1 % raw (#331787).
2. **A parity theorem.** It explains why every "mean-field" closure of the deep third cumulant fails.
3. **Cubic curvature-birth identities.** These come with readout factors, verified to ≤ 1e-14.
4. **Cost engineering under a metered budget.** Our tools are dead-neuron compaction, basis-side Strassen, aliased Grams, and an operation budget derived from a residual-time grader failure. Together they remove 23 % of the metered FLOPs for +2.8 % raw. With the zero-cost accuracy fixes they take the score from 5.418e-9 to 4.078e-9.
5. **An error map built from 6·10⁷-sample Monte-Carlo teachers.** The closures are exact given true inputs. The error is fresh input error that accumulates through depth. The single most valuable missing input is the **per-neuron pattern** of the κ₄ diagonal: the teacher's pattern at the chain's own level would take the raw MSE from 2.290e-8 to 1.551e-8, which is the leaders' raw range.
6. **A κ₄-pattern programme with graded gains.**
   - A zero-FLOP linear corrector (V55) graded −5.4 % (#334591).
   - A cubic Hermite-star feature built from the chain's own young legs (V56) graded a further −4.4 % (#334805), reaching 3.6521e-9 (#334870).
7. **A ceiling ladder that explains why every earlier κ₄ carrier stalled near correlation 0.45.**
   - Carriers whose births stop at second order in the Hermite expansion of the ReLU are capped at 0.45–0.56.
   - Cubic births over a 6-layer horizon reach 0.80–0.85.
   - An age-resolved ceiling (9 Oct) shows that ages 1–6 are jointly needed. Ages 1–4 give 0.68–0.74, ages 2–6 give 0.69–0.71, and ages 4–6 give only 0.37–0.41.
8. **A table of the more than 60 architecture and representation ideas we killed, with their ceiling numbers.** It includes outside-AI proposals and states which of them came with no measurement.
9. **Derived constraints on any better method.** The next design target is stated quantitatively.
10. **The κ₃ input and the κ₃-dressed C·T channel (9–10 Oct, oracle and component level; not graded).**
    - A joint oracle shows that the teacher's κ₃ input is the larger single lever: ×0.607 against ×0.686 for the κ₄ diagonal on MLP #0, and ×0.716 against ×0.885 on the synthetic net. Together the two give ×0.272.
    - A C·T readout built from the chain's own κ₃ legs, ct_i = 24 Σ_a g_a Ȳ_ai X_ai Q_ai, lifts the star's ceiling from 0.47–0.52 to 0.82–0.85. Used as a feature in a teacher-fitted (oracle) affine map, it takes the plain chain to ×0.773, which is about 75 % of the perfect-κ₄ gain.
    - An exact age-collapse identity, Γ_a = G_a + Ystep_a Γ_{a+1}, together with grouping into dense A/P anchors plus exact low-rank anchors, cuts the source contraction to two dense products per source and target, plus one shared backward recursion. Low-rank sketches keep only ×0.843–0.885. At an estimated 40 u this is about score-neutral, so the channel is real but not yet affordable. A rank-32 eigen-sketch (10 Oct) keeps ×0.874 at half the rank-64 thin-product count, an estimated 3–4 % score gain on the plain chain under the same oracle map (untested on V56).
11. **Where the κ₃ error is, and three mechanisms killed (10 Oct; oracle and component level; not graded).**
    - Measured against an estimate of the teacher's own sampling noise, the chain's early-layer κ₃ "error" is mostly teacher noise. The real (2,1)-slice error grows from about 3 % at L4 to about 7 % at L16, and the d3 level shrinks by 1.4 % at L16. The teacher's κ₃ helps only at L9–16 (§4.10).
    - A missing cubic-response C×κ₃ term, a feedback-scale confound and the κ₄→κ₃ feed channel were each tested and killed. The last kill holds only for the tested sample size, N = 10⁶ (§6.5).
    - Local corrections of one channel repeatedly make the final mean worse; only joint input oracles give large gains (§7.5).
12. **Where the shipped cost goes, and why the cost-side architectures fail (10–11 Oct; measured, partly on 100 MLPs).**
    - A call-site ledger of V56 (§3.6) puts 193.8 u and 20.6k operations in one warm call. About 70 u goes to the D21 hub readout, 63 u to young-leg transport and 20 u to the old basis; the rest is about 40 u.
    - One shared basis cannot hold the whole κ₃: at rank 512 the D21 error is 3.4–6.3 % at L11–15, and only rank 768 reaches 0.2–1 % (§6.6 (a)). Keeping only the youngest source exact (a deferred exact join) leaves 2.5–8 % (§6.6 (b)).
    - On 100 MLPs every age/rank setting near V56's scores worse (×1.026 to ×1.104), so the one-for-one law of §3.5 holds on V56 (§6.6 (c)).
    - Zeroing κ₃ in the closures for a window of layers costs ×15 to ×125, and the damage does not recover after the window (§6.6 (d)). A ±2 % change in the κ₃ level is not a gain either (§6.6 (e)).
    - The reference K = 3 implementation of Wu et al. costs about 4,150 u at our size, about 21× V56 (derived from their cost table, §3.6). Its trace projection of old κ₃ leaves 26–42 % D21 error (§6.6 (f)).
13. **A learned pair-memory student, PM364 (11 Oct; killed).** We distilled V56 into a source-free state with two n×n pair memories, an exact port of V56's closure and 364 learned parameters. The student fits under the 0.1 floor (about 93 u, estimate). It trained on 448 V56-labelled networks, and its held-out error moved by only −0.9 %. Its D21 error stays at 58–70 %, against a 2 % gate (§6.7). Per-row heads cannot rebuild the all-distinct post-ReLU κ₃, which agrees with a derived first-order obstruction for slice-only states (§6.6 (g)).

---

## 1. Setting and baseline

### 1.1 Task and scoring

For a bias-free ReLU MLP of width n = 1024 and depth 16, with He-initialised weights and input x ~ N(0, I), the estimator returns E[h^(l)] for every layer. The score is

  score = MSE_final × max(0.1, FLOPs / 2^41).

The final-layer MSE is measured against a 10⁹-sample Monte-Carlo reference. FLOPs are metered analytically by flopscope.

We use the unit **1 u = 2^31 FLOPs**, which is about one dense 1024³ product, so the budget is B = 1024 u. "util" denotes FLOPs/B, and the 0.1 floor equals 102.4 u.

The grader enforces three hard limits per MLP:
- 120 s wall time;
- 0.4 s of **residual** Python time outside metered operations;
- an 8 GB address space.

A capped MLP receives zero predictions with multiplier 1, which scores about mean(target²) ≈ 0.35. A single capped MLP in a hundred therefore ruins a submission.

### 1.2 The public chain (V29, 504aldo, MIT)

Our estimator is a modification of 504aldo's public "V29". We use it under the MIT licence; its copyright header and LICENSE are retained in every file we ship.

V29 implements the factorised K = 3 "simple" cumulant propagation of Wu et al. (arXiv:2605.05179, §4.3). Per layer it carries:
- the mean;
- the covariance C;
- the third cumulant κ₃, as a **sum of sources**.

Each ReLU injects one source. A source is stored as:
- a propagator leg P_s (n×n, the identity at birth, updated P ← Wᵀ diag(w₁) P);
- a birth leg A_s = P_s a_b;
- thin n×16 feedback legs.

The next ReLU needs the slices D3_i = κ_iii and D21_ic = κ_iic. These are obtained per source from contractions of Hadamard products of the legs.

Further features of V29:
- **Shared basis for old sources.** Sources older than four layers are confined to a shared basis of rank 384, and the oldest to a nested rank-224 sub-basis.
- **κ₄ handling.** κ₄ is not carried. It is regenerated each layer by a memoryless law, G_off = λ_l C_off.
- **D21 feedback.** D21 enters through feedback legs.
- **Pooled Strassen–Winograd.** Products use pooled, batched Strassen–Winograd at level 5.

The author's ledger prices V29 at 260 u, split as:

| Component | Cost |
|---|---|
| Young sources | 116 u |
| Old sources | 107 u |
| Thin and elementwise | 25 u |
| Covariance | 7 u |
| Closure | 6 u |

The author also reports a tolerance law: extra final MSE ≈ 4.2e-6·ε², where ε is the relative RMS error of D21. D21 must therefore be right to about 2 %.

**Graded baseline.** V29 verbatim graded 5.418e-9 (raw 2.133e-8, util 0.254; #331681), matching the author's own submission.

### 1.3 Measurement instruments

All of the following are described in §8.

- **Monte-Carlo teachers** with 6·10⁷ samples on public mini MLP #0 (logan-fitzgerald) and on a synthetic 1024×16 He network. They record, per layer, the pre- and post-ReLU means, covariances, κ₃ slices and κ₄ diagonal and slices.
- **Oracles** that replace one internal quantity of the chain by the teacher's value and rerun the chain end to end.
- **A deterministic 100-MLP paired harness** on Kaggle, using the public mini split with its 10⁹-sample truth.

**The project κ₄ metric.** Throughout §5, the metric is the correlation of the standardised κ₄ diagonal, κ₄/v², with the teacher's, computed on *decisive* neurons (|μ/σ| < 1.5) at layers 12, 15 and 16. Raw (unstandardised) correlations inflate every candidate because they contain v², which the chain already gets right. One outside proposal scored 0.80–0.87 on raw correlation and 0.43 on this metric.

---

## 2. Exact identities and derivations

### 2.1 Spherical-input identity (derived; shipped as V31)

Write X = (R/√n)·Y, where Y is uniform on the unit sphere scaled to radius √n and R ~ χ_n is independent of Y.

A bias-free ReLU network h is positively homogeneous of degree 1, so

  E[h(X)] = E[R/√n] · E[h(Y)] = a₁ E[h(Y)],  with a₁ = √(2/n) Γ((n+1)/2) / Γ(n/2) = 0.99975589 for n = 1024.

The sphere input has C = I and κ₃ = 0. Its fourth cumulant is κ₄ = c·Sym₃(I⊗I), with c = −2/(n+2). In the chain's harmonic-core representation this is a previous-layer core g = −6/(n+2).

The estimator therefore does three things:
- it injects that core at layer 0;
- it runs the existing κ₄ machinery unchanged;
- it rescales the output by a₁.

**Checks.**
- **Measured:** the injection reproduces the identity's scale shift to 0.3 % (+2.450e-4 against +2.442e-4 predicted). The control without a₁ gives 8.02e-8.
- **Graded:** raw 2.132e-8 → 2.108e-8 (−1.1 %), score 5.137e-9 (#331787), at zero cost.

The reparameterisation idea comes from a public Phase-2 repository (barnobarno666), surfaced in an outside-AI consultation (§9). On that repository's K3 chain it was worth −7.9 %. On V29, most of that gain is already absorbed by the fitted κ₄ regeneration.

### 2.2 Parity theorem: no mean-field part of the deep κ₃ (derived)

Take a source born at layer b. At any later layer l its legs are P = Wᵀ⁽ˡ⁾ D P′ and A = P a_b, where P′, a_b and the birth coefficients do not depend on W⁽ˡ⁾.

- Every quantity that is linear in the source's κ₃ (the products H, T2, T3 and T4, and the diagonal D3) is therefore an **odd** function of W⁽ˡ⁾.
- C⁽ˡ⁾, the leg Grams and the per-neuron variances are **even** functions of W⁽ˡ⁾.

Because the weight ensemble is sign-symmetric,

  E[ H_s⁽ˡ⁾ | C⁽ˡ⁾, Gram(A_s, P_s), all state before l ] = 0, exactly.

**Consequences.**
1. Any closure that replaces old sources by a function of even statistics discards the entire contribution. This covers mean-field closures, scale mixtures and polynomials in C. The failures of such closures reported in the public log (9.1e-7 and 2.3e-6) are therefore structural.
2. Replacing a source by its conditional expectation given one fresh weight row has a relative-error floor of √(6/(r+8)) in the retained support dimension r. That is 12 % at r = 384 and 16 % at r = 224, an order of magnitude above the 2 % tolerance.
3. κ₄ terms built from two κ₃ factors are even, and C⊗C is the only even invariant available. This explains the empirical law G_off = λ_l C_off.

The argument originated in an LLM consultation (Fable 5) and was checked by us. The floor was derived independently by two other models with the same constant. A final written hand-verification is still listed as pending in our notes **[OPEN]**.

The learned-residual experiment in §4.7 is the data-side counterpart of this theorem.

### 2.3 Exact structure of the births (measured to rounding)

- **Birth leg.** The birth leg satisfies A_s = P_s · diag(w₁,b) · C_off⁽ᵇ⁾ to 1e-5 at every slot and layer: the birth operator is the row-scaled off-diagonal post-ReLU covariance. It has a zero diagonal and a flat spectrum, needing 310–809 singular directions for 99 % of its energy, so it is not compressible.
- **Born tensor.** The born tensor splits into two families:
  - AAP, with |AAP|/|total| = 0.36–0.37;
  - MPP, with 0.74–0.75.
  
  Their mutual cosine is +0.56–0.57.

### 2.4 Edgeworth port deficit (derived and measured; shipped as FB 1.25)

Expanding the post-ReLU κ₃ to first order in C shows that the newborn Sym(X₁⊗I⊗Y₁) carries exactly **half** of both K×C all-distinct families.

**Measurement.** Regression coefficients on the slice residual are −0.55 to −0.69 and −0.6 to −0.9, against −1/2 predicted.

Restoring the missing families fully does not help, but a partial restoration does:

| Restoration | Effect on the teacher pair |
|---|---|
| Full (S = 2) | +3.4 % / +3.1 % (worse) |
| Quarter (S = 1.25) | −0.8 % / −0.35 % |

The fix is zero-FLOP. It was shipped together with an adaptive-λ exponent of 1.2. On the 100-MLP harness this scored ×0.9946 ± 0.0007, and it graded 5.1284e-9 (#332110). Why the full fix hurts the final mean even though it improves the slice at layers 4–10 is an open paradox **[OPEN]**. A 10 Oct test removed one explanation: the quadratic feedback cross term that also grows with the scale is inert, and the linear restoration itself is what hurts (§6.5).

### 2.5 Cubic curvature-birth cumulant identities (derived; numerically verified)

Expand the ReLU around a neuron's pre-activation mean μ and standard deviation σ, with α = μ/σ and standardised input u:

  relu(μ + σu) = Σ_k a_k He_k(u),  a_k = σ^k E[relu^(k)(μ+σu)] / k!.

Since relu″ = δ:
- a₂ = σφ(α)/2;
- a₃ = −σαφ(α)/6.

The second coefficient a₂ is the "curvature" coefficient. The third coefficient a₃ is the cubic coefficient used by the Round-18 star and by V56, where it appears as c = −αφ/6 in standardised units.

For jointly Gaussian standardised u, Wick's theorem gives the cubic-birth identity

  κ(He₃(u_a), u_b, u_c, u_d) = κ(u_a³, u_b, u_c, u_d) = 6 ρ_ab ρ_ac ρ_ad.

A cubic Hermite term at hub a thus injects a **star**-shaped fourth cumulant into all downstream neurons. Quadratic terms pair up into **path** (He₂–He₂) and C·T shapes.

**Numerical verification** (`r17_curvature_identities.json`):
- the cubic identity and its He₃ form hold to 8.1e-16 and 6.3e-16;
- the quadratic Q1 and Q2 identities hold to 5.1e-16 and 2.4e-16;
- the combinatorial readout factors hold to 9.0e-15 for the star (24), 3.5e-15 for C·T (24) and 4.1e-15 for the path (48).

These identities are the algebraic basis of §5.4–§5.5.

### 2.6 Counting results on the sum-of-sources family (derived or enumerated)

- **Young-tier floor.** An exact enumeration of contraction orders gives 193 square products for the young tier in every order: 39 P + 54 A + 100 D21 = 98.4 u at the measured 0.505 u per product. Two independent outside validators reproduced the count.
- **Duality.** Storing only birth matrices and layer maps and reading slices by reverse-mode differentiation (the "lazy cubic circuit") costs about 4l products at depth l, about 245 u in total, against 249 u for the source chain. The two are dual contraction orders of one tensor network.
- **Irreducibility.** Sym³ is an irreducible GL(n) representation. A linear state closed under every dense M^⊗3 therefore needs C(n+2, 3) ≈ 1.8·10⁸ numbers.
  - *Caveat:* with fixed, known weights, only what future readouts can observe matters, which is at most n² per remaining layer.
  - The statement is a count over known contraction orders, not a lower bound on arithmetic cost.
- **Algebraic floor per source.** Exactly evaluating one old source's D21 costs at least min(2n²r, 4nr³) per source-layer.

---

## 3. Cost engineering under a metered, op-capped budget

### 3.1 Where the FLOPs go

A per-call-site ledger of metered FLOPs (`m1_matmul_ledger.py`) on the V29-era chain gives 249 u per network. Of this, 220 u (88 %) is matrix products.

| Family | Cost |
|---|---|
| Young-source transport | ≈ 54 u |
| Young-source D21 contractions | ≈ 50 u |
| Joins (10 per network) | 41–46 u |
| Old sources in the basis | ≈ 41 u |
| Thin feedback | ≈ 15 u |
| Strassen additions | ≈ 12 u |
| Einsum | 9 u |
| Covariance and closure | ≈ 8 u |

The young tier sits at its enumerated floor (§2.6), so cost had to come from elsewhere: joins, dead neurons, the basis side, and the meter's pricing rules.

### 3.2 Graded cost steps

Each step below is exact or near-exact. "Util" is the grader's figure, or adjusted ÷ raw where that is the only source.

| Step (date) | Mechanism | Graded score (id) | Raw | Util |
|---|---|---|---|---|
| V29 verbatim (20 Sep) | — | 5.418e-9 (#331681) | 2.133e-8 | 0.254 |
| STRASSEN_MIN 32→8 | Two more Strassen levels on the rank-384/224 families. Prediction drift 4.9e-13, peak 5.97 GB | 5.220e-9 (#331686) | 2.132e-8 | 0.244 |
| + STRASSEN_FIRST 5 | First call per worker at full depth | 5.195e-9 (#331732) | 2.132e-8 | 0.2436 |
| + V31 sphere (§2.1) | Accuracy, 0 FLOPs | 5.137e-9 (#331787) | 2.108e-8 | 0.2437 |
| + FB 1.25, β 1.2 (§2.4) | Accuracy, 0 FLOPs | 5.1284e-9 (#332110) | 2.1046e-8 | — |
| Transport join (25 Sep) | Rebuild the shared basis at a join by transporting the previous basis (no range finder) | 4.9519e-9 (#332257) | 2.1788e-8 | 0.2273 |
| + AGE_OLD 5, R_FB 4 | One more dense young age; feedback rank 16→4 | 4.8066e-9 (#332324) | 2.1371e-8 | 0.2249 |
| v48 compaction (26 Sep) | w₁ := 0 for Φ < 1e-2; live-row contraction, widths padded to multiples of 32 | 4.6294e-9 (#332391) | 2.1462e-8 | — |
| v49 basis-side Strassen + aliased Grams | Projections, rotations, Qc and the D21 lift through Strassen; joiner Grams as YYᵀ (billed at half) | 4.4157e-9 (#332401) | 2.1463e-8 | — |
| v50 D21 live columns | D21 columns of dead neurons never computed | 4.2512e-9 (#332404) | 2.1537e-8 | — |
| v51 D21 rows, Φ < 3e-3 / 7e-3 | D21 rows of very dead neurons never computed | 4.1081e-9 (#332418) / 4.0902e-9 (#332420) | 2.1545e-8 / 2.1636e-8 | 0.189 |
| v52 op budget, safe B / A (27 Sep) | Strassen only where it saves ≥ ~2 u per 1,000 ops (§3.3) | 4.1696e-9 (#332566) / 4.1228e-9 (#332567) | — | — |
| + R_OLD 352, Tq from the QR R factor, R_FB 2 | Exact or near-exact trims | B: 4.0995e-9 (#332626); A: 4.0784e-9 (#332621) | 2.197e-8 (B) | — |

**Net effect.** From V29 to these variants, the trims removed 23 % of the metered FLOPs (util 0.244 → 0.189) for +2.8 % raw (2.105e-8 → 2.164e-8). On 26 Sep alone, the score fell from 4.8066e-9 to 4.1081e-9 (−14.5 %).

**What could not be compacted.** Only the *input* side of dead neurons can be compacted. Each of the following loses, measured on the teacher pair:

| Attempt | Effect |
|---|---|
| Removing dead neurons as hubs of the new source (their own κ₃ must propagate) | ×6.40 / ×6.74 |
| Zeroing their entries of the transported covariance | ×1.59 / ×1.46 |
| Zeroing rows with \|α\| > 3 | +6.5 % / +19.6 % |
| Treating always-on neurons as hubs | ×10.0 / ×8.0 |

### 3.3 The residual-time cap and the operation budget (graded failures)

Two variants that ran cleanly on our machines graded catastrophically:

| Submission | Grade | Share of MLPs capped |
|---|---|---|
| #332402 (v49 + R_OLD 320) | 0.0434 | ≈ 12 % |
| #332425 (v51, row threshold 1e-2) | 0.0204 | ≈ 6 % |

Memory was not the cause: the local peak was 5.51 GiB. We measured that the residual Python time tracks the number of metered operations, with no other dependence on the variant:
- about 0.07 ms per op on our laptop;
- 0.053–0.059 ms per op on Kaggle, for every variant;
- about 0.02 ms per op inferred for the grader.

The failing variants issued 21.7k–22.3k operations; v48, at ≈ 16.4k, had passed.

**The fix was an op budget.**
- Strassen is used only where it saves at least about 2 u per 1,000 ops. Each top-level Strassen call costs 110–240 ops.
- The covariance slot rides inside the compacted family.
- The op ledger names the expensive groups:

| Op group | Ops | Cost saved |
|---|---|---|
| Nested projections | 1.3k | 0.7 u |
| Qc | 1.25k | 1.7 u |
| D21 lift | 1.2k | 1.6 u |
| C slot | 1.27k | 0 u |

Both classes graded with no capped MLP (#332566 and #332567). This produced our two submission classes:
- **B (nomination-safe):** ≤ ~18k warm-call ops.
- **A:** ~19.9k–20k ops.

### 3.4 The 8 GB limit (graded failures)

The first V56 implementation batched the cubic star as a 3-D broadcast. It failed the grader's smoke test four times (#334786, #334787, #334798, #334799).

Kaggle reproduced the failure only under `whest run --runner subprocess`, which gave "Unable to allocate 450. MiB for an array with shape (118013952,)". In-process runs passed, with maxrss 5371 MB against 5346 MB for the V55 base.

A bisection submission with the star disabled graded exactly as V55 (3.8546e-9, #334803). The per-slot 2-D-plus-mat-vec form then passed (#334805). Strassen level 6 was ruled out earlier for the same reason, with a peak of 8.13 GB.

**Lesson.** `whest validate` catches neither the residual cap nor the address-space limit. Subprocess runs on the mini split are the minimum pre-submission check.

### 3.5 The one-for-one law, and where it breaks

Inside the chain, every *accuracy knob* we swept on the 100-MLP harness trades about 1 % cost for 1 % accuracy, and the shipped constants sit at the score optimum:

| Knob | Score ratio |
|---|---|
| β 1.2 | ×0.9993 |
| λ ×0.9 | ×1.0026 |
| Two tier-1 passes | ×1.073 |
| FB × β grid | within +0.12 % |
| R_OLD 448 | ×1.0113 |
| AGE_OLD 6 on V56 | raw ×0.978 but util +3.7 %, score ×1.014 |

The law does **not** hold for **cost levers that are exact or nearly exact**: compaction, Strassen placement, aliased Grams and the transport join. Our 27 Sep audit of 74 earlier kills found that all 8 kills that had to be revived were cost or knob levers. They had wrongly been killed on MSE alone, on mixed sub-populations, in one direction only, or on a different base.

No architecture kill was revived. Every one had missed its gate by a factor of 5 to 50.

A 100-MLP grid over the young-age gate and the old-basis rank on V56 (11 Oct, §6.6 (c)) confirms the law on the shipped estimator. Every one of the five settings scores worse than the shipped AGE_OLD 5, R_OLD 352, by ×1.026 to ×1.104.

### 3.6 Call-site ledger of the shipped V56 (measured 10 Oct)

`experiments/probes/site_ledger.py` wraps every public flopscope call. It charges each outermost call its metered FLOPs and one operation, and attributes both to the estimator line that requested it (the Strassen helpers are looked through). The ledger records the second, warm `predict()` call of the A-class V56 (`submissions/v56As_fs/estimator.py`) on one network. Result: `site_ledger_v56As_fs.json`.

The warm call costs **193.8 u** in **20,586 operations**. We group the call sites as follows (the grouping is ours):

| Group | Call sites | Cost | Ops |
|---|---|---|---|
| D21 hub readout | Strassen leaf products of the hub (`prod`, 59.6 u) and the live-row D21 readout `_d21_live` (10.2 u) | ≈ 70 u | 4,707 |
| Young-leg transport | The compacted transport family (52.9 u), which also carries the newborn's A leg and the covariance slot, and its uncompacted form in the first layers (10.1 u) | ≈ 63 u | 1,629 |
| Old basis | Dense legs of the old sources formed from their basis factors: tier 1 (7.4 u), tier 2 through the sub-basis (9.6 u), and the projection at a join (3.2 u) | ≈ 20 u | 2,925 |
| Everything else | Symmetric product of C_pre (6.2 u), thin feedback and D slices, rotations, QR, Grams, closure and κ₄ terms | ≈ 40 u | 11,325 |
| **Total** | | **193.8 u** | **20,586** |

Reading the ledger:
- **Two families dominate.** The D21 readout and the young transport are 133 u, about 69 % of the bill. These are the per-source contractions whose number grows with depth.
- **The old tier is now cheap.** The old basis costs about 20 u, against 107 u in the author's V29 ledger (§1.2) and about 41 u in our V29-era ledger (§3.1). Compaction, Strassen placement and the nested basis removed most of it.
- **Most operations are not in the expensive families.** The 40 u remainder issues 11.3k of the 20.6k operations. Removing FLOPs from the hot families therefore does not by itself relieve the residual-time budget (§3.3).
- **What a 60 % budget would need (derived).** At 0.6 × 194 u = 116.4 u, removing the whole 70 u readout still leaves 124 u, and removing the whole 63 u transport leaves 131 u. Any design at that budget must cut both hot families by more than half, or cut the remainder too.

This ledger is one network. The harness mean util of the same estimator is 0.1861, about 190.6 u (§6.5 (b)).

**Comparison with the reference implementation (derived).** Wu et al. price their factorised K = 3 algorithm at about 4,150 u for n = 1024 and L = 16 (their Table 1, naive matrix products, as read in `consultations/wu_paper_deep_read_2026-10-10.md`). V56 is therefore about 21× cheaper than the reference algorithm it descends from. The paper's factor count grows as 6nℓ, and it has no source merging, truncation or compression. The savings from V29's shared basis and from our trims (§3.2) are what make the method fit the budget at all.

---

## 4. The error map: where the error lives

All results in this section are measured on MLP #0 against the 6·10⁷-sample teacher, and partly replicated on the synthetic network. The plain chain's final MSE in these runs is 2.290e-8.

### 4.1 The closures are exact; the error is accumulated input error

- **Mean closure.** The Edgeworth mean closure, evaluated on the teacher's true marginals (μ, var, κ₃ᵢᵢᵢ, κ₄ᵢᵢᵢᵢ), matches the teacher's ReLU mean to 5e-11–1.5e-10 at every layer (9.3e-11 at L16).
- **Bivariate closure.** Teacher-forced at one layer, the bivariate covariance closure falls to the teacher's noise:

| | L7 | L14 |
|---|---|---|
| RMS error per entry | 7.04e-5 → 2.61e-5 (noise 2.5e-5) | 8.84e-5 → 2.76e-5 |
| Sensitivity-weighted next-layer variance error removed | 87 % | 91 % |

- **Accumulation.** The chain's per-layer MSE grows smoothly, from 0.7e-9 at L1 to 2.29e-8 at L16, adding about 1–2.5e-9 per layer.
- **Gain.** The inherited mean error passes each layer with a gain of |W row|²E[Φ²] = 0.65 at layer 2, rising to 0.8–0.96 at depth.
- **Inherited share.** Forcing the true mean at every layer takes the final MSE from 2.24e-8 to 4.2e-9, so about 80 % of the final error is inherited mean error.

### 4.2 Channel oracles

The true value of one channel is injected into the mean closure at every layer:

| Exact input | Final MSE | Ratio |
|---|---|---|
| Variance | 1.220e-8 | 0.533 |
| κ₄ diagonal | 1.609e-8 ¹ | 0.702 |
| κ₃ diagonal | 1.965e-8 | 0.858 |
| Variance + κ₃ | 0.817e-8 | 0.357 |
| Variance + κ₃ + κ₄ | 0.130e-8 | 0.057 (the truth's noise floor) |
| True κ₄ everywhere *except* the mean closure | 2.311e-8 | ≈ 1 |
| κ₄ diagonal + (2,2) + (3,1) slices | 1.738e-8 | 0.759 |
| (3,1) only / (2,2) only / both | 2.353 / 2.796 / 2.680e-8 | 1.03 / 1.22 / 1.17 |

¹ Two runs give 1.600e-8 and 1.609e-8 (`k4oracle2_target_base.json`: 1.6086e-8). An independent replication by a subagent gave ratios 0.532 / 0.865 / 0.702 / 0.057.

Single-channel ratios are upper bounds on each channel's own share, because each includes error inherited from the other channels.

Two facts follow from the table:
- κ₄ acts **only** through the mean closure, and only through its **diagonal**.
- The off-diagonal κ₄ slices are harmful when injected alone.

A first-order attribution of the L16 error gives:

| Source | Share |
|---|---|
| Pre-activation mean | 1.84e-8 (80 %) |
| Variance | 2.0e-9 |
| κ₃ᵢᵢᵢ | 1.0e-9 |
| κ₄ᵢᵢᵢᵢ | 1.0e-9 |

At depth, the *fresh* error per layer splits about 50 % variance, 25 % κ₃ and 25 % κ₄. The fitted coefficients are −0.99, −0.98 and −1.04, against −1 in theory, with R² 0.93–0.97 from L9 on.

### 4.3 The chain's per-channel error against the teacher

Teacher noise is given in parentheses where it was measured.

| Layer | Mean (σ units) | Variance | d3 | κ₄ diagonal (slope, corr over all neurons) | C off-diagonal | D21 off-diagonal |
|---|---|---|---|---|---|---|
| 7 | 1.9e-4 | 0.04 % (0.02) | 3.2 % (1.1) | 15 % (0.95, 0.78) | 0.39 % | 4.6 % |
| 11 | 3.1e-4 | 0.09 % | 3.8 % | 22 % (0.89, 0.78) | 0.59 % | 5.7 % |
| 15 | 4.6e-4 | 0.14 % | 4.6 % | 32 % (0.78, 0.83) | 0.73 % | 6.6 % |
| 16 | 5.1e-4 | 0.14 % | 4.7 % | 34 % (0.75, 0.82) | — | — |

Over all neurons the chain's κ₄ diagonal correlates 0.78–0.83 with the truth. On the **decisive** neurons that drive the mean closure it correlates only 0.26–0.34, at a level ratio of about 1.0. This gap between the two populations is why §5 uses the decisive-neuron metric.

### 4.4 Variance error is inherited covariance error

About 95 % of the variance error is dvarᵢ = wᵢᵀ δC wᵢ, a random projection of the off-diagonal post-ReLU covariance error. That covariance error has two parts:
- an inherited part, with slope 1.01 and R² rising from 0.52 to 0.79 with depth;
- a fresh part of about 4.2e-5 RMS per entry.

The audited missing covariance terms explain little:
- the C³, C⁴ and C×κ₃₁ patches have RMS 2–3e-6, against an error of 6.5–9e-5;
- the chain's own (1,1) terms explain only 3–8 %.

There is no missing covariance term. The variance channel's 47 % comes from accumulated κ₃ and κ₄ input error.

### 4.5 κ₃: the all-distinct entries, and memory

The chain's κ₃(i,i,j) slice has slope 1.00 and R² 0.996–0.998, but a random error that grows from 4.1 % at L3 to 6.7 % at L15. Of that error:
- 91–96 % sits in the **all-distinct** post-ReLU entries κ₃(hᵢ,hⱼ,h_k), which are themselves 9–16 % wrong;
- it is not truncation: with no confinement the slice error stays at 6.6 % and the final MSE moves only −3.9 %, for +19 % cost;
- about 55 % of it is unexplained even at second order in K, since the twelve K×K terms have RMS 3e-6 to 2e-5 against an error of 2.2e-4.

*Refinement (10 Oct, §4.10).* These percentages are measured against the 6·10⁷-sample teacher and include its sampling noise. A Gaussian estimate puts that noise at 2.5 % (L4) to 5.3 % (L2) on the slice, so most of the 4.1 % at L3 is teacher noise. Net of noise, the chain's slice error is about 3 % at L4 and grows to about 7 % at L16.

**Perfect κ₃ oracle.** A perfect κ₃ input (slice + d3) is worth ×0.6075 on MLP #0 (1.382e-8 vs 2.275e-8) and ×0.715 on the synthetic net. The synthetic figure is partial: 8 of 14 layers. Applied in layer windows on MLP #0 / synthetic:

| Window | Ratio |
|---|---|
| Layers ≥ 8 | 0.637 / 0.754 |
| Layers ≥ 12 | 0.704 / 0.863 |
| Layers ≤ 11 | 0.884 / 0.837 |

The windows compose multiplicatively: 0.884 × 0.704 ≈ 0.62.

**Memory.** The κ₃ memory is long and coherent but not low-rank in age. Keeping only sources of age ≤ h leaves this relative D21 error at L15:

| h | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| Error | 84 % | 65 % | 46 % | 31 % | 16 % |

- The age contributions are coherent: mean pairwise cosine 0.29–0.39.
- They are not low-rank: projecting on six age-modes still leaves 8.5 %.
- Removing the old tier entirely costs 32× (7.43e-7). Every age gate follows MSE ≈ 7.4e-7 × 0.66^(g−4).

### 4.6 κ₄: level transported, pattern regenerated

- **One-step transport.** Pairwise transport of the true post-ReLU slices gets the next layer's κ₄ diagonal **level** right (slope 1.00) but not its **pattern** (corr 0.24–0.59).
- **Exact slices are not enough.** Even with the exact (4), (3,1) and (2,2) slices, the next diagonal is 14 / 22 / 33 / 43 / 46 % wrong at L4 / L8 / L12 / L15 / L16.
- **Where the content lives.** The next layer's diagonal is about 90 % the (2,2) pair term. The next d3 comes 90–99 % from off-diagonal κ₃, so the diagonal-to-diagonal channel carries only 18 % (L4) down to 1.7 % (L16) in norm.

### 4.7 What does not carry the missing information

**Gaussian and copula closures.** The one-step closure floors (2e8 samples per layer) give these RMS errors of the next-layer mean:

| | L4 | L8 | L12 | L15 |
|---|---|---|---|---|
| Gaussian closure | 5.2e-4 | 5.1e-4 | 4.6e-4 | 4.5e-4 |
| Gaussian copula with exact marginals | 1.14e-3 | 9.2e-4 | 8.1e-4 | 7.5e-4 |

The Gaussian closure errs at about 3× V29's final error. The copula with exact marginals is worse than the plain Gaussian closure at every layer.

**Learned residual correctors.** On 100 MLPs (102,400 output neurons), learned correctors recover almost nothing:

| Model | Residual MSE recovered |
|---|---|
| Ridge, 19 features | 0.3 % |
| Ridge, 189-feature quadratic expansion | 0.7 % |
| Gradient boosting | 0.5 % |

- Per-network offsets explain 0.44 % of the residual variance.
- Every single feature correlates ≤ 0.04 with the residual.
- This is the parity theorem (§2.2) seen from the data side.

**Monte-Carlo routes.**
- A 10⁴-sample forward pass costs 156 u and delivers the variance only to 1.4 %, against a 0.1 % need.
- Control-variate surrogates decorrelate with depth:
  - the tangent network falls to 0.15–0.29 at L16;
  - the second-order Taylor surrogate falls from 0.967 at L2 to 0.00 at L16.
- The resulting variance reductions are only 1.0–2.2×.

### 4.8 The κ₃ and κ₄ inputs jointly (joint oracle, measured 9 Oct)

The oracle of §4.2 and §4.5 is rerun with both inputs at once (`experiments/variants/r20_joint_oracle.py`). The teacher's off-diagonal κ₃ (2,1) slice replaces the chain's D21 everywhere after it is computed, in the closure terms and in the feedback legs. The teacher's d3 = κ₃ᵢᵢᵢ replaces the chain's D3, and the teacher's κ₄ diagonal enters the mean closure.

The base is the plain V40-family chain (FB 1.25, β 1.2, spherical identity off) at util 0.2476. Its final MSE is 2.273e-8 on MLP #0 and 1.920e-8 on the synthetic net, slightly different from the 2.290e-8 base used in §4.1–§4.7. The synthetic teacher covers layers 2, 4, …, 14 and 15 only; at its other layers the chain keeps its own values, so the synthetic column is partial.

| Teacher input injected (every layer unless noted) | MLP #0 final MSE | Ratio | Synthetic final MSE | Ratio |
|---|---|---|---|---|
| None (plain chain) | 2.273e-8 | 1 | 1.920e-8 | 1 |
| κ₃ slice + d3 | 1.380e-8 | 0.607 | 1.376e-8 | 0.716 |
| κ₄ diagonal | 1.561e-8 | 0.686 | 1.700e-8 | 0.885 |
| d3 only + κ₄ diagonal | 1.204e-8 | 0.530 | 1.394e-8 | 0.726 |
| κ₃ slice (chain's own d3) + κ₄ diagonal | 0.995e-8 | 0.438 | 1.429e-8 | 0.744 |
| κ₃ slice (chain's own d3) + κ₄ pattern at the chain's level | 0.977e-8 | 0.430 | 1.441e-8 | 0.751 |
| **κ₃ slice + d3 + κ₄ diagonal** | **0.619e-8** | **0.272** | **1.092e-8** | **0.569** |
| κ₃ slice + d3 + κ₄ pattern at the chain's level | 0.669e-8 | 0.294 | 1.163e-8 | 0.606 |
| κ₃ slice + d3 at every layer, κ₄ diagonal from L8 | 0.653e-8 | 0.287 | 1.053e-8 | 0.548 |
| Both inputs from L8 only | 0.746e-8 | 0.328 | 1.102e-8 | 0.574 |

Reading the table:
- **κ₃ is the larger single lever on both networks:** ×0.607 against ×0.686 on MLP #0, and ×0.716 against ×0.885 on the synthetic net.
- **The two inputs compound more than multiplicatively.** The product of the single-input ratios would be 0.416 on MLP #0 and 0.634 on the synthetic net; jointly they give 0.272 and 0.569. Each channel's error is partly inherited from the other (§4.2).
- **With the true κ₄, the true d3 matters.** On MLP #0, adding the teacher's d3 to the teacher's slice takes 0.438 to 0.272.
- **The κ₄ pattern is again enough.** The teacher's pattern at the chain's level gives 0.294 against 0.272 for the full diagonal (compare §5.1). Injecting κ₄ only from L8 loses little (0.287).
- **What remains is mostly the variance channel.** After κ₃ and κ₄, the remaining ×0.27 is consistent with §4.2, where adding the true variance takes the joint ratio to the noise floor (0.057), and with §4.4, where the variance error is inherited covariance error.

### 4.9 Which Hermite order each channel needs, one step at a time (measured 9 Oct)

`r20_onestep_channels.py` runs 2·10⁶ exact samples of z_l through MLP #0. At one layer l, the ReLU is replaced by its Hermite expansion of order K around the teacher's (μ, σ) (§2.5), with common random numbers, and z_{l+1} = act(z_l) W⁽ˡ⁾. Every next-layer statistic is compared with the **exact** ReLU step on the same samples. A chain whose births stop at order K makes at least this error in each channel per step.

Values are given at source layer L2 → L15 (layers 2, 4, 8, 12 and 15 were measured; the trend is monotone unless a range is given):

| Order K | C off-diagonal | Variance | D21 off-diagonal | κ₃ diagonal | κ₄ pattern corr | κ₄ level ratio |
|---|---|---|---|---|---|---|
| 1 (linear) | 13.4 → 6.4 % | 19.0 → 4.9 % | 76.8 → 19.7 % | 75.0 → 16.1 % | 0.63 → 0.84 | 0.38 → 0.87 |
| 2 | 2.17 → 0.42 % | 3.26 → 0.99 % | 20.1 → 9.9 % | 21.2 → 10.0 % | 0.85–0.90 | 1.95 → 1.43 |
| 3 | 1.49 → 0.32 % | 2.14 → 0.63 % | 8.85 → 2.05 % | 9.26 → 2.05 % | 0.95 → 0.99 | 1.52 → 1.10 |
| 4 | 0.88 → 0.20 % | 1.29 → 0.36 % | 4.61 → 1.34 % | 4.30 → 1.38 % | 0.98 → 0.999 | 1.16 → 1.06 |

Errors are relative Frobenius errors against the exact step; the κ₄ columns use the project metric. For scale, this run's own distance to the 6·10⁷-sample teacher is 1.5 % → 0.53 % (C off-diagonal) and 0.10 % (variance). The common random numbers make the K-against-exact differences much more precise than that.

Findings:
1. **κ₃ needs at least the cubic response.** With quadratic births (K = 2), the one-step D21 and d3 errors are 10–21 %, against a D21 tolerance of about 2 % (§1.2). K = 3 reaches 2.1–2.7 % only at L12–L15 and is 7–9 % at L2–L4. K = 4 is 1.3–4.6 %.
2. **The covariance needs at least the quadratic response.** The linear step alone leaves 6–13 % of the off-diagonal covariance.
3. **One κ₄ step is easy; compounding is not.** K = 2 gets the one-step κ₄ pattern to 0.85–0.90 but overshoots the level by 1.4–1.9×. Compounded over six layers, the same truncation falls to 0.45–0.55 (§5.4, set A), while K = 3 holds.

### 4.10 Where the κ₃ error is real, and where its gain lives (measured 10 Oct)

**Error profile against teacher noise.** `chain_k3_error_profile.json` compares the plain chain's κ₃ (the captures used in §5.8) with the 6·10⁷-sample teacher at every layer. `teacher_k3_noise_estimate.json` gives a Gaussian approximation of the teacher's own sampling noise on the same quantities (an estimate). The "net" columns subtract that noise in quadrature, √(error² − noise²). This step is derived and assumes the noise is independent of the chain's error.

| Layer | D21 off-diagonal error | Teacher noise (est.) | Net D21 error | d3 error | d3 noise (est.) | Net d3 error | d3 norm ratio, chain/teacher |
|---|---|---|---|---|---|---|---|
| 2 | 4.4 % | 5.3 % | ≈ 0 | 3.3 % | 5.4 % | ≈ 0 | 1.002 |
| 4 | 4.2 % | 2.5 % | 3.4 % | 2.9 % | 2.2 % | 1.9 % | 0.998 |
| 8 | 4.9 % | 1.2 % | 4.8 % | 3.1 % | 0.9 % | 3.0 % | 0.996 |
| 12 | 5.9 % | 0.7 % | 5.9 % | 3.8 % | 0.6 % | 3.8 % | 0.993 |
| 16 | 6.9 % | 0.5 % | 6.9 % | 4.6 % | 0.4 % | 4.6 % | 0.986 |

Errors are relative Frobenius errors against the teacher.

Reading the table:
- **The early "error" is mostly teacher noise.** At L2 the estimated noise exceeds the measured difference on both the slice and d3.
- **The real slice error is carried and grows with depth**, from about 3 % at L4 to about 7 % at L16. At L7, L11 and L15 the profile gives 4.6 / 5.7 / 6.6 %, the same values as §4.3.
- **d3 has a level bias, not only scatter.** Its decisive-neuron correlation stays at 0.9965–0.9994, but its norm ratio falls steadily from 1.002 (L2) to 0.986 (L16), a 1.4 % shrink.

**Where the κ₃ gain lives** (`results/summaries/oct10/k3loc_{1,2,3}.json`). The teacher's κ₃ is injected only in a layer window, with the oracle of §4.8, on MLP #0. "Slice" is the off-diagonal (2,1) slice, "d3" the diagonal; "feedback legs" injects the slice into the feedback legs only. The base is the plain chain of §4.8 (2.2732e-8).

| Teacher κ₃ injected | Window | Final MSE | Ratio |
|---|---|---|---|
| Slice + d3 (§4.8) | L1–16 | 1.380e-8 | 0.607 |
| Slice + d3 | L1–4 | 2.282e-8 | 1.004 |
| Slice + d3 | L5–8 | 2.141e-8 | 0.942 |
| Slice + d3 | L9–12 | 1.949e-8 | 0.857 |
| Slice + d3 | L13–16 | 1.832e-8 | 0.806 |
| Slice only | L1–8 | 2.129e-8 | 0.937 |
| Slice only | L9–16 | 1.897e-8 | 0.834 |
| d3 only | L1–8 | 2.300e-8 | 1.012 |
| d3 only | L9–16 | 1.938e-8 | 0.853 |
| Feedback legs only | L1–16 | 2.276e-8 | 1.001 |

Reading the table:
- **The gain is in L9–16.** The product of the four slice + d3 windows is 0.653 (derived), against 0.607 when every layer is injected at once. L9–16 alone would give about 0.857 × 0.806 ≈ 0.69 (derived; not run as one window).
- **Early injection does not help.** Slice + d3 at L1–4 gives ×1.004, and d3 alone at L1–8 gives ×1.012. This fits the profile above: at L2–L4 the teacher's κ₃ is about as noisy as the chain's is wrong.
- **At depth, the slice and d3 matter about equally**: ×0.834 and ×0.853 at L9–16.
- **The feedback legs are not the lever**: ×1.001 (compare ×0.9999 in §6.3).

A κ₃ fix must therefore remove a carried late-layer error: 5–7 % in the slice and a 1.4 % shrink of the d3 level at L16. The three candidate mechanisms tested on 10 Oct are reported in §6.5.

---

## 5. The κ₄-diagonal pattern programme

### 5.1 The target is the pattern, not the level

Oracle (`k4_oracle_run2.py`): the teacher's κ₄ diagonal is injected into the mean closure of the full chain. "Share" is measured against the full gain, none (2.290e-8) → diagonal (1.609e-8).

| Injected target | Final MSE | Share of the gain |
|---|---|---|
| Teacher **pattern** at the chain's level | **1.551e-8** | **108 %** |
| Teacher pattern, layers ≥ 8 only | 1.707e-8 | 86 % |
| Teacher pattern, layers ≥ 12 only | 1.813e-8 | 70 % |
| Teacher pattern, layers ≤ 11 only | 2.064e-8 | 33 % |
| 50 % blend of teacher and chain patterns | 1.789e-8 | 74 % |
| Chain pattern at the teacher's **level** | 2.783e-8 | worse than none (+21.5 %) |

The whole κ₄ gap is the per-neuron *pattern* of the deep diagonal. The level the chain already produces is adequate.

**Joint oracle: cheap κ₃ plus true κ₄.** The same oracle on cheaper κ₃ chains shows that the gain sits on the full κ₃ chain only:

| κ₃ configuration | Util | Own κ₄ | True κ₄ diagonal | Ratio |
|---|---|---|---|---|
| Full chain | 0.2476 | 2.290e-8 | 1.609e-8 | ×0.702 |
| AGE_OLD 2 | 0.2303 | 5.240e-8 | 4.590e-8 | ×0.876 |
| AGE_OLD 2, R_OLD 256/128 | 0.1784 | 1.288e-7 | 1.244e-7 | ×0.966 |
| AGE_OLD 1, R_OLD 192/96 | 0.1381 | 6.511e-7 | 6.665e-7 | ×1.024 |

The route "cut κ₃ cost, recover with κ₄" is therefore closed. A κ₄ improvement must ride on the full κ₃ chain.

### 5.2 The deep pattern is carried memory

**One-step carry of the true state** (`opus12_k4_carry.json`, project metric):

| Carrier | L12 | L15 | L16 |
|---|---|---|---|
| Chain | 0.339 | 0.264 | 0.324 |
| Exact linear carry of the true κ₄ | 0.771 | 0.785 | 0.797 |
| Linear carry + quadratic ReLU curvature | 0.820 | 0.855 | 0.863 |

**Gaussian physics alone is not enough.** A Gaussian input with the true mean and covariance gives only 1–4 % of the deep κ₄ level from L8 on.

**Memory horizon.** Non-Gaussian history older than h layers is erased by a Gaussian restart, and the true network is run forward from there (`q18_k4_horizon.json`, N = 1e6). Pattern correlation at L12 / L16:

| h | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| L12 / L16 | 0.23 / 0.20 | 0.40 / 0.41 | 0.53 / 0.59 | 0.67 / 0.65 | 0.78 / 0.77 |

Correcting for the teacher's noise raises h = 6 to 0.85 and h = 2 to 0.52. The later N = 4e6 runs (§5.4, set B) give somewhat higher values at h = 3–4, which is consistent with lower noise attenuation.

A young carrier spanning 2–3 layers is therefore dead from the start. An ideal one, tested out of sample, moves the G4-weighted error only from 0.210 to 0.203.

### 5.3 V55: a zero-FLOP linear pattern corrector (graded)

**Construction.** At every layer ≥ LMIN, a fixed linear map turns per-neuron quantities the chain already holds into a predicted pattern of κ₄/v². The inputs are:
- s3sq = (D3/v^1.5)²;
- the row and column energies of the standardised D21;
- the correlation-row energy of C_pre;
- |μ/σ|;
- the chain's own κ₄/v²;
- a constant.

The pattern is rescaled to the chain's layer-mean level and fed to the **mean closure only**; the pair closure keeps the chain's κ₄.

The map is fitted once, offline, against the teacher on decisive neurons. The last layer uses a reduced map. The cost is 0 u and +310 metered ops (17,961 in total for the B class).

**Pattern correlation.**
- MLP #0: 0.41–0.55, against 0.24–0.41 for the chain (layers 6–15).
- Held out on the synthetic network: 0.41–0.52, against 0.25–0.35.
- The map transfers between networks: fitted on the synthetic net, it gives 0.43–0.56 on MLP #0.

**Teacher pair.**

| Class | MLP #0 (fit network) | Synthetic net (held out) |
|---|---|---|
| B | 2.1962e-8 → 2.0381e-8 (−7.2 %) | 1.8856e-8 → 1.8343e-8 (−2.7 %) |
| A | −7.0 % | −2.7 % |

**Harness and grader.** On 100 MLPs: ×0.9455 [90 % CI 0.9410, 0.9498], with 97/100 MLPs improved.

| Class | Projected grade | Actual grade |
|---|---|---|
| B | 3.8764e-9 | 3.8762e-9 (#334588) |
| A | 3.8562e-9 | 3.8582e-9 (#334591) |

Each is −5.4 % against its base. The s3sq coefficient ×1.25 then graded 3.8539e-9 (#334762) and 3.8331e-9 (#334763).

**Sweeps place the map at its optimum.** Ratios are on the 100-MLP harness.

| Change | Ratio |
|---|---|
| s3sq ×1.5 / ×1.75 / ×2.0 | ×1.0068 / ×1.0207 / ×1.0396 |
| c2 ×1.5 / ×0 | ×1.0563 / ×1.0131 |
| Chain coefficient ×0.9–×1.2 | within ×1.0011 |
| Last-layer corrector off | ×1.0116 |
| Lever-weighted "lean" fit | ×0.9957 against the decisive-neuron map, i.e. the lean fit is worse |

**Saturation.** Squares, products and logs of the features add only +0.02–0.03 in sample. Previous-layer features carried by (W∘W)ᵀ add nothing held out.

The cause is the feature set. With the same layer's teacher quantities, a least-squares combination reaches 0.65–0.70 out of layer, but almost all of that comes from the k22 row sum, a teacher-only feature that correlates 0.62–0.70 on its own. Without k22, out-of-layer fits fall to 0.39–0.54. **Per-layer chain features saturate near 0.5.**

### 5.4 Why carriers stalled: the Hermite-family ceilings

Between 27 Sep and 8 Oct we tested many κ₄ *carriers*: states transported from layer to layer with births at each ReLU. Every one stalled. Pattern correlation at L12 / L15 / L16:

| Carrier (birth content) | Correlation | Source |
|---|---|---|
| Skew-lift Σ_s Sym4(T_s⊗r_s) | 0.03 (best LSQR fit) | R13 |
| Dense pair core born at the first ReLU | −0.35 to −0.22 | R13 |
| Signed bilinear Sym(H⊗C), H carried by WᵀHW | 0.351 / 0.321 / 0.371 | R16 |
| Diagonal-only version of the same | 0.336 / 0.294 / 0.351 | R16 |
| True (4) + (3,1) + (2,2) slices carried one step | 0.30–0.45 | R16 |
| Pair-innovation births, all ages | 0.433 / 0.362 / 0.411 | R16b |
| Hub-pair C·T curvature births, h = 6 | 0.442 / 0.432 / 0.431 | R17 |
| Gaussian hypertree bank (GHQ-6) | 0.559 / 0.471 / 0.499 | R17 |

The pair-innovation carrier also made the closure error ×1.12–1.43 worse than the chain.

To find out why, we measured the ceiling of entire *families* of birth laws (`r17_hermite_family_ceiling.py`, Kaggle GPU, N = 4·10⁶). The construction:
1. Start from a Gaussian restart N(μ, C) of the teacher h layers back.
2. At each ReLU, replace the ReLU by its Hermite expansion of order K around the teacher's (μ, σ).
3. Measure the resulting κ₄ pattern.

**Set A and set B: which Hermite orders are needed.**

| Mode | L12 | L15 | L16 |
|---|---|---|---|
| Sanity: true ReLU, no restart | 0.967 | 0.973 | 0.980 |
| h = 6, true ReLU (restart ceiling) | 0.859 | 0.820 | 0.818 |
| h = 6, K = 2 (quadratic births) | 0.548 | 0.449 | 0.530 |
| h = 6, orders 0/1/3 only | 0.511 | 0.498 | 0.469 |
| **h = 6, K = 3 (cubic births)** | **0.848** | **0.799** | **0.817** |
| h = 6, K = 4 | 0.865 | 0.816 | 0.822 |
| h = 4, K = 3 / true ReLU | 0.749 / 0.753 | 0.687 / 0.702 | 0.711 / 0.710 |
| h = 3, K = 3 / true ReLU | 0.652 / 0.663 | 0.588 / 0.600 | 0.643 / 0.647 |
| h = 2, K = 3 / true ReLU | 0.517 / 0.525 | 0.436 / 0.434 | 0.485 / 0.494 |
| True state 1 back, K = 2 | 0.850 | 0.875 | 0.881 |
| True state 3 back, K = 2 | 0.687 | 0.689 | 0.738 |
| True state 6 back, K = 2 | 0.527 | 0.477 | 0.578 |
| True state 6 back, K = 3 | 0.915 | 0.927 | 0.949 |

Findings:
1. **Quadratic births are capped near 0.45–0.55.** This is exactly the band every second-order carrier landed in.
2. **K = 3 is necessary and sufficient.** K = 3 tracks the true ReLU at every horizon, and K = 4 adds ≤ 0.02.
3. **Both pieces are needed.** Dropping the quadratic term loses as much as dropping the cubic one.
4. **Compounding K = 2 diverges.** Applied over all 16 layers, it blows the level up by ×1e12. Even from the true state, quadratic births lose the pattern step by step: 0.85–0.88 after one step, 0.69–0.74 after three and 0.48–0.58 after six (one-step channel errors in §4.9).
5. **Horizon.** The 0.70 gate needs a horizon of about 5–6 layers.

**Set C: which ages need cubic births** (9 Oct, `r17_hermite_family_ceiling_C.json`). The restart is at h = 6. The full cubic response is applied only at the listed ages, and the response is linear elsewhere.

| Ages with cubic births | L12 | L15 | L16 | Level ratio L12 / L15 / L16 |
|---|---|---|---|---|
| 1–6 (all) | 0.848 | 0.799 | 0.817 | 0.81 / 0.53 / 0.48 |
| 2–6 | 0.709 | 0.708 | 0.691 | 0.61 / 0.44 / 0.35 |
| 1–4 | 0.736 | 0.682 | 0.713 | 0.28 / 0.19 / 0.19 |
| 1–3 | 0.647 | 0.582 | 0.637 | 0.13 / 0.09 / 0.09 |
| 1–2 | 0.525 | 0.458 | 0.456 | 0.05 / 0.03 / 0.04 |
| 1 | 0.343 | 0.239 | 0.231 | 0.02 / 0.00 / 0.02 |
| 4–6 | 0.367 | 0.397 | 0.409 | 0.30 / 0.19 / 0.20 |
| 6 | 0.026 | 0.116 | 0.187 | 0.05 / 0.04 / 0.03 |

Reading the table:
- **Ages are complementary.** No single age carries the pattern, and no late window does either: ages 4–6 give 0.37–0.41.
- **Dropping age 1** costs 0.09–0.14.
- **Truncating to ages 1–4** costs 0.10–0.12. Even so, it stays at or above the 0.70 gate at L12 and L16 (0.736, 0.713) and just below it at L15 (0.682).
- **Ages 1–3** fall to 0.58–0.65.
- **The level is a separate matter.** The level is mostly recovered from older ages (ages 4–6 alone give 0.19–0.30 of it). Since V55 and V56 rescale to the chain's level, and the pattern alone is worth 108 % of the oracle gain (§5.1), the pattern is the quantity a carrier must deliver.
- **Variance ratios** run from 0.64 (age 1 only) to 0.96 (all ages).

### 5.5 HTQB star → V56 (graded)

The Round-18 proposal "Hermite Tree Quartic Bank" (HTQB) uses the identities of §2.5 to read κ₄ from existing legs without a carried κ₄ state. It has two parts:
- a cubic **star** (Cu1: 24 Σ_a c_a P X³, with c_a = −α_aφ_a/6, over the last 6 ages);
- a quadratic **path** (Q2: 48 Vᵀ C V, over 3 ages).

Its proposer priced it at 27.2 u.

**Ceilings with teacher μ and C** (`round18_htqb.json`), at L12 / L15 / L16:

| Form | Correlation |
|---|---|
| Star + path, h = 6 | 0.637 / 0.559 / 0.586 |
| Shippable form (star h6 + path h3) | 0.616 / 0.538 / 0.541 |
| Star alone | 0.521 / 0.489 / 0.471 |
| Path alone | 0.074 / 0.024 / 0.150 |

Alone, HTQB is borderline. It is, however, **complementary** to the V55 features. With leave-one-layer-out fits on MLP #0, still using teacher μ and C (`r18_htqb_combo.json`):

| Features | Correlation, L7–L15 |
|---|---|
| V55 feature set alone | 0.48–0.56 |
| + star | 0.65–0.74 (0.70–0.74 at L12–15) |
| + star + path | 0.71–0.76 (L12 0.761, L15 0.711) |

**V56** ships the zero-FLOP half. The cubic star is computed from the chain's **own** young dense source legs (ages ≤ 6) and the per-source coefficient c = −αφ/(6·var) recorded at birth. It enters as an 8th feature of the V55 map. The path is not shipped.

The 8-feature map reaches 0.53–0.71 per layer on MLP #0 and 0.54–0.66 on the synthetic net, depending on the fitting network (`k4s_fit.json`), against 0.41–0.55 for the 7-feature map. Utilisation is 0.1857, against 0.1855 for the V55 reference.

**V56 grades.**

| Submission | Class | Score | Raw | Note |
|---|---|---|---|---|
| #334805 v56s_loop | B | 3.6829e-9 | 1.9706e-8 | −4.4 % vs the best B |
| #334811 v56As_loop | A | 3.6597e-9 | 1.9539e-8 | |
| #334869 v56s_fs (synthetic-fit map) | B | 3.6723e-9 | — | −0.29 % |
| **#334870 v56As_fs** | A | **3.6521e-9** | 1.950e-8 | −0.21 %, current best |
| #334871 fs + star ×1.25 | B | 3.6724e-9 | — | no extra gain |

**Tuning is closed** (`kaggle_v56_tuning.txt`). Ratios are against v56_ref (raw 2.0409e-8):

| Change | Ratio |
|---|---|
| Star ×0.75 / ×1.5 | ×1.0110 / ×1.0111 (score) |
| Star ages 4 | ×1.0045 |
| s3sq ×1.1 / ×1.25 | ×1.0028 / ×1.0093 |
| AGE_OLD 6 | raw ×0.9782 but score ×1.0138 |

The AGE_OLD 6 result is consistent with §5.4: a sixth dense age helps accuracy, but not at its present price.

### 5.6 Can Hermite trees replace the κ₃ chain? (measured; no)

Since the tree κ₃ diagonal has pattern correlation 0.975–0.986 with the truth, we tested whether a Hermite-tree estimator could drop the source chain altogether. The test is teacher-forced, one Edgeworth closure per layer (`r19_tree_ceiling.json`):

| Layer | Teacher floor | Chain | Tree κ₃ + κ₄ (h = 6) | Chain κ₃ + teacher κ₄ | Chain κ₃ + tree κ₄ |
|---|---|---|---|---|---|
| 8 | 4.3e-11 | 1.445e-9 | 2.99e-8 | 6.6e-10 | 1.08e-8 |
| 12 | 5.2e-11 | 2.004e-9 | 3.40e-8 | 9.5e-10 | 1.91e-8 |
| 16 | 6.2e-11 | 2.072e-9 | 4.18e-8 | 1.03e-9 | 1.93e-8 |

The tree is 10–20× worse. An oracle level rescale does not rescue it: L12 still gives 2.80e-8. The closure needs κ₃ to about 1 %, and a 0.98 correlation is far from that. The tree κ₄ is also harmful as a direct replacement of the chain's κ₄; it is useful only as a *feature* inside the rescaled pattern map.

### 5.7 Summary of the graded κ₄ gains

| Stage | Score (id) |
|---|---|
| Before V55 (A) | 4.0784e-9 (#332621) |
| V55 | 3.8582e-9 (#334591) |
| V55, s3sq ×1.25 | 3.8331e-9 (#334763) |
| V56 | 3.6597e-9 (#334811) |
| V56, synthetic-fit map | **3.6521e-9 (#334870)** |

In total this is −10.5 % on the score at unchanged cost, all of it from making the κ₄-diagonal *pattern* better at zero FLOPs.

### 5.8 Beyond the star: the κ₃-dressed C·T channel (measured 9–10 Oct; oracle and component level, not graded)

The V56 star is the *Gaussian* cubic birth. The Round-17 hub-pair C·T births were killed alone at 0.43–0.44 (§5.4). A design from a 9 Oct consultation, the "lagged κ₃-dressed vertex" (LKV), reads the chain's κ₃ through the quadratic and cubic Hermite vertices of §2.5. Every number below is our own measurement on public MLP #0.

**Readouts.** Take a vertex layer m and a target pre-activation layer t, at age j = t − m. Write u = z_m − μ_m and the K = 3 response of the ReLU at m as

  h − E h = Φ u + g (u² − v) + c (u³ − 3 v u),  g = φ/(2σ),  c = −αφ/(6σ²).

With the one-step linear response Ystep_l = diag(w₁⁽ˡ⁾) W⁽ˡ⁾ (legs move as L ← Ystep_lᵀ L, §1.2), define:
- the transports Ẏ = Ystep_m ⋯ Ystep_{t−1} and Ȳ = W⁽ᵐ⁾ Ystep_{m+1} ⋯ Ystep_{t−1} (the hub leg, with no w₁ on the hub);
- the C-leg X = C_m Ẏ;
- the κ₃ object Q_ai = Σ_bc κ₃(z_m)_abc Ẏ_bi Ẏ_ci.

The single-vertex births then read out on the diagonal at t (readout factors from §2.5):

  star_i = 24 Σ_a c_a Ȳ_ai X_ai³    (the Gaussian cubic star, shipped in V56)
  ct_i = 24 Σ_a g_a Ȳ_ai X_ai Q_ai   (κ₃ converted at a quadratic vertex: the C·T channel)
  tt_i = 36 Σ_a c_a Ȳ_ai YL_ai Q_ai,  YL = (D21 + diag d3) Ẏ   (κ₃ squared at a cubic vertex: T·T)

The prediction sums the readouts over ages j = 1–6.

**Ceiling** (`fable_ceiling_2026-10-09.py`, 2·10⁶ samples through the true network). Q, X and the exact births are computed from the samples. Project metric at L12 / L15 / L16:

| Readout, ages 1–6 | Pattern correlation | Level ratio |
|---|---|---|
| Sampling floor: sample κ₄ against the teacher | 0.938 / 0.949 / 0.963 | 1.00 / 0.99 / 1.00 |
| Exact one-vertex births + linear carry (telescoping identity, rel. err. 4.8e-6–1.0e-5) | 0.938 / 0.949 / 0.963 | 1.00 / 0.99 / 1.00 |
| Exact one-vertex births | 0.937 / 0.919 / 0.932 | 0.77 / 0.63 / 0.62 |
| K = 3 births with exact inputs | 0.926 / 0.893 / 0.918 | 1.35 / 1.10 / 1.08 |
| Star alone | 0.521 / 0.489 / 0.471 | −0.84 / −0.88 / −0.82 |
| C·T alone | 0.442 / 0.431 / 0.424 | 0.88 / 0.93 / 0.85 |
| T·T alone | −0.159 / −0.133 / −0.057 | −0.04 / −0.05 / −0.05 |
| Star + T·T | 0.490 / 0.445 / 0.436 | |
| **Star + C·T** | **0.846 / 0.835 / 0.824** | 0.04 / 0.05 / 0.03 |
| Star + C·T + T·T | 0.849 / 0.840 / 0.830 | 0.00 / 0.00 / −0.02 |
| Same, hub index of Q projected on the top-r eigenvectors of C_m, r = 32 / 64 / 128 | 0.67–0.71 / 0.74–0.75 / 0.78–0.81 | |

Reading the table:
- **Star + C·T matches the Gaussian-restart cubic-birth ceiling** of set A (§5.4: 0.848 / 0.799 / 0.817). It stays below the K = 3 births with exact inputs in the same run (0.89–0.93). Over ages 1, 2, 4 and 6 it builds up as 0.57 → 0.71 → 0.82 → 0.85 at L12 (0.43 → 0.65 → 0.80 → 0.84 at L15).
- **The Round-17 C·T kill was too narrow.** C·T alone sits exactly where the hub-pair carrier was killed (0.42–0.44), but it is complementary to the star.
- **T·T is dead.** It is negative alone and adds ≤ 0.006 on top of star + C·T.
- **The readouts carry the pattern, not the level.** The star and C·T levels cancel (sum level −0.02 to 0.05), so the prediction must enter as a *feature* of a rescaled map, as in V55/V56, not as a raw κ₄ diagonal.
- **The single-piece identities hold on the samples.** The exact one-vertex births split into joint cumulants of the linear, quadratic and cubic images of the response. Per age, the quadratic piece Q1 = 4 κ(quad, lin, lin, lin) correlates 0.934–0.974 with C·T. The cubic piece Cu1 = 4 κ(cub, lin, lin, lin) correlates 0.993–0.998 with star + T·T (plus a diagnostic C·K term).

**As a feature: oracle end-to-end test** (`fable_affine_fits.json`, `fable_affine_e2e.json`). At each layer 7–16 an affine map

  κ₄/v² ≈ a · κ₄^chain/v² + b · pred/v² + c

(three numbers per layer) is fitted by least squares to MLP #0's teacher, with pred taken from the sample-exact ceiling readouts. The fitted values replace the κ₄ diagonal in the plain chain's mean closure on the same network. **This is an oracle in two ways:** the map is fitted to the teacher on the network it is evaluated on, and the feature's FLOPs are not metered (util stays 0.2476).

| κ₄ diagonal in the mean closure, layers 7–16 | Fit corr, L7–L16 | Final MSE | Ratio |
|---|---|---|---|
| Plain chain (own κ₄) | 0.26–0.40 | 2.273e-8 | 1.000 |
| Teacher κ₄ diagonal | — | 1.591e-8 | 0.700 |
| Affine(chain) only: the fair baseline | — | 2.467e-8 | 1.085 |
| Affine(chain, star) | 0.37–0.51 | 2.296e-8 | 1.010 |
| **Affine(chain, star + C·T)** | **0.82–0.86** | **1.758e-8** | **0.773** |
| Affine(chain, star + C·T + T·T) | 0.81–0.86 | 1.801e-8 | 0.792 |

C·T recovers (1 − 0.773)/(1 − 0.700) ≈ 75 % of the perfect-κ₄ gain. Against the same-maturity affine baseline (×1.085), star + C·T is ×0.713 and star alone is ×0.931.

**Component test: the chain already holds the inputs** (`fable_component_2026-10-09.py`). The test uses captures of the plain chain with every source dense. Each of a source's three blocks has three legs, and each of a block's three Sym placements (a "role"; nine per source) pairs one leg at m (the anchor L) with the Hadamard product H of the other two legs at t. Then Q_m(t) = (1/3) Σ_s Σ_roles L_s(m) H_s(t)ᵀ. The legs at t are the legs at m transported by Ẏ (checked to 1.3e-7 and 8.2e-7).

Against the sample-exact readouts on decisive neurons, the chain's own readouts correlate:
- C·T: 0.993 (pair 14 → 15, age 1) and 0.996 (11 → 15, age 4);
- star: 1.000;
- T·T: 0.991–0.994.

The C·T content is therefore available from the chain's existing κ₃ legs. The obstacle is cost: formed per (source, role, age), Q costs one dense n³ product each.

**Age-collapse identity (derived; measured exact).** Write G_m = diag(g_m)(Ȳ_{m→t} ∘ X_{m→t}), an n × n hub-by-target matrix. Then

  ct_i = 8 Σ_m Σ_s Σ_roles Σ_a (G_m)_ai (L_s(m) H_s(t)ᵀ)_ai   (8 = 24/3).

Let a_s be the first window layer that holds source s. Its anchor legs are transported, L_s(m) = (Ystep_{a_s} ⋯ Ystep_{m−1})ᵀ L_s(a_s), so the sum over ages collapses onto one matrix per anchor layer:

  ct_i = 8 Σ_s Σ_roles Σ_h (L_s(a_s)ᵀ Γ_{a_s})_hi H_s(t)_ih,   Γ_a = G_a + Ystep_a Γ_{a+1}.

Γ is built once per target by a backward recursion over the window. On the captures, the collapsed form matches the per-age route to a relative error of 2.9–3.0e-7 at L12 / L15 / L16 (`acds_ct_probe.json`) and 2.9–3.9e-7 at L7–L16 (`acds2_ct_probe.json`), which is float32 rounding.

**Anchor grouping (derived; measured exact).** The nine anchor legs of a source are linear in its two dense legs, A_s and P_s, plus thin legs (the low-rank feedback and residual columns). Grouping the roles by dense anchor gives

  ct_i = 8 Σ_s [ Σ_h (A_sᵀ Γ_{a_s})_hi (H_A,s)_ih + Σ_h (P_sᵀ Γ_{a_s})_hi (H_P,s)_ih + thin_s,i ],

where H_A,s and H_P,s collect the target Hadamards of all roles anchored on A_s or on P_s. The thin_s,i terms are evaluated exactly through the thin products Z_sᵀ Γ. The grouped form matches the per-age route to 2.6–3.2e-7 at every target L7–L16. What remains is **two dense products per source and target**, plus building G_m and Γ.

**Sketches.** The dense products are replaced by Aᵀ Γ_a ≈ (Aᵀ S̃_a)(S_aᵀ Γ_a), with S̃ = S (SᵀS)⁻¹. S_aᵀ Γ_a = Σ_{m≥a} S_mᵀ G_m when the rank-r basis S is transported like a leg. Bases tested:
- "eig": the top-r eigenvectors of C at the window start;
- "rad": C Ω with a Rademacher Ω;
- "direct-Γ": an orthonormal basis of Γ_a Ω per anchor, optionally with target-aware column weights ω_i = dec_i √λ_i / v_i² and one power step.

The captures hold dense source stacks only at a few layers, so these tests use the consistent source subset: sources born between stack layers are absent until the next stack. On that subset, the exact chain C·T correlates 0.82–0.96 with the sample-exact one, against 0.993–0.996 for the full source set in the component test.

| C·T variant | Fidelity to the exact chain C·T (decisive neurons) | Teacher-metric corr of affine(chain, star + C·T), L12 / L15 / L16 or their range |
|---|---|---|
| Sample-exact (ceiling) | — | 0.859 / 0.858 / 0.843 |
| Exact chain C·T, source subset | 1 | 0.725 / 0.760 / 0.778 |
| Round 1 (all anchors sketched), eig / rad, r = 64 | 0.93–0.94 / 0.89–0.93 | 0.60–0.63 / 0.60–0.61 |
| Round 1, eig / rad, r = 128 | 0.95–0.97 / 0.94–0.96 | 0.67 / 0.65–0.66 |
| Round 1, eig / rad, r = 256 | 0.97–0.98 / 0.97–0.98 | 0.71 / 0.70–0.72 |
| Round 2 (grouped, low-rank anchors exact), eig r = 64 | 0.91–0.97 (L7–L16) | 0.659 / 0.665 / 0.674 |
| Round 2, eig r = 128 | 0.955–0.983 | 0.691 / 0.700 / 0.715 |
| Round 2, direct-Γ weighted + power step, r = 64 | 0.89–0.94 | 0.660 / 0.672 / 0.674 |
| Round 2, direct-Γ, r = 64 | 0.87–0.92 | 0.613 / 0.539 / 0.590 |

Two lessons:
- **Once the low-rank anchors are exact, the plain eigen-sketch of C is the best basis we tried.** At r = 64 the best direct-Γ basis ties it on the teacher metric at L12–L16, has lower fidelity and a worse end-to-end ratio (×0.885 against ×0.865, below), and costs more. The probe's thin-product count over targets 7–16 is 37.6–51.3 u against 27.6 u.
- **High fidelity is not enough.** In round 2 at L12–L16, sketches that correlate 0.96–0.98 with the exact chain C·T still lose 0.03–0.10 of teacher-metric correlation. The information C·T adds lies in the part the sketch drops.

**End to end with the chain's own C·T** (`acds_e2e.json`). This uses the same oracle affine map and protocol as above, with the chain's own star and the C·T variant from the captures:

| C·T input to the affine feature | Final MSE | Ratio |
|---|---|---|
| Teacher κ₄ diagonal (reference) | 1.591e-8 | 0.700 |
| Sample-exact C·T (ceiling) | 1.758e-8 | 0.773 |
| Exact chain C·T, source subset | 1.863e-8 | 0.819 |
| Eigen-sketch, r = 128 | 1.917e-8 | 0.843 |
| Eigen-sketch, r = 64 | 1.966e-8 | 0.865 |
| Direct-Γ weighted + power step, r = 64 | 2.012e-8 | 0.885 |

The gap from ×0.773 to ×0.819 is at least partly the capture subset, since the full-source component fidelity is 0.993–0.996. That share has not been measured separately.

**Layer windows** (10 Oct, `results/summaries/oct10/ctwin_{1,2}.json`). The same oracle affine feature is injected only at the target layers of a window. Elsewhere the chain keeps its own κ₄. The base is 2.2732e-8.

| C·T input to the affine feature | Window | Final MSE | Ratio |
|---|---|---|---|
| Sample-exact (ceiling, above) | L7–16 | 1.758e-8 | 0.773 |
| Sample-exact | L11–16 | 1.811e-8 | 0.797 |
| Sample-exact | L13–16 | 1.951e-8 | 0.858 |
| Sample-exact | L15–16 | 2.135e-8 | 0.939 |
| Sample-exact | L7–10 | 2.235e-8 | 0.983 |
| Eigen-sketch, r = 64 (above) | L7–16 | 1.966e-8 | 0.865 |
| Eigen-sketch, r = 64 | L11–16 | 2.074e-8 | 0.912 |
| Eigen-sketch, r = 64 | L13–16 | 2.164e-8 | 0.952 |

L11–16 carries (1 − 0.797)/(1 − 0.773) ≈ 89 % of the ceiling's gain, and L7–10 alone is nearly worthless (×0.983). With the r = 64 sketch, however, L11–16 keeps only (1 − 0.912)/(1 − 0.865) ≈ 65 % of the gain, while it holds 72 % of the probe's thin-product count (derived from `acds3_ct_probe.json`; the later targets hold more sources). Cutting target layers therefore saves less than it loses. Cutting rank is the better cut (round 3, below).

**Round 3: cheaper bases and source subsets** (10 Oct, `acds3_ct_probe.json`, `acds_e2e3.json`). The captures, protocol and oracle map are those of round 2. The new variants are:
- "rsi": a randomized subspace iteration on C at the window start (q = 2, three thin products with C), transported like the eigen-basis;
- "top-6": only the six sources with the largest ‖A_s‖‖H_A,s‖ + ‖P_s‖‖H_P,s‖ are kept, and the others are dropped;
- "young only" / "old only": only the sources anchored after / at the window start are kept.

The cost column is the probe's thin-product count summed over targets 7–16. It excludes forming G_m and Γ.

| C·T variant | Thin-product count | Teacher-metric corr of affine(chain, star + C·T), L12 / L15 / L16 | Final MSE | Ratio |
|---|---|---|---|---|
| Eig, r = 64 (round 2) | 27.6 u | 0.659 / 0.665 / 0.674 | 1.966e-8 | 0.865 |
| rsi, r = 64 | 27.6 + 1.9 u | 0.659 / 0.654 / 0.668 | 1.965e-8 | 0.864 |
| Eig, r = 48 | 20.7 u | 0.643 / 0.661 / 0.659 | 1.993e-8 | 0.877 |
| **Eig, r = 32** | **13.8 u** | 0.620 / 0.629 / 0.637 | **1.987e-8** | **0.874** |
| Eig, r = 64, top-6 sources | 20.9 u | 0.665 / 0.625 / 0.642 | 2.063e-8 | 0.907 |
| Eig, r = 32, top-6 sources | 10.4 u | 0.631 / 0.613 / 0.610 | 2.076e-8 | 0.913 |
| Eig, r = 64, young sources only | — | 0.675 / 0.660 / 0.694 | 2.051e-8 | 0.902 |
| Eig, r = 64, old sources only | — | 0.503 / 0.474 / 0.391 | — | — |

On every target the age-collapse and grouping identities again hold to 2.6–3.9e-7.

Reading the table:
- **Rank is cheap to cut.** r = 32 keeps ×0.874 against ×0.865 at r = 64, for half the thin-product count. r = 48 is no better than r = 32 (×0.877).
- **rsi ties the eigen-basis** (×0.864) and adds 1.9 u.
- **Dropping sources costs more than dropping rank.** Top-6 gives ×0.907–0.913 and young-only ×0.902. The old-anchored sources carry signal that no cheap subset keeps.
- **The three-layer teacher metric does not rank sketches end to end.** Young-only has the highest correlation at L12 / L15 / L16 (0.66–0.69) but ends at ×0.902, behind eig r = 32 (0.62–0.64, ×0.874). Sketch choices must be made on the end-to-end mean.

**Cost and verdict.** The score is MSE × max(0.1, FLOPs/2⁴¹). V56 uses about 190 u (util 0.186), and our estimate for a deployable eigen-sketch C·T at r = 64 is about 40 u (estimate). Adding 40 u multiplies util by about 1.21, so a raw gain of ×0.865 gives a score of about ×1.05, i.e. roughly neutral. Break-even at ×0.865 needs less than about 30 u (estimate). The probe's count of thin sketch products alone already sums to 27.6 u over targets 7–16 at r = 64, before G_m and Γ are formed.

Round 3 changes this arithmetic. Scaling the 40 u estimate by the probe's thin-product counts (13.8 against 27.6 u) puts eig r = 32 at about 18–20 u (estimate). Util then rises by a factor of about 1.09–1.11, and the score ratio is about 0.874 × 1.10 ≈ 0.96–0.97, i.e. a gain of 3–4 % (estimate). The open risk is the metered op count, which no prototype has measured yet.

Moreover, every ratio in this subsection was measured:
- on one network;
- with an in-sample teacher-fitted map;
- on the plain chain rather than on V56, which already contains the star and the V55 features.

The increment on V56 with an offline-fitted map is **(untested)**.

**Conclusion.** The κ₄ C·T channel is real. Its ceiling is 0.82–0.85 together with the star, it is worth ×0.773 under an oracle map, and the chain's own legs reproduce it at 0.993–0.996. At r = 64 it is **not affordable**. At r = 32 (round 3) it is at the edge of affordability, at an estimated 3–4 % score gain, with every caveat above still applying. Per the joint oracle (§4.8), κ₃ is the larger lever: ×0.607 against ×0.686 on MLP #0, and ×0.716 against ×0.885 on the synthetic net.

---

## 6. Negative results

Unless marked, ratios are final-MSE or score ratios against the chain, given as MLP #0 / synthetic net. "corr" is the decisive-neuron κ₄ pattern correlation. "Gate" is the pre-registered pass criterion.

The kill types are:
- **C:** a ceiling or oracle (an upper bound);
- **P:** a teacher-forced component test;
- **E:** an end-to-end prototype;
- **K:** a knob.

### 6.1 κ₃ representation and cost architectures

| Idea | Measured (gate) | Kill |
|---|---|---|
| Low-rank Hadamard products of source legs | A∘P needs 681–778 of 1024 directions for 99 % energy (flat spectrum) | C |
| Old content only in the last k layers | Keep last 1/2/3 layers: 4.85 / 4.17 / 3.16e-7 against a chain error of 2.1e-8; dropping one birth: 1.3–2.7e-7 | C |
| Query surrogates: trace, diagonal Hermite, TensorSketch | Error at L14: 98.2 %, 95.4 %, 790 % (d = 2048) | C/P |
| Pooled or aggregate states: vector metric, Tucker/TT, CP | Rank 32/128/256 → 35 / 7.8 / 1.6 %; k = 128 → 20 %; CP rank 4608 → 1.07e-7 | C |
| Slice-to-slice recursion | R² 0.030 / 0.047; RSC D21 error 35.9 % vs the chain's 6.6 % | C |
| Cubic source-index coding | 388 / 527 % error (q = 1); 83 / 130 % (32 channels) | P |
| Birth operator ≈ diagonal + rank k | 73 / 57 % (k = 32); 22 / 16 % (k = 384) | C |
| Adjoint low-rank response | Low-rank path carries 14 % of the response | C |
| ISDF skeleton / landmarks | 56–71 % (anchor), 79–94 % (transported) | P |
| Birth-column screening | −5.6 % cost for +25–30 % raw | K |
| Asymmetric contraction rank | Util +0.8 %, raw +2.9 % | E |
| Young tier at half cost | 193 products in every order (§2.6) | count |
| Explicit κ₃ tensor / lazy circuit | ≈ 1567 u per layer / ≈ 245 u against 249 u | count |
| Short age memory: age modes, shift carrier | h = 10 still 16 %; 6 modes 8.5 %; shift error 2.9 % (gate 2 %) | C |
| Source-free dressed hypertree (DHMP) | Next-layer D21 error 36–40 % (gate 2.5 %) | P |
| Diagonal-only κ₃/κ₄, diagonal + low-rank C | Next-d3 error 83–99 %; diagonal + rank-64 C 12–33 % | P |
| Deterministic moment-matched particles | D21 error 74–160 % against the chain's 6.6 % | E |
| Response-cashout propagator (use each source once) | Equals keep-last-1: +4.85e-7, about 23× the chain error | C |
| Richardson extrapolation in rank | Exponent 1.46 vs 2.07; every extrapolation worse | C |
| Strassen level 6 / 7 | 8.13 GB against the 8 GB cap; level 7 runs out of memory | cap |
| Compensating closure with no old tier | 7.43e-7 (32×); every knob 6.8–9.9e-7 | C |
| Hermite-tree κ₃ replacing the chain (§5.6) | Closure 10–20× worse | P |
| Single shared pool for the whole κ₃ (§6.6 (a)) | D21 error at L11–15: 7–11 % (r = 384), 3.4–6.3 % (r = 512); only r = 768 reaches 0.2–1 % (gate 2 % at r ≤ 512) | C |
| Deferred exact join, DEJ (§6.6 (b)) | Youngest source exact, rest at r = 352–512: 2.5–8 % (gate 2 %) | C |
| Age/rank re-split of V56, 100 MLPs (§6.6 (c)) | Score ×1.026 to ×1.104 (gate < 1) | K |
| Checkpointed birth archive, CBA-13 (§6.6 (d)) | κ₃ zeroed in the closures over a window: ×14.8 to ×125; no recovery | E |
| Trace projection of old κ₃ (Wu et al., §6.6 (f)) | 26–42 % of D21 | C |
| Learned pair-memory student, PM364 (§6.7) | Held-out MSE to V56 −0.9 % after training; D21 58–70 % (gate MSE ≤ 1e-9, D21 and d3 ≤ 2 %) | E |

### 6.2 κ₄ channel

| Idea | Measured (gate) | Kill |
|---|---|---|
| Per-row λ fitted to the (3,1) slice | 2.2911 → 2.2988e-8 (worse) | K |
| Depth multiplier, pairwise transport, 13-feature regression | ×1.25 / ×1.16; corr 0.24–0.59 | K/P |
| Young κ₄ carrier, h ≤ 3 | G4-weighted error 0.188 / 0.173 (ideal) | C |
| Quartic response kernel | −6.4 / −2.6 % at ≈ 42 u; not re-tuned | E (parked) |
| K₃×C all-distinct term | −8.8 % (MLP #0) at ≈ 140 u; cheap form −4.2 % at ≈ 13 u, synthetic run unfinished | E (parked) |
| Nine (2,1,1) hypertree classes | R² ≤ 0.07 | P |
| Carried pair-response modes | ≤ 3 % error cut against a 15 % gate | P |
| SSC 4-mode closure + drift MLP | corr 0.25–0.33 at L15; held-out R² ≤ 0.004 (gate 0.10) | P |
| Role-split λ | ×0.971 / ×0.998 (gate 0.97 on both); graded +0.77 % (#332382) | K |
| Per-source pair innovations γ, η (scalar form) | ×0.92 in sample, ×1.05 out of sample | P |
| Exchange vertex 3YC⁻¹Y | corr −0.07 to −0.23; error ×26–95 | P |
| Source-resolved skew energy | +≤ 0.004 corr | C |
| Every second-order carrier in §5.4 | 0.03–0.56 (gate 0.70) | C |
| Augmented dense κ₄ core (504aldo F105) | 2.04e-8 vs 2.23e-8 at +3 dense units per source-layer | C (cost) |
| T·T: κ₃ squared at a cubic vertex (§5.8) | Alone −0.16 to −0.06; adds ≤ 0.006 to star + C·T; oracle E2E ×0.792 against ×0.773 without it | C/E |
| LKV readouts injected as the raw κ₄ diagonal | Level ratio of the ages 1–6 sum −0.02 to 0.00 (§5.8); usable only as a feature | C |
| Low-rank C·T sketches, r = 32–256 (eig, Rademacher, direct-Γ) | Teacher-metric corr 0.60–0.63 at r = 64 against 0.73–0.78 exact (round 1); oracle E2E ×0.843–0.885 against ×0.773; ≈ 40 u (estimate) against a ~30 u break-even. Round 3 reopens it: eig r = 32 gives ×0.874 at an estimated 18–20 u (§5.8) | E (parked, cost; r = 32 reopened) |

### 6.3 Correctors and non-cumulant families

| Idea | Measured |
|---|---|
| Covariance patches | 2–3e-6 RMS against an error of 6.5–9e-5 |
| True D21 in the feedback legs only | ×0.9999 |
| Low-rank covariance-error correction | Needs rank ≈ 256 |
| Exact old tier | −3.9 % MSE for +19 % cost |
| Online mean-correction refit | R² 0.02 (chance level 0.013) |
| Learned residual: ridge / GBM | ×0.997 / ×0.995 |
| Monte-Carlo blends and control variates | Weight ≈ 0.002; surrogate corr 0.19 at L16 |
| Copula, TT-cross (60–500 % at width 32), TAP/AMP, SOS | Dead (§4.7) |
| ARC reference K3 at width 1024 | 1523 u, raw 3.7–4.5e-8: 5.9× dearer and 1.7× less accurate than V29 |

### 6.4 Outside-AI proposals

We consulted outside models in 18 numbered rounds and in further consultations on 10–11 Oct. Below, "In PDF?" states whether the proposal itself contained a measurement on our data. Every verdict is our own measurement.

| Round, author | Proposal | In PDF? | Our verdict |
|---|---|---|---|
| R1–2 (8 models) | Compensating closure without the old tier | No | Dead: 32× worse |
| R3, GPT Sol / Astra | Source-index coding; low-rank birth operator | No | Dead (§6.1) |
| **R4, GPT Sol** | **Spherical-input reparameterisation (from a public repo)** | No | **Shipped (V31), −1.1 % raw** |
| R5, Sol / Astra | 20 families (TAP/AMP, copula, TT-cross, SOS, …) | No | Dead |
| R7, Astra | Quartic κ₄ kernel; teacher-forced closure diagnostic | No | Kernel parked; the diagnostic shaped §4 |
| R7, Grok | Monte-Carlo surrogate, "40×" | No | Measured ×1.0–2.2 |
| R7, unknown pasted AI | "Closed-form" (1,1,1,1), "30×" MC | No | Wrong formula; claim baseless |
| R7–8, Fable (internal agent) | Error decomposition, FB scale, K₃×C | **Yes** | FB 1.25 shipped (#332110) |
| R9, GPT Sol | DHMP, FHQR-2, QCG ("promotion targets, not predictions") | No | Dead (36–40 %, 16 %, by arithmetic) |
| R10, several validators | Young-tier contraction search | Yes (counting scripts) | Confirmed 193 products; one answer's 49–55 u claim was an arithmetic error |
| R11, Copilot | Response-state closure | No | D21 35.9 % even from the true state |
| R11, SciMaster/Bohrium | Diagonal κ₃/κ₄ + low-rank C, "115 u at 1.18× chain MSE" | **Claimed, no run exists** | d3 error 90–99 % |
| R11, Astra | Channel-history state | Declined ("no update rule") | — |
| R12, Opus (internal agent) | Deep κ₄ is carried memory | **Yes** | Basis of §5.2 |
| R13, unsigned | Skew-lift carrier | No | corr 0.03 |
| R13, Astra | Dense pair core | **Yes**, self-rejected | corr −0.35 to −0.22 |
| R14, unsigned | Response-cashout; exchange vertex | No | Dead (§6.1, §6.2) |
| R15, Astra | Skew energy | **Yes**, reproduced identically | +0.004 |
| R16, unsigned | Signed bilinear quartic memory | Attempted; data host unreachable; declared none | 0.33–0.39 |
| R16b, Astra | Pair-innovation carrier | **Yes**, reproduced exactly | 0.36–0.44 |
| R17, Claude (ours) | Hub-pair C·T births | Yes | 0.43–0.44 |
| R17, unsigned | GHQ-6 hypertree bank | Script, own gate KILL | 0.47–0.56 |
| R17, SciMaster | Factorised curvature-birth carrier, "confirms ≥ 0.70, passes gate" | **Claimed; the script crashes** (`KeyError 'D21'`); patched, its own gate prints KILL | 0.39 / 0.32 / 0.40 |
| R17, SciMaster | Curvature-source factorised carrier, "exceeds 0.7 (PASS)" | **Claimed, no script or output** | The identical design measured 0.43 |
| **R18, unsigned** | **HTQB star + path** | Algebra self-test only (n = 4) | **Borderline alone; star shipped as V56** |
| 10 Oct consultation | κ₃ first-order audit: RLL term and the γ-patch | No (derivations only) | Both killed (§6.5 (a), (b)) |
| 10 Oct consultation | Cost-first κ₃ representation at 60 % of the cost | No; declared that no representation qualifies | Negative; slice-state obstruction (derived, §6.6 (g)) |
| 10 Oct consultation | Checkpointed birth archive (CBA-13) | No | ×14.8 to ×125 (§6.6 (d)) |
| 11 Oct consultation | Network-conditioned solver routing | Yes (audit of our archived runs, with script) | Oracle ceiling 0.80 %; not built (§6.6 (g)) |
| 11 Oct consultation | Learned pair-memory student (PM364) | No | Implemented by us; killed (§6.7) |

Two patterns stand out:
- **Renamed dead families.** Several proposals re-derived a dead family under a new name. The response-cashout propagator is keep-last-1; QCG is an n-term CP; FHQR is a short horizon.
- **Claims without runs.** Three documents reported passing measurements that no run supported.

Including the kill list and the project metric in every prompt is what filtered these out.

### 6.5 Hypotheses killed on 10 Oct

§4.10 places the remaining κ₃ error in the carried late-layer slice and in the d3 level. Three mechanisms for it, and for the FB 1.25 paradox of §2.4, were tested on 10 Oct. Each is killed as measured.

**(a) A missing cubic-response C×κ₃ term ("RLL"; P).** The hypothesis, from a 10 Oct outside-AI audit, was that the chain misses half of the term with one cubic and two linear Hermite vertices, at first order in the pre-activation κ₃ T.

*Derivation (derived).* With the K = 3 response h − E h = l u + q (u² − v) + r (u³ − 3vu) (§5.8), the connected partitions give

  κ(R_i, L_j, L_k) = r_i l_j l_k [3 C_ij T_iik + 3 C_ik T_iij]  (+ κ₅, dropped).

The proposal also contained a C_jk T_iii term. That partition is disconnected, because the block {i, i, i} does not meet {j, k}, so it does not belong in the joint cumulant. We tested both the connected form and the form as proposed.

*Test* (`astra_rll_test.py`). With the teacher's inputs at layer l, we map the term to its image dD on the (2,1) slice at l + 1. We compare dD with the chain's residual E = teacher − chain at l + 1 (the captures of §4.10), off the diagonal and on d3 separately. We report:
- β = ⟨E, dD⟩/|dD|²;
- ρ = corr(E, dD);
- r_half = |E − dD/2| / |E|, the residual left after adding half of the term.

The pre-registered kill was β ≈ 0 or negative, or ρ < 0.1 at L ≥ 9. Values are given for the transitions l → l + 1, l = 2–15, with l ≥ 9 in brackets:

| Form | Part | Size relative to the teacher's slice | ρ | r_half |
|---|---|---|---|---|
| Connected | Off-diagonal | 4.1–9.5 % | −0.04 to +0.07 (−0.04 to +0.02) | 1.08–1.42 (1.15–1.31) |
| Connected | d3 | 4.5–10.0 % | −0.24 to +0.07 (−0.24 to −0.12) | 1.23–2.03 |
| As proposed | Off-diagonal | 8.5–13.7 % | −0.06 to +0.02 | 1.23–1.89 |
| As proposed | d3 | 8.9–15.1 % | −0.23 to +0.01 | 1.54–2.95 |

*Verdict: KILL.*
- The connected term is comparable in size to the residual itself (4–10 % of the slice, against a residual of 4–7 %), yet it is uncorrelated with the off-diagonal residual (ρ ≈ 0, β −0.04 to +0.02 at l ≥ 9).
- On d3 it is anti-correlated at depth.
- Adding half of it makes the off-diagonal slice residual worse, by 15–42 % from L3 on (8 % at L2 → 3), and makes the d3 residual worse by 23–103 %.

This is consistent with the chain already carrying the term in its feedback legs.

**(b) A feedback-scale confound ("γ-patch"; E).** The same audit proposed the following explanation for the paradox of §2.4. The FB scale multiplies not only the linear K×C coefficient but also a quadratic Sym(Xt, P, Yt) cross term of the thin feedback legs, which grows as FB². FB 2.0 would then restore the linear term correctly but overshoot the quadratic one.

The patch (`experiments/variants/v56_fbg.py`) sets the linear coefficient to FB and multiplies only the Xt·Yt contributions by γ = (FB_base/FB)², which holds them at FB_base². We ran it on the 100-MLP harness (§8.2) on top of the V56 A-class estimator (v56As_fs, FB 1.25). No MLP failed in any run. Paired ratios are against the base:

| Variant | Linear FB | Quadratic Xt·Yt term held at | Raw MSE | Score | Paired geo-mean ratio (90 % CI) | Wins |
|---|---|---|---|---|---|---|
| base (V56 A, shipped) | 1.25 | 1.25² | 2.0194e-8 | 3.7586e-9 | 1 | — |
| Unpatched FB 2.0 | 2.0 | 2.0² | 2.0689e-8 | 3.8508e-9 | ×1.0244 (1.0211–1.0277) | 11/100 |
| γ-patch, FB 2.0 | 2.0 | 1.25² | 2.0686e-8 | 3.8522e-9 | ×1.0243 (1.0209–1.0273) | 10/100 |
| γ-patch, FB 2.0 | 2.0 | 1.0² | 2.0686e-8 | 3.8522e-9 | ×1.0243 (1.0212–1.0274) | 11/100 |
| γ-patch, FB 1.75 | 1.75 | 1.25² | 2.0403e-8 | 3.7995e-9 | ×1.0103 (1.0080–1.0124) | 25/100 |

*Verdict: KILL.*
- Holding the quadratic term at 1.25², at 1.0², or letting it grow to 2.0² gives the same result (×1.0243–1.0244), so the quadratic cross term is inert.
- The loss follows the linear coefficient: ×1.0103 at 1.75 and ×1.0243 at 2.0. The linear K×C restoration itself hurts the final mean.

The paradox of §2.4 remains open, with this explanation removed.

**(c) The κ₄→κ₃ feed channel (C/E).** A 10 Oct consultation (`consultations/fable_2026-10-10_k3_design.md`) proposed the following. Among the first-order terms of the all-distinct post-ReLU κ₃, the only one whose input the chain lacks is the feed g_c Φ_i Φ_j κ₄(z_m)_ccij from the pre-activation κ₄ at a quadratic vertex c (derived in the design note). According to the design note's reading of the code, the chain computes this feed from its memoryless κ₄ model (§1.2), whose only pattern is C and which does not depend on the vertex neuron c. The design would replace it with a carried cubic κ₄ (star + C·T, §5.8) in thin legs.

*Ceiling* (`fable_k3feed_ceiling_2026-10-10.py`; Kaggle CPU in five parts; N = 10⁶ exact samples of MLP #0; targets L9–16; `k3feed_mlp0_part{1..5}.json`). The script forms several versions of the feed read on the (2,1) slice at t:
- **exact:** the exact feed of every vertex;
- **bornJ4 / bornJ6:** only the κ₄ born within J layers, which is the ceiling of any carrier with J-layer births;
- **star_J4 / star_J6:** the Gaussian-star-fed feed in closed form.

The sampled C·T part of the design's representable feed was not run. As a check, the feed from the Gaussian layer z_1 sits at its own noise level (0.13–0.18 against noise 0.14–0.20).

| Quantity, L9–16 | Value |
|---|---|
| Exact feed size, off-diagonal, relative to \|Y21\| | 22.5–24.2 % |
| Exact feed size, diagonal, relative to \|d3\| | 23.3–23.6 % |
| Split-half reliability of the exact feed, off-diagonal / diagonal | 0.871–0.966 / 0.939–0.981 |
| Sampling noise of the exact feed, relative to \|Y21\| (derived) | 3.3–6.0 % |
| bornJ6 against exact, cross-half corr, off-diagonal / diagonal | 0.871–0.960 / 0.938–0.976 |
| bornJ4 against exact, same | 0.861–0.949 / 0.932–0.969 |
| star_J6 against bornJ6, same | −0.871 to −0.921 / −0.927 to −0.956 |
| star_J6 size relative to \|Y21\| | 12.9–15.9 % |

The pre-registered validity sub-gate, a split-half reliability of at least 0.90 at every target, is not met at L9 (0.871) or L10 (0.894).

*End to end* (`experiments/variants/r20_feed_oracle.py`). The variant removes the chain's own feed at the slice readout in a window and adds a feed read from file. It runs on the plain chain of §4.8, MLP #0. Hook validity: the plain configuration reproduces 2.2732e-8 exactly (`teacher_eval_feedor1.json`). The util of these runs is not comparable with the chain's, because the hook adds a second readout.

| Feed in the window | Window | Final MSE | Ratio |
|---|---|---|---|
| None (chain feed removed) | L9–16 | 1.715e-7 | 7.54 |
| None | L1–16 | 2.724e-7 | 11.98 |
| Exact (sampled) | L9–16 | 2.535e-8 | 1.115 |
| bornJ6 | L9–16 | 2.955e-8 | 1.300 |
| bornJ4 | L9–16 | 4.409e-8 | 1.940 |
| star_J6 | L9–16 | 4.783e-7 | 21.0 |
| star_J6, sign flipped | L9–16 | 4.255e-8 | 1.872 |
| Chain's own feed × 0.80 / 0.90 / 0.95 | L9–16 | 2.777 / 2.371 / 2.284e-8 | 1.222 / 1.043 / 1.0045 |
| Chain's own feed × 1.00 (dumped and re-injected) | L9–16 | 2.2725e-8 | 0.9997 |
| Chain's own feed × 1.10 / 1.20 | L9–16 | 2.487 / 3.011e-8 | 1.094 / 1.324 |

The removal runs are in `teacher_eval_feedor2.json` and the injections in `results/summaries/oct10/feedorE2E_{3,4,5,6}.json`. The pre-registered gates were: exact ≤ 0.80 PASS and > 0.92 KILL; bornJ6 ≤ 0.85 PASS and > 0.95 KILL. For reference, the teacher's κ₃ in L9–16 is worth about ×0.69 (§4.10).

*Slice-level attribution (measured; `results/summaries/k3feed/feed_attribution.json`, script `experiments/probes/k3_error_profile.py`).*
- At L9–L15, the chain's own feed correlates +0.95 to +0.97 with the exact feed. The exact feed is 12–26 % larger.
- The correction (exact − chain feed) correlates +0.41 to +0.59 with the chain's slice residual.
- Yet replacing the chain's feed with the exact one still increases the slice residual: from 5.2–6.2 % to 6.0–6.8 % of |Y21|.
- The best scalar rescaling of the chain's feed (β = +0.03 to +0.08) lowers the slice residual by only 0.5–3 %.

Reading:
- **The feed is large, and the chain needs it.** It is 22–24 % of |Y21|, and removing it costs ×7.5 (L9–16) or ×12 (all layers).
- **The chain's feed has the right shape and the wrong size**, and the correction points along the residual.
- **Every injection is nevertheless worse than the base, the exact feed included** (×1.115, KILL by its own gate). The chain's feed sits at a sharp optimum at scale 1.0: −5 % costs 0.45 % and ±10 % costs 4.3 % / 9.4 %. Scaling it up by 10 % hurts, although the exact feed is 14–24 % larger.
- **The sampled feed is noisy at the level that matters.** Its sampling noise, 3.3–6.0 % of |Y21|, is comparable to the chain's whole slice residual at L9–16 (about 5–7 %, §4.10). This would explain why the exact feed raises the slice residual although its correction is correlated with it.
- **The Gaussian star has the opposite sign to the sampled born feed** (corr −0.87 to −0.92). We have not established whether this is a sign-convention error in the closed form or a physical sign **[OPEN]**. Neither sign helps: ×21.0 as computed and ×1.872 flipped.

*Verdict: KILL as measured.* The experiment is decisive about the N = 10⁶ estimate of the exact feed, not about the true feed. The ceiling is noise-limited: reliability is 0.87–0.97, and the validity sub-gate failed at L9–10. Bringing the feed noise well below the residual would need about 10× more samples. With noise ∝ 1/√N, that would put it at about 1.0–1.9 % of |Y21| (estimate). A definitive ceiling, and the sampled C·T part, remain **[OPEN]**.

### 6.6 Cost-side architectures killed on 10–11 Oct

The ledger of §3.6 puts 133 u of V56's 194 u into per-source families whose size grows with depth. On the 10 Oct leaderboard our raw MSE was already in the range of entries ranked 11–40, while our util was 0.187 against 0.10–0.14 at the top (§7.3). The tests below ask whether the per-source accumulation can be removed, shortened or deferred.

**(a) Single-pool κ₃ ceiling (C).** A carrier of fixed size would keep the whole κ₃ in one shared rank-r basis U, for young and old sources alike. Its best case is to project every leg of every source, X → UUᵀX, and read D21 and d3 from the projected legs. `experiments/probes/pool_ceiling.py` does this on the plain-chain captures of §5.8 (every source dense) at L11, L14 and L15. U is either the top-r eigenvectors of the chain's own covariance at that layer ("C basis"), or the top singular vectors of the mode-1 unfolding of κ₃ from a randomized range finder ("unfolding basis"). Errors are relative Frobenius errors of the readout against the unprojected sources (`pool_ceiling.json`):

| Rank r | All sources pooled: D21 off-diagonal | All sources pooled: d3 | Only sources older than 5 layers pooled: D21 off-diagonal |
|---|---|---|---|
| 128 | 19.6–28.1 % | 13.2–20.5 % | 6.2–6.6 % |
| 256 | 11.4–17.3 % | 7.2–11.2 % | 2.4–2.5 % |
| 384 | 6.6–10.8 % | 4.1–6.9 % | 1.2–1.3 % |
| 512 | 3.4–6.3 % | 2.2–4.0 % | 0.6–0.7 % |
| 768 | 0.2–1.0 % | 0.1–0.6 % | 0.03–0.09 % |

All values use the C basis. The unfolding basis is worse at every rank: at r = 384 it gives 8.2–14.0 % for the whole κ₃.

*Verdict: KILL.* The pre-registered gate was a D21 error ≤ 2 % at L11–15 for some r ≤ 512. Only r = 768 passes, and a rank-768 pooled leg keeps three quarters of a dense leg's columns, so it saves little (derived). The table also shows where the obstruction sits. Pooling only the old sources at r = 384 costs 1.2–1.3 %, which is today's split. It is the **young** sources that no shared basis holds.

**(b) DEJ-4: a deferred exact join (C).** The deferred exact join (DEJ) would keep only the youngest source exact and fold every older source into a shared basis at once. The DEJ-4 falsifier runs the same script with the youngest k = 1, …, 4 sources exact and all older sources projected at r = 352, 448 or 512 (C basis; `oct10/dej_young{1..4}.json`). The older legs are projected from their exact values, so this is the best case of an exact join at age k + 1. D21 off-diagonal error at L11–15:

| Youngest sources kept exact | r = 352 | r = 448 | r = 512 |
|---|---|---|---|
| k = 1 (the DEJ design) | 5.7–8.1 % | 3.6–5.3 % | 2.5–3.9 % |
| k = 2 | 3.7–5.2 % | 2.4–3.3 % | 1.7–2.4 % |
| k = 3 | 2.6–3.2 % | 1.6–1.9 % | 1.1–1.5 % |
| k = 4 | 1.9–2.2 % | 1.2–1.4 % | 0.8–1.0 % |

*Verdict: KILL for k = 1* (2.5–8 % against the 2 % gate). The settings that pass, such as k = 4 at r = 448 (1.2–1.4 %, d3 0.9–1.2 %) and k = 3 at r = 512 (1.1–1.5 %, d3 0.9–1.3 %), are not a new architecture. They are other values of V56's own age gate and old-basis rank, so they were graded end to end in (c).

**(c) The age/rank grid on V56, 100 MLPs (K).** The 100-MLP harness (§8.2) ran five settings of the young-age gate AGE_OLD and the old-basis rank R_OLD on the A-class V56 (`kaggle_agegrid_100mlp.json`). The base is the shipped AGE_OLD 5, R_OLD 352, run in the same harness for the γ-patch test (`base_fbg` in `kaggle_fbg_100mlp.json`). No MLP failed. The ratios are ratios of the harness's aggregate scores.

| Variant | AGE_OLD | R_OLD | Raw MSE | Util | Score | Score ratio |
|---|---|---|---|---|---|---|
| base (V56 A, shipped) | 5 | 352 | 2.0194e-8 | 0.1861 | 3.7586e-9 | 1 |
| age4_r448 | 4 | 448 | 2.0090e-8 | 0.1920 | 3.8569e-9 | ×1.026 |
| age4_r352 | 4 | 352 | 2.2271e-8 | 0.1781 | 3.9666e-9 | ×1.055 |
| age4_r512 | 4 | 512 | 1.9849e-8 | 0.2021 | 4.0111e-9 | ×1.067 |
| age3_r512 | 3 | 512 | 2.0569e-8 | 0.1988 | 4.0889e-9 | ×1.088 |
| age3_r448 | 3 | 448 | 2.2268e-8 | 0.1863 | 4.1479e-9 | ×1.104 |

*Verdict: KILL; the one-for-one law holds (§3.5).*
- The best ceiling point, age4_r448, gains 0.5 % raw for 3.2 % more util.
- age4_r512 gains 1.7 % raw for 8.6 % more util.
- Saving cost at r = 352 (age4_r352, util −4.3 %) costs 10.3 % raw.

The ceiling ranking of (b) predicts the raw ranking, but every move away from the shipped setting costs more in one factor of the score than it gains in the other.

**(d) CBA-13: a checkpointed birth archive (E).** A 10 Oct consultation proposed deferring κ₃ work: run the early layers with a cheap κ₃, keep the births in an archive, and rebuild the exact sources at a checkpoint. The falsifier (`experiments/variants/r20_cba.py`) gives the proposal its best case. Inside a window of layers the closures see κ₃ = 0 (both D21 and d3), while the source bank keeps evolving **exactly**, so the exact κ₃ is available again for free after the window. In mode "zero", the births' feedback and repeated-index inputs see zero too. In mode "zero_closure", the births keep the true D21. The run uses the plain chain on MLP #0, with base 2.2732e-8 (`oct10/cba_{1,2,3}.json`):

| Window with κ₃ = 0 in the closures | Final MSE | Ratio | Per-layer MSE at the window's last layer → at L16 |
|---|---|---|---|
| L1–4 | 3.358e-7 | ×14.8 | 7.55e-7 → 3.36e-7 |
| L1–8 | 1.448e-6 | ×63.7 | 1.98e-6 → 1.45e-6 |
| L9–12 | 1.060e-6 | ×46.6 | 1.11e-6 → 1.06e-6 |
| L5–12 | 2.333e-6 | ×102.6 | 2.43e-6 → 2.33e-6 |
| L1–12 | 2.855e-6 | ×125.6 | 2.96e-6 → 2.85e-6 |
| L1–12, births keep the true D21 ("zero_closure") | 2.851e-6 | ×125.4 | — |

*Verdict: KILL.*
- **The damage does not recover after the window.** With the exact sources restored, the per-layer error falls by at most a factor of 2.2 by L16 (window L1–4), and by only 3.5–4.5 % for windows that end at L12. The mean and covariance error made inside the window is carried to the output (§4.1).
- **The damage enters through the closures.** Keeping the true D21 in the births changes nothing (×125.4 against ×125.6).
- **Early κ₃ is needed although it is not the lever.** Replacing the chain's κ₃ by the teacher's at L1–4 is worth nothing (×1.004, §4.10), yet removing it at L1–4 costs ×14.8.

**(e) κ₃ level scaling, at zero FLOPs (K).** §4.10 found that the chain's d3 level shrinks by 1.4 % at L16. The same harness (`r20_cba.py`, mode "scale") multiplies d3 and/or the D21 slice by a fixed factor in a window. The run uses the plain chain on MLP #0 and on the synthetic net, with bases 2.2732e-8 / 1.9201e-8. Results: `results/summaries/oct10/k3scale_{1,2,3}.json`.

| Change | Window | MLP #0 | Synthetic net |
|---|---|---|---|
| d3 ×1.02 | L9–16 | ×1.032 | ×1.015 |
| d3 ×0.98 | L9–16 | ×1.038 | ×1.047 |
| d3 ×1.01 | L1–16 | ×1.005 | ×1.013 |
| D21 ×1.02 | L9–16 | ×1.057 | ×1.016 |
| D21 ×0.98 | L9–16 | ×0.984 | ×1.021 |
| d3 and D21 ×1.02 | L9–16 | ×1.074 | ×1.017 |

*Verdict: KILL.* The κ₃ level sits at its optimum. Raising d3 toward the teacher's level, the direction that §4.10 suggests, is worse on both networks. The one gain, D21 ×0.98 on MLP #0 (×0.984), does not transfer to the synthetic net (×1.021). This is one more instance of §7.5: a local correction toward the teacher loses end to end.

**(f) The reference paper, read for unused ideas (Wu et al., arXiv:2605.05179).** We checked the paper's K = 3 ideas against `submissions/v56As_fs/estimator.py` (`consultations/wu_paper_deep_read_2026-10-10.md`):
- **Already in V56:** merging factor groups that share two legs (V56's M block fuses P·s + 3A·e + Z Lᵀ); the final-layer trim to the D3 diagonal (V19); the symmetric-output product that computes three of four blocks (`_sym_product`); a κ₄ diagonal carried with memory (dG = (W∘W) g_prev + λ-term, with only the off-diagonal core regenerated); and the mixed covariance-closure terms (V56's K3-simple term tables are a superset of the paper's k ≤ 3 frontier).
- **Cost:** the paper's factorised K = 3 costs about 4,150 u at our size (derived from its Table 1), about 21× V56 (§3.6).
- **Killed: the trace projection of old κ₃** (the paper's rotation-invariant part). On the captures at L11, L14 and L15, the old sources carry 35–57 % of |D21|. Their trace projection Sym(h₁ ⊗ I) leaves 74–75 % error in the old part, i.e. 26–42 % of the total D21, against a 2 % tolerance.
- **Untested:** the augmented K = 2 chain at about 88 u (the paper's estimate), which would sit under the 0.1 floor. The paper's own error theory makes it unlikely to reach our accuracy (estimate).

**(g) Cost-first consultations (10–11 Oct; negative).** We asked outside models for a κ₃ representation at 60 % of V56's cost (116.4 u) that keeps D21 and d3 within the 2 % channel tolerance. Three results came back.
1. **No qualifying representation.** The consultation declared that it had none, and it declined to present a measured-dead compression or slice recursion as one. The budget arithmetic is that of §3.6.
2. **A slice-only state cannot be exact to first order (derived in the consultation; we restate it because it is short).** Let S(T)_ik = T_iik be the (2,1) slice, and let U = Wᵀ diag(w₁) be the transport. Take three distinct indices a, b, c and the perturbation H = ε Sym(e_a, e_b, e_c). Then S(H) = 0 and diag₃(H) = 0, so the current mean, covariance and slice cannot see H. After one transport, with u_a the a-th column of U,

     S(U^⊗3 H) = (ε/3) [ (u_a∘u_b) u_cᵀ + (u_a∘u_c) u_bᵀ + (u_b∘u_c) u_aᵀ ],  diag₃(U^⊗3 H) = ε u_a∘u_b∘u_c,

   which is generically nonzero at first order in ε. A recursion on the slice alone therefore cannot be exact through first order for general third cumulants. A linear state that is closed under every invertible transport and returns every diagonal must be injective on Sym³, which takes n(n+1)(n+2)/6 ≈ 1.8·10⁸ numbers (compare §2.6). The argument does not exclude nonlinear encodings such as the source chain itself, restricted families, or approximate ones. Its measured counterparts are: 91–96 % of the chain's slice error sits in the all-distinct entries (§4.5); slice-to-slice recursion has R² 0.03–0.05 (§6.1); and PM364 fails (§6.7).
3. **Routing between budgets has a ceiling of 0.8 %.** The proposal was to inspect each network's weights cheaply and select one of several complete chains of different cost. An audit of our archived 100-MLP runs (`Architecture Analysis/routing_oracle_audit.py` → `archived_routing_audit.json`, supplied with the consultation) covers 86 variants matched on the same 100 MLPs. A selector that knew every variant's true per-MLP score would improve on the best fixed variant (3.7586e-9) by only 0.80 %, before any selector overhead (0.86 % over the 10 V56 tuning variants). The cheapest archived variant costs 188.9 u, so the bank holds no cheap expert. A bank with a genuinely cheaper expert is untested. The router was not built.

### 6.7 Round 20: a learned pair-memory student, PM364 (11 Oct; killed)

§6.6 shows that the per-source state cannot be pooled, deferred or shortened. The remaining route is a **source-free** state of fixed size whose missing content is *learned*. The design of a pair-memory student came from an 11 Oct consultation. We adapted it, implemented it (`experiments/student/`) and distilled it from V56.

**Design.** The student carries the mean, the covariance C and two n×n **pair memories** H₁ and H₂, with no per-ReLU sources. Per layer, with w the layer's weight in the row convention z = h w:
1. **Transport.** The mean and C are transported as usual. Each memory is transported by the restricted cubic operator P_k = (w∘w)ᵀ H_k w, which is the exact image of a κ₃ supported on its slice entries.
2. **Readout.** D21 = offdiag(P₁) + O, where O_ij = o_i (P₂)_ij − o_j (P₂)_ji is an explicitly odd correction. d3 uses the exact restricted diagonal rule (1 + g_i)·[3((P₁)_ii − c_i) + c_i], with c = (w∘w∘w)ᵀ diag(H₁). V56's dead-column rule is applied.
3. **κ₄ inputs.** Exactly V56's: the memory-carried diagonal, the adaptive λ, and the V55 pattern map.
4. **Closure.** V56's K3-simple closure, ported exactly. It returns the post-ReLU mean, variance, d3, κ₄, covariance K11 and slice K21.
5. **Memory update.** The post-ReLU slice is the closure's own output, P_full = K21 + diag(K3), which already contains the births, the κ₄→κ₃ feed and the linear dressing. The memories are updated as H₁ ← P_full + (ρ − 1)⊙L(H₁) + η⊙L(H₂) and H₂ ← (P_full − P_fullᵀ)/2 + η⊙L(H₁) + ρ⊙L(H₂), with L(H)_ij = w1_i² w1_j H_ij.

The four heads ρ, η, o and g are quadratic in 12 bounded per-neuron features, giving 4 × 91 = **364 parameters**. They are initialised at zero, so the zero model is the pure slice recursion.

**Engineering checks (measured).**
- **Closure port.** With teacher-forced inputs, the PyTorch port of the V56 closure matches the chain to about 5e-8 relative error in the post-ReLU mean per layer, and to 4e-6 relative in the final prediction (MSE 1.5e-11).
- **flopscope version.** `pm_flops.py` matches the PyTorch version to 2.8e-7.
- **Cost (estimate).** It issues about 9.7k operations. Extrapolated from n = 128, it costs about 93 u, below the 102.4 u floor. A successful student would therefore score about raw MSE × 0.1 (derived), against V56's × 0.186.

**Data and training.**
- **Labels.** 768 self-generated He networks (W ~ N(0, 2/n), float32, seeded with numpy `default_rng`) were labelled by one run each of the frozen V56 with a dump hook (`v56_dump.py`, `teacher_dump.py`). The seed ranges are 100000–100511 (train), 200000+ (validation) and 300000+ (test). The labels are V56's own per-layer quantities, so the student is distilled to V56, not to the Monte-Carlo truth.
- **Training.** `train_pm.py` uses AdamW with warm-up, cosine decay and gradient clipping. Every step is a free-running 16-layer rollout from layer 1, with no teacher forcing. The loss weights the final-layer MSE, the means, sampled covariance and D21 rows, d3 and κ₄ terms, with the D21 and d3 terms weighted 4× more at L9–16 than at L1–8.
- **Hardware.** One T4 GPU: 0.95 s forward and backward per network, 7.5 GB.

**Result** (seed 11; 448 training networks; 64 held-out networks from the training seed range, never trained on; 16 epochs; `results/summaries/student/pm364_s11_history.json`):

| Held-out quantity | Zero model (epoch 0) | After 16 epochs |
|---|---|---|
| Final-layer MSE to V56 | 9.64e-7 | 9.56e-7 (−0.9 %) |
| D21 relative error, L2 | 57.6 % | 57.6 % |
| D21 relative error, L3–16 | 63–70 % | 63–69 % |
| d3 relative error, L2–16 | 25–39 % | 25–39 % |
| Training loss (epoch 1 → 16) | 747.5 | 743.6 (−0.5 %) |

A second optimisation seed (23; same data and protocol; `results/summaries/student/pm364_s23_history.json`) reproduces this: held-out MSE to V56 goes from 9.64e-7 to 9.55e-7, and the D21 error stays at 57.6 % at L2 and 67–70 % at L15–16.

The pre-registered gate was a held-out MSE to V56 ≤ 1e-9 and D21 and d3 errors ≤ 2 %. *Verdict: KILL* (both seeds). The student's MSE to V56 is about 50× V56's own error against the truth (derived). The test split was not evaluated after the kill.

**Interpretation.** The D21 error at L2 is 57.6 % at the first transport and is not moved by training. At L1 the memory holds only the closure's slice. The share of the next slice that the transport would draw from the all-distinct post-ReLU κ₃ entries is therefore missing, and no per-row head (a function of per-neuron features) can rebuild it. This agrees with:
- the first-order obstruction for slice-only states (§6.6 (g));
- the location of the chain's error in the all-distinct entries (91–96 %, §4.5);
- the age-memory table of §4.5, where keeping only ages ≤ 2 leaves 84 % D21 error.

A source-free state must carry all-distinct content in some form. No fixed-size form we know does so at the 2 % tolerance (§6.1, §6.6 (a)).

---

## 7. What a better method must satisfy

### 7.1 Structural constraints (from §2 and §4–§6)

1. **Keep the κ₃ chain at full fidelity.**
   - The mean closure needs κ₃ to about 1 % (§5.6) and D21 to about 2 % (tolerance law).
   - Cheap κ₃ cannot be bought back with κ₄ (§5.1).
   - The κ₃ memory is ≥ 10 ages long, coherent, and not low-rank (§4.5).
   - No compressed linear κ₃ state is closed under transport (§2.6), and none of the more than 20 κ₃ representations we tried came within 5× of its gate (§6.1).
2. **Carry odd information.** Any closure built from even statistics of the weights loses the deep κ₃ entirely (§2.2).
3. **Deliver the κ₄ pattern, not the level, on decisive neurons.**
   - The pattern alone is worth 108 % of the κ₄-oracle gain.
   - Reaching correlation ≥ 0.70 at L12 / L15 / L16 requires births that contain **both the quadratic and the cubic Hermite response** of each ReLU.
   - These births must cover at least ages 1–4, and ideally 1–6 (§5.4, set C).
   - Second-order births of any form are capped near 0.45–0.56.
   - The V56 star plus a κ₃-dressed C·T readout meets this (ceiling 0.82–0.85), and the chain's own legs already carry its inputs (§5.8). It must enter as a feature of a rescaled map, because its level is near zero.
4. **Improve the κ₃ input first.**
   - The teacher's κ₃ slice and d3 are worth more than the κ₄ diagonal on both networks (×0.607 against ×0.686, and ×0.716 against ×0.885), and the two together give ×0.272 (§4.8).
   - A new κ₃ representation needs at least the cubic response of each ReLU: quadratic births make 10–21 % D21 and d3 error in one step, against a 2 % tolerance (§4.9).
   - The target is the carried late-layer error: net of teacher noise, 5–7 % in the slice at L8–16 and a 1.4 % shrink of the d3 level at L16. Teacher κ₃ helps only at L9–16 (§4.10). A missing C×κ₃ term, the feedback scale and the κ₄→κ₃ feed (at N = 10⁶) are ruled out as the cause (§6.5). A plain rescaling of the level is ruled out too (§6.6 (e)).
5. **Carry the all-distinct κ₃ content, at every layer, from the first layer on (§6.6–§6.7).**
   - A state built from the slice alone cannot be exact to first order (derived, §6.6 (g)). A learned 364-parameter student on such a state stays at 58–70 % D21 error (§6.7).
   - One shared basis cannot hold the young sources: the whole κ₃ needs rank 768 for ≤ 2 % (§6.6 (a)), and keeping only the newest source exact leaves 2.5–8 % (§6.6 (b)).
   - κ₃ cannot be skipped early and restored later: a zero window costs ×15 to ×125, and the damage persists after the window (§6.6 (d)).

### 7.2 Budget constraints

- **Ops and memory.** At most about 1,000 extra warm-call ops above the B-class base, which keeps the total ≤ ~18k for nomination. Total memory under 8 GB, measured under a subprocess runner. No host synchronisations.
- **FLOPs: break-even arithmetic.** A carrier worth 74 % of the oracle gain would cut raw to about ×0.78 (estimate, from the 50 %-blend oracle). On our ≈ 192 u:
  - at +30 u, the score improves by about 10 % (estimate);
  - at +60 u, it is about 2 % worse (estimate).
  
  So the carrier must cost ≤ ~30 u.
- **Starting point.** The V56 star already supplies part of the cubic signal at 0 u. A new carrier is worth only what it adds on top of V55 + star: 0.65–0.74 leave-one-layer-out, measured with teacher μ and C on one network.

### 7.3 Score targets (raw × util arithmetic)

| Target | Requirement |
|---|---|
| 3.0e-9 | κ₄-oracle raw (1.61e-8) at today's util of ≈ 0.187, or today's raw (1.95e-8) at util 0.154 (≈ 158 u) |
| Rank 12 on the 8–9 Oct board (2.2e-9) | raw 1.18e-8 at util 0.187, or 1.47e-8 at 0.15 |
| Leader class | raw 1.4–1.76e-8 at util 0.10–0.16 |

**Cost alone cannot get there.** Today's raw at the 0.1 floor would give 1.95e-9. That needs a 47 % cost cut, which §2.6 and §6.1 show is unavailable inside the sum-of-sources family.

The 1024-wide leaders therefore pair a raw MSE 1.2–1.35× lower with a cost 1.25–1.9× lower. Their mechanisms are not public.

**The 10 Oct board** (`consultations/competition_scan_2026-10-10.md`).
- The six highest entries have raw MSE 1.14–1.65e-8 at util 0.106–0.144.
- Our leaderboard entry shows raw 1.97e-8 at util 0.187 (rank 42). Entries ranked 11–40 have raw MSE comparable to ours (1.7–2.5e-8 among those listed).
- The entry at rank 11 has V29-level raw MSE (2.10e-8) at util 0.102, the floor.
- At our raw MSE, util 0.10 would score about 2.0e-9 and util 0.14 about 2.8e-9 (derived).

On that board the remaining gap is mostly **cost**, not accuracy. This motivated the cost-side tests of §3.6 and §6.6–§6.7. Within the sum-of-sources family, they confirm that cost and accuracy trade one for one (§6.6 (c)). Every source-free or pooled alternative we could build fails its 2 % D21 gate, by factors from about 1.25 (DEJ at r = 512) to about 35 (PM364).

### 7.4 Open design target [OPEN]

The design target is **a factorised κ₄ carrier whose births come from the cubic (and quadratic) Hermite response of each ReLU over ages 1–6, read through the existing κ₃ source legs, for ≤ 30 u and ≤ ~1,000 ops.**

Its ceiling is 0.80–0.85 (set C, ages 1–6) and 0.68–0.74 for ages 1–4. The value of reaching the oracle is about 3.0e-9 at the current cost (estimate).

**Status on 10 Oct (§5.8).** The readout side of this target now exists:
- star + C·T has a ceiling of 0.82–0.85, and the chain's own κ₃ legs reproduce C·T with fidelity 0.993–0.996;
- through an oracle affine map it is worth ×0.773 end to end;
- the age-collapse identity and A/P anchor grouping make it exact at two dense products per source and target.

The missing piece is cost:
- the eigen-sketch at r = 64 keeps ×0.865 at about 40 u (estimate), which is roughly score-neutral on V56's ≈ 190 u; at r = 128 it keeps ×0.843, with twice the probe's thin-product count (55.3 u against 27.6 u over targets 7–16);
- break-even needs ≤ ~30 u at that accuracy, or more accuracy at the same cost;
- round 3 (§5.8) found such a point: eig r = 32 keeps ×0.874 at an estimated 18–20 u, about a 3–4 % score gain (estimate), with the metered op count still unmeasured.

No estimator implementation exists yet, and the increment on top of V56, with a map fitted offline and transferred, is untested. In parallel, the joint oracle (§4.8) makes a better κ₃ input the larger open lever. Its first candidate mechanism, the κ₄→κ₃ feed, is killed at N = 10⁶ but is not ruled out at larger N (§6.5).

**Status on 11 Oct (§3.6, §6.6–§6.7).** The cost side has been mapped, and it is negative:
- the 133 u of per-source readout and transport cannot be pooled into one basis, deferred to a checkpoint or shortened in age without a one-for-one accuracy loss;
- routing between existing budgets is worth at most 0.8 %;
- a source-free learned student fails at the first layer, because its state has no all-distinct κ₃.

What remains open on the cost side is a representation that carries all-distinct κ₃ content at a cost that does not grow with the number of sources. We do not have a candidate.

Smaller open items:
- the cheap K₃×C term on the synthetic network (−4.2 % on MLP #0 at ≈ 13 u, unfinished);
- the HTQB path term end to end (+0.04–0.06 leave-one-layer-out, untested in the chain);
- the quartic kernel re-tuned jointly with λ and FB.

### 7.5 A lesson: local corrections against coupled errors

Again and again in this project, a change that makes one channel locally more correct made the final mean worse. All of the following were measured:
- **Full FB restoration** (§2.4) brings the slice closer to the derivation at L4–10, but the final mean gets 3.4 % / 3.1 % worse on the teacher pair. FB 2.0 is ×1.024 on 100 MLPs (§6.5).
- **The γ-patch** (§6.5), which keeps the linear restoration and removes its quadratic side effect, is no better (×1.0243).
- **Scaling the chain's κ₄→κ₃ feed** (§6.5) by ±5–10 % in either direction costs 0.45–9.4 %. This holds even upward, toward the exact feed, which is 12–26 % larger and whose correction correlates with the residual. Injecting the sampled exact feed gives ×1.115.
- **The analytic RLL term** (§6.5), of the right order and comparable size, makes the slice residual 15–42 % worse when half of it is added.
- **Single κ₄ inputs:** injected alone, the off-diagonal κ₄ slices give ×1.03–1.22 (§4.2), and the chain's κ₄ pattern at the teacher's level gives +21.5 % (§5.1).
- **The κ₃ level** (§6.6 (e)): the chain's d3 is 1.4 % too small at L16, yet raising it by 2 % at L9–16 costs ×1.032 / ×1.015 on the two networks. The one rescaling that helps on MLP #0, D21 ×0.98 (×0.984), hurts on the synthetic net (×1.021).

Only **joint** input oracles give large gains. The teacher's κ₃ and κ₄ together give ×0.272, against ×0.607 and ×0.686 alone (§4.8), and adding the variance takes the ratio to the noise floor (×0.057, §4.2).

The pattern is consistent with the chain's channel errors partly cancelling in the mean closure: correcting one channel uncovers the errors it was offsetting in the others. We have not measured this cancellation directly **[OPEN]**. In the feed case, Monte-Carlo noise in the injected quantity is a competing explanation.

The practical lesson for moment-chain estimators is narrower, and it rests only on the measurements above:
- a local test (a slice residual, a one-step error, a pattern correlation) is a necessary gate, but it is not a sufficient one;
- candidates must be promoted on the end-to-end mean;
- a single-channel correction should be expected to need joint re-tuning of the channels it is coupled to (λ, the FB scale, the feed scale).

FB 1.25 is the only correction in this list that gained (§2.4), and it did so at a quarter of the derived restoration.

---

## 8. Reproducibility

### 8.1 Versions and data

- Software: whestbench 0.16.1, flopscope 0.12.1.
- Dataset revision: v2-phase2.
- Every graded number in this paper carries its AIcrowd submission id. The shipped estimators are in `submissions/<variant>/estimator.py`, each with a lineage header naming every change; for example, `submissions/v56As_fs/estimator.py` (#334870) and `submissions/v56s_fs/estimator.py` (#334869).

### 8.2 Harness

The 100-MLP paired harness runs on the public mini split with its 10⁹-sample truth, using Kaggle CPU runners A–E (`kaggle/runners/`). It is compared with `experiments/probes/kaggle_pair_compare.py`, which reports:
- the geometric-mean ratio;
- wins;
- a bootstrap 90 % interval;
- the score MSE × max(0.1, util);
- the residual profile.

The base is bit-identical across runners, and paired standard errors are 0.01–0.15 %.

**Calibration against the grader:**

| Change | Harness | Graded |
|---|---|---|
| V55 B / A | Projected 3.8764e-9 / 3.8562e-9 | 3.8762e-9 / 3.8582e-9 (within 0.05 %) |
| Transport join | ×0.959 | −3.4 % |
| R_OLD 352 | ×0.9930 | −1.1 % |
| v52 safe B / A vs v51 | +2.1 % / +0.9 % | +1.9 % / +0.8 % |
| V56 synthetic-fit map | ×0.9964 | −0.29 % (B) / −0.21 % (A) |
| Star ×1.25 | ×0.9972 additional | No additional gain |

Every predicted sign matched the grade. For single changes worth more than about 1 %, the predicted size was accurate to within 0.05–0.7 percentage points. For gains under 0.5 % the size was unreliable: the graded gain came in at roughly 0 to 2× the predicted one.

The harness measures residual time but does not enforce the cap. Its in-process mode did not reproduce the V56 memory failure; only `whest run --runner subprocess` did (§3.4).

### 8.3 Scripts and result files

Result files are in `results/summaries/` unless a path is given.

| Section | Scripts → results |
|---|---|
| §2.5 identities | `experiments/probes/r17_curvature_identities.py` → `r17_curvature_identities.json` |
| §3.1 ledger | `experiments/probes/m1_matmul_ledger.py` → `m1_matmul_ledger.json`, `m1_matmul_ledger_callers.json` |
| §4 error map | Teacher scripts in `kaggle/k4/` (pre- and post-ReLU teachers), the error-map probes listed in `docs/build_notes.py`, and `opus12_errmap.py` |
| §4.8 joint oracle | `experiments/variants/r20_joint_oracle.py` → `r20_joint_oracle.json` |
| §4.9 one-step channels | `experiments/probes/r20_onestep_channels.py` → `r20_onestep_channels.json` |
| §4.10 κ₃ error profile and windows | `experiments/probes/k3_error_profile.py` → `chain_k3_error_profile.json`, `teacher_k3_noise_estimate.json` (and the §6.5 (c) slice attribution `k3feed/feed_attribution.json`); κ₃ windows through `r20_joint_oracle.py` (Kaggle wrappers `kaggle/k3loc/esfer_{1,2,3}/run.py`) → `oct10/k3loc_{1,2,3}.json` |
| §5.1 oracles | `experiments/chain/k4_oracle_run2.py` → `k4oracle2.json`, `k4oracle2_joint_*.json`, `k4oracle2_target_*.json` |
| §5.2 memory | `opus12_k4_carry.py`, `q18_k4_horizon.py`, `opus12_q18_noise.py` |
| §5.3 V55 | `experiments/variants/v55_k4c.py`, `experiments/probes/k4_pattern_features.py` → `k4c_fit.json`, `k4_pattern_features*.json`, `teacher_eval_v55_k4c.json`, `kaggle_v55_deep.txt`, `kaggle_v55_sweep1.txt`, `kaggle_v55_sweep2.txt` |
| §5.4 ceilings | `r16_sbqm_ceiling.py`, `r16b_astra/`, `experiments/probes/r17_hermite_family_ceiling.py` → `data/kaggle_out/r17_ceiling/r17_hermite_family_ceiling.json`, `r17_hermite_family_ceiling_B.json`, `data/kaggle_out/r17_ceiling_c/r17_hermite_family_ceiling_C.json` |
| §5.5 HTQB and V56 | `r18_htqb/`, `r18_htqb_combo.py` → `round18_htqb.json`, `r18_htqb_combo.json`; `experiments/variants/v56_star.py` → `k4s_fit.json`, `kaggle_v56_tuning.txt` |
| §5.6 tree test | `r19_tree_ceiling.py`, `r19_tree_rescale.py` → `r19_tree_ceiling.json`, `r19_tree_rescale.json` |
| §5.8 C·T channel | `fable_ceiling_2026-10-09.py` → `fable_ceiling_2026-10-09.json`; oracle affine fits and E2E through `r20_joint_oracle.py` (Kaggle wrapper `kaggle/fable_e2e2/kernel/run.py`) → `fable_affine_fits.json`, `fable_affine_e2e.json`; `fable_component_2026-10-09.py` → `fable_component_2026-10-09.json`; `acds_ct_probe.py`, `acds2_ct_probe.py` → `acds_ct_probe.json`, `acds2_ct_probe.json`, `acds_e2e.json`; layer windows (Kaggle wrappers `kaggle/acds/ctwin_{1,2}/run.py`) → `oct10/ctwin_{1,2}.json`; round 3: `acds2_ct_probe.py --lite 1 --ranks 32,48,64` (wrapper `kaggle/acds/kernel/run.py`) → `acds3_ct_probe.json`, E2E (wrapper `kaggle/acds/e2e3_kernel/run.py`) → `acds_e2e3.json` |
| §6.5 (a) RLL | `experiments/probes/astra_rll_test.py` → `astra_rll_test.json` |
| §6.5 (b) γ-patch | `experiments/variants/v56_fbg.py` (runner sets `kaggle/fbg/set_{1..5}`) → `kaggle_fbg_100mlp.json`; per-MLP outputs `data/kaggle_out/fbg_{1..5}/`, compared with `kaggle_pair_compare.py --ref base_fbg` |
| §3.6 V56 call-site ledger | `experiments/probes/site_ledger.py submissions/v56As_fs/estimator.py <weights.npz>` → `site_ledger_v56As_fs.json` |
| §3.6, §6.6 (f) Wu et al. read | `consultations/wu_paper_deep_read_2026-10-10.md` (cost table, idea-by-idea check against V56, trace-projection kill; no separate result file) |
| §6.6 (a) single pool | `experiments/probes/pool_ceiling.py --layers 11,14,15` → `pool_ceiling.json` |
| §6.6 (b) DEJ-4 | `pool_ceiling.py --ranks 352,448,512 --young k` (Kaggle wrapper `kaggle/dej/kernel/run.py`) → `oct10/dej_young{1..4}.json` |
| §6.6 (c) age/rank grid | `submissions/v56As_fs/estimator.py` with `V21_AGE_OLD` / `V21_R_OLD` (runner sets `kaggle/agegrid/set_{1..5}`, `variants.json`) → `kaggle_agegrid_100mlp.json`; base `base_fbg` in `kaggle_fbg_100mlp.json` |
| §6.6 (d) CBA-13 | `experiments/variants/r20_cba.py` (`CBA_MODE=zero` / `zero_closure`, `CBA_LMIN`, `CBA_LMAX`; wrappers `kaggle/cba/k1`–`k3`) → `oct10/cba_{1,2,3}.json` |
| §6.6 (e) κ₃ level scaling | `r20_cba.py` (`CBA_MODE=scale`, `CBA_D3S`, `CBA_D21S`; wrappers `kaggle/k3scale/k1`–`k3`) → `oct10/k3scale_{1,2,3}.json` |
| §6.6 (g) consultations | `Architecture Analysis/Astra_Cost_First_Architecture_Assessment_2026-10-10.pdf` (slice-state argument), `Architecture Analysis/Astra_Conditional_Budget_Architecture_2026-10-11.md`, `Architecture Analysis/routing_oracle_audit.py` → `Architecture Analysis/archived_routing_audit.json` |
| §6.7 PM364 student | `experiments/student/`: `pm_student.py` (PyTorch student), `pm_flops.py` (flopscope version), `v56_dump.py` + `teacher_dump.py` (V56 labels; Kaggle label kernels `kaggle/student/lab_*`), `train_pm.py` → `student/pm364_s11_history.json`, `student/pm364_s11_theta.npy` |
| §6.5 (c) κ₄→κ₃ feed | Design `consultations/fable_2026-10-10_k3_design.md`; ceiling `experiments/probes/fable_k3feed_ceiling_2026-10-10.py` (CPU wrappers `kaggle/k3feed/cpu_{1..5}/`) → `k3feed/k3feed_mlp0_part{1..5}.json`; E2E `experiments/variants/r20_feed_oracle.py` (wrappers `kaggle/feedor/k1`–`k6`) → `k3feed/teacher_eval_feedor1.json` (hook validity), `k3feed/teacher_eval_feedor2.json` (removal), `oct10/feedorE2E_{3,4,5,6}.json` (injections) |

The kill register is `docs/architecture_research/KILL_AUDIT.md` (27 Sep, 74 entries). The round verdicts are `consultations/round14_verdicts.md` through `round18_verdicts.md` and `consultations/new_architecture_step1.md`. The experiment queue is `docs/experiment_queue/queue.json`.

Unless a path is given, probe scripts are in `experiments/probes/`. All scripts will be released under MIT with the write-up after the competition closes.

### 8.4 Known inconsistencies in our records

- The κ₄-diagonal oracle appears as 1.600e-8 and 1.609e-8 in two runs.
- The κ₄-diagonal oracle ratio is ×0.702 in §4.2 (base 2.290e-8), ×0.686 in §4.8 (base 2.273e-8, every layer) and ×0.700 in §5.8 (same base, layers 7–16 only). The bases and injection windows differ; none of the three is a typo.
- The chain's κ₄ correlation is 0.78–0.83 over all neurons but 0.26–0.34 on decisive neurons. Both figures are correct for their populations.
- The grade of #332395 (last-block λ ×1.25) is not recorded in our files.
- The id of the failed v54A submission (7 Oct) is not recorded in our files.
- The chain's slice error is "6.7 % at L15" in §4.5 (base 2.290e-8) and 6.6 % at L15 in §4.10 (the §5.8 captures). Its d3 error is 3.2 / 3.8 / 4.6 / 4.7 % at L7 / L11 / L15 / L16 in §4.3 and 3.0 / 3.6 / 4.4 / 4.6 % in `chain_k3_error_profile.json`. The two come from different chain runs; both include teacher noise.
- Our working notes priced eig r = 32 at "≈ 18 u". The probe's thin-product count is 13.8 u (`acds3_ct_probe.json`), which excludes G_m and Γ. §5.8 therefore quotes 18–20 u as an estimate, scaled from the 40 u estimate for r = 64.
- The slice-level feed attribution of §6.5 (c) was first quoted for L10–L14 only (+0.96, 14–24 %, +0.45 to +0.58). The saved file `k3feed/feed_attribution.json` covers L9–L15: +0.95 to +0.97, 12–26 %, +0.41 to +0.59.
- The feed-oracle runs report util 0.3108–0.3425, because the hook performs a second slice readout. These values are not comparable with the chain's 0.2476.
- The κ₃ level-scaling ratios of §6.6 (e) were first read from the run logs; the result files were saved afterwards as `results/summaries/oct10/k3scale_{1,2,3}.json`.
- The PM364 engineering checks of §6.7 (closure-port parity 5e-8 / 4e-6 / MSE 1.5e-11, flopscope parity 2.8e-7, 9.7k operations, 93 u extrapolated) were measured in the 11 Oct session and recorded afterwards in `results/summaries/student/pm364_engineering_checks.json`.
- The V56 ledger counts 20,586 operations in the warm call (§3.6), against ~19.9k–20k quoted for the A class in §3.3. The ledger counts each outermost flopscope call as one operation on one network; the two counts are not from the same instrument.
- The ledger's 193.8 u is one network; the harness mean for the same estimator is util 0.1861 (≈ 190.6 u).
- The 10 Oct leaderboard lists our entry at raw 1.97e-8, while the grader reported 1.950e-8 for #334870. We have not established which submission or rounding the board's figure reflects.
- The CBA ratios are quoted to one decimal from the files (×63.7, ×125.6, ×102.6); our working notes rounded them to ×64, ×125 and ×103.

---

## 9. Disclosure of LLM use

**Primary assistant.** The main engineering and research assistant was Claude Code (Claude Opus 5 and 5.5), directed by the participant. It wrote and ran most of the probes, implemented the estimator variants, and maintained the notes. Internal Claude subagents did specific measurement tasks; every number they reported was measured. These subagents were:
- Fable 5 (Rounds 7–8: the all-distinct κ₃ decomposition, the FB scale and the K₃×C term);
- Opus (Round 12: κ₄ as carried memory);
- Sonnet (Round 10: the young-tier count).

**Outside models consulted.** We sent prompts to:
- GPT-5.6 Sol and Terra;
- Astra 6;
- Gemini 3.1 Pro, Grok, DeepSeek and Copilot;
- SciMaster/Bohrium;
- several unsigned answers.

**Origin of shipped ideas.**

| Shipped idea | Origin |
|---|---|
| Spherical-input reparameterisation | GPT Sol R4, pointing to a public repository |
| HTQB star | Round-18 proposal, built on our Round-17 ceilings |
| Parity argument | Fable 5 consultation |

**Verification.** Every outside claim was re-derived or re-measured by us before entering this paper. Several claimed measurements were found to be unsupported, and §6.4 says which. The 10–11 Oct consultations (§6.6 (g), §6.7) ran no network inference. Their one numerical result, the routing ceiling, is an audit of our own archived run files. The PM364 student was implemented, trained and evaluated by us. No leaderboard feedback entered any fitted coefficient. The V55 and V56 maps were fitted only on MLP #0 and the synthetic network against our own Monte-Carlo teachers.

**Licensing and attribution.** The baseline estimator is 504aldo's V29, used under the MIT licence with its copyright header and LICENSE retained. All other changes are ours.

---

## References

1. Wu et al., arXiv:2605.05179 (title to be filled in from the arXiv entry) (the K3 "simple" chain, §4.3, and the reference implementations used in §6.3).
2. 504aldo, *whest-p2-cumulant-k3* (GitHub, MIT licence): the V29 factorised third-cumulant chain with memoryless κ₄ regeneration, D21 feedback, an age-gated shared basis and pooled Strassen–Winograd. Forum post #18218. Finding F105 on the r = 1 κ₄ core.
3. WhestBench Phase 2 forum threads:
   - #18218 (504aldo chain);
   - #18216 (yujigon, zero-constant closure with CP κ₃);
   - #18214 (venture_agent, exact bivariate ReLU covariance);
   - #18200 (baghyeonsig, blended MC);
   - #18219 (nkosi_ndwandwe, MC economics);
   - #18199 (keenanpepper, no heavy tail across MLPs);
   - #18175 (Puffi, Phase 1 SSC stack).
4. barnobarno666, public Phase-2 repository, Phase 8 (spherical-input reparameterisation).
5. Organiser ruling of 11 Sep 2026 that Strassen–Winograd is legal at any recursion depth (competition forum).
