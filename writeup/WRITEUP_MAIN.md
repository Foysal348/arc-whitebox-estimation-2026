# Where the Error of the Factorised Cumulant Chain Lives, and What It Costs

**WhestBench Phase 2, Algorithm track. Team Hack2Publish: mdhamidhosen, esfer_sami, foyshal_sk.**

**Submission #337457** (team best): adjusted score **2.7858e-9**, raw final-layer MSE **1.8592e-8**, util 0.150. It runs our κ₄-pattern corrector (§5) on LordProcciuto's public e30_a chain (MIT; 3.1163e-9 as published), with a mean-correction table refitted on our trajectory with new non-local features and other measured additions (§6.5).
**Own-lineage best #337049** (e299safe, 3.2529e-9): our chain **#334870** (v56As_fs, 3.6521e-9), the chain this write-up analyses, plus md.hamidhosen's calibration and cost layer (§6.4) and the safe cost rewrites of §6.5.

*21 October 2026. This is the main write-up. The full technical report is attached as the **Appendix (full technical report)**, and "App. §x" refers to its sections. Each number and claim is labelled by its source:*
- *graded: an official grade, with its submission id;*
- *measured: a run of our own scripts, with the result file named in App. §8.3;*
- *derived: algebra or arithmetic, with the derivation in the Appendix;*
- *estimate or untested: not yet measured;*
- *conjecture: an interpretation we have not tested, marked as such.*

---

## 1. Summary

We started from 504aldo's public, MIT-licensed factorised third-cumulant chain ("V29"), which graded 5.418e-9 when submitted verbatim. Our changes take it to **3.6521e-9 (−32.6 %)**: raw MSE falls by 8.6 % (2.133e-8 → 1.950e-8) and util by 26 % (0.254 → 0.187) (derived). A final calibration and cost layer by md.hamidhosen on this chain (E282–E299, §6.4) takes it to **3.3946e-9 (#336377, −7.0 %)**; that layer is fitted and engineering, not new understanding. Moved onto a cheaper public chain (e30_a) with a refitted correction table, our corrector reaches **2.7858e-9 (#337457)** (§6.5).

**Mechanistic contributions.** These are what we consider our contribution to the understanding of white-box estimation:
1. **A spherical-input identity** (§3.1; derived, injection check measured, graded −1.1 % raw at zero cost).
2. **A parity theorem** (§3.2; derived, verified numerically): old κ₃ is odd in the later weights, so every closure built on even statistics loses it.
3. **An error map with channel oracles** (§4; measured on 6·10⁷-sample Monte-Carlo teachers). The closures are accurate when given true inputs; the error is input error made at each ReLU and accumulated through depth; the teacher's κ₃ and κ₄ inputs together would take the raw MSE to ×0.272 on MLP #0; the κ₄ gain lives in the **per-neuron pattern** of the κ₄ diagonal, not its level.
4. **A Hermite cubic-star feature, V56** (§5.2; identity derived and verified numerically to ≤ 8.1e-16; graded −4.4 %), motivated by a measured ceiling ladder: second-order births cap the κ₄ pattern at 0.45–0.55, cubic births reach 0.80–0.85.
5. **An analysis of the κ₃-dressed C·T channel** (§4.5; oracle and component level, not graded): with the star it reaches the cubic-birth ceiling, and the chain's own legs reproduce it, but not yet at an affordable cost.
6. **A negative-result atlas** (§7): 24 of the more than 60 architecture and representation ideas we killed, each with its gate and measured number, and three working conclusions that we label as empirical, not proven.

**Engineering and fitted components.** We state these plainly. Most of the gain from 5.418e-9 to 4.0784e-9 is **cost engineering**: Strassen placement, a transport join, dead-neuron compaction, D21 trimming and an operation budget for the grader's hidden limits (§6). This is performance optimisation, not new understanding. The κ₄ gain from 4.0784e-9 to 3.6521e-9 (−10.5 %) passes through **V55, a linear map fitted offline** to our own Monte-Carlo teacher (§5.1); the V56 star enters as one more feature of that fitted map. FB 1.25, β 1.2, the s3sq coefficient ×1.25, AGE_OLD 5 and R_OLD 352 are **tuned constants**. The last −7.0 % (§6.4) is a Gauss-Newton calibration fitted on public MLPs plus exact cost rewrites.

**Leaderboard context.** On the 10 Oct leaderboard the six highest entries are ahead of us on both factors (raw MSE 1.14–1.65e-8 at util 0.106–0.144). Our raw MSE is comparable to entries ranked 11–40. Their methods are not public, and we make no claim about what they are.

---

## 2. Starting point: the reference method (not claimed)

### 2.1 Task and score

For a bias-free ReLU MLP of width n = 1024 and depth 16, with He-initialised weights and input x ~ N(0, I), the estimator returns E[h⁽ˡ⁾] for every layer. The score is

  score = MSE_final × max(0.1, FLOPs / 2⁴¹),

with the final-layer MSE measured against a 10⁹-sample Monte-Carlo reference. FLOPs are metered by flopscope. We use **1 u = 2³¹ FLOPs**, about one dense 1024³ product; the budget is then 1024 u, "util" is FLOPs/budget, and the 0.1 floor is 102.4 u. The grader caps each MLP at 120 s of wall time, 0.4 s of residual Python time outside metered operations, and 8 GB of address space. A capped MLP scores about 0.35, so a single cap in a hundred MLPs ruins a submission.

### 2.2 504aldo's V29 (MIT)

Our estimator modifies **504aldo's public V29** (`whest-p2-cumulant-k3`, forum post #18218), which implements the factorised K = 3 "simple" cumulant propagation of Wu et al. (arXiv:2605.05179, §4.3). Per layer it carries the mean, the covariance C, and the third cumulant κ₃ as a **sum of sources**, one per ReLU. Each source is stored as a propagator leg P_s, a birth leg A_s = P_s a_b, and thin n×16 feedback legs. The next ReLU needs the slices D3_i = κ_iii and D21_ic = κ_iic, read per source from contractions of Hadamard products of the legs. Sources older than four layers live in a shared basis of rank 384 (the oldest in a nested rank-224 sub-basis). κ₄ is not carried: it is regenerated each layer as G_off = λ_l C_off, with a per-MLP adaptive λ. Products use pooled Strassen–Winograd.

The author prices V29 at 260 u and reports a tolerance law: extra final MSE ≈ 4.2e-6·ε², where ε is the relative RMS error of D21, so D21 must be right to about 2 %.

**Graded baseline.** V29, submitted verbatim, graded **5.418e-9** (raw 2.133e-8, util 0.254; **#331681**), matching the author's own submission.

**Licence and scope of our claims.** Every file we ship keeps 504aldo's MIT copyright notice and LICENSE, and carries a lineage header that names each change and its date. **Only the changes in §3–§6 are claimed as ours.** The sum-of-sources representation, the shared basis, the memoryless κ₄ law, the feedback legs and the pooled Strassen engine are the reference method.

---

## 3. Exact identities and derivations

### 3.1 Spherical-input identity (derived; V31)

Write the Gaussian input as X = (R/√n)·Y, with Y uniform on the sphere of radius √n and R ~ χ_n independent of Y. A bias-free ReLU network is positively homogeneous of degree 1, so

  E[h(X)] = a₁ E[h(Y)],  a₁ = √(2/n) Γ((n+1)/2)/Γ(n/2) = 0.99975589.

The sphere input has C = I and κ₃ = 0, and its κ₄ is an exact harmonic core, g = −6/(n+2) in the chain's representation. The estimator injects that core at layer 0, runs the existing κ₄ machinery unchanged, and rescales the output by a₁.
- **Measured:** the injection reproduces the predicted scale shift to 0.3 %.
- **Graded:** −1.1 % raw at zero cost (#331787).

The idea came from a public Phase-2 repository, surfaced by an outside-AI consultation (App. §2.1). The derivation for this chain and its integration are ours.

### 3.2 Parity theorem: no mean-field part of the deep κ₃ (derived and verified; App. §2.2)

Take a source born at layer b. At any later layer l its legs are P = Wᵀ⁽ˡ⁾ D P′ and A = P a_b, where P′, a_b and the birth coefficients do not depend on W⁽ˡ⁾. Every quantity linear in the source's κ₃ (the slice products and D3) is therefore **odd** in W⁽ˡ⁾, while C⁽ˡ⁾, the leg Grams and the per-neuron variances are **even**. Because the weight ensemble is sign-symmetric,

  E[ H_s⁽ˡ⁾ | C⁽ˡ⁾, Gram(A_s, P_s), all state before l ] = 0, exactly.

Consequences:
- Any closure that replaces old sources by a function of even statistics (mean-field closures, scale mixtures, polynomials in C) discards their entire contribution. This is consistent with the public log's failures of such closures (9.1e-7 and 2.3e-6) and with our learned residual correctors, which recover only 0.3–0.7 % of the residual (§7).
- Replacing a source by its conditional expectation given one fresh weight row has a relative-error floor of √(6/(r+8)) in the retained dimension r: 12 % at r = 384, far above the 2 % tolerance.
- κ₄ terms built from two κ₃ factors are even, and C⊗C is the only even invariant available. We read this as the reason the regenerated law G_off = λ_l C_off works (conjecture beyond the derivation; not tested separately).

*Status:* derived, then verified (11 Oct; `experiments/probes/parity_check.py`). On the frozen V56 chain (width 128), flipping W⁽ˡ⁾ leaves every earlier layer and C⁽ˡ⁾, var⁽ˡ⁾ bit-identical and flips μ, d3 and D21 exactly (difference 0.0; tested at l = 5, 10). The floor's algebra holds by Monte Carlo, and the bound ‖τ‖² ≤ (r+2)‖T‖²/3 is the exact operator norm (eigen-computation, r = 2–8); a written proof is in App. §2.2. The floor is a statement about the idealised fresh-row model. The mean μ⁽ˡ⁾ is also odd, so closures that condition on μ are not covered. The argument originated in a Claude subagent consultation.

### 3.3 A half-carried feedback term (derived and measured; shipped as tuned FB 1.25)

Expanding the post-ReLU κ₃ to first order in C shows that the newborn source carries exactly **half** of two all-distinct K×C families (derived, App. §2.4). Regression on the slice residual gives coefficients of −0.55 to −0.9, against −1/2 predicted: consistent in sign and size, not exact.

Restoring the missing families in full makes the final mean worse: +3.4 % / +3.1 % on the teacher pair, and ×1.024 on 100 MLPs at FB 2.0. A quarter restoration (FB 1.25) with the adaptive-λ exponent β = 1.2 scored ×0.9946 ± 0.0007 on the 100-MLP harness and graded 5.1284e-9 (#332110), at zero FLOPs. The shipped scale is therefore a **tuned constant**, not the derived one. Why the full restoration hurts is open; one explanation is ruled out (App. §6.5 (b)): the quadratic feedback cross term that also grows with FB is inert.

---

## 4. The error map: where the chain's error lives

All results in this section are measured, using our Monte-Carlo teachers with 6·10⁷ samples on public MLP #0 and a synthetic 1024×16 He network. They record the per-layer pre- and post-ReLU means, covariances, κ₃ slices and κ₄ diagonal. An **oracle** replaces one quantity inside the chain by the teacher's value and reruns the chain end to end. Two networks are a small sample; we state each result for the network it was measured on.

### 4.1 The closures are accurate; the error is accumulated input error

- **Mean closure.** Given the teacher's true marginals (μ, var, κ₃ᵢᵢᵢ, κ₄ᵢᵢᵢᵢ), the Edgeworth mean closure matches the teacher's ReLU mean to 5e-11–1.5e-10 at every layer.
- **Covariance closure.** Teacher-forced at one layer, the bivariate covariance closure falls to the teacher's noise.
- **Accumulation.** The chain's per-layer MSE grows smoothly, from 0.7e-9 at L1 to 2.29e-8 at L16.
- **Inherited share.** Forcing the true mean at every layer removes about 80 % of the final error.

We read these as: the error is **fresh input error** made at each ReLU and carried forward through depth.

### 4.2 Channel and joint oracles

The base is the plain chain at util 0.2476 (App. §4.8 gives the full table).

| Teacher input injected (every layer) | MLP #0 | Synthetic net |
|---|---|---|
| None (plain chain) | 2.273e-8 (×1) | 1.920e-8 (×1) |
| κ₃ slice + d3 | ×0.607 | ×0.716 |
| κ₄ diagonal | ×0.686 | ×0.885 |
| **κ₃ slice + d3 + κ₄ diagonal** | **×0.272** | **×0.569** |
| κ₃ slice + d3 + κ₄ *pattern* at the chain's level | ×0.294 | ×0.606 |

- **κ₃ is the larger single lever** on both networks.
- **The two inputs compound more than multiplicatively.** The product of the single-input ratios would be 0.416 and 0.634; jointly they give 0.272 and 0.569.
- **The remainder is consistent with the variance channel.** In the separate channel oracle of App. §4.2 (base 2.290e-8), the true variance, κ₃ diagonal and κ₄ diagonal together take MLP #0 to ×0.057, the truth's own noise floor. The two oracles inject different κ₃ quantities, so this is a consistency check, not a decomposition.

### 4.3 κ₃: where the error is real, and where the gain lives

We compared the chain's κ₃ with the teacher at every layer, and subtracted in quadrature a Gaussian estimate of the teacher's own sampling noise (derived; this assumes the two are independent):

| Layer | D21 off-diagonal error | Teacher noise (est.) | Net D21 error | d3 norm ratio, chain/teacher |
|---|---|---|---|---|
| 2 | 4.4 % | 5.3 % | ≈ 0 | 1.002 |
| 4 | 4.2 % | 2.5 % | 3.4 % | 0.998 |
| 8 | 4.9 % | 1.2 % | 4.8 % | 0.996 |
| 12 | 5.9 % | 0.7 % | 5.9 % | 0.993 |
| 16 | 6.9 % | 0.5 % | 6.9 % | 0.986 |

Under that noise estimate, the early "error" is mostly teacher noise, while the late slice error is real and grows with depth; the d3 level shrinks by 1.4 % at L16. Of the slice error, 91–96 % sits in the **all-distinct** post-ReLU entries κ₃(hᵢ, hⱼ, h_k) (App. §4.5).

**Localisation.** Injecting the teacher's κ₃ only in a window of layers (MLP #0, base 2.2732e-8):

| Window | Slice + d3 | Slice only | d3 only |
|---|---|---|---|
| L1–4 | ×1.004 | — | — |
| L5–8 | ×0.942 | — | — |
| L9–12 | ×0.857 | — | — |
| L13–16 | ×0.806 | — | — |
| L1–8 | — | ×0.937 | ×1.012 |
| L9–16 | — | ×0.834 | ×0.853 |

The gain lives at **L9–16**, where the slice and d3 matter about equally. Injecting the teacher's κ₃ into the feedback legs alone is worth ×1.001.

### 4.4 κ₄: the pattern, not the level

Injecting the teacher's κ₄ diagonal into the mean closure takes the raw MSE from 2.290e-8 to 1.609e-8. How that gain splits:
- **The teacher's pattern at the chain's own level** gives 1.551e-8, which is **108 %** of the gain.
- **The chain's pattern at the teacher's level** is *worse than no oracle* (+21.5 %).
- **In our oracles, κ₄ helps only through the mean closure's diagonal input.** True κ₄ everywhere except the mean closure gives ≈ ×1; the off-diagonal slices injected alone give ×1.03–1.22.
- **The chain's pattern is weak where it matters.** Over all neurons the chain's κ₄ diagonal correlates 0.78–0.83 with the truth; on the decisive neurons (|μ/σ| < 1.5) that drive the mean closure, only 0.26–0.34.

**A ceiling ladder for κ₄ carriers.** We measured the ceiling of whole *families* of birth laws by restarting a Gaussian h layers back and expanding each ReLU to Hermite order K (App. §5.4, N = 4·10⁶):
- **Quadratic births are capped:** births that stop at second order reach pattern correlation 0.45–0.55. Every second-order carrier we and outside models proposed stalled at or below this band (0.03–0.56, §7), which is consistent with, though does not prove, the cap being the cause.
- **Cubic births suffice:** cubic births over six layers reach 0.80–0.85, and K = 4 adds ≤ 0.02.
- **Ages are complementary:** ages 1–6 are needed jointly. Ages 1–4 give 0.68–0.74, and ages 4–6 alone give only 0.37–0.41.

This ladder is why V56 uses a *cubic* feature.

### 4.5 The κ₃-dressed C·T channel (measured at oracle and component level; not graded)

The V56 star (§5.2) is the *Gaussian* cubic birth. A κ₃-dressed C·T readout, ct_i = 24 Σ_a g_a Ȳ_ai X_ai Q_ai, reads the chain's κ₃ through the quadratic and cubic Hermite vertices (design from a 9 Oct consultation; every number below is ours, App. §5.8):
- **Ceiling.** Sample-exact star + C·T reaches 0.82–0.85, matching the cubic-birth ceiling of §4.4. C·T alone sits at 0.42–0.44, where an earlier hub-pair carrier was killed; it is complementary to the star.
- **The chain's own legs reproduce it** at fidelity 0.993–0.996 on decisive neurons (component test).
- **End to end under an oracle map** (fitted to the teacher on the evaluation network, FLOPs not metered): ×0.773 with the sample-exact feature, ×0.819 with the chain's exact C·T on the available source subset.
- **Cost.** An exact age-collapse identity, Γ_a = G_a + Ystep_a Γ_{a+1}, plus grouping by dense anchor, reduces it to two dense products per source and target (derived; matches the per-age route to 2.6–3.2e-7). At r = 64 a sketch keeps ×0.865 at about 40 u (estimate), roughly score-neutral. A rank-32 sketch keeps ×0.874 at an estimated 18–20 u. It is untested on V56 and is our main open lead.

---

## 5. The κ₄-pattern programme (graded)

### 5.1 V55: a fitted, zero-FLOP κ₄-pattern map (fitted component)

Given §4.4, V55 corrects the pattern, not the level. At every layer ≥ 6 (≥ 4 from #334762 on), a **fitted linear map** turns seven per-neuron quantities the chain already holds into a predicted pattern of κ₄/v²: s3sq = (D3/v^1.5)²; the row and column energies of the standardised D21; the correlation-row energy of C_pre; |μ/σ|; the chain's own κ₄/v²; and a constant. The pattern is rescaled to the chain's layer-mean level and fed **only** to the mean closure. The map is fitted offline, on decisive neurons, against our own teacher for MLP #0; the synthetic net is held out. It costs 0 u and +310 operations.

- **Pattern correlation:** 0.41–0.55 on MLP #0 against 0.24–0.41 for the chain, and 0.41–0.52 held out on the synthetic net.
- **100-MLP harness:** ×0.9455 [90 % CI 0.9410, 0.9498], with 97/100 MLPs improved.
- **Graded:** −5.4 % in both classes. The harness projected 3.8764e-9 / 3.8562e-9; the grades were 3.8762e-9 (#334588) and 3.8582e-9 (#334591).

V55 is a regression on chain features, not a mechanism; its features saturate near pattern correlation 0.5 (App. §5.3).

### 5.2 V56: the Gaussian cubic-star feature

The cubic Hermite coefficient of the ReLU, c = −αφ(α)/6 in standardised units, together with the Wick identity κ(u_a³, u_b, u_c, u_d) = 6ρ_ab ρ_ac ρ_ad, injects a **star**-shaped fourth cumulant downstream of every hub (derived; verified numerically to ≤ 8.1e-16, App. §2.5). V56 reads this star from the chain's **own** young dense source legs (ages ≤ 6), with the per-source coefficient recorded at birth (App. §5.8 notation):

  star_i = 24 Σ_a c_a Ȳ_ai X_ai³.

The star enters as an eighth feature of the V55 map, at zero added FLOPs: util 0.1857, against 0.1855 for the V55 reference. Alone its ceiling is 0.47–0.52, but it is complementary to the V55 features: with teacher μ and C, leave-one-layer-out fits rise from 0.48–0.56 to 0.65–0.74. The star idea came from an outside "Hermite Tree Quartic Bank" proposal that built on our own Hermite-family ceilings (App. §5.4–§5.5).

**Graded.** −4.4 % (#334805, B) and 3.6597e-9 (#334811, A). Refitting the map on the synthetic net gave a further −0.29 % / −0.21 %, reaching **3.6723e-9 (#334869) and 3.6521e-9 (#334870)**.

The κ₄ programme as a whole took the score from 4.0784e-9 to 3.6521e-9 (−10.5 %) at unchanged cost. Its tuning is closed: star ×0.75 / ×1.5 cost ×1.011, and star ages 4 cost ×1.0045.

---

## 6. Engineering components: cost under the meter and the grader's limits

These sections are performance engineering. We report them because they carry most of the cost reduction and because the grader's hidden limits shaped every later design, not as contributions to understanding.

### 6.1 Graded trajectory

| Step | Our change | Score (id) | Raw MSE | Util |
|---|---|---|---|---|
| V29 verbatim (reference) | — | 5.418e-9 (#331681) | 2.133e-8 | 0.254 |
| Strassen recursion floor 32 → 8 | Two more Strassen levels on the basis families | 5.220e-9 (#331686) | 2.132e-8 | 0.244 |
| + first call at full depth | Strassen depth on each worker's first call | 5.195e-9 (#331732) | 2.132e-8 | 0.2436 |
| + V31 spherical-input identity | §3.1, 0 FLOPs | 5.137e-9 (#331787) | 2.108e-8 | 0.2437 |
| + FB 1.25, β 1.2 | §3.3 (tuned), 0 FLOPs | 5.1284e-9 (#332110) | 2.1046e-8 | — |
| Transport join | Shared basis rebuilt at a join by transport, with no range finder | 4.9519e-9 (#332257) | 2.1788e-8 | 0.2273 |
| + AGE_OLD 5, R_FB 4 | One more dense young age; feedback rank 16 → 4 | 4.8066e-9 (#332324) | 2.1371e-8 | 0.2249 |
| v48 compaction | Dead neurons (Φ < 1e-2) dropped from the next transport | 4.6294e-9 (#332391) | 2.1462e-8 | — |
| v49 basis-side Strassen + aliased Grams | §6.2 | 4.4157e-9 (#332401) | 2.1463e-8 | — |
| v50 D21 live columns | D21 columns of dead neurons never computed | 4.2512e-9 (#332404) | 2.1537e-8 | — |
| v51 D21 rows | D21 rows of very dead neurons never computed | 4.1081e-9 (#332418) / 4.0902e-9 (#332420) | 2.1545e-8 / 2.1636e-8 | 0.191 / 0.189 |
| v52 operation budget, B / A | §6.3: safe against the residual cap | 4.1696e-9 (#332566) / 4.1228e-9 (#332567) | — | — |
| + R_OLD 352, Tq from QR, R_FB 2 | Exact or near-exact trims | B 4.0995e-9 (#332626); A, R_OLD 352 only, 4.0784e-9 (#332621) | 2.197e-8 (B) | — |
| V55 κ₄-pattern map, B / A | §5.1 (fitted) | 3.8762e-9 (#334588) / 3.8582e-9 (#334591) | — | — |
| + s3sq coefficient ×1.25, B / A | Map coefficient (tuned) | 3.8539e-9 (#334762) / 3.8331e-9 (#334763) | — | — |
| V56 cubic star, B / A | §5.2 | 3.6829e-9 (#334805) / 3.6597e-9 (#334811) | 1.9706e-8 / 1.9539e-8 | — |
| + map fitted on the synthetic net, B | | **3.6723e-9 (#334869)** | — | — |
| + map fitted on the synthetic net, A | | 3.6521e-9 (#334870) | 1.950e-8 | 0.187 |
| E282: 24-group calibration (md.hamidhosen) | §6.4 (fitted) | 3.5890e-9 (#335603) | 1.9154e-8 | 0.187 |
| E285–E291: rider, κ₄-corrector refits | §6.4 (fitted) | 3.5085e-9 (#335729) | 1.8722e-8 | 0.187 |
| E294–E299: exact cost rewrites, thresholds | §6.4 | 3.3946e-9 (#336377) | 1.8824e-8 | 0.180 |
| + e30_a's exact cost savings, (3,1) closure at 0.25 | §6.5 | 3.3325e-9 (#336757) | 1.8742e-8 | 0.178 |
| + fixed old basis, C_pre compaction (own lineage) | §6.5 | 3.2529e-9 (#337049) | 1.9107e-8 | 0.170 |
| Our corrector on public e30_a (scale 0.4), (3,1) closure | §6.5 | 3.0302e-9 (#337175) | 1.9953e-8 | 0.152 |
| + D21 ×0.98, closure mix 0.4, corrector split | §6.5 | 2.9956e-9 (#337258) | 1.9723e-8 | 0.152 |
| + e30_a's table refitted on our trajectory (N2) | §6.5 (fitted) | 2.8475e-9 (#337301) | 1.8747e-8 | 0.152 |
| + near-dead column cut (τ_B 2.4) | §6.5 | 2.8256e-9 (#337336) | 1.8869e-8 | 0.150 |
| + table features: star², D3·star, corrector-local (N3, 122) | §6.5 (fitted) | 2.7975e-9 (#337368) | 1.8681e-8 | 0.150 |
| + D21/C row energies, carried delta and star (N4a, 164) | §6.5 (fitted) | **2.7858e-9 (#337457)** | **1.8592e-8** | **0.150** |

Util is the grader's figure or adjusted ÷ raw. "—" means the value is not listed here. The B class keeps ≤ ~18k warm-call operations for a safety margin under the residual-time cap; the A class runs at about 20k.

### 6.2 Cost levers

- **Transport join:** the shared basis is transported through each layer instead of rebuilt by a range finder: −3.4 % (#332257).
- **AGE_OLD 5, feedback rank 4:** −2.9 % (#332324); nearby age/rank settings score ×1.026–1.104 on 100 MLPs.
- **Dead-neuron compaction (input side only):** neurons with Φ < 1e-2 leave the next transport's inner dimension, and their D21 columns and rows are skipped: −3.7 %, −3.7 %, −3.4 % (#332391, #332404, #332418). Compacting the output side failed in every form (×1.6 to ×10 on the teacher pair).
- **Strassen placement and aliased Grams:** recursion floor 32 → 8 (−3.7 %, #331686); basis-side Strassen and joiner Grams written as YYᵀ (billed at half): −4.6 % (#332401). Level 6 exceeded 8 GB.

Over the cost programme the trims removed 22 % of the metered FLOPs (util 0.2437 → 0.189) for +2.8 % raw.

### 6.3 The residual-time cap and the operation budget

Two variants that ran cleanly locally graded catastrophically (#332402 at 0.0434, #332425 at 0.0204: about 12 % and 6 % of MLPs capped). The residual Python time tracks the number of metered operations (0.053–0.059 ms per op on Kaggle for every variant); the failing variants issued 21.7–22.3k operations. Our fix is an **operation budget**: Strassen only where it saves at least about 2 u per 1,000 operations. Class **B** (nomination-safe) keeps ≤ ~18k operations, class **A** about 20k; neither has had a capped MLP since (#332566, #332567). A second hidden limit (an address-space cap) failed four 3-D forms of the V56 star (#334786–#334799) that passed in-process; `whest validate` catches neither limit, so our pre-submission check is an operation count plus a subprocess run.

### 6.4 Final calibration and cost layer (E282–E299, md.hamidhosen; fitted and engineering)

From 12 Oct md.hamidhosen built on #334870 (numbers from his ledger, re-checked against the grader):
- **Calibration (fitted).** A Gauss-Newton fit of 24 per-layer calibration groups (scales of the closure inputs and λ) on public mini MLPs 0–47 gave −1.66 % ± 0.20 % on unseen MLPs 48–99 (E282, graded 3.5890e-9); two further steps and a cubic final-layer ridge rider trained on 900 full-split MLPs followed. Four end-to-end refits of the V56 κ₄ corrector (per-block level, map coefficients) then gave −0.77 / −0.48 / −0.33 / −0.37 % unseen (E291, 3.5085e-9). Every lever converged: further steps of the groups, the corrector and the rider, and α-dependent scales, gave ≤ 0.08 % unseen.
- **Exact cost rewrites (engineering).** Grouped PK2 einsums (util −0.75 %), Strassen floor 4 and Strassen on the nested projections (−0.3 %, −0.48 %), and the Strassen-Winograd kernel (−0.77 % FLOPs) leave predictions unchanged up to rounding. With the D21 and dead-neuron thresholds raised (approximations; −0.73 % graded, E296 → E298) they give **3.3946e-9 (#336377, E299)**.

### 6.5 Final days: our corrector on a public chain, and a refitted table (17–20 Oct; measured and graded)

After the 17 Oct extension (grader on flopscope 0.12.3, which bills our chains identically on 100 Kaggle MLPs), we reviewed the new public write-ups and code and measured every candidate on the 100-MLP harness before grading (App. §3.8).
- **Exact cost savings from e30_a in our chain:** Gram-core skipping, R-factor rotation, folded constants: −1.4 % FLOPs, predictions unchanged up to rounding (#336711, #336719). kai_leal_gonzalez's two-term (3,1) closure (MIT, forum write-up) hurt at full strength (#336697, +0.7 %) but helped at mix 0.25 (#336757, −0.4 %). e30_a's fixed old basis and C_pre compaction gave our own-lineage best (#337049, 3.2529e-9).
- **Our corrector on LordProcciuto's public e30_a chain** (MIT; 158.8 u vs our 184.1 u per MLP, call-site ledger): at full strength it hurt (×1.085, MLP #0); at scale 0.3–0.4 it gave ×0.975–0.977 on 100 MLPs (94/100 at 0.3), graded ×0.979. The gain comes entirely from the V56 star; the local features duplicate e30_a's 87-feature mean-correction table.
- **Refitting e30_a's table on our trajectory (fitted).** The table was fitted on e30_a's own uncorrected trajectory. Refitted by ridge regression (grouped 5-fold CV) on teacher-forced dumps of public mini nets 0–69, with star modulators added (101 features), it gave ×0.930 on held-out nets 70–99 (30/30) and graded −4.9 % (#337301). A table only helps on the trajectory it was fitted on. New non-local features paid further (N3: ×0.990 held-out, graded −1.0 %, #337368; N4a: ×0.9966, graded −0.42 %, #337457); fitting on 341 instead of 70 nets, an on-policy refit and the (3,1) orientation fix of forum topic 18246 did not (×0.9997, ×0.997–1.005, ×1.0005).
- **Measured and rejected:** V31 on e30_a (×0.985 on MLP #0, graded +3.9 %, #337221); e30_a's row compaction in our chain (×1.063); refitting the table after cheaper settings (rank 352: ×1.04 even with refit).
- **Rules note:** the organisers allow building on public MIT code with attribution if the team adds substantial work of its own (forum topic 18240). Our own contribution in #337336 is the corrector (V55/V56), the closure settings and the refitted table; the base chain is LordProcciuto's.

---

## 7. Negative-result atlas

"Gate" is the pre-registered pass criterion. Kill types: C, a ceiling or oracle (an upper bound); P, a teacher-forced component test; E, an end-to-end prototype; K, a knob on the 100-MLP harness. Ratios are final-MSE or score ratios against the chain. Details are in App. §6.

| Idea | Gate | Measured | Kill |
|---|---|---|---|
| Compensating closure with no old tier (8 models proposed variants) | ≤ chain MSE | 7.43e-7, 32× the chain; every knob 6.8–9.9e-7 | C |
| Mean-field / even-statistic closures of old κ₃ | — | Zero by the parity theorem: old κ₃ is odd in later weights (§3.2) | derived, verified |
| Low-rank Hadamard products of source legs | low rank | A∘P needs 681–778 of 1024 directions for 99 % energy | C |
| Query surrogates: trace, diagonal Hermite, TensorSketch | D21 ≤ 2 % | 98.2 %, 95.4 %, 790 % at L14 | C/P |
| Source-free dressed hypertree (DHMP) | next D21 ≤ 2.5 % | 36–40 % | P |
| Hermite-tree κ₃ replacing the chain | closure ≤ chain | 10–20× worse | P |
| Every second-order κ₄ carrier (8 designs) | pattern corr ≥ 0.70 | 0.03–0.56 | C |
| Learned residual correctors (ridge, GBM; 102,400 neurons) | — | 0.3–0.7 % of the residual recovered | E |
| Monte-Carlo control variates | — | Tangent-surrogate corr 0.15–0.29 at L16; variance reduction 1.0–2.2× | P |
| C·T channel via age-collapsed sketches (ACDS), r = 64 | score gain at ≤ ~30 u | Oracle E2E ×0.865 at ≈ 40 u (estimate): score about neutral. r = 32 (×0.874, est. 18–20 u) is open | E (parked) |
| Missing cubic-response C×κ₃ term (RLL) | ρ ≥ 0.1 with the residual at L ≥ 9 | ρ ≈ 0; adding half the term worsens the slice residual by 15–42 % | P |
| γ-patch: FB 2.0 with the quadratic cross term held | ratio < 1 | ×1.0243, the same as unpatched FB 2.0 (×1.0244) | K |
| κ₄→κ₃ feed from the exact κ₄ (N = 10⁶) | exact ≤ 0.80 pass, > 0.92 kill | ×1.115 | C/E |
| κ₃ level rescaling, d3 or D21 ×0.98–1.02 | < 1 on both nets | ×0.984 to ×1.074; the one gain does not transfer | K |
| Single shared pool for the whole κ₃ | D21 ≤ 2 % at some r ≤ 512 | 3.4–6.3 % at r = 512; 0.2–1 % only at r = 768 | C |
| Deferred exact join (youngest source exact) | D21 ≤ 2 % | 2.5–8 % at r = 352–512 | C |
| Age/rank re-split of V56 (5 settings) | score ratio < 1 | ×1.026 to ×1.104 | K |
| Checkpointed birth archive (κ₃ skipped in a window, sources kept) | ≈ ×1 | ×14.8 (L1–4) to ×125.6 (L1–12); no recovery after the window | E |
| Trace projection of old κ₃ (Wu et al.) | D21 ≤ 2 % | 26–42 % | C |
| Cost-first κ₃ representation at 60 % of the cost | 2 % channel accuracy | None found; slice-only states are not exact to first order (derived) | derived |
| Network-conditioned routing over 86 archived variants | ≥ 3 % gain | Oracle ceiling 0.80 % before overhead | C |
| Learned pair-memory student PM364 (source-free, 364 parameters) | MSE to V56 ≤ 1e-9; D21, d3 ≤ 2 % | MSE −0.9 % after training; D21 58–70 %, d3 25–39 % | E |
| Exact old tier (no confinement) | score gain | −3.9 % MSE for +19 % cost | E |
| Strassen level 6 | ≤ 8 GB | 8.13 GB | cap |

### 7.1 Working conclusions (empirical; not proven)

**1. Within the sum-of-sources family, cost and accuracy traded roughly one for one in every knob we swept.** This is an empirical regularity over our sweeps, not a theorem. On the 100-MLP harness the shipped constants sit at the score optimum of the settings tested: β 1.2 (×0.9993), λ ×0.9 (×1.0026), the FB × β grid (within +0.12 %), R_OLD 448 (×1.0113), AGE_OLD 6 on V56 (×1.014), and the five-point age/rank grid on V56, where the best ceiling point gains 0.5 % raw for 3.2 % more util and saving 4.3 % util costs 10.3 % raw. The exceptions in our records are levers that are **exact or nearly exact** (transport join, compaction, Strassen placement, aliased Grams); these are where our cost gains came from. A 27 Sep audit of 74 earlier kills revived 8, all cost or knob levers killed on the wrong criterion; no architecture kill was revived.

**2. Local corrections of one channel made the final mean worse** in every case we tested: full FB restoration (+3.4 % / +3.1 %); the γ-patch (×1.0243); the RLL term (slice residual 15–42 % worse); scaling the chain's κ₄→κ₃ feed toward the larger exact feed (−5 % costs 0.45 %, +10 % costs 9.4 %); injecting the sampled exact feed (×1.115); raising d3 toward the teacher's level (×1.032 / ×1.015); single κ₄ slices (×1.03–1.22). Only **joint** input oracles give large gains (×0.272 against ×0.607 and ×0.686 alone). *Measured at one step (App. §7.5):* the chain's fresh closure-input errors cancel. Their norms sum to 1.9–2.6× their joint error, chiefly var against d3 (cosine down to −0.83). Correcting var alone (L2–L9) or d3 alone (L4–L10) makes the step worse by up to ×1.31 and ×1.13, while var and d3 together always help (×0.56–0.79). How this carries through depth to the final mean remains a conjecture, and for the feed, Monte-Carlo noise is a competing explanation. The practical rule we adopted: local tests are necessary gates, not sufficient ones; candidates are promoted only on the end-to-end mean.

**3. Every source-free or pooled κ₃ state we built failed its accuracy gate.** *Conjecture:* the common cause is that the all-distinct κ₃ entries, where 91–96 % of the slice error sits (§4.3), are information a cheap state does not hold. Evidence:
- **The slice-state argument (derived in an outside consultation; App. §6.6 (g)).** A perturbation H = ε Sym(e_a, e_b, e_c) with distinct a, b, c is invisible to the mean, the covariance and the slice, yet after one transport it produces a slice and a diagonal that are first order in ε. A recursion on the slice alone therefore cannot be exact to first order, and a linear state closed under every transport that returns every diagonal needs C(n+2, 3) ≈ 1.8·10⁸ numbers. The argument does **not** exclude nonlinear, restricted-family or approximate encodings (the source chain itself is one).
- **Age memory (measured).** Keeping only sources of age ≤ h leaves D21 errors of 84 / 65 / 46 / 31 / 16 % at h = 2 / 4 / 6 / 8 / 10; six age modes still leave 8.5 %.
- **Pooling (measured).** Pooling only the old sources costs 1.2–1.3 % at r = 384, but pooling everything needs r = 768.
- **Skipping and learning (measured).** κ₃ could not be deferred to a checkpoint (×14.8 to ×125.6), and our per-neuron-feature student PM364 did not learn it: its D21 error at L2, the first transport, is 57.6 % and does not move under training (two seeds). This rules out that student design, not learned states in general.
- **Averaging (derived).** By the parity theorem (§3.2), closures built on even statistics lose old κ₃ entirely.

**Where our cost goes.** A V56 call-site ledger puts 193.8 u in one warm call: about 70 u in the D21 hub readout, 63 u in young-leg transport, 20 u in the old basis, and about 40 u elsewhere, which issues 11.3k of the 20.6k operations. The 133 u in the two per-source families grows with depth, and nothing we tested removes it without at least a comparable accuracy loss. For comparison, the reference algorithm of Wu et al. costs about 4,150 u at our size (derived from their cost table), about 21× V56.

**Relation to the leaderboard.** On the 10 Oct board the six highest entries have raw MSE 1.14–1.65e-8 at util 0.106–0.144. On 20 Oct our row shows 2.8e-9 at rank 34 (tied; the board rounds to two significant figures). One entry at rank 11 has V29-level raw MSE (2.10e-8) at util 0.102. At our raw MSE, util 0.10 would score about 2.0e-9 (derived). We do not know the leaders' mechanisms. Our measurements exclude only the specific κ₃ representations in the table above as cheaper drop-in replacements inside our chain.

---

## 8. Reproducibility

**Versions.** whestbench 0.16.1, flopscope 0.12.1, dataset revision v2-phase2. Every graded number carries its AIcrowd submission id.

| Path | Contents |
|---|---|
| `submissions/<variant>/estimator.py` | Every shipped estimator, with a lineage header naming each change. #334870 is `submissions/v56As_fs/`, #334869 is `submissions/v56s_fs/`, the reference is `submissions/v29/`. |
| `submissions/e30*`, `experiments/variants/e30k_x8.py`–`e30k_x12.py`, `kaggle/x8*`, `kaggle/x10`–`x12` | The final-days line (§6.5): e30_a-based estimators, the 100-MLP harness rounds, and the self-contained Kaggle pipelines that dump, refit and evaluate the mean-correction table. e30_a is LordProcciuto's MIT release (github.com/siming767-creator/whestbench-lordprocciuto; copy in `external/`), the two-term (3,1) closure is kai_leal_gonzalez's (MIT, github.com/al07-alt/whest-p2-k31-two-term) |
| github.com/hamidhosen42/ARC-White-Box | md.hamidhosen's E-series: `experiments/k3/E2*_estimator.py` (E299 = #336377), the ledgers `experiments/results.csv` and `experiments/submissions.csv`, and the calibration tools |
| `experiments/variants/` | Research variants and oracles (`r20_joint_oracle.py`, `r20_cba.py`, `r20_feed_oracle.py`, `v55_k4c.py`, `v56_star.py`, `v56_fbg.py`) |
| `experiments/probes/` | Probes (`site_ledger.py`, `pool_ceiling.py`, `k3_error_profile.py`, `r17_hermite_family_ceiling.py`, `r20_onestep_channels.py`, `astra_rll_test.py`, `kaggle_pair_compare.py`) |
| `experiments/student/` | The PM364 student (PyTorch and flopscope versions, V56 label dump, trainer) |
| `kaggle/`, `kaggle/k4/` | Kaggle kernel wrappers, the 100-MLP harness runners, and the teacher scripts |
| `results/summaries/` | Result files, including `oct10/` and `student/` |
| `consultations/`, `docs/architecture_research/KILL_AUDIT.md` | Consultation notes and verdicts, and the kill register |
| `writeup/` | This document, the Appendix and the PDF builder |

App. §8.3 maps every Appendix section to its script and result file; known gaps in our records are in App. §8.4.

**The 100-MLP harness.** It runs the public mini split with its 10⁹-sample truth on five Kaggle CPU runners. `kaggle_pair_compare.py` reports the paired geometric-mean ratio, wins, a bootstrap 90 % interval, the score and the residual-time profile. The base is bit-identical across runners, and paired standard errors are 0.01–0.15 %. Against the grader, every predicted sign has matched the grade; for single changes worth more than about 1 % the predicted size was within 0.05–0.7 percentage points; for gains under 0.5 % the size was unreliable. The harness measures residual time but does not enforce the cap.

**Building the PDFs.** `python writeup/build_pdf.py` builds the Appendix; `python writeup/build_pdf.py writeup/WRITEUP_MAIN.md --out WRITEUP_MAIN` builds this document. All scripts will be released under MIT after the competition closes.

---

## 9. Disclosure of AI use

**Public code we build on.** #337336 runs on LordProcciuto's e30_a (MIT; its README discloses Claude assistance) and uses kai_leal_gonzalez's two-term (3,1) closure (MIT); both licences and attributions are kept in the shipped files. The corrector, the closure settings and the refitted table are ours (§6.5).

**Code.** The E-series layer of §6.4 (E282–E299) and its calibration fits are md.hamidhosen's, in his repository (§8). Apart from that layer and 504aldo's reference code, essentially all code and probe scripts were written by AI coding assistants (Anthropic Claude models, mainly Claude Opus 5 and 5.5, via Claude Code) under the team's direction. This includes every estimator variant, oracle, probe, harness runner and the PM364 student. Claude subagents also carried out specific derivations and measurement tasks.

**Ideas.** Many architecture proposals came from consultations with outside AI systems (GPT-5.6, Astra 6, Gemini 3.1 Pro, Grok, DeepSeek, Copilot, SciMaster/Bohrium, and several unsigned answers) over more than 18 rounds (App. §6.4, §9). Of the shipped ideas:
- the spherical-input reparameterisation was pointed out in an outside consultation, which referred to a public repository;
- the cubic-star feature comes from an outside proposal built on our own ceilings;
- the parity argument and the FB-scale derivation came from Claude subagent work with us;
- the slice-state argument (§7.1) came from an outside consultation.

Most proposals were killed (§7); several outside documents reported passing measurements that no run supported (App. §6.4).

**Write-up.** The text of this write-up and of the Appendix was drafted by AI and checked by the team.

**Validation.** Apart from the cited figures of 504aldo, Wu et al. and the public leaderboard, and the numbers marked as estimates, every number here was produced by our own scripts (or derived from them) or by the official grader. Every outside claim was re-derived or re-measured before we used it. On 11 Oct 2026 an automated, AI-run fact-check compared every number in this document with the result files, and the errors it found were corrected. No leaderboard feedback entered any fitted coefficient: the V55 map was fitted against our own teacher for MLP #0, the shipped V56 maps against our own teacher for the synthetic network, and the E-series calibration on public dataset MLPs with their published truth (mini 0–47, rider on full-split 100–999), validated on unseen mini MLPs 48–99; the refitted e30_a table (§6.5) on public mini nets 0–69, validated on unseen nets 70–99.

**Understanding: what was verified numerically and what is only derived.** The team directed the work and reviewed the results, but several derivations were produced with AI assistance.
- *Verified numerically:* the cubic-star and Wick identities (to ≤ 8.1e-16) and their readout factors (to ≤ 9.0e-15); the spherical identity's scale shift (to 0.3 %); the age-collapse and anchor-grouping identities (to 2.6–3.2e-7); the closure accuracy given true inputs (§4.1); the implementation parity checks (PM364's flopscope and PyTorch versions agree to 2.8e-7); the parity theorem's sign flip on the real chain (exact) and the √(6/(r+8)) floor (Monte Carlo plus exact operator norm, with a written proof).
- *Derived, consistent with measurements but not exactly confirmed:* the half-carried feedback term (regression −0.55 to −0.9 against −1/2).
- *Derived only:* the slice-state argument and the C(n+2, 3) count, and the Wu et al. cost comparison.
- *Conjectures:* the depth propagation of the (one-step measured) error cancellation of §7.1 (2), the common cause in §7.1 (3), and the link from parity to the G_off = λ_l C_off law (§3.2).

---

## References

1. Wu et al., arXiv:2605.05179 (v2). The factorised K = 3 "simple" cumulant chain (§4.3), its cost table, and the reference implementations.
2. 504aldo, *whest-p2-cumulant-k3* (GitHub, MIT licence), and WhestBench Phase 2 forum post #18218. The V29 factorised third-cumulant chain used as our reference method.
3. WhestBench (whestbench 0.16.1) and flopscope (0.12.1), the ARC White-Box Estimation Challenge 2026 benchmark and FLOP meter (Alignment Research Center / AIcrowd). The organiser ruling of 11 Sep 2026 states that Strassen–Winograd is legal at any recursion depth.
4. Appendix (full technical report): *Where the Error of the Factorised Cumulant Chain Lives: Exact Identities, Cost Engineering, and a Cubic-Birth κ₄-Pattern Programme for Deep ReLU MLPs*, Hack2Publish, 11 Oct 2026 (`writeup/ALGORITHM_TRACK_PAPER.md`).
