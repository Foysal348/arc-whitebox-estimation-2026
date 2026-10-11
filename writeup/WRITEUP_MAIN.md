# Where the Error of the Factorised Cumulant Chain Lives, and What It Costs

**WhestBench Phase 2, Algorithm track. Team Hack2Publish: mdhamidhosen, esfer_sami, foyshal_sk.**

**Submission #334870** (v56As_fs, A class): adjusted score **3.6521e-9**, raw final-layer MSE **1.950e-8**.
**Nomination-safe submission #334869** (v56s_fs, B class, ≤ ~18k warm-call operations): adjusted score **3.6723e-9**.

*11 October 2026. This is the main write-up. The full technical report is attached as the **Appendix (full technical report)**, and "App. §x" refers to its sections. Each number is labelled by its source:*
- *graded: an official grade, with its submission id;*
- *measured: a run of our own scripts, with the result file named in App. §8.3;*
- *derived: algebra or arithmetic on measured numbers;*
- *estimate or untested: not yet measured, and marked as such.*

---

## 1. Summary

We started from 504aldo's public, MIT-licensed factorised third-cumulant chain ("V29"), which graded 5.418e-9 when submitted verbatim. Our own changes take it to **3.6521e-9 (−32.6 %)**. In terms of the score's two factors, raw MSE falls by 8.6 % (2.133e-8 → 1.950e-8) and util falls by 26 % (0.254 → 0.187) (derived).

The gains come from three kinds of work:
1. **Two exact, zero-cost accuracy fixes.**
   - A spherical-input identity.
   - A derived partial restoration of a half-carried feedback term, shipped as FB 1.25 together with a retuned adaptive-λ exponent.
2. **Cost engineering under the metered budget and the grader's hidden limits.**
   - A transport join for the shared basis.
   - Dead-neuron compaction and D21 trimming.
   - Strassen placement and aliased Grams.
   - An operation budget derived from two graded failures of the residual-time cap.
3. **A κ₄-pattern programme at zero FLOPs.**
   - A linear pattern corrector in the mean closure (V55).
   - A Gaussian cubic "star" feature read from the chain's own young source legs (V56).

Alongside the estimator we built an **error map** of the chain from 6·10⁷-sample Monte-Carlo teachers:
- The closures are exact when they are given true inputs.
- The error is input error that is made fresh at each ReLU and then accumulates through depth.
- The teacher's κ₃ and κ₄ inputs together would take the raw MSE to ×0.272 of the chain's.

We also report 24 of the more than 60 architecture and representation ideas we killed, each with its gate and its measured number. Together they support three conclusions:
- inside the sum-of-sources family, cost and accuracy trade one for one;
- local corrections of one channel make the final mean worse;
- every source-free or pooled κ₃ state we could build lacks the all-distinct κ₃ content and fails its accuracy gate.

On the 10 Oct leaderboard our raw MSE is comparable to entries ranked 11–40. What separates us from the top is cost.

---

## 2. Starting point: the reference method (not claimed)

### 2.1 Task and score

For a bias-free ReLU MLP of width n = 1024 and depth 16, with He-initialised weights and input x ~ N(0, I), the estimator returns E[h⁽ˡ⁾] for every layer. The score is

  score = MSE_final × max(0.1, FLOPs / 2⁴¹),

with the final-layer MSE measured against a 10⁹-sample Monte-Carlo reference. FLOPs are metered by flopscope.

We use **1 u = 2³¹ FLOPs**, which is about one dense 1024³ product. The budget is then 1024 u, "util" is FLOPs/budget, and the 0.1 floor is 102.4 u.

The grader caps each MLP at:
- 120 s of wall time;
- 0.4 s of residual Python time outside metered operations;
- 8 GB of address space.

A capped MLP scores about 0.35, so a single cap in a hundred MLPs ruins a submission.

### 2.2 504aldo's V29 (MIT)

Our estimator is a modification of **504aldo's public V29** (`whest-p2-cumulant-k3`, forum post #18218). V29 implements the factorised K = 3 "simple" cumulant propagation of Wu et al. (arXiv:2605.05179, §4.3). Per layer it carries:
- the mean;
- the covariance C;
- the third cumulant κ₃ as a **sum of sources**, one per ReLU. Each source is stored as a propagator leg P_s, a birth leg A_s = P_s a_b, and thin n×16 feedback legs.

The next ReLU needs the slices D3_i = κ_iii and D21_ic = κ_iic. These are read per source from contractions of Hadamard products of the legs.

Further features of V29:
- **Shared basis.** Sources older than four layers live in a shared basis of rank 384, and the oldest in a nested rank-224 sub-basis.
- **κ₄.** κ₄ is not carried. It is regenerated each layer as G_off = λ_l C_off, with a per-MLP adaptive λ.
- **Products.** Products use pooled Strassen–Winograd.

The author prices V29 at 260 u, and reports a tolerance law: extra final MSE ≈ 4.2e-6·ε², where ε is the relative RMS error of D21. D21 must therefore be right to about 2 %.

**Graded baseline.** V29, submitted verbatim, graded **5.418e-9** (raw 2.133e-8, util 0.254; **#331681**). This matches the author's own submission.

**Licence and scope of our claims.** Every file we ship keeps 504aldo's MIT copyright notice and LICENSE, and carries a lineage header that names each change and its date. **Only the changes listed in §3 are claimed as ours.** The sum-of-sources representation, the shared basis, the memoryless κ₄ law, the feedback legs and the pooled Strassen engine are the reference method.

---

## 3. Our contributions and their effects

### 3.1 Graded trajectory

| Step | Our change | Score (id) | Raw MSE | Util |
|---|---|---|---|---|
| V29 verbatim (reference) | — | 5.418e-9 (#331681) | 2.133e-8 | 0.254 |
| Strassen recursion floor 32 → 8 | Two more Strassen levels on the basis families | 5.220e-9 (#331686) | 2.132e-8 | 0.244 |
| + first call at full depth | Strassen depth on each worker's first call | 5.195e-9 (#331732) | 2.132e-8 | 0.2436 |
| + V31 spherical-input identity | Exact identity, 0 FLOPs | 5.137e-9 (#331787) | 2.108e-8 | 0.2437 |
| + FB 1.25, β 1.2 | Derived partial restoration of the K×C feedback, plus the adaptive-λ exponent 1.0 → 1.2 | 5.1284e-9 (#332110) | 2.1046e-8 | — |
| Transport join | Shared basis rebuilt at a join by transport, with no range finder | 4.9519e-9 (#332257) | 2.1788e-8 | 0.2273 |
| + AGE_OLD 5, R_FB 4 | One more dense young age; feedback rank 16 → 4 | 4.8066e-9 (#332324) | 2.1371e-8 | 0.2249 |
| v48 compaction | Dead neurons (Φ < 1e-2) dropped from the next transport | 4.6294e-9 (#332391) | 2.1462e-8 | — |
| v49 basis-side Strassen + aliased Grams | §3.7 | 4.4157e-9 (#332401) | 2.1463e-8 | — |
| v50 D21 live columns | D21 columns of dead neurons never computed | 4.2512e-9 (#332404) | 2.1537e-8 | — |
| v51 D21 rows | D21 rows of very dead neurons never computed | 4.1081e-9 (#332418) / 4.0902e-9 (#332420) | 2.1545e-8 / 2.1636e-8 | 0.189 |
| v52 operation budget, B / A | §3.8: safe against the residual cap | 4.1696e-9 (#332566) / 4.1228e-9 (#332567) | — | — |
| + R_OLD 352, Tq from QR, R_FB 2 | Exact or near-exact trims | B 4.0995e-9 (#332626); A 4.0784e-9 (#332621) | 2.197e-8 (B) | — |
| V55 κ₄-pattern map, B / A | §3.9 | 3.8762e-9 (#334588) / 3.8582e-9 (#334591) | — | — |
| + s3sq coefficient ×1.25, B / A | Map coefficient | 3.8539e-9 (#334762) / 3.8331e-9 (#334763) | — | — |
| V56 cubic star, B / A | §3.10 | 3.6829e-9 (#334805) / 3.6597e-9 (#334811) | 1.9706e-8 / 1.9539e-8 | — |
| + map fitted on the synthetic net, B | | **3.6723e-9 (#334869)** | — | — |
| + map fitted on the synthetic net, A | | **3.6521e-9 (#334870)** | **1.950e-8** | 0.187 |

Util is the grader's figure or adjusted ÷ raw. "—" means our records do not hold it. The B class keeps ≤ ~18k warm-call operations for a safety margin under the residual-time cap; the A class runs at about 20k.

### 3.2 Spherical-input identity (derived; V31)

Write the Gaussian input as X = (R/√n)·Y, with Y uniform on the sphere of radius √n and R ~ χ_n independent of Y. A bias-free ReLU network is positively homogeneous of degree 1, so

  E[h(X)] = a₁ E[h(Y)],  a₁ = √(2/n) Γ((n+1)/2)/Γ(n/2) = 0.99975589.

The sphere input has C = I and κ₃ = 0, and its κ₄ is an exact harmonic core, g = −6/(n+2) in the chain's representation. The estimator injects that core at layer 0, runs the existing κ₄ machinery unchanged, and rescales the output by a₁.

- **Measured:** the injection reproduces the predicted scale shift to 0.3 %.
- **Graded:** −1.1 % raw at zero cost (#331787).

The idea came from a public Phase-2 repository, surfaced by an outside-AI consultation (App. §2.1). The derivation and its integration into this chain are ours.

### 3.3 Feedback scale and adaptive λ (derived and measured; FB 1.25, β 1.2)

Expanding the post-ReLU κ₃ to first order in C shows that the newborn source carries exactly **half** of two all-distinct K×C families (App. §2.4). Regression on the slice residual gives coefficients of −0.55 to −0.9, against −1/2 predicted.

Restoring the missing families in full makes the final mean worse: +3.4 % / +3.1 % on the teacher pair, and ×1.024 on 100 MLPs at FB 2.0. A quarter restoration (FB 1.25) together with the adaptive-λ exponent β = 1.2 scored ×0.9946 ± 0.0007 on the 100-MLP harness and graded 5.1284e-9 (#332110), at zero FLOPs.

Why the full restoration hurts is still open. One explanation is ruled out (App. §6.5 (b)): the quadratic feedback cross term that also grows with FB is inert. The linear restoration itself is what hurts.

### 3.4 Transport join

In V29, each time a source joins the shared basis, the basis is rebuilt with a randomized range finder. We instead **transport the previous basis** through the layer and append the joiner, with no range finder.
- Raw MSE rises slightly (2.1046e-8 → 2.1788e-8).
- Util falls from 0.2437 to 0.2273.
- The score improves by 3.4 % (graded #332257; harness predicted ×0.959).

### 3.5 AGE_OLD 5 and feedback rank 4

We keep one more young age dense before a source enters the shared basis, and cut the feedback rank from 16 to 4. Graded 4.8066e-9 (#332324), −2.9 % (derived).

On V56, a 100-MLP grid of five nearby age and rank settings scores ×1.026 to ×1.104 (§5). The shipped setting is the optimum of the settings we tested.

### 3.6 Dead-neuron compaction and D21 trimming

A neuron that is almost never active (Φ = P(z > 0) small) has a near-zero post-ReLU response. We exploit this on the **input side only**:
- **v48:** neurons with Φ < 1e-2 drop out of the next transport's inner dimension, with live widths padded to multiples of 32. −3.7 % (#332391).
- **v50:** the D21 columns of dead neurons are never computed. −3.7 % (#332404).
- **v51:** the D21 rows of very dead neurons (Φ < 3e-3 or 7e-3) are never computed. −3.4 % / −3.8 % (#332418 / #332420).

On 26 Sep alone, compaction, trimming and §3.7 took the score from 4.8066e-9 to 4.1081e-9 (−14.5 %). Over the whole cost programme, the trims removed 23 % of the metered FLOPs (util 0.244 → 0.189) for +2.8 % raw.

The *output* side cannot be compacted. We measured each of the following on the teacher pair:

| Attempt | Effect |
|---|---|
| Removing dead neurons as hubs of the new source | ×6.40 / ×6.74 |
| Zeroing their entries of the transported covariance | ×1.59 / ×1.46 |
| Treating always-on neurons as hubs | ×10.0 / ×8.0 |

### 3.7 Strassen placement and aliased Grams

- **Recursion floor.** Lowering V29's Strassen recursion floor from 32 to 8 added two levels on the rank-384/224 families: −3.7 % (#331686). The prediction drift was 4.9e-13, at a peak of 5.97 GB.
- **v49.** We routed the basis-side products through Strassen: join projections, factor rotations, the Qc formation and the D21 lift. The joiner Grams are written as YYᵀ, which the meter bills at half. Together these gave −4.6 % (#332401).
- **Level 6** was ruled out at a peak of 8.13 GB, above the 8 GB cap.

### 3.8 The residual-time and operation-budget discipline

Two variants that ran cleanly on our machines graded catastrophically:

| Submission | Grade | Share of MLPs capped |
|---|---|---|
| #332402 | 0.0434 | ≈ 12 % |
| #332425 | 0.0204 | ≈ 6 % |

Memory was not the cause. We measured that the residual Python time tracks **the number of metered operations**, and nothing else:
- 0.053–0.059 ms per op on Kaggle, for every variant;
- about 0.02 ms per op inferred for the grader.

The failing variants issued 21.7k–22.3k operations; one at about 16.4k had passed. The fix was an **operation budget**: Strassen is used only where it saves at least about 2 u per 1,000 operations, because each top-level Strassen call costs 110–240 operations. The budget defines our two classes:
- **B (nomination-safe):** ≤ ~18k operations.
- **A:** about 20k operations.

The B/A budget cost +1.9 % / +0.8 % graded against v51, and neither class has had a capped MLP since (#332566, #332567).

A second hidden limit appeared with V56. A 3-D broadcast of the cubic star passed in-process runs but failed the grader's address-space limit four times. Only `whest run --runner subprocess` reproduced the failure. The per-slot 2-D form passed (#334805).

`whest validate` catches neither limit. Our pre-submission check is therefore an operation count plus a subprocess run.

### 3.9 V55: a zero-FLOP κ₄-pattern map (graded)

The error map (§4) shows that the κ₄ gap is the **per-neuron pattern** of the κ₄ diagonal, not its level. At every layer ≥ 4, V55 maps seven per-neuron quantities the chain already holds to a predicted pattern of κ₄/v²:
- s3sq = (D3/v^1.5)²;
- the row and column energies of the standardised D21;
- the correlation-row energy of C_pre;
- |μ/σ|;
- the chain's own κ₄/v²;
- a constant.

The predicted pattern is rescaled to the chain's layer-mean level and fed **only** to the mean closure. The linear map is fitted offline, on decisive neurons (|μ/σ| < 1.5), against our own teachers for MLP #0 and the synthetic net. It costs 0 u and +310 operations.

**Pattern correlation.** 0.41–0.55 on MLP #0 against 0.24–0.41 for the chain, and 0.41–0.52 held out on the synthetic net.

**100-MLP harness.** ×0.9455 [90 % CI 0.9410, 0.9498], with 97/100 MLPs improved.

**Graded.** −5.4 % in both classes. The harness projected 3.8764e-9 / 3.8562e-9; the grades were 3.8762e-9 (#334588) and 3.8582e-9 (#334591).

### 3.10 V56: the Gaussian cubic star feature (graded)

The cubic Hermite coefficient of the ReLU, c = −αφ(α)/6 in standardised units, together with the Wick identity κ(u_a³, u_b, u_c, u_d) = 6ρ_ab ρ_ac ρ_ad, injects a **star**-shaped fourth cumulant downstream of every hub (derived; verified numerically to ≤ 8.1e-16).

V56 reads this star from the chain's **own** young dense source legs (ages ≤ 6), with the per-source coefficient recorded at birth. In the notation of App. §5.8,

  star_i = 24 Σ_a c_a Ȳ_ai X_ai³.

The star enters as an eighth feature of the V55 map, at zero added FLOPs: util is 0.1857, against 0.1855 for the V55 reference.

The star idea came from an outside "Hermite Tree Quartic Bank" proposal, which built on our own Hermite-family ceilings (App. §5.4–§5.5). Alone, its ceiling is 0.47–0.52. It is complementary to the V55 features, though: with teacher μ and C, leave-one-layer-out fits rise from 0.48–0.56 to 0.65–0.74.

**Graded.**
- −4.4 % (#334805, B) and 3.6597e-9 (#334811, A).
- Refitting the map on the synthetic net gave a further −0.29 % / −0.21 %, reaching **3.6723e-9 (#334869) and 3.6521e-9 (#334870)**.

The κ₄ programme as a whole took the score from 4.0784e-9 to 3.6521e-9 (−10.5 %) at unchanged cost. Its tuning is closed: star ×0.75 / ×1.5 cost ×1.011, and star ages 4 cost ×1.0045.

---

## 4. The error map: where the chain's error lives

All results in this section are measured. They use our Monte-Carlo teachers with 6·10⁷ samples, which record the per-layer pre- and post-ReLU means, covariances, κ₃ slices and κ₄ diagonal. The teachers cover public MLP #0 and a synthetic 1024×16 He network. An **oracle** replaces one quantity inside the chain by the teacher's value and reruns the chain end to end.

### 4.1 The closures are exact; the error is accumulated input error

- **Mean closure.** Given the teacher's true marginals (μ, var, κ₃ᵢᵢᵢ, κ₄ᵢᵢᵢᵢ), the Edgeworth mean closure matches the teacher's ReLU mean to 5e-11–1.5e-10 at every layer.
- **Covariance closure.** Teacher-forced at one layer, the bivariate covariance closure falls to the teacher's noise.
- **Accumulation.** The chain's per-layer MSE grows smoothly, from 0.7e-9 at L1 to 2.29e-8 at L16.
- **Inherited share.** Forcing the true mean at every layer removes about 80 % of the final error.

The error is therefore **fresh input error** made at each ReLU, carried forward through depth.

### 4.2 The joint oracle

The base is the plain chain at util 0.2476. App. §4.8 gives the full table.

| Teacher input injected (every layer) | MLP #0 | Synthetic net |
|---|---|---|
| None (plain chain) | 2.273e-8 (×1) | 1.920e-8 (×1) |
| κ₃ slice + d3 | ×0.607 | ×0.716 |
| κ₄ diagonal | ×0.686 | ×0.885 |
| **κ₃ slice + d3 + κ₄ diagonal** | **×0.272** | **×0.569** |
| κ₃ slice + d3 + κ₄ *pattern* at the chain's level | ×0.294 | ×0.606 |

Reading the table:
- **κ₃ is the larger single lever** on both networks.
- **The two inputs compound more than multiplicatively.** The product of the single-input ratios would be 0.416 and 0.634; jointly they give 0.272 and 0.569.
- **What remains is the variance channel.** Adding the true variance as well takes MLP #0 to ×0.057, the truth's own noise floor (App. §4.2).

### 4.3 κ₃: where the error is real, and where the gain lives

**Error against teacher noise.** We compared the chain's κ₃ with the teacher at every layer, and subtracted in quadrature a Gaussian estimate of the teacher's own sampling noise (derived; this assumes the two are independent):

| Layer | D21 off-diagonal error | Teacher noise (est.) | Net D21 error | d3 norm ratio, chain/teacher |
|---|---|---|---|---|
| 2 | 4.4 % | 5.3 % | ≈ 0 | 1.002 |
| 4 | 4.2 % | 2.5 % | 3.4 % | 0.998 |
| 8 | 4.9 % | 1.2 % | 4.8 % | 0.996 |
| 12 | 5.9 % | 0.7 % | 5.9 % | 0.993 |
| 16 | 6.9 % | 0.5 % | 6.9 % | 0.986 |

The early "error" is mostly teacher noise. The real slice error is carried and grows with depth, and the d3 level shrinks by 1.4 % at L16. Of the slice error, 91–96 % sits in the **all-distinct** post-ReLU entries κ₃(hᵢ, hⱼ, h_k) (App. §4.5).

**Localisation.** We injected the teacher's κ₃ only in a window of layers (MLP #0, base 2.2732e-8):

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

Injecting the teacher's κ₄ diagonal into the mean closure takes the raw MSE from 2.290e-8 to 1.609e-8. How that gain splits between pattern and level:
- **The teacher's pattern at the chain's own level** gives 1.551e-8, which is **108 %** of the gain.
- **The chain's pattern at the teacher's level** is *worse than no oracle* (+21.5 %).
- **κ₄ acts only through the mean closure, and only through its diagonal.** True κ₄ everywhere except the mean closure gives ≈ ×1. The off-diagonal slices injected alone give ×1.03–1.22.
- **The chain's pattern is weak where it matters.** Over all neurons, the chain's κ₄ diagonal correlates 0.78–0.83 with the truth. On the decisive neurons that drive the mean closure, it correlates only 0.26–0.34.

**Why κ₄ carriers stalled.** We measured the ceiling of whole *families* of birth laws, by restarting a Gaussian h layers back and expanding each ReLU to Hermite order K (App. §5.4, N = 4·10⁶):
- **Quadratic births are capped.** Births that stop at second order reach pattern correlation 0.45–0.55. That is exactly the band in which every second-order carrier we and outside models proposed had stalled.
- **Cubic births suffice.** Cubic births over six layers reach 0.80–0.85, and K = 4 adds ≤ 0.02.
- **Ages are complementary.** Ages 1–6 are needed jointly. Ages 1–4 give 0.68–0.74, and ages 4–6 alone give only 0.37–0.41.

This ceiling ladder is why V56 uses a *cubic* feature.

A κ₃-dressed C·T readout built from the chain's own legs raises the ceiling of star + C·T to 0.82–0.85. Under an oracle-fitted map it is worth ×0.773 end to end (App. §5.8). At r = 64 it is not affordable, at about 40 u (estimate). A rank-32 sketch keeps ×0.874 at an estimated 18–20 u. It is untested on V56 and is our main open lead.

---

## 5. Architecture search: what we tested and why it failed

"Gate" is the pre-registered pass criterion. The kill types are: C, a ceiling or oracle (an upper bound); P, a teacher-forced component test; E, an end-to-end prototype; K, a knob on the 100-MLP harness. Ratios are final-MSE or score ratios against the chain. Details are in App. §6.

| Idea | Gate | Measured | Kill |
|---|---|---|---|
| Compensating closure with no old tier (8 models proposed variants) | ≤ chain MSE | 7.43e-7, 32× the chain; every knob 6.8–9.9e-7 | C |
| Mean-field / even-statistic closures of old κ₃ | — | Zero by the parity theorem: old κ₃ is odd in later weights (derived, App. §2.2) | derived |
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

### 5.1 Synthesis

**The one-for-one law.** Inside the chain, every accuracy knob we swept on the 100-MLP harness trades about 1 % of cost for 1 % of accuracy, and the shipped constants sit at the score optimum:
- the adaptive-λ exponent and the λ level (β 1.2: ×0.9993; λ ×0.9: ×1.0026);
- the FB scale (the FB × β grid stays within +0.12 %);
- the old-basis rank (R_OLD 448: ×1.0113);
- the young-age gate (AGE_OLD 6 on V56: ×1.014), now including the five-point age/rank grid on V56.

On that grid the best ceiling point gains 0.5 % raw for 3.2 % more util. Saving 4.3 % util costs 10.3 % raw.

The law breaks only for levers that are **exact or nearly exact**: the transport join, compaction, Strassen placement and the aliased Grams. These are where our cost gains came from. A 27 Sep audit of 74 earlier kills revived 8, all of them cost or knob levers that had been killed on the wrong criterion. No architecture kill was revived.

**Coupled errors: local corrections hurt.** Again and again, a change that made one channel locally more correct made the final mean worse:
- full FB restoration (+3.4 % / +3.1 %);
- the γ-patch (×1.0243);
- the RLL term (slice residual 15–42 % worse);
- scaling the chain's κ₄→κ₃ feed toward the larger exact feed (−5 % costs 0.45 %, +10 % costs 9.4 %);
- injecting the sampled exact feed (×1.115);
- raising d3 toward the teacher's level (×1.032 / ×1.015);
- single κ₄ slices (×1.03–1.22).

Only **joint** input oracles give large gains: ×0.272 against ×0.607 and ×0.686 alone. This pattern fits channel errors that partly cancel in the mean closure. We have not measured the cancellation directly. The practical rule we adopted: local tests are necessary gates but not sufficient ones, and candidates are promoted only on the end-to-end mean.

**Information obstructions.** Several kills share one cause: the all-distinct κ₃ entries are information that a cheap state does not hold.
- **The slice-state argument (derived).** A perturbation H = ε Sym(e_a, e_b, e_c) with distinct a, b, c is invisible to the mean, the covariance and the slice. Yet after one transport it produces a slice and a diagonal that are first order in ε. A slice-only recursion therefore cannot be exact to first order. A linear state closed under every transport needs C(n+2, 3) ≈ 1.8·10⁸ numbers.
- **The age-memory spread.** Keeping only sources of age ≤ h leaves D21 errors of 84 / 65 / 46 / 31 / 16 % at h = 2 / 4 / 6 / 8 / 10. The age contributions are coherent but not low-rank: six age modes still leave 8.5 %.
- **The young sources resist pooling.** Pooling only the old sources costs 1.2–1.3 % at r = 384, but pooling everything needs r = 768.
- **κ₃ cannot be skipped early.** It cannot be deferred to a checkpoint, and it cannot be learned from per-neuron features: PM364's D21 error at L2, the first transport, is 57.6 % and does not move under training.
- **Old κ₃ cannot be averaged.** By the parity theorem it is odd in the later weights, so every closure built on even statistics loses it entirely.

**The remaining gap is cost.** On the 10 Oct leaderboard, the six highest entries have raw MSE 1.14–1.65e-8 at util 0.106–0.144. Our raw MSE, 1.97e-8 on the board, is comparable to entries ranked 11–40, and one entry at rank 11 has V29-level raw MSE (2.10e-8) at util 0.102. At our raw MSE, util 0.10 would score about 2.0e-9 (derived).

A V56 call-site ledger shows where our 193.8 u goes:
- about 70 u to the D21 hub readout;
- 63 u to young-leg transport;
- 20 u to the old basis;
- about 40 u to everything else, which issues 11.3k of the 20.6k operations.

The 133 u in the two per-source families is the bill that grows with depth. Nothing in the table above removes it without an accuracy loss of at least the same size. For comparison, the reference algorithm of Wu et al. costs about 4,150 u at our size (derived from their cost table), about 21× V56. The leaders' mechanisms are not public. Our measurements say they cannot be a cheaper drop-in κ₃ carrier of the kinds listed above.

---

## 6. Reproducibility

**Versions.**
- whestbench 0.16.1, flopscope 0.12.1, dataset revision v2-phase2.
- Every graded number carries its AIcrowd submission id.

**Repository layout.**

| Path | Contents |
|---|---|
| `submissions/<variant>/estimator.py` | Every shipped estimator, with a lineage header naming each change. #334870 is `submissions/v56As_fs/`, #334869 is `submissions/v56s_fs/` and the reference is `submissions/v29/`. |
| `experiments/variants/` | Research variants and oracles: `r20_joint_oracle.py` (joint and windowed oracles), `r20_cba.py` (CBA and κ₃ scaling), `r20_feed_oracle.py`, `v55_k4c.py`, `v56_star.py`, `v56_fbg.py` |
| `experiments/probes/` | Probes: `site_ledger.py`, `pool_ceiling.py`, `k3_error_profile.py`, `r17_hermite_family_ceiling.py`, `r20_onestep_channels.py`, `astra_rll_test.py`, `kaggle_pair_compare.py` |
| `experiments/student/` | The PM364 student (PyTorch and flopscope versions, V56 label dump, trainer) |
| `kaggle/` | Kaggle kernel wrappers; `kaggle/runners/` holds the 100-MLP harness runners |
| `results/summaries/` | Result files, including `oct10/` and `student/` |
| `consultations/`, `docs/architecture_research/KILL_AUDIT.md`, `docs/experiment_queue/queue.json` | Consultation notes and verdicts, the kill register and the experiment queue |
| `writeup/` | This document, the Appendix and the PDF builder |

App. §8.3 maps every section of the Appendix to its script and result file.

**The 100-MLP harness.** The harness runs the public mini split with its 10⁹-sample truth on five Kaggle CPU runners. `experiments/probes/kaggle_pair_compare.py` reports the paired geometric-mean ratio, wins, a bootstrap 90 % interval, the score MSE × max(0.1, util) and the residual-time profile. The base is bit-identical across runners, and paired standard errors are 0.01–0.15 %.

Calibration against the grader:
- every predicted sign has matched the grade;
- for single changes worth more than about 1 %, the predicted size was within 0.05–0.7 percentage points (V55: projected 3.8764e-9 / 3.8562e-9, graded 3.8762e-9 / 3.8582e-9);
- for gains under 0.5 %, the size was unreliable.

The harness measures residual time but does not enforce the cap. Before every submission we count warm-call operations and run `whest run --runner subprocess`.

**Teachers.** The Monte-Carlo teachers, with 6·10⁷ samples on MLP #0 and the synthetic net, are produced by the scripts in `kaggle/k4/`. The Hermite-family ceilings run on Kaggle GPU (`r17_hermite_family_ceiling.py`).

**Building the PDFs.**
- `python writeup/build_pdf.py` builds the Appendix (`ALGORITHM_TRACK_PAPER.pdf`).
- `python writeup/build_pdf.py writeup/WRITEUP_MAIN.md --out WRITEUP_MAIN` builds this document.

Both use python-markdown and a headless Edge print.

**Known gaps in our records** are listed in App. §8.4. The κ₃ level-scaling runs and the PM364 engineering checks are recorded in `results/summaries/oct10/k3scale_{1,2,3}.json` and `results/summaries/student/pm364_engineering_checks.json`. All scripts will be released under MIT after the competition closes.

---

## 7. Disclosure of LLM use

**AI assistance.** The work was done with AI coding assistants, under the team's direction. The main assistant was Claude Code (Claude Opus 5 and 5.5). It wrote and ran most probes, implemented the estimator variants and kept the notes. Internal Claude subagents did specific derivations and measurements.

**Outside models.** We also consulted outside models (GPT-5.6, Astra 6, Gemini 3.1 Pro, Grok, DeepSeek, Copilot, SciMaster/Bohrium) in more than 18 rounds.

**Origin of the ideas we shipped.**
- The spherical-input reparameterisation was pointed out by an outside consultation, which referred to a public repository.
- The cubic-star feature comes from an outside proposal built on our own ceilings.
- The parity argument comes from a Claude subagent consultation, and we checked it.
- The FB-scale derivation comes from a Claude subagent working with us.

**Verification.** Every number in this write-up was measured by our own scripts or graded by the official grader. Every outside claim was re-derived or re-measured before we used it. Several outside documents reported passing measurements that no run supported, and the Appendix (App. §6.4) says which. No leaderboard feedback entered any fitted coefficient: the V55 and V56 maps were fitted only against our own teachers for MLP #0 and the synthetic network.

---

## References

1. Wu et al., arXiv:2605.05179 (v2). The factorised K = 3 "simple" cumulant chain (§4.3), its cost table, and the reference implementations.
2. 504aldo, *whest-p2-cumulant-k3* (GitHub, MIT licence), and WhestBench Phase 2 forum post #18218. The V29 factorised third-cumulant chain used as our reference method.
3. WhestBench (whestbench 0.16.1) and flopscope (0.12.1), the ARC White-Box Estimation Challenge 2026 benchmark and FLOP meter (Alignment Research Center / AIcrowd). The organiser ruling of 11 Sep 2026 states that Strassen–Winograd is legal at any recursion depth.
4. Appendix (full technical report): *Where the Error of the Factorised Cumulant Chain Lives: Exact Identities, Cost Engineering, and a Cubic-Birth κ₄-Pattern Programme for Deep ReLU MLPs*, Hack2Publish, 11 Oct 2026 (`writeup/ALGORITHM_TRACK_PAPER.md`).
