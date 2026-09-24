# Round 12: Opus 5.5 report. The accuracy question (kappa4 / kappa3 slice), B1, B2

*25 Sep 2026. Brief: `ROUND12_fable_deep_research_prompt.md` (run by Opus instead of Fable). Public MLP #0 (n = 1024, depth 16),
its 6e7-sample teacher and 1e9 truth, and the synthetic teacher network. Every number below comes from one of the scripts
listed. Numbers I derived without running them are marked **(derived)**. Nothing was committed, pushed or submitted.*

Scripts: `experiments/probes/opus12_errmap.py`, `opus12_k4_gaussmc.py`, `opus12_k4_carry.py`, `opus12_b1_gate.py`,
`opus12_e2e.py` (a copy of `run_variant_teacher.py` that writes `opus12_e2e_<TAG>.json`), and the oracle variant
`experiments/variants/opus12_d21oracle.py`. Results are in `results/summaries/opus12_*.json`.

## 0. Verdict

1. **The final-MSE error is made late, and it is coherent across the three channels (measured, linear-response map).**
   Removing 25% of the fresh error of all three closure inputs (variance, kappa3_iii, kappa4_iiii) at every layer gives
   x0.573. Removing 25% of the variance channel's error alone gives x0.784. *[Corrected 25 Sep, see 7.2: the "fresh"
   errors include inherited error, so single-channel ratios are upper bounds on the gain of a per-layer fix of that
   channel. The all-three row is just the linear scaling (1-f)^2 = 0.5625. The sentence "only a 10-25% more accurate
   input channel is needed" is withdrawn.]*
2. **kappa4 (the brief's first thread): the deep kappa4 pattern is linear memory of the full pre-activation kappa4 tensor, not
   fresh ReLU physics.** Exact Gaussian inputs with the true mean and covariance give 1-4% of the true decisive-neuron
   kappa4 level from layer 8 on. The exact linear response of the true non-Gaussian z carries 83-90% of that level at
   L12-16, and its decisive-neuron pattern correlates 0.77-0.80 with the teacher (the chain: 0.26-0.34). The chain's
   Sym(G x I) core has no (1,1,1,1) sector, so it cannot carry this memory. The same holds for 504aldo's dense G, which
   explains why its ceiling is only -9% while the exact diagonal is worth -30%. **Verdict: no O(n^2) rule and no kernel
   of fresh physics can supply the pattern. The lever is alive only as a carried kappa4 object that has an all-distinct
   sector (needs a carrier; section 3).**
3. **Lead 2 (the missing half-families, B2 and WK12) is dead as a knob.** FB_SCALE 2 costs +3.8% / +3.2% (MLP #0 /
   synthetic), and refitting lambda makes it worse. Injecting the TRUE (2,1) slice into the feedback legs only changes
   nothing (x0.9999), so the "cross-talk" explanation of round 7 is dead. Injecting the true slice everywhere is worth
   x0.758. This is the first measured value of a perfect kappa3 slice.
4. **B1 dies at its first gate.** The best 2-product carrier of the old content misses 27% of the total D21 against a 2%
   gate. The one-step memory residual is 40% of the next slice, needs rank ~85 for 90% of its energy (~250 for 98%),
   and has zero alignment with the previous step's residual (corr -0.0004).
5. **B2 was not run.** No trajectory-specific state I could write down escapes three existing measurements: the R6
   query-subspace test, the R11 unfolding rank of the old kappa3, and the R9 duality argument (section 5).

## 1. Where the final MSE is made (`opus12_errmap.py`, analysis)

**Method.** The plain V29 chain's own channels and the teacher at every layer come from `errbudget_v29plain.npz`. The
fresh error of channel c at layer l is E(chain inputs) - E(chain inputs with c taken from the teacher). It is propagated
to layer 16 through v <- dE/dmu o (W^T v), which is exact for the mean. The linear map reproduces the measured oracles:
k4 0.702 (measured 0.699), var 0.532 (0.533), k3 0.865 (0.858), all three 0.057 (0.057). The reconstruction error of the
final error vector is <0.1%.

**Table 1. Final-MSE ratio when the named channel is fixed only at the listed layers**

| channel fixed | all layers | from L8 | from L12 | from L14 | from L15 | L16 only |
|---|---|---|---|---|---|---|
| kappa4 diagonal | 0.702 | 0.739 | 0.792 | 0.876 | 0.927 | 0.932 |
| variance | 0.532 | 0.514 | 0.629 | 0.729 | 0.773 | 0.911 |
| kappa3 diagonal | 0.865 | 0.879 | 0.835 | 0.889 | 0.943 | 0.917 |
| all three | 0.057 | 0.090 | 0.267 | 0.502 | 0.639 | 0.773 |

**Partial fixes (fraction of a channel's fresh error removed at every layer)**

| fix | ratio |
|---|---|
| kappa4, 25% | 0.869 |
| kappa4, 50% | 0.776 |
| variance, 25% | 0.784 |
| variance, 50% | 0.635 |
| all three, 25% | 0.573 |
| all three, 50% | 0.273 |

About 70% of the kappa4 value sits at layers >= 12. Fixing only |mu/sigma| < 1.5 keeps most of the value (k4 0.718,
var 0.515). Per-neuron features correlate at most 0.19 with the relative variance error on decisive neurons (inline
check). This is consistent with K3v: the variance error is a random projection of dC and has no cheap per-neuron
correction.

## 2. kappa4 thread

### Falsifier 1: is the missing pattern Gaussian physics at all orders in C? (`opus12_k4_gaussmc.py`, N = 6e5)

**Method.** z_l ~ N(mu_l, C_l) with the TEACHER's mean and covariance. Next-layer kappa4 by Monte Carlo: this is the
ceiling of every fresh kernel, including a non-perturbative version of Astra's.

**Table 2. Gaussian-input Monte Carlo against the teacher**

| next layer | decisive-neuron level, Gaussian MC / truth | errG4 of Gaussian MC (MC noise) | errG4 of chain |
|---|---|---|---|
| 2 (sanity: z1 is exactly Gaussian) | **1.021** | 0.72 (0.66) | 0.123 |
| 3 | 0.340 | 0.77 (0.38) | 0.120 |
| 8 | 0.038 | 0.97 (0.17) | 0.142 |
| 12 | 0.013 | 0.99 (0.12) | 0.193 |
| 16 | 0.011 | 1.00 (0.11) | 0.210 |

**Result.** Fresh ReLU physics from a Gaussian input makes 1-4% of the deep kappa4. A "stronger" (all-orders) Astra
kernel is capped by this.

**Verdict.** The lead "stronger kernel" is dead. Astra's measured -6.4% cannot come from fresh trees. It comes from the
kernel's d1-dressed transport of the chain's own core, f1 = metric (sum u^2)(u^T G u), plus the exact pair product.

### Falsifier 2: what carries the pattern? (`opus12_k4_carry.py`, exact Monte Carlo through the true net, N = 2e6)

**Method.** Two quantities are estimated from exact samples of z_l:

- **LIN** is the kappa4 of the linear response y = (Phi(a_l) o dz_l) @ W_l. It contains every sector of the true
  kappa4(z_l), carried linearly, with no fresh ReLU physics.
- **QUAD** adds the ReLU curvature term.

Comparisons are against the independent 6e7 teacher, on decisive neurons.

**Table 3. Decisive-neuron comparison against the teacher**

| layer | decisive corr: MC truth | decisive corr: LIN | decisive corr: chain | level LIN / T | errG4: chain | errG4: LIN, affine | errG4: QUAD, affine |
|---|---|---|---|---|---|---|---|
| 8 | 0.845 | 0.541 | 0.365 | 0.76 | 0.142 | 0.129 | 0.106 |
| 12 | 0.932 | 0.771 | 0.339 | 0.83 | 0.193 | 0.126 | 0.114 |
| 15 | 0.947 | 0.785 | 0.264 | 0.90 | 0.199 | 0.128 | 0.115 |
| 16 | 0.965 | 0.797 | 0.324 | 0.86 | 0.210 | 0.136 | 0.118 |

- The two errG4 columns marked "affine" use a 2-parameter in-sample fit (level + scale). The chain's slope is already
  about 1, so the chain column is not fitted.
- The "MC truth" correlation measures the Monte Carlo noise ceiling for that layer.

**Zero-cost structured predictors of LIN fail.** Both come from (mu, C) only:

| predictor | decisive corr with LIN |
|---|---|
| scale-mixture 3eps (u^T(C + mu mu^T)u)^2 | -0.05 to 0.08 |
| Sym(C x C) | -0.17 to 0.09 |

Inline check: the chain's kappa4 x var gives corr -0.33 to -0.27, and kappa4 / var gives 0.29-0.37 (chain alone:
0.26-0.37).

**Reading.** The per-neuron kappa4 pattern at depth is the linear image of the (2,1,1)/(1,1,1,1) sectors of the
pre-activation kappa4 tensor. Those sectors are themselves carried from earlier layers. *[Withdrawn 25 Sep, see 7.4:
"fed mostly by kappa3-driven births". Not measured, and QUAD overshoots the level 1.4-1.7x, so the second-order split
is unreliable.]*

- The chain's use side, Sym(G x mI), has a (1,1,1,1) sector of **0 by construction** (F51 table). The same is true of
  504aldo's dense G channel.
- This measures why the dense core is worth only -9% (F105) while the exact diagonal is worth -30% (K3ac).
- The regeneration's lambda C_off captures the level but not the pattern.

**Verdict.** O(n^2) rules are dead, and every kernel of fresh physics is dead. **What is needed is a carried kappa4
object with an all-distinct sector.** No cheap closed form of one was found: Sym(A x B) forms with A, B built from
(mu, C) fail above. A deployable carrier would need births from the young kappa3 sources plus transport (section 6).

### Lead 1, cheaper kernel (derived ledger, no run)

- The kernel's three products are Q = C_pre (d1 W), R = C_pre (d2 W o Q), and the merged pair product (6 K22 - S cH) @ W^2.
- Q can be read from the family's W^T C_post only through a division by Phi(a_p). That amplifies the chain's C^2 terms
  by (phi/Phi)^3 for a < -2, so it is not safe.
- At best one product is saved: 2 of 3 products = ~1.0 u/layer, ~14 u in total at Strassen L5.
- With a7's measured -6.4% / -2.6% this is still about net zero on MLP #0 and negative on the synthetic net. Applying the
  kernel only late loses its good layers (a7: LMIN 8 gives -3.5%). Table 1 says the value is late, but the kernel is
  worst there (r8).
- **Dead as a score lever.**

### Lead 2, the missing half-families, FB x lambda (`opus12_e2e.py` on the shipped v40fb, both teacher networks)

**Table 4. Paired ratio to the shipped setting (FB 1.25, LAM 0.95)**

| configuration | MLP #0 | synthetic |
|---|---|---|
| FB 2.0 (full Edgeworth families) | 1.0382 | 1.0318 |
| FB 2.0, LAM 0.85 | 1.0439 | 1.0305 |
| FB 2.0, LAM 0.75 | 1.0669 | 1.0446 |
| LAM 0.85 | 1.0181 | 1.0103 |

- Base absolute MSE: 2.2364e-8 on MLP #0 (layer 16) and 1.8895e-8 on the synthetic net (layer 15).
- Refitting lambda does not rescue the full families.

**D21 oracle** (`opus12_d21oracle.py`, plain Gaussian chain with FB 1.25 / BETA 1.2, MLP #0, base 2.2747e-8):

| injected slice | ratio |
|---|---|
| true slice into the feedback thin legs only | 0.9999 |
| true slice everywhere (closure, feedback, S21) | **0.7580** |

So the D21 error does not act through the feedback, and a perfect kappa3 slice is worth -24%. The "kappa4 to kappa3 feed
arm" is already in the chain (F68 X3/Y3/M_t1). **Dead as a knob.**

## 3. B1: old-content surrogate, first gate (`opus12_b1_gate.py`)

**Part 1 (captures L14 -> L15).**

- The transport check is exact: an L14 leg transported by diag(w1) W[14] equals the L15 leg to <5e-5.
- The old sources (10 sources, age >= 5) carry 65.4% of ||D21||.
- The one carrier tested (about 2 products) is the (2,1) slice of the old content under the exact repeated-index
  slice map (scale 0.9985). Its errors are below. *[Reworded 25 Sep, see 7.5: this is one candidate, not a proven
  optimum.]*

| measure of the slice-map carrier | error |
|---|---|
| against the total D21 | **27.3%** (gate: 2%) |
| on decisive rows | 24.2% |
| against the old content itself | 41.8% |
| old d3 | 24.7% |

**Part 2 (teacher, true post states).** The one-step missing part (true next slice minus the exact repeated-index
transport of the true post slice) is 40.6% / 40.1% of the next slice at L15 / L16.

| layer | energy rank for 90% | energy rank for 98% |
|---|---|---|
| L15 | 87 | 253 |
| L16 | 82 | 245 |

The L15 missing part, dressed and transported, predicts the L16 one with corr -0.0004. No single n x n matrix updated by
<= 4 products carries it: that is the slice-transport family, which differs from K3g/RSC only in using exact
coincidence algebra, and fails the same way. **Dead at the first gate.**

## 4. B2: nonlinear trajectory-specific state (not run)

I found no candidate that escapes three measured facts:

- Old-tier content acts through near-isotropic queries: 34 / 12 / 5% left at rank 128 / 256 / 384 (R6).
- Any state that is linear in a latent representation is bounded by the unfolding rank ~250 of the old kappa3 (R11).
- Precomputing per-network pull-backs is the reverse-mode contraction order, which costs what the chain costs (R9).

Section 2 adds one fact: the kappa4 channel has the same memory structure as the kappa3 channel. **No script spent;
status: needs a new idea.**

## 5. The single most promising next experiment

*[Superseded 25 Sep, see 7.4. The horizon gate below was answered by Q18 plus `opus12_k4_young_blend.py`: even the ideal
young carrier (h = 2) fails its own go criterion. The replacement proposal is at the end of section 7.]*

**Gate: the kappa4 memory horizon**, offline, 0 u, about 30 min of CPU. Extend `opus12_k4_carry.py` with LIN_k: the
kappa4 at L of the k-step linearised response from the true z_(L-k), for k = 1..4. Then T - LIN_k is what the last k
ReLUs gave birth to.

**Deployable mechanism it would gate (derived cost).**

- Carry the K3 x C kappa4 births of the youngest kappa3 source for h layers.
- Each birth is Sym(e_a w2_a x C_a x K3_s(a, ., .)). Its diagonal at a later layer needs the newborn's (A o P), which
  `_dslices` already forms, and two products X_s(k) (P o Y)^T per (source, birth layer, target layer).
- For one source and h = 2 that is 4 products/layer: ~2 u/layer, ~28 u in total, +11% cost **(derived)**.

**Go / no-go.**

- **Go** only if, on decisive neurons, the k <= 2 births carry at least half of the pattern that separates T from the
  chain: corr with the teacher rising from 0.26-0.34 toward 0.8. In that case, build the carrier and gate it end to end
  at <= 28 u for >= 12% raw.
- If LIN_2 already holds most of the pattern (a long memory), the kappa4 lever costs a full kappa4 source chain. It is
  then closed on cost, like the aug channel.

## 6. Status of every claim

| claim | status |
|---|---|
| Table 1 ratios, partial fixes; linear map equals the oracles | measured (`opus12_errmap.json`) |
| Gaussian-input kappa4 = 1-4% of the deep level; L2 sanity 1.02 | measured (`opus12_k4_gaussmc.json`, N 6e5) |
| LIN level 83-90% at L12-16, decisive corr 0.77-0.80 against the teacher; chain 0.26-0.34 | measured (`opus12_k4_carry.json`, N 2e6) |
| LIN / QUAD affine errG4 0.13 / 0.12 | measured, **2-parameter in-sample fit** |
| Scale-mixture and Sym(C x C) predictors fail | measured |
| The chain's and dense G's (1,1,1,1) sector is zero | derived (F51 use-side table) |
| FB x LAM paired ratios | measured (`opus12_e2e_fblam.json`) |
| D21 oracle: feedback 0.9999, everywhere 0.758 | measured (`opus12_e2e_d21oracle.json`) |
| B1: 27.3% vs 2% gate; missing part 40%, rank 87 / 253, alignment -0.0004 | measured (`opus12_b1_gate.json`) |
| Kernel "cheaper" ledger (~14 u), carrier cost 4 products/layer (~28 u) | derived, not run |
| B2 closed by R6 / R9 / R11 | argument on earlier measurements, no new run |
| Leaderboard: no public 504aldo release after 22 Sep (commit list); forum topic 18218 has nothing past F105 | checked on the web, 25 Sep |

## 7. Response to review (25 Sep)

This round used four new script runs:

- `opus12_e2e.py` with the extended `experiments/variants/opus12_d21oracle.py`, both networks
- `opus12_k4_carry_synth.py`
- `opus12_q18_noise.py`
- `opus12_k4_young_blend.py`

Their results are in `results/summaries/opus12_e2e_d21oracle2.json`, `opus12_k4_carry_synth.json`,
`opus12_q18_noise.json` and `opus12_k4_young_blend.json`.

**Chains and networks used.** The D21 oracle runs the shipped v40fb with `V31_ANGULAR=0` (FB 1.25, BETA 1.2). The
angular input has to be off because its internal state describes the spherical input, which the Gaussian teacher cannot
be compared with. `errmap` and the kappa4 comparisons use the plain V29 dumps (FB 1.0, BETA 1.0), on MLP #0 and on the
synthetic net (`errbudget_v29plain_synth.npz`).

| # | point | verdict | evidence / new wording |
|---|---|---|---|
| 7.1 | Single network | **proved on the second network** (b is partial there) | See **7.1** below. |
| 7.2 | "Fresh" error is not fresh | **corrected** | See **7.2** below. |
| 7.3 | The D21 oracle keeps the chain's d3 | **stated, and the perfect-kappa3 value measured** | "All" means the teacher's **off-diagonal** (2,1) slice only; the chain's own d3 = kappa3_iii is kept. New mode `all_d3` injects the teacher d3 = diag(Y21) as well, at every layer including the trimmed last one. A perfect kappa3 input (slice + d3) is worth **x0.6075 on MLP #0** and x0.7147 on the synthetic net (partial, 8 of 14 layers). |
| 7.4 | Unmeasured "fed by kappa3 births"; Q18 check | **sentence withdrawn; Q18 audited: correct, with one caveat; young carrier measured dead** | See **7.4** below. |
| 7.5 | "Best carrier" in B1 | **corrected** | New wording: "the one carrier tested (the exact repeated-index slice map of the old (2,1) slice, ~2 products) misses 27.3% of the total D21". The rank-87 / zero-alignment facts show that this family cannot be patched by a low-rank or recursive correction. They are **not** a lower bound over all carriers, so "best" is dropped. The B1 verdict now reads: dead for the slice-map family; other carriers were not tested. |
| 7.6 | Derived costs (~14 u kernel, ~28 u carrier) | **confirmed: no decision depends on them beyond "net <= 0"** | See **7.6** below. |

**7.1 Single network.**

- **(a) proved.** `opus12_k4_carry_synth.py`, synthetic net, N 1.6e6. Decisive-neuron correlation with the independent
  teacher, split-half noise-corrected values in brackets:

  | layer | LIN | chain | MC truth | LIN level |
  |---|---|---|---|---|
  | L8 | 0.539 (0.729) | 0.338 | 0.827 | 0.740 |
  | L12 | 0.736 (0.813) | 0.328 | 0.920 | 0.853 |
  | L14 | 0.776 (0.852) | 0.252 | 0.930 | 0.864 |
  | L15 | 0.816 (0.876) | 0.242 | 0.953 | 0.865 |

  - The Monte Carlo truth has level 1.00 against the teacher.
  - errG4 at L15: chain 0.223, LIN (affine, 2-parameter in-sample fit) 0.122.
  - Same picture as MLP #0: LIN ~0.8, chain ~0.25-0.34.
- **(b) proved on MLP #0, partial on the synthetic net.**
  - The true slice everywhere gives **x0.7580** on MLP #0, reproduced bit for bit (1.7243e-8 against 2.2747e-8).
  - On the synthetic net it gives **x0.8784** (1.6894e-8 against 1.9232e-8).
  - The synthetic teacher stores Y21 only at layers 2, 4, ..., 14, 15, so only 8 of the 14 injectable layers are
    corrected. x0.878 is therefore an upper bound on the ratio a full oracle would give (a lower bound on its gain).

**7.2 "Fresh" error is not fresh.**

- The map's per-layer term is E(chain inputs) - E(chain with channel c from the teacher). It contains the error
  inherited from layers < l, including error fed into channel c by the other channels' mean and covariance errors.
- Single-channel ratios (variance 0.532 / 25% -> 0.784, kappa4 0.702 / 25% -> 0.869) therefore measure the value of
  reducing the **total** error of that channel as it enters the mean closure. For a fix that improves only that
  channel's own per-layer generation, they are **upper bounds on the gain**.
- The "all three" rows are the linear scaling of the whole error: x0.573 at f = 0.25 and x0.273 at f = 0.5, against
  (1-f)^2 = 0.5625 and 0.25.
- New wording for section 0 item 1: *"The error is late: about 70% of the kappa4 value, and most of the variance value,
  sits at layers >= 12. Single-channel ratios are upper bounds for per-layer fixes."*
- The late-layer localisation stands as measured. I did not run a teacher-forced one-layer measurement: it needs a
  teacher-forced source state that does not exist. The statement is kept as a bound.

**7.4 Unmeasured claim; audit of Q18.**

- **Sentence withdrawn.** "Fed mostly by kappa3-driven births" was not measured.
- **Q18 method and indexing: correct.**
  - A negative control (h = 0: Gaussian z_L itself) gives level 0.001 +- 0.004 and correlation -0.05.
  - Replicated levels agree within noise: L16 h=2 0.032 +- 0.007 (Q18: 0.024), h=6 0.196 (0.195); L12 h=2 0.048
    (0.055), h=6 0.365 (0.353).
  - The decisive weighting is consistent: level and correlation unweighted on |mu/sigma| < 1.5, errG4 G4-weighted over
    all neurons. This is the same convention as mine.
- **Caveat: the pattern correlations are noise-attenuated at N = 1e6.** The per-neuron noise std in excess kurtosis is
  0.0044-0.0051, the same size as the h = 2 signal std (0.0051-0.0058). Split-half reliability is 0.50-0.62 at h = 2 and
  0.76-0.83 at h = 6.

  | case | Q18 correlation | noise-corrected |
  |---|---|---|
  | h = 2, L16 | 0.41 | **0.52** |
  | h = 2, L12 | 0.41 | **0.58** |
  | h = 6, L16 | 0.77 | **0.85** |
  | h = 6, L12 | 0.79 | **0.90** |

  So the LEVEL memory is long, as Q18 says (h = 6 holds 20-37%). The PATTERN memory is shorter: 6 steps hold 0.85-0.90
  of it and 2 steps about half.
- **Is the young content usable?** `opus12_k4_young_blend.py` blends the chain with the ideal young carrier (exact
  births of the last h layers, exact transport: chain + a K_h + b var^2). The fit is made on L12 and tested on L16, and
  vice versa.

  | carrier | errG4 at L16 (chain 0.210) | errG4 at L12 (chain 0.193) | decisive corr (chain 0.32-0.34) |
  |---|---|---|---|
  | h = 2 | 0.203 | 0.184 | 0.43-0.46 |
  | h = 3 | 0.188 | 0.173 | 0.54-0.56 |

  - Out-of-sample and in-sample errors agree.
  - At most a 10% cut of the kappa4 error, for an ideal carrier of 3 layers. From Table 1 this is of order -2 to -4%
    MSE **(derived)**.
  - My own gate (correlation toward 0.8) fails. **The young kappa4 carrier is dead**, and the section 5 experiment is
    closed.

**7.6 Derived costs.**

- **Cheaper kernel.** Break-even against a7's measured -6.4% / -2.6% is 249 (1/0.936 - 1) = **17 u** on MLP #0 and
  249 (1/0.974 - 1) = **6.6 u** on the synthetic net (arithmetic).
  - The kernel needs at least the two dense products Q = C(d1 W) and R = C(d2 W o Q) per layer, >= 28 products in total.
    That is >= 6.6 u at any price above 0.24 u per product, and Strassen L5 bills 0.51.
  - So "net about zero on MLP #0, negative on the synthetic net" holds for any metered cost in that range. Only the sign
    is used.
- **~28 u carrier.** Moot: the carrier is dead on value (7.4), whatever its cost.

**Replacement next experiment (0 u, oracle only).**

- The largest lever measured this round is the kappa3 input: slice + d3 is worth x0.61 on MLP #0, and the kappa4 carrier
  is closed.
- Proposed next run: `opus12_d21oracle.py` in `all_d3` mode restricted to layers >= 12 (and >= 14). By analogy with
  Table 1 it would price a late-layer-only kappa3 refinement.
- No deployable mechanism for it is identified. Round 7-8 found the slice error unexplained past second order.
