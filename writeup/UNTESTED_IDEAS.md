# Untested ideas from the 8-model consultations (rounds 1–2) — status board, 21 Sep 2026

Legend: **T** = testable now with existing dumps/tools · **K** = needs a Kaggle run · **R** = needs new code · prior = my estimate that it moves the score.

## A. Closure-side (does not touch the old tier's representation)

| # | Idea | Source | Test | Effort | Prior |
|---|---|---|---|---|---|
| A1 | **Compensating closure**: drop the old tier, retune λ-scale / BETA / young amplitude γ / AGE_OLD so that final-mean errors compensate | Astra Extra High Q3, GPT-Extra-Max H3 | **done**: noold 7.4e-7; λ 0.5–2.0 flat; BETA 0 flat; one-time γ 0.9/1.1/1.2/1.4 → 8.7/6.8/7.0/9.9e-7 (corrected test); age 5/3 → 4.9e-7/1.1e-6 (×0.66 per age) | done | **dead (measured)** |
| A2 | Per-network λ fit from a small MC probe (2^12 samples ≈ 4 units) instead of the fixed table | Fable H3 | on 10 mini MLPs: fit λ_l per network to the probe, compare raw MSE vs the table | K+R | low (author: λ spread only 1–2% after the adaptive rule) |
| A3 | Zero-constant chain: replace the 16-entry λ table by the online observable rule only | own J2-D | env V17_LAM_SCALE sweep already flat; a true zero-constant version needs code | R | low for score, useful for the write-up |
| A4 | Row screening: evaluate old-source slices only on rows whose closure coefficients w2 are appreciable | Astra Ultra #5 | measure the distribution of w2 per layer from the dumps: if 10% of rows carry >90% of Σ w2² it is worth a chain variant | T | low (He-init: all neurons near threshold) |
| A5 | Coupled-closure diagnostic | Astra Extra High | unconfined float64 chain vs V29: MSE 7.6e-10 (3 % of the error) at util 0.287 — the old tier's approximation error is too small to hide a cancellation | done | **closed (measured)** |

## B. Representation-side (old tier)

| # | Idea | Source | Test | Effort | Prior |
|---|---|---|---|---|---|
| B1 | **Metric-side rank screen**: the two "siblings" of U with the P leg on the right, (A∘A)Pᵀ and (P∘P)Mᵀ; if their required ranks sum to > 4 the strict four-query aggregate fit is dead (already effectively dead via pilot D, but the sibling spectra are cheap and complete the picture) | Astra Extra High Q1 | 2 SVDs from the L14 dump | T | closes a loophole |
| B2 | Quenched KL / Wiener-chaos residual: after subtracting the analytic first-Hermite term, PCA the residual query responses under 32–64 resampled W_l; if m ≤ 8 modes hold 99.96% the residual is compressible | GPT-high R2 Q2-2 | resample the next-layer weights on one frozen state (16–32 draws), compute U per draw, PCA | R (~1 h) | low ("my prior is that this fails" — its own author) |
| B3 | Finite-horizon observability (Hankel) rank of the old-source channel across layers 8–16 | Terra Q3, GPT-high Q2-1 | build perturbation responses of future D21 to old-source perturbations at layer l (needs the chain to run from a modified state: hook + multiple runs) | R+K (~3 h) | low-mid; pilot D's cross-source rank (26% for 2 modes) is the instantaneous version and already fails |
| B4 | Second-order chaos / Hessian-state formulation with a deterministic chaos basis m ≤ 40 | GPT-high R2 Q2-3 | project the stored old-source quadratic response on the best m = 32/64/96 directions | T (~30 min) | low (needs m ≪ 100; U's own rank is ~230) |
| B5 | Query-space response packets with data-dependent probes (PCA of the query covariance) transported with the state | Terra R2 §2 | equivalent to the SVD ceiling of pilot C (rank ~230 for 2%) — the ceiling itself says m ≈ 230, too expensive | T (done in effect) | dead by pilot C unless packets can be far fewer than the rank |
| B6 | Pooled static Tucker core at r = 128–150 | Gemini R1 H1 | **Pilot H (21 Sep)**: core route costs n r³ per layer (wins only at r ≤ 150); measured drift at R_OLD2 = 160/128/96: +1.2/+3.0/+7.0e-9 vs −3/−4/−5% util → net worse at every rank; R_OLD 320/256: +1.6/+6.3e-9 → worse | done | **dead (measured)** |
| B7 | Hierarchical Tucker with adaptive per-layer ranks (not confined to the shared basis) | DeepSeek R2 | HOSVD of the pooled core = pilot D (rank 250 for 2%) | T (done) | dead |
| B8 | TAP / cavity update with an explicit finite-n odd response equation | Terra R2 E, Grok | no closed equation exists; would need derivation first | R (days) | very low |

## R3. Round-3 proposals (21 Sep)

| # | Idea | Source | Test | Effort | Prior |
|---|---|---|---|---|---|
| R1 | Cubic source-index coding: q pooled channels with skew codes, unbiased for the cubic slice functional | GPT-Sol R3 §1 | probe_srccode.py on L10/L14 dumps: 250–530 % error (q ≤ 3), 83–130 % (32 channels) | done | **dead (measured)** |
| R2 | Birth-operator compression a_b ≈ diag + rank-k (k ≤ 40) | Astra R3 §5, GPT-Sol R3 §2 | probe_birth_dump.py + probe_birthcomp.py: a_b = diag(w1)·C_off, zero diagonal, r99 310–809; H error 57–73 % at k=32, 16–22 % at k=384 | done | **dead (measured)** |
| R3 | Query-optimal diagonal third leg (GPT variant, fit d at age 4 on the exact query) | GPT-Sol R3 §2 | subsumed: a_b has zero diagonal, the best diagonal surrogate of C_off has ~100 % error | – | dead by structure |

| R4 | Adjoint / linear-response contraction of the slices against the low-rank final sensitivity (own) | own, 21–22 Sep | probe_adjoint_rank/val.py, probe_jac_val.py, probe_d21late.py: sensitivity rank 944→173, response model exact (R² 0.996–0.9999) but the low-rank path carries 14 %; old D21/D3 both above budget at every layer | done | **dead for cost (measured)** |

| R5 | Angular / spherical-input reparameterization (κ4 injection at layer 0 + a₁ rescaling) | GPT-Sol R4 (GitHub: barnobarno666 Phase 8) | Kaggle 10 MLPs: −0.5 % raw, 5/10 wins (wash); control without a₁: +5.7e-8 as the identity predicts | done | **neutral (measured)** — free, optional in the final submission |
| R6 | Query-space row skeleton (ISDF): reusable left basis of the contracted query | GPT-Sol R4 | probe_isdf.py: anchor 56–71 %, transported 79–94 % at m ≤ 384 | done | **dead (measured)** |

| R7 | Float64 chain (precision share of the error) | own | drift vs float32: MSE 2.9e-13 | done | **dead: no precision error** |

| R8 | AAP vs MPP split of the old-source readout (Astra GitHub audit) | Astra R4 | probe_aap_mpp.py: shares 0.36 / 0.75, cos +0.56, both essential, MPP carries the diagonal | done | **diagnostic closed** |

| R9 | Learned per-neuron residual correction (V33; also Koopman/KME/Stein/RMT dictionaries of round 5) | own + GPT R5 | 100 MLPs, CV: ridge −0.3 %, quadratic −0.7 %, GBM −0.5 %; single-feature corr ≤ 0.04 | done | **dead (measured)** |
| R10 | TT-PCE / TT-cross of the network function (GPT R5 gate 3) | GPT R5 | width 32: one ReLU layer already 60–500 % integral error vs MC 1 % | done | **dead (measured)** |

| R11 | Copula + rich marginals (GPT R5 family 9) | GPT R5 | one-step oracle at 1024: worse than Gaussian closure at every layer | done | **dead (measured)** |
| R12 | AGE_OLD2 / QPASS2 knobs | own | Kaggle: all noise-level or worse | done | **dead** |

| R13 | Better κ₄ regeneration law from a Monte-Carlo teacher (per-row weight, κ₃ terms) | own, 22–23 Sep | law confirmed on 2 networks (R² 0.57 → 0.78), but in-chain it is monotonically worse (2.2911 → 2.2988e-8) | done | **dead (measured): λ is a compensator** |

| R14 | Exact κ₄ core instead of the λ law (the whole κ₄ accuracy line) | public record, 504aldo F105 / forum 18218 | end to end in the author's lean chain: λ·C_off 2.23e-8, dense exact core 2.04e-8, no channel 4.20e-8 → ceiling ≈ 9 % raw at +3 dense units per source-layer | done (public) | **dead (cost)** |
| R15 | Learned estimator from a published supervised corpus (HF `keenanpepper/*`: joint features, residual targets, 500k means, cumulant sketches) | HF sweep 23 Sep | their own numbers are Phase-1 scale (5.2e-6 → 1.9e-6); our 100-network residual study (R9) found the per-neuron residual unlearnable | done | **dead (measured)** |

| R16 | Richardson extrapolation in rank (two cheap runs cancel the leading truncation term) | own, 23 Sep | offline on stored predictions: exponent unstable (1.46 vs 2.07), every extrapolation worse than the better single run | done | **dead (measured)** |
| R17 | Per-family Strassen level (hub / C_pre / shared basis at level 6) | own, 23 Sep | hub and C_pre knobs are capped by `min(knob, s_lev)` → no effect; shared basis at L6 = 7.84 GB vs an 8 GB cap for −0.74 % | done | **dead (measured)** |
| R18 | Layer-dependent age gate (confine early-born sources sooner) | own, 23 Sep | every schedule net worse (best: −0.89 % cost for +7.6 % raw); confinement is a permanent state change, not a per-layer perturbation | done | **dead (measured)** |

| R19 | Unbiased Horvitz–Thompson column sampling of the sources (fixed at birth, and redrawn every layer) | own, 23 Sep | halves deterministic screening's error (2.57e-9 vs 5.11e-9 at q = 0.7) but per-layer redraw adds nothing (2.71e-9); column coherence 1.1–1.6 (Fable) | done | **dead: still +12 % raw for −5.6 % C** |
| R20 | Restrict the contraction to the dominant subspace of the downstream query | Fable R6 | residual response 34 / 12 / 5 % at rank 128 / 256 / 384 — the response is carried by short near-isotropic paths | done | **dead (measured)** |
| R21 | Asymmetric contraction rank (legs at 384, contraction at r_c) | Fable R6, implemented as V38 | end to end: util +0.8 % (overhead eats the tier-1-only saving), raw +2.9 % | done | **dead end to end** |

| R22 | Error budget of the chain against the MC teacher | own, 23 Sep | closure exact (9e-11); fresh error per layer = variance 50 % + kappa3 25 % + kappa4 25 %; mean error carried with gain 0.65-0.96 | done | **map made (K3ab)** |
| R23 | Depth multiplier on the kappa4 diagonal (V39) | own, 23 Sep | x1.16-1.26 worse on both teacher networks: the underestimate is in neurons the closure ignores | done | **dead** |
| R24 | kappa4 oracle with the units fixed | own, 23 Sep | **exact kappa4 diagonal alone: 2.29e-8 -> 1.61e-8 (-30 %)**; slices add nothing | done | **LIVE target** |
| R25 | kappa4 diagonal from cheap features / pairwise transport / star sources | own, 23 Sep | features 0.19 -> 0.18; pairwise corr 0.24-0.69; stars right at birth, 10 % at depth | done | dead as such; post-ReLU teacher next |

## C. Sampling / hybrid side (the budget above the floor is unused: V29 sits at 0.245 B, the floor is 0.1 B — but every unit above 0.1 multiplies the score)

| # | Idea | Source | Test | Effort | Prior |
|---|---|---|---|---|---|
| C1 | Blend V29 with a whitened antithetic MC branch (worked for cov-prop: −25% at r = 0.04) | baghyeonsig; own J2-E | Kaggle: compute V29 prediction and an MC-branch prediction (n samples chosen so total ≤ 0.35 B), measure error correlation and the optimal blend on 10 MLPs | K+R (~2 h) | low-mid: MC at 0.1 B has raw ~1e-5 ≫ 2e-8, so the blend weight would be ~0.002 — the MC branch is 500× noisier than V29; gain ≈ 0 |
| C2 | Control variate: use V29's per-layer means as the control for a small MC run (variance reduction needed 33–300× per author) | author item 12 | dead by the author's measurement | – | dead |

## D. Cost-side (inside V29)

| # | Idea | Source | Test | Effort | Prior |
|---|---|---|---|---|---|
| D1 | STRASSEN_FIRST 4 → 5 (first call per worker at full depth) | own | grader-only effect (mean util 0.2542 vs steady 0.2526 → ≤0.6%); residual risk on cold pools | submit | tiny |
| D2 | Strassen L6 only for the young families | own | measured 21 Sep: util 0.2407 (−1.0%), residual 2.00 s (2× L5), 8.10 GB; SB=6/MIN=4 variant: 0.2414, +45% residual, 7.84 GB | done | **dead (residual cap)** |
| D3 | Lower R_OLD 384 → 320 | own | pilot H: 320 → util −5.9%, drift +1.6e-9 (+7.5% raw): score 5.38 → 5.44e-9 | done | **dead (measured)** |
| D4 | Shared-basis range finder with a warm start from the previous basis (fewer dense products at the join) | own (Part K3c) | code change in the join block; verify drift | R | low (author: ≤ 4% headroom, and residual grows) |

## E. Strategy-side

| # | Idea | Source | Note |
|---|---|---|---|
| E1 | Two nominations: one max-score, one conservative (V29m8 as-is) | Overview rules | do at the end |
| E2 | Ask the 8 models a round 3 focused ONLY on A1/A5 (closure compensation) and D2, with pilots A–E as closed | user | prompt ready to write once A1 results are in |

## Priority order for the next 48 h
1. A1 (running) → decide whether compensation exists.
2. D2 (young-only L6): the one cost lever left with a rank-25 payoff; needs a per-family Strassen level knob.
3. B1 + A4 + B4 (cheap, from existing dumps): close loopholes for the write-up.
4. A5 (lean reference on Kaggle): the only experiment that can reveal *systematic* cancellation to exploit.
5. C1 only if A1 shows nothing.
