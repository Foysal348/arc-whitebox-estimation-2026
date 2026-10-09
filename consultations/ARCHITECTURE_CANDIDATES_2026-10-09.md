# Architecture Exploration Plan, 9-17 Oct 2026 (final, ranked)

Team Hack2Publish (foyshal_sk), ARC White-Box Estimation Challenge 2026, Phase 2.
Written 9 Oct 2026. Submissions close 18 Oct 2026, 05:59 Bangladesh time (so 17 Oct is the last working day). Write-up due 24 Oct.

## 0. Where we stand (from the context block)

- Best graded: #334870 v56As_fs **3.6521e-9** (raw 1.950e-8, A-class, ~20k ops). Best safe B: #334869 v56s_fs 3.6723e-9.
- Live LB (8-9 Oct): we are #41 (displayed 3.7e-9). Ranks 1-25: raw 1.4-1.76e-8 at util 0.10-0.16. Ranks 28-50 (public-chain family, incl. us): raw 1.9-2.2e-8 at util 0.16-0.20.
- Grader caps per MLP: 120 s wall, 0.4 s residual (keep warm-call ops <= ~18k for B-class), 8 GB address space.
- Newest ceiling (set C, MLP #0, N=4e6, Gaussian restart 6 layers back, cubic Hermite only at listed ages; `data/kaggle_out/r17_ceiling_c/r17_hermite_family_ceiling_C.json`): all ages 1-6 0.848/0.799/0.817 (L12/L15/L16); ages 2-6 0.709/0.708/0.691; ages 1-3 0.647/0.582/0.637; age 1 0.343/0.239/0.231.
- Inference from set C, re-run by the reviewer (`scratchpad/arch_tests/claude_superpos2.py`): best ADDITIVE sum of separately-born windows is 0.706/0.669/0.695 ({1-3}+{4-6}) and 0.735/0.726/0.706 (1+{2-6}). The joint run beats them by +0.11-0.14 corr. So the deep kappa4 pattern is not additive over birth ages. This is a ceiling for all diagrams; no deployable term has been measured against it yet.

Key facts that bound every candidate:
- The kappa4 oracle (`k4oracle2_target_*.json`, per the kappa4 digest) gives ~33% of the gain to layers <= 11 and ~70% to layers >= 12. **A corr gain at L6-11 is worth less than the same gain at L12-15.** Every gate below has a separate L12-15 condition for this reason.
- Teacher-forced corr shrinks when deployed: feat+star went from 0.69 (LOLO, teacher) to 0.61-0.67 (cross-network, chain features).
- Calibration used for estimates (one-point fit, V53 -> V56; an **estimate**): raw ~ R0 * (1 - 0.29 f(rho)), f = 1 - (1 - rho^2)/(1 - 0.3^2).

`scratchpad/` below means the session scratchpad
`C:/Users/hp/AppData/Local/Temp/claude/d--Machine-Learning-1-Year-Challenge-Kaggle-ARC-White-Box-Estimation-Challenge-2026/f76e7d52-e4d3-487f-9e05-d4df1b9f5ce8/scratchpad`.
Copy the probe scripts into `experiments/probes/` before the write-up, because the temp directory is not durable.

---

## 1. Ranked list

| Rank | Track | Reviewer score | Verdict | Cost | Expected score gain (estimate) | First decisive test |
|---|---|---|---|---|---|---|
| 1 | Depth-conditioned V56 readout map (+ merged: per-block kappa4 level multipliers as extra harness arms) | 7 (level arms: 2) | KEEP (+MERGE) | 0 u, ~0 ops | -0.5 to -1.2% | Laptop component split L6-11 / L12-15, then a 100-MLP harness arm |
| 2 | L16 cubic readout: HTQB star in the last-layer map (stage 1 only) | 7 | KEEP | 0 u, +10-20 ops | -0.5 to -2% | Price L16 with the K4O_LMIN=K4O_LMAX=16 oracle |
| 3 | V57 kappa4 feature bank: rank-32 metric Q2 path (diag-only fallback) | 5 | KEEP | ~1.1 u, +400-450 ops (rank-32); ~0 u, +150-200 ops (diag) | -0.3 to -0.8% | Chain-input path probe on top of the depth map |
| 4 | Lagged-vertex Q1 conversion features (+ merged: cubic-vertex TT term) | 4 (TT: 2) | KEEP, falsifier only (+MERGE) | lag-1 FLOP-neutral to +~4 u; TT +6-10 u | Q1: -1 to -4% (untested); TT: ~0 net | One Kaggle GPU falsifier run, laptop LOLO verdict in 0.5 day |
| 5 | Target-layer row liveness in the transport family (cost-first) | 3 | KEEP | -1 to -2 u realistic | -0.5 to -1% util | Laptop oracle-live-set row gather, measured util delta |
| 6 | Mild kappa3 cost-frontier re-solve on V56 (ledger + R_OLD/R_FB knobs, no AGE_OLD cuts, no oracle) | 3 | KEEP (scoped down) | -3 to -8% util (estimate) | 0 to -1.5% | Ledger (1 h, laptop) + one 100-MLP knob batch |

Ranking rule: reviewer score first, then quality of evidence (cross-network > single-network teacher-forced > ceiling-only > untested). Ranks 1 and 2 tie on score; rank 1 wins on evidence (passed held-out-network tests in both directions on deployed chain features). Ranks 5 and 6 tie on score; rank 5 is exact apart from a measured <= 0.10% accuracy cost, while rank 6 fights the one-for-one law that has held on every base so far.

The two merged items are not separate tracks:
- **Level multipliers** (score 2) become 4 extra arms in rank 1's harness batch.
- **TT births** (score 2) become extra accumulators in rank 4's falsifier run.

---

## 2. Track details

### Rank 1. Depth-conditioned V56 readout map (V57_DEPTH), with level-multiplier arms merged in

**Mechanism.** The V55/V56 corrector maps 8 chain features [s3sq, y21, y21T, c2, |mu/sigma|, chain k4/v^2, HTQB star, 1] to the kappa4-diagonal pattern with one coefficient vector for all layers >= LMIN. The change: beta(l) = b0 + (l/16) b1, fitted jointly (7 extra coefficients). It deploys as a precomputed per-layer coefficient table, so the warm-call dot product and op count do not change. The pattern is still rescaled to the chain's layer-mean level and feeds the mean closure only. Clamp l >= 6 for any layer below the fit range.

*Merged arms (former "score-fitted kappa4 level schedule"):* per-block multipliers m_B on the mean-closure kappa4 level only (pair closure and lambda untouched). Arms: m_B in {0.9, 1.1} for block L16 and for block L11-15 (4 arms).

**Evidence (verified by reviewer re-runs).**
- Cross-network, deployed chain features (`scratchpad/arch_tests/claude_depth_chain.py`; data `data/dumps/k4s_feats.npz`, teachers `data/kaggle_out/kaggle_out_k4pub`, `data/kaggle_out/kaggle_out_k4`), mean L8+:
  - fit synth -> test MLP#0: 0.674 -> 0.719 (L12 0.695 -> 0.719, L15 0.671 -> 0.692)
  - fit MLP#0 -> test synth: 0.613 -> 0.663 (L12 0.633 -> 0.669, L15 0.598 -> 0.626)
- Teacher-forced LOLO on MLP#0 (`scratchpad/arch_tests/claude_depth2.py`, output `scratchpad/claude_perage.json`): 0.694 -> 0.726 mean L8-15.
- **Caveat (reviewer).** The gain is concentrated at L6-11 (e.g. L7 0.637 -> 0.734). In the teacher-forced LOLO the late layers are flat or negative: L12 +0.011, L13 -0.003, L14 -0.001, L15 -0.008. Cross-network L14 gains only +0.006 / +0.016.
- Level arms: kappa4 digest gives teacher pattern at chain level 1.551e-8 vs full teacher diagonal 1.609e-8 vs chain pattern at teacher level 2.783e-8 (worse than none). Neighbouring axes graded flat: Q25 last-block lambda x1.25 0.989/1.001 (v48), V56 last-layer blend x1.25 x0.9999, LAM x0.9 x1.0026 (`results/summaries/kaggle_v56_tuning.txt` and the digest).

**Why not a dead idea.** Q33 extensions (squares/logs/products, carried previous-layer features) failed out of sample; V55 sweep 2 and V56 tuning only rescaled the global vector. This one passes held-out-network tests in both directions on the features the grader actually computes. The level arms isolate the mean-closure level, an axis no sweep has isolated; they piggy-back for free, so no separate days are spent on them.

**Expected gain.** Reviewer-adjusted: **-0.5 to -1.2% score** (estimate), because the gain sits where the oracle puts only ~33% of the value. The candidate's own -1.5 to -2.5% assumed a uniform gain and is probably 2-3x too high. Level arms: 0 to -0.7% (estimate, likely null).

**Cost.** 0 u, ~0 extra ops (table lookup). Memory unchanged.

**Decisive first test.**
1. Component (laptop, ~30 s): refit the depth map pooled on MLP#0 + synth at L >= 6. Report cross-network corr separately for L6-11 and L12-15. Also report per-layer pattern contrast std(pred)/mean(pred) against the teacher's, because the V55 s3sq x0.8 sweep showed that contrast shifts cost score (+1.24%).
2. E2E (Kaggle, one batch): v57d = v56As_fs + coefficient table, against v56As_fs; B-class twin against v56s_fs. Same batch: rank 2's L16-star arm and the 4 level arms.

**Gate.**
- PASS: geo ratio <= 0.993, >= 65/100 wins, residual median within 2%, no MLP worse than x1.02 (reviewer).
- KILL: geo ratio > 0.997 or wins < 55/100 (planner's threshold). The band in between: submit one A-class copy anyway (submit-freely rule) and keep it only if the graded score improves.
- Level arms PASS: geo <= 0.995 with >= 65/100 wins **and** the sign confirmed on a split half (fit on MLPs 0-49, verify on 50-99). Otherwise close the level axis for good.

**Risk.** Fit on only 2 networks; the synth teacher has even layers only (+L15), so L13/L15/L16 extrapolation rests on MLP#0. The depth intercept may shift contrast in an untested direction.

---

### Rank 2. L16 cubic readout: HTQB star in the last-layer map (stage 1 only)

**Mechanism.** V56 computes the star only inside the D21 branch. At the final layer need_d21 = False, so the map falls back to [s3sq, |mu/sigma|, chain, 1] with K4C_BL, and the star never reaches L16 (reviewer confirmed in `experiments/variants/v56_star.py` lines 1412-1444). The change: compute the star at L16 from the existing young slots (Py, Xs = A + P diag(w1 var), per-slot c = -alpha phi/(6 var)) and add it as a 5th last-layer feature. Use the V56b in-place per-slot form in the `bufs['pp']` / `bufs['t']` scratch, with no new (k, n, n) temporaries.

**Evidence (verified, teacher-forced, MLP#0 only).** `scratchpad/arch_tests/claude_l16.py` (teacher from `kaggle/r17/data/r17_data.npz`, plain-chain kappa3/kappa4 from `data/dumps/errbudget_v29plain.npz`):
- L16 last-layer set: 0.352 in-layer, 0.336 fit-on-L13-15
- with the star: 0.528 / 0.517 (+0.18)
- star alone 0.471 (also `results/summaries/round18_htqb.json` star_h6); chain alone 0.324
- Switching the last-layer map off cost x1.0116 on 100 MLPs (`results/summaries/kaggle_v55_sweep1.txt`).

**Why not a dead idea.** The last-layer blend x1.25 (x0.9999) and the LMIN/K4S_AGES knobs rescaled existing features. This adds the cubic-birth feature to the one layer V56 never gave it to.

**Expected gain.** -0.5 to -2% score (estimate). Weakness: the x1.0116 value of the last-layer map came with only a 0.324 -> 0.352 corr change, so level or contrast, not pattern, may drive L16. A +0.18 corr gain need not translate proportionally into MSE. There is no L16-only oracle yet; test (i) supplies it.

**Cost.** 0 u, +10-20 ops (one elementwise star block). Fits B-class.

**Decisive first test.**
1. Price L16 (laptop or Kaggle CPU, 1 run): `k4_oracle_run2.py` with K4O_PART=pattern, K4O_LMIN=16, K4O_LMAX=16, on MLP#0 and synth.
2. Component: extend `kaggle/k4s/data/k4s_dump.py` so the last-layer dump carries the chain-computed star. Fit the 5-coefficient map on chain features at L13-15 of MLP#0 + synth, test on MLP#0 L16. (The synth teacher in `data/kaggle_out/kaggle_out_k4/k4teach` has no L16, so a true cross-network L16 fit is impossible.)
3. Separate arm in rank 1's harness batch, against v56As_fs, plus a `whest run --runner subprocess` memory check.

**Gate.**
- Oracle step PASS: teacher-pattern-at-L16-only gain >= 3% raw. KILL below that.
- Component PASS: L16 corr >= 0.45 (baseline 0.352). KILL below.
- Harness PASS: geo <= 0.996, >= 60/100 wins, clean memory check. KILL: geo > 0.999 or any 8 GB failure that the in-place form cannot fix.

**Risk.** Validated on one network with teacher-forced legs. The young legs and scratch buffers must be valid at trim (need_d21 = False). The V56b 8 GB lesson applies.

---

### Rank 3. V57 kappa4 feature bank: rank-32 metric Q2 path (diag-only fallback)

**Mechanism.** Readout only, no new carried state. For each young dense slot of age <= 3, at layers >= 8, form V = (g_b P) o X from legs the star already reads. Path feature: path_i = sum_a var_a V_ai^2 + sum_m w_m (U_m^T V)_i^2, where (U, w) are the top 32 eigenpairs of the birth layer's C_off, estimated once at birth by a range finder (32+8 columns, 1 power iteration, QR) and stored as n x 32. Standardised by v^2, added to the V55/V56/V57 map, refit on synth (the *_fs recipe), mean closure only. Zero-cost fallback: diagonal metric (w = 0). The kd (kappa3-dressed quadratic vertex) piece is **dropped**: it adds +0.006 over rank-32 h3 (0.722 -> 0.728), which is noise level.

**Evidence (teacher-forced, MLP#0 only, LOLO over L8-15; reviewer recomputed from `scratchpad/path_metric_probe.json` and `scratchpad/k3c_probe.json`).**
- feat+star 0.694 -> +rank-32 path h3 0.722 -> +full path h3 0.727 -> +full path h6 0.737; diagonal-only 0.714 (h3) / 0.717 (h6); rank-32 + kd6 0.728.
- L12: 0.718 -> 0.745 -> 0.751. L14 full: 0.738 (h3) / 0.742 (h6).
- Late-layer rank-32 h3 increments are weak: L12 +0.027, L13 -0.004, L14 +0.029, L15 +0.007.
- corr(rank-32 path, full path) 0.92-0.96. The probe used the exact eigh of the teacher C_off; the range-finder version is untested.

**Why not a dead idea.** SBQM (0.33-0.39) and the pair-innovation carrier (0.36-0.44) died as standalone carriers; the HTQB path alone is 0.02-0.28; carried previous-layer features gave 0 or negative held-out gain. This is a same-layer complement inside the shipped map with a measured held-out increment. The 27 u full path is dead on cost; rank-32 is ~1/25 of that.

**Expected gain.** -0.3 to -0.8% score (reviewer estimate), near harness resolution. May not stack with rank 1, because both gain mostly at L8-11. That is why the test fits it on top of the depth map.

**Cost.** Rank-32: ~1.1 u (util +0.6%), +400-450 ops (estimate), so A-class only (A-class would reach ~20.4k ops, close to the residual cap). Diagonal-only: ~0 u, +150-200 ops (estimate), might fit B-class.

**Decisive first test (laptop, light).** Rebuild the path from chain inputs: chain C_off of the birth layer from `data/dumps/r7_cap_L*.npz`, chain legs A + P diag(w1 var), range finder instead of eigh. Fit on top of rank 1's depth-conditioned map, not on bare feat+star. MLP#0, plus synth if chain dumps exist; otherwise MLP#0 LOLO only, stated as such. Report the diag-only variant alongside.

**Gate.**
- Component PASS: >= +0.02 mean L8-15 **and** >= +0.015 mean L12-15 over the depth map. KILL if L12-15 < +0.01.
- Harness (arm against the V57 depth build) PASS: geo <= 0.995 including the util change, residual median within 2%. KILL: geo > 0.998.
- If only the diag variant passes the component gate, ship diag-only as the B-class candidate.

**Risk.** Every number so far is teacher-forced on one network; deployment has shrunk corr before. Ops headroom.

---

### Rank 4. Lagged-vertex Q1 conversion features, with cubic-vertex TT merged in (falsifier only)

**Mechanism.** At a ReLU vertex m = t-j, the He2 response converts incoming kappa3 into kappa4: kappa(z_a^2, z_b, z_c, z_d) = 2(C_ab T_acd + C_ac T_abd + C_ad T_abc). Diagonal readout q_j(i) = 24 sum_a g_a P_ai X_ai S~_j(i,a), with S~_j(i,a) = kappa3(z_i^t, z_i^t, z_a^m). In the deployed form, S~_1 would come from re-associating the D21 hub (D21 = S_1 M), FLOP-neutral at lag 1 in principle.
*Merged TT term:* the c3-vertex kappa3 x kappa3 birth, tt_j(i) = 12 sum_a c3_a P_ai S~_j(i,a) Y_j(a,i), with Y_j(a,i) = kappa3(z_a^m, z_a^m, z_i^t) (identity verified to 8.1e-16, `r17_curvature_identities.json`).

**Evidence.**
- Set-C interaction gap +0.11-0.14 corr (section 0), verified. This is a K=3 all-diagram envelope, **not** a Q1 or TT measurement.
- One-step carry (`results/summaries/opus12_k4_carry.json`): LIN carry of the true kappa4 0.771/0.785/0.797 -> + age-1 quadratic vertex 0.820/0.855/0.863.
- "Most of QUAD - LIN is conversion" is an inference. The age-1 Gaussian path figure (+0.013) is printed to console only and **unverified**.
- Q1 standalone, true kappa3 (`consultations/round17_verdicts.md` §1): H=2 0.420/0.392/0.439, H=4 0.473/0.447/0.454, H=6 0.442/0.432/0.431, i.e. **0.392-0.473** (the candidate's "0.42-0.47" was wrong). Q1 + Gaussian path was lower: 0.401/0.386/0.428.
- TT: sets A/B (`results/summaries/r17_hermite_family_ceiling_A.json`, `_B.json`): K=3 0.848/0.799/0.817, K=2 0.548/0.449/0.530, orders 0,1,3 0.511/0.498/0.469, roughly the star_h6 0.521/0.489/0.471. The claim that free star-leg contractions are dead (0.693/0.692/0.693 vs 0.694) is console-only and **unverified**.

**Why not a dead idea (and why only a falsifier).** Summed over lags, q_j is exactly the R17 Q1 hub-pair carrier's readout, and the full-rank, sample-exact ceiling was itself ~0.44. So the "died only from rank truncation" argument is false. The one untested question is **in-map complementarity** with feat+star. The star had the same standalone level and still gave V56's -4.4%. That justifies one 0.5-day falsifier, not the 3.5-day hub re-association. TT needs a real extra n^3 product per lag-layer (6-10 u), so it must clear a higher bar.

**Expected gain.** Q1 in-map rho +0.03-0.08 (untested estimate), i.e. -1 to -4% score. Reviewer view: low expected value per day, given the 0.44 standalone cap and possible collinearity with y21/y21T. TT: ~0 net after cost (estimate).

**Cost.** Falsifier: one Kaggle GPU run + laptop LOLO, 0.5 day. If it passes, deployment: lag-1 FLOP-neutral with +1.1-2.4k ops (Strassen), or ~+4 u (~2% util) and ~+10 ops (plain matmul); 3.5 days of hub engineering on the most fragile code (interleaved compacted Strassen, D21 must match to 1e-6). TT: +6-10 u, +3-5% util.

**Decisive first test.** Reuse `experiments/probes/r17_claude/r17_q1_hubpair_falsifier.py` on MLP#0 on Kaggle GPU (--targets 6..16, --H 3, N=2e6). Add one np.savez of the per-(age, target) Q1 diagonal from predict(), and in the same pass the TT contraction (one extra n x n accumulator Y per (m, t) pair, tt_j with teacher c3 and P). Laptop LOLO as in `claude_perage.py`: feat+star+q_1 (+q_2, q_3), then +tt_1.

**Gate.**
- Q1 PASS: mean L8-15 LOLO gain >= +0.03 over 0.694 **and** mean L12-15 gain >= +0.02. KILL otherwise, and close Q1 for good (put it in the write-up as a measured negative).
- TT PASS: >= +0.05 mean L8-15 and >= +0.03 at L12-15 over feat+star+q_1. Below that its 6-10 u cannot pay; KILL.
- If Q1 passes: next check chain-legs D21 equality on one layer (<= 1e-6 rel) before any hub change. Hard deadline: if a Q1 build has not cleared a 100-MLP harness by the end of 15 Oct, stop and do not ship it.

**Risk.** Engineering, ops budget (B-class has no room for an extra Strassen call per layer), collinearity with D21-energy features.

---

### Rank 5. Target-layer row liveness in the transport family (cost-first)

**Mechanism.** Neurons with Phi_l < 1e-5 are zeroed at the wick (w1 := 0) after their output rows have been computed in the young/old leg transport, the A products and the basis transport. Row-gather W_l to the live set, padded to the 32-row quantum (ROWC_Q) as V48 does, and set their own d3 to 0. Note: ROWC_THR already defaults to 1e-2 (`submissions/v56As_fs/estimator.py` lines 1518-1543), so rows of all Phi < 1e-2 neurons are currently computed only for their own D3.

**Evidence.** Dead fractions, MLP#0 teacher (`scratchpad/d3_corrector_and_mc_noise_check.txt`, verified): Phi < 1e-5: 0.4/0.8/1.6/3.5/7.3/9.2% at L6/8/10/12/14/16; Phi < 1e-4: 1.6/3.1/3.4/6.5/10.5/12.8%. B07/Q31: zeroing dead neurons' own d3 costs +0.02%/+0.10% MSE at 1e-5 and +0.37%/+0.52% at 1e-4. Precedents: V48 compaction exact to 6.5e-7, graded -3.7%; V50/V51 -3.7% / -3.4%.

**Why not a dead idea.** B06 and the Q31 always-on hubs removed hub columns, which carry K x C feedback. Q24 dropped D21 rows of live saturated neurons. This touches only output rows of already-wick-zeroed neurons.

**Expected gain.** Realistic ~1-2 u, i.e. 0.5-1% util (reviewer estimate). At ROWC_Q = 32, the L6-L10 dead counts (4-16 rows) round to zero, so savings come only from L12-16. MSE +0.02 to +0.10%.

**Cost.** Phi_l is not known before the transport into layer l (var_l comes from C_pre inside the same batched family). Either move the C slot (~1.27k ops for 0 u per the V52 ledger, pushing to A-class) or use a cheap diagonal var estimate with a margin that shrinks the dead set. 2 days if it proceeds.

**Decisive first test (laptop, one MLP; measures the saving, not the already-known accuracy).** Save per-layer Phi from a normal v56s_fs run. Re-run with an env flag that row-gathers W_l to the oracle live set (Phi >= 1e-5, padded to ROWC_Q) for all transports, using the saved Phi. Report the metered util delta and final MSE delta from flopscope.

**Gate.**
- Oracle-live PASS: util <= -0.8% and MSE <= +0.1%. KILL otherwise; this is an upper bound, so a failure here closes the track.
- Cheap-var variant: report the share of the oracle saving that survives and the op delta. Proceed to a 100-MLP pair only if >= 60% survives with <= +150 ops (planner's threshold).

**Risk.** Lumpy savings from quantum padding; residual time near the cap in A-class.

---

### Rank 6. Mild kappa3 cost-frontier re-solve on V56 (scoped down)

**Mechanism.** (a) Metered per-component ledger on v56As_fs (none exists for V56: `m1_matmul_ledger` is V40-era, `archA_ledger.py` never ran). (b) One 100-MLP knob batch against v56As_fs: R_OLD 320, R_OLD2 192, R_FB 1, and R_OLD 320 + R_FB 1. **No AGE_OLD cuts** (they shrink the star's young-slot coverage; K4S_AGES 4 measured x1.0045 worse) and **no oracle step** (the teacher-pattern oracle does not match the deployed map, corr ~0.65, so it cannot decide the deployed question).

**Evidence.** `results/summaries/kaggle_v56_tuning.txt`: AGE_OLD 6 on V56 gives raw x0.9782 [0.9749-0.9814], 89/100 wins, util 0.1857 -> 0.1926, score x1.0138 (-2.2% raw for +3.7% util, the up-slope). Joint oracle x0.702 (full chain) vs x0.876 (AGE_OLD 2), digest. Unverified and not used: "AGE_OLD 5 -4.8% raw before V55" and "R_OLD 320 x0.9988" (the digest's A13 says 0.993/1.005).

**Why not a dead idea.** "Cut kappa3, recover with kappa4" was closed only for drastic cuts. Base-dependent knob re-tests are the documented revival class (KA §2: 8/8 revived kills were cost or knob levers). The leaders' gap is mostly util (0.10-0.15 vs our ~0.187).

**Expected gain.** 0 to -1.5% score (estimate). Modest probability: the one-for-one law has held on every base, and diminishing returns make the down-slope steeper than the measured up-slope.

**Cost.** Ledger ~1 h on the laptop (also needed for the write-up). One Kaggle batch. 1 day total, then done.

**Decisive first test.** The ledger + knob batch above.

**Gate.** PASS: geo score ratio <= 0.995 with >= 60/100 wins on any arm. KILL: all arms > 0.998. Either way the track closes after this one batch.

---

## 3. Day-by-day schedule (9-17 Oct, at most 3 Kaggle tracks in parallel)

Kaggle slots: **K1** = main harness lane (100-MLP paired CPU runs), **K2** = second harness/knob lane, **K3** = GPU falsifier lane. The laptop does only light checks (component fits on stored data, one-MLP ledger/row-gather runs), never sweeps. Validate every remote script locally first and predict memory/OOM before pushing.

| Day | K1 (harness) | K2 | K3 (GPU) | Laptop (light) | Submissions |
|---|---|---|---|---|---|
| **Thu 9 Oct** (evening) | — | — | Prepare + smoke the Q1/TT falsifier patch locally on a tiny N; launch overnight | R1 component split (L6-11 / L12-15, contrast); build v57d table; R2 step (i) L16 oracle config | — |
| **Fri 10 Oct** | Batch 1: v57d (A) + v57d B-twin, L16-star arm, 4 level arms, all vs v56As_fs / v56s_fs | R6 knob batch (R_OLD 320, R_OLD2 192, R_FB 1, R_OLD 320+R_FB 1) | Q1/TT falsifier (MLP#0, N=2e6, H=3) | R2 step (ii) L16 chain-star dump + fit; R6 ledger (1 h) | Submit any arm that passes its gate (A and B) |
| **Sat 11 Oct** | Read batch 1; level arms: split-half check | Read R6 batch; R6 closes | Read falsifier output | **Q1/TT LOLO verdict** (0.5 day); R3 chain-input path probe on top of depth map; R5 oracle-live row-gather run | Submit V57 combo (depth + L16 star [+ level if passed]) |
| **Sun 12 Oct** | Batch 2: R3 path arm (rank-32 and diag) vs V57 combo | If Q1 passed: one-layer chain-legs D21 equality check; else R5 cheap-var 100-MLP pair (if gate passed) | free (or Q1 synth cross-check if passed) | R3 implementation in v57 (A) + diag in B; `whest run --runner subprocess` memory/ops checks | Submit R3 winners |
| **Mon 13 Oct** | Read batch 2 | Q1 lag-1 hub build (plain-matmul variant first) **or** R5 implementation | — | Ops/residual audit of the combined build | — |
| **Tue 14 Oct** | Batch 3: Q1 build arm (if alive) **or** R5 arm, vs best V57 | Re-run of any borderline arm with a fresh seed | — | Fix-ups from batch 3 | Submit passing arms |
| **Wed 15 Oct** | Read batch 3. **Hard stop for Q1**: no 100-MLP pass by end of day means it is not shipped | Final combined candidate: everything that passed, A and B | — | Write-up notes: ledger, negatives (Q1/TT, level, knobs) | Submit combined A/B |
| **Thu 16 Oct** | Confirmation run of the final combined build (second seed / full set) | Contingency lane: re-tune the map coefficients of the final feature set on synth (*_fs recipe) | — | Package checks (8 GB smoke, residual) | Submit final-candidate variants |
| **Fri 17 Oct** | Freeze. Only re-runs of already-validated packages | — | — | Final package verification | Last submissions well before 18 Oct, 05:59 Bangladesh time |

Stacking rule: ranks 1 and 2 are independent features of the same map and ship together if both pass. Rank 3 is fitted on top of rank 1. Rank 4 (if alive) and rank 5 change the chain or its cost and get their own arms against the best V57 build.

---

## 4. Dropped items (with one-line reasons)

No candidate was rejected outright by the reviewer. These parts were dropped or folded in:

- **L16 stage 2: decisive-row Q1 at the last layer.** Overlaps the R17 Q1 kill (standalone ~0.39-0.47); revisit only if rank 4's falsifier passes.
- **kd (kappa3-dressed quadratic vertex) in the feature bank.** +0.006 over rank-32 path (0.722 -> 0.728), noise level.
- **Full HTQB path and the full kappa3-slice K3 x C version.** 27 u and ~54 u respectively; dead on cost (kc6 adds only +0.007).
- **Lagged-vertex hub re-association as an immediate 3.5-day build.** Replaced by a 0.5-day falsifier; the full-rank Q1 ceiling (0.442/0.432/0.431) already capped it standalone.
- **TT births as a separate track.** It depends on Q1 infrastructure, costs 6-10 u, and has ~0 expected net; merged into the falsifier run.
- **Level-schedule stored-data pre-check via k4_oracle_run2.py.** Broken: it injects V56-chain features into the plain Gaussian chain, so it measures neither; replaced by harness arms.
- **Level multipliers as a separate track.** Every neighbouring axis (lambda, blend, chain coefficient) graded flat; reduced to 4 free arms in batch 1.
- **AGE_OLD cuts in the frontier re-solve.** They shrink the star's age coverage (K4S_AGES 4 x1.0045 worse); AGE_OLD 6 measured score x1.0138.
- **Teacher-pattern oracle step for mild kappa3 cuts.** The deployed map (corr ~0.65) does not match the teacher pattern, so it cannot predict the deployed outcome; the harness measures it directly.
- **Row-liveness accuracy-only first test.** It re-measures the known B07 accuracy and says nothing about the unknown FLOP saving; replaced by the oracle-live util measurement.
- **Phi < 1e-4 liveness threshold.** +0.37%/+0.52% MSE from own-d3 loss (B07/Q31) likely eats the extra saving; 1e-5 only, unless the oracle run shows otherwise.

## 5. Status labels

- Measured (files cited above): all corr ceilings and LOLO numbers in sections 0-2, the dead-neuron fractions, the AGE_OLD 6 harness result, the oracle ratios.
- Unverified (console-only or not found in files): age-1 Gaussian path +0.013; free star-leg contractions 0.693/0.692/0.693; "AGE_OLD 5 -4.8% raw pre-V55"; "R_OLD 320 x0.9988".
- Estimates (no measurement yet): every expected score gain, every u/ops cost for unbuilt code, the raw-vs-rho calibration, the range-finder version of the path, and the planner's KILL thresholds marked as such.
