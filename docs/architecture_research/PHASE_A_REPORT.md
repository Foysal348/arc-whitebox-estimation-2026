# Phase A report: candidate architectures and their falsifiers (26 Sep 2026)

Scope: the Phase A brief of `ARCH_RESEARCH_PLAN.md`. Four candidate architectures were designed, costed at n = 1024, L = 16
and falsified on stored data; each falsifier was run once, with its gate fixed in the script header before the run.
Every number below comes from a script listed next to it (results in `results/summaries/archA_*.json`); numbers marked
**(derived)** are arithmetic from measured unit prices, not runs. Times are BST. Nothing was committed, pushed, run on
Kaggle or submitted.

## 0. Verdicts in one table

| # | Candidate | Extra cost (derived) | Falsifier (script) | Measured | Verdict |
|---|---|---|---|---|---|
| A1 | **P-SSC**: Team Puffi's published stack (factorised K=3 carriers + SSC kappa4 + response memory + neighbour drift corrector) at 1024 x 16 | +36 to +44 u (util 0.24-0.26) | F1a `archA_ssc_mem.py`; F1b `archA_drift_nbr.py` | F1a: SSC on top of the chain, fitted to the consumed-mean target, errG4 x1.006-1.023 (L8-L16, neuron-split); memory mode x1.014. F1b: held-out R^2 <= 0.004 (gate 0.10) | **KILL** |
| A2 | **CPR**: carried pair-response kappa4 mode (budget moved from the old kappa3 tier to a kappa4 memory channel) | +13 u per carried mode, funded by R_OLD 384 -> 320 (-15 u) | `archA_cpr_oracle.py` (teacher-forced, 15 carried modes x 4 readout forms, out-of-layer) | best single mode errG4 x0.99-1.00 of the chain at L12/15/16; best pair x0.98 | **KILL** |
| A3 | **RSR**: role-split kappa4 regeneration (one lambda / dG feeds five uses; give each its own constant; per-neuron transported metric) | 0 u | `archA_teacher_eval.py` + `experiments/variants/archA_rsr.py` on the teacher pair | best split (feed x0.7, dG x1.25): x0.971 MLP #0, x0.998 synthetic (gate x0.97 on both) | **FAIL** (best partial lead) |
| A4 | **PIV**: kappa4 riding the kappa3 source legs (Round-13 pair innovations, queue Q22) | ~0 u (n^2 per source-layer) | `archA_q22_precheck.py` (the queue's own pre-check, generous form) | in-sample errG4 x0.92, out-of-sample x1.05-1.06 | **KILL** (out of sample worse than the chain) |

**Recommendation: none passed; stop the architecture track (its own kill rule). Best partial lead: A3's zero-FLOP role split
(-2.9% / -0.25%), to be settled by one 100-MLP paired Kaggle run inside the Q25 grid. Details in section 5.**

## 1. The baseline and the costing basis

- Baselines: the Phase-A falsifiers use `submissions/v44rfb8` (#332322: adjusted 4.860e-9, raw 2.210e-8, util 0.220 = 225 u;
  1 u = 2^31 FLOPs) or the plain V29 dumps for the kappa4 channel; the current best graded is #332324 (`submissions/v45a5rfb4`,
  4.807e-9, raw 2.137e-8, util 0.225). Nothing below depends on the difference (AGE_OLD / R_FB do not touch the kappa4 channel).
- Unit prices at n = 1024 (from the metered call-site ledger of the shipped chain, `results/summaries/m1_matmul_ledger_callers.json`,
  a script-generated ledger of the V40 chain whose young-tier and C_pre families are unchanged in v44/v45): one dense 1024^3 product in a Strassen L5 family = **0.51 u**
  (young transport: 54.5 u for 108 products); the second half of the block-symmetric sandwich W C W^T = **0.38 u**
  (5.3 u over 14 layers). Hence one carried symmetric n x n mode (U^T H U) costs **0.89 u per layer = 13.4 u** over the 15
  transported layers **(derived)**. A dense (1024 x 1024) x (1024 x r) product costs ~0.51 r/1024 u.
- kappa4 metrics (convention of `opus12_k4_young_blend.py`): errG4 = G4-weighted relative error of the pre-activation kappa4
  diagonal, G4 = (a^2 - 1) phi(a) / (24 s^3) (the weight with which the next mean closure consumes it); dec corr = pattern
  correlation of the excess kurtosis on the decisive neurons |mu/sigma| < 1.5. Chain (plain V29 dump) at L8/12/15/16:
  errG4 0.142/0.193/0.199/0.210, dec corr 0.37/0.34/0.26/0.32. Ceiling of an exact linear carry of the TRUE kappa4 tensor
  (Opus R12, affine, in-sample): errG4 0.129/0.126/0.128/0.136. Oracle value (K3ac): exact kappa4 diagonal = x0.70 final MSE;
  Opus's linear map: 25% of the kappa4 error removed = x0.87.
- Gate used for every kappa4 falsifier (pre-registered): out-of-sample errG4 <= 0.85 x chain at >= 2 of L12/L15/L16 and
  dec corr >= 0.45 there (a 15% cut of the kappa4 error is worth about -8% final MSE on the linear map); kill if > 0.93 x
  chain everywhere.

## 2. Candidates A1, A2, A4 (kappa4 memory family and Puffi's stack)

### A1. P-SSC: Puffi's published stack, reconstructed at 1024 x 16

**Design (as far as the PDF allows).** State per layer: mu (n), C (n x n), kappa3 as factorised carriers, kappa4 as an SSC
response. Puffi's carrier vocabulary maps onto our chain: "shared-middle groups" = our B1 hub blocks Sym(X1 x P x Y1),
"paired groups" = our B3 blocks Sym(M x P x P), "CP columns" = our thin legs; "atoms Sym(v, M)" and the "fresh cross-family
CP columns capped to 4" are small add-ons (F76 showed capped atoms cannot be the main carrier). Per layer: exact linear
transport of all carriers; Wick/Hermite ReLU update; kappa4 = sum_ab Lam_ab Sym(B_a x B_b) with B1 = C, B2 = Q/vbar (six-part
quadratic form of mu, C, kappa3), B3 = vbar I, B4 = gate-weighted C, diagonal read as 3 q^T Lam q with q_a = w^T B_a w;
a signed one-step response memory A_mem = sqrt|beta| D_g (W A W^T) D_g, beta = -0.3; a 31 -> 32 -> 32 -> 1 tanh drift
corrector on 14 per-neuron + 17 neighbour features (top-8 |R_ij| neighbours, kernel averages P_gamma) at layers >= 9.
**Unknown:** Puffi's Phase-2 method (their P2 raw 1.94e-8 @ 0.146 is not described anywhere public), the exact Q
coefficients and Lam tables, the atom/CP-cap scoring rule, and how their carriers cost 1.34x a K3 port at P1 but sit at
0.146 in P2.

**Cost at 1024 x 16 (derived).** Carriers = the chain (225 u; nothing in the PDF is cheaper than our V29 machinery). SSC readouts:
q1 = var and q3 = vbar ||w||^2 are free; q4 needs W D_g C (1 product), B2's C o C and D_m K^T parts one product each:
3 x 0.51 = 1.5 u/layer = **23 u**. Memory: one extra sandwich per remembered mode, ~0.9-1.4 u/layer = **13-21 u**. Corrector
and repair: O(n^2), ~0. Total **~261-269 u (util 0.25-0.26)**: it must cut raw by >= 14-16% just to tie.

**Accuracy argument.** Puffi's response branch was worth -64.6% against a plain K=3 chain at 256 x 32, and refitting Lam to
the consumed mean correction another -13.6%. Our chain already contains SSC's one-mode case (lambda C_off regeneration,
F68), so the question is what the remaining modes, the new fit target, the memory and the corrector add on top.

**Falsifier F1a** (`archA_ssc_mem.py`, 13 s): the four Puffi modes (and the full six-part Q) from the post-ReLU teacher at
layers 3/7/11/14/15, read at the next pre-activation layer, added to the chain's own kappa4 diagonal and fitted to the
G4-weighted residual (teacher - chain), i.e. the consumed-mean target. Results (errG4 relative to the chain):

| model | shared Lam, fit L8+12 -> test L15/L16 | fit L15+16 -> test L8/L12 | per-layer Lam, neuron-split CV (L4/8/12/15/16) |
|---|---|---|---|
| SSC 4 modes (pairs) | 0.198/0.209 (chain 0.199/0.210) | 0.143/0.193 (chain 0.142/0.193) | x0.93 / 1.02 / 1.02 / 1.02 / 1.01 |
| SSC + full Q | 0.198/0.216 | 0.156/0.197 | x0.94 / 1.05 / 1.05 / 1.06 / 1.08 |
| linear q (no products) | 0.199/0.207 | 0.228/0.196 | x0.88 / 0.98 / 1.00 / 1.00 / 0.99 |
| + one-step memory (L16 only, the one layer with a previous post-ReLU teacher) | - | - | x1.014 +- 0.017 (without memory x1.011) |

Only L4 improves (x0.88-0.93), where the kappa4 carries little of the final error (Opus R12 Table 1: ~70% of the kappa4
value sits at layers >= 12). At L8-L16 nothing beats the chain out of sample. Together with Q23 (stand-alone SSC fitted to
the marginal kappa4: +0.05 corr), the SSC kappa4 branch is dead on our chain in every form Puffi published.

**Falsifier F1b** (`archA_drift_nbr.py`, 4 min): Puffi's 31 corrector features computed from the 6e7 teacher state of
MLP #0 (a close proxy of the chain's state), the 13 Q19 chain features, and both; target = the chain's teacher-forced
one-step mean drift (Q19 dump, rms 5-7e-5 at layers 9-16); ridge and the 31 -> 32 -> 32 -> 1 tanh MLP. Held-out R^2:
neuron split 13 feats +0.004/+0.000, 31 feats -0.008/-0.006, 44 feats +0.004/+0.004 (ridge/MLP); held-out layers all
negative (-0.40 to -0.003). In-sample ridge R^2 is 0.008-0.022, i.e. chance level for 13-44 parameters on 4096 rows.
**Gate 0.10: KILL.** The fresh drift is not a function of any local or neighbour statistic of the lower-order state -
consistent with K3v ("the variance error is a random projection of dC").

**Verdict A1: KILL** (both untested components measured, neither moves the chain; the stack would cost +16% util).

### A2. CPR: a carried pair-response kappa4 mode, paid for by a smaller old kappa3 tier

**Design.** Keep the chain; add p = 1-2 carried symmetric matrices H (pre-activation index space), H <- U^T H U + birth
(U = D_Phi W, one block-symmetric sandwich per layer, 0.89 u), births built from n^2 objects the chain already has; read
kappa4_jj = ck4_j + 3 sum_ab Lam_ab H_a,jj H_b,jj (modes H_a in {H, C, I}). This is the first kappa4 memory channel with an
all-distinct (1,1,1,1) sector whose cost does not grow with the number of sources: Sym(H x M) has every sector, and the
carried H holds memory of all past births (D27 had one birth at layer 1 only; Q23/SSC regenerate the modes every layer).
**Cost (derived):** +13.4 u per mode; funded by R_OLD 384 -> 320 (K3j: -15 u at +1.6e-9 raw drift, i.e. +7% raw), so the
design needs >= -10% raw from the kappa4 channel to pay.

**Falsifier** (`archA_cpr_oracle.py`, 75 s, teacher-forced, deliberately generous): 5 birth types (C, C o C, the TEACHER's
(2,2) and (3,1) kappa4 slices, the Gaussian post-ReLU born covariance 1/2 D_w2 (C_off o C_off) D_w2) x 3 decays (1.0 / 0.7 /
0.4), transported with the teacher's Phi at every layer of MLP #0 (15 carried modes), each read in 4 forms (Sym(H x C),
Sym(H x H), Sym(H x I), chain core with H as its metric) plus the var^2 level term; blend fitted G4-weighted on the
residual (teacher - chain) on L8+L12 and tested on L15/L16, and vice versa (60 single features, 45 pairs).

| best out-of-layer | L8 | L12 | L15 | L16 |
|---|---|---|---|---|
| chain errG4 / dec corr | 0.142 / 0.37 | 0.193 / 0.34 | 0.199 / 0.26 | 0.210 / 0.32 |
| best single carried mode (Sym(H x I), H = carried C o C, decay 0.7) | 0.144 / 0.36 | 0.194 / 0.34 | 0.198 / 0.26 | 0.209 / 0.31 |
| best pair (chain core x carried K22 metric + chain core x carried C o C) | 0.138 / 0.43 | 0.190 / 0.38 | 0.197 / 0.29 | 0.207 / 0.35 |

The best of 105 carried constructions cuts errG4 by at most 3% (L8) and 1-2% at depth, against a 15% gate - even with the
TEACHER's own (2,2) slice as the birth. **Verdict A2: KILL.** A carried pair form Sym(H x M) cannot hold the deep kappa4
pattern: with Q23 (regenerated modes), D27 (single-birth core), K3ac (carried star births, ~10% of the level) and D26/A4
(source-coupled forms), every cheap kappa4 carrier form proposed so far has now been measured and failed. The
remaining kappa4 memory is the kappa3 x C birth family, whose births are not CP (per source q a vector x vector x matrix
term), so its readout needs one product per (source, birth layer, target layer): O(L^3) products, ~1100 u **(derived)**.

### A4. PIV: kappa4 riding the kappa3 source legs (Round-13 "carried pair innovations", queue Q22)

**Design.** Per source s two coefficient vectors (gamma_s, eta_s) of a kappa4 memory built on the same transported P leg,
readout q = 6 (Z gamma) o rowsum(Z) + (Z o Z) eta, Z = P o P: ~0.35 u **(derived)**, zero dense products.
**Falsifier** (`archA_q22_precheck.py`, the queue's own "26-scalar pre-check", generous form: scalars fitted directly to
the G4-weighted residual on the chain's captured P legs at L14 (13 sources) and L15 (14 sources)): in sample errG4 x0.92 /
x0.92 (0.205 -> 0.188, 0.199 -> 0.182), dec corr 0.30 -> 0.41 / 0.26 -> 0.34; **out of sample (half/half neuron split,
20 repeats) x1.053 / x1.056, corr 0.30 / 0.21 vs chain 0.30 / 0.25.** The in-sample gain is fitting noise. **Verdict: KILL.**

## 3. Candidate A3. RSR: role-split kappa4 regeneration (zero FLOPs)

**Design.** In the chain one regenerated coefficient lambda (and the transported diagonal dG built with it) serves five
different uses: the kappa4 diagonal in the closures (g4row), the (2,2) use side (wk4m), the (3,1) use side
(wk431 = lambda C_off), the lambda part of the kappa4 -> kappa3 feed (X3 = lambda A d(w1)), and the dG part of the feed.
They were fitted together (LAM table on the dense aug chain, one global 0.95 scale, BETA). K3ab measured that the true (3,1)
slope is ~2 lambda, and F68 that the feed "carries the gain", so the roles plausibly want different constants. A3 gives
each role its own multiplier, plus a switch that replaces the scalar metric METRIC_C = 2 by the per-neuron transported
metric ||w_j||^2 (the diagonal of W^T W). **Cost: 0 u** (scalars; the metric switch is one n^2 row sum).
**Accuracy argument.** A zero-FLOP change of the kappa4 -> kappa3 feed is the only mechanism in our record consistent with
AndreasHad04's -7.2% at identical util (their post names an unpriced kappa4 -> kappa3 feed arm).

**Falsifier** (`archA_teacher_eval.py` + `archA_rsr.py`, paired against the same variant with all multipliers 1, which is
bit-identical to v44rfb8; run at Strassen level 2 to stay under the memory cap while a second chain process was active -
the knobs are FLOP-free, so only the paired ratio matters). Gate: final MSE <= -3% on BOTH teacher networks.

| configuration (6 on MLP #0, then the best 3 on both) | MLP #0 (L16) | synthetic (L15) |
|---|---|---|
| base (2.3873e-8 / 1.8872e-8) | 1.000 | 1.000 |
| feed lambda x1.4 | 1.039 | - |
| (3,1) use side x2 (the raw-slice slope) | 1.125 | - |
| per-neuron metric ||w_j||^2 | 1.003 | - |
| lambda in dG x1.25 | 0.985 | - |
| feed lambda x0.7 | 0.984 | - |
| feed lambda x0.5 | 0.980 | 0.998 |
| **feed lambda x0.7 + dG lambda x1.25** | **0.971** | **0.998** |
| feed lambda x0.5 + dG lambda x1.5 | 1.011 | 1.050 |

Results: `results/summaries/archA_rsr_mlp0.json`, `archA_rsr_pair.json`. The feed wants a smaller lambda and the
diagonal a larger one - the roles do separate - but the best split gives -2.9% on MLP #0 and only -0.25% on the synthetic
net, the same one-network pattern as V31c/FB-scale, and the (3,1) role is sharply optimal at the shared value (x2 costs
+12.5%). **Verdict: FAIL** (sub-threshold; not a kill - it is a zero-FLOP knob with a consistent sign).

## 4. What else was considered and not run (arithmetic closes it)

- kappa3 x C kappa4 source chain (the physics Fable R8 found behind the kernel's depth deficit): births are not CP, readout
  O(L^3) products (~1100 u), see A2.
- Cheaper kappa3 carriers (atoms Sym(v, M) merged over sources, pooled Tucker/CP cores, background-mode mixtures): already
  measured dead (K3e: vector-metric k = 250-300 for 2%; F88/K3j: pooled core rank >= 224 costs more than the tier; F78:
  mixture 2.3e-6). The young tier's 193-product floor (D21) and the Khatri-Rao wall (C11) leave no cost-side representation
  change inside the sum-of-sources family.

## 5. Recommendation for Phase B

**No Phase-A candidate passed its falsifier.** Per the programme's kill rule, stop the new-architecture track now rather
than on 27 Sep evening, and put Phase B's days into the measured cost and constant levers. What this study adds to the
dead list, with numbers:

- **The whole cheap kappa4-memory family is closed** (A1 SSC with memory and consumed-mean fit, A2 carried pair modes with
  105 constructions including the teacher's own (2,2) slice as birth, A4 kappa4 on the kappa3 legs): none cuts the G4-weighted
  kappa4 error by more than 3% out of sample at L8-L16, against 15% needed. The remaining kappa4 memory (kappa3 x C births) is
  O(L^3) products (derived). No known carrier reaches the exact-diagonal oracle (x0.70) at an affordable cost.
- **Local drift correction is closed in its strongest published form** (Puffi's 31-feature neighbour MLP: held-out R^2 0.004).

**Best partial lead (and its next experiment).** A3's role split, feed lambda x0.7 + dG lambda x1.25: -2.9% raw on MLP #0,
-0.25% on the synthetic net, 0 FLOPs. It does not pass, but it is free, so the cheapest decisive test is a 100-MLP paired
Kaggle run of `experiments/variants/archA_rsr.py` with `ARCHA_KFEED=0.7,ARCHA_KDIAG=1.25` (and `ARCHA_KFEED=0.7` alone)
against v45a5rfb4, merged into the Q25 per-block lambda grid that the incremental track is already running (the dG role
overlaps with Q25's lambda blocks, so test them jointly, not twice). Expected (estimate, not measured): -0.5 to -1.5% score (the MLP #0 / synthetic
average), i.e. ~4.74-4.78e-9 from 4.807e-9 (estimate) - a knob gain, not the 3.5e-9 target.

**Where the target could still come from (not architecture):** the direction review's cost programme (Q20 input-side live-
column compaction, measured neutral at Phi < 1e-2, estimated -5 to -8% util) is the largest unexploited measured lever; the
3.5e-9 target needs about -27% score from 4.807e-9, which only a combination of cost trims and knob gains can approach on
this chain. If a new architecture idea appears, its entry gate is now concrete: it must carry kappa3 (2,1)-slice
information at < 1.1 u per old source-layer without dense legs, or improve the G4-weighted kappa4 diagonal by >= 15% out of
sample in a teacher-forced test like `archA_cpr_oracle.py` (75 s).

## 6. Files

Scripts: `experiments/probes/archA_cpr_oracle.py`, `archA_ssc_mem.py`, `archA_drift_nbr.py`, `archA_q22_precheck.py`,
`archA_teacher_eval.py` (a copy of the teacher harness writing `archA_<TAG>.json`), `archA_ledger.py`; variant
`experiments/variants/archA_rsr.py` (v44rfb8 + six zero-FLOP role multipliers and a per-neuron metric switch; all defaults
bit-identical to v44rfb8). Results: `results/summaries/archA_cpr_oracle.json`, `archA_ssc_mem.json`, `archA_drift_nbr.json`,
`archA_q22_precheck.json`, `archA_rsr_*.json`. `archA_ledger.py` (a fresh v44rfb8 ledger) was written but NOT run: a second
chain process (~4.7 GB) was active and the laptop had ~1 GB of free RAM; the unit prices above come from the existing
script-generated ledger, whose young-tier and covariance families are unchanged in v44/v45.
