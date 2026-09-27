# Kill audit: which of the killed ideas are really dead? (27 Sep 2026)

**Question.** Between 20 and 26 Sep we killed about 60 ideas. Were some of them killed unfairly? Two ways a kill can be unfair:
- an immature prototype was compared with our tuned chain;
- the idea was tested at only one setting, or on a mix of sub-populations.

**Sources read in full:**
- `docs/experiment_queue/queue.json`: all 51 items, including every one marked dead / fail.
- `docs/moonshot/Moonshot_Blueprint.html`
- `docs/architecture_research/PHASE_A_REPORT.md` and `ARCH_RESEARCH_PLAN.md`
- `consultations/moonshot_ideas_audit_2026-09-26.md` and `direction_review_2026-09-25.md`
- `writeup/WRITEUP_DRAFT_v3.md`, sections 4–8 and 11–13
- `docs/build_notes.py`, sections K1–K3ar

**Sources used for cross-checks:**
- `writeup/UNTESTED_IDEAS.md`
- `consultations/conversation_audit_2026-09-25.md`, items D-1 to D-14
- The stored result files in `results/summaries/` that the sources cite

**What this audit did not do.** No experiment was run and no other file was changed.

**Where numbers come from.** Numbers are quoted as the sources give them. **(derived)** marks simple arithmetic on stored result files; the file is named. **unverified** marks anything not found in the sources.

---

## 1. Method

**Conventions.**
- "M0 / syn" = the paired teacher networks: public MLP #0 and the synthetic 1024×16 He network. Numbers like "+1.1% / +1.7%" are MSE changes on M0 / syn.
- "score" = MSE × util.
- u = 2³¹ FLOPs. Our current util of ~0.193 is about 197 u.
- The local teacher harness costs about 2 min per configuration per network (`ARCH_RESEARCH_PLAN.md`).
- A Kaggle 100-MLP paired run takes about 3 h per variant.

**Kill type:**
- **C** — ceiling or oracle. Even a perfect version misses the needed gain. This includes exact counts; **C (count)** means counted, not run.
- **P** — component test. One component against the same component of the chain: one step, teacher-forced, or an upper-bound fit.
- **E** — end-to-end prototype scored against a chain.
- **K** — knob or threshold setting.

**Breadth:** how many settings were tried, how many networks, and whether sub-populations were mixed.

**Fidelity:** for other teams' or AIs' methods, whether the tested form was the proposer's form or a simplified one.

**Verdicts:**
- **DEAD** — a fair kill.
- **PARKED** — could be revived. This covers kill type E with a margin small enough that maturity or tuning could close it, a single-setting or MSE-only K kill, or a fidelity risk.
- **UNCLEAR** — the deciding split or test was never run.
- **REVIVED** — a kill that later measurements have already overturned. These are kept as calibration: they show where unfair kills came from.

**Fairness rules applied:**
1. **The maturity gap is small on accuracy and large on cost.**
   - Public V29 graded raw 2.133e-8 at util 0.254 (#331681).
   - Our safe base v52safeB graded raw 2.16345e-8 at util 0.19273 (#332566). Our best raw was 2.0771e-8 (#332318).
   - So our "tuned chain" is **not more accurate than V29**: raw −2.6% to +1.4%. It is ~24% cheaper (derived: 0.19273 / 0.254). The cheapening is mostly engineering any architecture could adopt: Strassen constants, the transport join, compaction, basis-side Strassen.
   - An E-type accuracy kill is therefore fair when the prototype's deficit is well above ~3%.
   - A cost comparison should credit the transferable engineering to the newcomer. The fair baseline for a new architecture is V29 plus the transferable trims.
2. **For a cost lever, the gate must be score, not MSE.** An MSE-only gate such as "≤ +0.3% MSE" wrongly kills a lever whose util saving exceeds its MSE cost.
3. **Several P-type gates favoured the newcomer.** The newcomer was teacher-forced, or fitted in-sample, while the chain ran free. A clear failure there is therefore strong evidence.
4. **The κ4 add-ons were inserted into a chain whose constants are co-adapted.** In K3ac, the TRUE (2,2) slice alone gives ×1.221 and the true (3,1) slice alone gives ×1.027. FB 2.0, the exact Edgeworth value, gives +3.4% / +3.1%. Add-ons tested without a joint re-tune of λ / β / FB are handicapped.
5. **Single-network evidence** is acceptable when the margin is ≥ 5×. It is a risk only for small margins, e.g. the cheap Q8 run and the κ4-diagonal and variance channel oracles (conversation audit D-3).

## 2. Findings in brief

- **74 entries** (near-duplicates grouped): **58 DEAD, 6 PARKED, 2 UNCLEAR, 8 REVIVED**.
- **All 8 revived kills are cost or knob levers.** None is a new architecture. The unfair kills came from five mechanisms:
  - **(a) MSE-only gates on cost levers.**
    - Transport join: killed on +6.3% / +1.0% MSE. The same runs score ×0.992 / ×0.943 (derived, `teacher_eval_v41join.json`).
    - D21 rows of dead neurons at Φ < 1e-2: killed on +1.1% / +1.7% MSE. They score 0.969 / 0.968.
  - **(b) Mixed sub-populations.**
    - Rows and columns tested together.
    - Dead and always-on neurons tested together.
    - LAM ×0.9 bundled with BETA 1.2.
  - **(c) Only one direction tested:** R_FB 128. R_FB 4 later won.
  - **(d) Base dependence:** AGE_OLD 5 lost before the transport join and won after it.
  - **(e) Prior-only kills:** "He-init: all neurons near threshold".
- **The architecture and representation kills survive the audit.** Every one missed its gate by a wide margin, usually on generous teacher-forced tests against the plain V29 chain or dump:
  - DHMP: 36–40% vs a 2.5% gate.
  - Diagonal closures: 22–99%.
  - Particles: 20–50× worse.
  - SSC and CPR: at most 3% κ4-error cut vs a 15% gate.
  - Maturity bias does not explain these kills.
- **What is left is thin but real.**
  - Two compaction axes were never judged on score: always-on neurons, and dead neurons' own d3.
  - One Round-13 κ4 carrier was tested only in a simplified (scalar) form.
  - Two κ4 add-ons were never re-tuned jointly with the chain's constants or metered.
  - One pair of variants was lost to the grader's residual cap, not to accuracy.

## 3. The audit table

| Idea | Source / date | Kill evidence (numbers as in the source) | Type | Breadth | Fidelity | Verdict | Cheapest fair re-test |
|---|---|---|---|---|---|---|---|
| **A. Carrying old κ3 more cheaply / new representations (cost side)** | | | | | | | |
| **A01** Compress the Hadamard products of source legs | Pilot A, K3b, 21 Sep | A∘P r90 / r99 = 358–504 / 681–778 of 1024 directions, flat noise-like spectrum (top sv 0.09–0.20 of the Frobenius norm); the legs alone have r99 96–504 | C | 3 layers × all source groups; 1 local MLP | own | DEAD | — |
| **A02** Omit or re-form old content late (keep old only in the last k layers; drop old at one layer; drop one birth; "young-tier shadowing") | Pilot B, K3c, 21 Sep (Fable R1 C1, Gemini) | Keep-last 1/2/3: 4.85e-7 / 4.17e-7 / 3.16e-7 vs chain error 2.1e-8. Dropping old at one layer, L10–L15: 5.2e-8–8.2e-8. Dropping one birth: 1.3–2.7e-7 (flat) | C | 26 runs, 1 local MLP | own (audited SOUND, K3o) | DEAD | — |
| **A03** Cheap surrogates of the contracted old query (isotropic trace, diagonal Hermite, TensorSketch); JL sketch and birth-index Grams rejected on paper | Pilot C, K3d/K3e, 21 Sep | Error at L14 / L10: trace 98.2% / 98.9%, Hermite 95.4% / 97.5%, TensorSketch d = 2048 790% / 810% (2% needs d ≈ 1e8) | C / P | 4 sketch widths, 2 layers | faithful; audited SOUND | DEAD | — |
| **A04** Aggregate / pooled states of old κ3: vector-metric sym3(G⊗v), pooled Tucker / TT / HT core, aggregate rank-k CP, quadratic-chaos Galerkin, second-order chaos basis | Pilots D and H (K3e, K3j), 21 Sep; D20 (Round 11); QCG (Sol R9); board B4, B6, B7 | Pooled unfolding rank 32/128/256 → 35 / 7.8 / 1.6%. Vector-metric k = 1/8/32/128 → 89/80/60/20% (L14). Pooled core at r = 128: −6% util for +14% raw. Public rank-4608 CP: 1.07e-7. QCG is an n-term CP; F76's cap at 4n leaves 39% D21 error (arithmetic, not run) | C | optimal-truncation ceilings at many ranks | the ceiling covers any implementation | DEAD | — |
| **A05** Slice-to-slice recursions (Gemini H2; Copilot RSC incl. long memory; DeepSeek sparse neighbour) | K3g, 21 Sep; D18, D7, 24 Sep | H2: R² 0.030 / 0.047. RSC from the true state, fitted in-sample: D21 35.9%, d3 21.6% (chain 6.6% / 4.7%). Long-memory RSC: 35.9% / 21.4% | C (generous) | 2 layer pairs; L14–16 | faithful | DEAD | — |
| **A06** Cubic source-index coding | R1, GPT Sol R3, 21 Sep | H error q = 1: 388% / 527%; 32 channels: 83% / 130% | P | q = 1…32, 2 layers | faithful | DEAD | — |
| **A07** Birth-operator compression a_b ≈ diag + rank k | R2 / R3, Astra and Sol R3, 21 Sep | a_b = diag(w1)C_off exactly (zero diagonal, r99 310–809). H error: k = 32 73% / 57%, k = 384 22% / 16% | C | k = 16…384 | faithful | DEAD | — |
| **A08** Adjoint / low-rank-sensitivity contraction; query-subspace restriction; Hankel observability; quenched-chaos residual | R4 / K3m, 21 Sep; R20 (Fable R6); board B2 / B3 | Sensitivity rank 944 → 173, but the low-rank path carries only 14% of the response. Subspace residual 34 / 12 / 5% at rank 128 / 256 / 384. B2 and B3 never run; B3 is answered by A17 | C | validated response model (R² 0.996–0.9999) | own | DEAD | — |
| **A09** ISDF / query-space row skeleton (also dNTC landmarks) | R6, GPT Sol R4, 22 Sep | Oracle basis 2.2% / 3.6% at m = 256, but anchor 56–71%, two sources 37–59%, transported basis 79–94% | P | m = 128…384, 4 layers, 4 basis choices | faithful | DEAD | — |
| **A10** Birth-column screening by energy (deterministic masks; Horvitz–Thompson sampling) | Pilot F (K3h), R19, 21–23 Sep | Every schedule is net worse. Deep q = 0.9: −1% cost, raw +4%. q = 0.7: −5.6% cost, raw +25–30%. HT halves the drift (2.57e-9 vs 5.11e-9) but is still +12% raw for −5.6% C | K | 7 schedules + 3 HT exponents | own | DEAD | — (the Φ criterion that later worked gives exact zeros, a different lever: B01) |
| **A11** Asymmetric contraction rank (V38) | R21, Fable R6, 23 Sep | End to end util +0.8%, raw +2.9% (the dump predicted −6.9% C). The tier-1-only saving (~4.6 u, 1.8%) is eaten by the per-layer eigendecomposition | E (with ceiling) | 1 setting, 1 local MLP | own | DEAD | — (ceiling 1.8%) |
| **A12** Layer-dependent age gate (V36) | R18, own, 23 Sep | e3s8: util −0.89% at +7.6% raw (score ×1.07). e2s8: −2.9% at drift 9.9e-9. e2s12: −7.5% at 1.4e-8. e1s6: −3.8% at 5.1e-8 | K | 4 schedules, 1 local MLP, V29 base | own | DEAD | — |
| **A13** Lower basis ranks / pooled core at r ≤ 150 (R_OLD, R_OLD2) | Pilot H (K3j), 21 Sep; Q21; v49 re-sweep, 26 Sep | R_OLD2 160/128/96: drift +1.2 / +3.0 / +7.0e-9 for −3 / −4 / −5% cost. R_OLD 320/256: +1.6 / +6.3e-9 for −6 / −12%. 100 MLPs: R_OLD 448 ×1.0113, R_OLD2 192 ×1.0003. v49: 448 1.020 / 1.016, 320 0.993 / 1.005, R_OLD2 256 1.015 / 0.998, 192 1.000 / 1.002 | K | ≥ 10 settings, 3 bases, 100 MLPs | own | DEAD | — (R_OLD 320's grader failure: B10) |
| **A14** Compensating closure without the old tier | K3k / K3o, board A1, 21–22 Sep | No old tier: 7.43e-7 (32×). λ 0.5–2.0: 7.45–7.56e-7. BETA 0: 7.42e-7. One-time γ 0.9 / 1.1 / 1.2 / 1.4: 8.70 / 6.83 / 6.98 / 9.91e-7. Age 5 / 3: 4.89e-7 / 1.10e-6 | C | 10 MLPs, 5 knob families; the flawed γ test was redone | own (audited) | DEAD | — |
| **A15** Young tier at half cost (A = P·a_b, shared propagator, closed forms, batching, contraction-order search, young-only speed-up) | Q17 (Round 10); D21 (Sonnet 5); D10 (Astra R9), 24–25 Sep | 193 square products (98.4 u core) in every order. Reverse suffix +75%, per-age chains +150%, batching FLOP-neutral. Removing all 116 u of the young tier still leaves 144 u | C (exact count) | exhaustive order search + 4 outside answers | faithful | DEAD | — |
| **A16** Explicit κ3 tensor; reverse / lazy cubic circuit | D10 (Astra R9); D15 (Sol), 24 Sep | Packed tensor 0.72 GB, ~1567 u / layer. Lazy circuit: ~480 products ≈ 245 u vs the chain's 249 u. Counted, never run; "not a proven lower bound" | C (count) | — | not built | DEAD | — |
| **A17** Short or compressed κ3 memory: FHQR-2 horizon, age-mode projection, shift-stable carrier | Q18a, D13, D14 (Sol R9), 24 Sep | Keeping ages ≤ h, error at L15: h2 84%, h4 65%, h6 46%, h8 31%, h10 16%. Age modes r = 1/4/6: 12.2 / 10.3 / 8.5%. Shift ε 2.9% (gate 2%); carrier + birth 13.9% | C | h = 2…10, r = 1…6, 4 layers | faithful | DEAD | — |
| **A18** Source-free DHMP (dressed tree rebuilds the all-distinct κ3) | D12, Sol R9, 24 Sep | Teacher-forced next-D21 error 40.4 / 38.4 / 36.4 / 36.7 / 36.3% (L4–L16); gate 2.5%, chain 4–7%. Tree vs true all-distinct part: corr 0.43–0.54 | P (teacher-forced) | 5 layers, M0; dressed and bare edges | faithful | DEAD | — |
| **A19** Channel-history state + nonlinear memory closure (Blueprint M1) | D16, Astra Round 11 (two answers) | The proposer could not give an update rule ("honest no"); injection argument; DHMP shows that half of the all-distinct content is memory | — (no mechanism) | — | n/a | DEAD | — (nothing to test until an update rule exists) |
| **A20** Diagonal-only κ3/κ4 transport, low-rank(+diag) covariance, rank-1 κ3 | D17, SciMaster Round 11, 24 Sep | One step from the TRUE state, next d3: 90–99% (independence), 83–98% (+ tree), 22–29% (exact repeated-index part + tree; chain 3.2–4.7%). Rank-1 D21: 35–49%. Diag + rank-64 C: 12–33% (bar 0.7%) | P | 5 layers, r = 4…256 | faithful (SciMaster's "1.18× prototype" was fabricated; disregarded) | DEAD | — |
| **A21** Deterministic moment-matched particles (dNTC) | D19, SciMaster Round 11 | Free run at L15, cross-polytope 2048: var 3.0%, C 16%, d3 47%, K4d 183%, D21 74%. K = 448: var 6.7%, D21 160%. Chain: var 0.14%, D21 6.6% | E | 3 designs, K = 448…4096 | proposer's size + larger | DEAD | — (margin 20–50× ≫ maturity gap) |
| **A22** Deeper Strassen: L6 (all, young-only, shared basis, per family), L7, memory-saving L6 | K2, R17, board D2, 21–23 Sep; D25, 25 Sep | L6 on all families: util −4.7% but 8.13 GB commit (cap 8 GB). Young-only L6: −1.0% at 8.10 GB, 2× residual. Shared-basis L6: −0.74% at 7.84 GB, +31–45% residual. L7: OOM. Hub / C_pre knobs are capped, so no effect | C (hard caps) | 5 variants | own | DEAD | — (Q30: residual ∝ op count makes it worse) |
| **A23** Precision tricks (float64 packing, float16) | R7, Astra R10, 22–24 Sep | float64 drift 2.9e-13 (no precision error in the chain). float16 is billed as float32, float64 at 2× | C | — | own | DEAD | — |
| **A24** Symmetric / alias pricing ("crumbs of 3–5%") | Moonshot audit C14 (DEAD-SOUND), 26 Sep | Killed as crumbs. Then aliased joiner Grams + basis-side Strassen shipped: util −4.3%, graded #332401 4.4157e-9 (−4.6%). The Grams' separate share is not reported | K (judged small) | desk estimate | own | REVIVED | — (part of v49's Strassen was later removed for the residual cap: v52safeB) |
| **A25a** Join without the range finder ("transport") | D11, own V41, 24 Sep | Killed on MSE: util −6.6%, MSE +6.3% / +1.0%. By score the same runs give ×0.992 / ×0.943 (derived, `teacher_eval_v41join.json`). Revived as Q16: 100 MLPs ×0.959 ± 0.003; graded #332257 −3.4% | K (MSE-only gate) | 1 setting, 2 networks | own | REVIVED | — |
| **A25b** Other join variants: aug16 / aug32 / aug64 / aug96 | D11, Q16, 24–25 Sep | aug16 / 32: score +1.3% / +0.8%. aug64 on 100 MLPs: ×0.9997 (48 / 100 better). aug96 costs more | K | 4 widths; 100 MLPs for aug64 | own | DEAD | — |
| **A26** Richardson extrapolation in rank (also: averaging two cheap runs) | R16, own, 23 Sep | Fitted exponent 1.46 vs 2.07. Best extrapolation 3.3e-9 vs 1.2e-9 for r = 160 alone | C | 3 ranks | own | DEAD | — |
| **B. Dead / always-on neuron compaction (the Q20 family)** | | | | | | | |
| **B01** Drop the D21 rows and columns of dead neurons (Φ < 1e-2) | Q20 mode 3, 25 Sep; Q29 oracle on v49, 26 Sep | Killed: rows+cols +1.65% / +2.59%; rows alone +1.1% / +1.7% (MSE). Split and scored:<br>• columns +0.06% / +0.9% → v50 graded #332404 4.2512e-9 (−3.7%)<br>• rows of very dead neurons (Φ < 1e-3) −0.13% / −0.06% → v51 at 3e-3 graded #332418 4.1081e-9 (−3.4%); at 7e-3 #332420 4.0902e-9 (a further −0.44%)<br>• rows at 1e-2: score 0.969 / 0.968 | K (rows+cols mixed, 1 threshold, MSE-only) | — | own | REVIVED | — |
| **B02** Row screening by closure coefficient w2 | Board A4 (Astra Ultra #5), 21 Sep | Never run: prior "low (He-init: all neurons near threshold)". Contradicted by teacher Φ statistics (Φ < 1e-3 for 6 / 12 / 17% at L8 / 12 / 16) and by B01's shipped rows | none (prior only) | 0 | — | REVIVED | — |
| **B03** Raise the input-row threshold (w1 := 0) above 1e-2 | Q20 3e-2 kill, 25 Sep; v48 threshold sweep, 26 Sep | Mode 9 (w1 + dead birth columns) at 3e-2: +8.5% / +7.9%. w1 alone on v48 (`teacher_eval_v48_thr.json`) at 0.02 / 0.03 / 0.05: score ×1.010 / ×1.009, ×1.031 / ×1.033, ×0.997 / ×1.074 vs 0.01 (derived from the stored MSE and util) | K | 4 thresholds, score-based | own | DEAD | — (the queue's Q20 "next" line is stale: this sweep was run) |
| **B04** Drop always-on neurons' birth columns (w2 := 0 for Φ > 1 − thr), then compact | Q20 mode 25, 25 Sep; direction review | Only ever run combined with dead neurons. Mode 25 at 1e-2: +0.77% / −0.79% MSE; relative to mode 9 alone +1.15% / −0.20% (derived). At 3e-2: +11.4% / +13.0%. The oracle cannot credit util. v48–v52 compact dead neurons only | K | 2 thresholds, never isolated, MSE-only | the oracle zeroes w2 only; a real compaction must remove the columns | PARKED | shortlist #1 |
| **B05** Drop D21 rows of far-from-threshold neurons, \|α\| > 3 | Q24 (moonshot audit), 26 Sep | \|α\| > 3: +6.5% / +19.6%. \|α\| > 2.5: +42% / +60%. Dead-only rows at Φ < 1e-3 (α < −3.09) are free (−0.13% / −0.06% on v50). The damage is therefore attributed to the always-on half by inference, never measured | K | 2 thresholds, dead + always-on mixed | own | UNCLEAR | Split the mask into α < −3 and α > +3 on the teacher pair: 2 configs × 2 networks × ~2 min |
| **B06** Remove dead neurons as hubs of the new source (P columns := 0) | v48_coldrop / coldrop2, 26 Sep | Φ < 1e-2: ×6.40 / ×6.74. Φ < 1e-4: ×2.32 / ×2.23. Φ < 1e-5: ×1.48 / ×1.79 | K | 3 thresholds | own | DEAD | — |
| **B07** Zero the dead neurons' own d3 (this would allow output-row compaction of the transports) | Q20 mode 4, 25 Sep; v51_d3mid, 26 Sep | At all layers (1e-2): ×1.79 / ×1.62. Intermediate layers only: Φ < 1e-3 +0.3% / +2.9%; Φ < 1e-2 ×1.79 / ×1.60. The saving from the output-row compaction it enables was never estimated | K (MSE-only; cost not credited) | 2 thresholds for the intermediate-only form | own | PARKED | shortlist #2 |
| **B08** Zero the dead rows / columns of the transported C (or only its dead off-diagonals) | v48_coldrop, v50_cdiag, 26 Sep | ×1.59 / ×1.46; off-diagonals only ×1.21 / ×1.15 | K | 2 forms | own | DEAD | — |
| **B09** Raise the D21-column threshold above 1e-2 | Q29, 26 Sep | Score at 0.02: 1.004 / 0.995; 0.03: 1.016 / 0.989; 0.05: 1.058 / 1.002 | K | 3 thresholds, score-based | own | DEAD | — (flat; the two networks disagree at 0.02–0.03) |
| **B10** Variants lost to the grader's 0.4 s residual cap: R_OLD 320 on v49 (#332402); D21-row threshold 1e-2 (#332425) | Q28–Q30, 26–27 Sep | Graded 0.0434 / 0.0204, i.e. ~12% / ~6% of MLPs capped (op count, not accuracy). Teacher: R_OLD 320 0.993 / 1.005; row 1e-2 0.969 / 0.968 vs 7e-3 0.973 / 0.961 | K (infrastructure kill) | 1 setting each | own | PARKED (≈ 0 upside) | Re-run on v52safeB within the ≤ ~18k warm-call-op budget, with the op ledger. A `submissions/v52Br352/` package exists; its result is not in the sources (unverified) |
| **C. κ4 channel (the exact κ4 diagonal is worth ×0.699–0.702, M0 only)** | | | | | | | |
| **C01** Astra quartic κ4 kernel (+ Sol's "3+2" K3+C term) | Q1, Astra R7, 23 Sep | End to end:<br>• A = 0.5, S = 1.5: −6.4% / −2.6% @ ~42 u<br>• A = 0.3, S = 1.25: −5.0% / −1.5%<br>• decisive neurons only: −5.2% / +0.3% @ ~22 u<br>• L8–16: −3.5% / −1.1% @ ~11.5 u<br>• L11–16: −2.8% / −0.2% @ ~7 u<br>Costs derived, not metered. Sol's term adds only +0.01–0.04 decisive corr | E (add-on on the plain chain; λ / β / FB not re-tuned) | 2 blends × 4 restrictions × 2 networks | own implementation of Astra R7 | PARKED | shortlist #4 |
| **C02** K3×C (1,1,1,1) "3+2" term on the chain's own sources (full and cheap versions) | Q8, Fable R8, 24 Sep | Full: −8.8% (2.0893e-8) @ ~140 u (M0).<br>Cheap (L8–16, decisive neurons, 2 sources): −4.2% @ ~35 u classical / ~13 u Strassen.<br>All layers: −6.25% @ ~52 u.<br>Same restriction without the term: −3.5%, so the term itself adds ×0.993 (derived).<br>The synthetic run was lost at a session restart | E | 3 settings, M0 only | own (Fable's derivation; coefficient 1.0 ± 0.1) | PARKED | shortlist #5 |
| **C03** Add a K4×C term to the kernel | Q6, Astra R7 appendix | Never run. Closed because Q1 died and D22 found the cheap kernel forms net ≤ 0 | none (killed by association) | 0 | — | UNCLEAR | One-layer diagnostic (non-pair corr / slope at L12 and L15), ~30 min; only if C01 revives |
| **C04** Take the (3,1) slice from the kernel gradient | Q7, Astra R7 | Fable R8: the (3,1) sector is 3e-5 within 1e-3 | C | — | faithful | DEAD | — |
| **C05** Nine cheap (2,1,1) hypertree classes (+ exact (2,1,1) law, triangle) | Q12, Sol High R7 | Fixed-coefficient R² ≤ 0.07; worse at every layer for the decisive neurons | P | the exact law was included | faithful | DEAD | — |
| **C06** Cheap κ4-diagonal rules: V39 depth multiplier, pairwise transport, 13-feature regression, star sources, \|κ3\| | D4, K3ab–K3ad, 23 Sep | V39: ×1.25 / ×1.16 (mean closure only), ×1.26 / ×1.16 (everywhere), ×1.09 / ×1.05 (half strength). Pairwise transport corr 0.24–0.59. Features 0.19 → 0.18. Stars ~10% of the level at depth | K / P | 3 strengths; 2 networks; cross-network fit | own | DEAD | — |
| **C07** Per-row λ law fitted to the raw (3,1) slice (V35) | R13, K3x, 22–23 Sep | Blend 0 / 0.25 / 0.5 / 1.0 / 1.5 → 2.2911 / 2.2923 / 2.2933 / 2.2958 / 2.2988e-8 (10 MLPs): monotonically worse. Object mismatch: the chain carries the r = 1 core, which the scalar λ fits at R² ≥ 0.95 (F105) | K | 5 blends, 10 MLPs | fitted to the wrong object, but the core leaves little to fit | DEAD | — |
| **C08** "κ4 line closed: a perfect core is worth ~9%" (F105); first κ4 oracle ×6–15 worse | R14, K3x / K3y, 22–23 Sep | The first oracle had a units bug. Corrected oracle: exact κ4 diagonal 2.290e-8 → 1.609e-8 (×0.702) | C (wrong ceiling) | M0 only | bug | REVIVED (as a target; no cheap carrier found since) | — |
| **C09** Young κ4 carrier (carry the last h births) | Q18, Opus R12, 25 Sep | κ4 level at decisive neurons, h = 1/2/3/4/6: L12 2 / 6 / 12 / 18 / 35%, L16 3 / 2 / 5 / 9 / 20%. Ideal carrier, out of sample: errG4 0.210 / 0.193 → 0.203 / 0.184 (h = 2), 0.188 / 0.173 (h = 3) | C | 5 horizons, 2 layers, N = 1e6 | own | DEAD | — |
| **C10** Opus R12 cheap κ4 rules: scale mixture, Sym(C⊗C), a cheaper Astra kernel, B1 two-product memory carrier | D22, 25 Sep | Fresh physics gives 1–4% of the κ4 level from L8. Scale mixture / C⊗C corr ≤ 0.09. Cheaper kernel ~14 u for −6.4% / −2.6% (derived), net ≤ 0. B1 misses 27.3% of D21 at L15 (gate 2%) | C / P | B1 is one carrier tested (report caveat 5) | faithful | DEAD | — |
| **C11** Source-coupled skew-lift quartic carrier Σ_s Sym4(T_s ⊗ r_s) | D26, Round 13, 26 Sep | The best LSQR fit leaves 50% of K22 and 83% of K31. Decisive corr 0.03 (chain 0.26, gate 0.70). 300 and 1500 iterations give identical results | P (upper-bound fit) | L14 → L15, M0 | the proposal's own oracle | DEAD | — |
| **C12** Dense pair core born at the first ReLU | D27, Astra Round 13 | Corr −0.35 to −0.22 at L12–16 (chain 0.26–0.34); level 1–2% | P | L12–16 | run by the proposer on our teacher | DEAD | — |
| **C13** Carried pair innovations: κ4 on the κ3 source legs (vectors γ_s, η_s per source) | Q22; Phase A A4, 26 Sep | 26-scalar pre-check (per-source scalars, not vectors).<br>In sample: errG4 ×0.92; corr 0.30 → 0.41 / 0.26 → 0.34.<br>Out of sample: ×1.053 / ×1.056; corr 0.30 / 0.21 vs the chain's 0.30 / 0.25 | P | 2 layers (L14, L15), neuron-split CV | simplified: the vector form was never fitted (queue: ~1.5–2 h CPU) | PARKED | shortlist #3 |
| **C14** Carried pair-response κ4 modes Sym(H⊗M), paid for by a smaller old tier (CPR) | Phase A A2, 26 Sep | Best of 105 teacher-forced constructions (incl. the teacher's own (2,2) slice as birth): errG4 0.138 / 0.190 / 0.197 / 0.207 vs chain 0.142 / 0.193 / 0.199 / 0.210 (L8 / 12 / 15 / 16), i.e. ≤ 3% vs a 15% gate. Cost +13.4 u per mode (derived) | P (generous) | 15 modes × 4 readouts, out of layer | own design | DEAD | — |
| **C15** Puffi's SSC κ4 response closure, alone and as Puffi's full stack at 1024 × 16 (with one-step response memory) | Q23, 26 Sep; Phase A A1-F1a | Q23, out-of-layer decisive corr: 0.43–0.45 (L8), 0.34–0.40 (L12), 0.25–0.33 (L15), 0.31–0.36 (L16) vs chain 0.37 / 0.34 / 0.26 / 0.32. The 9-feature form scores 0.46–0.52 in-layer but 0.00–0.18 out. Stacked on the chain: errG4 ×1.006–1.023 at L8–16 (×0.88–0.93 only at L4). Memory ×1.014 ± 0.017 | P | 4 forms; shared and per-layer Λ; memory at L16 only | reconstruction from the public PDF; Λ and Q refitted. Puffi's −64.6% was vs a chain without regeneration, and V29's λ·C_off is SSC's 1-mode case (2.23e-8 vs 4.20e-8 with no channel) | DEAD | — |
| **C16** κ4 / feedback constants: FB 2.0 (full Edgeworth), FB 1.15–1.40, λ 0.75–0.90, β grid | R30, D3, D22, Q2, Q21, 23–26 Sep | FB 2.0: +3.4% / +3.1% (+3.8% / +3.2% in R12). λ 0.85: +1.8% / +1.0%. FB 2.0 + λ 0.75: +6.7% / +4.5%. 100 MLPs: LAM 0.9 ×1.0026, FB 1.40 ×1.0004, LAM 0.90 ×1.0005, FB × β grid +0.00…+0.12%. Graded #332372 (FB 1.40) 4.8284e-9 vs 4.8066e-9 | K | ≥ 12 settings, 100 MLPs, grader | own | DEAD | — |
| **C17** Role-split κ4 regeneration (RSR: separate λ for feed / diagonal / (3,1) / metric) | Phase A A3, 26 Sep | Best split (feed λ ×0.7 + dG λ ×1.25): ×0.971 / ×0.998 (gate ×0.97 on both). (3,1) ×2 → ×1.125. Graded #332382 4.8438e-9 vs its base #332324 4.8066e-9 (+0.77%) | K | 8 configs on M0, 3 on both, 1 graded | own | DEAD | — |
| **C18** The sphere's remaining (2,2) κ4 term at layer 0 (V31c) | D24, 25 Sep | Scale 1: +0.04% / +0.15%. 0.5: +0.10% / +0.14%. 2: +0.02% / +0.19%. −1: −0.12% / +0.11%. The term is ~1e-6 per entry | K | 4 scales, 2 networks | own | DEAD | — |
| **D. κ3 / variance / mean accuracy, correctors, chain knobs** | | | | | | | |
| **D01** Missing covariance-closure terms (Astra's C³ / C⁴ / C×κ31 patch; Grok's D21 scaling; Gemini's and DeepSeek's κ3×κ3 terms) | D1, D5, D6b, D7 (Round 7), 23 Sep | Patch terms rms 2–3e-6 vs error 6.5–9e-5; projected variance error ×0.998–1.007. Grok's term R² ≤ 0.08. The κ3×κ3 terms are already in the (1,1) table, or impossible for a bivariate closure (Q5: the closure is exact given true inputs) | C / P | — | faithful | DEAD | — |
| **D02** Higher-order κ3 physics: 12 K×K terms, transport curvature, doubled edge, triangle, self-loop hubs, rank-1 D21 error, non-Gaussian vertex weight | D3, D9 (Fable R7 / R8), 23–24 Sep | K×K rms 3e-6–2e-5 vs error 2.2e-4. Curvature slope ±0.001, no coherent part. Each ≤ 2% of the error variance at depth. Doubled edge: R² 0.01 for +2 u / source-layer. Vertex ratio 1.00 | P | all sectors enumerated | own + Fable | DEAD | — |
| **D03** True D21 in the feedback legs only (closure cross-talk) | Q4, Opus R12, 25 Sep | ×0.9999 (the true slice everywhere gives ×0.758) | C | 1 oracle | own | DEAD | — |
| **D04** Low-rank correction of the covariance error | Q10, Terra R7, 23 Sep | The top 8 directions hold 26–49% of the energy, aligned with the mean (0.75–0.79). Decisive-neuron correction error 0.81–1.04 at k = 32; k = 256 would be needed | C | k = 4…256 | own | DEAD | — |
| **D05** Buy κ3 accuracy with cost: exact (unconfined) old tier, R_FB 128 | D3, R28, 23 Sep | Exact old tier: MSE −3.9% for +19% cost (slice error 6.7% → 6.6%). R_FB 128: −3.1% for +43% | C / K | — | own | DEAD | — (R_FB's cheap direction later won: D13) |
| **D06** Reallocate ranks / age limits by layer (toward late layers) | D23, 25 Sep | κ3 oracle by layer window: all 0.608 / 0.715; ≥ L8 0.637 / 0.754; ≥ L12 0.704 / 0.863; ≥ L14 0.848 / 0.903; ≤ L11 0.884 / 0.837 | C | 5 windows, 2 networks | own | DEAD | — |
| **D07** Refit the disabled online mean correction (F47, 13 per-neuron features) | Q19, 25 Sep | Teacher-forced in-sample R² 0.15 (L3), 0.02 (L8 / 12 / 15); chance ≈ 0.013 (F47 had 0.3–0.6 on V16b) | P | 4 layers, M0 | faithful to F47 | DEAD | — |
| **D08** Nonlinear neighbour-feature drift corrector (Puffi's 31 → 32 → 32 → 1 MLP) | Q26; Phase A A1-F1b, 26 Sep | Held-out R² ≤ 0.004 (13 / 31 / 44 features, ridge and MLP). Held-out layers −0.40 to −0.003. Gate 0.10 | P | 3 feature sets × 2 models × 2 splits | features computed from the teacher state (a proxy for the chain's state) | DEAD | — |
| **D09** Learned post-hoc residual corrections (ridge, quadratic, GBM; the HF supervised corpus; Koopman / Stein / RMT dictionaries) | R9 / V33, R15, 22–23 Sep | 100-MLP CV: ridge ×0.997, quadratic ×0.993, GBM ×0.995. Single-feature corr ≤ 0.04. Per-MLP offsets explain 0.44% | P | 102,400 output neurons | own | DEAD | — |
| **D10** Per-network λ from a small MC probe | Board A2 (Fable R1 H3), 21 Sep | Never run: 10⁴ samples cost 156 u; per-network offsets explain 0.44% | C (derived) | — | — | DEAD | — |
| **D11** Global chain knobs (AGE_OLD 6, AGE_OLD2, QPASS / QPASS2, row λ, LAM table, BETA 0.8, AGE_OLD 5 + R_OLD 448, …) | K3f, K3t, Q3, Q21, v49 re-sweep, 21–26 Sep | 100 MLPs: qpass_2 ×1.073, age2_8 ×1.0006, qpass2_3 ×1.0006, row0.25 ×1.0003, AGE_OLD 6 ×1.0134, AGE_OLD 5 + R_OLD 448 ×1.0244, AGE_OLD2 6 ×1.0127, AGE_OLD2 8 ×1.0013. Graded #332328–#332330: all worse than #332324 | K | 100 MLPs (paired SE 0.01–0.15%); grader | own | DEAD | — |
| **D12a** AGE_OLD 4 → 5 | K3f, 21 Sep → Q21, 26 Sep | Killed: raw −2.0%, cost +3.0% → adjusted +1.0% (10 MLPs). After the transport join: 100 MLPs ×0.9895 ± 0.0032; graded #332318 4.9057e-9 (new best at the time) | K (base-dependent) | 1 base | own | REVIVED | — |
| **D12b** BETA 1.2 | V31b graded #331805, 22 Sep → Q3, 24 Sep | Killed bundled with LAM ×0.9: graded 5.194e-9 vs 5.137e-9 (+1.1% raw). Alone on 100 MLPs: ×0.9993 ± 0.0005; shipped in #332110 | K (two knobs mixed) | 1 bundle | own | REVIVED | — |
| **D13** "One-for-one law" / "the cost side is closed within this formulation", used as a blanket kill of cost trims | Blueprint C5; notes K4 (line 1370), 22–24 Sep | Contradicted:<br>• R_FB 16 → 4 gives ×0.957 / ×0.956 (only the expensive direction, R_FB 128, had been tested)<br>• the transport join<br>• compaction + basis-side Strassen: 4.8066e-9 → 4.1081e-9 (−14.5%) on 26 Sep | meta (a generalised K result) | — | — | REVIVED | — |
| **E. Sampling and non-cumulant families** | | | | | | | |
| **E01** Monte Carlo, control variates (linear / quadratic surrogate), hybrids, MC sidecar blend, stratified / Rao-Blackwell MC | Q11, D2, D6c, D8, board C1 / C2, 21–23 Sep | Linear surrogate corr L2 0.73 → L16 0.19: variance cut ×1.0–2.2 (×100 needed). Quadratic surrogate 0.967 (L2) → 0.00 (L16). 10⁴ samples = 156 u for 1.4% variance error (0.1% needed). Blend weight ~0.002 | C | 4 layers, 1e4–2e4 samples | own | DEAD | — |
| **E02** Non-cumulant families: TAP / AMP, characteristic function, saddlepoint, GMM, polytope, EP, SOS, branch-and-bound, input chaos, TT-PCE / TT-cross, copula + rich marginals, 1/n and DMFT expansions | K3r, K3s, K3t, K3u (Sol R5, Astra R5b), 22 Sep | TT-cross at width 32: 60–500% vs MC 0.5–1%. Copula with exact marginals: 7.5e-4–1.14e-3 vs the Gaussian closure's 4.5–5.2e-4. Input chaos: 513 u per update. SOS Cholesky: 819 u. The rest by argument or by the proposer's own gates | C (mostly argument) | TT-cross only at width 32 | faithful where run | DEAD | — |
| **E03** ARC's reference augmented K3 / K4 chains as the base | K3i, K3w, 21–22 Sep | K3-simple factored at 1024: 1523 u, raw 3.7e-8 / 4.5e-8. K3-augment: 1.5× better for ~8× the cost at width 256. K4 factored: 516 u already at width 128 (n⁴ class) | C | widths 96–256, plus 1024 | ARC's own code | DEAD | — |
| **E04** Proposals with derivation errors (random-weight mixture memo, a "closed-form" (1,1,1,1) κ4, variance adjoint recursion, Hutchinson probes) | D5, D6a, D8, 23 Sep | Our weights are fixed (teacher κ3_iii rms ~1e-2; chain slope 1.00). The formula counts only the diagonal term. Hutchinson needs the unknown error. The claimed "40×" / "30×" MC gains measure ×1–2 | C (derivation) | — | faithful reading | DEAD | — |
| **E05** Gaussian-mixture closure; Puffi's A716 slice closure | 504aldo F78 / F82 (external) | 2.3e-6 and 9.8e-7 at the P2 shape | C (external) | — | not reproduced by us | DEAD (unverified by us) | — |

**Counts by section:**

| Section | Entries | DEAD | PARKED | UNCLEAR | REVIVED |
|---|---|---|---|---|---|
| A | 27 | 25 | 0 | 0 | 2 |
| B | 10 | 4 | 3 | 1 | 2 |
| C | 18 | 13 | 3 | 1 | 1 |
| D | 14 | 11 | 0 | 0 | 3 |
| E | 5 | 5 | 0 | 0 | 0 |
| **Total** | **74** | **58** | **6** | **2** | **8** |

## 4. Shortlist: the top 5 PARKED items, ranked by plausible upside × cheapness

**Common gates for every cost-lever re-test:**
- Paired against v52safeB on the teacher pair.
- Score = MSE × metered util, not MSE alone.
- Stay within the Q30 op budget (≤ ~18k warm-call ops), because residual ∝ op count.

The sixth PARKED item, B10 (R_OLD 320 / row threshold 1e-2 lost to the residual cap), is left out. Its teacher-measured upside is ≈ 0: 0.993 / 1.005, and 0.969 / 0.968 vs 0.973 / 0.961. A v52Br352 package already exists.

**#1 — B04: always-on birth-column compaction (Q20 mode 16, never isolated).**

*Why the kill was unfair.*
- Always-on neurons (Φ ≈ 1) have near-zero birth coefficients, just like dead ones. They are the one compaction axis where nothing is harvested: v48–v52 compact dead neurons only.
- The only data point is mode 25 at Φ 1e-2. It mixes always-on neurons with dead rows and columns.
- It was judged on MSE alone, and the oracle cannot credit util.
- The two networks disagree in sign: relative to mode 9, +1.15% / −0.20% (derived).
- The 3e-2 point (+11.4% / +13.0%) is the known threshold cliff, not evidence about tight thresholds.

*Cheapest fair re-test.*
1. **Oracle.** Run mode 16 alone at Φ > 1 − {1e-2, 3e-3, 1e-3}. Add a v48_coldrop-style variant that removes the always-on newborn columns completely. The w2 := 0 oracle leaves them in the M-leg terms, and for dead neurons the full column drop cost ×6.40 / ×6.74 (B06). About 6 configs × 2 networks × ~2 min ≈ 25 min (est.).
2. **Metered emulation.** Only if MSE ≤ +1% on both networks. Emulate the saved products on 32-padded shapes with `experiments/probes/m1_matmul_ledger.py` (~2 h). Gate: score ≤ ×0.98 on both networks.
3. **Build.** Only after that.

*Upside (estimate, not metered).* The direction review put dead + always-on birth columns at 1e-2 at ~19% of the young tier with 32-multiple packing, vs 9.5% for dead only. The always-on increment is therefore roughly 10 u, ~4% util on the 249-u ledger (derived). Teacher statistics show Φ > 1 − 1e-4 for 2% / 7% / 13% of neurons at L8 / L12 / L16.

**#2 — B07: drop the dead neurons' own d3 at intermediate layers, to allow output-row compaction of the transports.**

*Why the kill was unfair.*
- It was killed on MSE at two thresholds without crediting the saving it unlocks. That saving is the output half of Q20's original "~2 × dead fraction on each transport" estimate. The input half was measured at −3.4% util (v48).
- The M0 cost at Φ < 1e-3 is only +0.3%. The synthetic network (+2.9%) is the warning sign.
- Lower thresholds were never tried, although threshold matters a lot in the analogous hub-drop series: ×6.40 at 1e-2, ×2.32 at 1e-4, ×1.48 at 1e-5 on M0.

*Cheapest fair re-test.*
1. Run `v51_d3mid` at Φ < 1e-4 and 1e-5. Also run 1e-3 with the last two layers kept exact: their d3 is scored directly. About 4 configs × 2 networks ≈ 16 min (est.).
2. Estimate the output-row saving at those thresholds with the ledger (~1 h).
3. Gate: MSE ≤ +0.5% on both networks and metered saving ≥ 2× the MSE cost. Kill if the synthetic network stays above +1% at 1e-4.

*Upside.* Not estimated in the sources. It is bounded above by the input side's measured −3.4% util at 1e-2, since fewer neurons qualify at Φ < 1e-4 (estimate).

**#3 — C13: Q22 carried pair innovations, fitted in the proposal's own vector form.**

*Why the kill was unfair (fidelity).*
- The proposal carries two n-vectors (γ_s, η_s) per κ3 source. Phase A tested only the queue's 26-scalar pre-check: one scalar pair per source.
- The pre-check's own stop rule (in-sample corr < 0.3) was not triggered: it reached 0.41 / 0.34.
- The out-of-sample failure (×1.053 / ×1.056) is a statement about per-source scalars, not about the vector subspace.

*Cheapest fair re-test.*
- **Oracle, as specified in queue Q22 / moonshot audit F5.** Fit γ_s, η_s by LSQR on the L14 captures (13 sources) to the teacher's post-ReLU K22 / K31 off-diagonals. Then predict the L15 pre-activation κ4 diagonal, out of layer. About 1.5–2 h CPU; no chain change.
- **Baseline (same maturity):** the plain V29 dump, with errG4 0.199 / 0.210 and decisive corr 0.26 / 0.32 at L15 / L16.
- **Gate (pre-registered in Phase A):** errG4 ≤ 0.85 × chain and corr ≥ 0.45. Kill if corr < 0.3.

*Upside if it passes.* Phase A's linear map values a 15% cut of the κ4 error at ≈ −8% final MSE, at ~0.35 u (derived).

*Prior: low.* Its sibling D26 scored corr 0.03 in the same kind of oracle, and A2's 105 carried forms cut the κ4 error by ≤ 3%.

**#4 — C01: the Astra κ4 kernel, restricted, metered and re-tuned jointly with the chain's constants.**

*Why the kill was unfair.* This is an E-type add-on. It was judged against a chain whose κ4 constants are co-adapted (fairness rule 4). Only two (A, S) blends were tried, and the costs are derived, not metered. Its baseline was the plain chain (A = 1, S = 0: 2.2903e-8), so maturity is not the problem. Co-adaptation and cost accounting are.

*Cheapest fair re-test.*
- Run the L11–16 and L8–16 decisive-neuron kernels on v52safeB, with their three products per layer routed through the Strassen helper and metered.
- Add a joint 3 × 3 grid of λ-scale × FB_SCALE around each. About 20 configs × 2 networks × ~2 min ≈ 1.5 h (est.).
- Gate: score ≤ ×0.98 on both networks.

*Upside: small.* On today's ~197-u base, the 7 u and 11.5 u restrictions cost about +3.6% and +5.8% before Strassen (derived). It only pays if Strassen roughly halves that and the re-tune adds 1–2%.

**#5 — C02: the cheap K3×C (1,1,1,1) term (Q8).**

*Why the kill was unfair.* The cheap version was measured on M0 only: the synthetic run was lost at a session restart. Its ~13 u Strassen cost is derived.

*What limits the upside.*
- Its measured increment over the same-restriction kernel is small: ×0.993 (derived, 0.9583 vs 0.9650). It therefore matters only together with #4, or with a cheaper readout.
- The cheaper readout is Q14's mixed-coordinate form, claimed at 17–34 u for the full −8.8% term. Break-even on today's base is ~17 u (derived: 8.8% of ~197 u).

*Cheapest fair re-test.*
1. Finish the synthetic run with the flags recorded in `results/summaries/r8_k4_dist_e2echeap.json`. Its runtime is not recorded (unverified); expected under 1 h.
2. Meter the cost.
3. Fold it into #4's grid.
4. Gate: score ≤ ×0.98 on both networks.

**The two UNCLEAR items close in minutes:**
- **B05:** split the \|α\| > 3 row mask into dead and always-on halves: 2 configs × 2 networks. The expected outcome, by inference, is that the always-on half is costly and B05 becomes DEAD.
- **C03:** only worth a 30-min one-layer diagnostic if #4 revives the kernel.

## 5. Open items that were never killed (outside the verdict counts)

- **Q9 joint knob search:** open.
- **Q13 variance super-closure:** ceiling 9–13% of the closure residual (Q5).
- **Q14 mixed-coordinate readout:** not even the L14 identity check has been run.
- **Q15 S = 2 partner channel:** ceiling ~−3.4% (Astra's fit).
- **Q25 per-layer constant refit:** only per-block λ was run. The last block ×1.25 was submitted as #332395; its grade is not in the sources (unverified). The FB and κ4-diagonal-weight blocks were never run.
- **The weights-only angular attractor (B12):** never run.
- **GPT R4 §7 penultimate κ4-residual rider (conversation audit D-7):** never revisited.
- **The zero-constant chain:** write-up value only.
- **Umbrellas with no mechanism yet:** a nonlinear trajectory-specific state (Blueprint M3, Opus R12 B2) and a better all-distinct κ3 birth law.

## 6. Record corrections found while auditing (no file was changed)

- **`queue.json` Q20, "next".** It asks for a w1-only threshold sweep at 0.02 / 0.03 / 0.05. That sweep exists in `results/summaries/teacher_eval_v48_thr.json` and is worse at every threshold by score (B03).
- **`queue.json` D11.** It still reads "dead" for the transport join, which shipped as Q16 (#332257).
- **`UNTESTED_IDEAS.md` A4.** Its prior "He-init: all neurons near threshold" is contradicted by B01 / B02.
- **`consultations/moonshot_ideas_audit_2026-09-26.md` C14.** It lists alias pricing as DEAD-SOUND. Aliased Grams shipped in v49, together with basis-side Strassen (util −4.3%).
- **Stale wording in two places.** `docs/build_notes.py` K4 (line 1370, "Cost side is closed within this formulation") and the one-for-one wording of Blueprint C5 are contradicted by the 26 Sep trims (D13). The Blueprint's "halving is impossible inside this family" still stands: cost fell 249 → ~197 u, not by half.
