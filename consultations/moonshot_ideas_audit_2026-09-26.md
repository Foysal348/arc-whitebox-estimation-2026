# Moonshot ideas audit: big ideas that were raised and then dropped, parked, never tested or tested too narrowly (26 Sep 2026)

**Scope.**
- **Conversation.** The main transcript `f76e7d52-…jsonl`, 20 Sep 23:53 → 26 Sep 01:03 BST. It was streamed to `scratchpad/convo_extract3.txt` (7,772 lines, 733 records, including 4 compaction summaries) and read in full.
- **Earlier audits.** The three 25 Sep audits (conversation, direction, forum). This audit does not repeat their items; it only relabels some of them.
- **Team records.** `queue.json` (51 items), the Moonshot Blueprint, `UNTESTED_IDEAS.md` (R1–R31), and the notes (`build_notes.py`, grep only).
- **External record.** The 504aldo findings log (grep of F47–F105 and the open-questions section); the Round 9–13 external answers; the forum dump (93 topics).
- **Team Puffi's Phase-1 SSC write-up.** The public forum upload of topic 18175, fetched read-only; text extracted to `scratchpad/puffi_ssc.txt`.
- **Leaderboard.** `data/lb_scan/phase2_api.csv` (3,239 graded Phase-2 submissions) and `record_progression.json`. Team names for 44 key submission ids were read from their public submission pages; the organisers confirmed on 25 Sep that these pages are public. The raw pages are in `scratchpad/rec2/`.
- **Conventions.** All times are Bangladesh time (UTC+6). "util" = FLOPs/2^41; "u" = 2^31 FLOPs.

---

## Executive summary

1. **Count.** 47 moonshot-class ideas were found: 32 DEAD-SOUND, 3 DEAD-NARROW, 6 PARKED, 5 NEVER-TESTED (3 of these are umbrella questions with no mechanism) and 1 DONE. Most kills are sound.
2. **The one untested idea with published evidence behind it is Team Puffi's SSC κ4 response closure** (topic 18175, 16 Aug).
   - Mechanism: κ4 = Σ Λ_ab Sym(B_a⊗B_b) from ≤4 covariance-response modes, with the diagonal read as 3·qᵀΛq.
   - In Puffi's own chain this branch cut MSE by 64.6%.
   - None of our kills covers its form: F68 projected onto modes under a Sym(G⊗mI) use side, which has no all-distinct sector; Opus tested single fixed modes only; the D4 regression used 13 per-neuron features.
   - The forum audit's label "COVERED → F68" is wrong.
3. **The best-return DEAD-NARROW item is Q20's output side.** It was killed by a rows + columns test at Φ<1e-2. Rows-only compaction for |α|>3 was never tested.
   - The whole live-neuron compaction stack is worth about −6 to −12% util.
   - That is below the 20% bar on its own, but it is the largest missing trim.
4. **Two more cheap families are consistent with the leaderboard's zero-FLOP jumps.**
   - A per-layer end-to-end refit of the closure constants. Our sweeps were global scalars only. Puffi measured −13.6% from refitting to the consumed mean correction.
   - A nonlinear, neighbour-feature drift corrector. Q19 tested linear per-neuron features only.
5. **Leaderboard, attributed from public pages.**
   - **emanuel_ruzak started from V29 itself** (2.13e-8 @ 0.249) and reached 1.770e-8 @ 0.181 in 4 days: score −40%.
   - **jlacombe** gained −5.5% and **AndreasHad04** −7.2% at the same util to 4 digits.
   - **J2W** gained −11.9% in 27 h with util within ±0.3%. Its largest single step was −5.4%, not −8%.
   - So ≥10% accuracy is reachable on a V29-class chain by changes costing at most O(n²) per layer.
   - That rules out every idea costing ≥1 dense product per layer (Q8, Q14, κ4 carriers, the Astra kernel) as the source of these jumps.
6. **The leaders' util of ~0.15 at raw 2.0e-8 is not reachable by any trim stack in the record** (best ≈0.19). It needs a κ3 representation that none of the ~20 outside proposals delivered.
7. **Top 5 falsifiers.** About 15 h of offline work plus one Kaggle run:
   1. Q20 rows-only oracle plus metered emulation.
   2. SSC teacher-forced κ4-diagonal test.
   3. Per-layer end-to-end constant refit.
   4. Neighbour-feature drift corrector.
   5. Q22 oracle.
8. **Strategic.** The team-merge window closes 3 Oct 05:59 BST. It is the only option in the record with a demonstrated (by others) ≥15% payoff. It was dropped by default on 25 Sep, not by decision.

---

## 1. Leaderboard facts (checked against the API scan and public submission pages)

| Team (how it was identified) | Trajectory (BST) | Raw MSE | util | What it shows |
|---|---|---|---|---|
| suliman_tadros (record list) | 5 Sep 16:56 → 6 Sep 06:01 → 18:21 | 2.034 → 2.064 → 1.969e-8 | 0.1868 → 0.1645 → 0.1503 | −20% cost in about 25 h at flat raw, including one −12% step. This fits the 4–5 Sep board-wide Strassen adoption (504aldo F77/F78: 15–30% flat-raw drops for every team that resubmitted; marius_binner's P1 PDF discloses 5-level Strassen) |
| Puffi / andrew_epstein (record list) | 5 Sep 20:39 → 6 Sep 06:48 | 2.230 → 2.023e-8 | 0.1587 → 0.1478 | Before Strassen (2 Sep), Puffi was at 0.207 and J2W at 0.213, while 504aldo's then-current V25 cost 0.367. The base representation gap was about 1.75× |
| **J2W** (pages 331332/331347 + record list) | 17 Sep 02:47 → 18 Sep 05:58 (6 submissions) | 1.8698 → 1.8685 → 1.792 → 1.694 → 1.655 → 1.648e-8 | 0.1345–0.1353 | **−11.9% raw in 27 h at util ±0.3%**, i.e. at most ~0.5 u of change. Steps: −4.1, −5.4, −2.3, −0.4%. A further −3.8% came at fixed util on 24–25 Sep (1.639 → 1.577e-8) |
| **emanuel_ruzak** (pages 331935, 332033, 332103, 332137, 332238) | 21 Sep → 23 Sep 08:16 → 23 Sep 22:13 → 24 Sep 10:04 → 15:28 → 25 Sep 11:25 | **2.13** (V29 verbatim, our 24 Sep LB snapshot) → 1.895 → 1.911 → 1.822 → 1.801 → **1.770e-8** | 0.249 → 0.2287 → 0.1971 → 0.1980 → 0.1981 → **0.1810** | **Our exact starting point.** First own step: −11% raw at −8% util. Later steps −4.6% and −1.2% at near-fixed util. **Score 5.30 → 3.20e-9 (−40%) in 4 days** |
| **jlacombe** (pages 332114, 332175, 332242/3) | 24 Sep 11:39 → 22:24 → 25 Sep 12:00 | 1.9225 → **1.8172** → 1.780e-8 | 0.20533 → **0.20528** → 0.1998 | **−5.5% at identical util** (a zero-FLOP change). Started near chain accuracy (2.10e-8 @ 0.209 on 21 Sep, our snapshot) |
| **AndreasHad04** (pages 332093–332218) | 24 Sep 07:22 → 09:44 → 10:57 → 25 Sep 06:29 | 1.9985 → 1.864 → 1.854 → 1.855e-8 | 0.23286 → 0.23289 → 0.23289 → 0.22438 | **−7.2% at util identical to 4 digits**, then −3.7% cost at flat raw. Their own chain, not 504aldo's (their 22 Sep post in topic 18218). That post names "a kappa4 to kappa3 FEED arm sitting unpriced" and says F68 guided it. V29 already has the feed |
| **shiv_m** (pages 331852, 331916) | 22 Sep 18:14 → 23 Sep 06:07 | **2.1781 → 2.1882e-8** | **0.1960 → 0.1792** | **Our raw (2.1788e-8) at 14–21% lower util.** Also fklassen 1.997e-8 @ 0.201 and thibault_gounant 2.01e-8 @ 0.204 |
| us, #332257 | 25 Sep | 2.1788e-8 | 0.2273 | — |

**Correction to the brief.** J2W's largest single step was −5.4%, not −8%. The −8% figure (1.792 → 1.655) spans two submissions. The robust statement is "−11.9% over five submissions in 27 h at util ±0.3%".

---

## 2. Full table of moonshot-class ideas

Legend:
- **DEAD-SOUND**: the kill test covers the idea's strongest form.
- **DEAD-NARROW**: the kill test does not cover the strongest form.
- **NEVER-TESTED**
- **PARKED**: queued but not run.

"Sub-threshold" means that even full success would fall short of the ≥10% accuracy / ≥20% cost bar.

### A. Accuracy: the κ4 channel (corrected-units oracle: exact κ4 diagonal −30%; K3ab–K3ac)

| # | Idea (source, first raised) | Mechanism (one line) | What was measured (where) | Class |
|---|---|---|---|---|
| A1 | "κ4 accuracy at K3 cost" (K3i 21 Sep, K3w 22 Sep, Opus R12 25 Sep) | The leaders sit one cumulant rung above V29. The K3→K4 gain grows with width (1.15× at 96, 1.56× at 128). The deep κ4 pattern is carried memory that needs an all-distinct sector | K3i/K3w ladders. Opus R12: LIN carry gives decisive-neuron corr 0.77–0.80 vs the chain's 0.26–0.34. Fresh physics gives 1–4% of the level | NEVER-TESTED (umbrella; instances A2–A12) |
| A2 | Astra quartic κ4 kernel (Astra R7, 23 Sep; queue Q1) | Recompute the quartic response each ReLU; add its non-pair part | −6.4% / −2.6% raw at +7–10% cost. Cheap forms net ≤0 (D22). Late-layer/decisive-neuron cuts net ≤0 (Q1) | DEAD-SOUND |
| A3 | (1,1,1,1) K3×C birth term (Fable R8, 24 Sep; Q8) | The all-distinct κ4 birth from κ3 sources × C | −8.8% raw at ~140 u (MLP #0 only). Cheap: −4.2% at ~35 u classical / ~13 u Strassen; derived cost, synthetic run lost at the restart (Q8) | DEAD-SOUND as built (cost) |
| A4 | Mixed-coordinate readout of A3 (Astra R9, 24 Sep; Q14) | Sum sources before one B^T product; the identity leg saves a product | Nothing measured, not even the L14 identity check. Break-even 25 u; at 17 u, score −2.1% | PARKED, sub-threshold |
| A5 | Young κ4 carrier (Opus R12, 25 Sep; Q18) | Carry the last h births' κ4 exactly | h = 2 gives 2–6% of the level. The ideal h = 3 carrier gives errG4 0.188 vs 0.210, i.e. −2 to −4% MSE (Opus §7.5) | DEAD-SOUND |
| A6 | Cheap κ4-diagonal rules (own, 23 Sep; D4) | V39 depth multiplier; pairwise (W²)ᵀK22(W²); 13-feature regression; star sources; \|κ3\| | V39 ×1.16–1.26 worse. Pairwise corr 0.24–0.59. Features 0.19 → 0.18. Star 10% at depth (K3ab–K3ad) | DEAD-SOUND for these forms |
| A7 | Per-row λ from the κ4 teacher (own, 23 Sep; V35) | λ_l is the mean of κ4diag/var per row | Monotonically worse end to end (2.2911 → 2.2988e-8). F105 explains why: object mismatch (r=1 core) | DEAD-SOUND |
| A8 | Skew-lift quartic carrier (Round 13; D26) | κ4 = Σ_s Sym4(T_s⊗r_s) | Oracle upper-bound fit: corr 0.03 (gate 0.70) | DEAD-SOUND |
| A9 | Dense pair core born at the first ReLU (Astra R13; D27) | One carried G, no later births | Corr −0.35 to −0.22 (Astra's own run) | DEAD-SOUND for its no-births form |
| A10 | Carried pair innovations (Round 13; Q22) | Two vectors (γ, η) per κ3 source, read through Z = P∘P | Untested. D26's subspace (Sym4(T⊗r)) differs from Q22's (Σ γ_j Sym(p_j p_jᵀ ⊗ PPᵀ) + η_j p_j⊗4), so D26 does not cover it | PARKED |
| A11 | **SSC response closure** (Team Puffi P1 #327725, forum 18175, 16 Aug; public PDF) | κ4 = Σ_ab Λ_ab Sym(B_a⊗B_b) with modes B1 = C, B2 = Q/v̄ (Q a 6-feature off-diagonal form: v̄C, C⊙C, Sym(D_m Kᵀ), Sym(t mᵀ), vvᵀ, v̄²11ᵀ), B3 = v̄I, B4 = ½(g_i+g_j)C_ij. Diagonal readout 3 q_jᵀΛq_j with q_aj = w_jᵀB_a w_j. Λ is a 4×4 per-layer matrix fitted by ridge to the next pre-activation's marginal κ4. One-step response memory β = −0.3 | Never run by us. 504aldo F68 used the SSC "lens" only as a **linear** projection of the aug core G_off on 9 live modes under the **Sym(G⊗mI)** use side: C_off alone → 2.230e-8, 9 modes → 2.205e-8. That use side has **no (1,1,1,1) sector by construction** (Opus R12 §2). Opus tested only the fixed single modes Sym(C⊗C) and the scale mixture as LIN predictors (corr ≈0). The D4 regression used same-layer per-neuron features, not the quadratic forms w_jᵀB_a w_j. Puffi's numbers are at P1 shape (256 × 32): response branch −64.6% MSE (7.698e-6 → 2.728e-6); whole stack −77% vs Wu K=3, −48% vs K=4; cost 1.34× a K3 port | **NEVER-TESTED** (the kills above are too narrow for its form) |
| A12 | κ4 with memory via the augmented chain (504aldo F51/F58/F64/F105) | Per-source κ4 legs / dense G core | −9% raw at 0.5–1.0 B (restricted Sym(G⊗metric) channel); the dense core gives 2.04e-8 | DEAD-SOUND (cost) |
| A13 | Nonlinear, trajectory-specific state (Sol 24 Sep; Blueprint M3; Opus R12 B2) | A state that is not a linear representation of Sym³ | "B2 was not run … needs a new idea" (Opus R12 §4) | NEVER-TESTED (umbrella, no mechanism) |

### B. Accuracy: κ3, mean and variance channels

| # | Idea (source) | Mechanism | What was measured | Class |
|---|---|---|---|---|
| B1 | A better all-distinct κ3 birth law (the 25 Sep "option B" prompt, audit D-8) | Fix the 9–16% self-error of the post-ReLU all-distinct κ3 (91–96% of the D21 error) | Fable R7/R8: 55% unexplained through second order. K×K negligible (D9). Curvature incoherent (D3). DHMP (D12). The B1 carrier misses 27% (D22). Slice + d3 oracle ×0.61 (D23). The prompt was never written (Round 13 went to κ4) | NEVER-TESTED (umbrella, no mechanism) |
| B2 | Variance channel as its own lever (oracle −47%) | Fix the post covariance | Q5: the closure is exact given true inputs; 81% of the error is inherited. D1 covariance patch dead; Q10 not low-rank at the decisive rows | DEAD-SOUND |
| B3 | Reallocate rank/age by layer to buy κ3 accuracy late (D23, 25 Sep) | Spend cost where the error is born | Gain is spread over L8–16. The exact old tier gives only −3.9% (R28) | DEAD-SOUND |
| B4 | Refit 504aldo's disabled online mean correction (F47; Q19) | 13 per-neuron features, ridge, in-loop | Teacher-forced R² 0.02 at L8/12/15 (chance ~0.013) | DEAD-SOUND for linear per-neuron features |
| B5 | **Nonlinear drift corrector with neighbour features** (Puffi P1 §D: a 31→32→32→1 MLP; features include the top-8 correlated neighbours S_i(·) and kernel-weighted neighbour averages P_γ(·)) | Predict the one-step mean drift from lower-order state, in the loop | Only B4 (linear, per-neuron) and R9 (GBM on the **final-layer** residual, no neighbour features) were run. Mean-forcing oracle: −81% (Q19 by-product) | **DEAD-NARROW** |
| B6 | **Per-layer end-to-end refit of zero-FLOP closure constants** (queue Q9 "joint knob search"; Puffi future work) | Fit λ_l, the κ4-diagonal use weight, FB/β per layer (or per depth block) to the final or consumed mean error rather than to cumulant accuracy | Only global scalars were swept: λ×0.9/1.1, BETA, FB×β grid (flat), λ 0.85 (D22). V39 was a per-layer table but was fitted to κ4 **level**, and that hurt. Puffi: refitting Λ to the consumed mean correction instead of κ4 marginals gave **−13.6%** layer-32 MSE (26/30 networks) | **DEAD-NARROW** (Q9 parked P3 on the global-knob evidence) |
| B7 | Learned output/residual corrections, HF corpora (R9, R15, V33) | Post-hoc per-neuron correction | 100-MLP CV: ridge −0.3%, GBM −0.5%; single-feature corr ≤0.04 | DEAD-SOUND |
| B8 | Compensating closure without the old tier (Astra Extra High Q3; board A1) | Retune λ/β/γ/age to compensate | noold 7.4e-7 best (×35 worse), including the corrected γ sweep | DEAD-SOUND |
| B9 | Richardson extrapolation in rank (own, 23 Sep; R16) | Cancel the leading truncation term | Exponent unstable; every extrapolation worse | DEAD-SOUND |
| B10 | S = 2 partner channel (Astra R9; Q15) | Find the error that masks the fully restored Edgeworth term | Not run. Astra's fit gives E(t) = 1 − 0.034t + 0.068t², so the ceiling is ~−3.4% even if the partner were free | PARKED, sub-threshold |
| B11 | Variance super-closure (Sol High R7; Q13) | More bivariate diagrams | Not run. Q5 ceiling: 9–13% of the closure residual | PARKED, sub-threshold |
| B12 | Weight-only angular attractor (forum 18188, via the 25 Sep forum audit) | A weights-only direction aligned with the final mean | Not run. It is likely covered by R9: the attractor is aligned with the mean direction, and the pred/global-scale feature gave −0.3% | NEVER-TESTED (low prior) |
| B13 | Per-network λ from an MC probe (Fable R1 H3; board A2; audit D-12) | Calibrate per network | Per-network offsets explain 0.44%; 10⁴ samples cost 156 u | DEAD-SOUND |
| B14 | Exact / unconfined old tier (board A5, R28) | Remove truncation | MSE −3.9% for +19% cost; float64 drift 2.9e-13 | DEAD-SOUND |

### C. Cost side

| # | Idea (source) | Mechanism | What was measured | Class |
|---|---|---|---|---|
| C1 | Re-form old sources only in the last k layers (Fable R1 C1, 21 Sep) | Old tier ≈ free | keeplast 1/2/3: 4.9/4.2/3.2e-7 (sensitivity map, K3b/K3c) | DEAD-SOUND |
| C2 | Compress the old tier: Tucker/TT/CP/ISDF/vector-metric/aggregate state/rank ladders (Rounds 1–4, Terra, Gemini, Astra, pilots A–D, H; D20; 504aldo F88) | Fewer columns for old κ3 | Products r99 680–780 (pilot A). Vector-metric k=128 20% (pilot D). Pooled core net-worse at every rank (pilot H). ISDF 56–94% (R6). CP needs ≫250 (D20). TT rank 256 reaches 1.6%, but middle-core transport costs r²n² ≈ 32 u per layer. Sym³ irreducibility (D15) | DEAD-SOUND |
| C3 | Temporal/age compression: FHQR-2, age modes, shift carrier (Sol R9; D13/D14/Q18a) | Short or low-rank age memory | h=10 still 16% at L15; r6 age modes 8.5%; shift ε 2.9%; the total κ3 is already carried exactly | DEAD-SOUND |
| C4 | Source-free regeneration: DHMP, particles, diagonal-only, RSC, channel-history (Sol R9, Round 11; D12, D16–D19) | Drop the source bank | 36–40% D21 (DHMP); 74–160% (particles); 83–99% d3 (diagonal); 36% (RSC); Astra "no mechanism" | DEAD-SOUND |
| C5 | Young tier at half cost (Round 10; Q17/D21) | Reorder or share propagators | Exact order search: 193 square products in every order (Sonnet + 4 external answers) | DEAD-SOUND |
| C6 | Column screening / unbiased HT sampling of births (own; pilot F, R19) | Fewer birth columns | Deterministic: net negative. HT halves the error but is still +12% raw for −5.6% C | DEAD-SOUND |
| C7 | Asymmetric contraction rank (Fable R6; V38, R21) | Legs at 384, contraction at r_c | End to end: util +0.8%, raw +2.9% | DEAD-SOUND |
| C8 | **Live-neuron compaction, output side** (forum audit item 1 lever (b); Q20) | Form only the D21 rows i with \|α_i\| < τ: row i enters the next closure weighted by w2_i ∝ φ(α_i). Keep the columns and d3 | Only mode 2 was run: **rows AND columns** of neurons with Φ < 1e-2 (α < −2.33, where φ is still 6.6% of its peak): +1.65% / +2.59% (`teacher_eval_q20dead3.json`; code at `estimator_q20dead.py:1163`). **Rows-only at \|α\| > 3 (φ ≤ 1.1% of peak; 1e-5 of Σw2², forum audit table) was never run** | **DEAD-NARROW** |
| C9 | Live-neuron compaction, input side (Q20 step 1) | Drop dead neurons from the next-transport input and the newborn birth columns | Φ < 1e-2: −0.37% / −0.59% (safe). Saving 10–16 u estimated, not metered; the build has not started | PARKED |
| C10 | Join without the range finder ("transport"; Q16) | Carry the old basis | 100-MLP ×0.959; graded −3.4% (#332257) | DONE |
| C11 | Knob trades: AGE/R/QPASS, layer-dependent age gate, per-family Strassen (Q3, Q21, R12, R17, R18) | Reallocate within the chain | One-for-one law (C5 in the Blueprint). AGE_OLD 5: −1.05% on 100 MLPs (submitted #332318) | DEAD-SOUND (as a moonshot) |
| C12 | Deeper or faster matmul: L6, memory-saving L6, rank-40, alternative-basis Strassen (D2, D25, forum item 4) | Fewer billed FLOPs per product | ≤1–2%; memory and residual caps | DEAD-SOUND |
| C13 | Precision / packing (Fable R1 H1, Astra R10; R7) | Cheaper arithmetic | float16 billed as float32; float64 billed 2× | DEAD-SOUND |
| C14 | Symmetric-tag / alias pricing (504aldo F77) | Gram/sandwich discounts | Crumbs of 3–5%; the weighted gram raises SymmetryError at scale 1 | DEAD-SOUND |
| C15 | Quadratic-chaos Galerkin, QCG (Sol R9 architecture 3) | h_i ≈ u_iᵀξ + ξᵀH_iξ with H_i in a common n-mode basis | Not run. Its κ3 = Sym Σ_r (Ub_r)⊗(Ub_r)⊗S[:,r] is an n-term CP of a⊗a⊗c form. F76: a CP cap at 4n already leaves 39% D21 error, and 2% needs ~8n. Sol's own gate is ≤3% at early layers | DEAD-SOUND (by F76 arithmetic) |
| C16 | Lazy cubic circuit (Sol, 24 Sep; D15) | Store births + M, read by reverse mode | Counted, not run: ~480 products ≈ 245 u (the dual order of the chain) | DEAD-SOUND (counting) |

### D. Non-cumulant families and sampling

| # | Idea | Measured | Class |
|---|---|---|---|
| D1 | GPT Sol R5 (14 families) and Astra R5b (6 families): TAP/AMP, CF, saddlepoint, copula, GMM, polytope, EP, TT-PCE, SOS, input chaos | K3s/K3t/K3u: TT-cross 60–500% at width 32; copula worse at every layer; the rest by argument or Astra's executed gates | DEAD-SOUND |
| D2 | MC, control variates, hybrids (board C1/C2; Q11; D2; D6c; 504aldo F62) | Surrogate corr 0.19–0.73 (0.997 needed); 10⁴ samples = 156 u for 1.4% variance error | DEAD-SOUND |
| D3 | Gaussian-mixture closure (504aldo F78) and Puffi's A716 slice closure (504aldo F82) | 2.3e-6 and 9.8e-7 at the P2 shape | DEAD-SOUND |

### E. Strategic

| # | Idea | Status | Class |
|---|---|---|---|
| E1 | Team merge (option D, 25 Sep 09:30; direction review B-11) | The user chose "A, C"; merge was never decided explicitly. Freeze: 3 Oct 05:59 BST. §1 shows chain-derived teams holding a −15 to −17% raw and −20 to −27% util package | PARKED (user decision, deadline) |

**Counts (47).**
- DEAD-SOUND (32): A2, A3, A5–A9, A12; B2–B4, B7–B9, B13, B14; C1–C7, C11–C16; D1–D3.
- DEAD-NARROW (3): B5, B6, C8.
- PARKED (6): A4, A10, B10, B11, C9, E1.
- NEVER-TESTED (5): A11, B12, and the umbrellas A1, A13, B1. A11 goes here rather than under DEAD-NARROW because no test ever ran its actual form; the tests cited against it are too narrow.
- DONE (1): C10.

---

## 3. Ranking of the live items (DEAD-NARROW, NEVER-TESTED, PARKED)

Score = (plausible gain × probability) / cost-to-test. The top 5 are restricted to items whose success case meets or nearly meets the moonshot bar. The probabilities are judgement calls, stated so that they can be argued with.

| Rank | Item | Gain if it works | P(success) | Cost to test | EV per hour | Why this probability |
|---|---|---|---|---|---|---|
| 1 | C8 + C9 live-neuron compaction (rows-only oracle, then metered emulation) | −6 to −12% util at flat raw ≈ −6 to −12% score | 0.5 | 2.5 h | ~2%/h | Physics: w2 → 0 for \|α\| > 3. The input side is already validated. The risk is that always-on rows feed other terms, and residual time |
| 2 | A11 SSC κ4 law (teacher-forced diagonal test) | −8 to −15% raw at ≤1 u (n²-only form) | 0.08 | 3 h | ~0.3%/h | For: all-distinct sector, a published −64.6%, cheap. Against: Sym(C⊗C) alone has ~0 corr; the pattern needs ~6 steps of memory (Opus §7.5) |
| 3 | B6 per-layer end-to-end constant refit | −3 to −8% (≥10% unlikely) | 0.15 | 4 h + 1 Kaggle run | ~0.2%/h | For: Puffi's −13.6% and the zero-FLOP jumps in §1. Against: the global knobs are flat |
| 4 | B5 neighbour-feature drift corrector | −3 to −10% | 0.05 | 2 h | ~0.15%/h | The −81% mean-forcing ceiling exists, but the per-neuron R² was at chance |
| 5 | A10 Q22 pair innovations | −10 to −25% | 0.04 | 4 h | ~0.15%/h | A new subspace, but its sibling D26 scored corr 0.03 |
| 6 | B12 angular attractor | ≤ −5% | 0.01 | 0.25 h | ~0.2%/h | 15-minute closer; probably already covered by R9 |
| 7 | A4 Q14 mixed-coordinate readout | ≤ −2% score | 0.3 | identity check 0.5 h; real test needs the 1–2 day build | low | Sub-threshold |
| 8 | B10 Q15, B11 Q13 | ≤ −3% | — | 1 h | low | Sub-threshold |
| — | E1 team merge | −15 to −40% score (others' measured packages) | depends on a counterpart | user time | — | Not an experiment; deadline 3 Oct 05:59 BST |

---

## 4. Top-5 falsifier plans (all on stored data or the existing harness)

### F1. Live-neuron compaction, rows-only (C8), then metered emulation (C8 + C9). About 2.5 h.

1. **Oracle (30 min, 0 u).** Add bit 32 to `experiments/variants/estimator_q20dead.py`. It zeroes only the D21 **rows** i with α_i < −τ (dead), and separately bit 64 for rows with α_i > τ (always-on). Keep all columns and D3.
   - Run with `experiments/chain/run_variant_teacher.py` on MLP #0 + the synthetic net, paired, τ ∈ {3.0, 3.5}.
   - Also run the combination with the already-safe mode 9 at Φ < 1e-2.
2. **Gate.** Final MSE change ≤ +0.3% on both networks at τ = 3 for the dead rows; always-on rows are reported separately.
   - **Kill** if dead rows alone cost > +1% at τ = 3.5.
3. **Metered emulation (2 h).**
   - Take the per-layer counts m_l = 99th percentile of #(|α| < τ) over MLPs. Use `errbudget_v29plain.npz` for MLP #0 now; use a 100-MLP α dump on Kaggle for the final table (direction review B-2). Pad to multiples of 32.
   - Meter the young D21 contractions (50.5 u) and the old-tier contraction (the LA/LP left factors, rows i) with `experiments/probes/m1_matmul_ledger.py`, together with the input-side columns.
   - **Gate:** ≥5% metered util for the combined stack. **Kill** below 4%.
4. **Expected.** The source-weighted share of D21 rows with |α| > 3 is ~20%, of which the dead half is ~10%, giving 4–7 u from the output side. With the input side (10–16 u), the stack is 14–23 u (−6 to −10% util). This is what shiv_m's position (our raw at util 0.179–0.196) shows exists on a V29-class chain.

### F2. SSC κ4 law, teacher-forced (A11). About 3 h.

1. **Data.**
   - Post-ReLU teacher at l ∈ {3, 7, 11, 14, 15}: `data/kaggle_out/kaggle_out_k4post/k4post/layer{l}_post_seed0.npz` (mean m, C, Y21 = κ3(i,i,j), K4d).
   - Pre-activation teacher at l+1: `…/k4pub/k4teach/layer{l+1}_seed0.npz` (K4d target, mean, C → α, σ, G4 = (α²−1)φ(α)/σ³).
   - Weights: `weights_seed0.npz`. Use the transport convention of `r9_dhmp_gate.py`.
2. **Features.** For each next-layer neuron j (w_j = incoming weights):
   - q1 = w_jᵀCw_j
   - q2 split into its six parts: w_jᵀ(C⊙C)_off w_j; w_jᵀ Sym(D_m Kᵀ) w_j with K = Y21; (w_j·t)(w_j·m) with t = diag Y21; (w_j·v)²; v̄²(w_j·1)²; the v̄C part
   - q3 = v̄‖w_j‖²
   - q4 = Σ_a (w_ja g_a)(Cw_j)_a, with g the standardized centered Φ(α_l)
   - Model: κ̂4_j = 3 Σ_ab Λ_ab q_aj q_bj, i.e. ~45 product features, ridge.
3. **Fit/test.** Leave one layer pair out, on:
   - (a) the raw target, teacher K4d;
   - (b) the residual target, K4d − chain `L{l+1}_ck4` from `errbudget_v29plain.npz`. This shows whether SSC adds to the chain's transported dG.
4. **Gate.** Out-of-pair decisive-neuron (|α| < 1.5) corr ≥ 0.60 **and** errG4 ≤ 0.15 at L15 and L16.
   - Reference points: chain 0.26–0.32 / 0.199–0.210; ideal h=3 young carrier 0.54 / 0.188; ideal linear carry (LIN) 0.79 / 0.12–0.13.
   - **Kill** if corr < 0.45 or errG4 > 0.18 at both L15 and L16.
5. **If it passes.** Build the n²-only version. q1, q3, q4 and three q2 parts reuse V29's existing W·C product; drop the C⊙C and D_m Kᵀ parts, which need 2 extra products per layer ≈ +15 u.
   - Compute q from the chain's own state (captures `r7_cap_L{7,11,14,15}` hold C_post, K21, K1–K4, mu).
   - Inject κ̂4 as the κ4 diagonal through the corrected oracle harness (`k4_oracle_run2`, diag mode).
   - **Gate:** raw ≤ −5% on both teacher networks at ≤ +1% util.
   - Then check the κ4→κ3 feed as well. SSC's (2,1,1) sector could replace the regen G in the X3 feed, the part F68 found "carries the gain".

### F3. Per-layer end-to-end refit of zero-FLOP constants (B6). About 4 h plus one Kaggle run.

1. **Variant.** A new `make_*.py` variant of v42tr with 12 multipliers: λ_l, the κ4-diagonal use weight (g4row), and FB_SCALE, each over 4 depth blocks (L2–5, 6–9, 10–13, 14–16).
2. **Gradient.** Forward finite differences at ±10% on MLP #0 + synthetic (`run_variant_teacher.py`, paired, bit-identical base): 24 runs per network, about 2–3 h locally. Fit a separable quadratic and take one Newton step, capped at ±30%.
3. **Gate.**
   - At least −3% on **both** networks after the step (a cross-check: fit on one network, apply to the other).
   - Then Kaggle 100-MLP paired: score ≤ ×0.98 and < 1 − 2 SE.
   - **Kill** if the held-out network improves < 1%.
4. **Optional second target.** Following Puffi, minimise the G4-weighted κ4-diagonal error in the consumed mean term instead of the final MSE. It is cheaper to evaluate on the teacher arrays and needs no chain runs.

### F4. Neighbour-feature drift corrector (B5). About 2 h.

1. **Dump (30 min per network).** Re-run the teacher-forced dump (`experiments/variants/estimator_q19corr.py` via `experiments/chain/q19_run.py`) on MLP #0 and the synthetic net. Save, per layer, the 13 base features, the fresh mean error (target), and C_pre.
2. **Features.** Add Puffi's neighbour features: for each i, the 8 largest |R_ij| (pre-activation correlations), with S_i(x) = Σ_t r_it x_{j_it} and U_i(x) over x ∈ {α, Φ, φ, η = κ3/σ³, log σ-ratio}, plus the kernel averages P_γ(x) for γ ∈ {8, 32}.
3. **Fit.** A small numpy MLP (or ridge on pairwise products) on one network, tested on the other, layers ≥ 8.
4. **Gate.** Cross-network R² ≥ 0.10. Chance level is ≈0.013; Q19 reached 0.02 in-sample. **Kill** if R² < 0.05.
5. **If it passes.** In-loop correction variant, then a paired teacher eval, then the 100-MLP harness. Cost is O(n²) per layer, consistent with the zero-FLOP jumps in §1.

### F5. Q22 carried pair innovations, oracle (A10). About 4 h.

1. **Oracle, as planned in the queue.**
   - At L14 (`r7_cap_L14.npz`: 13 sources, dense P legs), fit γ_s, η_s by LSQR to the teacher's post-ReLU K22/K31 off-diagonals at L14 (`layer14_post_seed0.npz`), as in the D26 oracle `r13_skewlift_oracle.py`.
   - That costs one n³ product per source per LSQR matvec, ~1.5–2 h CPU.
   - Then predict the L15 pre-activation κ4 diagonal with the readout q = Σ_s [6 (Z_sγ_s) ∘ rowsum(Z_s) + (Z_s∘Z_s) η_s], Z_s = (transported P_s)∘(transported P_s).
2. **Gate.** Decisive-neuron corr ≥ 0.70 against the teacher (chain 0.26; D26 0.03). **Kill** below 0.45.
3. **Cheap pre-check (20 min).** Fit only per-source scalars γ_s, η_s (26 parameters). If even the best 26-parameter fit gives corr < 0.3, stop.

---

## 5. Leaderboard consistency: which ideas the facts support and which they rule out

### Fact A: util ~0.15 at raw ~2.0e-8 by 6 Sep (Puffi 0.148, suliman 0.150); suliman at 0.134 by 13 Sep

**What it is.** Two layers:
- The 4–6 Sep drops (−15 to −30% at flat raw for every resubmitting team) coincide with Strassen adoption. We already have that.
- Underneath is a base-representation gap: 0.207–0.213 vs 504aldo's 0.367 before Strassen, 0.15 vs our 0.227 after. That is 1.5–1.75×.

**Consistent with it.**
- 504aldo's F86 arithmetic: the leaders' bill ≈ V29 without the old tier (153 u = 0.150 B).
- In our record only the umbrella A13/M3 fits: an expansion whose old κ3 content does not cost per-source dense legs.
- SSC (A11) does **not** explain the cost. Puffi's P1 SSC cost 1.34× a K3 port; it is an accuracy mechanism.

**Ruled out as the explanation of Fact A** (each is already in our chain or capped far below 1.5×):
- Strassen depth; precision; alias pricing.
- The knob trades: one-for-one.
- Join trims.
- Young-tier reordering: the 193-product floor.
- Every linear compression of κ3: the Sym³ wall.
- The compaction stack: C8 + C9 ≈ −10%.

The best trim stack in the record (transport, done, + compaction + AGE_OLD 5 + crumbs) reaches util ≈0.19–0.20, not 0.15.

**What that stack does explain.** The near-chain cluster in §1 (shiv_m at our raw with util 0.179–0.196; fklassen and thibault_gounant at 0.20). So F1 is the record's route to the ~0.19 tier, not to 0.15.

### Fact B: large accuracy steps at fixed util (J2W −11.9% in 27 h; jlacombe −5.5%; AndreasHad04 −7.2%; emanuel_ruzak −4.6% after a −11% first step)

**What it is.** J2W's util stayed within ±0.3% (≲0.5 u). jlacombe's and AndreasHad04's util matched to 4 digits. One dense Strassen product per network would move J2W's util by +0.37%, and one per layer by +5.5%. So these steps cost at most O(n²) per layer: constants, closure coefficients, per-neuron corrections, or cheap κ4/mean laws.

**Consistent with it** (and still open for us):
- A11 SSC in its n²-only form (F2).
- B6 per-layer constant refit (F3), which matches the monotone step shape of J2W and emanuel.
- B5 in-loop drift corrector (F4).

**Consistent with it, but no longer a lever for us:**
- The spherical-input trick. It is public in barnobarno666's repo (last updated 13 Sep), which reports −7.9% on an independent K3 chain; J2W's jump followed 4 days later. It gave us −1.1% graded, because V29's vec+ww radial channel already covers the radial κ4.
- The F68 κ4→κ3 feed/regeneration. AndreasHad04's own post names it; V29 has it.

These alternative explanations mean Fact B alone does **not** prove that a zero-cost lever exists for V29.

**Ruled out as the source of these steps** (each adds ≥1 dense product per layer and would have shown in util):
- Q8 (≥13 u), Q14 (17–34 u), the young κ4 carrier (~28 u), the Astra kernel (≥14 u).
- The exact old tier (+19%), R_FB 32 (+6%), AGE_OLD 5 (+4%).
- Q22 (~0.35 u, +0.25% util) is borderline: detectable at 4 digits, but not excluded if applied at a few layers only.

### Fact C (new): a V29-derived team found both levers within 4 days

emanuel_ruzak's trajectory (§1) starts at V29 verbatim: −11% raw and −8% util in the first own step, and a 40% better score by day 4. Nothing in our record reproduces a −11% raw step from V29. Our best zero-cost gains are sphere −1.1% and FB 1.25 −0.37%, and our knob and oracle searches bound every other cheap edit we tried at ≤1%.

This is the strongest evidence that a ≥10% accuracy lever compatible with V29 exists and is unknown to us. Its signature (a large step, then small zero-FLOP refinements) matches F2 and F3 more than any cost-bearing idea. Caveat: we cannot see whether that first step replaced part of the chain.

---

## 6. Record corrections found along the way

- **`consultations/forum_audit_2026-09-25.md`, appendix, "Puffi SSC κ4 response modes — 18175 → findings F68 trigger" (COVERED).** This is inaccurate. F68 took the SSC idea only as a trigger and tested a linear mode projection under the Sym(G⊗mI) use side. SSC's bilinear Sym(B_a⊗B_b) use side, its marginal-κ4 fit and its response memory were never evaluated. See A11.
- **The brief's "−8% in a single submission" for J2W.** The largest single step was −5.4%; the robust figure is −11.9% over five submissions in 27 h (§1).
- **`docs/build_notes.py` line 1175.** It still says "the deep kappa4 is REGENERATED at every ReLU, not carried from its births". Opus R12 (Falsifier 2) and Q18 show the opposite at depth: fresh physics gives 1–4% of the level, and the linear carry of the earlier κ4 gives 83–90%.
- **`Round 6b` prompt (`consultations/prompts/ROUND6b_prompt_astra_error_vs_cost.md`).** No answer was ever filed. It is analytic, with no direct score value; close it or send it.

*Scratch artefacts (session scratchpad only):* `convo_extract3.txt`, `extract3.py`, `queue_dump.txt`, `puffi_ssc.txt` (text of the public Puffi PDF), `rec2/*.html` (44 public submission pages used for attribution). No repository file other than this report was written; nothing was committed, pushed or run on Kaggle.
