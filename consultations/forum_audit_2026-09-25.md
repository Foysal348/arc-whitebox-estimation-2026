# Forum audit against our knowledge base — 25 Sep 2026

**Scope.** The whole AIcrowd category (discourse.aicrowd.com/c/2991): **93 topics (17981 to 18219), 275 non-empty posts** (288 including empty system posts). Everything was read, oldest to newest, Phase 1 and Phase 2, with all replies. Fetched 25 Sep 2026 at about 10:15 Bangladesh time (04:15 UTC). The raw dump (one `.txt` per topic, plus the JSON) is in the session scratchpad: `.../scratchpad/forum_dump/` and `.../scratchpad/all_forum.txt`.

**Knowledge base checked:**
- `docs/build_notes.py`. Part G digests all 28 Phase 2 threads, read on 21 Sep. Phase 1 is covered only where it transfers.
- `docs/experiment_queue/queue.json` (updated 25 Sep).
- `docs/moonshot/Moonshot_Blueprint.html`.
- `whest-p2-cumulant-k3/docs/findings_log.md`. The local copy is at `18c17e2`. Upstream has one more commit, `1a4083f` (F105, 22 Sep: the F68 claim corrected). F105 is already digested in notes K3x/K3y, so nothing is lost.

**Labels.** COVERED = the KB states the item and acts on it. PARTIAL = the KB has a related piece, but not this piece or not this use. NOT COVERED = no trace. "Post says" means the post states it. "Inference" means my reasoning, not stated by anyone.

**Bottom line.** The KB already covers the Phase 2 forum almost completely. The real gaps are:
- one Phase 1 structural idea that transfers to our chain as a **cost** lever (item 1);
- two final-evaluation rulings nobody has written down (items 2 and 3);
- a few low-value leftovers.

Item 1 is the only one I would spend a full day on.

---

## (a) NOT COVERED and PARTIAL items, most promising first

| # | Status | Source | Claim (short) | Matters for | Cheapest test |
|---|---|---|---|---|---|
| 1 | PARTIAL | 18106#1 natasha_stewart 27 Jul; 18149#1 konstantin_baltsat 11 Aug; 18157#1 (4c, amalgonim appendix) 13 Aug; 18176#1 ely2sh 17 Aug | Per-neuron **dead / on / kink structure** (from μ/σ) lets you drop work on inactive or linear neurons | **Cost**: est. 8–12 % fewer FLOPs (inference) | Offline mask emulation on 2 teacher MLPs plus a metered ledger run (~half a day) |
| 2 | PARTIAL | 18154#1 hyojun_kwon 12 Aug; 18216#1 yujigon 6 Sep; flopscope issue #257 (still OPEN on 25 Sep) | Symmetry-inferred ops **validate numerically** (`allclose`, atol 1e-6). They can raise `SymmetryError` on some inputs only, which zeroes that MLP | **Private-rerun robustness** | Local-only counter on the validation hook, then a full-split run |
| 3 | NOT COVERED | 18078#1 aicrowd_team 17 Jul (Townhall); 18125#1 mohanty 3 Aug | The final private evaluation runs on a **fixed (pinned) flopscope version**. The sponsor may **patch the cost model during final evaluation** and rerun everyone | **Rules and cost**: the metered FLOPs on which our utilisation is scored can change | Ask which version, then re-meter the designated tarball under it |
| 4 | NOT COVERED (rank-40) / PARTIAL (rewrites) | 18157#9 jamesrahenry 18 Aug (summarising andrei_bulzan 18183); 18157#8 (pranay212 18173); 18149#1 baltsat | Faster-than-Strassen bilinear schemes (Tichavský rank-40) and exact "compiler" rewrites | **Cost**: probably ≤1–2 % at our shapes (inference) | Desk calculation on the ledger's product shapes (15 min) |
| 5 | NOT COVERED | 18188#1 zacharymartinot 19 Aug; paraphrased in 18157#14 | The angular map s → normalize((W s)₊) has an **attractor computable from the weights alone**, strongly aligned with the true final-mean direction | Accuracy (speculative) | cos(chain error, attractor) on stored dumps (10 min) |
| 6 | NOT COVERED | 18180#1 violeta 17 Aug | A **100-machine parallel evaluator** (fly.io script; "Modal is probably slightly faster, more free credits"), a GPU flopscope fork (~2× local) and a residual profiler | **Iteration speed**: our 100-MLP paired checks are the bottleneck | Decide yes or no; no experiment |
| 7 | NOT COVERED | 18078#1 Townhall 17 Jul; 18159#1 baltsat 14 Aug; 18154#1 kwon 12 Aug | ARC's own open line: cumulant or Edgeworth series **diverge**, and ARC tried **Borel resummation**. baltsat measured Borel–Padé unstable at 256×32 | **Write-up framing** (algorithmic prize), not score | No experiment. One paragraph in the write-up |

### 1. Structure-aware gating of the chain's dense products (PARTIAL; cost)

**What the posts say (Phase 1, sampling estimators):**
- SOX (18106#1): moment propagation classifies neurons as dead, on or kink. On neurons are treated linearly in the last two layers, and activation sparsity is exploited in the matmuls. Graded raw 2.18e-7, adjusted 1.55e-7.
- baltsat (18149#1): a coordinate is omitted only when the analytic and pilot tests agree. The recurrence mean of omitted inputs is added back analytically. A dead/on/kink fold is used in the final three layers.
- amalgonim appendix, as summarised in 18157#1 item 4c: about 97.5 % of closure error sits in the ~66 neurons within 2σ of the kink (256×32).
- ely2sh (18176#1): a gate-support pilot removed about 8.3 % of effective compute at a 0.07 % raw-MSE change.
- **Warning.** 18157#9, quoting ely2sh, reports a sharp "support-removal cliff": firing threshold 16 cost 2.08× MSE for 7.2 % more narrowing, and threshold 64 was catastrophic (1.3e-5). In their words, "rarely firing units carry disproportionate downstream value".

**What the KB has:**
- K3h (birth-column screening by energy quantile, net negative).
- K3ab (700/1024 neurons at layer 16 have |μ/σ| > 1.5).
- Q8/Q14, which restrict a *new* κ4 term to a |μ/σ| < 1.5 query of about 350 rows.

Nobody has gated the **existing** transports and contractions by the operating point α = μ/σ. The shipped code has no masking: I grepped `estimator_v29.py` and `v41join.py` for take/mask/argsort.

**New measurement (this audit, zero cost).** The chain's own pre-activation μ and σ come from `data/dumps/errbudget_v29plain.npz` (MLP #0).

| layer | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|
| dead: Φ(α) < 1e-3 | 0.7 % | 3.3 % | 5.8 % | 6.5 % | 12.1 % | 15.8 % | 17.4 % |
| \|α\| > 3 | 1.2 % | 5.9 % | 11.5 % | 16.9 % | 23.4 % | 32.9 % | 36.9 % |
| share of Σ w2² (w2 = φ(α)/σ) carried by the \|α\|>3 rows | 1e-6 | 3e-6 | 6e-6 | 8e-6 | 1.1e-5 | 1.8e-5 | 1.3e-5 |
| share of Σ w1² (w1 = Φ(α)) carried by dead rows | 1e-8 | 1e-8 | 2e-8 | 1e-8 | 3e-8 | 3e-8 | 2e-8 |

**Two levers (inference):**
- **(a) Transports** P ← W d(w1) P. Drop the inner index where Φ(α) < 1e-3. Those rows are multiplied by w1 < 1e-3, so this is nearly exact. The same applies to the shared-basis formings and to the C_pre sandwich.
- **(b) D21 contractions** (LA Aᵀ + LP Pᵀ). Form only the rows i with |α_i| < τ. Row i of D21 enters the next layer weighted by the second Hermite weight w2_i. The |α|>3 rows carry about 1e-5 of that weight energy, far below the ε² ≈ 4e-4 tolerance implied by the 504aldo law (extra MSE ≈ 4.2e-6 ε²). Columns stay full: on-neurons still need D21[c, i].
- **Caveat.** The κ4 weight (α² − 1)φ(α)/σ³ decays more slowly: at |α| ≈ 3–3.5 it is 2–4 % of its peak. So test τ = 3, 3.5 and 4.

**Size (inference, ledger model).**
- Per-layer bill: young tier 2.17 u per source, old tier ~1.62 u per old source (fits 504aldo's 116 + 107 u).
- About half transport and half contraction per layer, with the savings fractions above.
- Result: about **30 u of ~249 u, roughly 12 %**, as an upper bound. The last layer is already trimmed, and shapes must be padded to multiples of 32.
- Realistic 8–12 %: utilisation 0.244 → about 0.215–0.225. Adjusted 5.13e-9 → about 4.5–4.7e-9 if raw MSE is unchanged (our notes put rank ~14 at 4.5e-9).

**Implementation constraint (inference, from 18202#4).** A data-dependent mask read on the host is a graph flush, and every flush is billed to the 0.4 s residual. So the mask sizes should come from a **fixed per-layer table m_l**, set above the 99th percentile across MLPs; keenanpepper (18199) shows the Phase 2 MLPs have no heavy tail. The mask itself stays server-side: `argsort(|α|)` (billed 4×n·log n, negligible) plus `take`, with pooled full-size buffers used through leading-slice views (views are free). FLOPs stay shape-deterministic.

**Cheap experiment:**
1. Take α per layer from the chain for 20 mini MLPs and set m_l (dead and kink counts) with a margin.
2. In the float64 lean chain, or as a variant on `estimator_v41join.py`, zero the inner rows with Φ < τ₁ (τ₁ ∈ {1e-4, 1e-3}) and the D21/LA/LP rows with |α| ≥ τ₂ (τ₂ ∈ {3, 3.5, 4}). Measure final-MSE drift against the unmasked chain on MLP #0 and on the synthetic net. **Gate: drift ≤ 3 % of raw.**
3. Metered run to get the real saving after padding.
4. Kaggle 100-MLP paired run, logging residual time and peak memory.

**Kill** if drift exceeds 3 % at τ₂ = 4, or the metered saving is below 4 %.

### 2. The flopscope #257 symmetry-validation cliff (PARTIAL; robustness of the private rerun)

**What the posts say:**
- kwon (18154#1): on 0.9.x, a symmetry-inferred three-operand einsum was validated with atol 1e-6. It failed on one public-mini MLP after about 19 layers of float32 cancellation, and the forfeited MLP moved their 100-MLP proxy from 3.6e-7 to 6.6e-2.
- yujigon (18216#1) filed flopscope #257. On GitHub (read 25 Sep) the issue is still open: on 0.12.1, the symmetry check is `np.allclose(atol=1e-6)` on near-zero off-diagonals, and "the same input can pass on one platform and fail on the other".

**What the KB has.** Notes G3.2 mention #256/#257 as yujigon's bugs. The findings log (F-gram probe) shows that an aliased weighted Gram raises `SymmetryError` at scale 1.0, and 504aldo avoided that path. But no audit of **our** shipped files exists.

**What I checked:**
- `estimator_v29.py:1405` and `v41join.py:1458` tag C with `as_symmetric`. C is built as (K11 + K11ᵀ)/2 plus `fill_diagonal`, so it is bit-exactly symmetric. Pointwise ops on it are safe.
- `v41join.py` adds `fnp.linalg.eigh(Gz)`. Flopscope 0.12.1 `eigh` does **not** validate symmetry (checked in `whest-starterkit/.venv/.../numpy/linalg/_decompositions.py`).

So the risk looks low. But a single failed MLP in the private rerun adds about 0.9/N to the mean MSE, which is catastrophic.

**Cheap experiment.**
1. In the **local harness only** (never in the submission), wrap `flopscope._pointwise._validate_result_symmetry` and `_prepare_symmetric_out` with a counter.
2. Run 3 MLPs of the nomination candidate. If the count is above zero, list the call sites and either symmetrise exactly or drop the tag.
3. Do the full-split run the notes already plan (L631/L650) on the exact nomination tarball.

### 3. Pinned flopscope for the final evaluation, and patches during it (NOT COVERED; rules and cost)

**What the posts say:**
- Townhall summary, 18078#1, 17 Jul: "Final decisions will be based on the private evaluation round, run on a fixed version of FlopScope with full reproducibility details published."
- mohanty, 18125#1, 3 Aug (still standing): "During final evaluation, we may also make targeted changes to the cost model if we identify an exploit … patch the gap, apply the revised rule consistently to all affected submissions, and rerun them."

**Why it matters.**
- Our score is raw × metered utilisation, so any repricing of Strassen additions, `fill_diagonal`, `take` or `where` moves our number.
- The evaluator runs flopscope 0.12.0 (18202#2), while the kit pins 0.12.1. PyPI's latest release is still 0.12.1 (29 Aug).
- flopscope #256 and #257 are open. Fixing them could ship a new version.

**Action.**
1. Email arc-whestbench@aicrowd.com: which flopscope and whestbench versions will the Phase 2 private rerun pin, and will they be announced before 17 Oct?
2. Keep a 3-MLP re-meter script: `flops_used`, residual, peak RSS for the designated tarball under any given version.

### 4. Fast-matmul beyond Strassen–Winograd, and exact "compiler" rewrites (NOT COVERED / PARTIAL; cost, low)

**What the posts say:**
- andrei_bulzan's report, as quoted in 18157#9: a rank-40 rectangular identity (Tichavský, arXiv:2104.05323) takes 54 → 40 block products. Nested twice over rank-7 leaves this is "48 % of ordinary leaf products (82.08B → 49.45B including transforms)", with bit-identical predictions and an execution-layer trim of 0.65–0.79 %.
- pranay212 (18157#8): an exact-algebra pass (one-write assembly, Winograd formulas, sparse packing) cut billed FLOPs by about 6.7 % and gave +7.5 % adjusted.
- baltsat (18149#1): four exact rewrites gave −3.1 %.

**What the KB has.** Strassen is covered: 504aldo F79–F81, our STRASSEN_MIN 8 and FIRST 5, and "pricing polish ≤ 4 % on paper, 0 % shippable" (504aldo row 23, K3ah).

**Why the value is probably small (inference, my arithmetic).**
- At 1024³ with 5 levels and 32-leaves, leaf multiplies are about 93 % of a product's bill and the 15 additions per level about 7 % (8.1e7 of 1.165e9).
- 1024 = 2¹⁰, so 3-way splits need padding, and rank 7 is already optimal for 2×2 blocks.
- The one thing left is additions. An alternative-basis Strassen (Karstadt–Schwartz: 12 instead of 15 additions per level; my addition, not from the forum) saves at most about 1.5 % before its basis-change cost.

**Test.** A 15-minute desk calculation over the actual family shapes (n×n×n, n×n×384/224, n×n×16). Drop it unless the calculation shows more than 3 %.

### 5. Weight-only angular attractor (NOT COVERED; accuracy, speculative)

**What the post says** (as paraphrased in the census, 18157#14; the claim itself is in the 18188 PDF, which I did not open): s → normalize((W s)₊) is contracting. Under periodic extension it converges to an attractor computable from the weights alone, "highly correlated with the direction of the true final mean". The author adds: "not sure how it could be exploited".

**What the KB has.** Our angular work is input-level only (V31 spherical input, shipped, −1.1 % raw). K3v says the chain's residual is unpredictable from cheap **per-neuron** features. The attractor is a network-level vector, so it has not been tried.

**Test (10 minutes, zero FLOPs).**
1. On the stored `errbudget` dumps (MLP #0 and synthetic), compute the attractor a.
2. Compute cos(e_final, a) and cos(e_final, pred), and run a leave-one-network-out scalar fit.
3. Drop it if |cos| < 0.1. The prior is low.

### 6. Parallel remote evaluator and other tools (NOT COVERED; process)

**What the post says** (violeta, 18180#1): `github.com/01-1/arc-wbe/blob/main/cloud/fly_runner.py` evaluates one MLP per remote machine on 100 machines; the author recommends writing your own and suggests Modal. There is also a GPU-accelerated flopscope fork (`github.com/01-1/flopscope`, about 2× locally) and a residual-time profiler.

**Why it matters.** The notes (K3z-B, queue Q3/Q9) show that our paired 100-MLP decisions wait on the Kaggle harness. **Caveat:** FLOP and residual numbers from a fork or a cloud box are not the grader's. Use them only for relative raw MSE.

**Action.** A yes/no decision only.

### 7. Borel resummation / divergence of the cumulant series (NOT COVERED; write-up)

**What the posts say:**
- Townhall (18078#1): higher-order Edgeworth "improves results initially but eventually diverges. ARC has explored Borel resummation … believe this matters more for very narrow networks (e.g. width 16)".
- baltsat (18159#1): the Borel–Padé ladder at 256×32 had a mean drift of 9.765e-6 against a 1.650e-6 bound, and only 0.297–0.398 of cases were stable.
- kwon (18154#1): K=3 gain over K=2 collapses 46× → 1.3× from depth 4 to 32 at width 256. kwon calls this a counterexample to ARC's width-16 remark.

**Why it matters.** It gives the algorithmic write-up a hook ARC itself raised. Our K3w (the K3 → K4 gain grows with width) and K3x/F105 (λ is a compensator) speak directly to it.

**Action.** No score experiment. One paragraph in the write-up.

---

## (b) Organiser rulings relevant to the final evaluation, prizes and team changes

Clock times are UTC, with Bangladesh time (UTC+6) in brackets. Coverage in the KB is noted in square brackets.

1. **23 Aug, 18197#1 (Phase 2 launch).**
   - Submissions close 17 Oct 23:59 UTC (18 Oct 05:59 BST).
   - **Registration and team freeze: 2 Oct 23:59 UTC (3 Oct 05:59 BST).**
   - Write-up due 24 Oct 23:59 UTC (25 Oct 05:59 BST).
   - All computation must go through flopscope. λ = 0 and the residual cap is 0.4 s; a breach zeroes that MLP. Wall cap 120 s, memory 8 GB, 10 submissions per day.
   - Review continues after grading. "Unusual residual wall-time usage receives additional scrutiny."
   - Mispricing must be reported, not worked around.
   - [COVERED: A1, B, G1]
2. **3 Aug, 18125#1, edited 7 Sep:** "Registration and the team freeze have not closed. Both are 2 October 2026, 23:59 UTC." [COVERED]
3. **31 Jul, 18118#2; restated 3 Aug, 18125#1.** Prizes come *only* from a private rerun:
   - fresh MLPs, from private seeds not used in either phase;
   - same distribution, "same width and layer-count ranges" and same budget calibration;
   - suite size set by statistical analysis;
   - ties are resolved by adding more MLPs.

   [COVERED: A4 L124. "Ranges" is not noted, but Phase 2 has a single shape; our shape gates keep a fallback.]
4. **17 Jul, 18078#1 (Townhall).**
   - The designation locks at the Phase 2 deadline. [COVERED]
   - **The final private evaluation runs on a fixed flopscope version, with reproducibility details published.** [NOT COVERED → item 3]
   - Winners must release their code under an OSI licence; declining forfeits the prize to the next team. [COVERED]
   - **ARC will review roughly the top 10 submissions (possibly more)** for the algorithmic prize. [PARTIAL: J2 says only "starts from the top-scoring".]
5. **3 Aug, 18125#1.** During final evaluation the cost model may be patched for exploits, "applied consistently to all affected submissions" and rerun. Finding an exploit is not by itself grounds for disqualification. [NOT COVERED → item 3]
6. **10 Aug, 18143#1 (Phase 1 mechanics; expect the same for Phase 2):**
   - team-owned selection slots, a panel under Submissions → "Your team";
   - selection closed one day after the phase;
   - a failed submission can be nominated explicitly;
   - with no pick, the top public entries are used.

   The Phase 2 slot count is "up to two" per the Overview page. [COVERED: G1, A4]
7. **23 Jun and 6 Jul (edited 7 Sep), 18041#1/#4. Algorithmic prize criteria:**
   - One PDF plus **exactly one** graded submission ID, sent by email or posted on the forum.
   - The week after the close is for writing only.
   - ARC will "award the prize to the submission where novel mechanistic ideas made the largest improvement … over **previously-known methods**".
   - ARC is less interested in fine-tuned constants, careful performance optimisation and opaque LLM code. LLM use must be disclosed and hedged.

   [COVERED: J2, including "a cost trim of 504aldo's public chain is not a contribution".]
8. **7 Jul, 18053#2:** to count for the **community** prizes, a write-up also needs a **PDF plus the associated submission ID**. [PARTIAL: J3 lists channels but not this condition.]
9. **9–10 Aug, 18129#4:** C_m > B_m zeroes the predictions and forces the multiplier to 1.0. The evaluator was patched and older submissions re-scored. [COVERED]
10. **11 Sep** (reported by 504aldo in 18218#3 on 14 Sep; second-hand): the sponsor confirmed Strassen–Winograd written as flopscope ops is permitted, with no limit on recursion depth. [COVERED]
11. **27 Aug, 18202#2:** the evaluator runs whestbench 0.16.0 with flopscope client and server 0.12.0. [COVERED]
12. **19 Aug, 18183#4:** mohanty (not a reviewer) says write-ups need "genuine human touch". Phase 2 write-up guidelines were promised; **none were posted as of 25 Sep** in this dump. [COVERED]

---

## (c) Count summary

I read all 93 topics (275 non-empty posts; 26 posts carry PDF write-ups). I extracted **69 distinct technical or rules items**: 22 rules and evaluation, 17 metering and infrastructure, 30 method and accuracy. Against the KB:
- **57 COVERED**;
- **5 PARTIAL**: dead/on/kink gating, the #257 symmetry cliff, compiler rewrites, "ARC reviews ~top 10", and the community-prize PDF condition;
- **7 NOT COVERED**: the pinned final flopscope version, cost-model patches during final evaluation, the rank-40 fast-multiplication identity, the angular attractor, the parallel-evaluator tools, Borel resummation, and `flops.current_budget()` (no action needed, since our code never calls the budget summaries).

The KB's Phase 2 digest (21 Sep) plus K3y (22–23 Sep) already absorbed every Phase 2 post. The only posts newer than the digest are three from 22 Sep: 18218#4 and #5 are covered by F105/K3y, and 18145#2 (mcdvoice007) has no technical content. The remaining value is in Phase 1 write-ups whose ideas transfer to the analytic chain, and item 1 is the only one with a plausible, measurable score effect (inference: about −10 % utilisation at unchanged raw MSE).

---

## Appendix: item ledger (where each COVERED item lives)

**Rules (22):**
- COVERED:
  - Phase 2 spec and caps — 18197 → B, G1
  - key dates including the 2 Oct team freeze — 18197; 18125 edit of 7 Sep → A1
  - private rerun: fresh seeds and suite size — 18118, 18125 → A4
  - selection mechanics — 18143, 18130 → G1
  - designation lock — 18078 → A4, J1
  - OSI licence — 18078 → A4
  - algorithmic-prize criteria and deliverable — 18041 → J2
  - C > B zero fallback — 18129 → G1, B5
  - Strassen ruling — 18218#3 → G1
  - evaluator versions — 18202 → G1
  - write-up quality guidance — 18183#4 → G1
  - error-text leak — 18217 → G1
  - residual scrutiny and post-grading review — 18197 → B4
  - Phase 1 unmetered-compute policy — 18132, 18135, 18108 → B2 (superseded)
  - bit-packing: allowed in Phase 1, banned in Phase 2 — 18125#9 → B2
  - 4 GiB per-array cap — 18039 → B5
  - separate setup budget — 18102 → B5
- PARTIAL: "ARC reviews ~top 10" (18078); community-prize PDF + ID condition (18053#2).
- NOT COVERED: pinned flopscope for the final (18078); cost-model patches during final evaluation (18125).

**Metering and infrastructure (17):**
- COVERED:
  - float64 billed 2× — 18125 → D3
  - `stats.norm` promotes to float64 — 18127 → D3, D4
  - f16 billed as f32 — 18125#10, 18151 → F3
  - batched-matmul price and Strassen-depth trade-off — 18184 → G3.6
  - `matmul(out=)` — 18101 → 504aldo buffer pools
  - residual time = sync points — 18202#3-4 → B5 warn, G4
  - big batched einsum failures — 18202#5 → G4
  - smoke-test shapes — 18214, 18218 → G4
  - Gram alias at 0.5× — 18154 → findings F-gram (unshippable)
  - tensordot bug — 18170 → B4
  - no billing arbitrage — 18179 → 504aldo row 14
  - Strassen levels — 18145, 18151, 18174 → F79–F81, K1
- PARTIAL: #257 symmetry cliff (18154, 18216); exact compiler rewrites (18149, 18173, 18171, 18183).
- NOT COVERED: rank-40 identity (18183 via 18157#9); violeta's tools (18180); `current_budget()` (18092, no action).

**Method and accuracy (30):**
- COVERED:
  - Puffi SSC κ4 response modes — 18175 → findings F68 trigger
  - Puffi A716 — 504aldo row 3
  - compensation dynamics and trajectory fitting — 18097 → Q19, K3x
  - state-keyed stabiliser — 18097#3 → V25 adaptive λ
  - global scale 0.992 / 0.99278 / 0.998 — 18063, 18183, 18216 → learned-output-correction dead (`pred` is among the features)
  - closure floor and Edgeworth sign — 18151, 18157 1f → K3ab (closure exact)
  - (c_i, c_j, ρ) pair table — 18157 4d → Q5 (the error comes from the inputs, not the closure)
  - kwon: unit-gain loop, K3 depth collapse, K4 memory — F2, K3i
  - trim_qewas ceilings and 6-dim Lyapunov subspace — K3m, Q10, D23, Q19
  - keenanpepper oracle ceiling and the inert module — G3.6
  - width/depth crossover — G3.6
  - Price + Plackett — F2, G3.3
  - radial conditioning worth ~1 % — V31 shipped
  - exact bivariate covariance only 7.4 % — G3.3
  - yujigon: constants absorb drift — G3.2
  - complementary-branch blend — K3k, D2
  - the 0.1 floor — G3.5
  - no tail in Phase 2 — G5
  - 504aldo post — G3.1
  - F105 — K3y
  - ARC low-rank direction — shared subspace, K3m
  - mid-network control variates — D2
  - thylinao oracle ladder — G3.6
  - Phase 1 sampling families — F3
  - violeta screening rule — tolerance law
  - keenanpepper datasets — G6
  - label-noise floor — A3
- PARTIAL: dead/on/kink gating (18106, 18149, 18157 4c, 18176).
- NOT COVERED: angular attractor (18188); Borel resummation (18078, 18159).

**Linked external artefacts** (none needed fetching: the only Phase 2 method repos, 504aldo and agentaiventure-dot, are already in G6):
- Code: SkyeNygaard/arc-whitebox; jamesrahenry/arc-whitebox-replication; Xieyangxinyu/whest-kprop-experiments; Oishi1029/arc-whestbench-2026; 01-1/arc-wbe and 01-1/flopscope; kaileh57/whitebox-phase1-327786; AndreiBulzan/arc-whitebox-submission-and-documentation.
- Hugging Face datasets: keenanpepper/{arc-whestbench-higher-moments-2026, whest-k3-tensors-2026, whestbench-relu-mlp-means-500k, whestbench-relu-mlp-moments-10k}.
- Other pages: thylinao1.github.io; the Kaggle notebook baltsat/how-203-experiments-reached-public-16; the Townhall video youtu.be/pvCgbUJ-nTo.
