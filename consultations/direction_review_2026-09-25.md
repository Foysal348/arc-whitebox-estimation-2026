# Direction review, 25 Sep 2026, 12:15 BST (independent; builds on the morning audit)

**Scope.** The main transcript `f76e7d52-…jsonl` (20 Sep 23:53 → 25 Sep 11:57 BST) was streamed to `scratchpad/convo_extract2.txt`: 7,369 lines of user messages, assistant text and 4 compaction summaries. I read all of 24 Sep 23:10 → 25 Sep 11:57 and the earlier summaries, and searched the rest. I also used the raw leaderboard tool output of 24 Sep 23:26 BST, the forum dump (topic 18218), `results/summaries/teacher_eval_q20dead{,2,3,4}.json` (the last one written at 12:03), `queue.json`, the Blueprint, the write-up draft and `correspondence/`. There are two new zero-cost computations, both on `data/dumps/errbudget_v29plain*.npz`: the dead and always-on fractions per layer, and the effect of the packing granularity on the Q20 saving. This review does not repeat the morning audit (`conversation_audit_2026-09-25.md`, items D-1…D-14 and C-1…C-10). All times are BST (UTC+6).

---

## Executive verdict

1. **The current direction (A: cost trims) is broadly right, but the assistant values it for the wrong reasons.** Cost trims are the only lever with positive measured evidence. The leaderboard shows the lever is bigger than the assistant says: **7 teams sit at the 504aldo chain's own accuracy (raw 2.12–2.18e-8) at util 0.19–0.22**, which is 10–23% cheaper than us. The assistant's 09:30 framing ("A = small improvement, ~#30") and the notes' "cost side is closed within this formulation" are contradicted by that data.
2. **The last 36 hours were steered by wrong rank arithmetic.** The claims "MSE ≤1.8e-8 → rank ~#14" and "20% accuracy → ~#12–14" are wrong: at our util they give **#24 and #20**. That arithmetic drove the overnight chain of accuracy oracles (5 lines, all dead).
3. **A false claim is now with the organisers.** "The 7–22% trim 504aldo saw was us" (23 Sep) is impossible: his post is dated 14 Sep and our first submission was on 20 Sep. The claim closed the most direct cost lead, and it is in the K3 email the user confirmed today was sent. Correct it today.
4. **Q20: go, but fix the plan before building.** Packing live columns into 1024-wide chunks (11:49 plan) erases most of the saving: dead-only compaction then saves **0%** of the young tier. Pack to multiples of 32 instead. Accuracy has a cliff (Φ-threshold 3e-2 costs +8–13%, and 1e-2 already costs +0.77% on MLP #0), so use a conservative fixed per-layer table. Run a metered emulation first. A realistic gain is **5–8% util**, not ~10%.
5. **Write-up (C) should get ≤30 min a day until 18 Oct.** Its deadline is 25 Oct 05:59 BST, a week after the close. ARC judges the algorithmic prize on roughly the top 10 submissions, by mechanisms that improved the score. At rank ~35, with a dead-end map, our odds for the $20k are low. The same text still fits the community prize.
6. **Stop new external-AI rounds and open-ended accuracy oracles. Keep Kaggle busy at all times. Submit each validated gain:** the grader is the only true measurement of residual and wall time.

---

## A. Direction

### A1. What the leaderboard says about where rank can come from

The leaderboard of 24 Sep 23:26 BST, from the transcript's raw fetch, gives this score-to-rank map: 3.1e-9 → #8; 3.6e-9 → #14; 4.0e-9 → #19; 4.1e-9 → #20; 4.2e-9 → #22; 4.4e-9 → #24; 4.6e-9 → #26; 4.7e-9 → #28; 4.9e-9 → #29; 5.0e-9 → #32; 5.1e-9 → #35 (us).

**The chain-accuracy, lower-cost cluster.** These teams have raw MSE within ~3% of the chain's 2.13e-8 but lower util:

| Team | Raw | Util |
|---|---|---|
| LuDoe | 2.12e-8 | 0.199 |
| tristan_miano | 2.13e-8 | 0.205 |
| whest_winner | 2.18e-8 | 0.206 |
| pricop_tudor | 2.18e-8 | 0.188 |
| Hack2Publish | 2.12e-8 | 0.219 |
| huang_chung_yi | 2.15e-8 | 0.214 |
| bgrubbs1984 | 2.18e-8 | 0.217 |
| oqaris (outlier) | 2.15e-8 | 0.146 |

Our util is 0.244 after the Strassen trims.

504aldo saw this already on **14 Sep 03:58 UTC** (topic 18218 #3): "someone trimmed 7-22% of the chain's cost (4.2e-9 to 5.2e-9) without losing accuracy".

| Our candidate (raw 2.105e-8 unless stated) | Score | Rank on 24 Sep |
|---|---|---|
| today | 5.13e-9 | #35 |
| + aug64 (−0.5%) | 5.10e-9 | #35 (no change) |
| + Q20, realistic (−5 to −8%) | 4.72–4.87e-9 | #28–30 |
| util 0.20 (what the cluster reached) | 4.21e-9 | **#22** |
| raw 1.8e-8 at util 0.244 (the "accuracy" target) | 4.39e-9 | #24 |
| raw 1.8e-8 at util 0.20 | 3.6e-9 | #14 |

Standing still also costs rank. The same 5.13e-9 was #28 on 22 Sep, #33 at 24 Sep 12:17 and #35 at 23:26.

**Reading.** An ~18% cost cut at unchanged accuracy is worth about as much as the ~15% accuracy trick the 24 Sep plan chased. Unlike the accuracy trick, we already hold one concrete candidate for it (Q20), and the forum audit names more (exact-algebra rewrites: pranay212 −6.7%, baltsat −3.1%). On 24 Sep at 12:17 the assistant had it right: "rank-এর সবচেয়ে বড় বাধা খরচ" [the biggest barrier to rank is cost]. At 23:29 it reversed this on the strength of the wrong rank mapping (§C, C-2).

### A2. Is the current plan right?

**Track A, Q16 (aug64 / transport joins, running on Kaggle): right. Keep it.**
- aug64 alone (−0.5% on the teacher networks) will not change the rank. The #332110 precedent also warns that a small local gain shrinks on the grader: −0.54% locally became −0.17% graded (queue Q2).
- The "transport" join (util −6.6%, MSE +6.3% / +1.0%, so score −0.8% / −5.7%) is the one Q16 variant that could matter. The queue still lists it as dead (D11); re-open it.

**Track A, Q20 (dead-neuron compaction): right lead, wrong execution plan as of 11:49.** The measured facts:

| Mode (bits) at Φ-threshold | MLP #0 | Synthetic | File |
|---|---|---|---|
| 1 (w1 := 0, next-transport input) @1e-2 | −0.25% | −0.66% | q20dead |
| 9 (+ dead birth columns) @1e-2 | −0.37% | −0.59% | q20dead3 |
| 3 (+ D21 rows/cols of dead neurons) @1e-2 | +1.65% | +2.59% | q20dead3 |
| 4 (d3 of dead neurons) @1e-2 | **+79%** | **+62%** | q20dead3 |
| 25 (+ always-on birth columns) @1e-2 | **+0.77%** | −0.79% | q20dead4 (12:03) |
| 25 @3e-2 | **+11.4%** | **+13.0%** | q20dead4 |
| 9 @3e-2 | +8.5% | +7.9% | q20dead4 |

So the output side (rows, d3) cannot be dropped. The input side is neutral at 1e-2 but sits next to a cliff at 3e-2. This matches the old K3h result (1% per-source screening error became 32% end-to-end) and the forum's "support-removal cliff" (ely2sh via 18157#9).

**Savings.** My estimate uses the chain's own α per layer (MLP #0 dump) and assumes the young-tier bill (54.5 u transport + 50.5 u D21) is spread evenly over source-layers.

| Rule | Young-tier saving, ideal shapes | 1024-chunk packing (the 11:49 plan) | 32-multiple packing |
|---|---|---|---|
| dead-only, Φ < 1e-2 (validated neutral) | ~16 u (6.4% of 249 u, incl. dead inner rows) | columns: **0%** of young | columns: 9.5% of young |
| dead + on, 1e-2 (borderline +0.77%) | ~25 u (10.2%) | 9% of young (~4% total) | 19% of young (~8% total) |
| dead + on, 5e-3 | ~21 u (8.4%) | 7% of young | 15.5% of young |

The dead + on fraction per layer at 1e-2 is 10% (L5), 23% (L8), 30% (L10), 37% (L12), 45% (L14) and 50% (L16). With 1024-wide chunks, four young sources save a whole chunk only when ≥25% of all their columns are dropped. That happens only at the deepest layers.

After a fixed-table safety margin (needed because fixed counts mean the effective threshold drifts per MLP, toward the cliff) and the joiner savings, the **realistic metered gain is 5–8% util**. The 10:56 message's "~8–10%, score ~4.6–4.7e-9" rested on dropping "input and output" rows, which the 11:31–11:47 runs disproved.

**Track C, write-up now: poor use of pre-close time.**
- The deadline is 25 Oct 05:59 BST, and the rules reserve "the week after the close … for writing only" (18041).
- The criterion (18041): "award the prize to the submission where novel 'mechanistic' ideas made the largest improvement to performance over previously-known methods". ARC "will review roughly the top 10 submissions (possibly more)" (Townhall, 18078).
- Our draft is honest about its content: "Most of this report is a measured map of where the remaining error comes from". Our own improvements over the public chain are Strassen constants (engineering), a spherical input (idea credited to barnobarno666) and fitted constants. The notes' own rule J2 says "a cost trim of 504aldo's public chain is not a contribution".
- The notes (`build_notes.py`) already hold every number. A thin cadence loses nothing. The draft is also a reasonable community-prize candidate (PDF + submission ID, 18053#2).

**Accuracy and moonshot: parked by the user (10:45: "na, apatoto off" [no, off for now]). Correct to keep parked.**
- Five accuracy lines died on 25 Sep. The 09:30 conclusion "বাকি accuracy K3 পদ্ধতির জন্মের গঠনে আটকে আছে … গবেষণা-স্তরের প্রশ্ন" [the remaining accuracy is locked in the structure of K3 births … a research-level question] over-reaches, though (§C, C-4).
- Do not restart this without a concrete hypothesis for how emanuel_ruzak and jlacombe moved 2.13 → 1.82e-8 in ≤4 submissions.

**Better use of the time?** Yes, a small but real re-framing: make the target **util ≈ 0.20 at unchanged raw**. Q20 is item 1 of a cost-trim programme, not the whole programme (see B-5). The 72-hour plan at the end is built on that.

### A3. Where time was lost, and whether it is happening now

| Pattern | Evidence | Cost | Now? |
|---|---|---|---|
| External-AI rounds | Rounds 3–12 (21–25 Sep). The queue holds 37 dead entries (D1–D25 plus 12 Q-items), most of them from outside proposals. On 24 Sep, 71 of 101 assistant messages were about AI rounds or prompts. At 24 Sep 20:23 the assistant said "বাইরের উত্তর যাচাইয়ে আমি বেশি সময় দিয়ে ফেলেছি … দিনে সর্বোচ্চ ১ ঘণ্টা" [I spent too much time verifying outside answers … at most 1 h a day]. The next ~6 h (20:44–02:50) went to Round 10/11 triage, a Sonnet contraction search, the Round 12 prompt, Opus R12 and its review. Partly user-driven: the user pasted PDFs and asked for deep analysis | Most of 23–24 Sep | No (none today after 02:50) |
| Oracle chains, where each negative spawns the next | Channel, horizon, Age-Gram, shift, D21, late-layer, κ4 horizon, Q19, V31c: about 12 oracles on 23–25 Sep, all sound, all negative. Each was pitched as a lead: 02:23 "বাস্তব আর সস্তা সূত্র" [a real and cheap lead], dead at 03:12. Several were predictable: the κ4 horizon (Q18) mirrors the κ3 horizon; D23 follows from K3ad (exact old sources only −3.9%); V31c's term (~1e-6 per entry) could have been sized in 2 minutes before a 20-minute, 10-run sweep | ~8 h on 25 Sep, 00:23–04:08 and 09:07–10:56 | Q20's four mask runs (~70 min) are quick and decision-relevant, which is fine. The risk now is a multi-day refactor on an unmeasured estimate |
| A false conclusion closed a live lead | 23 Sep 12:04: "সেই 'কেউ'-টা আমরা (MIN=8 + FIRST=5)" [that "someone" is us]. The cost cluster was never examined again; notes line 1329 says "Cost side is closed within this formulation" | Strategic | Still in the notes, the correspondence and the sent email |
| Idle capacity | Kaggle runners idle from 24 Sep ~20:45 (the 20:49 message said all three were free; Q16 was promised "next") until 25 Sep 10:07, about 13 h with no push between the 24 Sep 12:45 grid and 10:07. The assistant also sat idle from 04:31 to 09:07 waiting for "শুরু করো" [start] on a 0 u, 15-minute local run (rule 6: "অপেক্ষা কেন করছো?" [why are you waiting?]) | ~13 h of Kaggle, 4.5 h of local time | Watch it: one runner should always hold a queued job |
| Payoff arithmetic not checked against the leaderboard | §C, C-2 | Drove the 24→25 Sep pivot | Fix with the table in A1 |

### A4. Rules and priorities today

- Fable was not called. Replies are in Bengali and times in BST. The user's choice of A,C (09:49) was followed. The 80% moonshot rule was suspended by the user (10:45).
- Minor: Q16 was pushed at 10:07 before the local 2-MLP `whest run` finished. That run then failed at 10:19 because the mini-split download was incomplete. The variant had passed teacher-network runs on 24 Sep (`teacher_eval_v41join*.json`), so the risk is low, but this breaks the letter of rule 5 ("নিজে রান না দিয়ে রান দিও না" [don't launch without running it yourself]).
- Minor: the user's answer to the K-1/K-4/K-5 question was "আমাকে এখনো মেসেজ দেওয়ার অনুমতি মিলে নি" [I still don't have permission to send messages]. That conflates forum posting with GitHub issues, which need no forum permission. The assistant did not clarify.

### A5. Status of the morning audit's items

| Item | Status since the morning audit |
|---|---|
| D-1 public-code re-sweep | **Done** (10:38): no forks, no code after 22 Sep, barnobarno666 last changed 13 Sep. The 7-team cost cluster was not examined (B-5) |
| D-2 V31c | **Done, dead** (D24; +0.02…+0.19%, noise) |
| D-3 one-network oracles | Wording fixed in the draft ("MLP #0 only so far"). Not replicated |
| D-4 K-1/K-4/K-5 | Asked; the answer was a misunderstanding (A4). Open |
| D-5 Strassen L6 | **Decided: dropped** (D25). The "max −0.7%" understates the measured young-only −1.0%; the decision is still right |
| D-6 zero-constant chain, D-7 R4 rider | Not done (low; D-6 is write-up only) |
| D-8 B-prompt, 80% rule | Resolved by the user: moonshot off |
| Queue clean-up (Q4, Q6, Q17 dead; Q18a) | **Done** |
| C-1 "orthogonal ages" | Notes fixed. **`queue.json` "headroom" still says "roughly equal and orthogonal"** |
| C-5 Blueprint premise | Revised, but it still lists "Priorities since 24 Sep: (1) accuracy, (2) cost trims", which the user's A,C choice superseded |
| C-9 two drafts | Done (master v3, condensed HTML) |
| §5 external: K3 email | The user confirmed it was sent to two addresses, with no reply. It carries the false claim (C-1 below) |

---

## B. Skipped experiments and checks, ranked by (EV × P) / cost

| # | Item | Why it matters | EV × P | Cost |
|---|---|---|---|---|
| B-1 | **Metered FLOP emulation of Q20 before the refactor.** Build the per-layer live-column table (α from ≥10 MLPs), pad to multiples of 32, and meter the young transport + D21 hub on those shapes (flopscope on a stub, or the ledger) | Decides whether a 1–3 day build is worth it. The current ~10% is a guess, and 1024-chunk packing gives ~0–4% | High × 0.9 | 1–2 h |
| B-2 | **Q20 robustness table.** Live-column counts per layer across the 100 mini MLPs (one Kaggle dump run), set to the worst case plus a margin; confirm accuracy on held-out networks | Fixed counts move the effective Φ-threshold per MLP; the cliff is at 3e-2 | Protects the gain | 1 Kaggle run + 1 h |
| B-3 | **Stack Q20 on the Q16 winner** (aug64 or transport). Run a 100-MLP paired test against both the winner and v40fb | Both touch the join (41–46 u); the effects are not independent | Medium × 0.8 | Kaggle 6–8 h (0 of our time) |
| B-4 | **Submit every validated gain the same day** (10 a day; 7 used in 5 days) | The grader is the only true residual/wall clock: the local clock is ~5× the grader's (K-5), and Kaggle's maximum was 134 s against a 120 s cap. Submitting also gives an independent 50-MLP check | Safety + information | 15 min each |
| B-5 | **"util 0.20" trim hunt (desk audit, time-boxed).** Rank the remaining exact trims by ledger share: join 41–46 u (range-finder products, 7 × 3.75 u); thin legs 15 u; `C_pre` sandwich 5.3 u (drop dead rows and columns of C_post); `V26_STRASSEN_NEW=0` (0.5% for ~30 ms residual, if the grader residual margin allows); Strassen additions (~7% of each product; an alternative-basis Strassen ≤1.5%); pranay212/baltsat-style exact rewrites. Seven teams found ~20% at chain accuracy, so it exists | Target #22 instead of #28 | High × 0.3 | 2 h desk + prototypes |
| B-6 | **Re-tune AGE_OLD / R_OLD / R_OLD2 after Q20.** Compaction changes the cost of each knob. Cheaper young sources make a longer young tier more attractive (age2_8 gave raw −0.9% at +1% util before), and fewer live columns may lower the rank the old tier needs | The "one-for-one" optimum was measured on the uncompacted cost structure | Medium × 0.4 | One Kaggle grid (0 of our time) |
| B-7 | **Held-out confirmation** of the nomination candidate on public full-split MLPs outside the 100 used for tuning, plus a local `SymmetryError` counter (forum audit item 2) | FB/β were tuned on the same 100 MLPs; −0.54% local became −0.17% graded. One failed MLP in the private rerun costs ~0.9/N | Protects the final result | Kaggle 2 × 6–10 h; do by ~8 Oct |
| B-8 | Email the organisers: which flopscope/whestbench version the private rerun pins (forum audit item 3). **Bundle with the correction (C-1)** | The price of `take`/`argsort`/Strassen additions affects Q20 | Small × certain | 10 min (user sends) |
| B-9 | Residual and wall check for Q20's new ops: per-MLP residual from the Kaggle log against the base; gate ≤ base + 10%. Fixed shapes only, no host reads | A residual over 0.4 s zeroes that MLP | Protects the gain | With B-3 |
| B-10 | Write-up only, on idle Kaggle time: D-6 zero-constant chain; replicate the κ4-diagonal and variance channel oracles on the synthetic network (D-3) | Evidence quality for the write-up | Low | Kaggle only |
| B-11 | **Decision: team merge, deadline 3 Oct 05:59 BST** | The only way to acquire an accuracy trick we have failed to find in 5 attempts. The user's call; decide consciously this week rather than by default | — | — |

**Explicitly not recommended now:** new external-AI rounds; Fable; Q14 (break-even at best, per its own queue entry); Q9/Q13; more κ3/κ4 oracles.

---

## C. Consistency: statements contradicted by the data

| # | Statement (when) | Contradicting data | Fix |
|---|---|---|---|
| **C-1** | "সেই 'কেউ'-টা আমরা (MIN=8 + FIRST=5)" [that "someone" is us] (23 Sep 12:04). Also in `build_notes.py` line 1124 ("That someone is us"), `correspondence/K3_email_to_organisers.md` lines 31–32 ("That was my submissions"), `K3_forum_new_topic.md` line 24 and `K3_forum_reply_final.md` line 17 | 504aldo's post #3 is dated **2026-09-14 03:58 UTC** (forum dump 18218). Our first submission, #331681, was on **20 Sep**. Our trims took util 0.254 → 0.244 (−4%, best 5.195e-9), never 4.2e-9 | **Today:** a 3-line correction email to the same two addresses; fix the notes and both embargoed drafts. The algorithmic-prize guidelines warn that unhedged dubious claims make ARC "more skeptical" |
| **C-2** | "শুধু accuracy ২০% বাড়ালেই … ~#১২" [just 20% better accuracy … ~#12] (24 Sep 23:29); "MSE … ≤১.৮e-8, যাতে rank ~#১৪" [MSE ≤1.8e-8, so rank ~#14] (23:38); "২০% কমলে ~#১৪" [20% lower → ~#14] (25 Sep 03:16) | At util 0.2437: 1.684e-8 → 4.10e-9 → **#20**; 1.8e-8 → 4.39e-9 → **#24** (24 Sep leaderboard). The 00:48 table was correct (top 15 needs 1.8e-8 **at 0.19**) | Keep a score-to-rank table (A1) in the queue; recheck before every "rank ~#N" claim |
| C-3 | "A. খরচ ছাঁটাই … ছোট উন্নতি, ~#৩০" [A. cost trims … small improvement, ~#30] (09:30). Notes line 1329: "Cost side is closed within this formulation". Draft §3: "the chain sits at its own score optimum" | 7 teams at chain accuracy with util 0.19–0.22 (A1). Q20 itself breaks the "floor" | Reword the draft ("within the knobs we tested"); update the notes |
| C-4 | "বাকি accuracy K3 পদ্ধতির জন্মের গঠনে আটকে আছে … গবেষণা-স্তরের প্রশ্ন" [the remaining accuracy is locked in the structure of K3 births … a research-level question] (09:30). Draft §0: "resists every cheap fix we and others tried" | The assistant's own 24 Sep 23:38 table: emanuel_ruzak 2.13@0.249 → 1.82@0.198 in 4 submissions; jlacombe 2.10 → 1.82 in 4. The oracles show *where* the error is, not that a fix must be expensive | Say "unknown to us; others found it quickly"; drop "and others" from the draft |
| C-5 | "মৃত neuron বাদ দিলে transport-এ ইনপুট আর আউটপুট দুদিকেই ছোট হয় … ~৮–১০%, score ~৪.৬–৪.৭e-9" [dropping dead neurons shrinks both the input and the output of the transport … ~8–10%, score ~4.6–4.7e-9] (10:56); queue Q20 "~2 x dead fraction" | q20dead3: rows +1.65% / +2.59%, d3 +79% / +62%. Only the input side is free | Update Q20's evidence: realistic 5–8% (B-1) |
| C-6 | "জীবিত column যত কম, খরচ ঠিক তত অনুপাতে কম" [cost falls exactly in proportion to the live columns] together with "১০২৪-চওড়া টুকরোয় ভাগ করে" [split into 1024-wide pieces] (11:49) | The two are inconsistent: with 1024 chunks the cost is quantised. Dead-only compaction saves 0% of the young tier; dead + on at 1e-2 saves ~9% of it (my computation, A2) | Pack to multiples of 32. Rectangular Strassen leaves already exist in the ledger, e.g. `(6, 2401, 32, 12)` |
| C-7 | ARC "এই পুরস্কার দেয় score দেখে না, mechanistic বোঝাপড়া দেখে" [gives this prize not by score but by mechanistic understanding] (00:48); "C … $২০k-এর প্রতিযোগী … মাঝারি" [C … a $20k contender … medium certainty] (09:30) | 18041: "largest improvement to performance over previously-known methods"; 18078: "review roughly the top 10 … mechanistic analysis that measurably improved the score" | Tell the user: the $20k odds are low unless we climb and our own mechanism moves the score; the community prize is realistic |
| C-8 | Queue D11 "transport … dead"; queue "headroom": "orthogonal"; Blueprint "(1) accuracy, (2) cost trims" | D11 is being re-tested (Q16); Age-Gram showed coherent ages; the user chose A,C | Housekeeping edits |

No contradiction was found in the V31c, Q19, D23, Q18 or channel-oracle numbers reported today. They match their JSON files.

---

## Next 72 hours (25 Sep 12:15 → 28 Sep 12:15 BST), in order

**Thu 25 Sep**
1. **12:30–12:45: correction email.** The assistant drafts 3 lines ("the 7–22% trim 504aldo observed on 14 Sep predates our participation; our Strassen trims were ~4%"). The user sends it to the same two addresses and adds the pinned-flopscope question (B-8). Fix the notes line 1124, the K3 drafts, and queue items D11, Q20 and "headroom" (30 min).
2. **12:45–13:15:** add the score-to-rank table (A1) to the queue and the Blueprint. Make the target explicit: util ≈ 0.20 at raw 2.1e-8.
3. **~13:30: read the Q16 results.** If aug64 or transport passes (< 1 − 2 SE): package it, run the local `whest` validation (the mini split must be fully cached first), and **submit today**. Record the grader residual and wall time.
4. **13:30–16:00: B-1 and B-2 (metered emulation).** Queue one Kaggle job that dumps α per layer for the 100 mini MLPs (this also keeps a runner busy). Set the gate: **≥5% metered util at ≤ +0.3% raw on both teacher networks**; otherwise stop Q20. Start with dead-only at 1e-2, since it is validated neutral.
5. **16:00–24:00: Q20 stage 1** (dead-only birth columns plus dead inner rows; fixed table; 32-multiple packing; server-side `argsort`/`take`; no host reads). Validate against the mask oracle and check util, local residual and peak RSS.

**Fri 26 Sep**
6. Morning: stage 1 on both teacher networks, then a Kaggle 100-MLP paired run stacked on the Q16 winner (runner A) and alone (runner B) (B-3, B-9). About 6–8 h of wall time.
7. Midday: **B-5 desk audit** (2 h, time-boxed). Pick at most 2 exact trims to prototype.
8. Evening: if the 100-MLP result passes, **submit**; read the grader residual.

**Sat 27 Sep**
9. Q20 stage 2: always-on columns at a conservative threshold (Φ > 0.995, since the 0.99 point was borderline) and joiner compaction. Teacher networks first, then Kaggle.
10. Prototype the best B-5 trim. Queue the B-6 knob re-tune on post-Q20 code on an idle runner.

**Sun 28 Sep (to 12:15)**
11. Read stage 2 and B-5 results; submit if they pass. Start the B-7 held-out confirmation for the leading candidate (runs ~10 h in the background).
12. **User decision before 3 Oct 05:59 BST:** team merge, yes or no (B-11).

**Standing rules for the period**
- Write-up: ≤30 min a day (append numbers and figure data only). Full writing happens 18–24 Oct.
- No new external-AI rounds or oracles unless one names a concrete mechanism for the 1.82e-8 jump.
- Never let all Kaggle runners sit idle.
- Before stating any "rank ~#N", check it against the A1 table.
