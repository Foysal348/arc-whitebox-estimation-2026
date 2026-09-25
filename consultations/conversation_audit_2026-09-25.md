# Conversation audit: dropped, half-done and contradicted items (20–25 Sep 2026)

**Scope.** The main session transcript `f76e7d52-…jsonl` (40 MB, 20 Sep 17:53 UTC → 25 Sep 04:16 UTC), streamed and reduced to 691 records (160 user messages, 25 mid-turn user messages, 493 assistant text blocks, 6 subagent hand-backs, 4 compaction summaries); plus every subagent launch and every Kaggle push / AIcrowd submit found in the tool calls. Each item was cross-checked against `docs/experiment_queue/queue.json`, `writeup/UNTESTED_IDEAS.md`, `docs/build_notes.py`, `docs/moonshot/Moonshot_Blueprint.html`, `writeup/WRITEUP_DRAFT_v3.md`, `docs/writeup/Writeup_Draft.html`, `correspondence/README.md`, `results/summaries/`, `experiments/` and the git log (HEAD `e9ca2cf`).

**Conventions.** All clock times are **Bangladesh time (BST = UTC+6)**, converted from the transcript's UTC stamps. Bengali quotes are followed by an English translation in [brackets]. Status labels: **DONE** (cite where), **OPEN-TRACKED** (open in the queue or Blueprint), **DROPPED** (not done and not tracked as an open item).

---

## 1. DROPPED items (most valuable first)

### D-1. Re-sweep public code after the 21→24 Sep leaderboard jumps (not done)
- **When / quote.** 24 Sep 23:38: the leaderboard comparison showed teams starting from the public chain and improving within days: emanuel_ruzak "২.১৩e-8 @ ০.২৪৯ (হুবহু 504aldo chain, আমাদের মতো)" [exactly the 504aldo chain, like ours] → "১.৮২e-8 @ ০.১৯৮" in 4 submissions; jlacombe 2.10 → 1.82e-8 in 4; MeatProxy 2.99 → 1.79e-8 in 1. Conclusion then: "এটা সম্ভবত ছোট, স্থানীয় কোনো পরিবর্তন" [probably a small, local change]. 25 Sep 02:19: "they probably read 504aldo's public findings log" → only F47 (the disabled online correction, later Q19, dead) was checked.
- **What was never checked.** Whether the change is itself public: newer commits or estimator files in `504aldo/whest-p2-cumulant-k3` after 22 Sep (our local clone `whest-p2-cumulant-k3/` is still at 9 Sep, commit `18c17e2`; the 23 Sep sweep read F105 on the web only); forks of that repo; and the rest of the barnobarno666 "Phase-8" repo, which already gave us the angular trick and reportedly "−7.9% raw on an independent K3 chain" (22 Sep 01:23). The 25 Sep forum-audit agent covers the forum only.
- **Why it matters.** The 24 Sep plan made accuracy the first priority ("MSE ২.১১ থেকে ≤১.৮e-8" [MSE 2.11 → ≤1.8e-8], rank ~#14). Five own accuracy lines have died since: κ4 cheap rules and the missing half-families (Opus R12), the young κ4 carrier (Q18), the online correction (Q19), and rank reallocation (D23). A cheap public diff is the most direct test of the "small local change" hypothesis.
- **Next step.** `git -C whest-p2-cumulant-k3 fetch && git log origin/HEAD --since=2026-09-09 --stat`, then diff any new estimator against `submissions/v40fb/estimator.py`; a GitHub search for whestbench / whest-p2 repos and forks pushed since 20 Sep; re-read the barnobarno666 repo for anything besides the spherical input. All read-only, well under an hour.

### D-2. V31c: the remaining sphere κ4 term (2c·M_ij²) at layer 0 (never run)
- **When / quote.** 22 Sep 07:06, the assistant's own experiment list: "**V31c**: sphere-এর (2,2) κ4 slice-এর বাকি অংশ (2c·M_ij²) layer 0-তে যোগ … হয়তো আরও −0.3% … লোকাল 5 মিনিট" [add the rest of the sphere's (2,2) κ4 slice at layer 0 … maybe another −0.3% … 5 min locally], followed by "A দিয়ে শুরু" [starting with A]. The next tool call located the injection block; then the session moved to the AGE_OLD2 sweep and V31c never ran.
- **State.** Not in the queue. `docs/build_notes.py` K4 still lists "the sphere's 2 M_ij^2 term" under NEXT. No `estimator_v31c*` in `experiments/variants/`.
- **Why it matters.** It is zero-cost, completes the only accuracy idea that has shipped so far (#331787, −1.1% graded), and matches the user's 25 Sep 09:49 choice of track "A" (sure small gains).
- **Next step.** Implement it in `v40fb`, run it on the two teacher networks, and add it to the next 100-MLP batch next to Q16.

### D-3. The headline channel oracles rest on one network (not confirmed)
- **When.** 23 Sep 17:20–17:32: the corrected κ4 oracle ("diag 0.702"), and later the channel map (variance −47%, κ4 −30%, κ3 −14%, all three −94%), all on public MLP #0. On 25 Sep 02:12, the Opus review caveat 1 read "শুধু একটা network-এ মাপা" [measured on only one network]. Opus then replicated the κ3 slice (and slice+d3) oracle on the synthetic network, but only partially: the synthetic teacher has 8 of the 14 injectable layers. The κ4-diagonal and variance oracles were never replicated.
- **Conflict in the write-up.** The abstract of `WRITEUP_DRAFT_v3.md` says "measured with … teachers … on two 1024×16 networks", but §11.2 (channel oracles) is "final MSE of MLP #0" only.
- **Why it matters.** These numbers are the central claim of the algorithmic-prize write-up and the premise of Blueprint C2 and M4.
- **Next step.** Rerun the corrected channel oracle (`k4_oracle_run2` / `meanfix_oracle`) on the synthetic teacher for its available layers, or push one more public-MLP pre-activation teacher (a validated kernel exists; about 45 min on a T4). At the least, correct the abstract wording.

### D-4. Community-contribution filings K-1, K-4 and K-5 (status "to do" since 21 Sep)
- **When.** 21 Sep 03:08, PDF Part K3 "contribution candidates": K-1 "starter-kit doc is stale about the prepared-Arrow fast path", K-4 dataset sizes (fold into K-1), K-5 "local residual clock ~5× the grader's … false failures". The notes' K4 section still reads "Write … the K-1 docs issue". The table still says "to do".
- **Why it matters.** The user asked on 21 Sep 03:04 to compete for every prize, including community ("সবগুলোর জন্যই লড়বো" [we will fight for all of them]). These three items disclose nothing about our method. The embargo's stated reason (`correspondence/README.md`) covers only the Strassen trims and the angular trick. They also do not need forum posting rights: they are GitHub issues on `AIcrowd/whest-starterkit`. **Caveat:** on 23 Sep 12:46 the user chose "সব public পরে" [everything public later], so get the user's explicit OK first.
- **Next step.** Draft the three issues (English, with evidence), ask the user whether to file them from their GitHub account now or after 18 Oct, and log them in the PDF under Part J3.

### D-5. Memory-saving Strassen L6 (promised twice, never started)
- **When / quote.** 24 Sep 20:23: "ধারা B. নিশ্চিত লাভ … (২) memory বাঁচিয়ে Strassen L6 … ধাপে ধাপে হিসাব করলে memory কম লাগবে … কাল থেকে" [stream B, sure gains … (2) Strassen L6 with memory saving … computing step by step needs less memory … from tomorrow]. 24 Sep 20:37: "নিশ্চিত লাভের কাজগুলো (aug64, Strassen L6) চালাব" [I'll run the sure-gain tasks (aug64, Strassen L6)].
- **State.** aug64 became Q16 and is running on Kaggle. L6 is in neither the queue nor the Blueprint (M5 lists only aug64 and the grid).
- **Evidence it is small but real.** Young-only L6 gave util −1.0% at 7.89–8.10 GB with 2× residual (21 Sep). MIN=6 gave −0.74% at 7.84 GB and +31% residual (23 Sep). The residual cap (0.4 s) is the binding risk.
- **Next step.** Either add it to the queue with the residual/memory gate written out, or record an explicit decision to drop it.

### D-6. Zero-constant chain: online-λ-only, no fitted table (never run)
- **Sources.** Board A3 ("useful for the write-up") and notes K4 "Question D: online-lambda-only variant (drop the fitted table) on 10 mini MLPs". On 25 Sep 02:23 the assistant noted "ARC fitted constant পছন্দ করে না" [ARC does not like fitted constants].
- **Why it matters.** Its value is for the write-up, not the score: it shows how much of the 5.13e-9 depends on fitted constants. The FB 1.25 and β 1.2 changes added two more fitted constants.
- **Next step.** One 100-MLP paired Kaggle run. A 0 u cost for the result; report it in the write-up however it comes out.

### D-7. GPT R4 §7 "penultimate K4-residual rider" (~1%): the blocker was removed but the rider was never revisited
- **When / quote.** 22 Sep 02:14 table: "GPT R4 §7 penultimate K4-residual rider | ~1% accuracy rider; 1024-এ augmented teacher নেই | সম্ভব নয় এখন" [no augmented teacher at 1024 | not possible now].
- **State.** 1024-wide κ4 teachers (pre and post) have existed since 22–23 Sep. The item is not in the queue or on the board.
- **Next step.** Re-read GPT R4 §7 and check it against C15 (the κ4 memory is long). If the rider is a last-layer fitted correction, the Q19 result (no signal at depth) probably kills it too; if so, record it as dead.

### D-8. Option B of 25 Sep: a sharp research prompt on the birth of the all-distinct κ3 (recommended, then not written)
- **When / quote.** 25 Sep 09:30: "আমার পরামর্শ: A আর C একসাথে … সাথে B-র জন্য একটা ধারালো প্রম্পট" [my advice: A and C together … plus a sharp prompt for B]. The user answered "A,C" (09:49), and no B prompt was written.
- **Why it matters.** Every accuracy measurement now points to the same place: C3, D23, and Fable R8's "~55% unexplained through second order". The standing rule from 23 Sep 14:48 is "moonshot-এ ৮০% সময়" [80% of the time on the moonshot]. The 25 Sep choice suspends that rule without saying so.
- **Next step.** Ask the user whether the 80% moonshot rule still holds. If it does, write a one-question prompt for external models (no Fable without per-run permission).

### D-9 to D-14. Lower value (record the decision; no work needed unless the user asks)
| # | Item (when) | Status | Suggested action |
|---|---|---|---|
| D-9 | Sol's lazy-cubic-circuit exact-parity falsifier ("the concrete mechanism I would test once", 24 Sep 16:27). The assistant replied "আমি এটা implement করিনি" [I did not implement it] (16:30) | Recorded dead as D15 on a counting argument, never measured | Only adds a measured "dual contraction orders" statement to the write-up; skip unless there is time |
| D-10 | Astra R11 injection test (24 Sep 18:22: "চালাইনি" [not run]) | Skipped by argument | None |
| D-11 | "round-6 prompt focused only on … cheap κ4-with-memory representations" (22 Sep 11:43) | Never written. Round 6 went to cost instead; Q18 later showed that κ4 memory is as long as κ3's | Superseded; none |
| D-12 | Fable R1 H3 = board A2: per-network λ fit from a 2¹² MC probe (21 Sep 11:08) | Never run. Later evidence kills it: 10⁴ samples cost 156 u, and per-network offsets explain 0.44% | Mark dead on the board |
| D-13 | Board B2 (quenched chaos residual), B3 (Hankel observability rank), Terra's implicit mat-vec oracle, Fable R1's "[M] R² of the λ fit should rise as 1−O(1/√n) with width" | Never run; low prior. B3 is effectively answered by the horizon oracle and the Age-Gram test | Mark closed |
| D-14 | 23 Sep 22:14: the DeepSeek R7 PDF was a byte-identical copy of Terra 7.1: "DeepSeek-এর আলাদা উত্তর থাকলে আবার দিন" [if DeepSeek has a separate answer, give it again] | Never resent | Optional; ask the user once |

---

## 2. OPEN-TRACKED items (still open in the queue or the Blueprint)

| Item | Where tracked | Origin / quote | Note |
|---|---|---|---|
| **Q16** join aug64 (plus the "transport" join) 100-MLP check | queue Q16 (P2); commit `e9ca2cf` | 24 Sep 13:32: "aug64 … score ~০.৫% ভালো … 100-MLP-তে যাচাইয়ের তালিকায়" [aug64 … score ~0.5% better … on the 100-MLP validation list] | Running on Kaggle runners A and B since 25 Sep ~10:07. Promised earlier (24 Sep 20:49: "runner খালি হওয়ায় পরের কাজ: Q16" [runner free, next task: Q16]) but pushed only 13 h later. Check the wall-time margin before shipping: the Kaggle max was ~134 s against the 120 s cap (24 Sep 00:17 note) |
| **Q14** Astra R9 mixed-coordinate κ4 readout | queue Q14 (P2, open) | 24 Sep 12:23: "Q14 … এটাই একমাত্র প্রার্থী যা পরের ধাপে নিতে পারে … ১–২ দিন" [Q14 … the only candidate that can take us to the next rounding step … 1–2 days] | Never measured, not even the layer-14 identity check. Parked by the user's 12:41 decision "এলবি টিউন করবো না আপাতত" [no LB tuning for now]; the 25 Sep choice "A" reopens small gains |
| **Q15** S = 2 partner-channel replay | queue Q15 (P2, "সময় পেলে" [when time permits]) | 24 Sep 00:48: "আমি: একটা অমীমাংসিত রহস্যের পেছনে যাব … S = 2" [I will chase an unsolved mystery … S = 2] | Handed to Fable R9, which died in the restart and was not resumed. Astra R9 explained it as a quadratic cancellation. The write-up §11.4 still calls it "an open paradox" |
| **Q9** joint knob search | queue Q9 (P3) | 23 Sep 13:56, idea B: "যৌথ knob optimisation" [joint knob optimisation] | Deprioritised on 100-MLP evidence (knobs sit at the optimum) |
| **Q13** variance super-closure | queue Q13 (P3) | GPT Sol High R7 | Low ceiling (Q5) |
| **M3** nonlinear, trajectory-specific state | Blueprint §8 M3 "Open" | Sol, 24 Sep 16:27: leaders must "exploit a nonlinear, trajectory-specific sufficient statistic … or bypass the explicit κ3 state" | Opus R12: "B2 was not run … needs a new idea". Fable never ran (429). No action scheduled |
| **Write-up (track C)** | `docs/writeup/Writeup_Draft.*` v0 (25 Sep) and `writeup/WRITEUP_DRAFT_v3.md` | 23 Sep 14:48: "write-up-টা নিশ্চিত সম্পদ … আপডেট করতেই থাকবো" [the write-up is a sure asset … we will keep updating it] | Still to do: figures, hand-checks of the derivations, the parity proof, human edit, ≤10 pages. See §4 on the two parallel drafts |
| **E1** two nominations (one max-score, one conservative) | board E1 "do at the end" | 23 Sep 13:15 plan: "১৫–১৬ অক্টো: দুটো nomination বাছাই" [15–16 Oct: choose the two nominations] | Deadline item (§5) |
| **Embargoed forum post** | `correspondence/K3_forum_new_topic.md`, `K3_forum_reply_final.md` | 23 Sep 12:46, user: "শুধু email (আজ) + সব public পরে" [only email (today) + everything public later] | Needs posting rights first (§5) |

**Tracked as open but actually done or dead (update the queue):**
- **Q4 / board R31** ("TRUE D21 injected into the feedback legs only"). Queue: P3 open; board: "open, queued". **DONE** by Opus R12: "শুধু feedback leg-এ দিলে কিছুই হয় না" [injected only into the feedback legs, nothing happens] (25 Sep 02:00; queue D22 evidence "×0.9999"). Mark it dead.
- **Q6** (K4×C term in the Astra kernel). Queue: **P1 open**, next "Q1 পাস করলে" [if Q1 passes]. Q1 is dead, and D22 found the cheap kernel variants net ≤ 0. Mark it dead.
- **Q17** (young tier at half cost). Queue: **P1 open**, next "Round 10 AI উত্তরের অপেক্ষা" [waiting for the Round 10 AI answers]. Round 10 is closed (D21, the Sonnet search; Blueprint M2 "CLOSED", 193-product floor). Mark it dead.
- **Duplicate id `Q18`** in `queue.json`: one entry for FHQR-2 (dead) and one for the κ4 memory horizon (dead). Renumber one of them.
- **Board rows A4, B1, B4** carry no status, although all three were done by `probe_loopholes.py` (21 Sep). Board D4 (warm-start join) was done as V41 / Q16 / D11.

---

## 3. Rule violations found

Standing rules as the user gave them (chronological):
1. The API key is never pasted in chat; the user runs `whest login` (21 Sep 01:30).
2. "যখন … Fable 5, Astra 6 থেকে আইডিয়া নিয়ে চাও বলিও" [tell me when you want ideas from Fable 5 or Astra 6] (21 Sep 02:43).
3. Compete for all prizes; log community and algorithmic material in the PDF during experiments (21 Sep 03:04).
4. "আর ফেবল ৫ ইউজ করবে না,আমার অনুমতি ছাড়া" [don't use Fable 5 without my permission] (21 Sep 11:58).
5. "নিজে রান না দিয়ে রান দিও না,আর এরর বাগ আগে থেকেই দেখবে" [don't launch (on Kaggle) without running it yourself first; check for errors and bugs in advance] (21 Sep 18:15); stop and fix buggy notebooks (18:16).
6. Kaggle CPU can run many jobs in parallel; don't wait (22 Sep 02:12).
7. "সাবমিশন দিয়েই দিবে,হাতে রেখে তো লাভ নেই" [just submit; there's no gain in holding submissions back] (22 Sep 02:27).
8. Never wait for the leaders' write-ups; never give up (21 Sep 13:47).
9. 80% of the effort on the moonshot; "আমাকে ডিমোটিভেড কখনোই করবে না" [never demotivate me] (23 Sep 14:48).
10. Fable is permitted for the 23 Sep work: "ফেবল ৫ কে তুমি কল করে ওকে দিয়ে কাজ করিয়ে নাও" [call Fable 5 yourself and get the work done] (23 Sep 15:00).
11. Reply in Bengali with English only for technical terms: "বেশি ইংরেজি বুঝি না" [I don't understand much English] (23 Sep 21:10); "বাংলায় কথা বলো" [speak in Bengali] (21:27).
12. Every AI answer worth testing goes into a priority-ordered PDF (the queue) (23 Sep 21:27).
13. Clock times in Bangladesh time, "সর্বদা" [always] (24 Sep 00:46).
14. Fable is expensive in tokens, so ask before each launch (24 Sep 00:55).
15. Codex only for small error-checking tasks (24 Sep 17:27), then not at all: "কোডেক্স লাগবে না" [Codex is not needed] (17:41).
16. Documents in English: "ইংরেজিতে,বাংলায় না" [in English, not Bengali] (24 Sep 17:48, said of the Blueprint PDF; the assistant generalised it to all PDFs and documents).
17. Small, focused prompts for external AIs, and put every finding in the notes (24 Sep 13:36).
18. "আমার অনুমতি ছাড়া আর কখনো ফেবলকে ডাক দিবা না" [never call Fable again without my permission] (25 Sep 01:26).

Violations and near-violations:

| # | When | What happened | Rule | Severity |
|---|---|---|---|---|
| V-1 | 24 Sep 00:48 | Fable subagent "Fable: new estimator architecture" launched on the assistant's own initiative. The user's turn had asked "আর্কিটেকচার পরিবর্তন করব?" [shall we change the architecture?], not for Fable. Seven minutes later the user objected: "ফেবল অনেক টোকেন খায় … দ্রুত লিমিট শেষ হয়ে যাবে" [Fable eats a lot of tokens … the limit will run out fast]. The run died in the restart and was not resumed | 4 (21 Sep). The assistant relied on a blanket reading of rule 10 | **Likely violation** (of the rule's intent) |
| V-2 | 23 Sep 19:38 | Fable R7 ("missing kappa3 transport terms") launched with no Fable request in that turn, under the same blanket reading of rule 10. The user did not object | 4 / 10 | Grey area |
| V-3 | 25 Sep 01:25 | "Credit ফিরলে Fable-কে নতুন করে পাঠাব" [when the credit returns I'll send Fable again]: an intent to relaunch without asking. The user immediately made rule 18, and the assistant acknowledged "ওটা ঠিক হয়নি" [that was not right] | 4 | Near-violation (not executed) |
| V-4 | 23 Sep 16:47–21:10 | After the `/model` switch (15:58), every reply was in English, although the user had always written in Bengali. This triggered rule 11. After the rule, a few short English lines remained: 24 Sep 13:37 "Now the write-up (Algorithm Track) and the queue:"; 24 Sep 16:00 and 16:01 "Result: eps_shift = 2.9% …" | 11 (after 21:10) | Minor |
| V-5 | 24–25 Sep | Rule 16 as recorded by the assistant ("PDF আর ডকুমেন্ট ইংরেজিতে লিখব" [I'll write PDFs and documents in English], 24 Sep 17:50). Queue entries D14–D23, Q18 and Q19, added after that point, are still written in Bengali, and so is `Experiment_Queue.pdf` | 16 | Minor / needs the user's call: the queue was created in Bengali on 23 Sep, before the rule |
| V-6 | 22 Sep 06:40 → 08:18, 11:41 → 23 Sep 00:40 | After rule 5, the one-step GPU kernel was validated locally at width 64 but failed twice on Kaggle (GPU `torch.sort` OOM, then a non-PD Cholesky from fp32 accumulation). The width-160 dense-K4 push "doesn't fit on a T4 (n⁴ intermediates)": the OOM was not predicted | 5 ("predict OOM first", the assistant's own memory note) | Minor–moderate (Kaggle hours lost) |

No violations were found for rules 1, 6, 7, 13, 15 or 17. Rule 9: no demotivating message after 23 Sep 14:48, but see D-8 (the moonshot share was silently reduced by the 25 Sep A,C choice). The 24 Sep 23:07 request about extracting information via the reported API leak was stopped by the safety classifier and then declined ("ওই রিপোর্টে বর্ণিত পথে কোনো তথ্য বের করব না" [I will not extract information by the route described in that report]). That is correct and should stay closed.

---

## 4. Decisions and measurements whose conclusion changed: stale text still in the notes

| # | Old conclusion | Corrected by | Where the old text survives | Fix |
|---|---|---|---|---|
| C-1 | "Age contributions nearly orthogonal (squares sum ≈ 1)" (24 Sep 14:22) | Sol's correction, accepted 24 Sep 15:13: squares sum 0.24–0.49, the ages are coherent (Age-Gram) | `queue.json` "headroom": "বয়সের অবদান প্রায় সমান ও লম্ব" [age contributions nearly equal and orthogonal]; queue Q18 (FHQR) evidence "অবদানগুলো প্রায় পরস্পর-লম্ব" [the contributions are nearly mutually orthogonal]; `build_notes.py` K3aj **heading** "long, flat and nearly orthogonal across ages", with the body line 1260 still stating it (the correction follows at 1261); commit message `06ef6ea` | Edit the headroom string, the Q18 evidence and the K3aj title/body |
| C-2 | "λ is a fitted COMPENSATOR" (23 Sep 04:16) | 23 Sep 12:04: object mismatch (the chain carries the r = 1 core, F105); 23 Sep 17:20: the old oracle had a units bug | `build_notes.py` K3x **heading** (the body corrects it at line 1112); `UNTESTED_IDEAS.md` R13 "dead (measured): λ is a compensator" | Retitle K3x; amend R13 |
| C-3 | "Perfect κ4 is worth only ~9% (F105 ceiling); κ4 line closed" (23 Sep 12:04 and 14:43) | 23 Sep 17:20–17:32: the corrected-units oracle gives the true κ4 diagonal −30% | `UNTESTED_IDEAS.md` R14 "ceiling ≈ 9 % raw … dead (cost)". The write-up v3 and the notes line 1173 are already corrected | Amend R14 |
| C-4 | "A = P·diag(a_b)" (prompts, 24 Sep); then "A and P have no fixed relation, P⁻¹A drifts 100–970%" (24 Sep 21:22) | 24 Sep 23:01: A = P·R_b with a fixed full birth matrix, verified to 1e-14; the drift was an ill-conditioned inverse | Notes, prompts and queue are corrected (notes line 1303, queue D21). The 24 Sep 19:00 answer relayed to an external AI stated "A = P·diag(a_b)", and some Round 10 answers were built on the diag premise (triaged as such) | Nothing more, unless that AI's later answers are reused |
| C-5 | "Leaders = our accuracy at half the cost; moonshot = same accuracy at ≤ ~115 u" (Blueprint §1, 24 Sep 12:48–17:48) | 24 Sep 23:29–23:38: the top ~12 teams have 15–26% lower MSE, and several reached ~1.8e-8 from the public chain within days. New priority: "১. Accuracy … ২. খরচ … ৩. মুনশট" [1. accuracy … 2. cost … 3. moonshot] | Blueprint §1 "Key point" and target line; M4 "Partial"; M5 does not record the FB×β grid result (flat optimum, nothing to ship); no Q19 or D23 (25 Sep) in C-list or M4 | Update the Blueprint: add the LB-trajectory evidence, the reordered priorities, Q19, D23, and Q16 now running |
| C-6 | "Channel oracles measured on two networks" (write-up abstract) | §11.2 is MLP #0 only (see D-3) | `WRITEUP_DRAFT_v3.md` abstract | Reword, or replicate (D-3) |
| C-7 | Board and notes "next" lists frozen at 21–22 Sep | Work since | `build_notes.py` K4 "Next experiments queued" (22 Sep state: "Question D", "K-1 docs issue", "post K-3 after #331686 grades"); Part K3 table "K-3 … post pending" (now embargoed until after 17 Oct 23:59 UTC); `UNTESTED_IDEAS.md` "Priority order for the next 48 h" (21 Sep), R31 open, A2 untested | Mark the board as superseded by the queue, or sync it; refresh K4 and the K3 table |
| C-8 | `correspondence/README.md` "**send today**" for `K4_permission_request.md` and `K3_email_to_organisers.md` | K4 was sent as a Discourse PM to @mohanty on 23 Sep 13:14. Whether the K3 research email was sent is unconfirmed (see §5) | README | Update after asking the user |
| C-9 | Two write-up drafts | `writeup/README.md` names `WRITEUP_DRAFT_v3.md` (24 Sep) "current"; `docs/writeup/Writeup_Draft.html` "draft v0" (25 Sep) is newer and carries the 25 Sep results | Both files | Pick one master and say so in both READMEs |
| C-10 | S = 2 "open paradox" | Astra R9: a plain quadratic cancellation (fits −0.35…−0.8%); Fable R8: the K×K explanation is dead | Write-up v3 §11.4 still says "an open paradox" | Resolve through Q15, or cite Astra's explanation as a hypothesis |

---

## 5. Pending external actions and dates (BST)

| Date | Action | Owner | State |
|---|---|---|---|
| now | Confirm whether `K3_email_to_organisers.md` (the dated research record, "send today" on 23 Sep) was actually sent. The user's 23 Sep 13:14 "হয়েছে। রিপ্লে আসলে জানাবো" [done; I'll tell you when a reply comes] referred to the PM | user | unconfirmed |
| now → before 18 Oct | Posting rights: the PM to @mohanty (23 Sep 13:14) has no recorded reply. Follow up via route B `devops@aicrowd.com` well before the embargo lifts, otherwise the community post cannot go out | user | waiting |
| now | K2 security report (sent 21 Sep 03:39): no reply from `arc-whestbench@` as of 23 Sep. Offer stands to re-test once fixed; the endpoint must not be used for others' data | organisers / user | waiting |
| now (optional) | File the community issues K-1/K-4/K-5 on the starter-kit GitHub (D-4) | user (GitHub) | not started |
| 25 Sep ~13:30 | Q16 (aug64 / transport) 100-MLP results from Kaggle runners A and B | assistant | running |
| **2 Oct** | Team freeze / merge deadline. Option D ("Team merge") was offered on 25 Sep 09:30; the user chose A and C, so no merge is planned. The deadline itself is still live | user | decision taken (no merge), unless revisited |
| ~2 Oct | Blueprint gate: "One mechanism passes a one-layer falsifier (otherwise shift time to M4/M5)". In effect already shifted by the 25 Sep A,C choice | — | update the Blueprint |
| **15–16 Oct** | Nominate the two final submissions (board E1: one max-score, one conservative). Current best #332110 (5.1284e-9), then #331787 (5.137e-9) | user + assistant | open |
| **18 Oct 05:59** (17 Oct 23:59 UTC) | Phase 2 submissions close; daily limit 10 | — | — |
| after 18 Oct 05:59 | Publish the embargoed forum content (new topic, or reply in 18218) | user | embargoed |
| **24 Oct** | Write-up deadline (algorithmic $20k, community $500–5k); still missing figures, derivation hand-checks and a human edit | assistant + user | draft v0 / v3 |
| 15 Nov | Winners announced | — | — |
| (20 Sep advice) | "Final submission-এর আগে একবার login করে Rules ট্যাবটা নিজে পড়ে নেবেন" [before the final submission, log in once and read the Rules tab yourself]. Rules acceptance is implied by the accepted submissions; reading the full rules is unconfirmed | user | unconfirmed |

---

## 6. Items verified DONE (compact)

| Request, promise or idea (BST) | Where done |
|---|---|
| PDF handbook, Phase 2 reorganisation, HF card, forum digest, prize tracks J1–J3 (20–21 Sep) | `docs/ARC_WhestBench_2026_Complete_Notes.pdf` Parts A–K |
| Submissions V29 → MIN=8 → FIRST=5 → angular → V31b → v40fb | #331681, #331686, #331732, #331787, #331801/#331805, #332110 |
| Young-only L6 and R_OLD2 192 "queue those variants next" (21 Sep 02:46) | board D2 (dead), knob sweep K3f |
| Old-tier knob sweep (R_OLD2, LAM_SCALE, BETA, AGE_OLD) | K3f |
| Question A pilot, Fable R1 C1 keep-last and sensitivity map | K3b, K3c |
| Write-up backbone in the PDF; TODO sections filled (21 Sep) | J2.1; `WRITEUP_DRAFT_v2/v3` |
| Re-read all AI answers and list the untested ideas (21 Sep 13:47) | `writeup/UNTESTED_IDEAS.md` |
| Compensating closure (Astra Extra High Q3), including the corrected γ | board A1, K3k/K3o |
| Audit of the failed experiments for bugs (22 Sep 01:26) | K3o (13/14 sound; the γ axis re-run) |
| Float64 reference, angular re-tune, rank-cliff recheck, A5 coupled closure | K3p, K3n, board A5 |
| Residual-structure study; one-step closure and copula; TT-cross | board R9–R11, K3t/K3v |
| AGE_OLD2 / QPASS2 sweep; K4-vs-K3 width ladder (96/128; 160 does not fit) | R12, K3w |
| κ4 teacher → per-row λ (V35); corrected-units oracle | K3x/K3y, K3ab |
| Professional GitHub reorganisation + private push (22 Sep) | repo `Foysal348/arc-whitebox-estimation-2026` |
| Net-wide GitHub/HF sweep, recorded in English (23 Sep) | K3y, `results/summaries/websweep/` |
| 100-MLP harness; re-check of the small knobs lost to 10-MLP noise | queue Q3 |
| Own ideas A (per-family Strassen) and C (Richardson); layer-dependent age gate | R16–R18 |
| Unbiased HT column sampling; Fable R6 asymmetric contraction rank (V38) | R19–R21 |
| Error budget, channel oracles, teacher-forced closure (Q5) | K3ab–K3ad |
| Fable R7 / R8 (K×C half families; (1,1,1,1) term); cheap late-layer kernel | K3ae–K3ag, queue Q1/Q8/D9 |
| Experiment-queue PDF, priority-ordered (23 Sep request) | `docs/experiment_queue/` |
| FB_SCALE 1.25 (+β 1.2) validated and shipped; FB×β grid | queue Q2; #332110 |
| Cost ledger; V41 join (local); young-tier algebra; Sonnet contraction search | K3ah, queue D11/D21 |
| DHMP gate, horizon oracle, Age-Gram, shift carrier, irreducibility | K3ai/K3aj, queue D12–D15 |
| Stage-1 per-channel bar; Round 11 falsifiers and re-verification | K3ak/K3al, queue D16–D20 |
| LB re-analysis and competition re-read (24 Sep 23:16 request) | Round 12 prompt (`dddb9a1`) |
| Opus R12 + six-point review; Q18 κ4 horizon; Q19 online correction; D23 late-layer κ3 oracle | K3am, queue D22/Q18/Q19/D23 |
| Fable's feedback-only D21 oracle (Q4 / R31) | Opus R12 (×0.9999); the queue is stale (§2) |
| Fable's (2,1,1) cost–gain question (23 Sep 19:17) | Fable R7/R8: dead |
| Moonshot Blueprint, in English (24 Sep) | `docs/moonshot/` |
| Permission-prompt fix (the user's explicit request, 25 Sep 00:46–01:18) | `.claude/settings.local.json` |

---

## 7. Count summary

- **Records examined:** 691 (185 user messages, 493 assistant text blocks, 6 hand-backs, 4 summaries). Also checked: 10 subagent launches (6 Fable, 3 Opus, 1 Sonnet) plus 1 SendMessage, ~47 Kaggle push/upload commands (including retries) and 7 AIcrowd submissions.
- **Requests, promises and ideas (types 1–3):** about 78.
  - **DONE:** about 55 (section 6 groups them into 29 rows).
  - **OPEN-TRACKED:** 9: Q16 (running), Q14, Q15, Q9, Q13, M3, the write-up, the E1 nominations, the embargoed forum post.
  - **DROPPED:** 14 (D-1 … D-14). Five matter for score or prize: D-1 to D-5.
- **Stale tracking:** 5 queue or board entries marked open but actually done or dead (Q4/R31, Q6, Q17, duplicate Q18, the A4/B1/B4 rows).
- **Contradicted conclusions still in the notes (type 4):** 10 (C-1 … C-10).
- **Rule violations:** 6 entries: 1 likely violation (V-1, Fable), 1 grey area (V-2), 1 near-violation (V-3), 3 minor (V-4 language, V-5 document language, V-6 unpredicted Kaggle OOMs). No violations found on the API key, UTC/BST, Codex, submit-freely, parallel-Kaggle or leak-exploitation rules.
- **Pending external actions:** 13 (§5). The hard dates are 2 Oct (team freeze), 15–16 Oct (nominations), 18 Oct 05:59 BST (close), 24 Oct (write-up).
