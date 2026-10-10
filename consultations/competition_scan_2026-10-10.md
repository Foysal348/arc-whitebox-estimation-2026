# Competition scan, 10 Oct 2026: overview, leaderboard, forum, write-up rules

## Leaderboard (Phase 2, top 50, fetched 10 Oct evening)

| rank | entry | adjusted | final MSE | util |
|---|---|---|---|---|
| 1 | J2W (T) | 1.4e-9 | 1.36e-8 | 0.107 |
| 2 | marius_binner | 1.6e-9 | 1.14e-8 | 0.144 |
| 2 | Luna (T) | 1.6e-9 | 1.40e-8 | 0.118 |
| 2 | oliver_chen | 1.6e-9 | 1.23e-8 | 0.131 |
| 2 | mliston | 1.6e-9 | 1.39e-8 | 0.117 |
| 6 | mk1981 | 1.7e-9 | 1.65e-8 | 0.106 |
| 9 | Masato (19 entries) | 2.0e-9 | 1.45e-8 | 0.140 |
| 11 | lode_dockx | 2.1e-9 | 2.10e-8 | 0.102 |
| 24 | reds / carlos_rodriguez | 2.8e-9 | 1.73e-8 | 0.162 |
| 36 | hyojun_kwon | 3.4e-9 | 2.47e-8 | 0.138 |
| 42 | **Hack2Publish** (106 entries) | 3.7e-9 | 1.97e-8 | 0.187 |

**Reading.** Our raw MSE (1.97e-8) is already in the range of ranks 11–40. Several entries reach 1.1–1.7e-8, but the dominant gap is COST.
- lode_dockx (#11) has V29-level accuracy (2.10e-8) at util 0.102, the floor.
- At our MSE, util 0.10 would score about 2.0e-9 (about rank 9). Util 0.14 would score about 2.8e-9 (about rank 24).
- The score has a floor at util 0.10 (forum 18219, verified): below it, only MSE counts.

## Forum facts that matter

- **504aldo (thread 18218):**
  - "The top of the board (raw 1.7–2.0e-8 at 0.145–0.16 × B) is better on both axes."
  - The leaders' bills roughly equal 504aldo's own without the old-source tier (0.150 × B).
  - 504aldo's guess, unvalidated: the leaders' method "is not a per-source hub-tensor expansion at all".
- **Wu et al. (arXiv 2605.05179).** The factorised K ≥ 3 algorithm costs about L²·n^K. The L² comes from per-source accumulation. A representation without per-source accumulation would remove the depth factor that dominates our bill.
- **yujigon (thread 18216, Phase 2 write-up).** A rank-4,608 CP channel carrying the inherited κ₃ at 9.9 % of budget. The accuracy is poor for other reasons, but a global CP / hub pool is a cheap κ₃ carrier.
- **Selection and private re-evaluation (thread 18118, mohanty, 4 Oct).**
  - Submission selection opens one week before the Phase 2 close, announced on the forum and by email.
  - Before that, a new flopscope version fixing the reported billing issues (GitHub #264–#285) will be released, and affected submissions re-evaluated.
  - Prizes are decided only by private re-evaluation on a new suite with fresh seeds.
- **Eligibility (thread 18240).**
  - Building on MIT-licensed forum code is fine with clear credit and substantial own changes.
  - Retaining the MIT notice and disclosing the change in the write-up is sufficient.
  - For the algorithmic-contribution prize, describe the reference method separately; only your own contribution is judged.

## Write-up rules

- **Deadline:** 24 Oct 2026, 23:59 UTC (25 Oct, 05:59 Bangladesh time).
- **How to submit:** email arc-whestbench@aicrowd.com or post on Discourse, including the submission number it relates to.
- **Format, from community examples:** a forum post (~1,500 words) with an attached PDF and a public repository for reproduction, plus an LLM-use disclosure.
- **Not specified:** no length or format limits are published.
- **Prize pool:** Phase 2 prizes total $100k. Community-contribution prizes are typically $500–5,000 per individual. The algorithmic-contribution prize amounts are not stated on the pages read.
- **Our gaps:**
  - Our PDF is 44 pages. A concise main document (8–12 pages) with the full paper as an appendix will read better.
  - Name the reference (504aldo V29) and our deltas in a separate section.
  - Cite the submission number (#334870).
  - The team has no forum posting rights, so submit by email.

## Ideas this suggests

1. **Cost, not accuracy, is the lever.** Target util 0.12–0.14 at roughly today's MSE. Within our representation, cost cuts trade one-for-one with MSE (paper §3.5), so this needs a carrier without per-source accumulation, e.g. a global hub pool or CP pool of fixed size recompressed each layer.
2. **Watch the flopscope billing-fix release.** It may reprice our submissions before selection.
