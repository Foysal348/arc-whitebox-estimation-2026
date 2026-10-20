# Shared brief for the 19 Oct idea hunt (read this first)

## Competition and score
- ARC White-Box Estimation Challenge 2026, Phase 2 (WhestBench). Predict the per-neuron means of every post-ReLU layer of a
  random bias-free He-initialised ReLU MLP (width 1024, depth 16), input x ~ N(0, I), without sampling.
- **Score:** final-layer MSE × max(0.1, FLOPs/2^41). 1 u = 2^31 FLOPs, so the budget is 1024 u.
- **Grader limits:** residual Python time 0.4 s per MLP (scales with the number of flopscope ops; keep warm ops near
  ≤ 20k), 8 GB, 120 s wall. Estimators must never import plain numpy.
- **Deadline:** submissions close 22 Oct 2026 05:59 Bangladesh time, i.e. **about 50 hours** from now. Anything proposed
  must be buildable and testable on Kaggle CPU within ~1 day.

## Where we are
| Submission | What | Score | Raw MSE | util |
|---|---|---|---|---|
| #337175 (grading) | e30ak4c40k31 = e30_a + our κ₄ corrector (scale 0.4) + two-term (3,1) closure (mix 0.25) | ~3.034e-9 (predicted) | ~2.05e-8 | 0.152 |
| #337041 | e30ak4c40 | 3.0506e-9 | 2.0093e-8 | 0.152 |
| #337049 | e299safe (our own lineage: E299 + e30_a cost savings + FIJA + C_pre + (3,1)) | 3.2529e-9 | 1.9107e-8 | 0.170 |

- **Target:** below 2.95e-9 (the board shows 2 significant figures), i.e. at least **−2.8 %** from 3.034e-9. Gains
  multiply: raw MSE × util.
- **Base file to improve:** `submissions/e30ak4c40k31/estimator.py`. It is LordProcciuto's public e30_a (MIT; see
  `external/e30_a_estimator.py`, `external/e30_a_WRITEUP.md`) plus our corrector knobs X5_* and the X2_K31* closure.
- **Cost ledger of e30_a on MLP #0:** `data/kaggle_out/ledger/ledger_e30a.json` / `.txt` (158.8 u). Ours for comparison:
  `ledger_e299xrk.json` (184.1 u).
- **Leaderboard context:** top-10 at 1.4–1.9e-9; teams at 2.2–2.7e-9 have raw 1.45–2.0e-8 at util 0.125–0.16.

## Already measured (do NOT re-propose without a new angle)
- **K4C blend/scale, λ, β, FB, thresholds** on our lineage were tuned. Re-tunes of e30_a's own knobs (X_LAM_TILT,
  X_DEAD_TAU, X_VB_HOT, (3,1) mix, corrector scale) are **running on Kaggle now**.
- **e30_a's row compaction in our chain:** ×1.063 (bad). C_pre compaction: ×0.988. Fixed basis in our chain: ×0.972 (used
  in e299safe).
- **Two-term (3,1) closure:** full strength hurts; mix 0.25 helps 0.4–0.55 %.
- **Corrector on e30_a:** scale 0.3–0.4 ×0.975–0.977; the star is essential; the E299 rider does not transfer.
- **Our paper's kill table** (`writeup/ALGORITHM_TRACK_PAPER.md` §6) has 60+ dead architectures. Main lesson:
  single-channel local fixes often hurt end to end; only joint fixes or cost cuts paid.
- **flopscope 0.12.3** billing does not change our chains.

## Rules for every agent
- **Read-only.** Do not run any estimator on this laptop (out of RAM), do not push to Kaggle, do not submit, do not
  git commit. Small pure-python arithmetic is fine.
- **Be sceptical.** Separate *measured* (with a number and a source file or post) from *claimed by others* and from
  *your speculation*. Outside-AI documents often contain unsupported claims; our paper §6.4 lists some.
- **Output** (your final message): a ranked list of at most 8 candidate experiments. For each give:
  - one-line idea;
  - where it comes from (file, section or post);
  - why it might work on e30ak4c40k31 specifically;
  - expected gain in % of score, with its basis;
  - implementation effort in hours, and which code region / knob;
  - risk (ops / residual / memory / rules);
  - a concrete Kaggle test: arms and env knobs.

  Also give up to 3 "failed ideas that might work now", with the reason. Keep the report under ~900 words.
