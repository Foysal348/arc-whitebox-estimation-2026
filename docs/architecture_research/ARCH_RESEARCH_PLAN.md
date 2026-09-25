# New-architecture research programme (26 Sep – 3 Oct 2026)

Goal: an estimator architecture different from the public sum-of-sources K=3 chain that beats our best graded
submission (#332322: adjusted 4.860e-9, raw 2.210e-8, utilisation 0.220) within 7 days, target ≤ 3.5e-9
(e.g. raw 2.0e-8 at util 0.175, or raw 1.8e-8 at util 0.195). Submissions close 18 Oct 05:59 BST; 10 per UTC day.

## What we know (constraints any new architecture must respect)

- Closures are exact given true inputs; the error is in the inputs (variance, κ3, κ4 channels), accumulated over depth.
  Perfect κ3 input ×0.61 (MLP #0) / ×0.71 (synthetic); perfect κ4 diagonal ×0.70; forcing the true mean −81%.
- κ3 and κ4 "all-distinct" content is carried memory over ≥ 6 layers; it is not low-rank, not compressible by any linear
  representation closed under dense maps, and not predictable from per-neuron (even) statistics.
- Cost: dense n³ products are the unit (1 u ≈ one 1024³ product, 0.51 u with Strassen L5). 150 u ≈ util 0.146.
- ~50 alternative formulations have been measured dead (Moonshot Blueprint dead list; queue D1–D27;
  consultations/moonshot_ideas_audit_2026-09-26.md). Do not re-propose them without showing why the kill was narrow.
- Leaderboard facts (data/lb_scan): top teams built their own methods from 24 Aug; by 2 Sep J2W was at raw 2.23e-8 @ 0.211
  and Puffi at 2.15e-8 @ 0.207 (close to our chain); Strassen adoption (4–6 Sep) cut 20–30% cost board-wide; marius_binner
  was at raw 1.81e-8 as early as 2 Sep; J2W gained −12% raw at fixed cost on 17 Sep; jlacombe / AndreasHad04 gained
  −5.5% / −7.2% raw at identical util (zero-FLOP changes). Puffi (Phase 1 public write-up, forum 18175) describes its stack:
  factorised K=3 carriers (CP columns, atoms Sym(v,M), shared-middle and paired groups) + SSC κ4 response closure + a
  nonlinear drift corrector; Phase 2 score 2.8e-9 (raw 1.94e-8 @ 0.146).

## Resources

- Teachers (6e7-sample MC): pre-activation mean/C/Y21/K4d/K22/K31 at every layer of public MLP #0; post-ReLU at 3/7/11/14/15;
  synthetic net (even layers + 15). Baked 1e9 truth for MLP #0 (all layers); HF mini split (100 MLPs, all-layer truth).
- Dense captures of the chain's sources at L7/11/14/15; chain channel dump (errbudget_v29plain.npz).
- Local harness: experiments/chain/run_variant_teacher.py (two teacher networks, ~2 min per config per net).
- Kaggle: 3 CPU runners (100-MLP paired evaluation, ~3 h per variant); grader: 10 submissions/day.
- Code: submissions/v44rfb8 (current best), 504aldo's repo + findings log (whest-p2-cumulant-k3), ARC's reference
  mlp_cumulant_propagation (factor_k3.py; see notes K3i), Puffi's PDF (scratchpad/puffi_ssc.txt), barnobarno666 repo.

## Plan

- Phase A (26–27 Sep): architecture design study. Enumerate 4–6 concrete candidate architectures (not tweaks), each with
  state, per-layer update, exact cost at n = 1024, the accuracy argument, and the cheapest falsifier on stored data.
  Include at least: (1) a reconstruction of Puffi's published stack as far as the PDF allows, re-costed at 1024 × 16;
  (2) a design that spends the budget differently from our chain (e.g. a cheaper κ3 carrier plus a stronger κ4/variance
  channel); (3) any design consistent with a zero-FLOP −5–7% accuracy step. Run the falsifiers; pick ONE.
- Phase B (28 Sep – 1 Oct): prototype the chosen architecture as an estimator; iterate on the two teacher networks;
  first 100-MLP Kaggle run; first submissions.
- Phase C (2–3 Oct): tune, combine with proven trims (transport-style joins, R_FB, Strassen), submit.
- Kill rule: if no Phase-A candidate passes its falsifier by 27 Sep evening, report and stop the programme; if the
  prototype is not within 20% of the chain's score by 1 Oct, stop and fold the useful parts into the chain.

The incremental track (submissions, zero-FLOP tuning) continues in parallel.
