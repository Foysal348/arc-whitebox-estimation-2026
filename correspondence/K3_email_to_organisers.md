> CORRECTION (25 Sep 2026): the claim that the 7-22% trim seen by 504aldo on 14 Sep was ours is FALSE - our first submission was 21 Sep and our trims are ~4%. A correction email was drafted (K3b_correction_email.md). Rewrite this draft's opening before any public use.

HOW TO SEND
===========
To:      arc-whestbench@aicrowd.com
Subject: Phase 2, dated record + posting-permission request: where the 7-22% cost trim on the public K=3 chain is
Body:    everything below the line. Attach nothing; all numbers are inline.

Do NOT post this publicly yet. Public release is planned for after the submission deadline
(17 October 2026, 23:59 UTC) - see correspondence/README.md for the reason and the schedule.
A private message to @504aldo is also deferred for the same reason: he is a competitor.

---

Hello,

Two things in one message: a community contribution for Phase 2, and a small account request.

**The request first.** My forum account (foyshal_sk, trust level 2) has neither a reply control nor
a "New Topic" button in the ARC White-Box Estimation Challenge 2026 category, so I cannot post in it
at all. Other users, including first-time posters, are posting there normally, so I assume a
group-permission sync issue on my account rather than a category setting. If you can enable posting
for that account I would be grateful.

**On timing, so there is no misunderstanding.** I intend to publish the material below on the forum
myself, but after the Phase 2 submission window closes on 17 October, because two of the items are
directly applicable cost reductions on a public estimator and publishing them now would simply hand
my current standing to the thirteen-odd accounts submitting that estimator verbatim. I am sending
everything to you today so that it is on record with a date, and so that the answer to 504aldo's
open question exists somewhere other than my own notes. Please treat this email as a dated record
rather than as something to publish on my behalf.

**The contribution.** In topic 18218, post #3, 504aldo asked what the 7-22% cost trim visible on
the leaderboard was, since his own cost anatomy had not found it. That was my submissions. The
answer is two constants in the block-recursion of his public V29 chain, and the numbers are below.
Everything is the same arithmetic as V29 (prediction drift MSE <= 6e-13 against it, i.e. float32
noise); only the Strassen recursion changes. Graded numbers are from the live leaderboard; local
numbers from `whest run` on the public mini split with the baked 1e9-sample truth.

  variant                              util (steady)   peak commit   drift MSE   graded
  V29 as shipped (L5, MIN 32)          0.2540          5.56 GB       -           5.418e-9  (#331681)
  L5, STRASSEN_MIN = 8                 0.2442          5.97 GB       4.9e-13     5.220e-9  (#331686)
  L5, MIN = 8 + STRASSEN_FIRST = 5     0.2432          6.18 GB       4.9e-13     5.195e-9  (#331732)
  L6, MIN 16                           0.2421          8.13 GB       6.3e-13     not submittable
  L7, MIN 8                            -               OOM (1.12 GiB single pooled buffer)  not submittable

1. STRASSEN_MIN 32 -> 8 is the trim. With MIN = 32 the shared-basis families (rank 384 / 224) stop
   at Strassen level 3 / 2 because their block sides fall below 32; with MIN = 8 the 384-families
   reach level 5 (384/32 = 12 >= 8) and the 224-families level 4 (224/16 = 14 >= 8). The 1024-side
   families already ran all five levels either way, so memory barely moves (+0.4 GB). Net -3.9% of
   C locally, -3.7% graded, at unchanged raw MSE.

2. STRASSEN_FIRST 4 -> 5 is the second, smaller one. The first predict() of each grader worker pays
   the pooled-buffer page touch, which is metered; letting that call run at full depth anyway is
   worth another -0.4%, graded 5.220e-9 -> 5.195e-9.

3. Level 6 is blocked by the 8 GB address-space cap, not by FLOPs. Each level adds pooled
   (b, 7, 7^(k-1), n/2^k, n/2^k) buffers, about 1.75x the previous level's; at level 6 the commit
   peaks at 8.13 GB on the second predict() of the process, above RLIMIT_AS, so the worker would
   die. It would only have bought -4.7% anyway: on a 1024-side family the level-6 block additions
   (18 * 7^5 * 16^2 ~ 77 MFLOP) eat over half of that level's matmul saving ((1/8)(7/8)^5 * 2n^3 ~
   138 MFLOP), and at level 7 (135 vs 120 MFLOP) they cancel it outright. On a 31 GB machine
   without the cap, L6 scores identically to L5 on 10 mini MLPs (raw 2.302e-8 vs 2.303e-8) at util
   0.2416, but wall time goes 90 -> 150 s per MLP on four cores.

**One accuracy item that is not engineering.** The networks have no biases, so they are positively
homogeneous and the Gaussian input factorises exactly as X = (R/sqrt(n)) Y, with Y uniform on the
sphere of radius sqrt(n) and R ~ chi_n independent of it. Hence E[h_l(X)] = a1 * E[h_l(Y)] exactly
at every layer, with a1 = sqrt(2/n) * Gamma((n+1)/2) / Gamma(n/2) = 0.99975589 at n = 1024. The
spherical input has C = I and kappa3 = 0 like the Gaussian one, but kappa4 = c * Sym3(I x I) with
c = -2/(n+2), which a chain that already carries a kappa4 channel can absorb as a layer-0 injection,
with the output rescaled by a1. Cost +0.02% of budget; graded raw 2.132e-8 -> 2.108e-8 (-1.1%),
score 5.195e-9 -> 5.137e-9 (#331787). The idea is not mine: it is from barnobarno666's public
Phase-8 report, where it was worth -7.9% on a plainer K=3 chain.

**A kappa4 measurement, relevant to the F68/F105 exchange in that thread.** I ran a Monte-Carlo
kappa4 teacher (single-pass raw moments to central cumulant slices; 6e7 samples per layer; two
independent 1024x16 networks, one a public mini MLP) and fitted the lambda law against the raw
cumulant slices, before F105 was posted. Consistent with 504aldo's correction that the chain's G is
the r=1 core rather than a raw slice:

  - On the raw (3,1) slice, lambda * C_off gives R^2 0.35-0.60 rising with depth; adding the kappa4
    diagonal terms gives 0.55-0.71; adding the chain's own D21 gives 0.74-0.80. Same on both nets.
  - The true weight on that slice is per row, kappa4(i)/var(i), and the per-layer scalar lambda_l
    equals the mean of that ratio to within 2% (6.8e-3 vs 6.9e-3 at layer 15).
  - But implementing the per-row weight makes the final mean monotonically worse: raw 2.2911 ->
    2.2923 -> 2.2933 -> 2.2958 -> 2.2988e-8 as the blend goes 0 -> 0.25 -> 0.5 -> 1.0 -> 1.5 (10
    mini MLPs, baked truth). The lesson is that channel-level fidelity arguments prove nothing here:
    the chain propagates the core, the scalar already fits it, and improving the model against a
    different object degrades the result.

Two further negative results, in case they save someone time: replacing the old-source tier's
shared basis by a pooled Tucker/TT core needs multilinear rank ~250 for 2% query error and is net
negative at every rank measured; and the chain's final-layer residual is not learnable from cheap
per-neuron features - cross-validated over all 100 public networks (102,400 output neurons), ridge
on 19 features recovers 0.3%, a quadratic expansion 0.7%, gradient boosting 0.5%, with every
single-feature correlation <= 0.04.

Disclosure: the experiments and this text were produced with LLM assistance (Claude), directed by
me; every number above is from `whest run`, the grader, or a stated Monte-Carlo run.

Best regards,
Foysal Emon Shanto (AIcrowd: foyshal_sk)
