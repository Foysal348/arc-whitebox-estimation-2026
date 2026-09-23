EMBARGOED UNTIL 18 OCTOBER 2026
===============================
Do not post before the Phase 2 submission window closes (17 Oct 2026, 23:59 UTC).
Reason: sections 1-2 are directly applicable cost reductions on a public estimator; publishing them
during the window would hand our standing to the accounts submitting that estimator verbatim.
The write-up deadline is 24 Oct, so posting on 18-20 Oct still counts for the contribution tracks.
Sent to the organisers by email on 23 Sep 2026 as a dated record (K3_email_to_organisers.md).

Reply to: https://discourse.aicrowd.com/t/18218 (504aldo, "Everything we tried: a factorized K=3 cumulant-propagation estimator at 0.25 x B...")

Paste everything below the line into the reply box. It is Markdown; Discourse renders the table as-is.

---

**The 7–22% trim you saw on the leaderboard was Strassen block size, and here are the numbers**

In post #3 you said you could see someone had trimmed 7–22% of the chain's cost without losing accuracy, and that the cost anatomy hadn't found it for you. That was me, so here it is in full. Thanks for the MIT release — it is what got me on the board (#331681, V29 verbatim: 5.418e-9 adjusted, raw 2.133e-8, util 0.254, 0 failed).

Everything below is the *same arithmetic* as your V29 — prediction drift against it is MSE ≤ 6e-13, i.e. float32 noise. Only the block recursion changes.

| variant | util (steady) | peak commit (local) | drift MSE vs V29 | graded |
|---|---|---|---|---|
| V29 as shipped (L5, MIN 32) | 0.2540 | 5.56 GB | — | 5.418e-9 (#331681) |
| **L5, STRASSEN_MIN = 8** | **0.2442** | 5.97 GB | 4.9e-13 | **5.220e-9 (#331686)**, raw 2.132e-8, 0 failed |
| L5, MIN = 8 + STRASSEN_FIRST = 5 | 0.2432 | 6.18 GB | 4.9e-13 | **5.195e-9 (#331732)** |
| L6, MIN 16 | 0.2421 | **8.13 GB** | 6.3e-13 | not submittable |
| L7, MIN 8 | — | OOM: one pooled (10, 7, 16807, 16, 16) buffer is 1.12 GiB | — | not submittable |

**1. `STRASSEN_MIN` 32 → 8 is the trim.** With MIN = 32 the shared-basis families (rank 384 / 224) stop at level 3 / 2, because their block sides fall below 32; with MIN = 8 the 384-families reach level 5 (384/32 = 12 ≥ 8) and the 224-families level 4 (224/16 = 14 ≥ 8). The 1024-side families already run all five levels either way, so memory barely moves (+0.4 GB). Net −3.9% of C locally, −3.7% graded, at unchanged raw MSE.

**2. `STRASSEN_FIRST` 4 → 5 is the second, smaller one.** The first `predict()` of each grader worker pays the pool page-touch, which is metered; letting it run at full depth anyway is worth another −0.4% (0.2442 → 0.2432), graded 5.220e-9 → 5.195e-9.

**3. Level 6 is blocked by the 8 GB cap, not by FLOPs.** Each level adds pooled `(b, 7, 7^(k-1), n/2^k, n/2^k)` buffers, about 1.75× the previous level's; at level 6 the commit peaks at 8.13 GB on the *second* `predict()`, above the grader's `RLIMIT_AS`, so the worker would die. It would only have bought −4.7% anyway, because the 18 block additions per level start to cost as much as the matmul saving: on a 1024-side family the level-6 additions (18·7⁵·16² ≈ 77 MFLOP) eat over half of that level's matmul saving (⅛·(7/8)⁵·2n³ ≈ 138 MFLOP), and at level 7 (135 vs 120 MFLOP) they cancel it. On a 31 GB machine with no cap, L6 scores identically to L5 on 10 mini MLPs (raw 2.302e-8 vs 2.303e-8) at util 0.2416, but wall time goes 90 → 150 s per MLP on four cores, which would also flirt with the 120 s limit.

So on this representation the block recursion is finished at MIN = 8 / L5 / FIRST = 5 (≈ 0.243). The patch is literally two constants in `estimator_v29.py`; happy to post the diff and the measurement script if useful.

**One accuracy item that is not engineering.** The network has no biases, so it is positively homogeneous, and the Gaussian input factorises exactly as X = (R/√n)·Y with Y uniform on the sphere of radius √n and R ~ χ_n independent of it. Then E[h_l(X)] = a₁·E[h_l(Y)] exactly, with a₁ = √(2/n)·Γ((n+1)/2)/Γ(n/2) = 0.99975589 at n = 1024. The spherical input has C = I and κ₃ = 0 like the Gaussian one, but κ₄ = c·Sym₃(I⊗I) with c = −2/(n+2), which your own κ₄ machinery can absorb at layer 0 (a previous-layer core g = −6/(n+2), i.e. dG = g·var and a (3,1) slice g·C_off), and then the output is rescaled by a₁. Cost: +0.02%. Graded: raw 2.132e-8 → 2.108e-8 (−1.1%), 5.195e-9 → **5.137e-9** (#331787). The idea is not mine — it comes from barnobarno666's public Phase-8 report, where it was worth −7.9% on a plainer K3 chain; on V29 most of it is already absorbed by the κ₄ regeneration and the mean rider, but the remainder is free.

**And one negative result on F68/F105, since the thread is already on it.** I ran a Monte-Carlo κ₄ teacher (single-pass raw moments → central slices, 6e7 samples per layer, two independent 1024×16 networks, one of them a public mini MLP) and fitted your λ law against the *raw* slices before F105 was posted. Two things, both consistent with your correction:

* On the raw (3,1) slice, λ·C_off gives R² 0.35–0.60, rising with depth; adding the κ₄ diagonal terms gives 0.55–0.71, and adding the chain's own D21 gives 0.74–0.80 — same pattern on both networks. But that is the raw slice, not your r = 1 core, so it does not contradict the ≥ 0.95 pre-activation fit.
* The true weight on the raw (3,1) slice is per-row, κ₄(i)/var(i) — and your per-layer scalar λ_l equals the **mean** of that ratio to within 2% (6.8e-3 vs 6.9e-3 at layer 15). Implementing the per-row version in the chain makes things monotonically *worse* (raw 2.2911 → 2.2923 → 2.2933 → 2.2958 → 2.2988e-8 as the blend goes 0 → 0.25 → 0.5 → 1.0 → 1.5, 10 mini MLPs with baked truth). Read together with F105: the object the chain propagates is the core, the scalar already fits it, and "improving" it against a different object degrades it. Your 2.23e-8 / 2.04e-8 / 4.20e-8 ladder is the honest ceiling for that channel.

Disclosure: the experiments and this note were produced with LLM assistance (Claude), directed by me; every number above is from `whest run`, the grader, or a stated Monte-Carlo run.
