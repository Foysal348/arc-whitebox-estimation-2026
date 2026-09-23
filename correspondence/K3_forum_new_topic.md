HOW TO POST THIS
================
The reply button is not available on topic 18218 for this account, but the "+" (New Topic) button in the
left sidebar is. Post this as its own topic in the "ARC White-Box Estimation Challenge 2026" category.

TITLE (paste into the title box):

[Phase 2] Where the 7-22% cost trim on the public K=3 chain is: two Strassen constants, measured, plus a free -1.1% from a spherical input

TAGS (optional, if the box offers them): phase-2

BODY (paste everything below the line):

---

@504aldo asked in [his thread](https://discourse.aicrowd.com/t/18218) (post #3) what the 7-22% cost trim visible on the leaderboard was, since his own cost anatomy had not found it. That was my submissions, so here are the numbers. I cannot reply in that thread (no reply button on my account), hence a separate topic.

Everything below is the **same arithmetic** as the public V29 chain — prediction drift against it is MSE ≤ 6e-13, i.e. float32 noise. Only the block recursion changes. All graded numbers are from the live leaderboard, all local numbers from `whest run` on the public mini split with baked 1e9 truth.

| variant | util (steady) | peak commit (local) | drift MSE vs V29 | graded |
|---|---|---|---|---|
| V29 as shipped (L5, MIN 32) | 0.2540 | 5.56 GB | — | 5.418e-9 (#331681) |
| **L5, `STRASSEN_MIN` = 8** | **0.2442** | 5.97 GB | 4.9e-13 | **5.220e-9** (#331686), raw 2.132e-8, 0 failed |
| L5, MIN = 8 + `STRASSEN_FIRST` = 5 | 0.2432 | 6.18 GB | 4.9e-13 | **5.195e-9** (#331732) |
| L6, MIN 16 | 0.2421 | **8.13 GB** | 6.3e-13 | not submittable |
| L7, MIN 8 | — | OOM: one pooled (10, 7, 16807, 16, 16) buffer is 1.12 GiB | — | not submittable |

**1. `STRASSEN_MIN` 32 → 8 is the trim.** With MIN = 32 the shared-basis families (rank 384 / 224) stop at Strassen level 3 / 2, because their block sides fall below 32. With MIN = 8 the 384-families reach level 5 (384/32 = 12 ≥ 8) and the 224-families level 4 (224/16 = 14 ≥ 8). The 1024-side families already ran all five levels either way, so memory barely moves (+0.4 GB). Net −3.9% of C locally, −3.7% graded, at unchanged raw MSE.

**2. `STRASSEN_FIRST` 4 → 5 is the second, smaller one.** The first `predict()` of each grader worker pays the pooled-buffer page touch, which is metered; letting that call run at full depth anyway is worth another −0.4% (0.2442 → 0.2432), graded 5.220e-9 → 5.195e-9.

**3. Level 6 is blocked by the 8 GB address-space cap, not by FLOPs.** Each level adds pooled `(b, 7, 7^(k-1), n/2^k, n/2^k)` buffers, about 1.75× the previous level's; at level 6 the commit peaks at 8.13 GB on the *second* `predict()` of the process, above the grader's `RLIMIT_AS`, so the worker would die. It would only have bought −4.7% anyway, because the 18 block additions per level start to cost as much as the matmul saving: on a 1024-side family the level-6 additions (18·7⁵·16² ≈ 77 MFLOP) eat over half of that level's matmul saving (⅛·(7/8)⁵·2n³ ≈ 138 MFLOP), and at level 7 (135 vs 120 MFLOP) they cancel it outright. On a 31 GB machine without the cap, L6 scores identically to L5 on 10 mini MLPs (raw 2.302e-8 vs 2.303e-8) at util 0.2416, but wall time goes 90 → 150 s per MLP on four cores, which would also flirt with the 120 s limit.

So on this representation the block recursion is finished at MIN = 8 / L5 / FIRST = 5 (≈ 0.243). The change is two constants in `estimator_v29.py`; I am happy to paste the diff and the measurement script if anyone wants them.

## One accuracy item that is not engineering

The networks have no biases, so they are positively homogeneous, and the Gaussian input factorises exactly as X = (R/√n)·Y with Y uniform on the sphere of radius √n and R ~ χ_n independent of it. Therefore

    E[h_l(X)] = a₁ · E[h_l(Y)],   a₁ = √(2/n)·Γ((n+1)/2)/Γ(n/2) = 0.99975589 at n = 1024,

exactly, at every layer. The spherical input has C = I and κ₃ = 0 just like the Gaussian one, but a non-zero fourth cumulant, κ₄ = c·Sym₃(I⊗I) with c = −2/(n+2). In a chain that already carries a κ₄ channel this is a layer-0 injection (a previous-layer core g = −6/(n+2), i.e. dG = g·var and a (3,1) slice g·C_off) plus a final rescaling by a₁.

Cost: +0.02% of budget. Graded: raw 2.132e-8 → 2.108e-8 (−1.1%), score 5.195e-9 → **5.137e-9** (#331787).

Credit where it is due: the idea is from **barnobarno666**'s public Phase-8 report, where it was worth −7.9% on a plainer K=3 chain. On V29 most of it is already absorbed by the κ₄ regeneration and the mean rider, but the remainder is free, and the identity itself is exact rather than fitted.

## A κ₄ measurement, since the thread is already on F68/F105

Before F105 was posted I ran a Monte-Carlo κ₄ teacher (single-pass raw moments → central cumulant slices; 6×10⁷ samples per layer; two independent 1024×16 networks, one of them a public mini MLP) and fitted the λ law against the **raw** cumulant slices. Two results, both consistent with @504aldo's correction that the chain's G is the r = 1 core and not a raw slice:

* On the raw (3,1) slice, λ·C_off gives R² 0.35–0.60 (rising with depth); adding the κ₄ diagonal terms gives 0.55–0.71; adding the chain's own D21 gives 0.74–0.80. Same pattern and depth trend on both networks.
* The true weight on that slice is **per row**, κ₄(i)/var(i) — and the per-layer scalar λ_l equals the **mean** of that ratio to within 2% (6.8e-3 vs 6.9e-3 at layer 15). So the fitted constant has a derivation.
* But implementing the per-row weight in the chain makes the final mean monotonically **worse**: raw 2.2911 → 2.2923 → 2.2933 → 2.2958 → 2.2988e-8 as the blend goes 0 → 0.25 → 0.5 → 1.0 → 1.5 (10 mini MLPs, baked truth). Read together with F105, the lesson is that a fidelity argument about one channel proves nothing here: the chain propagates the core, the scalar already fits it, and "improving" the model against a different object degrades the result. The 2.23e-8 / 2.04e-8 / 4.20e-8 ladder @504aldo posted is the honest ceiling for that channel.

Two further negative results from the same work, in case they save someone a week: replacing the old-source tier's shared basis by a pooled Tucker/TT core needs multilinear rank ≈ 250 for 2% query error and is net negative at every rank we measured; and the final-layer residual of the chain is **not** learnable from cheap per-neuron features — cross-validated over all 100 public networks (102,400 output neurons), ridge on 19 features recovers 0.3%, a quadratic expansion 0.7%, gradient boosting 0.5%, with every single-feature correlation ≤ 0.04.

Disclosure: the experiments and this write-up were produced with LLM assistance (Claude), directed by me; every number above is from `whest run`, the grader, or a stated Monte-Carlo run.
