Measured on your V29: where the cheap 4% trim is, and why levels 6-7 are not shippable

Thanks for releasing the chain under MIT — it got me on the board (#331681, V29 verbatim: 5.418e-9 adjusted, raw 2.133e-8, util 0.254, 0 failed). Since you asked what the 7-22% trims might be, here is what I measured on the Strassen side, with numbers. Everything below is the same arithmetic (prediction drift MSE ≤ 6e-13 vs your V29, i.e. float32 noise); only the block recursion changes.

Variant → util (steady) / peak commit (local) / drift MSE vs V29 / graded
• V29 as shipped (L5, MIN 32) → 0.2540 / 5.56 GB / 0 / 5.418e-9 (#331681)
• L5, STRASSEN_MIN = 8 → 0.2442 (graded mean 0.245) / 5.97 GB / 4.9e-13 / 5.220e-9 (#331686), raw 2.132e-8, 0 failed
• L6, MIN 16 → 0.2421 / 8.13 GB / 6.3e-13 / not submitted
• L7, MIN 8 → OOM: a single (10, 7, 16807, 16, 16) pool buffer is 1.12 GiB / not submitted

Two mechanisms:

1. STRASSEN_MIN 32 → 8 is the trim. With MIN = 32 the shared-basis families (rank 384 / 224) stop at level 3 / 2 because their block sides fall below 32; with MIN = 8 the 384 families reach level 5 (384/32 = 12 ≥ 8) and the 224 families level 4 (224/16 = 14 ≥ 8). The 1024-side families already run all 5 levels either way, so memory barely moves (+0.4 GB). Net −3.9% of C locally, −3.7% graded (5.418e-9 → 5.220e-9), for free.

2. Level 6 is blocked by the 8 GB cap, not by FLOPs. Each level adds pooled (b, 7, 7^(k-1), n/2^k, n/2^k) buffers, ~1.75× the previous level's; at level 6 the commit peaks at 8.13 GB on the second predict(), above the grader's RLIMIT_AS, so the worker would die. It would only have bought −4.7% anyway, because the 18 block additions per level start to cost as much as the matmul saving: on a 1024-side family the level-6 additions (18·7^5·16² ≈ 77 MFLOP) eat over half of that level's matmul saving (⅛·(7/8)^5·2n³ ≈ 138 MFLOP), and at level 7 (135 vs 120 MFLOP) they cancel it. On Kaggle (31 GB, no cap) L6 grades identically to L5 on 10 mini MLPs (raw 2.302e-8 vs 2.303e-8) at util 0.2416, but wall time goes 90 → 150 s per MLP on 4 cores, so it would also flirt with the 120 s cap.

So on this representation Strassen is done at MIN = 8 / L5 (~0.244); the remaining gap to the 0.236 band is not in the block recursion. Happy to share the one-line patch (it is literally the STRASSEN_MIN default) and the measurement script.

Disclosure: measurements and this note were produced with LLM assistance (Claude) directed by me; all numbers are from whest run / the grader as stated.
