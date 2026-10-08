# Round 17 verdicts (8 Oct 2026)

Both runs below used Kaggle GPU (account foysalemonshanto) on public MLP #0, scored with the project metric:
- decisive neurons |μ/σ| < 1.5 from the 6e7-sample teacher;
- pattern = corr(κ4 / v_t², K4d_t / v_t²).

## 1. Claude's hub-pair Q1 carrier: KILL (pre-registered gate)

Design: the Q1 C·T curvature birth at each ReLU, carried linearly by U for up to 6 ages and read out as the pre-activation diagonal. The κ3 contraction is computed exactly from samples, so this is the generous version.

Script: `experiments/probes/r17_claude/r17_q1_hubpair_falsifier.py`, the author's version with our metric added. Results: `results/summaries/r17_q1_hubpair.json`, N = 2e6. MC sanity: the sampled κ4 vs the teacher is 0.936 / 0.960 / 0.964.

| config | L12 | L15 | L16 | level ratio |
|---|---|---|---|---|
| ceiling Q1 (H = 6, r = n) | 0.442 | 0.432 | 0.431 | 0.84-0.92 |
| Q1 + Gaussian path | 0.401 | 0.386 | 0.428 | 1.8-1.95 |
| Q1, H = 4 | 0.473 | 0.447 | 0.454 | 0.62-0.69 |
| Q1, H = 2 | 0.420 | 0.392 | 0.439 | 0.32-0.41 |
| Q1, H = 6, r = 64 | 0.255 | 0.203 | 0.361 | ~0 |
| Q1, H = 4, r = 32 | 0.096 | 0.126 | 0.255 | ~0 |

The author's own metric (raw κ4, top-128 neurons by φ|κ4|/σ³) gives 0.80-0.87. It is not comparable: the raw diagonal contains v², which the chain already gets right.

## 2. Family ceiling: which Hermite order of the ReLU must the births carry? (our own test)

Method:
- Monte Carlo with the ReLU replaced by its Hermite expansion around the teacher's per-layer μ, σ.
- "gauss_h6": the state 6 layers before the target is a Gaussian restart N(μ, C) of the teacher, so all older non-Gaussian history is erased; this is the best a 6-age carrier could do.
- "true_hX": the true state X layers back.

Script: `experiments/probes/r17_hermite_family_ceiling.py`. Results: `results/summaries/r17_hermite_family_ceiling_A.json`, N = 4e6.

| mode | L12 | L15 | L16 |
|---|---|---|---|
| true ReLU (sanity) | 0.967 | 0.973 | 0.980 |
| Gaussian restart 6 back, true ReLU | 0.859 | 0.820 | 0.818 |
| Gaussian restart 6 back, Hermite K = 2 | 0.548 | 0.449 | 0.530 |
| **Gaussian restart 6 back, Hermite K = 3** | **0.848** | **0.799** | **0.817** |
| true state 1 back, K = 2 (the quoted QUAD) | 0.850 | 0.874 | 0.881 |
| true state 3 back, K = 2 | 0.687 | 0.688 | 0.738 |
| true state 6 back, K = 2 | 0.527 | 0.477 | 0.578 |
| true state 6 back, K = 3 | 0.915 | 0.927 | 0.949 |

Two notes:
- Compounding K = 2 over all 16 layers diverges (level ×1e12); K = 2 over h layers degrades with h.
- The memory-horizon reference (Gaussian restart 6 back, true ReLU) is 0.82-0.86 here, slightly above the 0.77 quoted earlier.

## Verdict

- **The second-order (Section B) birth family is dead at the ceiling.** Even with exact cumulants and unlimited sectors it reaches only 0.45-0.55 over a 6-layer horizon. This is consistent with Q1 at 0.43-0.47.
- **Third-order births pass with a wide margin.** Adding the CUBIC Hermite term c3 He3, with c3 = φ(α) He1(−α)/3! = −α φ(α)/6, reaches 0.80-0.85. That equals the true ReLU at the same horizon.
- The next design target is a factorised carrier whose births come from the cubic response of each ReLU.
- Set B (running) asks two further questions:
  - is the quadratic term needed at all (orders 0, 1, 3)?
  - how short can the horizon be (h = 4, 3, 2)?
