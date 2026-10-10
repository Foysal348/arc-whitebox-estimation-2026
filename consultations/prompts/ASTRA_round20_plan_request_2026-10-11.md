# Request: a complete, executable plan for the Round 20 learned pair-memory test

We want to run your Round 20 proposal ("A learned pair-memory estimator", 11 Oct 2026) as a full bounded test, exactly as
you specified it:
1. the resource falsifier;
2. the trajectory-distillation falsifier, with free-running rollouts and held-out networks;
3. only if both pass, the scale-up.

Please write the plan so that we (an AI coding assistant plus two Kaggle accounts with GPU) can execute it step by step
without further design decisions. Aim for the version with the best chance of passing; size it to the resources below. Below are our real constraints and assets. Where the plan depends on something we have not
told you, state the assumption explicitly.

## Time and compute
- **Time.** Submissions close 18 Oct 2026, 05:59 Bangladesh time (17 Oct 23:59 UTC). Today is 11 Oct. We give this test
  4 full days, 11–14 Oct, so that a passing student can be exported, validated and submitted on 15–17 Oct.
- **Compute.**
  - Kaggle GPU: 50+ GPU-hours across our two accounts, and about 30 more can be added if needed. Plan the training on
    GPU (P100 / T4 class).
  - Kaggle CPU kernels (4 cores, 30 GB RAM, 12 h each, up to 5 concurrent per account, 2 accounts) for teacher-label
    generation and evaluation.
  - PyTorch is available on Kaggle.
  - The laptop is for light work only.
- **Grader limits (deployment).**
  - 120 s wall per network;
  - 0.4 s residual Python time per network, which scales with the number of flopscope ops (keep about ≤ 18–20k ops per
    predict());
  - 8 GB address space;
  - flopscope-metered NumPy-like ops only (no torch at inference);
  - float32 bills the same as float16 (rate 1.0); float64 bills 2×;
  - the smoke test rejects some batched 3-D in-place broadcasts, so 2-D ops are safest.
- **Scoring.** Final MSE × max(0.1, FLOPs/2^41); 1 u = 2^31 FLOPs; floor 102.4 u. Our current chain: raw MSE about
  1.97e-8 (grader) at 194 u.

## Assets
- **Teacher chain** (frozen production estimator): `submissions/v56As_fs/estimator.py`.
  - About 2,000 lines, flopscope API; a predict() on one network takes about 2–3 min on a Kaggle CPU and about 5.5 GB.
  - A research copy of the plain chain, `experiments/variants/r20_joint_oracle.py`, has hooks where we can log, per
    layer: pre-activation mean, the full covariance C (n×n), the D21 slice incl. diagonal, d3, the κ₄ diagonal g4row,
    and the closure outputs.
  - The variant `experiments/variants/r20_feed_oracle.py` already dumps D21/D3 per layer via an environment variable.
- **Networks.**
  - The public "mini" split: 100 networks of width 1024 and depth 16, with 1e9-sample Monte-Carlo truth of the
    final-layer means.
  - We can also generate any number of new He-initialised networks ourselves (no truth unless we run sampling).
  - Teacher statistics (6e7-sample Monte Carlo) exist for public network #0 only: mean, C, Y21, K4d at all 16 layers.
- **Reusable code.** The closure functions (Edgeworth univariate mean closure; bivariate covariance closure tables
  TERM_SPECS with c_off, d21, d21T, d3, g4, wk4m, wk431 terms; the regenerated κ₄ core λ·C_off) live inside the
  estimator. We can factor them out into a NumPy/torch module.

## What we need from you
1. **Exact student specification.**
   - Every formula of the forward pass, including the Gaussian ReLU (2,1) third-cumulant birth slice B from (a, S), and
     how the student calls our existing closure (which inputs it replaces: D21, d3, g4row; what to do with wk4m, wk431
     and the κ₄ off-diagonal core).
   - Normalisation floors.
   - Initialisation of the 364 parameters.
   - The rule for the last layer.
2. **Resource falsifier.**
   - The flopscope implementation sketch and the expected op count per layer.
   - Which products to batch or Strassen.
   - The pass/stop thresholds (you proposed ≤ 116 u).
3. **Teacher data plan.**
   - How many training and held-out networks; which per-layer tensors to log (we must keep storage feasible: one n×n
     float32 is 4 MB; 16 layers × several matrices × N networks).
   - Whether to log full matrices or sampled entries.
   - Whether to use public-mini networks (with truth) or self-generated ones.
4. **Training procedure (GPU).**
   - Loss terms and weights (final mean, per-layer channels, normalisation).
   - Optimiser and learning rate.
   - BPTT through 16 layers of n = 1024 dense products on GPU: memory per rollout, checkpointing, minibatching over
     networks, and the expected wall time per epoch (each rollout is about 128 dense 1024³ products forward).
   - How to split the 50+ GPU-hours across runs and seeds.
   - Whether we should train at a smaller width first (e.g. n = 256) and transfer, given your parameters are shared
     and width-independent.
5. **Free-running evaluation protocol.**
   - The pre-registered gates you proposed: student-to-teacher final-mean MSE ≤ 1e-9 on held-out networks; D21/d3 ≤ 2 %
     at layers 9–16; no failures.
   - Seeds, how many runs.
   - What counts as a stop.
6. **Ablations, if cheap:** without H2; without the odd readout O; without g.
7. **Day-by-day schedule** for 11–14 Oct, with the go/no-go point for export on 15 Oct.
8. **Inference export** (only if everything passes): how the trained heads become flopscope ops within the op budget,
   and how to validate on our 100-network harness before any grader submission.

Mark every number as derived, estimated or speculative. Do not report measured results you did not run.
