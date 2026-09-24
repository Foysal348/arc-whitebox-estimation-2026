# Round 12 — Fable 5 deep research: the young tier (Round 10) and a different represented state (Round 11)

*Prepared 24 Sep 2026. Updated after every external research round. Do not launch without the user's go-ahead
(token budget).*

You are working inside the repository `D:\Machine Learning 1 Year Challenge\Kaggle\ARC White-Box Estimation Challenge 2026`.
You can read files and run Python. **Every number you report must come from a script you ran. Do not report expected
or projected results as measured.** Put scripts in `experiments/probes/fable12_*.py` and results in
`results/summaries/fable12_*.json`. Do not edit anything else, and do not commit, push or start Kaggle runs. Final report:
`consultations/fable/ROUND12_fable_report.md`, at most ~4 pages.

Budget discipline: at most three falsifier scripts per track. If a track dies at its first gate, say so and stop that track.

## Read first (in this order, skim what you already know)

1. `docs/moonshot/Moonshot_Blueprint.html` — the design specification: target, constraints C1–C13, dead list.
2. `consultations/prompts/ROUND10_prompt_young_tier_cost.md` and `ROUND11_prompt_change_what_is_represented.md`.
3. The estimator: `submissions/v40fb/estimator.py` (V29/V31 factorised K3 chain, 504aldo MIT, + FB_SCALE 1.25, BETA 1.2).
4. Existing gate scripts, as templates for loading data: `experiments/probes/r9_dhmp_gate.py`,
   `r9_horizon_oracle.py`, `r11_chain_channel_ref.py`, `r11_diag_closure_gate.py`, `m1_matmul_ledger.py`.

## Data you have (public MLP #0, n = 1024, depth 16)

- Teacher (6e7-sample Monte Carlo), pre-activation at every layer 1–16:
  `data/kaggle_out/kaggle_out_k4pub/k4teach/layer{l}_seed0.npz` → `mean, C, Y21 = κ3(z_i,z_i,z_j), K4d, K22, K31, relu_mean`;
  weights `weights_seed0.npz["W"]` (z_{l+1} = h_l @ W[l]).
- Teacher post-ReLU at layers 3/7/11/14/15: `data/kaggle_out/kaggle_out_k4post/k4post/layer{l}_post_seed0.npz`.
- Dense source captures of the chain at L7/11/14/15: `data/dumps/r7_cap_L{l}.npz` (every live source's legs
  `A_st, P_st, Z_st, L_st, Zf_st, R1T_st, R2T_st` + birth vectors; index s is the same source across files; age = k − s).
  `blocks_of_source` in `r9_horizon_oracle.py` rebuilds a source's D21 contribution exactly.
- Free-running chain channels at every layer: `data/dumps/errbudget_v29plain.npz`.
- Cost meter: `m1_matmul_ledger.py` attributes the metered FLOPs to estimator lines.

## Settled facts (measured; do not re-derive, do not re-propose the dead)

- **Target:** our accuracy (raw 2.1e-8) at ≤ ~115 u instead of 249 u reaches the top 3; ≤ 102 u ties #1.
  1 u = 2³¹ FLOPs ≈ one dense 1024² product; ≈ 0.51 u with Strassen L5. So ≤ 115 u ≈ ≤ 15 square products per layer in total.
- **Ledger (249 u):** young transport 54, young D21 contraction 50, join 41–46, old tier 41, thin legs 15, Strassen adds 12,
  covariance + closure only 8.
- **One-for-one law:** every knob inside the chain trades ~1% cost for ~1% accuracy. Halving is impossible by tuning.
- **Error map:** the closures are exact given true inputs; channel oracles give variance −47%, κ4 diagonal −30%, κ3 −14%,
  all three −94%. The chain's own channel errors vs the teacher at L15: variance 0.14%, C off-diagonal 0.73%, d3 4.6%,
  κ4 diagonal 32% (slope 0.78, corr 0.83), D21 off-diagonal 6.6%.
- **κ3 memory:** long (keeping ages ≤ 10 still leaves 16% of D21 at L15), coherent (one age mode ≈ 88%), not low-rank
  across ages (6 modes leave 8.5%); the total κ3 is transported exactly by one linear map, so the problem is spatial.
- **Walls:** no linear representation of the total κ3 closed under dense transport is smaller than C(n+2,3) (Sym³
  irreducible). The lazy cubic circuit (reverse mode) costs ≈ 245 u, the same as the source chain: the two are dual
  contraction orders. Hadamard products of factored legs are full rank (Khatri–Rao).
- **Memory is needed one step ahead:** from the TRUE post-ReLU state, a source-free dressed Gaussian tree + the exact
  repeated-index slices still miss 22–29% of the next d3 and 36–40% of the next D21. Diagonal-only transport
  (Hadamard powers) misses 90–99% of the next d3: **d3 is fed almost entirely by off-diagonal κ3**.
- **Dead:** Tucker/TT/CP/ISDF/vector-metric/shared-basis compression; short horizons; age modes; shift-stable carrier;
  source-free trees; diagonal-only κ3/κ4 transport; low-rank(+diag) covariance (r = 64 still 12–33% off-diagonal);
  rank-1 κ3; sampling / control variates; adjoint low-rank paths; learned corrections; join rebuilds; A = P·a_b tricks,
  P_old = P_new·M, batching (same FLOPs; checked 24 Sep).

## Track A (Round 10): the young tier, ~104 u → ≤ 55 u at ≤ 2% D21 error

The young tier is four dense products per source-layer (two transports, two contractions) for ages 1–4. My own
algebra found no saving inside the sum-over-sources form. Candidate threads to examine (take or kill each with a number):

- **A1. Mixed contraction order.** Source-forward and reverse-mode are dual orders of one tensor network. Is there a mixed
  or hierarchical order over the (birth b, target layer l) pairs, e.g. forward for ages 1–2 and reverse sweeps shared by
  several targets, whose product count is below both? Count products exactly for the young tier at our shapes.
- **A2. Closed-form ages 1–2.** The age-1 legs are P = W-like and A = P·a_b; can the age-1 (and age-2) contributions
  to D21 be written as a few products of W, C and diag vectors already computed for the covariance step, so the legs
  are never transported?
- **A3. Any Round 10 answer** that arrives before launch (this section will be updated with them).

Gate: on the captures, the reorganised young-tier D21 must match `blocks_of_source` summed over ages 1–4 to ≤ 2%
relative error, with a product count ≤ 55 u over the network (count at real shapes; Strassen only for square products).

## Track B (Round 11): represent something different at ≤ ~15 products per layer

Carrying the κ3 tensor, in any factorisation, costs what the chain costs. What must be carried is whatever the next means
need: variance, d3, κ4 diagonal and the covariance feeding them, **plus the hidden all-distinct memory**, since visible
channels provably do not close (Round 11, Astra: an all-distinct perturbation invisible in the current slice changes
the next slice). Candidate threads:

- **B1. The memory residual, one step.** Take the true post-ReLU state at L11/L14 and the exact next D21 and d3. Measure what
  the missing 22–29% (d3) / 36–40% (D21) actually is: its rank, its alignment with the previous layer's all-distinct
  residual transported by W, and whether a single extra n × n state matrix per layer, updated with ≤ 4 products, predicts it
  to ≤ 2%. If no such matrix exists, say so with the numbers. The prior "slice transport" (K3g) is dead, so show what differs.
- **B2. A nonlinear, trajectory-specific state.** The Sym³ wall applies only to linear states closed under every M. Our M
  are fixed and known. Propose one state that exploits that (e.g. a per-network precomputed operator applied to a small
  carried object) and gate it on the captures.
- **B3. κ4 pattern (accuracy side, cheap).** The chain's κ4 diagonal is 32–34% wrong at depth with slope 0.75–0.79 and
  corr ~0.8; the κ4 oracle alone is worth −30% MSE. Find an O(n²)-per-layer correction of the pattern (not the scale:
  the scale constants already sit at their score optimum) and gate it on the teacher (K4d, K22, K31 are stored).

Gate for B1/B2: one-step, teacher-forced where possible, and free-running from layer 1 if the state is not
reconstructible from teacher arrays; every channel within the chain's own error (numbers above) at L11/14/15/16.
Gate for B3: κ4 diagonal error at L14–16 reduced from ~33% to ≤ 20% with an O(n²) rule, validated on the layers it was
not fitted on.

## Report format

For every thread: one paragraph of mechanism, the exact product count, the script name, the measured numbers, and a
verdict (alive / dead / needs X). End with the single most promising next experiment and what it would cost.
