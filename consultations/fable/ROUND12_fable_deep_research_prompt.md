# Round 12 — Fable 5 deep research: a different represented state (Round 11); the young tier (Round 10) is closed

*Prepared 24 Sep 2026. Updated after every external research round. Do not launch without the user's go-ahead
(token budget).*

You are working inside the repository `D:\Machine Learning 1 Year Challenge\Kaggle\ARC White-Box Estimation Challenge 2026`.
You can read files and run Python. **Every number you report must come from a script you ran. Do not report expected
or projected results as measured.** Put scripts in `experiments/probes/fable12_*.py` and results in
`results/summaries/fable12_*.json`. Do not edit anything else, and do not commit, push or start Kaggle runs. Final report:
`consultations/fable/ROUND12_fable_report.md`, at most ~4 pages.

Budget discipline: at most three falsifier scripts per thread; order B3 (κ4 / accuracy) first, then B1, then B2. If a track dies at its first gate, say so and stop that track.

## Read first (in this order, skim what you already know)

1. `docs/moonshot/Moonshot_Blueprint.html` — the design specification: target, constraints C1–C13, dead list.
2. `consultations/prompts/ROUND11_prompt_change_what_is_represented.md` (Round 10 is closed; see Track A below).
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
  source-free trees; diagonal-only κ3/κ4 transport; slice-state recursions (Copilot RSC: carry Y21 + response matrix
  R' = (W∘W)RWᵀ + αY21 + βC — 36% D21 error even in-sample from the true state); low-rank(+diag) covariance (r = 64 still 12–33% off-diagonal);
  rank-1 κ3; aggregate rank-k CP of the old κ3 (unfolding bound: rank ≫ 250); deterministic
  moment-matched particles (2048 points: variance 3%, D21 74% at L15 — no better than random); sampling / control variates; adjoint low-rank paths; learned corrections; join rebuilds; A = P·a_b tricks,
  P_old = P_new·M, batching (same FLOPs); any contraction-order change of the young tier (193 products is the floor);
  lower-precision arithmetic (float16 billed like float32).

## LEADERBOARD FACT THAT RE-ORDERS THIS ROUND (public LB, 24 Sep 2026)

The frontier's edge over us is ACCURACY as much as cost. Our #332110: raw final MSE 2.11e-8 at utilisation 0.244.
At least 12 teams are at 1.56–1.86e-8 raw MSE (15–26% better), most at utilisation 0.13–0.20:
J2W 1.61e-8 @ 0.127; suliman_tadros 1.63e-8 @ 0.138; marius_binner 1.56e-8 @ 0.147; Luna 1.67e-8 @ 0.155 (only 7
entries); reds 1.73e-8 @ 0.162; ben3 1.68e-8 @ 0.184; MeatProxy 1.79e-8 @ 0.182; thegamers 1.79e-8 @ 0.202 (already by
10 Sep); emanuel_ruzak 1.82e-8 @ 0.198 (5 entries); AndreasHad04 1.86e-8 @ 0.233. The ~20 teams at 2.1–2.2e-8 @ 0.23–0.25
are the public 504aldo chain (= us). So a 15–25% ACCURACY mechanism exists that many independent teams found, at no
extra cost. Our own oracles say where it can be: the chain's closures are exact given true inputs; the κ4-diagonal
oracle alone is −30% final MSE, and the chain's κ4 diagonal is its weakest channel (32–34% wrong at depth, slope ~0.78).
**Therefore B3 (κ4) is now the FIRST thread**, with the question widened: what cheap (O(n²) or a few n³ per layer) change
to the κ4 / variance channels removes ≥ 15% of the final MSE? Gate it end-to-end if the channel gate passes (the chain
harness `experiments/chain/run_variant_teacher.py` measures final MSE against the teacher on MLP #0).

## Track A (Round 10) is CLOSED — do not spend budget on it

The young tier (ages 1–4) is an exact dense computation of 193 square products per network (39 P transports + 54 A
transports/applications + 100 D21 contractions; 98.4 u core, 105 u metered). An exact contraction-order search (Sonnet 5
subagent, opt_einsum + hand-built mixed/hierarchical orders) and four external answers agree: every order — forward,
reverse, interval maps, on-demand A, batching — costs the same 193 products; the Hadamard hyperedge forces
materialisation before each n³ contraction. Also measured: A = P·R_b with a FIXED full n×n birth matrix R_b (A and P share
the same left transport to 1e-14; R_b is not diagonal), which saves nothing because forming A entry-wise is itself a
product. The meter bills float16 = float32 = 1.0 and float64 = 2.0, so there is no precision discount (numerical packing
fails the cost gate). **A cheaper young tier must change the representation, not the order.** Approximating the young legs
is dead too (they are near full rank; their Hadamard products are full rank).

## Track B (Round 11): represent something different at ≤ ~15 products per layer

Practical framing (use it to prioritise): the leaders' bill (~130 u) is about our exact young tier alone (~100 u) plus
cheap work. So the most concrete target architecture is **exact young tier + an old-content surrogate costing ≤ ~20 u in
total** that replaces the join + old tier + thin legs (~100 u). The surrogate must supply what sources aged ≥ 5 contribute
(keeping only ages ≤ 4 leaves 65% of D21 at L15; the old content is coherent — one age mode ~88% — but not low-rank).

Carrying the κ3 tensor, in any factorisation, costs what the chain costs. What must be carried is whatever the next means
need: variance, d3, κ4 diagonal and the covariance feeding them, **plus the hidden all-distinct memory**, since visible
channels provably do not close (Round 11, Astra: an all-distinct perturbation invisible in the current slice changes
the next slice). Candidate threads:

- **B1. The memory residual, one step.** Take the true post-ReLU state at L11/L14 and the exact next D21 and d3. Measure what
  the missing 22–29% (d3) / 36–40% (D21) actually is: its rank, its alignment with the previous layer's all-distinct
  residual transported by W, and whether a single extra n × n state matrix per layer, updated with ≤ 4 products, predicts it
  to ≤ 2%. If no such matrix exists, say so with the numbers. The prior "slice transport" (K3g) is dead, so show what differs.
  Then the architecture question: on the captures (L7/11/14/15, every source dense), can the summed D21 of the sources aged
  ≥ 5 be produced by a carried object updated with ≤ ~1.3 square products per layer (≈ 20 u over the network) to ≤ 2% of
  the total D21? This is the old-content surrogate of the framing above.
- **B2. A nonlinear, trajectory-specific state.** The Sym³ wall applies only to linear states closed under every M. Our M
  are fixed and known. Propose one state that exploits that (e.g. a per-network precomputed operator applied to a small
  carried object) and gate it on the captures.
- **B3. κ4 pattern (accuracy side, cheap).** The chain's κ4 diagonal is 32–34% wrong at depth with slope 0.75–0.79 and
  corr ~0.8; the κ4 oracle alone is worth −30% MSE. Find an O(n²)-per-layer correction of the pattern (not the scale:
  the scale constants already sit at their score optimum) and gate it on the teacher (K4d, K22, K31 are stored).

Gate for B1/B2: one-step, teacher-forced where possible, and free-running from layer 1 if the state is not
reconstructible from teacher arrays; every channel within the chain's own error (numbers above) at L11/14/15/16.
Gate for B3: κ4 diagonal error at L14–16 reduced from ~33% to ≤ 20% with an O(n²) rule, validated on the layers it was
not fitted on. Known (24 Sep, r11_verify.py V5): the next κ4 diagonal is ~90% the (2,2) pair term from the post K22
off-diagonal; the exact post (4)+(3,1)+(2,2) terms still leave 14/22/33/43/46% (L4/8/12/15/16) — the (2,1,1) +
(1,1,1,1) remainder — and the chain's (2,2) model wk4m is 15–28% off (pattern corr 0.65–0.73). So B3 must model the
(2,1,1)/(1,1,1,1) post-ReLU structure or improve the (2,2) pattern; a per-neuron scalar rule cannot work.

## Report format

For every thread: one paragraph of mechanism, the exact product count, the script name, the measured numbers, and a
verdict (alive / dead / needs X). End with the single most promising next experiment and what it would cost.
