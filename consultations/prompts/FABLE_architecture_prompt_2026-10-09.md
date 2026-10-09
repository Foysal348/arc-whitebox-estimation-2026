# Fable brief (9 Oct 2026): design and ceiling-test ONE new architecture for the κ₃ + κ₄ input channels

You are working inside our project repository:
`D:\Machine Learning 1 Year Challenge\Kaggle\ARC White-Box Estimation Challenge 2026`.

Read this brief fully before acting. Every number below is measured by us; the source file is given in brackets.

## 0. Task and scoring

**Problem.** Estimate the final-layer post-ReLU means of a bias-free ReLU MLP:
- width n = 1024, depth L = 16;
- He weights, input x ~ N(0, I);
- row convention z_{l+1} = h_l W[l].

**Score.** Final-layer MSE × max(0.1, FLOPs / 2⁴¹). One u = 2³¹ FLOPs ≈ one 1024³ product.

**Hard grader limits per network:**
- 120 s wall time;
- 0.4 s of Python time outside metered operations, which is proportional to the op count: keep ≤ ~18,000 metered ops;
- 8 GB address space;
- a smoke test that rejected batched 3-D broadcast operations computed in place (`multiply(k×n×n slice, k×1×n broadcast, out=…)`); use 2-D ops and mat-vecs.

**Time.** Submissions close **18 Oct 2026 05:59 Bangladesh time**. You have about 7 days.

## 1. Where we stand

| | score | raw MSE | util |
|---|---|---|---|
| Our best, #334870 (`submissions/v56As_fs/estimator.py`) | 3.652e-9 | 1.950e-8 | ≈ 0.187 |
| Best nomination-safe, #334869 (`submissions/v56s_fs`) | 3.672e-9 | — | ~18k ops |
| Live leaderboard: we are #41 | | | |
| Top 25 teams | — | 1.4–1.76e-8 | 0.10–0.16 |
| #10 | ≈ 2.1e-9 | — | — |
| marius_binner | — | 1.14e-8 | — |

**Our estimator** is the public MIT factorised third-cumulant chain of 504aldo (V29, see `docs/build_notes.py`, `writeup/ALGORITHM_TRACK_PAPER.md` §1):
- mean, dense covariance C;
- κ₃ as a sum over per-ReLU sources with dense young legs (ages ≤ 5) and a rank-352 shared basis for old sources;
- a memoryless regenerated κ₄ core;
- univariate Edgeworth mean closure and bivariate covariance closure.

**Our additions:**
- V31, spherical-input identity;
- cost engineering (dead-neuron compaction, Strassen, op budget);
- V55, a linear κ₄-diagonal pattern corrector from per-layer features;
- V56, the Gaussian-tree cubic "star" 24 Σ_a c_a P_ai X_ai³ read from the young legs as an extra feature;
- the synthetic-network refit.

## 2. The decisive new measurements (9 Oct)

### 2.1 Joint input oracle on the plain chain

Teacher inputs come from the 6·10⁷-sample Monte-Carlo teacher [results/summaries/r20_joint_oracle.json]. The chain propagates everything else, including C.

| injected exactly | MLP #0 final MSE (ratio) | synthetic net (ratio) |
|---|---|---|
| nothing | 2.273e-8 (1.000) | 1.920e-8 (1.000) |
| κ₃ (2,1) slice + κ₃ diagonal | 1.380e-8 (0.607) | 1.376e-8 (0.716) |
| κ₄ diagonal (mean closure) | 1.560e-8 (0.686) | 1.700e-8 (0.885) |
| **κ₃ slice + κ₃ diag + κ₄ diag** | **6.19e-9 (0.272)** | **1.092e-8 (0.569)** |
| κ₃ slice + κ₄ diag | 9.95e-9 (0.438) | 1.429e-8 (0.744) |
| κ₃ diag + κ₄ diag | 1.204e-8 (0.530) | 1.394e-8 (0.726) |
| κ₃ slice + diag + κ₄ PATTERN (chain level) | 6.69e-9 (0.294) | 1.163e-8 (0.606) |
| as row 4, but only from layer 8 on | 7.46e-9 (0.328) | 1.102e-8 (0.574) |

Notes on the synthetic net:
- Its "truth" is itself a 6·10⁷-sample estimate, so its noise floor is higher.
- With true variance, κ₃ and κ₄ fed directly into the mean closure, MLP #0 reaches 1.3e-9, the noise floor [results/summaries/meanfix_oracle.json].

**Reading.**
- κ₃ and κ₄ are synergistic: 0.607 × 0.686 = 0.417 if independent, but 0.272 measured.
- Together they hold about 73 % of MLP #0's error.
- Most of the value is from layer 8 on.

### 2.2 One-step channel errors of truncated ReLU responses

Common random numbers, N = 2·10⁶ [results/summaries/r20_onestep_channels.json].

**Method:**
- Take exact samples of z_l.
- Replace the ReLU at layer l by its Hermite expansion of order K around the teacher's μ, σ.
- Compare z_{l+1} statistics with the exact ReLU step on the same samples.

At L12 → 13:

| order | C off-diagonal RMS / entry | variance rel. | κ₃ (2,1) slice rel. | κ₄ pattern corr |
|---|---|---|---|---|
| K = 2 | 1.2e-4 | 1.3 % | 12 % | 0.88 |
| K = 3 | 9.2e-5 | 0.8 % | 2.7 % | 0.99 |
| K = 4 | 5.7e-5 | 0.5 % | 1.6 % | 0.998 |
| **chain** | fresh ≈ 4.2e-5 | 0.09 % (accumulated) | 4.6–6.6 % (accumulated) | **0.29–0.34** |

**Reading.**
- The chain's covariance and variance machinery is already better than a 4th-order Hermite response; do not try to beat it there.
- Its κ₃ slice is comparable to cubic order per step, but accumulates to 4.6–6.6 %.
- Its κ₄ diagonal pattern is the weakest channel by far.

### 2.3 Hermite-family ceilings for the κ₄ pattern

Gaussian restart h layers back, then K-th order response; decisive-neuron pattern corr at L12 / L15 / L16 [results/summaries/r17_hermite_family_ceiling_{A,B,C}.json].

| response | horizon | L12 / L15 / L16 |
|---|---|---|
| true ReLU | 6 | 0.859 / 0.820 / 0.818 |
| K = 2 | 6 | 0.548 / 0.449 / 0.530 |
| K = 3 | 6 | 0.848 / 0.799 / 0.817 |
| K = 3 | 4 | 0.749 / 0.687 / 0.711 |
| K = 3 | 3 | 0.652 / 0.588 / 0.643 |

Full cubic births only at the listed ages:

| birth ages | L12 / L15 / L16 |
|---|---|
| 1–4 | 0.736 / 0.682 / 0.713 |
| 2–6 | 0.709 / 0.708 / 0.691 |
| 4–6 | 0.37–0.41 |
| 1 | 0.34 / 0.24 / 0.23 |

Every κ₄ carrier built from second-order births stalls near 0.44 (see §4).

### 2.4 Shipped κ₄ features: chain-computed correlation vs the teacher

[results/summaries/k4s_fit.json, r18_htqb_combo.json]

| map | correlation |
|---|---|
| V55 map | ~0.51 |
| + HTQB star (V56) | 0.61–0.67 (held out across networks) |
| teacher-input ceiling of features + star + path | 0.71–0.76 |

### 2.5 Exact algebra you can use

Second- and third-order curvature-birth cumulant identities and readout factors (24 for C·T and star, 48 for the Gaussian path), verified to ≤ 1e-14 [consultations/prompts/ROUND18_prompt_cubic_birth_carrier.md §3, experiments/probes/r17_curvature_identities.py].

## 3. What a top-10 result requires (derived)

**Target.** Top 10 needs a score of ≈ 2.1e-9. For example:
- raw ≈ 1.4–1.5e-8 at util ≈ 0.14–0.15; or
- raw ≈ 1.1e-8 at util ≈ 0.19.

**What closes the gap.** On MLP #0, the plain chain + exact κ₃ and κ₄ inputs gives raw 6.2e-9. Capturing about half of that joint-oracle gain (raw ≈ 1.45e-8) AND cutting cost by about 20 % would reach the top 10. κ₄ alone, even perfect, reaches only about #24.

## 4. Do NOT re-propose (measured dead; details in docs/architecture_research/KILL_AUDIT.md and consultations/round14..18_verdicts.md, new_architecture_step1.md)

**Second-order κ₄ carriers** (ceiling ~0.44 at L12-16):
- SBQM, Sym(H⊗C): 0.32–0.39;
- pair-innovation core: 0.36–0.44;
- Claude's Q1 C·T hub-pair carrier with exact κ₃: 0.43–0.44;
- GHQ-6: 0.47–0.56;
- CSF / "Factorised Curvature-Birth" (SciMaster): 0.06–0.40, with fabricated PASS claims.

**HTQB (Gaussian star + path).** 0.56–0.64 on its own. The star is already shipped (V56); the path costs 27 u for little gain.

**Readouts that do not carry the pattern:**
- source-resolved skew energy: +0.004;
- κ₄ riding κ₃ legs: ×1.05;
- per-neuron regressions across networks;
- squares, logs, carried (W∘W)ᵀ features;
- lever-weighted fits (worse end to end).

**Replacing the κ₃ source chain** with Gaussian Hermite-tree κ₃: closure error 10–20× the chain's even with an oracle level.

**Cost-side architectures (60+ kills):**
- κ₃ cuts recovered by κ₄ (×0.70 → ×1.02 as κ₃ gets cheaper);
- age gates / ranks are at their optimum (one-for-one law);
- MC control variates correlate ≤ 0.2 at depth;
- DHMP, diagonal closures, particles, aggregate states, ISDF, adjoint, RCMP, EVC, exchange term.

**Parity theorem.** No statistic even in the later weights carries the deep pattern.

## 5. The question

**Design ONE architecture change (at most two) that captures a large part of the joint κ₃ + κ₄ oracle gain (§2.1) and is affordable.**

Guidance from the evidence:
- **κ₄:** a carried quartic state whose births include the CUBIC response (§2.3; ages 1–4 suffice for ~0.70, ages 1–6 for ~0.8) and that uses the chain's existing legs (A, P per young source; `experiments/variants/v56_star.py` shows how to read X = A + P·diag(w1·var) from them).
- **κ₃:** the accumulated (2,1)-slice error, 4.6–6.6 %, 91–96 % of it in all-distinct post-ReLU κ₃ entries that are 9–16 % wrong, and about 55 % of it unexplained at second order in K. Is there a cubic-order or carried correction to the κ₃ births or slices?
- **Cost:** any proposal must state its u and metered-op cost. The breakeven is ~+30 u for a 74 %-of-κ₄-oracle gain. A combined proposal that also removes cost (for example fewer old-source ranks once κ₃ is more accurate) is welcome but must be measured.

## 6. Protocol (mandatory; this is how we judge, fairly)

1. **Ceiling first.** An exact, untruncated version on stored data or exact Monte Carlo, scored by:
   - κ₄: decisive-neuron pattern corr — `dec = |μ/σ| < 1.5` of the teacher; `corr(pred/v², K4d/v²)` on `dec`. Do NOT use raw κ₄ or top-k sets.
   - κ₃: relative error of the (2,1) slice.
   - **Preferably end to end:** final MSE through the joint-oracle harness `experiments/variants/r20_joint_oracle.py` + `experiments/chain/run_variant_teacher.py` with your quantity injected.
2. **Component test** with the chain's own quantities.
3. **Same-maturity baseline**, then Kaggle 100-MLP paired run (`kaggle/runners/*`, `experiments/probes/kaggle_pair_compare.py`). The harness predicts grades to 0.05 %.
4. **Pre-register** PASS / KILL thresholds before each run.
5. **Heavy runs go to Kaggle:**
   - Ask the user before using accounts.
   - CPU kernels: up to 5 per account. Accounts: foysalemonshanto (default `~/.kaggle`) and esfersami50 (`KAGGLE_API_TOKEN` from `C:/Users/hp/.kaggle-esfer/access_token`; never print it).
   - GPU on foysalemonshanto.
   - The laptop has ~1–2 GB free RAM: no full-size local runs.
6. **Validate before pushing anything remote.** Before any grader submission, run the Kaggle smoke reproduction (`kaggle/smoke/kernel/run.py`: whest run `--runner subprocess --n-mlps 1 --debug --fail-fast`).

## 7. Data and code map

| item | location |
|---|---|
| Teachers, MLP #0, all 16 layers (mean, C, Y21 = κ₃(i,i,j) incl. diag, K22, K31, K4d, relu_mean) | `data/kaggle_out/kaggle_out_k4pub/k4teach/layer{l}_seed0.npz`, `weights_seed0.npz`, `truth.npz` (1e9-sample means) |
| Synthetic net teachers (layers 2, 4, …, 14, 15) | `data/kaggle_out/kaggle_out_k4/k4teach` |
| Post-ReLU teachers, L3/7/11/14/15 | `data/kaggle_out/kaggle_out_k4post/k4post` |
| Packed GPU input (W, mean, C, K4d) | `kaggle/r17/data/r17_data.npz`; Kaggle dataset `foysalemonshanto/r17-ceiling-data` |
| Slim oracle data | `foysalemonshanto/r20-joint-oracle-data`, `esfersami50/r20-joint-oracle-data` |
| Chain κ₃ source captures | `data/dumps/r7_cap_L{1..16}.npz` (reconstruction: `blocks_of_source` in `experiments/probes/r7_srcgain.py`) |
| Monte-Carlo templates | `experiments/probes/r17_hermite_family_ceiling.py`, `r20_onestep_channels.py`, `opus12_k4_carry.py` |
| Current best estimator | `submissions/v56As_fs/estimator.py` (A-class) and `submissions/v56s_fs/estimator.py` (B-class, safe) |
| Full record | `writeup/ALGORITHM_TRACK_PAPER.md`, `docs/build_notes.py`, `docs/experiment_queue/queue.json`, `consultations/ARCHITECTURE_CANDIDATES_2026-10-09.md` |

## 8. Deliverable

1. The architecture: state, birth, transport, readout, equations with shapes.
2. A cost estimate in u and metered ops.
3. Why it is not a dead item.
4. **The ceiling run, executed** (on Kaggle if heavy), with its script and JSON committed, and a pre-registered verdict.
5. If the ceiling passes: the component test and a prototype estimator variant, validated by the smoke reproduction, ready for a Kaggle 100-MLP run.

Mark every number as measured / derived / estimate. Do not claim a result you did not run.
