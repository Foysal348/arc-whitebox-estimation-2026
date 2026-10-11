# A conditional budget for the cumulant chain

Architecture assessment and a bounded handoff to Claude | 11 October 2026

Verdict: I have not found a defensible new cumulant representation that reaches your requested accuracy at 60% of the FLOPs. The concrete alternative assessed here is network-conditioned solver selection: inspect the weights cheaply, select one computation budget, then run one complete cumulant chain. This changes compute allocation; it does not remove the per-source tensor machinery.

The first audit is negative. On your archived 100-network evaluations, even a selector that knows the true error of every available solver improves on the best fixed solver by only 0.80-0.86%, before selector overhead. I would not build a learned selector for that existing solver bank. A wider-budget version remains untested and earns at most one bounded falsification test, not a full implementation campaign.

Labels: [D] derived; [M] measured here from archived JSON, not new network inference; [Q] supplied by you or recorded in project files; [E] arithmetic/engineering estimate; [S] speculative proposal or proposed acceptance threshold.

## What remains intact

Your CBA-13 result is decisive against delayed construction or omitted early cumulants: zeroing windows costs roughly 15-125 times baseline MSE [Q], and restoring the sources does not repair the accumulated mean/covariance error. Both branches proposed here retain every birth and process every layer. The cheaper branch can approximate the carried state, but cannot erase a time window or later recover it for free.

## The single mechanism [S]

```text
network weights W_1,...,W_16
      -> diagonal-Gaussian pilot -> small frozen gate
      -> choose H OR C before inference
      -> selected full-depth chain -> final means
```

| Component | Role |
| --- | --- |
| H: reference expert | The current frozen production chain; about 194 u and raw MSE 1.97e-8 [Q]. |
| C: cheaper expert | One same-code, reduced-budget chain; cost and accuracy unmeasured. No different closure physics. |
| Gate | Predict which expert has lower MSE-times-cost on this network. No truth, network ID or teacher at inference. |

This is not a new compression result. The reduced ranks in C are already familiar knobs. The only untested hypothesis is that their error/cost tradeoff is predictable across networks, so a knob that loses on average might be useful on a detectable subset. Your existing negative results make this a low-confidence hypothesis [S].

Scope: analytical design, literature check and archived-score arithmetic only. No network inference, router training, estimator editing or leaderboard submission was performed.

# State and arithmetic

A cheap selector is useful only if a genuinely useful cheap expert exists.

## Pilot state and deterministic update [D]

For row-vector activations z = hW, store vectors m and v, initialized by m = 0, v = 1. The following deliberately ignores off-diagonal covariance. It is only a feature generator; none of its moments enter the selected chain.

```text
a = W.T @ m
b = (W*W).T @ v
s = sqrt(max(b, tiny));  alpha = a/s
p = Phi(alpha);  f = phi(alpha)
m_new = s*f + a*p
v_new = max((a*a + b)*p + a*s*f - m_new*m_new, 0)
```

Use a fixed normal-CDF implementation and a documented small positive floor. At layers 4, 8, 12 and 16 record four permutation-invariant scalars: mean(v), mean(m*m), fraction with |alpha| < 1, and mean(phi(alpha)). This is a fixed 16-feature vector [S]. Scaling and decision thresholds are learned offline and frozen.

The two matrix-vector products cost about 4n2 scalar operations; squaring W costs n2. Thus the leading 16-layer pilot cost is 80n2 FLOPs, or 40/n = 0.0390625 u at n = 1024 [D]. A provisional arithmetic allowance of 0.06 u includes vector operations and reductions [E]. This is not a metered guarantee: CDF charges, allocation, dispatch and residual wall-time charges must be measured.

Additional pilot memory is one n by n square-weight buffer (4 MiB in float32) plus O(n) vectors [D]. Release that buffer before chain execution. No additional n-cubed product is used. The pilot can process the already supplied weights; it need not duplicate the entire model.

## The actual estimator state [D/S]

The chosen chain still carries m (n), C (n2), the dense young A/P legs, shared old basis and factors, birth coefficients, thin feedback legs and existing fourth-cumulant features. With fixed layer count and ranks its storage is the same order as the existing chain. It retains the signed third-cumulant contributions through its existing transport/readout; the gate does not reconstruct their odd fluctuation from even statistics.

Choose the expert before any branch-specific approximation. Executing H and C and then selecting would pay both bills. Switching from a previously truncated C to H midway would not restore its missing history. Neither operation is part of this proposal.

## One concrete low-cost branch to falsify [S]

Freeze H first. In an isolated copy change only old shared ranks 352/224 to 256/160 and R_FB from 4 to 1; retain age thresholds 5/7, R_RES = 16, all young sources and the existing closure/κ4 settings. These starting values match the inspected archived implementation [Q]; verify them against the actual current submission before applying the comparison.

R_FB is a class constant in the inspected file, not an environment-variable knob. This branch is a test of conditional usefulness of known rank cuts, not a claim those cuts are newly viable. Its numerical u count cannot be honestly supplied without metering the actual branch. H averages 194/16 = 12.125 u per layer [D from Q]; C averages its measured total divided by 16. Layer costs are not uniform.

# What a selector must predict

Optimize the evaluator score on each network; do not optimize utilization alone.

## The exact objective [D]

```text
L_ij = MSE_ij * max(0.1, (C_ij + g_i)/1024)
delta_i = L_iH - L_iC
choose C iff predicted E[delta | pilot features] > 0
```

Here C and g are effective compute expressed in u, including any evaluator wall-time conversion; g is the measured pilot overhead. At deployment only the chosen expert is run. Train a small cost-sensitive classifier with label sign(delta) and sample weight |delta|, or a shallow regression tree for delta [S]. False choices on large-loss networks must count more than harmless mistakes.

For a two-expert bank, the unattainable outcome oracle has score mean(min(L_H,L_C)). Its improvement against always using H is mean(max(L_H-L_C,0)). This remains possible even when the two experts have equal average score. For example, scores H = (1,1), C = (0.9,1.1) give an oracle score of 0.95 [D; illustrative]. Predictability is an additional requirement, not a consequence of this identity.

## Cost versus allowable error [D, illustrative]

When both branches are above the score floor and pilot cost is negligible, C wins on a network only if MSE_C/MSE_H < C_H/C_C. The following uses H = 194 u solely for break-even arithmetic; these are not predicted branch costs or errors.

| C budget (u) | Largest break-even MSE ratio |
| --- | --- |
| 180 | 1.0778 |
| 176 | 1.1023 |
| 170 | 1.1412 |
| 160 | 1.2125 |
| 116.4 (60% of H) | 1.6667 |

At or below the 102.4 u floor, further arithmetic reductions do not improve this score multiplier. Apply the max() separately on each network, and include overhead before applying it. For instance, if the cheap expert were 160 u, routing could never reduce the average bill to 116.4 u: a mixture of 160 and 194 u has average cost at least 160 u [D].

## Necessary ceiling before learning [D]

```text
oracle_gain = 1 - mean_i min_j L_ij / best_fixed_score
best_fixed_score = min_j mean_i L_ij(without pilot)
```

For a fair deployment comparison the routed losses include pilot overhead while each fixed expert does not. Existing archived results below set pilot cost to zero, making their ceiling optimistic. A perfect label selector is not an implementable algorithm: it has access to final errors that inference cannot observe.

The per-layer 2% D21/d3 constraint is stronger than improving the final score. This proposal does not prove that constraint, does not evade your slice-state obstruction, and does not establish a 60%-cost architecture. If the constraint is mandatory, it must be checked independently for every network assigned to C.

# Executed falsifier: the archived bank

100 matched networks; same dataset hash; no new inference [M]

I recomputed selection from per-network adjusted scores, matching by mlp_name rather than array position. Included runs have width 1024, depth 16, the same dataset SHA-256, the same 100 unique names, no reported failures, and scores consistent with MSE times the evaluator multiplier. Means below are arithmetic means of per-network scores, not products of aggregate means.

| Bank | Experts | Best fixed score | Oracle score | Extra gain |
| --- | --- | --- | --- | --- |
| V56 tuning | 10 | 3.77639079e-09 | 3.74387700e-09 | 0.861% |
| All matched | 86 | 3.75860143e-09 | 3.72847903e-09 | 0.801% |

Best fixed V56 variant: v56t_e/run_v56_fit_synth.json. Best fixed across the full bank: fbg_1/run_base_fbg.json. The full-bank oracle is lower in absolute score, but its gain relative to its own stronger fixed baseline is smaller. Those percentages should not be compared as gains over one common baseline.

The historical reference run_v56_ref.json scores 3.78992640446e-9; its reported raw MSE is 2.04094e-8 and its cost is about 190.156 u [M]. These numbers are not the current user-reported 194 u / 1.97e-8 baseline. Relative to this older reference, the oracle improvements are 1.215% and 1.621%; those larger values would overstate the benefit of routing over simply choosing the best fixed variant.

## Why this already stops one tempting project

Do not train a gate over these archived experts. Even perfect selection cannot deliver the proposed 3% development gate on these recorded outcomes. The archive contains no genuinely low-budget expert: its smallest mean FLOP bill is 188.860 u [M]. Therefore this result rejects a routing build for the existing bank, but does not measure a 140-170 u expert that is absent from it.

This is a finite-bank, finite-sample ceiling, not a theorem for unseen networks or future estimators. The 86 variants also differ in maturity and tuning; they are not a controlled ablation of one baseline. Selecting the minimum of many noisy teacher errors can exaggerate gains. Reusing the same 100 networks for design and claimed validation would add selection bias.

## Reproducibility

The accompanying routing_oracle_audit.py reads data/kaggle_out/**/run_*.json and emits archived_routing_audit.json. It records the source-file hashes, per-run mean costs, winner counts and eligibility checks. The reproduced audit accepted 86 runs and rejected 46 because their network names/counts did not match [M]. No predictor was fitted.

```text
python routing_oracle_audit.py --root PROJECT_DIRECTORY
    --out archived_routing_audit.json
```

The stored historical runs use zero residual-time conversion in run_config [Q/M]. That fact does not establish the current competition billing rule. Claude must use the active evaluator and resource caps when measuring a proposed pilot.

# Claude handoff: test before building

One candidate pair, explicit stop conditions; thresholds below are proposed [S].

## 1. Freeze and meter before a dataset run

Hash the current H source and its configuration. Create only the C branch specified on page 2, keeping all closure coefficients identical. Run a correctness/resource smoke test on one representative network and measure the complete bill, including pilot, allocation and residual time. Check for unhandled rank assumptions. If C saves less than 8% of H compute, stop this pair; do not train a gate or sweep many ranks.

## 2. The decisive paired oracle test

Run H and C offline on 32-40 representative networks not selected for favorable outcomes, using existing official final-mean truth. Record per-network MSE, effective cost and all failure flags. Include measured pilot cost in each routed loss, but not in the fixed-expert comparator. Compute the outcome-oracle gain over the better fixed expert. If gain is below 3%, kill this pair before feature engineering. This is a screening threshold, not a significance claim or an assertion that the cheap branch will pass.

Also inspect late D21 and d3 differences against H where captures exist. Those comparisons diagnose branch damage; they are not replacements for paired final-mean score. If the 2% channel criterion remains a hard requirement, enforce it at every required layer independently. Do not repair C with teacher injections.

## 3. Only if the oracle screen passes

Collect a training set and a separate validation set, ideally at least 100 independent networks each. Generate only the 16 pilot features specified here. Fit a frozen depth-1 or depth-2 tree with sufficiently large leaves (for example 20 training networks). Split by complete MLP, never by neuron. Exclude network index/name, oracle errors and ground-truth moments from features. If competition rules restrict auxiliary training data or fitted policies, verify eligibility first; this report does not establish it.

Evaluate one selected expert per validation network and compare with the best fixed expert. A proposed promotion gate is at least 1.5% lower mean score and a paired-bootstrap 95% confidence interval for the score ratio entirely below 1, with no resource failures. Keep model selection inside training; reserve untouched validation for the final decision. Previously mined public networks are exploratory, not pristine validation.

## Failure modes that make a negative result likely

Random He networks may be too homogeneous for useful conditional allocation. The pilot discards covariance, so its features may not predict sensitivity to rank truncation. Your prior local corrections and average rank tradeoffs are adverse evidence. Even with a wide outcome-oracle gap, unobservable error cancellation can make the best expert unpredictable. Any of these is sufficient reason to stop, rather than add a larger gate.

## Evidence and provenance

Project evidence inspected: docs/architecture_research/KILL_AUDIT.md; consultations/ARCHITECTURE_CANDIDATES_2026-10-09.md; results/summaries/kaggle_v56_tuning.txt; kaggle/arch/code/v56As_fs.py (rank defaults, transport and readout); the matched JSON files enumerated with hashes in the audit. Your latest CBA-13 report takes precedence over earlier conjectures.

Related primary literature: Bolukbasi et al., Adaptive Neural Networks for Efficient Inference, ICML 2017, studies cost-sensitive network selection. This supports the policy formulation only; its classification results provide no evidence for ARC mean-estimation accuracy. https://arxiv.org/abs/1702.07811

Recommendation: archive the narrow-bank routing idea as negative. If you want one bounded new test, ask Claude for the same-base cheap-expert oracle screen above. I cannot justify a large architecture build, a predicted leaderboard gain, or a claim that this solves the original 60%-FLOP problem.
