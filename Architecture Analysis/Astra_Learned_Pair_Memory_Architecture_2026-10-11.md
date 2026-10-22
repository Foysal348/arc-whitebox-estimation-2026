# A learned pair-memory estimator

One high-risk architecture with a large potential cost reduction | 11 October 2026

Proposal: distill the complete cumulant chain into a nonlinear recurrent estimator with two dense pair-memory matrices. It reads the real weights, computes mean/covariance through the existing analytic closure, and learns the missing third- and fourth-cumulant information jointly. There are no birth-indexed legs, source joins or source-age loop at inference.

Status: a falsifiable research architecture, not a demonstrated breakthrough. Its dominant arithmetic can fit near the score floor. There is no evidence yet that two matrices can preserve enough history for the required accuracy. Your failed slice recursions and learned corrections make that a severe risk. I am recommending one capacity test, not promising leaderboard improvement.

[D] derived; [Q] user/project evidence; [E] estimate; [S] untested hypothesis or proposed threshold; [T] small algebra test executed here.

## Why this is a different wager

The old chain is an explicit approximate representation of a third-order tensor. This student represents a nonlinear predictive state for the next queries. Two states that have the same current D21 may have different second memory matrices and therefore different future predictions. The state is trained across complete trajectories, not fitted as a local correction to the final means.

This is related to the previously rejected channel-history idea A19, but supplies an explicit recurrence, a parameter budget and a training target. It is not an exact closure. Irreducibility rules out an exact universal linear compression; it does not establish that this particular learned nonlinear state succeeds on random He networks. The latter is the entire empirical gamble.

## State and one inference pass

```text
state_l = (m_l, C_l, H_l[1], H_l[2])
shapes:      n     n x n   n x n    n x n
initial: m_0=0, C_0=I, H_0[1]=H_0[2]=0
```

For each layer: transport C exactly by WTCW; transport both H matrices with the cubic pair operator on page 2; read out D21, d3 and a learned fourth-cumulant diagonal; apply the analytic ReLU mean/covariance closure; update the two H matrices with a shared small pairwise neural network. All 16 layers run. All birth contributions are introduced through the local update and may persist through the recurrent state.

Persistent float32 storage is 3n2 + n scalars, about 12 MiB at n = 1024 [D], excluding model weights, scratch buffers and multiplication workspace. This is storage for the proposed state, not a measured process-memory footprint.

## What is and is not retained

Full-rank covariance remains. The transport retains the current weight signs and cubic dependence. A learned second matrix is responsible for information that the current slice alone cannot predict. There is no claim that it carries every source accurately, represents the true off-slice tensor, or satisfies a generic 2% tensor approximation bound. If the 2% D21/d3 requirement is mandatory, it remains an independent pass condition.

# Transport and a nonlinear query

Row-vector convention: pre-activation z = hW.

## Three-product cubic transport [D]

```text
J(H)_ab = H_ab           if a != b
J(H)_aa = H_aa / 3
V = J(H) @ W
R_W(H) = (W*W).T @ V + 2*(W*V).T @ W
```

The operator is exactly the transported (2,1) slice of the symmetric tensor whose repeated entries are K(a,a,b) = H(a,b), and whose all-distinct entries are zero. Setting J(a,a) = H(a,a)/3 avoids triple-counting the diagonal. It costs three dense products, not two. This identity is exact only for that restricted tensor; applying it to a true slice does not recover the true full-tensor transport.

```text
a = W.T @ m;   S = W.T @ C @ W
Z1 = R_W(H1);  Z2 = R_W(H2)
Y21_ij = Z1_ij + t_ij * O_theta(o_ij,e_ij)
d3 = diag(Y21);   k4_i = S_ii^2 * g_theta(q_i)
```

Essential nonlinear readout [S]: t_ij = S_ii*sqrt(S_jj). Use six odd inputs o: normalized Z1_ij, Z1_ji, Z2_ij, Z2_ji, alpha_i, alpha_j; and eight even inputs e: correlation, log(S_ii), log(S_jj), layer/16, diagonal indicator, row mean-square correlation, and squared normalized Z1_ij and Z2_ij. Define O(o,e) = [r(o,e)-r(-o,e)]/2 with a shared 14-to-8-to-1 MLP r. This preserves odd parity exactly. Reading Y21 = Z1 alone would collapse back to a restricted-tensor slice closure and is not the proposed model.

g is a shared 8-to-8-to-1 MLP [S]. Use eight dimensionless features: alpha2, log(S_ii), d32/S_ii3, alpha*d3/S_ii1.5, normalized row and column mean-squares of Y21, normalized row mean-square of Z2, and layer/16. Here alpha = a/sqrt(diag(S)); normalize any pair Z_ij by S_ii*sqrt(S_jj). All eight features are even under a simultaneous sign flip of the current W.

Run the existing pointwise mean and bivariate covariance closure with (a,S,Y21,d3,k4), producing (m_new,C_new). Force k4 = 0 at the first Gaussian pre-activation. Replace all per-source inputs to that closure with these predicted channels. Do not retain V56 star features that require source legs: the small g head replaces their fourth-cumulant role and must learn it jointly.

## Nonlinear post-ReLU memory [S]

```text
p_i = Phi(alpha_i);   t_ij = S_ii*sqrt(S_jj)
H1_new_ij = p_i^2*p_j*Y21_ij + B_ij + t_ij*f1(F_ij)
H2_new_ij = p_i^2*p_j*Z2_ij        + t_ij*f2(F_ij)
```

B is the Gaussian ReLU third-cumulant (2,1) slice computed from (a,S), using existing univariate/bivariate Gaussian formulas; it is a baseline birth, not the full chain birth with all compensation. Any omitted compensation is part of the learned residual. f is one shared 16-to-8-to-2 MLP with ReLU hidden units. No neuron-specific parameters or network IDs are used.

The 16 inputs F_ij are normalized Y21_ij, Y21_ji, Z2_ij, Z2_ji; correlation S_ij/sqrt(S_ii*S_jj); alpha_i, alpha_j; p_i, p_j; normalized B_ij, B_ji; log(S_ii), log(S_jj); layer/16; the diagonal indicator; and the row mean-square correlation at i. Floors for normalization must be fixed and documented. Transposed entries use their own index-appropriate normalizers.

A fixed hidden state gives R_(-W)(H) = -R_W(H), and the antisymmetrized readout preserves this for Y21 and d3 [D]. The pointwise shared maps and row reductions respect neuron permutations [D]. Neither property guarantees accuracy. A 4-wide toy tensor check reproduced the restricted transport to 8.88e-16 and the sign-flip identity exactly [T]; no ARC inference was tested.

# Why it has breakthrough-sized upside

Arithmetic permits the target; accuracy and metering remain the gates.

| Per layer | Dense products | Reason |
| --- | --- | --- |
| Covariance transport | 2 | W.T @ C @ W |
| Two cubic memory transports | 6 | Three products per H matrix |
| Births, closures and small heads | 0 | Pointwise functions and O(n squared) reductions |
| Total | 8 | Independent of source count and depth history |

With 16 layers this is at most 128 square products [D]; zero initial H and last-layer omissions can only reduce that count. Standard Strassen with 18 block additions per recursion node has the following exact arithmetic count at width 1024. These numbers do not assume a faster Winograd variant.

```text
B_d = 7^d*(2*m^3-m^2)
      + 18*sum_{j=0}^{d-1} 7^j*(1024/2^(j+1))^2
m = 1024/2^d;     product_u = B_d / 2^31
```

| Strassen depth | One product (u) | 128-product core (u) |
| --- | --- | --- |
| 3 | 0.680077 | 87.0498 |
| 4 | 0.606150 | 77.5872 |
| 5 | 0.550050 | 70.4064 |
| Classical | 0.999512 | 127.9375 |

The post-ReLU pair head uses about 288 multiply/add operations per pair per layer, or 2.25 u over 16 layers [D]. Two evaluations of the odd readout add about 480 operations per pair, or 3.75 u. Together these cost about 6 u, before activations, normalization and feature construction. Mean/covariance closures, Gaussian B, fourth-cumulant features and allocations are additional. A 100-116 u total with depth-3 multiplication is an engineering target [E], not a measured bill. Classical multiplication alone misses 125 u. Batch the two memory channels within each multiplication family; do not assume batching reduces FLOPs.

The residual-time/op-count cap can kill this design despite favorable FLOPs. Meter one frozen, untrained forward pass before collecting training data. No estimate here establishes compliance with the 8 GB or residual-time limit. Training and autodiff are offline; inference exports only small fixed coefficient arrays and permitted NumPy-like operations.

## A quantitative score opportunity [D, conditional]

Let teacher true-mean MSE be E_T and student-to-teacher mean discrepancy be e. The triangle inequality gives E_S <= (sqrt(E_T)+sqrt(e))2, over the same evaluated coordinates/networks. This requires neither independent errors nor helpful cancellation.

```text
E_T = 1.97e-8;  e <= 1e-9
=> E_S <= 2.958e-8
If student cost <= 102.4 u, baseline cost = 194 u:
score_S / score_T <= 0.7925
```

Thus even a student slightly less accurate than the chain could lower score by about 21% in this conditional example. At exactly the same raw MSE and floor cost the score ratio would be 102.4/194 = 0.5278. These are targets and mathematical implications, not predicted outcomes. Heterogeneous per-network billing must be scored per network, not by multiplying aggregate averages.

The required mean discrepancy e = 1e-9 corresponds to RMSE 3.16e-5. That is demanding. No theorem or cited metanetwork result establishes this accuracy, and the small learned fourth-cumulant head may itself be inadequate.

# The decisive experiment for Claude

Test representational capacity before polishing or submitting anything.

## First, the cheap resource falsifier

Implement only this two-memory forward pass with dummy fixed f/g/r parameters. Run one width-1024, depth-16 network under the real meter. Stop this prototype if total cost exceeds 116 u, or if memory/residual-time caps fail after ordinary batching [S]. This test needs no teacher training. Save a call-site ledger so there is no hidden n-cubed term in B or a closure.

## Then, one trajectory-distillation falsifier [S]

Use the frozen current chain as teacher on a small initial batch, for example 16 training and 8 held-out random networks. This produces cheap labels relative to fresh billion-sample Monte Carlo, but is still work for Claude. Log teacher pre/post m,C,Y21 and k4 at all layers. Stored one-network captures help debugging; they cannot establish generalization or supply a complete multi-network trajectory test.

Train f, g and r jointly through complete student rollouts from layer 1. Supervise normalized channels and mean/covariance trajectories, with a strong final-mean imitation loss. The teacher supplies labels only, never the student state during validation. A short teacher-forced warm start is optional, but the decisive test must be free-running. Fixed shared heads keep the first model small: the post-ReLU pair head has 154 parameters, the antisymmetrized readout 129, and the vector head 81, including biases: 364 total [D].

Initial pass gate: held-out student-to-teacher final-mean MSE at most 1e-9; stable variances and covariance closure; no numerical/resource failures; and, if the original channel requirement is retained, relative D21 and d3 errors at most 2% at every layer 9-16 using the project metric [S]. Count a failed seed as a failure, not an excluded observation. A reproducible large error floor closes this prototype; finite training failure is not an impossibility proof.

Only after this passes should Claude increase independent training/validation networks and evaluate actual final-mean MSE against official truth. A student can imitate a biased teacher or overfit eight networks. The score inequality on page 3 must be checked on the same held-out population, and the final score must include all billed work. Offline fitting eligibility and the use of synthetic teacher labels must be checked against current competition rules before deployment.

## Why prior negative results remain relevant

A pure slice recursion discards most of the next-layer signal; disabling the learned memory/readout is expected to fail badly. Local 13-44-feature correctors had near-zero held-out value [Q]. Here those features come from a learned full-matrix memory, and losses train all channels together, but that is a change of hypothesis rather than evidence of success. Two hidden matrices may still be far too small. If they need many more channels, the dense-product bill erases the benefit.

The student does not keep an exact CP/Tucker representation of the teacher tensor. Consequently the measured rank ceilings do not directly give a numerical error bound for its query-specific recurrent predictions. They do make any claim of universal exactness untenable. The nonlinear readout is essential to avoid restricting each predicted query to the transported repeated-index tensor. This removes that immediate algebraic objection, not the empirical accuracy problem. No early κ3 window is skipped, and no later state reset is supposed to repair earlier closure errors.

## Primary precedent and what it supports

Zhou et al., Permutation Equivariant Neural Functionals (2023), develops symmetry-aware networks that process neural-network weights: arxiv.org/abs/2302.14040. Lim et al., Graph Metanetworks for Processing Diverse Neural Architectures (ICLR 2024), studies equivariant weight-processing models: arxiv.org/abs/2312.04501. These support the design direction, not this recurrence, cost/accuracy tradeoff or ARC performance.

Recommendation: this is a high-risk architecture worth one bounded capacity test because a successful result could remove most per-source work. I would choose it over another marginal rank sweep. The two-memory sufficiency assumption is unresolved; no claimed breakthrough should precede the held-out free-rollout test.
