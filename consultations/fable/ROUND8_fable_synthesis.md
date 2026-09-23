# Round 8 (Fable 5.1, 23-24 Sep 2026): synthesis of the round-7 answers - what they missed, what is dead, what to run

Scope: public MLP #0, plain Gaussian chain (`V31_ANGULAR=0`), the 6e7-sample pre- and post-ReLU teachers, the 1e9 baked
truth, the round-7 captures (`data/dumps/r7_cap_L*.npz`, every source dense) and the round-7 kernel dumps
(`data/dumps/astra7_k4_mlp0.npz`). New scripts: `experiments/probes/r8_*.py`; summaries `results/summaries/r8_*.json`;
logs `results/logs_r8_*.txt`. Everything below is measured unless marked "derived"; nothing is fitted unless said so.
Chain runs used: 2 of the allowed 3 (section 6); everything else is numpy on stored arrays.

## 0. Verdict in ten lines

1. **The idea the eleven answers missed, measured positive**: the quartic kappa4 kernel's depth deficit is not a missing
   diagram class, not memory, not its Gaussian vertex weights - it is that GPT Sol's K3 x C "3+2" term with the
   repeated-index slice approximation t_rep is, by index bookkeeping, a (2,1,1)/(3,1)-sector term. The (1,1,1,1) 3+2 term
   needs the ALL-DISTINCT pre-activation kappa3, which is exactly what the chain's own sources carry (Astra's section-5
   "directional query", queue Q8). Evaluated from the captured sources it enters the kernel residual with coefficient
   1.0 +- 0.1 (theory 1) at layers 4, 8, 12, 16, lifts the decisive-neuron non-pair correlation from 0.69-0.75 to
   0.75-0.81, and cuts the G4-weighted kappa4 error to 0.095 / 0.116 / 0.157 / 0.193 / 0.197 at layers 4 / 8 / 12 / 15 / 16
   (chain 0.124 / 0.142 / 0.193 / 0.199 / 0.210; kernel + t_rep 0.101 / 0.137 / 0.186 / 0.204 / 0.220). Ages 1-4 carry
   ~90% of it at layers 4-12. End to end: section 6.
2. The (3,1) sector of the true non-pair kappa4 content is negligible (3e-5 of 1e-3 at depth): Q7 (kernel (3,1) slice) is
   dead as a kappa4-diagonal lever. The non-pair pattern is (2,1,1) + (1,1,1,1).
3. Sol's nine (2,1,1) hypertree classes (Q12) derived exactly with the B_r = (h_r - m_r)^2 - Var vertex weights and the
   -2 Cov Cov corrections: total 6e-4 (layer 4) -> 9e-5 (layer 16), fixed-coefficient R2 <= 0.07 (negative at layers 4-8),
   every deploy combination WORSE on the decisive neurons. Dead. The triangle is <= 5e-5 and uncorrelated.
4. Memory (transported star births, the kappa4 analogue of the kappa3 sources): anti-correlated with the kernel residual
   at depth (coefficient -0.9, corr -0.5 at layers 15-16). The deep multi-index kappa4 is fresh. Dead.
5. Non-Gaussian (Edgeworth) vertex weights in the kernel: ratios to the Gaussian weights 1.00 (median), no change in
   slope or error. Dead.
6. The second order in K of the kappa3 all-distinct tensor (12 K x K terms, coefficient 1/4 each, derived and tested):
   rms 3e-6 / 1e-6 / 2e-5 against an error of 2.2e-4. Dead. It also kills my own round-7 explanation of why S = 2
   hurts (it is not a doubled cross term), so no chain run was spent on it. The kappa3 all-distinct error stays 55-65%
   unexplained through second order; within this formulation it is a floor.
7. The coherent part of the covariance error (Terra) exists at the kappa3 level too: the D21 error has a rank-1 component
   along the pre-activation mean mu (overlap 0.97-0.99, 20-29% of the energy at depth), but it multiplies mu_a on the
   way to every closure input of neuron a, so the decisive neurons (mu ~ 0) never see it. Dead as a lever, consistent
   with Q10.
8. The variance channel has no separate lever: its fresh error is 77% the kappa3 (2,1)-slice input (TF numbers), the
   slice error is 91-96% the all-distinct kappa3 error, and that error is a floor (items 6-7). Only the kappa4 channel
   moved this round.
9. Ranked experiments (section 7): (i) kernel + t_rep + all-distinct 3+2 from the young sources, late layers and
   decisive neurons only, ~25-40 u; (ii) the same with the kernel's cost cut to the decisive neurons; (iii) nothing else
   from the round-7 answers survives.
10. Everything else in the eleven answers (sampling, adjoints, low-rank covariance, super-closure, mixture memo, Rao-
    Blackwell, aggregate matrices) was already dead on the queue's evidence and nothing here revives it.

## 1. The map (what is established, restated in one picture)

Error accumulates through the mean (gain 0.65-0.96) and the covariance (inherited share 0.52 -> 0.79). The fresh error
per layer is made by three closure inputs: the variance (50%), kappa3_iii (25%), kappa4_iiii (25%). Q5 (teacher-forced
closure) says the fresh covariance error is 77% the kappa3 (2,1)-slice input (0.285 of 0.376 at layer 7; 0.147 of 0.185
at layer 14) and ~20% closure residual + noise; round 7 says the slice error is 91-96% the all-distinct post-ReLU
kappa3 tensor, 9-16% wrong relative to itself. So the whole covariance/variance channel hangs on one object (the
all-distinct kappa3 content), and the kappa4 channel on another (the multi-index post-ReLU kappa4 content). This
round tested the remaining candidates for both.

## 2. The kappa3 all-distinct error: second order in K, the coherent part, and why S = 2 hurts (r8_kk_terms.py)

Derived. For distinct i, j, c the joint cumulant kappa3(h_i,h_j,h_c) depends only on the trivariate cumulants of
(z_i,z_j,z_c). At second order in the (2,1)-slice entries K[x,y] = kappa3(z_x,z_x,z_y) the connected terms are the 12
pairs of slice hyperedges (xxy) that cover {i,j,c}, coefficient (1/2)(1/2)(1/2) x 2 = 1/4 each:

| type | term (hub h; x, y the other two) | count | in the port |
|---|---|---|---|
| P | (1/4) w2_h (K_xxh w2_x)(K_yyh w2_y) | 3 | 0 |
| Q | (1/4) w4_h (K_hhx w1_x)(K_hhy w1_y) | 3 | 0 |
| R | (1/4) w3_h [(K_xxh w2_x)(K_hhy w1_y) + (x<->y)] | 6 | 1/8 (half), through Xt x Yt |

The port's newborn Sym(X1 x I x Y1) carries type R at half coefficient through its Xt x Yt cross term, and the round-7
knob S scales Xt and Yt, hence type R by S^2 - at S = 2 families A, B are exact but R is doubled. That was my hypothesis
for the +3.4% of S = 2. Measured (all-distinct part of every term, its own (2,1)-slice content removed this time; true
layer-l cumulants; target = the chain's all-distinct slice error e_dist(l+1)):

| post layer | e_dist rms | famA rms / slope / R2 | R rms / slope | P rms | Q rms / slope / R2 | joint(6) R2 | fixed theory (-1/2 A,B,R; -1 P,Q,dedge) R2 |
|---|---|---|---|---|---|---|---|
| 3 | 2.63e-4 | 1.08e-4 / -0.55 / 0.052 | 4.0e-6 / +3.1 | 1.8e-6 | 7.3e-6 / +2.1 / 0.003 | 0.105 | 0.089 |
| 7 | 2.27e-4 | 1.37e-4 / -0.60 / 0.133 | 4.3e-6 / +2.6 | 1.0e-6 | 1.3e-5 / +1.2 / 0.005 | 0.168 | 0.145 |
| 11 | 2.12e-4 | 1.28e-4 / -0.67 / 0.164 | 3.8e-6 / +1.5 | 5.9e-7 | 1.5e-5 / +1.0 / 0.005 | 0.184 | 0.154 |
| 14 | 2.24e-4 | 1.17e-4 / -0.72 / 0.142 | 3.0e-6 / +2.6 | 3.2e-7 | 1.8e-5 / +1.0 / 0.007 | 0.154 | 0.119 |
| 15 | 2.18e-4 | 1.19e-4 / -0.65 / 0.126 | 3.2e-6 / +1.8 | 3.7e-7 | 1.9e-5 / +0.8 / 0.005 | 0.133 | 0.111 |

The K x K terms are 1-8% of family A in size and carry <= 0.7% of the error variance each (their fitted coefficients are
noise). The S = 2 penalty is therefore NOT a doubled cross term; the chain-level XY experiment I had prepared
(`r8_xy_chain.py`, S and XY separated consistently at birth and in `_dslices`) was not run. Family A's coefficient is
-0.55..-0.72 with the (2,1) content removed (round 7: -0.55..-0.69), the chain-state terms give the same numbers as the
true-state terms (deployable), and a rank-16 range-finder K keeps 46-78% of family A (fine at depth, poor at layer 3).
The thin-leg deployment of P and Q is therefore possible but pointless.

Structure of the residual (r8 inline check): the error is uniform across alpha bins (relative error 0.10 at layer 8,
0.15 at layer 15 for every bin of the row and of the column index), 50/50 symmetric/antisymmetric, and LOW-RANK in part:
its top singular vector carries 11% (layer 8) -> 29% (layer 16) of the energy, the top four 19% -> 47% (random reference
0.4%), and those vectors lie along the pre-activation mean mu (|overlap| 0.97-0.99; also 0.94-0.96 with the true kappa3
diagonal, which is itself ~mu-shaped). That is Terra's coherent mean-direction error one level up. It does not matter
for the score: the next-layer variance proxy built from the rank-4 part is 1/4-1/3 of the full one on the decisive
neurons with correlation 0.26-0.29 (0.94 at layer 4 only), because every contraction of an mu-directed error into a
closure input of neuron a is proportional to mu_a. Same verdict as Q10, now for kappa3.

Bottom line for kappa3: first plus second order in K explain 35-45% of the all-distinct error; the coherent part is
irrelevant; the remaining random ~55% is unattributed. No fragment of the round-7 answers touches it. Floor.

## 3. Where the kappa4 kernel's deficit is NOT (r8_k4_memory.py, r8_k4_211.py, r8_k4_ngvertex.py, inline)

The kernel (Astra) + Sol's t_rep term has non-pair correlation 0.73-0.83 with the true non-pair part of the next kappa4
diagonal but the joint fit wants a scale 1.25 / 1.17 / 1.45 / 1.80 / 1.85 (kernel) and 1.15 / 1.13 / 1.41 / 1.75 / 1.92
(t_rep term) at layers 4 / 8 / 12 / 15 / 16, and its residual (true non-pair - kernel - t_rep term) is 1.47e-3 / 9.0e-4 /
7.1e-4 / 7.1e-4 / 6.9e-4 against a target of 2.1e-3 / 1.3e-3 / 1.0e-3 / 1.05e-3 / 1.0e-3. Four candidates for the residual:

**(a) The (3,1) sector.** With the post-ReLU teacher's K31 (convention verified: K31[j,k] = kappa4(h_j,h_j,h_j,h_k),
Gaussian birth gamma_j Phi_k C_jk fits it with slope 0.90 / R2 0.69 at layer 7), 4 sum W^3 W K31 is 1.1e-4 at layer 8 and
3.0e-5 at layer 15 of a non-pair target of 1.3e-3 / 1.05e-3. Negligible: the pattern is (2,1,1) + (1,1,1,1). Q7 dead.

**(b) Memory** - the kappa4 analogue of the kappa3 sources: the star births k(h_r,h_r,h_k,h_m) = beta_r a_kr a_mr,
k(h_r,h_r,h_r,h_k) = gamma_r a_kr and the all-distinct star w3_r a_ar a_br a_cr (= the kernel's own f2 at age 0)
transported linearly with exact dense transport from the chain's own state, every age, split by sector. Their diagonal
at a later layer is elementwise on the kappa3 legs (6 beta P^2 At^2 + 4 gamma P^3 At + 4 w3 P At^3), i.e. free. The
transport gain ~0.8/layer would make the steady state 1.7x the fresh birth - the observed deficit. Measured: the old
stars (ages >= 1) regress on the kernel residual with coefficient +0.07 / -0.09 / -0.47 / -0.88 / -0.93 (R2 0.007 /
0.006 / 0.06 / 0.26 / 0.27, corr -0.5 at depth); every deploy combination with them is worse (errG4 0.204 -> 0.242 at
layer 15). The linearly transported (2,1,1)/(1,1,1,1) star content of old births is anti-correlated with what is
missing; the deep multi-index kappa4 is fresh (consistent with K3ac(d)). Dead.

**(c) The exact (2,1,1) law (Q12).** Derived at leading order with the B_r vertex weights v_k = E[B_r He_k]/s^k
(v1 = 2m(1-Phi), v2 = 2Phi - 2m phi/s, checked by quadrature) and the -2 Cov Cov corrections: 11 hypertree classes
(G1 star, G2 path, H3 doubled edge at r, G4, G5, K1-K6 with the slice hyperedges K_rrb, K_bbr, K_bbc paired with one C
edge) plus the triangle (n^3 per neuron, done on 64 neurons). Directional contraction (hub-r forms 6 sum_r w_r^2 g_r
[2 (L^T w)(M^T w) - 2 (L o M)^T w^2]; hub-b forms 12 sum_b g_b w_b [(L^T w^2)(M^T w) - (L o M)^T w^3]) on the true
layer-l cumulants and on the chain's state (identical numbers):

| layer | 211 total rms (residual rms) | largest classes | fixed-coefficient R2 (G4-w) | errG4 kernel+t_rep -> +211 | decisive corr -> |
|---|---|---|---|---|---|
| 4 | 6.3e-4 (1.47e-3) | K2 5.0e-4, G2 3.6e-4, G1 3.3e-4, G4 2.2e-4 | -0.11 (-0.09) | 0.101 -> 0.113 | 0.746 -> 0.712 |
| 8 | 2.7e-4 (9.0e-4) | K2 3.0e-4, G4 1.9e-4, G1 1.6e-4 | -0.03 (-0.10) | 0.137 -> 0.154 | 0.738 -> 0.724 |
| 12 | 1.4e-4 (7.1e-4) | K2 1.7e-4, G4 1.2e-4 | 0.05 (-0.04) | 0.186 -> 0.199 | 0.739 -> 0.719 |
| 15 | 9.6e-5 (7.1e-4) | K2 1.2e-4, K1 9.3e-5 | 0.08 (0.04) | 0.204 -> 0.217 | 0.722 -> 0.706 |
| 16 | 9.2e-5 (6.9e-4) | K2 1.1e-4, G4 8.4e-5 | 0.07 (0.02) | 0.220 -> 0.237 | 0.692 -> 0.689 |

The exact (2,1,1) content is too small at depth to matter and at layers 4-8 it is already inside the kernel's own
polynomial (adding it double counts: the star G1 alone has corr -0.25 with the residual). Sol's kernel of truth: the
classes are real and cheap; his error: they are not where the pattern is. Dead.

**(d) Non-Gaussian vertex weights.** Kernel recomputed with E[h^(k)] = w_k + (kappa3/6) w_{k+3} + (kappa4/24) w_{k+4} +
(kappa3^2/72) w_{k+6} from the chain's D3 and g4: median ratios to the Gaussian weights 1.000 (d1), 0.985-1.002 (d2),
1.000-1.011 (d3); slope, correlation, errG4 unchanged to three digits at every layer. Dead.

## 4. Where it IS: the (1,1,1,1) K3 x C term with the all-distinct kappa3 (r8_k4_1111.py)

Derived. Sol/ARC's 3+2 hypertree K_jkl C_jm contracted four ways with x = d1 o w reads 12 sum_j s_j t_j with
s = (d2 o w) o (C_off x) and t_j = sum_{kl} K_jkl x_k x_l. Sol's t_rep = D21^T(x o x) + 2 x o (D21 x) - 2 D3 o (x o x) is
the part of that sum with k = l, k = j or l = j - i.e. a monomial with a repeated index - so it is a (2,1,1)/(3,1)-sector
contribution. For four distinct output indices the hyperedge must have three distinct indices: t_dist_j = sum_{k != l,
both != j} K_jkl x_k x_l, the all-distinct kappa3 content, which the chain represents as hub forms. For a block
Sym(X x P x Y) (the chain's Sym = (1/6) sum of permutations), t = (1/3)[X (P^T x o Y^T x) + P (X^T x o Y^T x) +
Y (X^T x o P^T x)]; summed over the three blocks (X1,P,Y1), (M,P,P), (X3,P,Y3) of every live source, minus each
source's own repeated part (its (2,1) slice and diagonal, computed from the same legs). The pre-activation kappa3 at
layer l is the source stack at l (ages >= 1); the "newborn" of layer l belongs to the post-ReLU tensor and correctly
gets coefficient ~0 (0.06-0.24) when included as a control. Measured on the captured sources (chain state):

| layer | target rms / kernel residual rms | t_rep term rms | all-distinct term rms (ages>=1) | residual on it: coef / R2 (G4-w coef / R2) | decisive corr | errG4: chain / kernel+t_rep / +all-distinct / +ages 1-4 only | decisive non-pair corr: kernel+t_rep -> +all-distinct | non-pair slope after |
|---|---|---|---|---|---|---|---|---|
| 4 | 2.10e-3 / 1.47e-3 | 4.8e-4 | 6.4e-4 | +1.01 / 0.19 (+1.02 / 0.18) | 0.45 | 0.124 / 0.101 / 0.095 / 0.095 | 0.746 -> 0.800 | 1.15 |
| 8 | 1.34e-3 / 9.0e-4 | 3.5e-4 | 4.1e-4 | +1.06 / 0.24 (+0.98 / 0.22) | 0.49 | 0.142 / 0.137 / 0.116 / 0.116 | 0.738 -> 0.808 | 1.14 |
| 12 | 1.03e-3 / 7.1e-4 | 2.3e-4 | 2.8e-4 | +1.19 / 0.22 (+1.40 / 0.24) | 0.43 | 0.193 / 0.186 / 0.157 / 0.161 | 0.739 -> 0.792 | 1.30 |
| 15 | 1.05e-3 / 7.1e-4 | 1.7e-4 | 2.3e-4 | +1.31 / 0.19 (+1.00 / 0.08) | 0.32 | 0.199 / 0.204 / 0.193 / 0.197 | 0.721 -> 0.752 | 1.55 |
| 16 | 0.99e-3 / 6.9e-4 | 1.6e-4 | 2.2e-4 | +1.04 / 0.11 (+0.95 / 0.08) | 0.46 | 0.210 / 0.220 / 0.197 / 0.201 | 0.692 -> 0.767 | 1.66 |

Per-age coefficients on the residual (joint fit) are 0.8-1.4 for ages 1-8 at layers 4-12 (theory 1); the term's rms by
age decays as 5.1e-4, 3.6e-4 (layer 4), 2.4, 1.9, 1.5, 1.1, 0.86, 0.57e-4 (layer 8, ages 1-6), 1.2, 1.0, 0.97, 0.78, 0.67,
0.52e-4 (layer 12). The joint fit of the target on [kernel, t_rep term, all-distinct term] moves the kernel's scale from
1.25 / 1.17 / 1.45 / 1.80 / 1.85 to 1.07 / 0.97 / 1.29 / 1.69 / 1.83 with the new term at 0.94 / 0.80 / 0.75 / 0.63 / 0.65
(unweighted R2 0.535 -> 0.628, 0.560 -> 0.637, 0.582 -> 0.634, 0.677 -> 0.703, 0.658 -> 0.683). So the term is the
identified missing physics at layers 4-12 and about a third of it at layers 15-16, where a ~1.5x deficit of unknown
origin remains (the kernel alone is WORSE than the chain's regeneration there, 0.204/0.220 vs 0.199/0.210; with the
term it is better, 0.193/0.197).

Cost (derived, Astra's ledger confirmed): with x = d1 o w the legs P^T x and A^T x are the products the chain already
forms to transport the sources (the transported leg is (A^T d(Phi) W)^T), and every block's other leg is a scaling or
thin combination of A and P (M = P s + 3 A e + Z L^T, X1 = 3A + Xt, Y1 = A w2 + Yt, X3 = A c1 + P c2, Y3 = y 1^T). The
new work is the products with s: A^T S and P^T S (n x n x m each) per source-layer plus thin work, i.e. 2 m/n u per
dense source-layer (Astra section 5: 54 u at m = 350 over the whole schedule with the confined ranks; ages 1-4 only, 15
layers: 120 m/n u = 41 u at m = 350, 60 u at m = 512; last 6 layers only: 16-24 u). Plus the kernel's own 3 m/n u per
layer (18-26 u at m = n; 6 u at m = 350 for 6 layers).

## 5. What the round-7 answers got right and wrong (question 2)

| answer / fragment | kernel of truth | what was mis-stated | status |
|---|---|---|---|
| Astra: fresh quartic response per ReLU | right: captures the multi-index pattern (corr 0.63-0.78) | its truncation drops the (1,1,1,1) K3 x C term, priced correctly in section 5 but deferred behind the wrong gate | alive, needs section 4's term |
| Astra: K3 x C directional query (Q8) | the missing term (section 4), same cost ledger (2 legs A^T S, P^T S per source-layer) | none - it was the right proposal, ranked last | alive |
| Astra: covariance patch, K4 x C terms, sampling, radial Rao-Blackwell | - | patch dead (D1); K4 x C untested but the (2,1,1) is not the sector; sampling dead (Q11) | dead |
| Sol: 3+2 with t_rep | right term, wrong sector: t_rep is (2,1,1)/(3,1) by construction; it still helps (corr +0.05) | "if the weighted scalar has <20% error use the slice form": the slice form cannot see the (1,1,1,1) sector at all | partly alive |
| Sol: nine (2,1,1) hypertree classes | the classes exist and are cheap (section 3c) | not where the pattern is; the kernel's polynomial already holds their shallow-layer content | dead |
| Sol: bivariate super-closure (Q13) | - | Q5: closure exact given inputs | dead |
| Terra: coherent mean-direction covariance error; local-birth factorization of K4(j,j,:,:) | both true: the D21 error is mu-directed too (section 2); K4(j,j,:,:) = the (2,1,1) classes | the coherent part never reaches the decisive neurons; (2,1,1) is not the sector | dead |
| Grok: pairwise kappa4, D21-scaled covariance term, 40x CV | - | measured dead (D5, Q11) | dead |
| Copilot: transported M211/M1111 matrices | memory (section 3b) | anti-correlated at depth | dead |
| Gemini: kappa3 x kappa3 covariance term; kappa3 = 0 memo; K3 x C query at 0.01 u | the K3 x C query is the term of section 4 | its 0.01 u factorization is wrong (Hadamard coupling, 2 m/n u per source-layer); the rest dead (D6) | dead / superseded |
| DeepSeek 7 / 7.1: mixture memo, sparse pairs, rank-R kappa4 CP, kappa3 x kappa3 (2,1) term | - | D6, D7 | dead |
| Fable R7: half families A/B, S = 1.25 | still right (-0.8% / -0.35%) | "S = 2 hurts because of over-propagation" stands; my round-8 hypothesis (doubled K x K) is dead | S = 1.25 queued |

## 6. End to end (chain runs; the mean closure only, kappa4 input replaced as in a7_kernel_e2e.py)

Configuration: k4_new = P_chain + (F0 - P_H) + K3C_rep + delta1111(ages 1-4, from the live source stacks in `_dslices`),
S = 1, A = 0, all layers, all neurons; paired against the plain chain (same harness: 2.2903e-8 on MLP #0, 1.9262e-8 on
the synthetic net) and against round 7's best kernel setting A = 0.5, S = 1.5 (2.1447e-8 = -6.4%; 1.8766e-8 = -2.6%).

Three chain runs (`r8_k4_dist_e2e.py`; `results/summaries/r8_k4_dist_e2e.json`, logs `results/logs_r8_k4_dist_e2e*.txt`):

| run | network | configuration (mean closure only) | final MSE | ratio to plain | reference without the new term |
|---|---|---|---|---|---|
| 1 | MLP #0 | A = 0, S = 1, + t_rep term, + all-distinct term (ages 1-4), all layers, all neurons | 2.2683e-8 | **0.990** | A = 0, S = 1.25, no terms: 0.952 (a7) |
| 2 | synthetic | same | 1.9081e-8 | **0.991** | A = 0, S = 1.25: 1.000 (a7) |
| 3 | MLP #0 | A = 0.5, S = 1.5, no t_rep term, + all-distinct term (ages 1-4) - round 7's best blend plus the term | 2.0893e-8 | **0.912** | same blend without the term: 0.936 (a7, -6.4%) |

Reading. The term is real end to end: in round 7's best setting it adds 2.4 points (-6.4% -> -8.8%, the best kappa4-channel
result so far; the kappa4 oracle is -30%). But the pure replacement with S = 1 (runs 1-2) is worth only -1% on both
networks, less than the S = 1.25 replacement without any term on MLP #0 (-4.8%): end to end the blend with the chain's own
regeneration (A = 0.5) and the kernel scale (S = 1.5) matter more than the offline G4-weighted error suggests, and the
kernel alone is worse than the chain at layers 15-16 (section 4). The live term was checked against the offline arrays
(`r8_dist_check.py`): identical (corr 1.000, slope 1.000 at layers 8 and 12, as are the live kernel and P_chain), and the
live G4-weighted errors of the run-3 combination are 0.120 / 0.166 at layers 8 / 12 (chain 0.142 / 0.193; blend without
the term 0.131 / 0.179; decisive non-pair correlation 0.63 -> 0.72 -> 0.77 and 0.69 -> 0.73 -> 0.77). The synthetic-net
version of run 3 was not run (run budget).

Cost of run 3 as it stands (classical products, m = all 1024 neurons, 14 layers): kernel 3 products/layer = 42 u; the
all-distinct term 2 products per source-layer x 4 sources = 112 u; total ~150 u (+58%) for -8.8%: score x1.44. Not
shippable. The score arithmetic at 260 u: an add-on of c units must cut the raw MSE by more than c/(260 + c): 14% for
43 u, 8% for 22 u, 4.4% for 12 u.

## 7. Recommended experiments (question 3), ranked by expected score gain per unit of compute

**1. The cheapest version of run 3 (kernel blend + all-distinct term), late layers, decisive neurons, youngest sources.**
State: nothing new (the term reads the live source legs A, P, Z, L, Zf and the chain's C_off, D21, D3; the kernel reads
C, g4, lam). FLOPs per layer at query width m (classical): kernel 3 m/n u; term 2 m/n u per source used (A^T S, P^T S;
the x-legs P^T x, A^T x are the transports). For layers 8-16 with |mu/sigma| < 1.5 (m ~ 420 on average, a7's DEC runs)
and the 2 youngest sources: kernel 3 x 9 x 0.41 = 11 u, term 2 x 2 x 9 x 0.41 = 15 u, total ~26 u (13 u if the products
go through the level-5 Strassen at 0.51). Falsifier already run on the stored arrays: ages 1-2 keep ~60-70% of the
term's G4-weighted gain (errG4 0.124 vs 0.116 at layer 8, 0.172 vs 0.161 at layer 12, chain 0.142 / 0.193). End-to-end
gate: run 3's harness with `A=0.5,S=1.5,KC=0,DIST=1,AGE=2,LMIN=8,DEC=1.5` on both networks; ship only if the raw MSE gain
exceeds 9% (26 u) or 5% (13 u). Expected from a7's restriction ladder (-3.5% at LMIN = 8, DEC = 1.5 without the term;
the term added 2.4 points at full width): -5 to -6%, i.e. neutral to slightly positive at Strassen cost, negative at
classical cost. This is the only experiment in the round-7 material with a plausible positive score outcome, and it is
small.

**2. Cut the kernel's own cost before adding the term.** The kernel's three products (C U, C T, Delta22 W^2) are the
same size as the term's two; at all layers and all neurons the kernel + term is 150 u. Any deployment must first
answer whether C U and C T can share the covariance transport (C W is formed for the variance: C U = C (d1 o W) is a
different right operand, so no) - if not, the kappa4 route stays a late-layer, decisive-neuron add-on of at most ~10-20 u.
No falsifier needed; a FLOP ledger on the shipped estimator settles it.

**3. Nothing else.** The variance channel has no lever left in the round-7 material: its fresh error is the kappa3
slice input (77%), the slice error is the all-distinct kappa3 content, and that error is unexplained past second
order in K with no coherent part the decisive neurons can see. The kappa3-diagonal channel (-14%) has the same source.
The remaining ~1.5x deficit of the kappa4 kernel at layers 15-16 after the all-distinct term is not memory, not the
(2,1,1) sector, not the vertex weights, and not a K x K term (the (1,1,1,1) K x K needs a per-query n^3 Gram; not
affordable): it is unattributed. Reviving any of the dead items (sampling, adjoints, low-rank, super-closure, mixture
memo) would need new evidence that none of the eleven answers provided.

What I would tell the lead in one line: the kappa4 kernel's missing physics is identified and measured (the
all-distinct K3 x C term from the chain's own sources, coefficient 1, +2.4 points end to end on top of the best blend),
but under this score it pays only as a ~13-26 u late-layer add-on worth ~5%; the leaders' 1.3x is not in the round-7
answers.

## 8. Files

- `experiments/probes/r8_kk_terms.py` -> `results/summaries/r8_kk_terms.json`, `results/logs_r8_kk_terms.txt` (section 2)
- `experiments/probes/r8_xy_chain.py` (prepared, not run: section 2)
- `experiments/probes/r8_k4_memory.py` -> `r8_k4_memory.json`, `data/dumps/r8_k4_memory.npz` (section 3b); `r8_k4_memory_e2e.py` (prepared, not run)
- `experiments/probes/r8_k4_211.py` -> `r8_k4_211.json` (section 3c)
- `experiments/probes/r8_k4_ngvertex.py` -> `r8_k4_ngvertex.json` (section 3d)
- `experiments/probes/r8_k4_1111.py` -> `r8_k4_1111.json`, `results/logs_r8_k4_1111.txt` (section 4)
- `experiments/probes/r8_k4_dist_e2e.py` -> `r8_k4_dist_e2e.json`, `results/logs_r8_k4_dist_e2e.txt` (section 6)
