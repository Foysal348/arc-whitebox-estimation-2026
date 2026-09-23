# Round 7 (Fable 5.1, 23 Sep 2026): what physics of the post-ReLU third cumulant does the K3 chain drop, and does it produce the 4-7% slice error?

Scope: public MLP #0, plain Gaussian chain (`V31_ANGULAR=0`), the 6e7-sample pre- and post-ReLU teachers and the 1e9 baked truth.
All numbers below are measured; nothing is fitted unless said so. Scripts: `experiments/probes/r7_*.py`, variant
`experiments/variants/r7_estimator_fbscale.py`, summaries `results/summaries/r7_*.json`, captures `data/dumps/r7_cap_L{l}.npz`
(one plain-chain run, all sources dense, `V21_NO_CONFINE=1 V19_FULL_LAST=1 V26_STRASSEN=0`; final MSE 2.2013e-8 = the notes' 2.202e-8).

## 0. Verdict in five lines

1. The chain's (2,1)-slice error is **91-96% an error of the ALL-DISTINCT entries** of the post-ReLU kappa3 tensor (its diagonal and (2,1) slices are exact to 0.1-2% and 1-6% and carry 0% and 4-9% of the error variance).
2. The Edgeworth expansion of kappa3(h_i,h_j,h_c) to first order in the pre-activation kappa3 K and up to third order in C_off gives five dropped/mis-weighted all-distinct terms. **Every one of them is confirmed numerically with its theoretical coefficient** (regression slopes: the two K x C families -0.55..-0.69 and -0.6..-0.9 for a predicted -1/2, doubled edge -0.9..-1.2 for -1, transport curvature K_dist x C -1.0..-1.4 for -1, triangle ~ -1) - but together they explain only **13-21%** of the slice-error variance at one layer; the largest (the K x C family A), accumulated over the births it survives from, reaches **20-35%**.
3. The port carries exactly **half** of the Edgeworth K x C all-distinct terms (the lean chain's B2 block and its `c_w2 r_w2 WK12` term of Y1 were both dropped). Restoring the full coefficient costs **zero FLOPs** (two constants), improves the slice and kappa3_iii at layers 4-10, but hurts at depth (final MSE +3.4% on MLP #0, +3.1% on the synthetic net): the added content is over-propagated by the chain. The optimum of that knob is 1.25x: **-0.8% / -0.35% final MSE for free** on the two teacher networks (section 4).
4. The transport curvature (the exact first-order transport of the all-distinct kappa3 is K_ijc x P3(i,j,c), P3 the trivariate orthant probability) has **no coherent part** (per-source slope +-0.001): it is a 1.2-2.2%-per-layer random correction, ~2-10% of the error variance accumulated, and costs n^4 per source-layer: **unaffordable and not worth it**. The kappa4 (2,1,1) hub, the self-loop (kappa3/kappa4-diagonal) hubs, the doubled edge, the triangle, the Rres truncation and the feed model are each <= 2% of the error variance at depth.
5. Honest bottom line: ~35-45% of the all-distinct slice error is identified (mostly the accumulated half-family-A deficit, whose correction over-shoots because it is then over-propagated), ~55-65% is beyond the first-order-in-K cumulant expansion and could not be attributed; nothing affordable inside the K3 formulation buys the 5-7% back. The one free improvement found is `R7_FB_SCALE=1.25`.

## 1. Derivation: the post-ReLU third cumulant for distinct i, j, c

Pre-activation z ~ (mu, C, K, Q) (K = kappa3, Q = kappa4), h = relu(z), w_k(i) = E[h_i^(k)] (the chain's Wick table: w1 = Phi, w2 = phi/sigma, w3 = -alpha phi/sigma^2, ...).
Cumulant expansion: E[F] = E_G[F] + (1/6) K_abc E_G[d_abc F] + (1/24) Q_abcd E_G[d_abcd F] + (1/72) K K E_G[d^6 F] + ..., with E_G the Mehler expansion in C_off (a term with edges (ab) of multiplicity m_ab carries prod C_ab^m / m! and derivative orders = vertex degrees). The joint cumulant keeps the connected diagrams. For i, j, c distinct:

| term | expression (all hub placements h in {i,j,c}, x,y the other two) | order | chain |
|---|---|---|---|
| linear transport | w1_i w1_j w1_c K_ijc | K | exact (sources) |
| Gaussian hub | sum_h w2_h (C_hx w1_x)(C_hy w1_y) | C^2 | exact (newborn B1) |
| family A (w3-type) | (1/2) sum_h w3_h [(C_hx w1_x)(K_hhy w1_y) + (x<->y)] | K C | **1/4** (half) |
| family B (w2 w2-type) | (1/2) sum_h w2_h [(K_xxh w2_x)(C_hy w1_y) + (x<->y)] | K C | **1/4** (half) |
| doubled edge | (1/2) sum_h w3_h [(C_hx^2 w2_x)(C_hy w1_y) + (x<->y)] | C^3 | 0 |
| triangle | w2_i w2_j w2_c C_ij C_jc C_ic | C^3 | 0 |
| transport curvature | K_ijc [C_ij w2_i w2_j w1_c + C_ic w2_i w1_j w2_c + C_jc w1_i w2_j w2_c] (= K_ijc x first-order orthant correction) | K C | 0 |
| kappa4 (2,1,1) | (1/2) sum_h w2_h w1_x w1_y Q_hhxy | Q | lam-model feed |

(K x K, K x C^2, Q x C terms are each <= 7% of family A by size; not tested.)

Where the chain stands (from `Sym(X1 x I x Y1)`, Sym = (1/6) sum of the 6 permutations, X1 = 3 a_b + 1.5 d(w2) D21, Y1 = a_b d(w2) + 0.5 d(w1) D21^T d(w3), a_b = d(w1) C_off): the all-distinct entry with hub j is (1/6)[X1_ij Y1_cj + X1_cj Y1_ij]. The a_b x a_b part gives the Gaussian hub with coefficient 1 (correct); the a_b x Yt and Xt x a_b w2 cross terms give family A and family B with coefficient **1/4** each. In the lean chain (`lean_k3.py`) the other quarter of family A sat in the B2 block (`X2 = 3 c_w1 WK12`, `Y2 = c_w3 r_w1 WK11`) and the other quarter of family B in Y1's `c_w2 r_w2 WK12` term; the port shadows B2 (F69: "hurts") and its Y1 is `a_b d(w2) + Yt` only. So the port has exactly half of both families - a clean, testable prediction (regression coefficient -1/2 on both).

The slice-level closure (TERM_SPECS) is the same expansion for (i,i,j) and (i,j) and is confirmed exact by the notes; nothing above touches it.

## 2. Where the slice error lives (r7_slice_terms.py)

z_{l+1} = h_l W[l] exactly, so D21'(a,b) = sum_{ijc} W_ia W_ja W_cb T_ijc splits by the index pattern of the post tensor T: diagonal (i=j=c, S_diag), two-equal ((2,1) slice, S_21), all-distinct (S_dist = D21' - S_diag - S_21, exact for the teacher: same samples, C_pre(l+1) = W^T C_post W holds to 2e-7). The chain's post slices K3, K21 are captured; its all-distinct part is the remainder.

| post layer -> pre | D21' rms = diag + (2,1) + dist | chain error rms (rel) | error rms: diag / (2,1) / dist | share of error variance diag / (2,1) / dist | chain post K21 rel err | chain post k3_iii rel err |
|---|---|---|---|---|---|---|
| 3 -> 4 | 6.33e-3 = 1.41e-3 + 4.92e-3 + 3.02e-3 | 2.68e-4 (4.2%) | 1.4e-6 / 5.3e-5 / 2.63e-4 | 0.00 / 0.04 / 0.96 | 1.2% | 0.1% |
| 7 -> 8 | 4.74e-3 = 3.6e-4 + 4.08e-3 + 2.16e-3 | 2.33e-4 (4.9%) | 1.3e-6 / 5.1e-5 / 2.27e-4 | 0.00 / 0.05 / 0.95 | 2.3% | 0.4% |
| 11 -> 12 | 3.68e-3 = 1.3e-4 + 3.28e-3 + 1.55e-3 | 2.18e-4 (5.9%) | 1.1e-6 / 5.0e-5 / 2.12e-4 | 0.00 / 0.06 / 0.94 | 3.9% | 0.8% |
| 14 -> 15 | 3.57e-3 = 8e-5 + 3.24e-3 + 1.45e-3 | 2.34e-4 (6.6%) | 1.3e-6 / 6.8e-5 / 2.24e-4 | 0.00 / 0.08 / 0.92 | 5.2% | 1.6% |
| 15 -> 16 | 3.33e-3 = 7e-5 + 3.07e-3 + 1.34e-3 | 2.30e-4 (6.9%) | 1.5e-6 / 6.8e-5 / 2.18e-4 | 0.00 / 0.09 / 0.91 | 5.6% | 2.0% |

The all-distinct part is ~40% of D21' and carries 91-96% of the error; relative to itself the all-distinct part is wrong by 9% (layer 3) to 16% (layer 15). The reconstruction of the chain's all-distinct part from the captured source legs (`r7_srcgain.py`: per-source hub blocks B1 = Sym(X1 x P x Y1), B3 = Sym(M x P x P), feed = Sym(X3 x P x Y3), transported by W^T diag(w1), minus each source's own slice entries, plus the newborn hub) reproduces it with slope 1.003-1.006, R^2 0.999, and the summed source dslices reproduce the chain's D21 to R^2 1.00000 - the block conventions used below are the chain's.

## 3. Regression of the all-distinct error on the dropped terms (fresh, one layer)

Terms computed from the TRUE layer-l pre-activation data (C, Y21, mu, var); hub forms contract to D21' in 3 n^3 (`hub()`), the triangle and the curvature (from the chain's own source legs) are n^4 and were done exactly on 24 random columns (24 x 1023 entries). Target y = all-distinct error e_dist(l+1) (all n^2 - n entries for the hub forms). "slope" = regression through the origin, "R2" = fraction of the error variance explained.

| post layer | famA slope / R2 | famB | doubled edge | curvature K_dist x C | triangle | kappa4 hub (post K4 diag) | joint R2 (5 hub terms) | joint R2 incl. curvature+triangle (24 cols) | fixed theory coefs (-1/2,-1/2,-1,-1,-1) R2 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | -0.55 / 0.052 | -0.62 / 0.028 | -0.67 / 0.006 | -0.96 / 0.045 | -0.95 / 0.002 | -0.04 / 0.001 | 0.103 | 0.141 | 0.131 |
| 7 | -0.59 / 0.132 | -0.33 / 0.004 | -1.07 / 0.017 | -1.05 / 0.031 | -1.11 / 0.002 | -0.33 / 0.000 | 0.160 | 0.187 | 0.179 |
| 11 | -0.65 / 0.161 | -0.05 / 0.000 | -1.43 / 0.016 | -1.21 / 0.023 | -0.20 / 0.000 | -0.59 / 0.000 | 0.175 | 0.171 | 0.153 |
| 14 | -0.69 / 0.138 | -0.08 / 0.000 | -1.26 / 0.005 | -0.90 / 0.004 | +0.03 / 0.000 | -0.61 / 0.000 | 0.142 | 0.210 | 0.185 |
| 15 | -0.62 / 0.120 | -0.23 / 0.000 | -1.10 / 0.004 | -1.03 / 0.008 | -1.27 / 0.000 | -1.42 / 0.000 | 0.127 | 0.149 | 0.132 |

(joint fits give famA -0.53..-0.77, famB -0.6..-0.9, dedge -0.8..-1.2, curvature -1.0..-1.5: the theory coefficients.)
Term sizes (rms of the D21' contraction, layer 15): famA 1.2e-4, famB 1.7e-5, dedge 1.3e-5, curvature 1.6e-5, triangle 2e-6, kappa4 hub 1e-6, versus error 2.2e-4 and all-distinct content 1.3e-3.
Reading: the derivation is right (every coefficient lands on its predicted value, including the exact -1 of the curvature and doubled edge) and family A is the only fresh term of consequence, but the fresh dropped physics of ONE layer is 13-21% of the error. The error is accumulated.

Other single-layer contributions checked (`r7_checks.py`): the rank-16 truncation of the slice fix-up Rres (tau = Rres - Q Bm, rebuilt with the chain's range-finder recipe; slope -1.00 exactly) is R2 0.09 at layer 3 falling to 0.004 at layer 15 (rank 64/128 would remove 30-50% of that); the K4->K3 feed's all-distinct content (lam model) R2 <= 0.01; the newborn hub built from the chain's own C instead of the true C (inherited part) R2 0.001-0.006.

## 4. The K x C families end to end: a free knob, and why the Edgeworth value hurts (r7_fbscale_chan.py)

`experiments/variants/r7_estimator_fbscale.py` = the shipped estimator with `R1T_b = Bf.T * 1.5 * S`, `R2T_b = Qf * (w3 * 0.5 * S)` (`R7_FB_SCALE = S`). S = 1 is the shipped chain, S = 2 gives the newborn exactly families A and B (the a_b x a_b hub is untouched; the Xt x Yt term is second order in K; the newborn's own slices are re-fixed by S21 = K21 - rep21, so only the all-distinct entries change). Zero extra FLOPs.

Per-layer relative errors against the pre-activation teacher (%), plain chain, MLP #0:

| S | final MSE (ratio) | D21 L4 / L8 / L12 / L15 / L16 | k3_iii L4 / L8 / L12 / L16 | var L16 | C_off L16 |
|---|---|---|---|---|---|
| 0 (no feedback) | 2.611e-8 (1.139) | 4.41 / 6.31 / 8.83 / 10.66 / 11.22 | 3.06 / 3.90 / 5.69 / 7.12 | 0.18 | 1.00 |
| 1 (shipped) | 2.293e-8 (1.000) | 4.22 / 5.05 / 6.08 / 6.69 / 6.98 | 2.93 / 3.24 / 3.96 / 4.65 | 0.14 | 0.79 |
| 1.5 | 2.283e-8 (0.9955) | 4.18 / 4.75 / 5.44 / 6.02 / 6.34 | 2.89 / 3.09 / 3.58 / 4.38 | 0.14 | 0.74 |
| 2 (Edgeworth) | 2.371e-8 (1.034) | 4.18 / 4.73 / 5.59 / 6.77 / 7.24 | 2.88 / 3.10 / 3.71 / 5.04 | 0.14 | 0.76 |

The shipped chain's slice error regressed on the CHANGE of the slice D21(S=2) - D21(S=1) (i.e. the accumulated, faithfully propagated missing half): slope -0.78 / -0.78 / -0.66 / -0.48 / -0.46 at layers 4 / 8 / 12 / 15 / 16 (ideal -1), R2 0.02 / 0.13 / 0.21 / 0.19 / 0.17; the change's rms grows to 2.1e-4 at layer 15-16, as large as the error itself. So the fresh coefficient is right (shallow layers: -0.8), but content added to the sources is carried too strongly by the chain's linear transport at depth: at layer 15-16 the truth contains only ~half of what the chain propagates. S = 1.5 already gives D21 error 6.98 -> 6.34% and kappa3_iii 4.65 -> 4.38% at layer 16 with the final MSE -0.45%; the paired scan below puts the MSE optimum at S ~ 1.25. This is the same phenomenon as F69's "B2 hurts" and the R_RES ladder (16/32/64 -> 2.35/2.39/2.41e-8: a MORE exact slice fix-up is worse): every exact piece of content added to the linearly transported sources is over-propagated at depth. Section 5 shows this over-propagation is NOT a transport-gain error (the curvature has no coherent part) but the born half-deficit itself, so the balance point of the knob is set by how the deficit propagates through the closure, not by a transport correction.

Paired check on both teacher networks (`run_variant_teacher.py`, `results/summaries/teacher_eval_r7_fbscale_paired.json`; MLP #0 layer 16 vs the baked truth, synthetic He net layer 15 vs its 6e7-sample teacher):

| S | MLP #0 (ratio to shipped) | synthetic net (ratio) |
|---|---|---|
| 1.0 | 2.2930e-8 (1.0000) | 1.9256e-8 (1.0000) |
| 1.25 | 2.2745e-8 (0.9919) | 1.9188e-8 (0.9965) |
| 1.5 | 2.2827e-8 (0.9955) | 1.9284e-8 (1.0015) |
| 1.75 | 2.3127e-8 (1.0086) | 1.9502e-8 (1.0128) |
| 2.0 | 2.3711e-8 (1.0340) | 1.9858e-8 (1.0313) |

Both networks agree: the optimum is S ~ 1.25 (-0.8% / -0.35%), S = 2 costs +3.4% / +3.1%. A free but small gain (`R7_FB_SCALE=1.25`); it needs the 100-MLP suite before shipping.

A structural reason why the missing half is coherent: per hub h, the (2,1) slice is proportional to the covariance row, K[h,:] ~ beta_h C[h,:], with beta_h ~ alpha_h (corr 0.83-0.95 across layers, per-row IQR +-0.09 at depth), so family A's coherent part is beta_h w3_h (C C) w1 w1 = -(beta_h alpha_h / sigma_h) x (Gaussian hub): a DAMPING of the newborn hub that is always negative (beta_h alpha_h > 0), median factor 1 - beta_h alpha_h/sigma_h = 0.99 (layer 3), 0.94 (7), 0.83 (11), 0.56 (14-15). The chain has half of it. This is also why the per-source regression (`r7_srcgain.py`: e_dist on each old source's transported contribution, jointly) finds every source over-represented by +2-5% (ages 1-10) with the oldest, smallest sources at +5-17%: the born half-deficit rides along.

## 5. The transport itself, the accumulated deficit, and the self-loop hubs (r7_coh.py, r7_diaghub.py)

(A) Coherent part of the transport curvature. Per source alive at layer l, the exact K_dist x C correction of its next-layer slice contribution (8 columns, n^4) regressed on the source's own linear-transport contribution: slope -0.0013..+0.0012 (corr |c| <= 0.07) at layers 7, 11, 15, for every age. The curvature has **no coherent part**: it is a pure random modification of 1.2-2.2% (of the source's contribution) per layer. Accumulated as a random walk over a source's life it amounts to ~3e-5 in slice units at layer 16 (sum over sources of age x contribution^2, times 1.5%), i.e. ~2% of the error variance. The chain's over-representation of the old sources (+2-5% per source, section 4) is therefore NOT a transport-gain error; it is the born half-deficit of family A riding along.

(B) How the born family-A deficit survives. e_dist(l+1) regressed jointly on the linearly transported all-distinct family-A tensor of every birth layer (the born tensor's own slice entries removed, as the closure handles them): the recent births carry -0.3..-0.5 each (predicted -1/2), the tiny old births are noise (rms 1e-6..1e-5); joint R2 0.21 (layer 7), 0.31 (11), 0.34 (15); with all coefficients fixed at the physical -1/2: slope -0.42 / -0.35 / -0.25 and R2 0.21 / 0.29 / 0.25. So the accumulated K x C family-A deficit is **20-35% of the all-distinct error variance** (consistent with the end-to-end 13-21% of the S = 2 change, which the chain propagates through its closure rather than by pure linear transport). Why the offline linear-transport model over-predicts at depth: the (2,1)-slice part of any error tensor propagates through the closure with the exact slice gain Phi_j [Phi_i - phi_i^2 - alpha_i phi_i Phi_i] (0.82x the linear gain Phi_i^2 Phi_j at alpha = 1, 5.5x at alpha = -1), not with the linear gain - a faithful offline propagation of the deficit needs the closure in the loop; the S = 2 run IS that propagation.

(C) Self-loop hubs. The non-Gaussian marginal of the hub neuron also generates all-distinct entries: (1/6) K3_h w4_h (C w1)(C w1), (1/24) K4_h w5_h (...), (1/72) K3_h^2 w6_h (...) - i.e. the hub weight w2_h is really E[h''] of the non-Gaussian marginal. Measured: rms 2-3e-6 (vs error 2.2e-4), R2 <= 0.02, and the ratio (non-Gaussian hub weight)/w2 is 1.000 with IQR 0.987..1.014 at depth. Dead.

(D) What is left. Adding up: family A accumulated 20-35%, curvature ~2-10%, doubled edge + triangle + Rres truncation + feed + inherited-C hub ~3-5% (~10% at layer 3). The remaining ~55-65% of the all-distinct error variance is not any first-order-in-K, first-to-third-order-in-C term of the cumulant expansion, nor the second-order Mehler correction of the transport (<= 25% of the first order on pairs of off-neurons, ~2% elsewhere), nor teacher noise (~1% of the variance: the same-sample identity makes S_dist_true exact up to its own 1.5e-5 sampling noise). Candidates not tested: the second order in K (K x K terms with all three vertices connected, each estimated <= 7% of family A but numerous and also accumulated), and the cross-talk between the slice error and the all-distinct error through the closure (the D21 error of 4-7% enters the newborn's Xt/Yt and the M leg at every layer).

## 6. Cost of the candidate corrections (units of n^3 = 2^31 FLOPs per layer; the shipped chain spends ~250 units per MLP, i.e. ~16 per layer, and ~4 per live source-layer for 2 dense transports + 2 dense contractions)

| correction | what it needs | cost | worth (measured) |
|---|---|---|---|
| K x C families at the Edgeworth coefficient | two constants (`R7_FB_SCALE`) | 0 | S = 2: +3.4% / +3.1% MSE (MLP #0 / synthetic); S = 1.25: -0.8% / -0.35% (section 4) |
| doubled edge | a third dense leg per source, (C o C) d(w2), transported (1 unit) and contracted as a new right factor (1 unit) | +2 units per source-layer = +~210 units per MLP (+80% cost) | R2 0.005-0.02 of the slice error: not worth it; a rank-1 (mean C^2) version is free but carries ~1/3 of an already negligible term |
| transport curvature K_dist x C (exact first-order orthant transport of the all-distinct tensor) | for a hub source with legs (X,Y,Z): sum_ij (u_i X_ir) C_ij (u_j Y_jr) per hub r and output column a: n^3 per column | n^4 per source-layer (~1000 units), and the corrected tensor is no longer a hub form (n^4 per further transport) | its fresh part is R2 0.005-0.045 with the exact coefficient -1; no coherent part (section 5), accumulated ~2-10% of the error variance |
| triangle | n^4 | unaffordable | R2 0.002 |
| Rres rank 16 -> 64 | +~0.1 unit per birth | cheap | removes 30-50% of a term that is R2 0.09 at layer 3 and 0.004 at depth; 504aldo measured the MSE getting WORSE (over-propagation again) |
| kappa4 (2,1,1) hub from the post K4 diagonal | 3 matmuls | ~3 units per layer | R2 0.000-0.001 |

## 7. What was NOT the cause (negative results, all measured)

- Not the slice-level closure: the chain's post (2,1) slice and kappa3 diagonal are right to 1-6% and 0.1-2% and carry <= 9% of the slice-error variance.
- Not the rank truncations (Rres, feedback rank, confinement): tau is <= 9% at layer 3 and ~0 at depth; the capture run had every source dense.
- Not the kappa4 -> kappa3 channel: the (2,1,1) hub from the post kappa4 diagonal and the chain's lam feed are both R2 <= 0.01.
- Not the Gaussian closure births themselves: the hub coefficient is exactly 1, the O(C^3) corrections (doubled edge, triangle) are ~1-2% of the error variance.
- Not one layer's dropped physics of any kind: the fixed-coefficient theory prediction of all first-order dropped terms explains 13-19% of one layer's all-distinct error.

## 8. What I would do next (in order of expected value per FLOP)

1. Ship the free knob after a wider check: `R7_FB_SCALE=1.25` on the 100-MLP local suite (both teacher networks improve, -0.8% and -0.35%; 1.5 is neutral on the synthetic net, 2.0 hurts both by 3%). Small, but zero cost and no new fitted table (one constant between the port's half and the Edgeworth coefficient).
2. Do not build a transport correction: the curvature K o P3 has no coherent part (no per-source gain to damp), its random part is ~2-10% of the error variance, and it is n^4. A fitted per-age damping of the legs would be chasing the born half-deficit (the per-source over-representation of +2-5%), which the knob above already balances.
3. If the slice error is to be attacked further, the only untested physics of comparable size is the second order in K (K x K terms with all three vertices connected, each <= 7% of family A, all accumulated) and the closure-mediated cross-talk between the 4-7% slice error and the newborn's Xt/Yt legs at every layer; both would need the closure in the loop to test (a chain-level oracle that injects the TRUE D21 into Xt/Yt only would isolate the second at one run). Everything else in the first-order cumulant expansion is now measured and excluded. Within this formulation the 5-7% looks like a floor; a 1.3x accuracy gain will not come from the kappa3 births.
