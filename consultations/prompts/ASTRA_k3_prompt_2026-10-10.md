# Request (10 Oct 2026): find the first-order flaw in a carried third-cumulant recursion for a deep ReLU MLP

This is a new problem, separate from the C·T / ACDS thread. All numbers below were measured by us. You cannot run our
code, so please answer with derivations, explicit equations and FLOP counts. We will run every test ourselves.

## 1. Setting
**Problem.**
- Estimate the final-layer post-ReLU means of a bias-free ReLU MLP: width n = 1024, depth 16, He-initialised
  weights, input x ~ N(0, I), row convention z_{l+1} = h_l W[l].
- Score = final MSE × max(0.1, FLOPs/2^41). One u = 2^31 FLOPs, about one 1024^3 matmul.
- Our estimator costs about 190 u.

**The estimator** is a moment chain: the public factorised K = 3 cumulant propagation of Wu et al.
(arXiv:2605.05179, §4.3). Per layer it carries:
- the mean and the dense covariance C;
- the third cumulant kappa3, as a SUM OF SOURCES.

Each ReLU injects one source s, stored as:
- a propagator leg P_s (n×n, the identity at birth, transported P <- W^T diag(w1) P, where w1 is the per-neuron
  Wick / linear-response coefficient of the ReLU, about Phi(mu/sigma));
- a birth leg A_s = P_s diag(w1_b) C_off^(b), i.e. the row-scaled off-diagonal post-ReLU covariance at the birth
  layer b, exactly;
- thin feedback legs (n × 16 and n × 18).

kappa3 is a sum over sources of three symmetric blocks of legs (X, P, Y), roughly:
```
Sym(3A + thin, P, A*w2 + thin) + Sym(P*s + 3A*e + Z L^T, P, P) + Sym(A*c1 + P*c2, P, y 1^T)
```
with static per-source hub vectors w2, s, e, c1, c2.
- **Read-out.** The next ReLU needs D3_i = kappa3(i,i,i) and the slice D21_ic = kappa3(i,i,c).
- **Closures.** The mean uses a univariate Edgeworth closure in (mu, sigma, kappa3 diagonal, kappa4 diagonal);
  the covariance uses a bivariate closure.
- **Cost-saving tiers.** Sources older than 4–5 layers live in a shared rank-384/224 basis. kappa4 is not carried;
  it is regenerated each layer by a memoryless law.

## 2. Measurements

**(a) Value of a perfect kappa3.** Exact teacher kappa3 (slice D21 + diagonal d3), injected in a layer window,
final MSE ratio on a test network:

| window | ratio |
|---|---|
| all layers | 0.607 |
| L1–4 | 1.004 |
| L5–8 | 0.942 |
| L9–12 | 0.857 |
| L13–16 | 0.806 |

- Slice only: L1–8 0.937, L9–16 0.834. Diagonal only: L1–8 1.012, L9–16 0.853.
- Perfect kappa3 + kappa4 together: 0.272.
- Our best estimator is far from the top-10. A large part of this kappa3 gain is the lever we need.

**(b) The chain's kappa3 error vs a 6e7-sample teacher** (relative Frobenius norm):

| layer | D21 off-diagonal error | teacher noise (est.) | d3 error | d3 norm ratio chain/teacher |
|---|---|---|---|---|
| 2 | 0.044 | 0.053 | 0.033 | 1.002 |
| 4 | 0.042 | 0.026 | 0.029 | 0.998 |
| 8 | 0.049 | 0.012 | 0.031 | 0.996 |
| 12 | 0.059 | 0.007 | 0.038 | 0.993 |
| 16 | 0.069 | 0.005 | 0.046 | 0.986 |

The teacher noise uses a Gaussian approximation of Var[z_i^2 z_j] / N. Reading:
- The real error is about 3 % at L4 and grows steadily to about 7 % at L16.
- The d3 level shrinks systematically, by 1.4 % at L16.
- The slice has slope 1.00 and R² 0.996–0.998. The error is random-looking, not a gain error.

**(c) Where the error sits.**
- 91–96 % of the accumulated slice error comes from the all-distinct post-ReLU entries kappa3(h_i, h_j, h_k),
  which are themselves 9–16 % wrong.
- About 55 % of it is unexplained even at second order in the post-ReLU cumulant expansion: twelve K×K terms with
  RMS 3e-6 to 2e-5 against an error of 2.2e-4.
- It is not truncation: removing the old-source basis tier changes the slice error from 6.7 % to 6.6 %.

**(d) One exact step from samples.** The ReLU is replaced by its Hermite expansion of order K around the true
(mu, sigma), one layer, from exact samples. Relative error of the next D21 / d3:

| source layer | K = 2 | K = 3 | K = 4 |
|---|---|---|---|
| 4 | 0.20 / 0.21 | 0.066 / 0.071 | 0.033 / 0.033 |
| 8 | 0.155 / 0.163 | 0.046 / 0.050 | 0.024 / 0.025 |
| 12 | 0.123 / 0.126 | 0.027 / 0.028 | 0.016 / 0.017 |

**(e) Known partial fix.** Expanding the post-ReLU kappa3 to first order in C shows that the newborn source carries
exactly HALF of both K×C all-distinct families. Restoring them fully makes the final MSE worse (+3 %). Restoring a
quarter (scale 1.25) helps slightly (−0.8 %). This is an open paradox.

**(f) Dead ends.**
- Replacing the source chain by Gaussian Hermite-tree kappa3: closure error 10–20× worse.
- Feedback legs alone: ×1.001.
- Rank or age truncations follow a one-for-one cost/error law.

## 3. Questions
1. Which approximation in this recursion is first-order in the per-layer non-Gaussianity and would produce a slowly
   growing, random-looking 3–7 % slice error dominated by the all-distinct post-ReLU entries? Candidates we can think
   of:
   - kappa4 regenerated memorylessly, while its true value enters the next kappa3 through cross terms;
   - the K×C all-distinct families (the half-carried paradox in (e));
   - missing cubic-response (Hermite order 3) births.
   Rank them with an argument or derivation.
2. Give ONE correction for the top-ranked cause, with equations, shapes and a cost in u. Target: at most +30 u and
   at most about 3k extra elementwise/matmul operations. It may reuse the existing legs (A_s, P_s) of the young
   sources (age ≤ 5).
3. Give a cheap, decisive test of your ranking that we can run on stored per-layer teacher statistics (mean, C,
   kappa3 slice incl. diagonal, kappa4 diagonal, all 16 layers) and chain captures (all source legs at layers
   3, 7, 11, 14, 15; mu, var, D3, D21, C_off at every layer).
4. On the d3 level drift (−1.4 % at L16): is a level renormalisation of d3 principled here? If so, what quantity
   should set it?

Mark each statement as derived, estimated or speculative. Do not report numbers as measured unless you ran them.
