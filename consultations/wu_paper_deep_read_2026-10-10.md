# Deep read of Wu et al. (arXiv 2605.05179v2) for our K=3 chain (10 Oct 2026)

Source: subagent report from the full text (68 pages). Each idea was checked against `submissions/v56As_fs/estimator.py`.

## Bottom line
The paper offers no source merging, truncation or compression. Its factor count grows as J(ℓ) = 6nℓ.

| method | paper's cost at n = 1024, L = 16 (Table 1, naive matmul) |
|---|---|
| factorised K=3 | ≈ 4,150 u |
| basic K=3 | ≈ 21,100 u |
| factorised augmented K=3 | ≈ 11,300 u |
| K=2 | ≈ 56 u |
| augmented K=2 | ≈ 88 u |

Our 194 u is already about 21× below the paper's own K=3 implementation. K=4 (factorised 24n⁴L²) is out of reach. The
conjectured error is c_K·(L/n)^K, so higher K only helps when n/L is large.

## Ideas and their status in V56

| # | idea (paper reference) | status in V56 | expected value |
|---|---|---|---|
| 1 | κ4 carried as a scalar T4 with recursion (S.4.1, S.4.3(a), eq. 20), cupped with M = WWᵀ (or C_off) | ours is memoryless g4 = λ C_off with an adaptive λ (BETA 1.2, tuned) | low: the level is already tuned and the κ4 pattern is the weak channel; A/B is cheap |
| 2 | next-order birth terms folded into existing legs (eqs. 18–20; k2 = 1 set free, one (2,1,1) term is a rescale of the centre coefficient) | partly present (the F68 feed); the paper's own term selection (eq. 14) is inconsistent, so the code is the authority | low to moderate; A/B needed |
| 3 | covariance-closure mixed terms at the error frontier (eq. 13 for λ = (1,1)): f3f1 κ4_iiij/6, ½f3f2 κ3_iij C_ij, ¼f3f3 κ4_iijj C_ij | unknown which are in our bivariate closure | ≈ 0 FLOP; risk of breaking error cancellations (§7.5 lesson) |
| 4 | merge factor groups sharing two legs (multilinearity of eq. 15) | done: the M block fuses P·s + 3A·e + Z Lᵀ | — |
| 5 | final layer: D3 diagonal only, no D21, no births | done: `trim = last` (V19) | — |
| 6 | symmetric-output congruence (App J): compute 3 of 4 blocks | done: `_sym_product` (V29, 2×2) | 4×4 blocks would save a further ~1 u |
| 7 | project old κ3 onto its rotation-invariant trace part h1 (Thm S.2.4) | not done | probably dies (the traceless part dominates D21); cheap falsifier |
| 8 | augmented K=2 at ≈ 88 u (under the floor) | not done | would need MSE ≤ 3.7e-8; theory says O(n⁻²), likely ≥ 10× short |

## Confirmations of what we already do
- factorised K=3 sources;
- power cumulants;
- the diagonal-slice corrections;
- univariate Edgeworth mean closure;
- a bivariate covariance closure that is a superset of the paper's Algorithm 2;
- closed-form ReLU Hermite coefficients.

The paper also states:
- dropping power cumulants or the odd-K trace gives O(1) MSE;
- the expansion is asymptotic, not convergent, so adding Hermite orders is not guaranteed to help;
- wall-clock is dominated by entrywise operations.

## Open problems the authors list
- Depth-independent error.
- Narrow networks.
- Non-Gaussian or correlated weights.
- Trained networks.
- Black-box inputs.

## Follow-up checks (10 Oct, evening)
1. **Carried κ4:** already present.
   - The diagonal is carried with memory: dG = WW·g_prev + var·λ, with the post-ReLU K4v/K22 core g_prev.
   - Only the off-diagonal core λ·C_off is regenerated, and its level is adapted per layer (BETA 1.2 against REF_R).
2. **Mixed closure terms:** already present.
   - The K3-simple TERM_SPECS tables (pruned at rel ≥ 1e-5) contain c_off×d21T, c_off×wk4m, d21×d21, d21×wk4m,
     g4row×d21, d3row×c_off, and so on.
   - This is a superset of the paper's k ≤ 3 frontier terms.
3. **Trace projection of old κ3:** killed.
   - Measured on captures L11/14/15: the old sources carry 35–57 % of |D21|.
   - Their trace projection Sym(h1 ⊗ I) leaves 74–75 % error in the old part, i.e. 26–42 % of the total D21 (tolerance
     2 %).
