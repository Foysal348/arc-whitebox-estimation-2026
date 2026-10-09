# Reply to your RLL proposal: measured, and killed. Next question below.

We ran your residual-projection test on all 15 transitions (L2->3 ... L15->16). Teacher inputs at layer l and the true
W_l give dD21 at l+1. We compared it with the plain chain's residual E = D21_teacher(l+1) - D21_chain(l+1).

## 1. Your formula has one wrong term
The C_jk T_iii term in kappa(R_i, L_j, L_k) comes from the partition {i,i,i}{j,k}. That partition is NOT connected
(the i-block never touches j or k), so it does not belong in the joint cumulant. The same holds for the
2 diag(R^T d) G and d (Q^T C Q)_pp pieces of your slice formula. The connected first-order part is
`r_i l_j l_k [3 C_ij T_iik + 3 C_ik T_iij]`, which gives
`dD21 = 6 (R o B)^T E + 6 (R o E)^T B + 6 (B o E)^T R`.
We tested both versions.

## 2. Result (measured)

| quantity | value |
|---|---|
| size of dD21, relative to the teacher slice | 4-9 % (comparable to the 4-7 % residual) |
| corrected version, off-diagonal: rho | -0.044 to +0.074 |
| corrected version, off-diagonal: beta | -0.042 to +0.075 |
| corrected version: residual after adding dD/2 | 15-42 % WORSE (r_half 1.15-1.42) |
| your version, off-diagonal: rho / beta | -0.06 to +0.02 / ~0 |
| your version: residual after adding dD/2 | 23-89 % worse |
| diagonal (d3), corrected version, L6-16: rho | -0.09 to -0.24 (negative) |
| diagonal (d3), corrected version, L6-16: beta | -0.03 to -0.16 |

Reading:
- The residual has no component along RLL, so the chain already carries the full RLL term.
- The negative diagonal projection suggests the chain's d3 slightly over-counts it.

## 3. Next question
With RLL excluded, which first-order mechanism remains? We still have from your ranking:
- QQL C x T: it needs the all-distinct T_ijk, which we do not have dense, but the chain has it in factorised form;
- the memoryless K_211 kappa4 sector.

For each, please give:
1. a residual-projection test computable from teacher C, Y21 (slice incl. diag), K4d (diagonal only), mean, and W,
   or from chain captures (source legs at L3/7/11/14/15);
2. the expected sign and size of beta if the mechanism is the cause.

Also: is a negative d3 projection on RLL consistent with the observed -1.4 % d3 level shrink at L16? If so, what
does it imply?
