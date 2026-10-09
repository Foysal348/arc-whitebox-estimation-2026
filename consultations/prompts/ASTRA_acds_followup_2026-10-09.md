# Follow-up on ACDS-64 (Age-Collapsed Dual Sketch): what we found on our side, and three questions

Thank you for ACDS. We checked it against our code before running it. The algebra is sound. Your script cannot run
unchanged, and the cost changes because of how our kappa3 sources are actually represented. Details follow.

## 1. Why your script cannot run as written
- `TripletAdapter` imports `fable_component_2026-10-09.py` as a module. That file is a script: argparse and the main
  loop run at import, and the SystemExit escapes your `except Exception`.
- The fallback `blocks_of_source(c, s, rres, rfb)` needs `rres` and `rfb` (scalars stored in the capture), so a
  `(cap, s)` call fails.
- Transport: the chain moves legs with the chain's own Wick coefficient w1 = W_all[:, 4], not with Phi(alpha):
  leg(l+1) = (diag(w1_l) W_l)^T leg(l). We verified this to a relative error of 1e-7 on P_st. The hub path is
  Ybar = W_m diag(w1_{m+1}) W_{m+1} ... (no w1 at the vertex), and X = (C_off + diag var)(m) Ydot.
- Captures: the dense source stacks exist only at layers 3, 7, 11, 14 and 15 (2, 6, 10, 13 and 14 sources).
  Layers 1-2, 4-6, 8-10, 12-13 and 16 hold only mu, var, D3, D21, C_off, W_all, ... . Your loader expects stacks at
  every layer.
We re-implemented ACDS in our conventions (experiments/probes/acds_ct_probe.py). It checks the age-collapse identity
against the per-age route and measures sketch fidelity with a Rademacher range finder C Omega and with top-r
eigenvectors of C, at r = 32, 64, 128 and 256. A, P and Z legs transport exactly (relative error 3e-7 from L14 to
L15); L_st, R1T/R2T and the per-source scalars are static. So legs at non-stack layers are obtained by transport from
the latest earlier stack layer (consistent source subset).

First measurement (small local check, target L8, window 6-7, 6 sources, 54 source-role terms):
- Age-collapse identity: relative error 4.7e-7 against the per-age route. The identity is exact.
- Sketch fidelity, corr vs the chain's exact C.T on decisive neurons:

  | sketch | corr |
  |---|---|
  | Rademacher C Omega, r = 32 | 0.679 |
  | top-32 eigenvectors of C | 0.815 |

- Affine teacher-metric corr: chain alone 0.365; with exact chain C.T 0.722; rad32 0.539; eig32 0.597.

So the sketch, not the algebra, is the open problem. The full run (L12/15/16, r up to 256) is on Kaggle now.

## 2. The width factor is 3, not 1, but it seems to fold back
Each source has three blocks (legs are [neuron, hub], n = 1024 on both sides; static per-source hub vectors):
```
M  = P*s_b + 3 A*e_b + Z L^T                     Z: [n,18] neuron leg, L: [n,18] hub side
X1 = 3A + Zf[:, :rfb] R1T^T     Y1 = A*w2b + Zf[:, rfb:] R2T^T     (Zf: [n,32], R1T/R2T: [n,16] hub side)
X3 = A*c1 + P*c2                Y3 = y 1^T  (y = Z[:, rres+1])
blocks: Sym(X1, P, Y1) + Sym(M, P, P) + Sym(X3, P, Y3)
```
That gives 9 (anchor, H) placements per source. Our worst case is about 146 u, not 48.75 u.
However, every anchor leg is (dense A or dense P) times a hub-diagonal, plus low-rank terms (Z L^T, Zf R^T, y 1^T).
Group the placements by dense anchor:
```
sum_roles (H_role o w_role[None, :]) (A^T S~) = H_A B_A,   with H_A = sum over A-anchored roles of H_role o w_role
```
and likewise H_P for P. Per (source, target) this needs only 2 thin GEMMs for B (A^T S~ and P^T S~) and 2 for H B.
The low-rank parts are O(n r 18). That gives about 13 x 10 x 4 x 0.0625 = 32.5 u, plus your shared ~8 u, so about
40 u in total. Please check this grouping. The H matrices are formed at the target and mix legs at t (X1(t), P(t), ...).
We believe they stay elementwise, so the grouping is exact, but we want a second pair of eyes.

## 3. Deployment differences
- The deployed chain (V56) keeps dense young legs only for ages <= AGE_OLD = 5. Older sources live in a SHARED old
  basis (one common [n, r_old] leg plus per-source small factors). Anchors older than t-6 are therefore already
  low-rank in the deployed chain. Does ACDS simplify for them, for example A_old^T Gamma = U_old^T Gamma with one
  shared U_old?
- Grader constraints: 120 s wall time; the residual time cap means at most ~18k metered ops in total; 8 GB;
  2-D ops only (in-place batched 3-D broadcasts are rejected by the grader smoke test).
- The whole estimator uses ~190 u of the 1024 u budget, and the score scales with max(0.1, FLOPs/2^41). Every extra
  u costs score unless the MSE gain pays for it. C.T is worth up to 0.773x MSE on MLP #0 with an oracle map.

## 4. Questions
1. Is the anchor grouping in section 2 exact, including the Sym(M, P, P) block, where P appears twice and M contains
   both P and A?
2. Sketch choice. S_{t-6} = C_{t-6} Omega with no power iteration may lose mass. Which costs less for equal fidelity:
   one power step (C^2 Omega), a top-r eigenbasis (eigh is O(n^3) but needed once per window), or a sketch built from
   the anchors themselves (e.g. A_s(a) Omega pooled over sources)? Give the cost in u for each.
3. A plain range finder loses too much at r = 32. Can the sketch be made target-aware at low cost? For example,
   sketch the column space of the anchors (the A and P legs at a, pooled over sources) instead of C. Or avoid the
   projection entirely: Gamma_a = sum_m Ydot_{a->m} G_m has its own structure (each G_m is g o Ybar o (C Ydot)), so
   a factorisation of Gamma might be cheaper than projecting it.

Please answer with formulas and FLOP/op counts. Do not report fidelity numbers as measured unless you ran them; we
will run the measurements on our side.
