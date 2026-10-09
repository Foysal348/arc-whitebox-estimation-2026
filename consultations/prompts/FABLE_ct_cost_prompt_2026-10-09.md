# Fable follow-up (9 Oct 2026): make C.T deployable inside the grader budget

## What your LKV design achieved (all measured, public MLP #0)
- Ceiling (2e6 samples, ages 1-6): star + C.T pattern corr 0.846 / 0.835 / 0.824 at L12 / L15 / L16. T.T adds nothing.
- Raw injection of the prediction failed: its mean is 1-4% of the teacher's and 40% of the values are negative.
- As a FEATURE it works. Per layer 7-16, an oracle affine map K4d/v^2 ~ a chain_ck4/v^2 + b pred/v^2 + c (3 numbers per
  layer) was fitted to the teacher and injected into the plain chain's mean closure:

| kappa4 input (layers 7-16) | final MSE | ratio |
|---|---|---|
| plain chain | 2.273e-8 | 1.000 |
| teacher K4d (perfect kappa4) | 1.591e-8 | 0.700 |
| affine(chain) only | 2.467e-8 | 1.085 |
| affine(chain, star) | 2.296e-8 | 1.010 |
| **affine(chain, star + C.T)** | **1.758e-8** | **0.773** |
| affine(chain, star + C.T + T.T) | 1.801e-8 | 0.792 |

  So C.T recovers about 76% of the perfect-kappa4 gain.
- Component gate: C.T built from the chain's OWN kappa3 legs (r7 captures) matches the sample-exact C.T with corr 0.993
  (14->15) and 0.996 (11->15). Star fidelity is 1.000 and T.T 0.991-0.994.

## The blocker: cost
- The young legs are dense: A_st and P_st are (13 sources, 1024, 1024). Q_m(t) = (1/3) sum_s [X(m)(P(t) o Y(t))^T + ...]
  costs one 1024^3 matmul (about 1 u = 2^31 FLOPs) per (source, term, age). Over 6 ages x 13 sources x 3 terms x 16 layers
  that is more than 1000 u. The whole budget is 2^41 = 1024 u, and V56 uses about 190 u (util 0.186).
- The rank-64 C(m) eigen-sketch loses about 0.09 corr in the ceiling (0.75) and still costs about 0.06 u per term, which is
  about 230 u in total if nothing is shared.
- Grader constraints: 120 s wall time; residual op count <= ~18k ops (0.4 s); 8 GB; 2-D ops only (no batched 3-D in-place
  broadcasts).

## Questions
1. Find an algebraic route to ct_i = 24 sum_a g_a Ybar_ai X_ai Q_ai that never forms an n x n product per source. Examples:
   - share work across ages through the transport recursion (the legs at t are the legs at m times Ydot);
   - fold the sum over sources into the existing factorised chain objects (shared old basis, Z_st/L_st with 18 columns,
     Zf_st with 32);
   - a recursion of ct over t, as the chain carries C_off.
2. Which ages and which of the three Sym terms carry most of the 0.77? Prune before sketching.
3. Target: at most about 60 u extra and at most about 4k extra ops, with corr at least 0.75 on the ceiling probe.
4. Deliver the formula, a FLOP and op count, and a numpy prototype that runs on the captures
   (experiments/probes/fable_component_2026-10-09.py layout) and reports fidelity against the exact C.T.

## Data
- Captures: data/dumps/r7_cap_L{1..16}.npz (A_st, P_st, Z_st, L_st, Zf_st, R1T/R2T, mu, var, D3, D21, C_off, W_all).
- Ceiling: data/kaggle_out/fable_ceiling_all/fable_ceiling_all.npz (keys L{t}_j{j}_{star, ct, ct_r32/64/128, tt, ...}).
- Fits: results/summaries/fable_affine_fits.json. E2E: results/summaries/fable_affine_e2e.json.
