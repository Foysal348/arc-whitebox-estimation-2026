# Experiment 5: the E299 kappa4 corrector ported into e30_a (17 Oct 2026)

**Variant:** `experiments/variants/e30a_k4c.py`. This is e30_a (`external/e30_a_estimator.py`, the MIT derivative of 504aldo's V29) with these additions behind env knobs:

- `X5_K4C`: the V55 kappa4-diagonal pattern corrector with the V56 HTQB cubic star (`X5_K4S`, default 1 under K4C).
- `X5_K4C_SCALE`: scales the corrector's deviation from e30_a's own g4row.
- `X5_K4LVL`: E299's per-block kappa4 level (V59_K4LVL GN values).
- `X5_RIDER`: E299's E284 cubic final-layer rider.

All coefficients are copied unchanged from `kaggle/e299b/E299_estimator.py`. With every knob unset, the variant is **bit-identical** to e30_a. Checked: final output `np.array_equal` = True, with the same FLOPs and op count.

**Runner:** `experiments/variants/e30a_k4c_warm_check.py`. It follows the template of `kaggle/e299b/warm_check.py`: each variant runs in a fresh subprocess, makes a cold and a warm `predict()` call, and runs under `kaggle/_runlock`. Raw numbers are in `results/summaries/exp5_k4c_into_e30a.json`.

**Data:** MLP #0 from k4teach (weights_seed0 / truth.npz final_means). u = 2^31 FLOPs.

## What the corrector does in E299 (step 1)

The corrector runs at every layer with mode 1 and layer index li+1 >= 4 (LMIN 4, through the last layer). It acts after D3, D21 and g4row = dG·METRIC_C are formed and before the term program runs.

1. **Features.** It builds standardised per-neuron features from quantities the chain already has:
   - s3sq = (D3/var^1.5)^2
   - row and column sums of (D21/(var_i·s_j))^2
   - the row sum of (C_pre/(s_i s_j))^2
   - |mu/s|
   - chain = g4row/var^2
   - V56 star: 24·Σ_j c_j P_ij X_ij^3 over the youngest ≤6 dense source slots, with X = A + P·diag(w1·var at birth) and c = −alpha·phi/(6 var) at birth
   - a constant
2. **Pattern map.** A fixed linear map turns these features into a kappa4-diagonal pattern. The map has 8 coefficients for hidden layers and 4 for the last layer (s3sq, |a|, chain, 1).
3. **Level.** The pattern is multiplied by var^2 and rescaled to the layer mean of g4row, so only the **shape** changes, not the level.
4. **Where it enters.** The result replaces g4 in the d=1 vector part of the term program (`b1["g4"]`), which is the mean closure. The pairwise g4col/g4row blocks keep the raw diagonal.
5. **Level hook.** E287/E288 multiplies a separate per-block level (V59_K4LVL: 1.026 for layers 1-4, about 0.98 afterwards) onto the same g4.

**Cost.** Each young slot needs 5 elementwise n² operations and one mat-vec, plus a few n² feature sums. It is not free in ops.

**Port into e30_a.** e30_a compacts dead neurons and stores each source's columns permuted by that layer's `cperm`. The per-source birth vectors (w1·var and the c coefficient) are therefore stored through the same `cg()` permutation that e30_a uses for w2/S3c/e. The block sits after e30_a's alive mask, so D21 is masked as in E299's dead-column handling.

## Results, MLP #0 (step 3)

Base e30_a: warm **158.841 u**, **19,791 warm ops**, final MSE **2.0887e-8**. For reference, E299 on the same MLP is 186.67 u and 1.7485e-8.

| variant | MSE | MSE ratio | Δwarm FLOPs | Δwarm ops | score ratio (MSE × warm u) |
|---|---|---|---|---|---|
| e30a (base) | 2.08870e-8 | 1 | 0 | 0 | 1 |
| X5 all off | 2.08870e-8 | 1.00000 (bit-identical) | +0.000 u | +0 | 1.00000 |
| K4C scale 0.25 | 2.04751e-8 | **0.98028** | +0.214 u (+0.13 %) | +815 | **0.98160** |
| K4C scale 0.50 | 2.06411e-8 | 0.98822 | +0.214 u | +815 | 0.98955 |
| K4C scale 0.75 | 2.13273e-8 | 1.02108 | +0.214 u | +815 | 1.02245 |
| K4C scale 1.00 (as in E299) | 2.26626e-8 | 1.08501 | +0.214 u | +815 | 1.08647 |
| K4C 1.00, no star (V55 7-feature map) | 2.29509e-8 | 1.09881 | +0.053 u | +368 | 1.09918 |
| K4C 1.00 + E299 K4LVL | 2.20790e-8 | 1.05707 | +0.214 u | +830 | 1.05849 |
| rider only | 2.09513e-8 | 1.00308 | +0.002 u | +26 | 1.00309 |

Run 3 never ran: level only, scale 0.3, scale 0.3 + level, and scale 0.3 without the star were queued on the shared lock and cancelled at 16:43 when the laptop was overloaded. **Those arms are untested locally.**

**Projection from the measured points.** A quadratic through scales 0, 0.5 and 1 puts the optimum near **scale 0.30, with an MSE ratio of about 0.980**. Least squares of (K4C(1) − base) against the base error on the final layer gives the same 0.31. The correction direction correlates +0.15 with the base error.

## Reading

- **Scale 1 makes e30_a worse (+8.5 % MSE).** The likely cause is double counting. e30_a's mean correction is the 87-feature `CORR_FEATS` table: a ridge fit on 70 networks of e30_a's **own uncorrected closure**. It already contains Hermite(alpha) × modulator terms (`h_k*s4` with s4 = g4/var², `h_k*s3`, `s33`, `dn`, `cr`, `c2`), which are essentially the K4C features. Part of the per-neuron kappa4 pattern error is therefore already absorbed downstream, and the full E299 deviation overshoots. A scale of about 0.25-0.3 recovers a gain of about 2 %.
- **The star carries real signal.** Without it, scale 1 is worse (1.099 vs 1.085). That also suggests the star's column-frame port under e30_a's dead-neuron permutation is right.
- **E299's K4LVL helps at scale 1** (1.085 → 1.057). These levels are fitted to E299's chain, so the same caution applies (level only and level at scale 0.3 are untested locally).
- **The rider is not portable.** It costs +0.3 % MSE on e30_a. It was refitted for E299's V31 spherical-input chain and its 13-feature table. Not recommended.
- **Cost:** +0.214 u (+0.13 % FLOPs), but **+815 warm ops** (19.8k → 20.6k). e30_a already exceeds our ~18k residual-safe op class. The no-star map costs only +368 ops but loses the star's signal. Before a nomination, the star loop (6 slots × 6 ops) could be fused into fewer ops, for example as one batched einsum over the young slots.

## Caveats

- This is **one network (MLP #0)**, which is weak evidence. The best scale (about 0.25-0.3) was picked on that same network, so the ~2 % gain is in-sample and optimistic.
- The coefficients (map, level, rider) were fitted for E299's chain (V29 + V31 + V40-V59 + GN10-GN16), not for e30_a's. Refitting the map against e30_a's table, or refitting e30_a's table with K4C on, is the principled fix. A small fixed scale is a stopgap.

## Recommendation for a 100-MLP Kaggle test

Run a paired comparison of e30_a against `X5_K4C=1` at **X5_K4C_SCALE ∈ {0.2, 0.3, 0.4}**, with K4S on and the rider off. Add arms for **0.3 + X5_K4LVL=1** and **X5_K4LVL=1 alone**, both untested locally. Use 100 MLPs and report the paired score ratio with a CI and the win count.

The variant is worth nominating only if the 100-MLP ratio is ≤ ~0.985 and the warm op count passes the residual check. Today it is 20.6k ops against e30_a's 19.8k; e30_a itself graded fine at 19.8k on the board.

The higher-ceiling follow-up is to refit e30_a's 87-feature correction table with the K4C-corrected closure, or to refit the 8-coefficient map on e30_a's closure using the E-series GN tools, rather than shrinking E299's map.

## Exact Kaggle arm list (100 mini MLPs, flopscope 0.12.3)

Estimator: `experiments/variants/e30a_k4c.py` for every arm; the base arm with all knobs off is bit-identical to e30_a.

```
{
 "e30a_base":        {},
 "k4c_s020":         {"X5_K4C": "1", "X5_K4C_SCALE": "0.2"},   # untested locally
 "k4c_s025":         {"X5_K4C": "1", "X5_K4C_SCALE": "0.25"},  # tested: x0.9803 MSE on MLP #0
 "k4c_s030":         {"X5_K4C": "1", "X5_K4C_SCALE": "0.3"},   # untested locally (projected optimum)
 "k4c_s040":         {"X5_K4C": "1", "X5_K4C_SCALE": "0.4"},   # untested locally
 "k4c_s050":         {"X5_K4C": "1", "X5_K4C_SCALE": "0.5"},   # tested: x0.9882
 "k4c_s030_lvl":     {"X5_K4C": "1", "X5_K4C_SCALE": "0.3", "X5_K4LVL": "1"},  # untested locally
 "lvl_only":         {"X5_K4LVL": "1"},                         # untested locally
 "k4c_s030_nostar":  {"X5_K4C": "1", "X5_K4C_SCALE": "0.3", "X5_K4S": "0"},    # untested locally (cheap-op option, +368 ops)
 "k4c_s030_rider":   {"X5_K4C": "1", "X5_K4C_SCALE": "0.3", "X5_RIDER": "1"}   # untested locally; rider alone was x1.0031 (worse)
}
```

Priority if the arm count must be cut: e30a_base, k4c_s025, k4c_s030, k4c_s040, k4c_s030_lvl, lvl_only. The rider arm is lowest priority.
