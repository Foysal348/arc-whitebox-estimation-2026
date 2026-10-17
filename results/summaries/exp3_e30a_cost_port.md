# Experiment 3+6: porting e30_a's exact cost savings into E299 (17 Oct 2026)

Variant file: `experiments/variants/e299_x3.py`. It is E299 (`kaggle/e299b/E299_estimator.py`, team best #336377) plus env knobs `X3_*`, all off by default. The ported pieces credit LordProcciuto's e30_a (MIT): https://github.com/siming767-creator/whestbench-lordprocciuto.
Runner: `experiments/variants/e299_x3_check.py` (template `kaggle/e299b/warm_check.py`; uses the `kaggle/_runlock` lock). Raw numbers: `results/summaries/exp3_e30a_cost_port.json`.

Setup: MLP #0 (`k4teach/weights_seed0.npz`, truth `k4teach/truth.npz`). Two `predict()` calls in one fresh process (cold, then warm). Python 3.14 with flopscope 0.12.1. u = 2^31 FLOPs.

**Base check.** With all knobs off, `e299_x3.py` matches E299 exactly: same predictions (all 16 layers, `array_equal`), same FLOPs and same ops.

| | cold u | cold ops | warm u | warm ops | MSE |
|---|---|---|---|---|---|
| base (all X3 off) | 194.583 | 23,273 | 186.674 | 21,953 | 1.74848e-8 |
| E299 reference file | 194.583 | 23,273 | 186.674 | 21,953 | 1.74848e-8 |

## Per-knob results (each knob alone vs base, warm call)

"maxdev" is the largest |p - p_base| over all 16 layer outputs, divided by max |p_base|. On a single MLP, MSE ratios within about ±0.2 % of 1 are fp32 rounding noise.

| knob | what it does | exact? | Δ warm FLOPs | Δ ops | MSE ratio | maxdev |
|---|---|---|---|---|---|---|
| `X3_SGDEAD=1` | Analogue of e30_a e29 `sgdead`. With `V41_JOIN=transport`, nothing reads the tier-1 Gram core Sg. Skips the joiner Gram Sj (two aliased r×n×r Grams), the Sg rotation Tq Sg Tqᵀ, and the Sg updates. | **bit-exact** | **-0.921 %** | -94 | 1.000000 | 0 |
| `X3_TQR=1` | Our own V53 (graded -0.33 % on 27 Sep, never merged into the A line), same idea as e30_a `tri`. The transport-join rotation Tq = Qnᵀ Qp is the R factor of the QR, so the r×n×r product is dropped. | rounding-level | **-0.427 %** | -8 | 0.998119 | 7.9e-7 |
| `X3_NOSYMC=1` | e30_a e29 `nosymc`. Drops the `as_symmetric` tag on C (billed 7 n² per call; no later op reads the tag). | **bit-exact** | -0.029 % | -16 | 1.000000 | 0 |
| `X3_FOLD2=1` | e30_a `X_FOLD2` (part 1). The -2 / 4 constants of K21 / K22 / e_b move into the vector (powers of 2, so bit-exact). C = K11 copy. No redundant `zero_diag` on K21, K22, S21, S_sep. | **bit-exact** | -0.020 % | -135 | 1.000000 | 0 |
| `X3_DDMM=1` | e30_a `X_FOLD2` (part 2). The dA / dP Gram weights come from one (n,3)@(3,2) product. | rounding-level | 0.000 % | -43 | 1.000662 | 7.7e-7 |
| `X3_KVEC=1` | e30_a `X_KVEC`. The vector K parts (1,)..(4,) come from 11 monomials and one matmul (same PK2K_TABLE). | rounding-level (just above 1e-6) | 0.000 % | **-299** | 0.999717 | 1.4e-6 |
| `X3_PWR=1` | e30_a `X_PWR`. Powers of alpha, 1/σ and σ via `fnp.power`. | rounding-level (just above 1e-6) | +0.001 % | -157 | 1.000223 | 1.2e-6 |
| `X3_RELOJ=1` | e30_a `X_RELOJ`. The PK2 group einsum becomes in-place multiply, multiply, sum. | bit-exact | 0.000 % | **+90** | 1.000000 | 0 |
| `X3_HOT=3.0` | e30_a e24hot `X_VB_HOT`. Old-level dense legs (formation, D21 inner, lift, inner Qcᵀ) are computed only for rows with alpha ≤ 3, rounded up to 32 rows; hot rows stay zero. Formation is deferred until after C_pre. | **approximation** | -1.431 % | +419 | **1.050782** | 4.1e-5 |

Summary:
- SGDEAD and TQR carry all the real FLOP savings.
- NOSYMC, FOLD2, DDMM, KVEC and PWR mostly cut op count (about 650 ops together), which helps the residual budget.
- RELOJ is FLOP-neutral and adds ops (it was a wall-clock fix for e30_a's einsum form). Do not use it.
- HOT 3.0 costs +5.1 % MSE for -1.4 % FLOPs (score about x1.036, worse), unlike e30_a, where it was "at equal MSE". A likely reason: E299's D3-driven parts (K4C/K4S features, final rider) use the hot rows' old-source D3 and D21, and e30_a has no such parts.

Combined runs:

| combination | cold u | warm u | warm ops | Δ warm FLOPs | Δ ops | MSE ratio | exact? |
|---|---|---|---|---|---|---|---|
| `exact_all` = SGDEAD + NOSYMC + FOLD2 | 192.772 | 184.864 | 21,708 | **-0.970 %** | **-245** | 1.000000 | **bit-identical** (all 16 layers) |
| `exact_round` = exact_all + TQR + KVEC + PWR + DDMM | not run | | | expected about -1.40 % | expected about -752 | expected 1 ± rounding | rounding-level |

The knobs add up exactly: exact_all = -0.921 - 0.029 - 0.020 % and -94 - 16 - 135 ops. `exact_round` and `exact_round_hot3` were **not run locally**: the laptop was overloaded and the coordinator stopped new local runs. The exact_round expectation (warm about 184.06 u, about 21.2k ops) is that same sum.

## e30_a mechanisms with no port, or one already in E299

| e30_a mechanism | status in E299 |
|---|---|
| `ueye` (U = [I; 0], so Qc U and the inner2 Uᵀ lift become copies) | Not portable as an exact change. It needs e30_a's fixed basis (X_FIJA). E299's sub-basis U rotates at every join (U = Tq U) and at every nest move (U = Un), so Qc U (n·r1·r2, about 0.075 u per layer) and the lift are real products. Making U = [I; 0] means re-basing tier 1 at each nest move; that is an architecture change worth about 1 u at most. |
| `tri` (block triangular solve for M = R⁻¹Qᵀ) | E299 has no solve; its transport join uses QR plus a rotation. The matching saving is X3_TQR (Tq = R). |
| `nest` / `nest2` / `nestg` (fixed-basis least-squares move into tier 2) | Not applicable. E299 moves the oldest source into tier 2 with a range finder on G2 (r1×r1, small). |
| `fj` (projection written straight into the factor slot) | Already in E299: with SB2, the joiner projection `smm.mm` writes directly into `fap4_new`. |
| `oldc` (old legs written compact, no take back to natural order) | Not applicable. E299 forms the old legs at full n rows directly into the leg slots; there is no permutation round-trip. |
| `sgdead` | Ported as X3_SGDEAD (E299's dead Sg is larger than e30_a's: -0.92 %). |
| `X_CORR_FUSE` (87-feature mean-correction table fused) | Not applicable. E299 runs with `V17_NO_CORR=1`, so there is no correction table. |
| `X_WINO_LEAF` (Winograd leaf) | Already in E299 (`V63_WINO`, E299's own change). |
| DD_team "omit the first identity transport (Wᵀ I = Wᵀ), defer the newest identity block" | Already in E299 through the V29 lineage. The newborn's P slot carries C through the transport family, and its P leg is then set by `copyto(legs[k,1], W)` (n² copy, no product). e30_a's `ueye` is a different thing (fixed sub-basis selection). |

## Recommendation for the 100-MLP Kaggle test

Recommended combination: `X3_SGDEAD=1 X3_TQR=1 X3_NOSYMC=1 X3_FOLD2=1 X3_KVEC=1 X3_PWR=1 X3_DDMM=1`. Expected warm FLOPs about -1.4 %, warm ops about -750 (21.95k → about 21.2k), MSE unchanged within rounding. Also run the exact-only combination `X3_SGDEAD=1 X3_NOSYMC=1 X3_FOLD2=1` (-0.97 %, bit-identical predictions) as the safe fallback. Leave RELOJ and HOT off. HOT is only worth a probe at higher thresholds (for example 4.0 or 5.0); it has not been tested locally.
