# Experiment 7: e30_a young-source / C_pre dead-neuron compactions in e299xrk (17-18 Oct 2026)

Source of the ideas: LordProcciuto's e30_a (MIT; `external/e30_a_estimator.py`, X_DEAD_* "fis/comp" machinery,
`_vivas` rule: tau 2.85, rounding 'cerca' to multiples of 32). Variant: `experiments/variants/e299_x7.py`
(e299xrk + knobs; header credits e30_a). Base `submissions/e299xrk/estimator.py` untouched.

## What was ported

| Knob | e30_a piece | What it does in e299_x7 | Exact for us? |
|---|---|---|---|
| `X7_YROW=1` | `pre_done` + fis 2 (c) rows `ml` (+ fis 5 'f' thin legs, `X7_YROW_Z=1`) | W C and C_pre are formed **before** the young family (C leaves the family; W C keeps our cidx + CSPLIT exact split), so this layer's alpha is known; the young legs + newborn A leg (and Z / Zf) are transported only into rows with alpha > -2.85 (rounded to 32), other rows zero | Nearly. Rows below -2.85 are already gone from D21 rows/columns (Phi < 0.005 / 0.03), so in principle only their young D3 changes. The measured MSE says otherwise (see below) |
| `X7_CPRE=1` | fis 1 (a) inner `mp` + comp 'p' | W C = W[:, P] C[P, P] (P = previous layer's kept set), C_pre only on this layer's kept rows/columns (inner P), full diagonal = row sums of (W C) * W | No. This is e30_a's mask semantics for C: dropped neurons lose their C_off rows/columns, so they stop being hubs of the newborn (a_b = w1 C_off) |
| `X7_DEAD_TAU` 2.85, `X7_DEAD_ROUND` cerca/arriba/abajo, `X7_DEAD_MULT` 32 | `_vivas` | kept-set rule | n/a |

**Not ported: source-column compaction (e30_a fis 4, `mcf` / `cg()` permutation).** In our chain it is not exact.
`_dslices` D3 has a pure-P term `sum_j s_j P_ij^3` (M = P diag(s) + ...), and our dead neurons keep s, e, w2 and C_off.
Dropping columns would therefore remove dead neurons as hubs, which is the change that was catastrophic before (x6.4 at Phi 0.0125).
It would only be consistent on top of CPRE plus masking s/e/w2 of dropped neurons (e30_a's full mask). Not attempted.

## MLP #0 check (Kaggle kernel `esfersami50/whest-x7`, flopscope 0.12.3, 2 predict() calls, warm shown)

Raw output: `data/kaggle_out/x7/` (log, `x7_results.json`, call-site ledgers `ledger_x7_yrow.*`, `ledger_x7_both.*`).

| Variant | warm u | warm ops | resid (Kaggle) | final MSE | MSE ratio | FLOPs ratio | score ratio |
|---|---|---|---|---|---|---|---|
| ref e299xrk | 184.110 | 21,341 | 1.227 s | 1.72202e-8 | - | - | - |
| base (all off) | 184.110 | 21,341 | 1.227 s | 1.72202e-8 | 1 (max abs diff vs e299xrk = 0.0) | 1 | 1 |
| yrow | 180.165 | 23,226 | 1.312 s | 1.87000e-8 | 1.0859 | 0.9786 | **1.0627** |
| cpre | 182.199 | 23,100 | 1.295 s | 1.72796e-8 | 1.0035 | 0.9896 | **0.9930** |
| both | 177.762 | 23,289 | 1.318 s | 1.85516e-8 | 1.0773 | 0.9655 | 1.0402 |
| yrow arriba | 180.688 | 23,157 | 1.312 s | 1.80022e-8 | 1.0454 | 0.9814 | 1.0260 |
| yrow, no Z rows | 180.412 | 23,166 | 1.309 s | 1.85932e-8 | 1.0797 | 0.9799 | 1.0580 |

Default-off is bit-identical to e299xrk (same FLOPs, same ops, max abs diff 0).

### Why the gap to e30_a is mostly not reachable by rows

- Young transport + W C at the same sites: base 62.0 u (52.0 + 10.1) -> yrow 57.7 u (46.4 young on kept rows,
  4.5 young on layers without a kept set, 5.25 + 1.65 W C), so only -4.3 u. At tau 2.85 only about 8 % of rows drop.
  e30_a's 36 u comes mainly from the **columns** (`mcf`) and its full mask, not from `ml`.
- Our inner dimension (cidx, Phi >= 0.0125, i.e. alpha >= -2.24) is already narrower than e30_a's `mp` (alpha > -2.85).
- C_pre: the ledger comparison "6.1 u vs 1.4 u" was misleading. e30_a's C_pre is 3.9 u (kept-row family, line 2395)
  + 1.4 u (`_sym_product`) + 5.0 + 1.65 u for W C, about 11.9 u in total, against our 6.1 + about 6.8 u. CPRE got us
  W C 5.0 u + C_pre 4.9 u, warm -1.91 u.
- Dropping the young rows of alpha < -2.85 costs +4.5 % MSE ('arriba') and +8.6 % ('cerca', which also drops up to 16 live rows).
  This is far more than D3 of dead neurons alone should cost. The likely amplifiers are the K4S star / K4C features
  (`g4new` is rescaled by a global mean, so changes in dead rows move every row) and the ridge mean correction.
  Not investigated further.

### Op count

The pre stage adds about 1.8k warm ops (+8 %: W C becomes its own Strassen family, plus gathers).
Kaggle-measured residual is 1.29-1.32 s against 1.23 s for base (Kaggle CPU is much slower than the grader).
Given the residual-op budget note (keep warm ops <= ~18k; base is already 21.3k), CPRE's +1.76k ops is a risk.
To trim it, run the W C / C_pre families of the pre stage at a lower Strassen level (fewer combos), at a small FLOPs cost.

## 100-MLP run (launched)

Only CPRE was worth it (x0.993 on MLP #0, below the 0.97 gate but a certain -1.04 % FLOPs).
- Kernel `esfersami50/whest-runner-x7-1` (dataset `esfersami50/whest-estimators-x7-1`), arms `cpre` and
  `cpre_arriba` (X7_DEAD_ROUND=arriba), 100 mini MLPs, flopscope 0.12.3. Launched 18 Oct 00:15 (Bangladesh time);
  about 2 x 100 x 110 s, so expect results around 6-7 am.
- Paired base: arm `xr_k31_025` of `esfersami50/whest-runner-xrk31-1` (still running). It is e299xr_k31.py with X2_K31TWO=1,
  X2_K31_MIX=0.25, which differs from e299xrk only in those two defaults.
- Harness: `kaggle/x7h/make_sets.py`, `kaggle/x7h/launch.sh`. Comparison: `kaggle/x7h/compare.py` (fetch both outputs first;
  the commands are in its docstring).

## Files

- `experiments/variants/e299_x7.py`: the variant (all X7 knobs off = e299xrk).
- `kaggle/x7/data/`, `kaggle/x7/kernel/run.py`: the MLP #0 check (dataset `esfersami50/whest-x7-data`, kernel `esfersami50/whest-x7`).
- `data/kaggle_out/x7/`: its outputs.
- `kaggle/x7h/`: the 100-MLP harness.

No AIcrowd submission, no git commit.
