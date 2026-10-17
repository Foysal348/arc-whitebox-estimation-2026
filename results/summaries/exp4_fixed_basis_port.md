# EXP4: e30_a fixed-basis old level ported into E299 (17 Oct 2026)

Source: LordProcciuto's e30_a (MIT): `external/e30_a_estimator.py`, write-up sections 3A, 5.4 and 5.7.
Variant: `experiments/variants/e299_fija.py`. It is E299 plus two knobs. Both default to OFF, which keeps the E299 code path unchanged.
Harness: `experiments/variants/exp4_warm_check.py`. It follows the `kaggle/e299b/warm_check.py` template and takes `kaggle/_runlock` around each child run.

## Status: no measurements yet

No local estimator run finished. Other agents held the run lock for about 1.5 h. After that, the user stopped all local runs because the laptop was overloaded, and the queued harness was killed before it ever took the lock.

Only static checks have been done:
- syntax is valid
- pyflakes is clean
- the module imports
- there is no plain numpy import

All numbers below have to come from Kaggle.

## What e30_a does and how it compares with E299

| | E299 (V21/V24/V42) | e30_a fixed basis (X_FIJA) |
|---|---|---|
| First join | range finder, rank R_OLD 352 | same range finder; then, if tier 2 is on, ordered once by the first joiner's Gram (8 subspace passes, the R_OLD2 leading directions first) |
| Later tier-1 joins | transport: QR of Qp, joiner = Qn^T A, old factors rotated by R (Tq), Gram core Sg kept | basis kept as transported (not orthonormal); joiner by least squares, M = R^-1 Q^T (block-bisection triangular solve); no rotation, no Sg |
| Tier 2 (age > AGE_OLD2) | at every nest: range finder on G2 = S_s + U S2 U^T, a new U, and the old tier-2 factors are re-projected (T2 = Un^T U, which loses information each time) | U = [I; 0] fixed (the first R_OLD2 columns); the mover is projected by least squares in the basis metric, F2 = F[:r2] + R11^-1 R12 F[r2:]; old tier-2 factors are never touched |
| Qc U and the D21 lift through U | products | column copies (e28 "ueye") |

Tier 1 is the same as E299's V42 transport join in exact arithmetic: the same span, only different coordinates. Expected effects:

- **FLOPs fall:** the rotations Tq·FAP and T2·FAP2, the Gram cores Sj, S_s and S2, the tier-2 range finder, and the U products all go away. My rough estimate is about 5–10 u out of 186.7 u (–3 to –5 %), but this is not measured.
- **MSE changes only through tier 2** (fixed instead of adaptive sub-basis) and through rounding. e30_a's –4.5 % raw MSE was measured against e16, which ran a range finder at every join. E299 already has the transport join, so most of that gain is probably already in E299. The likely MSE ratio is about 1.00, and it could go either way.

**Hot-row rule (X_VB_HOT, e24hot):** rows with alpha = mu/sqrt(var) > t get zero old-level dense legs. Young sources are untouched. In the port:

- Old-leg formation moves to after C_pre, because alpha needs this layer's variance.
- Formation runs only on the non-hot rows, padded with the least-hot rows to a multiple of 32, and is scattered back with zeros elsewhere.
- The old part of the D21 contraction (`inner`) runs only on those rows.
- The old D21 columns of hot neurons are zeroed (Qc is masked) for consistency.

e30_a reported –2.4 % cost at equal MSE (submission 335727).

## Knobs (environment variables)

- `X4_FIJA=1`: fixed-basis old level (default 0).
  - Sub-knobs: `X4_FIJA_PASS` (8), `X4_FIJA_TRI` (2), `X4_FIJA_TRI2` (1).
- `X4_HOT=t`: hot-row threshold (default 0 = off; e30_a uses 3.0).
  - Sub-knob: `X4_HOT_Q` (32).

## Kaggle test (100 mini MLPs, flopscope 0.12.3); all untested locally

```
{"base": {}, "fija": {"X4_FIJA": "1"}, "hot3": {"X4_HOT": "3.0"}, "fija_hot3": {"X4_FIJA": "1", "X4_HOT": "3.0"}}
```

Optional: `{"fija_hot2.5": {"X4_FIJA": "1", "X4_HOT": "2.5"}}`.

Checks to run first:
1. **`base` must match E299 exactly.** On MLP #0 that means MSE 1.7484848188035433e-08, warm 186.674 u and 21,953 ops. If it does not match, the default path is broken.
2. **Warm op count for each variant.** The residual budget is about 18k ops; E299 itself is at 21.9k.
   - FIJA should remove ops (no rotations, Gram cores or tier-2 range finder).
   - HOT adds a few gathers per layer.

What to report per variant: paired score ratio against base (mean and CI), MSE ratio, warm FLOPs ratio, ops, and how many MLPs failed.
