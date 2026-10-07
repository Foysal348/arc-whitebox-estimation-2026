# Round 16 verdict (8 Oct 2026)

Source: `Architecture Analysis/Round 16 - Signed Bilinear Quartic Memory.pdf` (Astra). Astra could not reach our data, so we ran its falsifier ourselves.

## The candidate: Signed Bilinear Quartic Memory (SBQM)

- **State.** A symmetric dense matrix H, with implied quartic cumulant Q = Sym(H ⊗ C).
- **Readout.** The κ4 diagonal is 3 H_ii C_ii.
- **Transport.** W^T H W.
- **Refresh.** At each ReLU only the diagonal of H is refreshed, from the chain's κ4 diagonal.
- **Cost.** 30 u (two dense products per map).

## Falsifier: run by us, with Astra's own gate

Script `experiments/probes/r16_sbqm_ceiling.py`, results in `results/summaries/r16_sbqm_ceiling.json`.

- **Method.** Teacher-forced, one step. Fit H to the post-ReLU teacher's K31 and K22 slices (C = teacher), with H_ii = K4d/(3 C_ii). Transport once with the real W and correlate with the next pre-activation teacher's κ4 pattern on decisive neurons.
- **Sanity check.** W^T C W reproduces the next teacher covariance to 0.000 relative error.

| post → pre | SBQM (Astra fit) | K31-only fit | K22-only fit | H = C | diagonal only | true (4)+(3,1)+(2,2) slices |
|---|---|---|---|---|---|---|
| 3 → 4 | 0.177 | 0.200 | −0.016 | 0.028 | 0.189 | 0.208 |
| 7 → 8 | 0.397 | 0.406 | 0.041 | 0.011 | 0.392 | 0.446 |
| 11 → 12 | 0.351 | 0.364 | −0.025 | −0.019 | 0.336 | 0.392 |
| 14 → 15 | 0.321 | 0.325 | 0.006 | −0.015 | 0.294 | 0.296 |
| 15 → 16 | 0.371 | 0.389 | −0.048 | 0.009 | 0.351 | 0.377 |

The fit quality is poor: R² of the off-diagonal K31 slice is 0.66-0.84, and of K22 strongly negative. Sym(H ⊗ C) cannot represent both slices at once.

## Verdict

**Dead at the ceiling** under Astra's own gate (≥ 0.70 at L12/15/16). The best variant reaches 0.33-0.39, which is no better than carrying the diagonal alone.

Sharpened fact for any next round (one step from the TRUE post-ReLU state):

| state carried one step | pattern corr |
|---|---|
| (4), (3,1) and (2,2) slices | 0.30-0.45 |
| full κ4 tensor (earlier measurement) | 0.77-0.80 |

The missing ~0.4 of correlation lives entirely in the post-ReLU κ(a,a,b,c) and κ(a,b,c,d) entries. No state built from slices or from matrices bilinear with C can supply it.
