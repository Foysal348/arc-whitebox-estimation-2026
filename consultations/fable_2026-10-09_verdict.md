# Fable LKV design: ceiling verdict (9 Oct 2026)

Design: `consultations/fable_2026-10-09_design.md`. Script: `experiments/probes/fable_ceiling_2026-10-09.py`. Run by the main agent on a Kaggle GPU: N = 2e6, MLP #0, exact κ3 from samples, project metric. Results: `results/summaries/fable_ceiling_2026-10-09.json`.

## Decisive-neuron κ4 pattern correlation, births at ages 1-6

| prediction | L12 | L15 | L16 |
|---|---|---|---|
| star (shipped in V56) | 0.521 | 0.489 | 0.471 |
| **star + C·T (Q1 conversion)** | **0.846** | **0.835** | **0.824** |
| star + C·T + T·T (the full LKV) | 0.849 | 0.840 | 0.830 |
| star + T·T | 0.490 | 0.445 | 0.436 |
| full K = 3 births (exact inputs) | 0.926 | 0.893 | 0.918 |
| K = 3 births + exact linear carry | 0.936 | 0.931 | 0.957 |
| ARCH with the hub index sketched to rank 64 | 0.752 | 0.753 | 0.744 |

By age window (star + C·T + T·T), L12 / L15 / L16:

| ages | L12 | L15 | L16 |
|---|---|---|---|
| 1-2 | 0.703 | 0.643 | 0.654 |
| 1-3 | 0.763 | 0.736 | 0.726 |
| 1-4 | 0.815 | 0.798 | 0.768 |

Checks: telescoping error ≤ 1e-5; Cu1 ≈ star + T·T + C·K, with corr 0.993-0.998.

## Verdict

- **Fable's central hypothesis is killed** by its own pre-registered clause: T·T adds less than 0.05 over star + C·T at all three layers (+0.003 / +0.005 / +0.006).
- **The ceiling passes for star + C·T.** It reaches 0.82-0.85, equal to the full-cubic family ceiling.
- **Lesson:** Q1 (C·T) was killed alone in Round 17 at 0.43-0.44, and the star alone sits at 0.47-0.52. Their SUM reaches 0.82-0.85. The Round-17 Q1 kill was too narrow: it was never tested together with the star.

## Next

1. E2E: inject the prediction as the κ4-diagonal pattern into the plain chain and measure the final MSE (kernel `fable-lkv-e2e`).
2. Component: the C·T readout built from the chain's own κ3 sources (`experiments/probes/fable_component_2026-10-09.py`).
3. Cost: an exact C·T readout needs S = T(a; Ẏ_i, Ẏ_i) for all hub indices a. The rank-64 sketch loses about 0.09 correlation.
