# New-architecture track, step 1 (8 Oct 2026): can Hermite-tree inputs replace the κ3 source chain? NO (gate failed)

Scripts: `experiments/probes/r19_tree_ceiling.py` and `r19_tree_rescale.py`. Results: `results/summaries/r19_tree_ceiling.json` and `r19_tree_rescale.json`.

## Method

Teacher-forced, per layer, on public MLP #0. The univariate Edgeworth mean closure is evaluated from the teacher's μ and v and a (κ3, κ4) diagonal pair, and its error against the teacher's post-ReLU mean is measured.

The tree inputs are Gaussian Hermite-tree births at every layer:
- κ3 star: 6 Σ g P X²;
- κ4 star + path, as in HTQB.

Births are carried linearly for h ages and built from the TEACHER μ and C, i.e. oracle inputs.

## Closure MSE vs relu_mean (selected layers)

| layer | teacher inputs (floor) | chain | tree h = 6 | tree h = 6, κ3 with an oracle level scalar + chain κ4 | chain κ3 + tree κ4 |
|---|---|---|---|---|---|
| 8 | 4.3e-11 | 1.44e-9 | 2.99e-8 | 2.24e-8 | 1.08e-8 |
| 12 | 5.2e-11 | 2.00e-9 | 3.40e-8 | 2.79e-8 | 1.91e-8 |
| 16 | 6.2e-11 | 2.07e-9 | 4.18e-8 | 4.38e-8 | 1.93e-8 |

The tree κ3 diagonal has pattern correlation 0.98 with the teacher, against 1.00 for the chain. Yet in the closure it is 10-20× worse than the chain, even after an oracle per-layer level rescale.

Raw tree κ4 is also worse than the chain's in level: chain κ3 + tree κ4 gives 1e-8 to 2e-8, against 2e-9 for the chain. As a pattern feature inside V55, however, it is useful (V56).

## Verdict

The gate fails: a Gaussian-tree κ3 cannot replace the source chain. The mean closure needs κ3 to about 1% precision, and the non-Gaussian corrections the chain carries are essential.

The new-architecture track via Hermite trees is closed. The tree terms stay useful as κ4-pattern features (V56) inside the chain.
