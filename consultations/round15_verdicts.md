# Round 15 verdict (8 Oct 2026)

Source: `Architecture Analysis/Round_15_Astra_Kappa4_Pattern_Assessment.pdf` (Astra, 7 Oct). Its scripts are copied to `experiments/probes/r15_astra/`.

## The candidate: source-resolved skew energy

- **Definition.** Keep the birth index b in the chain's κ3-diagonal contraction: B_bi = Σ_j [X1 P Y1 + M P² + X3 P y]_bij, and E_i = Σ_b B_bi² / v_i³.
- **Use.** Append E to the V55 feature span. Fit, rescale to the chain's level, and feed the mean closure only.
- **Cost.** About 0.0002 u and +72 metered ops for the whole network.

## Falsifier: run by Astra, reproduced by us 8 Oct with identical numbers

The test reconstructs B from the dense source captures r7_cap_L{7,11,14,15} of public MLP #0. The ceiling relaxation is generous: arbitrary coefficients per layer, a free intercept, and no level constraint.

| layer | E alone corr | feature span corr | + E corr | G4-weighted error ratio (+E / span) |
|---|---|---|---|---|
| 7 | 0.125 | 0.581 | 0.581 | 0.9995 |
| 11 | 0.032 | 0.526 | 0.527 | 1.0000 |
| 14 | 0.021 | 0.478 | 0.482 | 0.9967 |
| 15 | 0.032 | 0.455 | 0.456 | 0.9921 |

Frozen transfer (fit on L7/L11, apply at L14/L15) gives 0.359 / 0.329.

## Verdict

**Dead at the ceiling** (fair-evaluation step 1). Even the oracle-fitted family adds ≤ 0.004 correlation.

Lesson: the full signed cubic history of the sources does not carry the deep κ4 pattern. Together with the earlier kills, the only remaining route to correlation ≥ 0.7 is a genuinely quartic carried state. Those earlier kills are:
- κ4 riding the κ3 legs (×1.05 out of sample);
- a 2-3-layer young κ4 carrier;
- per-layer chain features saturating at ~0.5 (V55).

The memory-horizon measurement shows that exactly carrying κ4 births from the last 6 layers gives 0.77.
