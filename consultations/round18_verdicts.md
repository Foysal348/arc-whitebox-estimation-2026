# Round 18 verdicts (8 Oct 2026)

## HTQB-6/3: Hermite Tree Quartic Bank

Source: `Architecture Analysis/Round 18 - Hermite Tree Quartic Bank.pdf` and `round18_htqb_falsifier.py`. The script is copied verbatim to `experiments/probes/r18_htqb/`.

**Design.** It keeps the two lowest connected Gaussian trees of the degree-3 Hermite ReLU response:
- the Cu1 star (3,1,1,1): 24 Σ_a c_a P_ai X_ai³;
- the Q2 path (2,2,1,1): 48 V_:iᵀ C V_:i, with X = C_b (Φ_b P) and V = (g_b P) ∘ X.

Each birth is built from the teacher μ and C at that layer and carried linearly by P_(b→t) = W[b] U[b+1] … U[t−1].
- Shippable form: the star kept for 6 ages, the path for 3, at L8-16.
- Claimed cost: 27.2 u.

**Run by us** (packed teacher, project metric; `results/summaries/round18_htqb.json`). The self-test algebra matches its claim (star 2.1e-15, path 5.2e-16).

| | L12 | L15 | L16 |
|---|---|---|---|
| ceiling, star + path, h = 6 | 0.637 | 0.559 | 0.586 |
| shippable, star h6 + path h3 | 0.616 | 0.538 | 0.541 |
| star only, h = 6 | 0.521 | 0.489 | 0.471 |
| path only, h = 6 | 0.074 | 0.024 | 0.150 |

**Verdict.** BORDERLINE on its own: 0.56-0.64, between the KILL line 0.55 and the PASS line 0.70. It is nonetheless the best cheap carrier so far; every earlier one sat at ~0.44.

## Complementarity with the shipped V55 features (`experiments/probes/r18_htqb_combo.py`, `results/summaries/r18_htqb_combo.json`)

Pattern correlation on decisive neurons. "LOLO" means linear maps fitted on the other layers ≥ 8 and applied to the held-out layer.

| layer | chain | V55 | star | star+path | V55+star (LOLO) | V55+star+path (LOLO) | V55 features+star+path (LOLO) |
|---|---|---|---|---|---|---|---|
| 8 | 0.376 | 0.517 | 0.501 | 0.625 | 0.578 | 0.660 | 0.730 |
| 10 | 0.412 | 0.547 | 0.503 | 0.647 | 0.599 | 0.675 | 0.741 |
| 12 | 0.345 | 0.523 | 0.521 | 0.637 | 0.619 | 0.675 | 0.761 |
| 13 | 0.290 | 0.496 | 0.549 | 0.639 | 0.607 | 0.644 | 0.741 |
| 14 | 0.304 | 0.506 | 0.486 | 0.611 | 0.596 | 0.657 | 0.742 |
| 15 | 0.270 | 0.473 | 0.489 | 0.559 | 0.578 | 0.610 | 0.711 |

The V55 per-layer features and the HTQB trees are complementary. Together they clear the 0.70 gate at every layer from 6 to 15 (0.71-0.76, held out by layer). The star alone, which is nearly free, adds +0.06 to +0.11 on top of V55.

**Caveats**
- HTQB here uses the teacher μ and C (an oracle). The deployed version must use the chain's μ and C.
- The combined map is fitted on MLP #0 only, so a second network is still needed.
- The end-to-end MSE effect is still unknown; next comes a mean-closure injection.
