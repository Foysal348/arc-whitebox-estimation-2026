# Team Hack2Publish — Research Effort Summary

**ARC White-Box Estimation Challenge 2026, Phase 2.** Team: mdhamidhosen, esfer_sami, foyshal_sk. Status as of 21 October 2026.

Counted from the repositories and experiment ledgers, the team ran **more than 500 experiments**. Because the counts come from files, a few experiments may be counted in two places; the overall scale is correct.

## Architectures and new methods

- **More than 60** distinct architecture and representation ideas were tested to a decision.
- **20 consultation rounds** with outside AI systems (GPT, Astra, Gemini, Grok, DeepSeek, Fable and others); every proposal was measured before use.
- The kill register lists **about 80** measured and closed avenues.

**Architectures that survived and were submitted:**

| Architecture | What it does | Graded effect |
|---|---|---|
| V31 | Spherical-input identity (exact, zero FLOPs) | raw −1.1 % |
| V55 | κ₄-diagonal pattern corrector of the mean closure | −5.4 % |
| V56 | Hermite cubic-star feature for the κ₄ pattern | −4.4 % |
| E-series calibration (md.hamidhosen) | Gauss-Newton refit of 24 calibration groups and the κ₄ corrector | −3.9 % |
| e30_a + κ₄ corrector | Our corrector on another team's public (MIT) chain | −2.1 % vs e30_a |
| e299safe | Fixed-basis old level + C_pre compaction in our own lineage | −2.4 % |
| Refitted table (N2 → N4a) | e30_a's correction table refitted with our non-local features | −6.3 % |

## Experiments and measurements

| Item | Count |
|---|---|
| Probe and measurement scripts (V-series) | 130 |
| Estimator variants (with knobs or oracles) | 50 |
| Result files | 280 |
| 100-MLP Kaggle harness runs (result folders) | 159 |
| Kaggle kernels created | 132 |
| E-series ledger (md.hamidhosen) | 133 experiments, 117 estimator versions |
| Planned-experiment queue | 62 |

## Grader submissions

- About **134 graded submissions** in total for the team (board count, 20 October): about 66 E-series submissions by md.hamidhosen and the rest from the V-series and the final-days line.
- Score improved from **5.418e-9 to 2.7858e-9 (−48.6 %)**.
- Best public rank was **24** (26 September); the rank on 20 October is **34** (displayed 2.8e-9).

## Writing and verified results

- Main write-up: 12 pages. Full technical report (appendix): 58 pages.
- Four mathematical results verified: the parity theorem, the proof of the √(6/(r+8)) floor, the sign of the Gaussian star, and the direct measurement of channel-error cancellation.
