# Round 17 — ONE question: the cheapest FACTORISED κ₄ carrier that keeps the all-distinct post-ReLU sectors

*(Answer in English, at most 4 pages. Give ONE concrete carrier, its exact cost at the real shapes, and ONE falsifier as a
SELF-CONTAINED PYTHON SCRIPT that we run here: you cannot reach our data, so we execute it and report back. Mark every number
as derived, estimated, quoted or measured. Do not re-propose anything in the dead list unless you show why its kill was too narrow.)*

## Setting (short)

The network is a bias-free ReLU MLP of width n = 1024 and depth L = 16, with He weights and input x ~ N(0, I). The task is to estimate the final-layer post-ReLU means.

Score = final MSE × FLOPs / 2⁴¹. One u = 2³¹ FLOPs ≈ one 1024³ product; a level-5 Strassen product ≈ 0.51 u.

Our estimator is the public factorised third-cumulant chain at ≈ 192 u:
- it carries the mean, a dense covariance C, and κ₃ as a sum over per-ReLU "sources";
- young sources have dense n × n legs; old sources share a rank-352 basis;
- every source leg is transported each layer by U = diag(Φ(α)) W.

Its κ₄ is memoryless: at each layer a core diag(g) + λ C_pre,off is regenerated, and only the diagonal is carried, by (W∘W). The κ₄ diagonal enters only the univariate Edgeworth mean closure.

Graded now: raw 2.06e-8 at utilisation 0.187, score 3.86e-9. The grader allows ≤ ~18,000 metered operations per network.

Leaderboard reference:
- J2W: raw 1.47e-8 at utilisation 0.102.
- Masato: raw 1.46e-8 at 0.151.

## Established (public MLP #0; teacher = 6×10⁷-sample Monte Carlo)

1. **The accuracy gap is the PATTERN of the deep κ₄ diagonal.** The teacher's diagonal pattern, at the chain's level, fed into the mean closure takes raw 2.29e-8 → 1.55e-8 (108% of the full gain). A 50% blend gives 74%; layers ≥ 8 give 86%.
2. **The pattern is carried memory.**
   - The exact LINEAR carry of the full true pre-activation κ₄ tensor one layer, y = (Φ(α) ∘ (z − μ)) W, gives decisive-neuron pattern correlation 0.77-0.80 at L12-16 (Monte Carlo: `experiments/probes/opus12_k4_carry.py`, LIN).
   - Erasing the non-Gaussian history h layers back leaves correlation 0.41 (h = 2), 0.59 (h = 3), 0.77 (h = 6).
   - Fresh ReLU physics from a Gaussian input is only 1-4% of the deep level.
3. **NEW (Round 16, measured): the missing pattern lives in the post-ReLU entries κ(a,a,b,c) and κ(a,b,c,d).** One step from the TRUE post-ReLU teacher state:

   | state carried one step | pattern corr at L8 / L12 / L15 / L16 |
   |---|---|
   | (4), (3,1) and (2,2) slices, carried exactly | 0.45 / 0.39 / 0.30 / 0.38 |
   | full tensor | 0.77-0.80 |

   So about 0.4 of correlation comes only from entries with three or four distinct indices. Every slice-based or matrix-based state is capped near 0.4.
4. **Per-layer chain features saturate at ≈ 0.5 (V55, shipped):** −5.4% on the grader.
5. **The κ₄ is odd in the later weights** (parity), so even statistics carry nothing.

## Measured dead (do not propose)

**Matrix or slice states (capped by fact 3)**
- Round 16 SBQM (Sym(H ⊗ C), H carried by Wᵀ H W, diagonal refresh). Teacher-forced one-step ceiling 0.32-0.39 at L12/15/16, the same as carrying the diagonal alone. The K31/K22 slices cannot both be fitted by Sym(H ⊗ C).
- CPR, one or two carried response matrices H ← Uᵀ H U with births and a pair readout (×0.98-1.00).
- Scale-mixture / Sym(C ⊗ C) predictors (correlation ≤ 0.09).
- A carried pair-response mode; SSC response closure.

**Readouts of cubic history**
- Round 15 source-resolved skew energy: +0.004 correlation at a per-layer oracle. The signed cubic history does not predict κ₄.
- κ₄ riding the κ₃ source legs (×1.05 out of sample).

**Short horizons and closed forms**
- A young-source κ₄ carrier over only 2-3 layers (0.210 → 0.203 error). Its horizon is below the measured 6 layers: you may argue it was too narrow.
- The exchange term 3 Y C⁻¹ Yᵀ; per-layer refits of λ; learned per-neuron regressions; cutting κ₃ cost and recovering it with κ₄.

## The question

**Design the CHEAPEST factorised representation of the carried pre-activation κ₄ tensor that keeps its all-distinct sectors.**
- Its diagonal must reach decisive-neuron pattern correlation ≥ 0.7 at L12-16.
- The representation must be a sum over births (ReLUs / layers) of structured terms, like our κ₃ sources, carried for ≥ 6 layers.
- State the birth term at a ReLU: which products of the chain's existing quantities (C, κ₃ source legs, Φ, φ) create the (2,1,1) and (1,1,1,1) entries.
- State how each term is transported by U = diag(Φ) W and how the diagonal is read out.
- State what can be truncated (age, rank, sector) while keeping correlation ≥ 0.7.

**Cost envelope (derived break-even)**

A carrier worth 74% of the oracle gain multiplies raw MSE by ≈ 0.78. At +30 u on ≈ 192 u the score improves ≈ 10%; at +60 u it is already ≈ 2% worse. So it must cost ≤ ~30 u, ≤ ~1,000 extra metered operations, and fit 8 GB of memory.

If nothing meets the cost, give the cheapest carrier that meets the CEILING (exact, untruncated) and its cost.

**Deliverable**
1. The equations: birth, transport and readout, with shapes.
2. The cost per layer: dense products, FLOPs and metered operations.
3. Why it captures the κ(a,a,b,c) and κ(a,b,c,d) sectors with ≥ 6-layer memory, and why it is not a dead item.
4. ONE self-contained falsifier script, ceiling first. It must run the exact (untruncated, Monte-Carlo-backed if needed) version of your carrier on public MLP #0 and print decisive-neuron pattern correlation at L12, L15 and L16.
   - Follow the Monte Carlo template of `experiments/probes/opus12_k4_carry.py`: exact samples through the true MLP, float32 forward, float64 accumulation, N = 2×10⁶ by default.
   - It must run on one CPU or GPU in ≤ ~2 hours. We will run it on Kaggle if needed.
   - Pre-register the gate: PASS ≥ 0.70 at all of L12/L15/L16; KILL if < 0.55 at all three.

## Stored data (paths relative to the project root)

- **Pre-activation teachers, MLP #0, all 16 layers:** mean, C, D21 = Y21, K22, K31 and K4d, in `data/kaggle_out/kaggle_out_k4pub/k4teach/layer{l}_seed0.npz`, plus `weights_seed0.npz` (W stacked 16 × 1024 × 1024, row-vector convention z_{l+1} = h_l @ W[l]).
- **Post-ReLU teachers, layers 3/7/11/14/15:** the same arrays, in `data/kaggle_out/kaggle_out_k4post/k4post/layer{l}_post_seed0.npz`.
- **Slice conventions:** K31_ij = κ(i,i,i,j); K22_ij = κ(i,i,j,j).
- **Synthetic second network:** pre-activation teachers at layers 2, 4, 6, 8, 10, 12, 14 and 15, in `data/kaggle_out/kaggle_out_k4/k4teach`.
- **Dense captures of the chain's κ₃ sources at all layers:** `data/dumps/r7_cap_L{1..16}.npz`. Factor reconstruction is `blocks_of_source` in `experiments/probes/r7_srcgain.py`.
- **The chain's own κ₄ diagonal:** `data/dumps/errbudget_v29plain.npz` (`L{l}_ck4`).
- **The graded estimator:** `submissions/v55k4c/estimator.py`.

## Time budget

Today is 8 Oct 2026; the competition closes 18 Oct (Bangladesh time). We need the ceiling verdict by 10 Oct and a prototype by 12 Oct. A carrier whose ceiling passes but costs more than 30 u is still worth reporting.
