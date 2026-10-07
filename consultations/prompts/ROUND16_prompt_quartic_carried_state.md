# Round 16 — ONE question: a cheap QUARTIC carried state with ≥ 6-layer memory

*(Answer in English, at most 4 pages. Give ONE concrete state (at most two), its exact cost at the real shapes, and ONE
falsifier that you RUN on our stored data, as in Round 15. Mark every number as derived, estimated, quoted or measured.
Do not re-propose anything in the dead list unless you show why its kill was too narrow.)*

## Setting (short)

The network is a bias-free ReLU MLP of width n = 1024 and depth L = 16, with He weights and input x ~ N(0, I). The task is to estimate the final-layer post-ReLU means.

Score = final MSE × FLOPs / 2⁴¹. One u = 2³¹ FLOPs ≈ one 1024³ product; a level-5 Strassen product ≈ 0.51 u.

Our estimator is the public factorised third-cumulant chain:
- it carries the mean, a dense covariance C, and κ₃ as a sum over per-ReLU "sources";
- young sources have dense legs; old sources share a low-rank basis;
- its κ₄ is memoryless: at each layer a core diag(g) + λ_l C_pre,off is regenerated, and only its diagonal g is carried by (W∘W);
- the κ₄ diagonal enters ONLY the univariate Edgeworth mean closure.

Graded now: raw MSE 2.06e-8 at utilisation 0.187 (≈ 192 u), score 3.86e-9. The grader allows ≤ ~18,000 metered operations per network.

Public leaderboard for comparison:
- J2W: raw 1.47e-8 at utilisation 0.102.
- Masato: raw 1.46e-8 at 0.151; jumped from #27 to #11 with 10 submissions.

## What is established (measured on public MLP #0; teacher = 6×10⁷-sample Monte Carlo)

1. **The accuracy gap is a κ₄-diagonal PATTERN gap.** The teacher's κ₄ diagonal fed into the mean closure, coupled from layer 1:
   - full chain 2.290e-8 → 1.609e-8;
   - the teacher's pattern rescaled to the chain's per-layer level 1.551e-8 (108% of the gain);
   - a 50% blend 74%;
   - layers ≥ 8 only 86%.

   The chain's level is right; its per-neuron pattern is wrong. Pattern correlation chain vs teacher is 0.24-0.41 at layers 7-16 (decisive neurons |μ/σ| < 1.5).
2. **The pattern is carried memory.**
   - Exact LINEAR carry of the TRUE full κ₄ tensor of the previous layer, y = (Φ(α) ∘ (z − μ)) W, gives pattern correlation 0.77-0.80 at layers 12-16.
   - Erasing all non-Gaussian history h layers back leaves correlation 0.41 (h = 2), 0.59 (h = 3), 0.77 (h = 6).
   - Fresh ReLU physics from a Gaussian input is only 1-4% of the deep κ₄ level.
3. **It lives in the all-distinct sectors.** The next diagonal is ~90% the (2,2) pair term 3 Σ_{a≠b} W_ai² W_bi² K22_ab by norm. Yet transporting the TRUE post-ReLU (4) and (2,2) slices one step gives the right level and only correlation 0.24-0.59. The pattern sits in the (2,1,1) and (1,1,1,1) sectors.
4. **It is odd in the later weights.** Statistics that are even in the weights (per-neuron variances, kurtosis, ensemble averages) carry no conditional information about it (parity theorem; measured correlation ≤ 0.04).
5. **Per-layer chain features saturate at correlation ≈ 0.5 (shipped as V55).** A linear map uses:
   - (κ₃ᵢᵢᵢ/v^1.5)²;
   - the row energy of the standardised (2,1) slice;
   - the correlation-row energy of C_pre;
   - the chain's own κ₄ᵢᵢᵢᵢ/v².

   The map is rescaled to the chain level and used in the mean closure only. It reaches correlation 0.41-0.56, and on the grader −5.4% (3.858e-9). On 100 public networks it is better on 97/100 (×0.9455).

   These additions do not help:
   - squares, products and logarithms of those features;
   - the previous layer's features carried by (W∘W)ᵀ;
   - re-weighting the fit by each neuron's mean-closure lever (worse end to end, ×1.004).

## Measured dead (do not propose)

**κ₄ readouts of cubic or per-neuron information**
- Round 15 (yours, reproduced by us), source-resolved skew energy E_i = Σ_b B_bi²/v³ from the signed per-birth κ₃ diagonal: +0.004 correlation even at a per-layer oracle fit. The full cubic source history does not carry the κ₄ pattern.
- κ₄ riding the κ₃ source legs (×1.05 out of sample).
- Learned per-neuron or 13-feature regressions across networks.

**Carriers and regenerated cores**
- A young-source κ₄ carrier over 2-3 layers (out of sample 0.210 → 0.203 error). Note: its horizon was shorter than the measured 6-layer memory. Argue if you think this kill was too narrow.
- Scale-mixture / Sym(C⊗C) κ₄ predictors (correlation ≤ 0.09).
- Fresh-physics κ₄ kernels.
- A carried pair-response mode (×0.98-1.00).
- SSC response closure (×1.006-1.023).
- Role-split or per-layer refits of λ.

**Closed-form terms and trade-offs**
- The exchange term 3 Y C⁻¹ Yᵀ: pattern correlation −0.07 … −0.23, and its level is 10× the truth.
- Using each κ₃ source once and carrying only (m, C): +4.85e-7.
- Cutting κ₃ cost and recovering with κ₄: κ₄ helps less the cheaper κ₃ gets (×0.70 on the full chain, ×0.88 / ×0.97 / ×1.02 on cheaper ones).

## The question

**Design ONE carried quartic state, and its readout of the per-neuron κ₄ diagonal, with all of the following properties:**
- it carries κ₄ content born at ReLUs across ≥ 6 layers;
- it includes the all-distinct sectors;
- it reaches pattern correlation ≥ 0.7 with the teacher at layers 12-16;
- it costs ≤ ~30 u for the whole network and ≤ ~1,000 extra metered operations.

The level stays the chain's (rescale per layer). You may reuse anything the chain already carries:
- μ and C;
- the κ₃ sources' dense legs and shared-basis factors;
- the (2,1) slice D21 and κ₃ diagonal D3;
- the regenerated core.

If no state can meet the cost, give the cheapest one that meets the CEILING (correlation ≥ 0.7 when computed exactly, unfactorised) and say what it costs. We then decide whether to pay for it with cost trims elsewhere.

**Deliverable**
1. The state: shapes, birth rule at each ReLU, transport rule through W and Φ, and the readout. Give exact equations.
2. Cost per layer in dense products, FLOPs and metered operations.
3. Why it carries the 6-layer memory and the (2,1,1)/(1,1,1,1) sectors, and why it is not one of the dead items.
4. ONE falsifier that you RUN on our stored data, ceiling first: the exact, unfactorised version of your state, scored by pattern correlation at L12-16 on decisive neurons. Then the cheap version if the ceiling passes.

## Stored data you can use (same project paths as in Round 15)

- **Teachers, MLP #0, all 16 layers:** mean, C, D21 slice, K22, K31 and the κ₄ diagonal (`data/kaggle_out/kaggle_out_k4pub/k4teach/layer{l}_seed0.npz`) and the weights (`weights_seed*.npz`).
- **Synthetic second network:** the same teachers at layers 2, 4, 6, 8, 10, 12, 14 and 15 (`data/kaggle_out/kaggle_out_k4/k4teach`).
- **Post-ReLU teachers** at layers 3/7/11/14/15.
- **Dense captures of the chain's κ₃ sources** at ALL layers 1-16: `data/dumps/r7_cap_L{1..16}.npz`. The factor reconstruction is `blocks_of_source` in `experiments/probes/r7_srcgain.py`, as you used in Round 15.
- **The chain's own κ₄ diagonal per layer:** `data/dumps/errbudget_v29plain.npz` (`L{l}_ck4`).
- **The current graded estimator:** `submissions/v55k4c/estimator.py`.

## Time budget

The competition closes 18 Oct 2026 (Bangladesh time). We need a ceiling verdict within 1-2 days and a working prototype by 12 Oct. A state whose ceiling passes but costs more than 30 u is still useful: say so clearly.
