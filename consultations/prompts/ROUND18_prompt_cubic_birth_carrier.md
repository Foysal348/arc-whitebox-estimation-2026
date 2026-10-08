# Round 18 — ONE question: a cheap κ₄ carrier whose births come from the CUBIC (third-order) response of each ReLU

*(Answer in English, at most 4 pages. Give ONE concrete carrier, its exact cost at the real shapes, and ONE self-contained
falsifier script, ceiling first. Run it on our data yourself if you can reach it and embed the script and its JSON; otherwise
we run it here on a Kaggle GPU. Mark every number as derived, estimated, quoted or measured. Score ONLY with the project metric
in section 4.)*

## 1. Setting (short)

- **Network and task.** Bias-free ReLU MLP, n = 1024, L = 16, He weights, x ~ N(0, I). We estimate the final-layer post-ReLU means.
- **Score.** Final MSE × FLOPs / 2⁴¹. One u = 2³¹ FLOPs ≈ one 1024³ product.
- **Our estimator.** The factorised third-cumulant chain at ≈ 192 u:
  - it carries the mean, a dense C, and κ₃ as per-ReLU sources with legs transported by U = diag(Φ) W;
  - its κ₄ is memoryless (a regenerated core; only the diagonal is carried);
  - the κ₄ diagonal enters only the univariate Edgeworth mean closure.
- **Graded today.** 3.833e-9 (raw 2.05e-8).
- **The target.** The κ₄-diagonal PATTERN. The teacher's pattern at the chain's level would take raw 2.29e-8 to 1.55e-8; a 50% blend is worth 74% of that gain.

## 2. What was settled since Round 17 (all measured on public MLP #0, N = 4×10⁶ Monte Carlo, Kaggle GPU)

**The second-order family is dead.** Any carrier whose births come from the quadratic Hermite response (L4 pass-through + Q1 C·T + Q2 C·K / T·T / path) is capped near 0.5:
- Claude's Q1 hub-pair carrier, with the κ₃ contraction exact from samples, reached 0.44 / 0.43 / 0.43 at L12 / L15 / L16 → KILL;
- every earlier carrier (SBQM, pair-innovation) also sat at ~0.44.

**Family ceiling.** We replaced the ReLU by its Hermite expansion around the teacher's per-layer μ, σ,

    h = σ Σ_{k≤K} c_k He_k((z − μ)/σ),  c_0 = αΦ + φ, c_1 = Φ, c_k = φ(α) He_{k−2}(−α)/k!  (k ≥ 2),

for the last h layers before the target. The state h layers back is a Gaussian restart with the teacher's (μ, C), so all older non-Gaussian history is erased. This is the best any h-age carrier with K-th-order births can do.

| births (h = 6 unless stated) | L12 | L15 | L16 |
|---|---|---|---|
| true ReLU | 0.859 | 0.820 | 0.818 |
| K = 2 (quadratic) | 0.548 | 0.449 | 0.530 |
| **K = 3 (quadratic + cubic)** | **0.848** | **0.799** | **0.817** |
| K = 4 | 0.865 | 0.816 | 0.822 |
| orders 0, 1, 3 only (cubic without quadratic) | 0.511 | 0.498 | 0.469 |
| K = 3, h = 4 | 0.749 | 0.687 | 0.711 |
| K = 3, h = 3 | 0.652 | 0.588 | 0.643 |
| K = 3, h = 2 | 0.517 | 0.436 | 0.485 |

At every horizon, K = 3 tracks the true ReLU to within ±0.02. Requirements:
- **(a)** births must include BOTH the quadratic and the cubic response;
- **(b)** K = 3 is enough;
- **(c)** the carrier needs a horizon of about 5-6 layers (4 is borderline, 3 fails).

## 3. The cubic birth terms (exact; verified numerically on an exact empirical measure, max relative error ≤ 9e-15)

**Response model** (u = z − μ; g_a = φ(α_a)/(2σ_a); c_a = −α_a φ(α_a)/(6σ_a²)):

    h_a − E h_a ≈ Φ_a u_a + g_a (u_a² − v_a) + c_a (u_a³ − 3 v_a u_a)

κ(h_a, h_b, h_c, h_d) is multilinear in the three pieces L = Φu, Q = g(u² − v) and Cu = c(u³ − 3vu): sum over all assignments of a piece to each argument.

**Verified pieces**

- (L4) Φ_a Φ_b Φ_c Φ_d K_abcd — the pass-through, which carries memory.
- (Q1) g_a Φ_b Φ_c Φ_d [κ5_aabcd + 2 (C_ab T_acd + C_ac T_abd + C_ad T_abc)]
- (Q2) g_a g_b Φ_c Φ_d [κ6_aabbcd + 2 C_ac K_abbd + 2 C_ad K_abbc + 2 C_bc K_aabd + 2 C_bd K_aabc + 4 C_ab K_abcd
       + 4 T_abc T_abd + 2 T_aab T_bcd + 2 T_abb T_acd + 4 C_ab (C_ac C_bd + C_ad C_bc)]
- **(Cu1, new)** c_a Φ_b Φ_c Φ_d [κ6_aaabcd + 3 (C_ab K_aacd + C_ac K_aabd + C_ad K_aabc)
       + 3 (T_abc T_aad + T_abd T_aac + T_acd T_aab) + 6 C_ab C_ac C_ad]
  (the −3 v_a u_a part of He3 cancels a 3 v_a K_abcd term exactly)

Mixed assignments (for example one Q and one Cu, or two Cu) are of the same or higher order. We have not written them out; the K = 3 Monte Carlo ceiling above contains all of them.

**Verified readouts** (row convention z′ = h W, U = diag(Φ) W, C U the product the chain already forms):
- Q1 C·T: 24 Σ_a g_a W_ai (C U)_ai T(a; U_:i, U_:i)
- Q2 Gaussian path: 48 v_iᵀ C v_i, with v_a = g_a W_ai (C U)_ai
- **Cu1 Gaussian star: 24 Σ_a c_a W_ai ((C U)_ai)³** — elementwise after one existing product, so almost free.

## 4. The project metric (use exactly this; other metrics are not comparable)

    dec = |mu_t / sqrt(v_t)| < 1.5                    # teacher pre-activation mean / variance at target layer t
    corr = pearson( (pred / v_t**2)[dec], (K4d_t / v_t**2)[dec] )

**Do NOT score raw κ₄ or a top-k set.** Claude's raw-κ₄ / top-128 metric gave 0.80-0.87 for a carrier whose project-metric correlation was 0.43. The raw diagonal contains v², which the chain already gets right.

## 5. The question

**Design the cheapest factorised carrier with these properties:**
- its births at every ReLU contain the quadratic AND the cubic response (section 3), plus the L4 pass-through of earlier births;
- it is carried for 5-6 ages;
- its diagonal readout reaches ≥ 0.70 on the project metric at L12, L15 and L16.

**State:**
1. which terms you keep (Q1, Q2 parts, Cu1 parts, mixed terms) and why;
2. how each is factorised (hub / source legs; the chain's κ₃ sources are available as factors) and transported;
3. the readout;
4. the truncations (age, hub rank, sectors) and the cost.

Note that the Gaussian parts (path, star) are nearly free to read out one step ahead. Say whether they carry pattern, or only level, before relying on them.

**Cost envelope (derived).** A carrier worth 74% of the oracle gain multiplies raw MSE by ≈ 0.78. At +30 u on ≈ 192 u the score improves ≈ 10%; at +60 u it is already ≈ 2% worse. Target ≤ ~30 u, ≤ ~1,000 extra metered ops and ≤ 8 GB.

**Falsifier (ceiling first)**
- Exact Monte Carlo through the true MLP #0, float32 forward, float64 accumulation, N = 2×10⁶.
- Births use cumulants or contractions computed from exact samples of the true network, as in Claude's script: T(a; x, x) = E[u_a (u·x)²] from samples.
- Print the project metric at L12 / L15 / L16 for the untruncated carrier and for the shippable truncation.
- Gate: PASS ≥ 0.70 at all three; KILL < 0.55 at all three.
- Templates you may reuse:
  - `experiments/probes/r17_claude/r17_q1_hubpair_falsifier.py` (Claude's Q1 script with our metric);
  - `experiments/probes/r17_hermite_family_ceiling.py` (the family ceiling).

## 6. Data (paths relative to the project root; also packed for Kaggle)

- **Packed teacher.** `kaggle/r17/data/r17_data.npz` holds:
  - W (16 × 1024 × 1024; z_{l+1} = h_l @ W[l]);
  - teacher mean (17 × n), C (17 × n × n) and K4d (17 × n), index = layer.
- **Full teachers.** `data/kaggle_out/kaggle_out_k4pub/k4teach/layer{l}_seed0.npz` (mean, C, Y21, K22, K31, K4d).
- **Post-ReLU teachers, layers 3/7/11/14/15.** `data/kaggle_out/kaggle_out_k4post/k4post/`.
- **Dense captures of the chain's κ₃ sources.** `data/dumps/r7_cap_L{1..16}.npz`; reconstruction via `blocks_of_source` in `experiments/probes/r7_srcgain.py`.
- **The graded estimator.** `submissions/v55As125/estimator.py`.

## 7. Time budget

Today is 8 Oct 2026; the competition closes 18 Oct (Bangladesh time). We need the ceiling verdict by 10 Oct and a prototype by 12 Oct.
