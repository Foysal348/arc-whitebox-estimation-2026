# Fable design (9 Oct 2026): the lagged κ₃-dressed vertex carrier (LKV) for the κ₄ pattern, with a κ₄→κ₃ feed as stage 2

Answer to `consultations/prompts/FABLE_architecture_prompt_2026-10-09.md`. Every number is marked **measured** (file), **derived** (arithmetic on measured numbers) or **estimate**. Nothing below was run at full size; the only runs are the two smoke runs in §8.

Files delivered:
- this document;
- `experiments/probes/fable_ceiling_2026-10-09.py` — the exact, untruncated ceiling (Monte Carlo on MLP #0, `kaggle/r17/data/r17_data.npz`), accepts `N` and `device`, writes the κ₄ predictions in the teacher-file layout for the end-to-end injection;
- `experiments/probes/fable_component_2026-10-09.py` — the same readouts from the chain's own κ₃ legs (`data/dumps/r7_cap_L{m}.npz`), laptop-light.

## 0. One-paragraph summary

The deep κ₄ diagonal pattern is carried memory (exact linear carry of the full κ₄ tensor one step: 0.771 / 0.785 / 0.797 at L12 / L15 / L16, **measured**, `results/summaries/opus12_k4_carry.json`), and the K = 3 Hermite family reaches it (0.848 / 0.799 / 0.817 over 6 ages, **measured**, `r17_hermite_family_ceiling_A.json`) while the quadratic-only family (0.548 / 0.449 / 0.530) and the cubic-only family (0.511 / 0.498 / 0.469, `_B.json`) are each capped near 0.5. The cubic-only family contains no κ₃ at all (every κ₃ birth needs an even total Hermite degree), so it is the Gaussian star plus C·K dressing, and it scores the same as the star alone (0.511 vs 0.521): **C·K dressing of κ₄ is worthless**. The quadratic-only family is the Q1 C·T conversion of κ₃ plus the Gaussian path (Q1 with exact κ₃: 0.442 / 0.432 / 0.431, `consultations/round17_verdicts.md`; path 0.074 / 0.024 / 0.150, `round18_htqb.json`). The ≈ +0.3 that K = 3 adds over the union of both families must therefore sit in the **cross-family terms, where κ₃ born by quadratic vertices is consumed at a cubic vertex**. At first order in the births these are Cu1's T·T term (κ₃ squared at a cubic vertex) and the mixed Q×Cu term (C·C·T). Neither has ever been measured. The design carries the T·T birth (plus the free C·T birth, which shares its expensive object) on the chain's existing κ₃ legs, truncated by a rank-r sketch of the vertex layer's legs, at an estimated +9 to +14 u and ≈ 700 metered ops. Stage 2, untested, feeds the (2,1,1) slice of the carried κ₄ back into κ₃ through the chain's existing thin-leg machinery; it is the only route to the joint oracle's synergy.

## 1. The mechanism

### 1.1 Notation (chain conventions)

Row convention z_{l+1} = h_l W[l], n = 1024, L = 16. At the ReLU of layer m: pre-activation z_m with the chain's mean μ and variance v = σ², α = μ/σ, Φ = Φ(α) (= the chain's wick w₁), φ = φ(α), u = z_m − μ. K = 3 response (ROUND18 §3, verified ≤ 1e-14):

    h − E h = Φ u + g (u² − v) + c (u³ − 3 v u),   g = φ/(2σ),   c = −α φ/(6 v)        (c is V56's per-slot k4s_c)

Legs are n×n arrays indexed [neuron i, hub h]; the chain transports every leg by leg ← (diag(w₁,l) W[l])ᵀ leg. The chain's κ₃ at the pre-activation stage of layer m is a sum over sources s < m of hub blocks (`blocks_of_source`, `experiments/probes/r7_srcgain.py`):

    T_abc(m) = Σ_{s<m} Σ_blocks Σ_h Sym(X_s[a,h] P_s[b,h] Y_s[c,h]),   blocks (X1, P, Y1), (M, P, P), (X3, P, Y3)

with X1 = 3A + Xt, Y1 = A·w2b + Yt, M = P·s + 3A·e + Z Lᵀ, i.e. every dense left factor is A_s or P_s with a static per-hub column scaling, plus thin (rank-16) terms.

Transports from vertex m to target t:

    Ẏ_{m→t} = diag(w₁,m) W[m] diag(w₁,m+1) W[m+1] ⋯ diag(w₁,t−1) W[t−1]      (n×n; linear-response legs; the chain's leg transport)
    Ȳ_{m→t} = W[m] diag(w₁,m+1) ⋯ W[t−1]                                       (hub leg: no w₁ on the hub)
    X_m(t)  = (C_off(m) + diag v) Ẏ = A_m(t) + P_m(t) diag(w₁ v)               (the C-leg; exactly what V56 reads from the newborn slots)
    YL_m(t) = (D21(m) + diag D3(m)) Ẏ                                          (the (2,1)-slice leg, NEW)

### 1.2 Births (κ₄ at the pre-activation of layer m+1, before transport)

From the Cu1 and Q1 identities (ROUND18 §3; `experiments/probes/r17_curvature_identities.py`, max rel err ≤ 9e-15):

    B^TT_abcd = c_a Φ_b Φ_c Φ_d · 3 (T_abc T_aad + T_abd T_aac + T_acd T_aab)   summed over the 4 positions of the cubic piece
    B^CT_abcd = g_a Φ_b Φ_c Φ_d · 2 (C_ab T_acd + C_ac T_abd + C_ad T_abc)      summed over the 4 positions of the quadratic piece
    B^star    = c_a Φ_b Φ_c Φ_d · 6 C_ab C_ac C_ad                               (shipped, V56)

The terms NOT carried, with the reason: Cu1 C·K and Q2 C·K (worthless by the K013 result, §0); κ5, κ6 pieces (not representable, expected small; the ceiling script measures them as the residual Cu1 − star − tt − ck and Q1 − ct); the Gaussian path (0.07, measured); the mixed Q×Cu C·C·T term and the multi-piece terms (measured by the ceiling script as `QCu`, `CuCu`, `higher`; if `QCu` turns out to be the large one, see §7, INCONCLUSIVE branch).

### 1.3 Transport

Every leg of B^TT and B^CT is transported by Ẏ (hub by Ȳ); this is the chain's own leg transport, so the κ₄ state is a linear functional of objects the chain already carries plus the new YL leg. The diagonal at target t = m + j (j = age) is, exactly (derived from the identities by the same contraction as the readout factors 24 / 48 of `r17_curvature_identities.py`):

    tt_i(t) = 36 Σ_a c_a Ȳ_ai YL_ai Q_ai,      ct_i(t) = 24 Σ_a g_a Ȳ_ai X_ai Q_ai,      star_i(t) = 24 Σ_a c_a Ȳ_ai X_ai³

    Q_m(t)[a, i] = Σ_bc T_abc(m) Ẏ_bi Ẏ_ci = T(a; Ẏ_i, Ẏ_i)
                 = (1/3) Σ_{s<m} Σ_blocks [ X_s(m) (P_s(t)∘Y_s(t))ᵀ + P_s(m) (X_s(t)∘Y_s(t))ᵀ + Y_s(m) (X_s(t)∘P_s(t))ᵀ ]      (n×n)

Q is "the chain's D21 contraction with the left factors taken at the vertex layer m instead of at t": the Hadamard right factors at t are exactly the fused right factors the chain forms for its own D21(t) (`_dslices`: two fused einsums per layer with right factors A and P), so they are free; the left factors are the legs of the sources at the vertex layer, which are static once m has passed. At lag 1, Q_{t−1}(t)ᵀ Ẏ... = D21(t) is the chain's own slice (this is the re-association behind candidate rank 4 of `ARCHITECTURE_CANDIDATES_2026-10-09.md`).

### 1.4 Truncation (the deployable form)

- Vertex window J = 4 ages (set C: ages 1–4 give 0.736 / 0.682 / 0.713, **measured**, `_C.json`); sources s ≥ t − 5 (the chain's dense young legs).
- Rank-r sketch of the vertex index: Q ≈ Ω_m Ω_mᵀ Q with Ω_m (n×r) the top-r eigenvectors of the chain's C_pre(m) (one range finder per layer, r = 64 default; 32 and 128 are measured by the ceiling script). The stored memory per vertex is Ω_mᵀ A_s(m), Ω_mᵀ P_s(m) (r×n each, static) for its sources.
- Thin YL: D21(m) ≈ Q_fb B (the chain's rank-R_FB range finder of D21, the same factors that feed the V18 thin legs); YL ≈ Q_fb (B Ẏ), an n×r_y leg transported at n² r_y per layer (r_y = 16 default; 64 measured).
- Readout to the mean closure: tt and ct (summed over ages ≤ J, divided by v²) enter the V56 map as two extra standardised features and are level-rescaled to the chain's layer mean exactly as V55/V56 do (the pattern is worth 108 % of the oracle gain, paper §5.1). Layers ≥ 9 only (the oracle puts ≈ 70 % of the value at L ≥ 12, `k4oracle2_target_*.json`).

Array shapes at n = 1024: Q, X, YL, Ȳ, Ẏ: n×n per (m, t) pair (formed and discarded per target; with the sketch, Q never exists at full rank: only (Ω_mᵀ A_s(m)) (n×n right factor)ᵀ = r×n pieces). Memory: 4 vertices × ≤ 5 sources × 2 × r × n × 4 B = 10 MB at r = 64; YL thin legs 4 × n × r_y × 4 B.

### 1.5 Stage 2 (second change, untested): the κ₄→κ₃ feed from the carried state

At the next quadratic vertex the κ₃ birth contains the κ₄-dressed term (partition formula; the same bookkeeping as the Q1 identity): κ(u_a² − v_a, u_b, u_c) = κ₄(a,a,b,c) + 2 C_ab C_ac. The chain feeds this term from its memoryless core (`K4->K3 feed: X3 = d(w1) G_pre d(w1)`, V17 docstring). The (2,1,1) slice of the carried TT state has the hub form Σ_h G[:,h] ⊗ P_s(t)[:,h] ⊗ Y_s(t)[:,h] with G = (c∘Ȳ∘YL)ᵀ Ω_m (Ω_mᵀ X_s(m)), a rank-r leg, i.e. a thin-leg block (G₁ = (c∘Ȳ∘YL)ᵀ Ω_m transported, G₂ = Ω_mᵀ X_s(m) static) attached to source s, exactly the chain's Z Lᵀ machinery at O(n² r) per layer. This is the only mechanism in the design that can produce the joint oracle's synergy (κ₃ × κ₄: 0.607 × 0.686 = 0.417 if independent vs 0.272 measured, `r20_joint_oracle.json`). It is proposed, not measured, and it is gated on stage 1 passing.

## 2. Which measured facts make it plausible

| Fact | Value | Source |
|---|---|---|
| Deep κ₄ pattern is linear memory of the full tensor | 0.771 / 0.785 / 0.797 | **measured**, `opus12_k4_carry.json` |
| K = 3 births over 6 ages reach the restart ceiling | 0.848 / 0.799 / 0.817 (true ReLU 0.859 / 0.820 / 0.818) | **measured**, `r17_hermite_family_ceiling_A.json` |
| Quadratic-only family capped | 0.548 / 0.449 / 0.530 | **measured**, `_A.json` |
| Cubic-only family (orders 0, 1, 3) capped, and equal to the star alone | 0.511 / 0.498 / 0.469 vs star 0.521 / 0.489 / 0.471 | **measured**, `_B.json`, `round18_htqb.json` |
| Orders 0, 1, 3 carry no κ₃ (odd-degree Gaussian cumulants vanish) | — | **derived** (Hermite degree parity) |
| Chain κ₃ (2,1) slice is at cubic-order per-step accuracy | K3 one step 2.7 %, chain 4.6–6.6 % accumulated | **measured**, `r20_onestep_channels.json`, paper §4.3 |
| Chain κ₄ pattern is the weak channel | 0.29–0.34 vs K3 one step 0.99 | **measured**, same files |
| Ages are complementary; ages 1–4 suffice for ≈ 0.70 | 0.736 / 0.682 / 0.713 | **measured**, `_C.json` |
| The star readout from the chain's own legs is exact | V56 star fidelity 1.000 at 14→15 (2 sources) | **measured**, §8 smoke |
| Joint oracle, synergy | 0.272 vs 0.607 × 0.686 | **measured**, `r20_joint_oracle.json` |

## 3. Why it is not a dead item

- Every killed κ₄ carrier (SBQM, pair-innovation, Q1 hub-pair, GHQ-6, CSF; `docs/architecture_research/KILL_AUDIT.md` C11–C15, `consultations/round16..17_verdicts.md`) has births of second order in the response or pure-Gaussian. The T·T birth is at the cubic vertex and quadratic in κ₃; no previous carrier contained it (`ARCHITECTURE_CANDIDATES_2026-10-09.md` rank 4: "TT births ... unmeasured").
- It is not HTQB's path (Gaussian Q2) and not a C·K dressing.
- It is not "κ₄ riding κ₃ legs" (C13, ×1.05): no κ₄ mode is fitted per source; the quantity is an exact cumulant identity evaluated on the chain's κ₃, with no free coefficient except the V56 map's.
- It is not the young κ₄ carrier C09, which was killed on the κ₄ LEVEL at h ≤ 6; the protocol now scores the pattern, and set C shows ages 1–4 carry 0.70 of it.
- Q1 (C·T) alone is dead at 0.44 and is included only because Q is shared; the gates are written on the increment that T·T brings over star + ct.
- The cost side uses the chain's own fused right factors and its rank-r sketch machinery; nothing in the dead cost list (ranks, age gates, closures) is re-proposed.

## 4. Cost (estimate; n = 1024, L = 16, u = 2³¹ FLOPs; nothing metered yet)

Per target t ∈ {9..16} (8 targets), vertices m = t−1..t−4, sources s with t−5 ≤ s < m: (s, m) pairs 4 + 3 + 2 + 1 = 10.

| item | count | FLOPs each | total |
|---|---|---|---|
| sketched Q: (fused right factor, n×n) × (Ω_mᵀ A_s(m) or Ω_mᵀ P_s(m))ᵀ (n×r) | 2 per pair → 20 per target | 2 n² r = 1.34e8 at r = 64 | 1.25 u per target → **10 u** (r = 32: 5 u; r = 128: 20 u) |
| sketch Ω_m (range finder on C_pre(m), 2 passes) | 1 per layer | ≈ 6 n² r | ≈ 0.2 u per layer → 2 u |
| thin YL transport | 4 vertices per layer | 2 n² r_y | 0.5 u total at r_y = 16 |
| Ȳ (hub leg without w₁): divide c and g by w₁ (floored) instead | 0 | — | 0 |
| readouts Σ_a c Ȳ YL Q, Σ_a g Ȳ X Q, over r-space | elementwise, 10 pairs per target | ≈ 6 n² | negligible |

Total ≈ **13 u at r = 64 (≈ 8 u at r = 32)**, i.e. util 0.187 → ≈ 0.200 (0.195). Metered ops (estimate): per (s, m, t) 2 matmuls + 2 adds; per (m, t) 3 Hadamards + 2 reductions; per layer 1 range finder (≈ 10 ops): 8 × (10 × 4 + 4 × 5) + 16 × 10 ≈ **640–700 extra ops**, inside the ≈ 1,000 headroom of A-class; a J = 3 version (≈ 400 ops) for B-class. Memory +≈ 20 MB. Stage 2 adds 2 thin products per (s, m) per layer: ≈ 2 u.

Break-even (derived from the brief's rule: +30 u ↔ 74 % of the κ₄-oracle gain): at +13 u (util ×1.07) the raw MSE must fall by ≥ 6.5 %; at +8 u by ≥ 4.2 %.

## 5. Expected gain (estimate, with the reasoning)

- Ceiling: if the cross-family hypothesis is right, star + ct + tt over ages 1–6 should land between the K013/K2 band (0.5) and the full K = 3 (0.85); the design is built for ≥ 0.70 at ages 1–6 and ≥ 0.60 at ages 1–4 with r = 64 (gates, §7).
- Deployment shrinkage (V56: 0.69 teacher-forced → 0.61–0.67 deployed, `k4s_fit.json`): ×0.9 on the correlation.
- With the one-point calibration of the candidate plan (raw ≈ R0 (1 − 0.29 f(ρ)), f = 1 − (1−ρ²)/0.91, **estimate**): deployed map ρ 0.65 → 0.78 gives raw ×0.934; at r = 32 (+8 u) score ×0.978, at r = 64 (+13 u) score ×1.00. **Stage 1 alone is therefore worth −2 % to 0 % of score**; it only pays if the deployed correlation reaches ≈ 0.80 or the cost stays ≤ 8 u.
- The κ₄ channel alone is bounded by the κ₄ oracle (×0.70 raw, ≈ #24); the joint-oracle gain (×0.27) requires stage 2. Realistic combined upside if both work: raw ×0.80–0.85 (**estimate**), i.e. score ×0.86–0.91 at +15 u, which is the top-15 band, not the top 10.

## 6. The ceiling test (`experiments/probes/fable_ceiling_2026-10-09.py`)

Exact Monte Carlo through MLP #0 (float32 forward, float64 accumulation), teacher μ, C, K4d from `r17_data.npz`. For every (target t, vertex m = t − j, j ≤ 6):

1. the four linear images of the K = 3 pieces and the exact remainder, yA = (Φu)Ȳ, yB = (g He2)Ȳ, yC = (c He3)Ȳ, yR = (ReLU − pieces)Ȳ, and their joint 4th-order cumulants per target neuron (69 raw moments, exact moment → cumulant conversion). This gives, exactly on the samples: the linear carry κ₄(yA) (= carry of the full κ₄ tensor of z_m over j layers), the K = 3 / K = 2 / K013 / exact one-vertex births with EXACT inputs, and the single-piece and two-piece decomposition Q1, Cu1, Q2, QCu, CuCu, higher;
2. the cross-moments S = E[u_a (uẎ)_i²], YL = E[u_a² (uẎ)_i], K2 = κ₄(u_a, u_a, y_i, y_i), hence the factorised readouts star (teacher C), ct, tt, ck, the rank-r sketches of Q (r ∈ {32, 64, 128}, Ω = top eigenvectors of C_m) and the thin YL (r_y ∈ {16, 64}, from the teacher Y21(m) when `--teacher-dir` points at `k4teach`, else the oracle space of YL);
3. self-checks: Σ_j births_exact_j + κ₄(yA at age J) must equal the sample κ₄(z_t) (telescoping identity; printed as `telescope err`), and Cu1 ≈ star + tt + ck, Q1 ≈ ct (the κ6 / κ5 residues);
4. the project metric for every combination, cumulative over ages 1..J, J = 1..6; the verdict on the gates of §7; `<out>.npz` with every per-(t, j) vector for the component test; `<out>_k4pred_{ARCH_J6, ARCH_J4, ARCH_r64_J4, ARCH_r64_ry16_J4, ...}/layer{t}_seed0.npz` (key K4d) for the end-to-end run.

Kaggle commands (GPU on foysalemonshanto, dataset `foysalemonshanto/r17-ceiling-data`):

    python fable_ceiling_2026-10-09.py 2e6 cuda --data /kaggle/input/r17-ceiling-data/r17_data.npz --out /kaggle/working/fable_ceiling            # 3 targets, estimate 3–6 min, < 2 GB GPU
    python fable_ceiling_2026-10-09.py 2e6 cuda --data ... --targets all --diag-targets 12,15,16 --out /kaggle/working/fable_ceiling_all   # all layers for the E2E dirs, estimate 10–20 min, ≈ 3 GB GPU

End-to-end (laptop or Kaggle CPU, from `whest-starterkit`, MLP #0 only, paired against the same base as `r20_joint_oracle.json`):

    NETS=mlp0 OPUS_TEACHER=<dir>/fable_ceiling_all_k4pred_ARCH_J6 python -m uv run python ../experiments/chain/run_variant_teacher.py ../experiments/variants/r20_joint_oracle.py "V31_ANGULAR=0" "V31_ANGULAR=0,R20_K4=pattern"

(`_r20_k4` reads only `K4d` from `OPUS_TEACHER/layer{l}_seed0.npz`; layers without a file keep the chain's κ₄; `pattern` rescales to the chain's layer mean, so only the pattern is injected. Reference: none 2.273e-8; teacher pattern alone ≈ 1.55e-8, ×0.68, **measured** per the κ₄ digest; teacher κ₄ full 1.5605e-8, `r20_joint_oracle.json`.)

## 7. Pre-registered gates

Ceiling (N = 2·10⁶, MLP #0, project metric at L12 / L15 / L16; the script prints the verdict from exactly these rules):

- **PASS**: ARCH = star + ct + tt, ages 1–6, exact Q ≥ 0.70 at all three AND ≥ star + 0.12 at all three. Deployability sub-gate (reported, not part of PASS/KILL): ARCH_r64 ages 1–4 ≥ 0.60 at all three.
- **KILL**: ARCH ages 1–6 < 0.60 at all three, OR the increment of tt over star + ct < 0.05 at all three (the cubic T·T dressing is then not the carrier).
- **INCONCLUSIVE** otherwise. Reading guide: if `births_K3` (sum of one-vertex K = 3 births with exact inputs, linear carry) is itself < 0.70 while `births_K3+L4carry` is ≈ 0.9, the deep pattern needs multi-vertex dressing and no first-order carrier can reach the gate; if `QCu` or `higher` carries the missing correlation, the mixed C·C·T term is the next (more expensive, pair-hub) candidate and the T·T design stops here.

End to end: PASS if the final MSE ratio with ARCH_J6 injected (pattern mode) is ≤ 0.85 (≥ 45 % of the teacher-pattern gain); KILL if > 0.95.

Component (chain legs, `fable_component_2026-10-09.py` against the ceiling npz, decisive neurons): PASS if corr(chain tt_j, exact tt_j) ≥ 0.80 at 14→15 and ≥ 0.60 at 11→14 and 11→15; KILL if < 0.50 at 14→15. The rank-64 fidelity decides r.

Only after all three: prototype in `v56_star.py` (A-class), smoke reproduction, Kaggle 100-MLP pair with gate geo ≤ 0.985 at the measured util (break-even ×0.935 raw at +13 u).

## 8. Smoke runs (executed here; tiny N, CPU; these prove execution, not the hypothesis)

`fable_ceiling_2026-10-09.py 2000 cpu --batch 1000` (35 s): 18 (target, vertex) pairs; every sample-based term is noise at N = 2000 (ct, tt, Q1, Cu1 ≈ 0 ± 0.1, as expected); the teacher-C star summed over 6 ages gives 0.521 / 0.489 / 0.471 at L12 / L15 / L16, identical to Round 18's `star_h6` (0.521 / 0.489 / 0.471, `round18_htqb.json`), which validates the transport and leg conventions; the telescoping identity holds to 1–5e-5 (float32 rounding); the verdict logic runs and is flagged "NOT VALID: N < 5e5".

`fable_component_2026-10-09.py --pairs 14:15 --max-sources 2` (2 s): slot/transport check rel err 1.3e-7 (the chain's slot s is the same source at both layers and the transport convention is right); the chain-built star has fidelity 1.000 against the sample-exact star at 14→15.

## 9. What is NOT addressed

- κ₃ accuracy directly (brief §5, "cubic-order or carried correction to the κ₃ births"): the chain's κ₃ birth already contains the C·T dressing at the cubic vertex through the D21 feedback legs (X1 = 3a_b + 1.5 d(w₂) D21, Y1 = a_b d(w₂) + 0.5 d(w₁) D21ᵀ d(w₃), rank R_FB), so the missing κ₃ piece at second order is most plausibly the κ₄-dressed birth g κ₄(a,a,b,c) fed from the wrong (memoryless) κ₄; that is stage 2, untested, and the ceiling script has no κ₃ diagnostic.
- The synthetic network: the ceiling and component tests are MLP #0 only; the E2E can be run on both networks once the `all`-targets prediction dirs are also made for the synthetic teacher (the script accepts any packed npz with W, mean, C, K4d).
