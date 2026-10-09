# Fable brief (10 Oct 2026): the κ₃ channel, the larger lever. Design and ceiling-test ONE fix for the chain's late-layer κ₃ error

You are working inside our project repository:
`D:\Machine Learning 1 Year Challenge\Kaggle\ARC White-Box Estimation Challenge 2026`.

Read this brief fully before acting. Every number below was measured by us, and the source file is given in brackets.
§0 (task, scoring, grader limits), §6 (protocol) and §7 (data map) of your 9 Oct brief
`consultations/prompts/FABLE_architecture_prompt_2026-10-09.md` still apply unchanged; read them first.

## 1. What your 9 Oct design achieved (the κ₄ channel)

**Ceiling.** Your LKV star + C·T is real.

| injection on the plain chain, MLP #0, layers 7–16 | final MSE ratio | source |
|---|---|---|
| teacher K4d | ×0.700 | `fable_affine_e2e.json` |
| star + C·T as a feature: K4d/v² ≈ a·ck4/v² + b·pred/v² + c, oracle-fitted per layer | **×0.773** | `fable_affine_e2e.json` |
| star alone | ×1.010 | `fable_affine_e2e.json` |
| star + C·T + T·T | ×0.792 | `fable_affine_e2e.json` |

The raw prediction cannot be injected directly: its level is 1–4 % of the teacher's, 40 % of the values are negative, and the result is MSE ×100+.

**Component.** The chain's own κ₃ legs reproduce the sample-exact C·T: corr 0.993 / 0.996 [`fable_component_2026-10-09.json`].

**Affordability work** (another assistant's "ACDS", re-implemented and measured by us) [`acds_ct_probe.json`, `acds2_ct_probe.json`, `acds_e2e.json`, `acds3_ct_probe.json`]:
- **Exact age collapse.** Γ_a = G_a + Ystep_a Γ_{a+1}, with Σ_m A_s(m)ᵀ G_m = A_s(a)ᵀ Γ_a; relative error 3e-7. The six age products collapse to one per anchor.
- **Exact anchor grouping.** All 9 Sym placements of the 3 blocks per source reduce to two dense anchors (A, P) with target matrices H_A, H_P, plus exactly computable thin anchors (Z Lᵀ, Zf Rᵀ, y 1ᵀ); relative error 3e-7.
- **Sketching A_sᵀΓ_a.** Top-r eigenvectors of C at the window start, transported like a leg, are best. A randomized subspace iteration with q = 2 matches them. Direct-Γ range finders (plain, column-weighted, with a power step) are worse.
- **E2E MSE, MLP #0, oracle affine map, layers 7–16:**

  | variant | MSE ratio | extra cost |
  |---|---|---|
  | sketch-free, full sources | ×0.773 | over budget |
  | sketch-free, consistent source subset | ×0.819 | over budget |
  | eig r = 64 | ×0.865 | ~40 u |
  | eig r = 128 | ×0.843 | ~80 u |

- **Layer windows** (sketch-free ceiling C·T injected only in a window) [`data/kaggle_out/ctwin_*`]:

  | window | MSE ratio |
  |---|---|
  | L11–16 | ×0.797 |
  | L13–16 | ×0.858 |
  | L15–16 | ×0.939 |
  | L7–10 | ×0.983 |
  | eig64, L11–16 | ×0.912 |
  | eig64, L13–16 | ×0.952 |

- **Round 3: cheaper source sets** [ROUND3_E2E_PLACEHOLDER].

**Verdict so far.** Break-even against the ~190 u V56 is about +30 u for ×0.865. C·T is real, but at its present cost it is score-neutral. We continue the cost work ourselves; you may comment on it (§4, Q3), but it is NOT your main task.

## 2. The new κ₃ evidence (10 Oct)

### 2.1 Where the κ₃ gain lives

Teacher κ₃ (D21 slice and/or d3 diagonal) injected only in a layer window, plain chain, MLP #0
[`data/kaggle_out/k3loc_*`; full-depth numbers from `r20_joint_oracle.json`]:

| mode (all_d3 = slice + d3, all = slice only, d3 = diagonal only, fb = feedback legs only) | MSE ratio |
|---|---|
| all_d3, L1–16 | ×0.607 (synthetic net ×0.716) |
| all_d3, L1–4 | ×1.004 |
| all_d3, L5–8 | ×0.942 |
| all_d3, L9–12 | ×0.857 |
| all_d3, L13–16 | ×0.806 |
| all (slice only), L1–8 | ×0.937 |
| all (slice only), L9–16 | ×0.834 |
| d3 only, L1–8 | ×1.012 |
| d3 only, L9–16 | ×0.853 |
| fb only, L1–16 | ×1.001 |
| all_d3 + K4d (κ₃ and κ₄ jointly), L1–16 | ×0.272 (synthetic ×0.569) |

The window gains are close to multiplicative/additive. Almost all the κ₃ gain is in L9–16, and both the off-diagonal slice and the diagonal d3 matter there.

### 2.2 How large the chain's κ₃ error is, per layer

Chain (plain V29, r7 captures) vs the 6e7-sample teacher
[`chain_k3_error_profile.json`; teacher sampling noise from a Gaussian approximation, `teacher_k3_noise_estimate.json`]:

| layer | D21 off-diag rel. error | teacher noise (est.) | d3 rel. error | d3 norm ratio chain/teacher |
|---|---|---|---|---|
| 2 | 0.044 | 0.053 | 0.033 | 1.002 |
| 4 | 0.042 | 0.026 | 0.029 | 0.998 |
| 8 | 0.049 | 0.012 | 0.031 | 0.996 |
| 12 | 0.059 | 0.007 | 0.038 | 0.993 |
| 16 | 0.069 | 0.005 | 0.046 | 0.986 |

**Reading.**
- Up to about L4 the "error" is mostly teacher noise.
- From L8 on, a real error of 5–7 % grows with depth.
- d3 shows a systematic level shrink of 1.4 % at L16, i.e. a bias, not noise.
- A 5–7 % κ₃ error is worth ×0.61 in final MSE. The mean closure is very sensitive to it.

### 2.3 One-step truncation errors

One step L→L+1 from exact samples, ReLU replaced by its Hermite-K expansion. Relative errors
[`r20_onestep_channels.json`]:

| source layer | K2: D21 / d3 | K3: D21 / d3 | K4: D21 / d3 |
|---|---|---|---|
| 4 | 0.199 / 0.213 | 0.066 / 0.071 | 0.033 / 0.033 |
| 8 | 0.155 / 0.163 | 0.046 / 0.050 | 0.024 / 0.025 |
| 12 | 0.123 / 0.126 | 0.027 / 0.028 | 0.016 / 0.017 |
| 15 | 0.099 / 0.100 | 0.021 / 0.020 | 0.013 / 0.014 |

**Earlier finding** (9 Oct brief §1/§5): of the chain's accumulated (2,1)-slice error, 91–96 % sits in all-distinct post-ReLU κ₃ entries that are 9–16 % wrong. About 55 % of it is unexplained at second order in K.

## 3. Dead list additions (do NOT re-propose)

- **Replacing the κ₃ source chain** with Gaussian Hermite-tree κ₃: closure error 10–20× worse [`r19_tree_ceiling.json`].
- **Feedback-leg (fb) changes alone:** ×1.001.
- **Direct-Γ sketches for C·T:** worse than eig(C) at equal rank.
- All items in §4 of the 9 Oct brief.

## 4. The question

**Main (κ₃).** Design ONE change to the κ₃ source chain (births, transport, closure of the all-distinct post-ReLU entries, or a carried correction state) that removes a large part of the 5–7 % late-layer (2,1)-slice and d3 error, and is affordable.

Targets:
- **Ceiling:** at least half of the κ₃ window gain, i.e. MSE ≤ ×0.80 on MLP #0 for L9–16. Check the synthetic net too.
- **Cost:** ≤ +30 u and ≤ +3k metered ops, or a matching cost cut elsewhere.

Hints from the evidence, not constraints:
1. The error grows steadily with depth. It looks carried, not born fresh each layer. Which approximation in the chain's κ₃ recursion is first-order in the per-layer non-Gaussianity?
2. The d3 level drift (−1.4 % at L16) is systematic.
3. The all-distinct post-ReLU entries are the dominant wrong part.

**Secondary (κ₄ cost; answer briefly):**
- Q3. Given §1, is there a cheaper exact or near-exact C·T route for sources anchored inside the window? Young sources alone keep the signal [ROUND3_E2E_PLACEHOLDER].
- Q4. Is C·T for L11–16 only the right cost cut?

## 5. Deliverable

Same as the 9 Oct brief §8:
1. The design: equations with shapes.
2. Cost in u and metered ops.
3. Why it is not dead.
4. **The ceiling run executed** (Kaggle if heavy: you may use up to 5 CPU kernels on esfersami50 and 2 on foysalemonshanto; the laptop has ~1 GB free RAM), with the script and JSON committed and a pre-registered PASS/KILL.
5. If it passes: the component test and a prototype variant.

Mark every number as measured / derived / estimate. Do not claim a result you did not run. Do not submit to the grader.
