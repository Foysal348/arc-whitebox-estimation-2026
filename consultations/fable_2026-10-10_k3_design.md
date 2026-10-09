# Fable design (10 Oct 2026): the κ₃ error is the mean-field κ₄→κ₃ feed; replace it by a carried cubic-κ₄ feed in thin legs

Answer to `consultations/prompts/FABLE_k3_prompt_2026-10-10.md`. Every number is marked **measured** (file), **derived** (algebra or arithmetic on measured numbers) or **estimate**. Nothing in this note was run at full size; the only runs are the self-test and the laptop smoke run in §6. The ceiling run and the E2E are for the main session (§5).

Files delivered (new files only):
- this note;
- `experiments/probes/fable_k3feed_ceiling_2026-10-10.py` — the exact ceiling (Monte Carlo on the true network, GPU), self-test, and the attribution analysis mode;
- `experiments/variants/r20_feed_oracle.py` — `r20_joint_oracle.py` plus 58 inserted lines (nothing removed): the chain's own feed removed at the dslice readout in a layer window and the file's feed added; a dump mode for the attribution.

## 0. Summary

The chain's late-layer κ₃ error (D21 off-diagonal 5–7 %, d3 −1.4 % level drift at L16, 91–96 % of it in all-distinct post-ReLU entries, carried not born fresh) is, by elimination among the first-order terms of the post-ReLU κ₃, the **κ₄→κ₃ feed** `g_c Φ_i Φ_j κ₄(z_m)_{ccij}`: the chain feeds it from the memoryless core as `κ₄(z_m)_{ccij} ≈ (m/6)[λ_m C_ij + δ_ij dG_i]`, which does not depend on the vertex neuron c at all (a mean-field feed), while every other first-order term is either exact in the chain or has been oracle-tested as not the problem (§1). The fix (§2) replaces the F68 feed block by feed blocks built from the carried **cubic κ₄ (star + C·T)** already identified on 9 Oct, written as **thin legs** on the chain's existing source legs (the Z/L and feedback-leg machinery), at an estimated +20–30 u at r_f = 32 (+12–18 u at r_f = 16) and ≈ 1.5k ops (§3). The ceiling script tests three levels in one run: the exact feed (oracle: is the feed the lever?), the feed from κ₄ born within J layers (ceiling of any carried-κ₄ design), and the representable content (star-fed, closed form, plus the one hub-separable C·T placement), with pre-registered E2E gates (§4–5).

## 1. Diagnosis

### 1.1 The first-order terms of the all-distinct post-ReLU κ₃ (derived)

At a ReLU layer m with pre-activation z_m, u = z_m − μ, K = 3 response h − E h = Φ u + g (u² − v) + c (u³ − 3 v u) (g = φ/(2σ), c = −αφ/(6v)), the joint cumulant κ₃(h_i, h_j, h_c) for distinct i, j, c, to first order in the non-Gaussianity of z_m (one κ₃ or one κ₄ of z_m per term, the rest covariances), is by the connected-partition expansion:

| vertex orders | term | in the chain |
|---|---|---|
| (1,1,1) | Φ_i Φ_j Φ_c κ₃(z)_ijc | the legs: exact up to confinement (no confinement: −3.9 % MSE only, paper §4.5) |
| (2,1,1) | g_c Φ_i Φ_j [2 C_ci C_cj + **κ₄(z)_ccij**] + 2 perms | 2CC: birth leg A = w1 C_off, exact (paper §2.3). **κ₄ feed: F68 block, mean-field model** |
| (2,2,1) | g_i g_j Φ_c [4 C_ij κ₃_ijc + 2 C_ic κ₃_ijj + 2 C_jc κ₃_iij] + perms | V18 feedback legs (FB 1.25, R_FB 4); true D21 in them: ×1.001 (dead list) |
| (3,1,1) | c_i Φ_j Φ_c [3 C_ij κ₃_iic + 3 C_ic κ₃_iij] + perms | feedback legs; the RLL test of 10 Oct (`astra_rll_test.json`): the chain carries it fully (ρ −0.04…+0.07 with the residual) |
| (2,2,2), (3,2,1), (4,1,1) | κ₄ × C | wk431 / g22c model; second order in the vertex, small |
| K × K | κ₃ × κ₃ | measured small: 12 terms, RMS 3e-6–2e-5 vs error 2.2e-4 (paper §4.5) |

So the only first-order term whose INPUT the chain does not possess is the κ₄ feed. Reading the F68 code (`estimator.py` lines 1605–1630 and `_dslices`): the feed block is Sym(X3 ⊗ I ⊗ y 1ᵀ) + the M_t1 column u vᵀ with X3 = d(w1)[λ C_off + diag(dG)]d(w1), y = (m/4) w2, i.e. (derived from the chain's Sym normalisation, checked against its own birth-dslice formula `D21 += (1/3) R yᵀ + (2/3)(y∘X3)`):

    κ₄(z_m)_{ccij} ≈ (m/6) [λ_m C_ij + δ_ij dG_i]       (independent of the vertex neuron c)

and consistently K22_ab ≈ (m/6)(dG_a + dG_b), K31_ab ≈ (m/2) λ C_ab (the chain's `wk4_mat`, `wk431`). This is the κ₄ whose only pattern is C: the parity theorem (paper §2.2) says no such statistic carries the deep κ₄ pattern, and the chain's own κ₄ diagonal pattern correlates 0.29–0.34 with the teacher on the decisive neurons (paper §4.3). The feed with the wrong c-dependence lands exactly in the all-distinct post-ReLU entries (hint 3), is born at every vertex and then transported with the source's legs for the life of the source (hint 1: carried, growing with depth), and its diagonal readout 3 Σ_c g_c Ȳ_cp κ₄(c,c,ỹ_p,ỹ_p) is a level term for d3 (hint 2).

### 1.2 What is measured, derived, estimated here

- **derived**: the term table above; the chain's feed model (from the code); the sample-exact forms of §4 (verified by the self-test to 1e-15).
- **measured (by others, cited)**: every oracle in the table; the RLL kill (10 Oct, main session); the author's own "exact (2,1,1) κ₄ core: −9 % raw at 0.5–1.0 B" (`docs/build_notes.py` F3 list) — a dense carried κ₄ core with Gaussian (second-order) births already moved the feed channel by 9 %, which is consistent with the feed being a lever and with second-order births capping at the 0.44 pattern.
- **estimate**: that the feed error is *most* of the 5–7 %. The brief's κ₃×κ₄ synergy (0.272 vs 0.417 if independent, `r20_joint_oracle.json`) and the "with the true κ₄ the true d3 matters" line are consistent with one mechanism feeding both channels, but the share is exactly what the ceiling measures (the `--analyze` mode reports the fraction of the chain's D21 error energy removed by replacing the feed, with the sampling noise subtracted).
- **light check run** (§6): the pipeline, the Gaussian-layer self-consistency (F4 from the Gaussian layer z_1 sits at its own noise level) and the memory footprint; no physics number at N = 2·10⁴ is meaningful.

## 2. The fix: carried cubic-κ₄ feed in thin legs (CFF)

Chain conventions: legs are n×n arrays [neuron, hub]; every leg moves as leg ← Ystep_lᵀ leg, Ystep_l = diag(w1_l) W[l]; a source born at m'' carries A (birth leg, = w1 C_off at birth) and P (propagator, = I at birth); its C-leg is X'' = A + P diag(w1 v) (as V56 reads it) and its hub leg is Ȳ'' = P diag(1/Φ_{m''}) (floored at Φ ≥ 1e-3; g/Φ and c/Φ stay bounded by the Mills ratio).

**State.** Per feed vertex m (every ReLU layer from L4 on): one transported thin basis S_m (n×r_f), born as the top-r_f eigenvectors of the chain's C_pre(m) (the "eig" basis of the ACDS round, the best sketch basis measured; shared with V57 if it ships), transported S_m ← Ystepᵀ S_m. Per (κ₄-carrier source born at m'', feed vertex m) with 1 ≤ m − m'' ≤ J: static right factors V (n_hub × r_f), see births. Memory: 12 bases × n × r_f × 4 B ≈ 1.5 MB at r_f = 32; V factors ≈ 30 × 3 × n × r_f × 4 B ≈ 12 MB.

**Births (exact identities; derived, verified by the self-test).** The star-born κ₄ of source m'' at vertex m is κ₄_pqrs = 6 Σ_e c_e Sym₄(Ȳ''_ep X''_eq X''_er X''_es) (hub e = birth neuron of source m''); its κ₃ feed at m, g_c κ₄(c,c,·,·), is exactly the sum of two hub-form κ₃ blocks on source m'''s own legs:

    B₁ = Sym(X̃₁ ⊗ X'' ⊗ X''),   X̃₁[c,e] = 12 g_m,c c''_e Ȳ''[c,e] X''[c,e]
    B₂ = Sym(X̃₂ ⊗ Ȳ'' ⊗ X''),   X̃₂[c,e] =  6 g_m,c c''_e X''[c,e]²

(n×n Hadamards of the source's legs, no product). The C·T-born κ₄ (κ₃ of an older source s' converted at the quadratic vertex m'') has one hub-separable placement in its (c,c,i,j) slice — both vertex indices on the (quadratic vertex, C-partner) pair — giving per κ₃ block (X_s', P_s', Y_s') of s' the three feed blocks (X̃, P_s', Y_s'), (X̃', X_s', Y_s'), (X̃'', X_s', P_s') with

    X̃ = diag(g_m) Γ_aᵀ X_s'(a),   Γ_a = Σ_{m''≥a} Ẏ_{a→m''} G''_{m''→m},   G'' = diag(4 g_{m''}) (Ȳ_{m''→m} ∘ X_{m''→m})

i.e. exactly the ACDS age-collapsed object with the roles (target → feed vertex m); sketched at rank r on the eig basis, X̃ ≈ (diag(g) Π_aᵀ)(S̃_aᵀ X_s'(a))ᵀ with Π_a = S_aᵀ Γ_a, which V57 forms anyway. The other five placements of the C·T-born feed are not hub-separable (they couple the vertex index with a free index through C_ab) and are dropped; the ceiling measures what that costs (`starct` vs `bornJ`).

**Sketch.** Every new left leg is rank-r_f on the basis S_m: X̃_k ≈ S_m Ṽ_kᵀ with Ṽ_k = X̃_kᵀ S_m (one n² r_f product per block at birth). With X'' = A + P d (d = w1 v per hub) and Ȳ'' = P d' (d' = 1/Φ''), the blocks reduce to three types on the existing legs:

    (thin, P, P):  Ṽ = Ṽ₁ d² + Ṽ₂ d d'      -> r_f extra columns of the source's Z/L stacks (Z = S_m transported, L = Ṽ static)
    (thin, P, A):  Ṽ = 2 Ṽ₁ d + Ṽ₂ d'       -> a V18-type feedback leg pair (F = S_m in Zf, R = Ṽ static)
    (thin, A, A):  Ṽ = Ṽ₁                  -> new but same cost class: D21 += (2/3)(X̃₁∘A)Aᵀ [fused into LA] + (1/3)(A∘A) Ṽ₁ S_mᵀ [thin]; D3 += rowsum(X̃₁∘A∘A)

**Transport.** Only S_m is transported (shared by every feed block born at m): n² r_f per layer per live basis. The V's are static. Feed blocks live as long as their source; for sources in the shared basis (age > 5) the Z stacks are already transported thin, so the extra columns ride along.

**Readout.** Nothing new: the feed blocks enter D21 and D3 through `_dslices` (fused left factors for the dense-right-factor terms, thin products for the thin-right-factor terms, the existing PPL/Z path for the M-type columns). The F68 memoryless feed is switched off from the first layer where the carried feed exists (m ≥ 3; the ceiling's E2E removes it in the window); a blend is a prototype knob, not part of the ceiling.

**Shapes at n = 1024, r_f = 32:** S_m n×r_f; per (source, vertex) block Ṽ n×r_f (×3 types); dense X̃ formed transiently per source-layer (n×n, pooled buffer); no n×n×k stacks beyond the chain's own.

## 3. Cost (estimate; u = 2³¹ FLOPs; nothing metered)

Per source-layer with feed blocks (young dense sources, ≤ 5 per layer, L4–16 → ≈ 60 source-layers): form the three dense thin products S Ṽᵀ (3 n² r_f), two thin right-factor contractions (2 n² r_f), Z-stack transport of r_f extra columns (n² r_f): ≈ 12 n² r_f flops = 0.19 u at r_f = 32 → **11 u**. Births: per (source, vertex) pair (≤ 5 × 12 = 60), two Hadamard legs (n²) and two n² r_f sketches: ≈ 0.13 u → **8 u**. Basis S_m: a two-pass range finder on C_pre(m) ≈ 6 n² r_f = 0.2 u × 12 → **2.4 u**. Old-source feed columns (15 sources × 12 layers × n² r_f): **≈ 5 u** if kept to full depth (the ceiling's `cumulative_by_age_off` says whether vertices older than 5 layers matter). C·T (a,b) blocks: ≈ 2 n² r per (s', m) on top of V57's Π (**≈ 3 u**), or the ACDS thin cost (11–18 u at r = 32–64, `acds3_ct_probe.json`) if V57 does not ship.

Total: **≈ 21–30 u at r_f = 32 on V57 (21 u without old-source columns, 29 u with them), ≈ 30–45 u standalone** (r_f = 16: about 60 % of that; r_f = 64: about 1.8×). Metered ops: ≈ 12 per birth pair + ≈ 10 per source-layer + ≈ 8 per basis → **≈ 1.4–1.8k ops**. Within the brief's +30 u / +3k only with V57 sharing or at r_f = 16; the rank is the knob, and the ceiling script reports the representable content before any rank truncation (the sketch fidelity is the component-stage question, as it was for C·T).

Break-even (derived from the brief's rule, V56 ≈ 190 u): +20 u needs raw ≤ ×0.905; +30 u needs ≤ ×0.864. The κ₃ window target (×0.80 ceiling, L9–16) leaves room for the usual oracle→deployed shrinkage only if the representable content holds most of the born feed; the ceiling measures that ratio directly (`starct_J6` vs `bornJ6`, cross-half correlation and LS share).

## 4. Why it is not on the dead lists

- Not "replacing the κ₃ source chain by Hermite-tree κ₃": the sources, legs, births and feedback stay; only the F68 feed block's content changes.
- Not a feedback-leg (fb) change: the K×C terms are untouched (and the RLL test of 10 Oct confirms they are carried fully).
- Not a direct-Γ sketch: the C·T part uses the eig(C) basis and the age-collapsed Γ of the ACDS round; the star part needs no Γ at all.
- Not C17 (role-split λ) nor any memoryless feed model: the feed is carried, with the vertex-neuron pattern of the κ₄ births.
- Not C13 ("κ₄ riding κ₃ legs", ×1.05): no fitted per-source coefficients; exact identities with the chain's own Hermite constants; and the κ₄ is used in the κ₃ channel, not as a κ₄ readout.
- Not the author's dense "(2,1,1) κ₄ core" (−9 % at 0.5–1.0 B): that carried a second-order (Gaussian-birth) κ₄ in dense legs; this carries the cubic (star + C·T) content in thin legs on existing legs.
- Not the 9 Oct stage 2 as written (κ₄→κ₃ feed from the T·T state): T·T is dead; the carrier here is star + C·T, which passed its own ceiling (0.82–0.85, `fable_2026-10-09_verdict.md`), and the feed is now the primary target, not a bolt-on.

## 5. The ceiling test

Script `experiments/probes/fable_k3feed_ceiling_2026-10-10.py` (docstring has the full derivation). Per target t it forms, from N exact samples of the true network with pass-1 sample μ, C (so every image is exactly centred and every identity is exact on the samples):

- `exact`: F4(t) = Σ_{m<t} [2 E[yB_p yA_p yA_r] + E[yA_p² yB_r] − Gauss(m→t)], the exact κ₄ feed of every vertex read on the (2,1) slice at t (diagonal = d3 part);
- `bornJ4`, `bornJ6`: F4 with κ₄(z_m) → κ₄(z_m) − κ₄(linear image of z_{m−J}), the ceiling of any carrier with births within J layers;
- `star_J4`, `star_J6`: the star-fed feed, closed form (no samples); `ct_J6`: the hub-separable C·T placement, sampled; `starct_J6` = star + ct: the representable content of §2;
- `zero`: the control (chain feed removed, nothing added);
- per-vertex profile and cumulative-by-age norms (carried vs fresh), split-half reliabilities and cross-half correlations (unbiased by shared sampling noise), the Gaussian-layer check (F4 of the Gaussian layer z_1 must vanish to noise), the sample Y21 vs the teacher.

Outputs: `<out>.json`, `<out>.npz`, and `<out>_feed_<name>/layer{t}_seed0.npz` (key F21) for the E2E variant.

**Pre-registered gates** (E2E, plain chain, MLP #0, window L9–16, `FEED_ORACLE=inject` with the chain's feed removed in the window, paired ratio to the plain chain; reference: teacher κ₃ slice + d3 in the same window ≈ ×0.69, derived from the k3loc windows L9–12 ×0.857 and L13–16 ×0.806):

| file | PASS | KILL |
|---|---|---|
| `exact` (diagnosis: is the feed the lever?) | ratio ≤ 0.80 | > 0.92 (0.80–0.92: PARTIAL, the feed is a lever but not the whole carried error) |
| `bornJ6` (ceiling of any J = 6 carried-κ₄ design) | ≤ 0.85 | > 0.95 |
| `starct_J6` (the representable content of §2) | ≤ 0.90 | > 0.97 (then the non-separable C·T placements are required and §2 is not enough) |
| validity sub-gate (reported, not pass/kill) | split-half reliability of `exact` ≥ 0.90 at every window target | else rerun with larger N before reading the E2E |

Synthetic network: same window; a PASS needs the MLP #0 PASS and synthetic ≤ 0.90. The attribution (`--analyze`) is reported alongside: the fraction of the chain's D21 error energy at L9–16 removed by replacing the feed (noise-corrected), the d3 level after the replacement, and corr(chain error, −(exact − chain feed)).

### Exact run instructions (main session)

Ceiling, GPU kernel on foysalemonshanto (dataset `foysalemonshanto/r17-ceiling-data` for `r17_data.npz` [key W], plus the k4teach files from `r20-joint-oracle-data` symlinked as in `kaggle/r20o/kernel_e2e/run.py` for the teacher comparisons; the script needs only W, everything else is sampled):

    python fable_k3feed_ceiling_2026-10-10.py 2e6 cuda --targets 9-16 --J 4,6 --Jct 6 --ct-targets same \
        --data /kaggle/input/r17-ceiling-data/r17_data.npz --teacher-dir <k4teach mlp0> --out /kaggle/working/k3feed_mlp0 --batch 8192
    python fable_k3feed_ceiling_2026-10-10.py 2e6 cuda --targets 9-16 --J 4,6 --Jct 6 --ct-targets same \
        --data <k4teach synth>/weights_seed0.npz --teacher-dir <k4teach synth> --out /kaggle/working/k3feed_synth --batch 8192

Runtime (estimate): ≈ 2,700 batched GEMMs of 8192×1024×1024 per batch, ≈ 1.2·10¹⁶ FLOPs at N = 2·10⁶ → 45–70 min on a P100/T4 (the 9 Oct ceiling took 36 min for a lighter pass); GPU memory ≈ 5–6 GB (per-pair float64 accumulators 2.9 GB, legs 0.5 GB, batch 0.5 GB); host RAM < 4 GB. If the validity sub-gate fails, rerun at 4·10⁶–10⁷ with `--ct-targets 12,15,16` (the C·T triples are the costly part; `exact` and `bornJ` keep all targets). Set `--seed` differently for the second network or keep it (independent samples anyway).

E2E, CPU kernels (as `kaggle/r20o/kernel_e2e/run.py`, with `r20_feed_oracle.py` beside `run_variant_teacher.py` in the dataset and the `k3feed_*_feed_*` dirs as a kernel source):

    # 1. hook validity + dump (plain chain must still give 2.273e-8 on mlp0; dump = D21/D21_nofeed/D3/D3_nofeed per layer)
    NETS=mlp0 FEED_DUMP=/kaggle/working/feeddump python run_variant_teacher.py r20_feed_oracle.py "V31_ANGULAR=0"
    # 2. the gates (one kernel, ~3 min per config: the second dslice readout doubles the dense contractions; plain = 120 s)
    NETS=mlp0 python run_variant_teacher.py r20_feed_oracle.py "V31_ANGULAR=0" \
        "V31_ANGULAR=0,FEED_ORACLE=inject,FEED_LMIN=9,FEED_LMAX=16,FEED_DIR=<dir>/k3feed_mlp0_feed_zero" \
        "V31_ANGULAR=0,FEED_ORACLE=inject,FEED_LMIN=9,FEED_LMAX=16,FEED_DIR=<dir>/k3feed_mlp0_feed_exact" \
        "V31_ANGULAR=0,FEED_ORACLE=inject,FEED_LMIN=9,FEED_LMAX=16,FEED_DIR=<dir>/k3feed_mlp0_feed_bornJ6" \
        "V31_ANGULAR=0,FEED_ORACLE=inject,FEED_LMIN=9,FEED_LMAX=16,FEED_DIR=<dir>/k3feed_mlp0_feed_bornJ4" \
        "V31_ANGULAR=0,FEED_ORACLE=inject,FEED_LMIN=9,FEED_LMAX=16,FEED_DIR=<dir>/k3feed_mlp0_feed_starct_J6" \
        "V31_ANGULAR=0,FEED_ORACLE=inject,FEED_LMIN=9,FEED_LMAX=16,FEED_DIR=<dir>/k3feed_mlp0_feed_star_J6" \
        "V31_ANGULAR=0,OPUS_D21ORACLE=all_d3,OPUS_OR_LMIN=9,OPUS_OR_LMAX=16"
    # synthetic: NETS=synth and FEED_DIR_SYNTH=<dir>/k3feed_synth_feed_<name> (same configs)
    # 3. attribution (laptop-light): 
    python experiments/probes/fable_k3feed_ceiling_2026-10-10.py --analyze <feeddump> --net mlp0 --out <dir>/k3feed_mlp0 \
        --teacher-dir data/kaggle_out/kaggle_out_k4pub/k4teach --targets 9-16

Memory of the E2E: the chain's pool (≈ 5.5 GB) plus one extra n×n copy; Kaggle CPU kernels (30 GB) are fine, the laptop is not. Note the dirs in the config strings must not contain commas (the harness splits on them).

**Reading guide.** `exact` PASS and `bornJ6` PASS and `starct_J6` PASS → component test next (the chain's own legs: build X̃₁, X̃₂ from `r7_cap` legs at one (m'', m, t) triple and correlate with the ceiling's per-vertex star feed; the ACDS probe already has every ingredient for the C·T part), then the prototype in `v56_star.py`. `exact` PASS but `bornJ6` KILL → the feed needs κ₄ memory longer than 6 layers: a carried κ₄ with the chain's linear carry (the 0.77–0.80 "linear carry" of `opus12_k4_carry.json`) rather than births only. `exact` PARTIAL → the feed is one of two carried errors; the attribution's residual (the part of the chain error orthogonal to the feed replacement) is the next object to map. `exact` KILL → the κ₃ error is not in the feed; the next candidates are the (2,2,1)/(3,1,1) coefficients themselves (the FB 1.25 paradox) rather than their inputs.

## 6. Light checks executed here (laptop)

- `--selftest` (n = 6, brute-force index loops vs every closed form and sample estimator on an arbitrary non-Gaussian sample set): exact-feed estimator 1.6e-15, star closed form 2.1e-16, C·T (a,b) estimator 3.0e-15, carry form 6.6e-15 → **PASS** (measured). This validates all placements and factors of §2/§5 independently of sampling.
- Smoke run, MLP #0 weights, N = 2·10⁴, CPU, targets 3 and 5, J = 2, Jct = 2: 69 s, far under 1 GB; the pipeline writes the JSON, the npz and the five E2E dirs; the Gaussian-layer check (F4 from z_1, which must be zero) is 1.47 against its own noise 1.61, i.e. consistent with zero; all split-half reliabilities ≈ 0 at this N, as expected (no physics can be read from this run). The sample C_off at N = 2·10⁴ differs from the teacher by 11–23 % (that is 1/√N scaling: 1–2 % at 2·10⁶).
- `r20_feed_oracle.py` compiles; the hook reuses the exact `_dslices` call of the base and only zeroes the three feed lists and the M_t1 column of a copy of L; it is **not runtime-tested** here (the chain needs ≈ 5.5 GB). Step 1 of the E2E instructions is that test: the plain configuration must reproduce 2.273e-8 and the dump's D21 − D21_nofeed must be a few per cent of Y21 with corr > 0 to `exact` (the chain's feed is a crude version of it).

## 7. Secondary questions (κ₄ cost; brief §4)

**Q3 (cheaper exact C·T for in-window sources).** The ×0.902 (young only) vs ×0.865 (all, r = 64) gap is an *old-source* loss, not a young-source sketch loss, so making the young sources exact does not recover it. Exactness for young sources costs 2 n³ per (source, target) = 2 u (≈ 100 u over 5 sources × 10 targets); for old sources the chain's own representation is already rank-352, so Aᵀ Γ_a = FA_sᵀ (Qcᵀ Γ_a) is "exact" at n² r_old = 0.34 u per source-target after one shared Qcᵀ Γ_a (0.34 u per target): ≈ 35 u for 10 old sources × 10 targets — still above eig32 on everything (×0.874 at ≈ 18 u). No exact route under 30 u exists in this representation (derived). The actionable version: eig32 on all sources, and share Γ/Π with the feed blocks of §2, which turns the C·T machinery into a two-channel asset (κ₄ readout + κ₃ feed) for one Γ.

**Q4 (C·T for L11–16 only).** With the brief's numbers (derived): eig64 L11–16 ×0.912 at ≈ 24 u (0.6 of 40 u) → score ×0.912 × 214/190 ≈ 1.03; eig64 all ×0.865 at 40 u → ≈ 1.05; eig32 all ×0.874 at 18 u → ≈ 0.96. Cutting targets loses accuracy faster than cutting rank: the window cut is not the right cut. Rank 32 on all targets (and, if the ops permit, top-6 sources only when the old anchors dominate the count) is the better cost cut; the layer window should only drop L7–8 (ctwin L7–10 alone is ×0.983: nearly worthless, and those targets are the cheapest anyway).
