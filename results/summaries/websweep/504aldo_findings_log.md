# Insights — running write-up material

Capture write-up-worthy findings the moment they appear. This feeds the algorithmic-contribution
write-up (PDF + submission ID; $10K Phase 1 deadline Aug 7, $20K Phase 2 deadline Sep 26),
judged on score, algorithmic ideas, and write-up quality.

Format: `F<N> (date) — finding`, with enough detail (numbers, experiment file) to reconstruct.

## Findings

- F1 (2026-07-10) — Local baseline reproduction on mini split (100 MLPs) matches published
  numbers exactly: mean-prop MSE_final 9.48e-4 (published ~9.5e-4), cov-prop 8.37e-5
  (published ~8.4e-5). Both hit the 0.1x discount floor (mean C/B: 8.4e-4 and 7.6e-3).
  Our local harness is trustworthy as the primary signal. (whest run, --runner local,
  v1-phase1 mini.)
- F2 (2026-07-10) — Paper (arXiv:2605.05179) core: cumulant propagation, MSE ~ c_K (L/n)^K.
  At competition shape (L=32, n=256), L/n = 0.125, so each +1 in K buys only ~8x error
  reduction, vs ~64x at the paper's L=4. Depth-32 is deliberately outside the paper's
  validated sweet spot — this is the competition's actual frontier and why leaders are only
  ~7.6x over sampling. See docs/papers/2605.05179-notes.md.
- F3 (2026-07-10, corrected same day by F5) — Budget arithmetic: B = 2.72e11 FLOPs.
  Basic kprop K=3 linear update is einsum(ijk,ia,jb,kc->abc) = 6n^4 ≈ 2.57e10/layer
  → 8.2e11 for L=32: 3x OVER budget. Factorized K=3 (~L²n³ scale) is not an optimization
  but a REQUIREMENT at this shape. K=3 factorized + MC correction up to ~B remains the
  natural hybrid operating point given the 0.1x floor (H2/H4).
- F4 (2026-07-10) — Reference implementation exists and is public:
  github.com/alignment-research-center/mlp_cumulant_propagation (cloned to data root).
  It is PyTorch-based; the grader requires flopscope.numpy — a careful port is needed, and
  their FLOP counting assumes a "fictitious symmetric kernel" discount that flopscope may
  not grant. Verify flopscope's counting of einsums on symmetric tensors early.
- F5 (2026-07-10) — flopscope probe (C:\kaggle_local\whest\scratch\probe_flopscope_einsum.py):
  NO symmetric discount; einsums charged at naive contraction cost, n²(2n-1) per matmul-like
  stage (mults + adds counted exactly). einsum(ij,ia,jb->ab) == two chained matmuls (4n³·0.998)
  — no penalty for einsum form, no reward for symmetry. To get the paper's symmetric savings
  we must structurally compute only unique entries. Elementwise costs: norm.pdf 27 FLOPs/elt,
  norm.cdf 48/elt, ReLU (maximum) 1/elt. One batch-1 forward pass = 4.19e6 FLOPs → budget
  B = 2.72e11 ≈ 64,900 MC passes, matching the overview's arithmetic.

- F6 (2026-07-10) — Literature triage: from five candidate resources, kept two.
  (1) ARC blog "Competing with sampling" (alignment.org/blog/competing-with-sampling/):
  the competition's conceptual frame — Matching Sampling Principle, and its "random MLPs
  via cumulant propagation" instance IS this problem; names Hermite polynomial bases and
  tensor-network contractions as the techniques beyond second order. Read with 2605.05179.
  (2) arXiv:2403.16163 (analytic covariance propagation): closed-form mean/covariance of
  Gaussians through ReLU/Heaviside/GELU — the machinery behind the cov-prop baseline
  (~8.4e-5). Caveat: core ReLU-Gaussian identities are classical (Cho & Saul 2009
  arc-cosine kernels); gets us to baseline, not past it.
  Discarded: 2202.04925 (trained-net interpretability), 2410.13211 (LM rare outputs —
  right authors, wrong problem), 2211.06738 (heuristic-estimator philosophy; only useful
  as framing language in the algorithmic write-up).

- F7 (2026-07-11) — Measured the torch kprop reference (mlp_cumulant_propagation) on 8 REAL
  competition MLPs (mini ids 0-7, n=256, L=32, float64; harness
  C:\kaggle_local\whest\scratch\kprop_bench.py, data dumps in scratch/mlp_dumps/).
  Aggregate MSE_final (mean over 8):
  - kprop k_max=2 SIMPLE == cov-prop ex03 to 4 digits on every MLP (6.79e-5 both). The
    starterkit cov-prop baseline IS the paper's K=2; no free lunch left at K=2.
  - k_max=3 SIMPLE factored: 3.83e-5 — only 1.8x better than K=2, and WORSE than K=2 on
    3 of 8 MLPs at the final layer (despite 4x better all-layer MSE). At L=32 the simple
    variant's error accumulates; the augment pieces are not optional at this depth.
  - k_max=3 AUGMENT factored: 7.47e-6, 9.1x better than cov-prop, consistent across all 8.
    All-layer 2.19e-6.
  - Wall time: 74 s/MLP (torch f64, 16 threads) — OVER the 60 s grader cap, even though
    FLOPs are only ~6% of budget. Paper's Sec 6.8 warning confirmed: non-matmul overhead
    dominates. Any port must be engineered for wall time, not just FLOPs.
  - Context: pure MC's flat adjusted score (F9) is ~5.9e-7 on these MLPs — kprop-augment
    raw (7.5e-6) is 12.7x worse; even at the 0.1 discount floor its adjusted ~7.5e-7 merely
    TIES trivial MC, with wall-clock risk on top. And the optimal bias-variance combination
    of kprop (bias^2=7.5e-6, cost 0.06B) with MC on the remaining 0.94B improves MSE by
    only ~3-8% over MC alone. Pure analytics caps out here; K=4 factorized (~4.4e12 FLOPs)
    is 16x over budget.

- F8 (2026-07-11) — Two classic variance-reduction tricks are DUDS at L=32 (8 mini MLPs,
  4000 samples, scratch/cv_probe.py):
  - Affine control variate (ReLU -> c + Phi(alpha)*z gains from cov-prop; E[g] exact by
    affine propagation): Var(h-g)/Var(h) = 0.80 mean. With the surrogate doubling per-sample
    cost, equal-compute gain is 0.63x — LOSES to pure MC. The surrogate decorrelates from
    the true network over 32 layers (corr(h,g) ~ 0.45).
  - Antithetic pairs (h(X)+h(-X))/2: variance ratio 0.92 -> 1.09x gain. Depth-32 ReLU nets
    are far from odd-symmetric.
  Write-up-worthy as negative results: cheap surrogates and symmetry tricks die at depth.
- F9 (2026-07-11) — Scoring arithmetic: pure MC has a FLAT adjusted score in N above the
  discount floor: score = (avg_var/N) * (N*c_pass/B) = avg_var * c_pass / B ~ 5.4e-7,
  independent of N for N in [~6.5k, ~65k] (below N~6.5k the 0.1 floor binds and score
  worsens as 1/N). Two consequences: (1) a trivial batched-MC submission scores ~5.4e-7
  adjusted — 12x better than our submitted cov-prop (6.62e-6); do this next as the new
  baseline. (2) Leaders (0.9-1.6e-7 adjusted) beat the MC invariant by only 3-6x — THAT is
  the actual size of the algorithmic frontier, not the 72x gap from cov-prop. Any winning
  idea must either cut variance per sample or cut FLOPs per sample by that factor.

- F10 (2026-07-11) — QMC WORKS at L=32: scrambled Sobol on the 256-dim input beats iid MC
  by 1.7-2.3x in final-layer MSE at equal N (4 MLPs, R=20, N=2048-32768;
  scratch/qmc_probe.py, scratch/qmc_scaling.py). Details:
  - The gain is a roughly CONSTANT factor, not a better exponent: measured MSE ~ N^-1.03
    for Sobol vs N^-1 iid (depth-32 ReLU too rough for asymptotic QMC rates). So the
    optimal operating point STAYS at the discount floor; use the smallest power of 2 above
    it (N=8192 vs floor ~6460; Sobol needs power-of-2 N for full balance).
  - LHS: 1.14x, moment-matched iid: 1.20x, antithetic: ~1.0x (the +8% assumed for v1 was
    within noise; Cov(h(X),h(-X)) = -0.004 vs Var 0.036 — drop it, DOF for Sobol).
  - Sobol+antithetic no better than Sobol alone (scrambled nets already balanced).
  - Methodology trap (cost us an hour): MSE-vs-GT comparisons at R<=3 replicates are
    wildly noisy because neurons share samples (few effective DOF, heavy tail) — a R=3 run
    showed antithetic 3.5x WORSE, pure fluke. Use R>=20 and report replicate SEM
    (scratch/anti_diag.py).
  - Grader-side feasibility verified: fnp has bitwise ops (1 FLOP/elt) + uint dtypes;
    flops.stats.norm.ppf exists (83 FLOPs/elt); Sobol gen + ppf add <1% to per-sample
    cost. Direction numbers (256 dims x 30 bits, uint32, 32 KB) extracted from scipy and
    our vectorized Gray-code generator reproduces scipy's unscrambled points exactly
    (scratch/sobol_numpy.py -> scratch/sobol_directions_d256.npy). Submission packaging
    ships extra files present in the folder.
  - Projected: MSE_sobol(8192) ~2.8e-6 x C/B 0.128 -> ~3.6e-7 adjusted, ~1.9x better than
    submission 315857. Pending: digital-shift-only scrambling vs Owen (fnp-implementable).

- F11 (2026-07-11) — GRADER ENVIRONMENT FACTS (from submission 315871's failure page +
  config panel; user screenshot):
  - `import numpy` raises ModuleNotFoundError in the grader sandbox. Only flopscope
    (`flops`, `fnp`) and the Python stdlib are available to estimator code. Anything
    numerical must be fnp; anything else must be pure-Python stdlib.
  - A setup()-stage exception = SETUP_FAILED at the "smoke test" stage: the submission is
    rejected before scoring but does NOT count against the daily quota.
  - The submission page (View -> Overview/Configuration) shows FULL tracebacks for failed
    submissions — always check it before debugging blind. Config panel confirms grader
    runs: 100 MLPs, depth 32, width 256, budget 2.72e11, smoke test with 3 worker tries.
  - Confirmed chain: 315871 failed on the numpy import (not the sidecar .npy); 315881's
    try/except caught the same import and silently scored as iid MC (7.33e-7).

- F12 (2026-07-11) — Active-subspace ROTATION for Sobol is a DUD (scratch/rot_probe.py,
  4 MLPs, R=20, N=8192): rotating inputs so leading Sobol dims align with eigenvectors of
  G = E[J^T J] (estimated via backprop probes) gives 0.92x aggregate (per-MLP 0.80-1.21x,
  within noise). This despite G being extremely anisotropic (top/median eig ~1e17, ~10
  directions carry 50% of gradient energy). Explanation: every Sobol coordinate is already
  perfectly stratified in 1-D; rotation only reshuffles which 2-D+ interaction projections
  get the best equidistribution, and a depth-32 net's interactions are spread across many
  pairs. Grader-side feasibility had been confirmed (fnp.linalg.eigh = 1.5e8 FLOPs at 256,
  0.06% of budget) but there is no gain to ship.

- F13 (2026-07-11) — Layer-1 exact-mean regression control variates are a DUD ON TOP OF
  SOBOL, and the reason is structural (scratch/cv_layer1_probe.py, 4 MLPs, R=20):
  - Features f = [z1, relu(z1), z1^2] have EXACTLY known Gaussian means/covariances
    (Stein + arc-cosine closed forms; Cov(h_i, z_j^2) = S_ij^2/(s_i sqrt(2pi))).
  - R^2(h_L | f) ~ 0.50: HALF the final-layer variance is explained by layer-1 statistics.
  - Under iid MC the CV works as theory says: 1.38x. Under dshift-Sobol: 0.97x = NOTHING.
  - Depth profile of the QMC gain (H3 answered): 21x at layer 1, ~3.4x at layer 4, ~2.4x
    at layer 8, decaying to 1.4-2.1x at layer 32. Sobol already integrates away exactly
    the layer-1-explained half; what remains is high-order-in-x "deep" noise.
  - CONSEQUENCE: the whole input-side class — QMC scramble upgrades, rotations, control
    variates with known input-side means, moment matching, whitening, affine surrogates
    (F8) — caps out around ~2x over iid at L=32. Leaders at 3.7-6.5x past the MC invariant
    must be attacking the deep/high-order component or per-sample FLOPs. Write-up-worthy
    as a unified negative result with a clean mechanism.
  - Also measured: digital-shift scrambling beats iid 1.61x aggregate (vs CP's ~1.4x on
    the LB), confirming the v3 upgrade; last-layer Gaussian smoothing (fit mu/sigma of
    z_32, closed-form relu mean) is 0.76x — z_32 across x is far from Gaussian at L=32.

- F14 (2026-07-11, CORRECTED same night) — Early probes suggested superlinear N-scaling
  (exp ~1.10) and an antithetic gain (1.18x) for dshift-Sobol; a PAIRED R=32 remeasure
  (scratch/paired_nv_probe.py; variants share the scramble, log-ratio SEM 5-7%) shows
  both were largely 4-MLP seed noise: paired gains vs plain8192 are 2.12x @16384 (mildly
  superlinear, +6% score), 3.91x @32768 (wash), anti8192 0.93x (slightly HURTS),
  anti32768 4.19x (+15% score projected, but only ~7% over plain32768, within noise).
  METHODOLOGY (extends F10's trap): even AGGREGATE MSmeans over 4 MLPs x R=16-20 carry
  ~15-17% seed noise — decisions on <20% effects need paired designs AND more MLPs, or
  the local 100-MLP runner. Unresolved small upside: N=16384 (+6%) and anti32768; test
  through whest run (real multiplier accounting) before any v4. Non-power-of-2 prefixes:
  12288=8192+4096 on the 1/N line, 6656 poor.

- F15 (2026-07-11) — LB intelligence from the Jul 12 board (user screenshots, we are #49
  at 2.91e-7 after +75):
  - There is a dense PLATEAU at 2.87-2.95e-7 (at least ranks ~46-51, all "2.4x vs
    sampling"): many teams independently hit the same ceiling we predicted in F13 —
    scrambled-QMC-class input-side methods cap there. External validation of F13.
  - Top-10 decomposes into two archetypes via adjusted/raw = multiplier:
    (a) FLOOR-DWELLERS (mult ~0.10-0.12): pluto #2 raw 9.27e-7, joe_wanza #5 raw 1.22e-6,
        andrei_bulzan #8 raw 1.30e-6, d0rb #10 raw 1.52e-6 — i.e. 2-6x better RAW
        variance than our Sobol at comparable-or-less compute. They have real
        deep-variance tech (or strong hybrids).
    (b) COMPUTE-SPENDERS (mult 0.21-0.58): andrew_epstein #1 raw 3.55e-7 at mult 0.26,
        shiv_m #3 0.36, kaileh57 #4 0.58, SOX #6 0.36, bitter-lesson-pilled #7 0.21,
        ionel_chiosa #9 0.47 — their estimators SCALE BETTER THAN 1/N with compute
        (otherwise spending above the floor would be pointless, F9/F14), i.e. genuinely
        superlinear methods exist and are worth up to ~1.9x at high mult.
  - Both archetypes imply ~3-4x variance beyond us, consistent with the H5 frontier.
    The cutoff (~1.53-1.58e-7) is ~1.85x from us and drifting down; most of the top-11
    submitted within the last 2 days (exceptions: #7 Jul 6, #9 Jul 3, #11 Jul 3).

- F16 (2026-07-12) — Forum triage (docs/platform/temp_forum/, 4 threads read in full):
  - **External confirmation of F13** (thread "Unbiased randomized-QMC + Rao-Blackwell...",
    user evaaaz): an independent team's Phase-1 estimator — randomly-shifted rank-1 lattice
    (RQMC) input sampling + exact closed-form layer-1 mean (Rao-Blackwell) — lands at
    adjusted ≈4.10e-7 (mult 0.42, well above the discount floor). That is essentially our
    v2.2 CP-Sobol number (4.15e-7 at mult 0.135) reached less efficiently — consistent with
    F9 (flat-score-above-floor), i.e. they are leaving score on the table by not dialing N
    down to the floor. More important: they tested the SAME "obvious upgrades" we did
    (linearization/layer-1 CV, block-covariance CVs, scrambled-Sobol-vs-lattice, a learned
    bias-corrector on block-cov) and found them all near-dead (~x1.0-1.2, or 2.3x cut on a
    problem needing ~17x) — independent replication of F13's structural claim that
    input-side tricks cap out and the residual is high-order "kink" variance. Strengthens
    H5 as the real frontier and lowers the priority of re-testing more input-side/CV
    variants ourselves.
  - **flops.Module packing** (thread "Flopscope v0.8.0 Release Candidate"): submissions can
    bundle up to 50MB/50 files of arrays into a `flops.Module` (`.save()`/`.from_file()`,
    JSON+named-arrays, no pickle); loading at predict-time costs 0 FLOPs and 0 wall-time.
    Relevant if an H5 attack ends up needing precomputed per-layer tables (e.g. a bias-
    correction lookup, extra QMC point sets, or a small learned corrector). We're already on
    flopscope 0.8.0rc5 / whestbench 0.12.0rc3 (ahead of the rc1 announced in that post) —
    no version action needed, but the post's "computation vs data movement" cost-model
    clarification (arithmetic/matmul/FFT billed; copy/reshape/stack/slice/gather free) is
    worth re-checking against our dshift-Sobol table-building code for any op we assumed
    free that a future evaluator revision could start billing.
  - **Algorithmic Contribution Prize mechanics** (thread "Algorithmic contribution prize
    guidelines"): submit exactly one PDF write-up + one graded submission-id, either
    privately (arc-whestbench@aicrowd.com) or publicly (challenge forum, which also makes it
    eligible for Community Contribution Prizes). Deadlines: Phase 1 Aug 7 2026 23:59 UTC
    (one week after Phase 1 closes Jul 31 — but the referenced submission must already be
    successfully graded by Jul 31, since the Phase 1 evaluator closes then), Phase 2 Sep 26
    2026 23:59 UTC. STRATEGIC FLAG: ARC states they are "less interested in methods that
    rely heavily on sampling, fine-tuned constants... and opaque LLM-optimized code" for
    THIS prize specifically, and most interested in "mechanistic" estimation ideas. Our
    current line (QMC/Sobol sampling-based) is a sampling method by construction — likely
    a weaker algorithmic-prize candidate even if it scores well on the leaderboard (score-
    based prizes are unaffected by this). If we want a shot at the $10K/$20K algorithmic
    track specifically, H5's per-layer/coupled-forward ideas (if they pan out) would read as
    more "mechanistic" than pure QMC and are the better write-up angle regardless.
  - **Higher-moments dataset** (thread "Datasets with higher moments?"): user keenanpepper
    published huggingface.co/datasets/keenanpepper/arc-whestbench-higher-moments-2026 — full
    O(W^2) moment-tensor components up to 4th order, pre- and post-activation, every layer,
    computed via MC on the public competition MLPs (~$30 GPU-hours, saves us recomputing).
    Potential use: cheap structural diagnostic for H5 (layer-by-layer skewness/kurtosis of
    the "irreducible kink variance", without spending our own MC budget) or as training data
    for a small bundled corrector (packable per flops.Module above). Caveat: evaaaz's own
    learned-corrector attempt (this same finding, above) only achieved a 2.3x MSE cut where
    ~17x was needed — treat this as a cheap exploratory experiment, not an expected win.
    Rules check (docs/platform/Challenge_rules.mhtml §5.2/§5.7): no restriction on using
    externally published, publicly-available derived statistics for offline research;
    anything actually bundled into a submission must remain attributable and the submission
    itself must not require network access at eval time (consistent with flops.Module
    packing). Not prioritized — H5's own probes (per-layer smoothing, coupled forwards)
    come first.

- F17 (2026-07-12) — HARD 5.0s setup() TIMEOUT in whestbench
  (whestbench.cli._default_resource_limits, setup_timeout_s=5.0), discovered when the
  v4 N=16384 K=8 probe died with SETUP_TIMEOUT (8.57s) on the local runner. Not mentioned
  in any doc we'd read; a setup timeout fails the whole run (all-MLP zero fallback /
  rejected submission), not a single MLP. Consequences:
  - v3 (315894, our LB best) builds K=8 z-tables in 4.28-4.35s locally — only ~14% margin
    on hardware we can't benchmark (grader could be slower per-core). It DID pass grading,
    but re-runs (and the Phase-2 designated rerun!) carry avoidable risk.
  - v3.1 (mc_v3_1_safe_setup.py): K_TABLES 8 -> 3, setup ~1.6-1.9s (>60% margin). Local
    mini confirms no score impact: adj 4.429e-7 vs v3's 4.415e-7 (both mult 0.140, 0/100).
    K only sets how many distinct scrambles the 100 MLPs share; each draw is unbiased.
  - Any future setup-heavy design (bigger QMC tables, precomputed correctors) must fit 5s
    or move work to predict()/flops.Module packing (F16: loading bundled arrays is free).

- F18 (2026-07-12, CLOSED) — The Gaussian-smoothing family for H5, measured end-to-end (scratch/h5_smooth_probe.py, h5_telescope_probe.py,
  h5_covprop_debias_probe.py; mini MLP 0, N=8192, replicate-variance methodology — dshift
  Sobol is unbiased so Var over R scrambles == MSE, no GT needed for modified nets):
  - MECHANISM CONFIRMED: replacing relu by its Gaussian smoothing
    s_sigma(z) = sigma*phi(z/sigma) + z*Phi(z/sigma) (= E_eps[relu(z + sigma*eps)]) at
    layers 2+ makes Sobol ~5x more effective at sigma=0.5 (MSE 6.0e-7 / 1.9e-7 across two
    runs vs plain 3.0e-6 / 1.7e-6 at N=8192). F13's claim is now causal, not just
    correlational: the deep ReLU kinks ARE what caps QMC at L=32.
  - But every way of paying the smoothing bias b(sigma) = E[h_sigma] - E[h] LOSES:
    (1) 2-level sampled correction (coupled h - h_sigma): Var(diff) at sigma=0.5 is ~40%
        of full variance; best allocation 0.27-0.37x. DEAD.
    (2) MLMC telescope over sigma ladder (0.5/sqrt2 steps): the coupled level diffs get
        NO QMC gain (0.9-1.3x — they carry exactly the kink noise), optimal-allocation
        MSE 9.9e-6 vs 3.0e-6 plain. Splitting variance into kinky corrections forfeits
        the QMC advantage. DEAD.
    (3) Richardson extrapolation in sigma^2: bias2 scales ~sigma^4.6 (not polynomial
        enough); 2-pt/3-pt residual bias2 1.8e-5..1e-3 vs the ~1e-7 needed. DEAD.
    (4) Analytic debias via cov-prop difference (covprop[h_sigma] - covprop[h], using the
        elegant fact that s_sigma is cov-prop-natural: same closed forms with variance
        inflated by sigma^2, gain Phi(mu/sqrt(v+sigma^2)), diag var by 1-D Gauss-Hermite):
        residuals only partially cancel (corr ~0.6 between the two runs' errors; the
        smoothed net is MORE Gaussian so cov-prop's error is sigma-dependent). resid2
        3.4e-5 at sigma=0.5 down to 1.6e-6 at 0.125; best corner 0.59x. DEAD.
  - (5) smooth_from sweep (h5_smoothfrom_sweep.py, 2 MLPs, R=8): smoothing only DEEP
        layers keeps much of the QMC unlock (from=16 sigma=0.5: 4x; even from=28: 1.5x —
        late-layer kinks matter disproportionately), but bias2 falls only ~1.6x when the
        smoothed layers halve, and covprop resid2 tracks bias2 at ~2-8e-4 RELATIVE in
        every corner (cov-prop captures ~98% of the bias amplitude; we need 99.99%).
        Best corner 0.31x. DEAD at every (smooth_from, sigma).
  - Write-up value: unified negative result with mechanism — at L=32 the kink variance
    can neither be integrated (QMC), sampled cheaply (any telescope), extrapolated away
    (Richardson), nor absorbed analytically (gain-method cov-prop), regardless of which
    layers are smoothed. Whatever the top-10 floor-dwellers do, it is none of these.
    Next distinct axis (untested): convergence RATE — higher-order interlaced digital
    nets on the smoothed net (h_sigma is C^inf for sigma>0), where a rate win compounds
    exactly where bias is smallest (small sigma).

- F19 (2026-07-12) — Two more H5-adjacent closures:
  - HIGHER-ORDER (order-2 interlaced) Sobol: NO consistent gain, with or without
    smoothing (scratch/h5_interlace_probe.py, 2 MLPs, R=12, N=1024..16384, prefix-coupled
    replicate variance). MLP 0 teased exponent 1.27 vs 1.07 and 1.5x at sigma=0.25/N=16k;
    MLP 1 flatly contradicts (exponent 1.08-1.09 vs 1.00-1.04, WORSE constant, loses at
    every N). Depth-32 kinks (and even smoothed nets' effective roughness) deny the
    smoothness that interlacing needs. Also kills the leading hypothesis for how F15's
    compute-spenders scale superlinearly — their mechanism remains unknown.
  - MULTIPLIER DECOMPOSITION (v3.1 local per-MLP breakdowns + LB): on the GRADER the
    multiplier is FLOP-dominated — flops_used/B = 0.1266 (N=8192 matmuls), residual wall
    only ~0.002 (~6 ms/MLP); local runs inflate mult to ~0.14 via local wall-time, an
    artifact (harsh local lambda), not a grader signal. Nothing to squeeze: dropping to
    N=6144 (=4096+2048, on the 1/N line per F14) to pin the 0.1 floor trades MSE x1.33
    for mult x0.78 — net slightly negative. N=8192 at mult ~0.127 is the optimal
    operating point for the v3 design. Score ~ flat in N (v4_k2 confirmed: +4.4%
    aggregate at N=16384 was a tail artifact; paired per-MLP ratio 0.994, z=-0.1).

- F20 (2026-07-12) — Paper re-read (2605.05179 Hermite/Wick machinery) with F18's lens,
  hunting for an exact variance-reduction identity. Verdict: the paper contains NO such
  identity, but the surrounding Wick/Hermite world contains three it never uses.
  - The paper's machinery = one exact identity + truncation. The diagram summation formula
    (Thm A.2, exact for polynomials and ReLU under analytic conditions; proof via
    collecting Taylor terms into Hermite coefficients, supplement S.1.4 eq (4)->(5)) needs
    the input marginals' cumulants EXACTLY; only layer 1 is exactly Gaussian, and layer-1
    exactness is already exploited and dead (F13). Grep of all 68 pages: zero occurrences
    of control variate / variance reduction / unbiased / hybrid / importance — the
    sampling-hybrid angle is untouched by the authors. Sec 7.3 explicitly lists large-L
    asymptotics as an open problem: the competition's L=32 regime is THEIR open frontier.
    Confirms D1 (no kprop port) from the theory side.
  - Exact identity (a), OU/Mehler semigroup: P_rho h(x) = E[h(rho*x + sqrt(1-rho^2)*xi)|x]
    has EXACTLY the same mean as h for every rho, while damping Hermite-chaos order k by
    rho^k. Mean-preserving smoothing exists in principle — the bias that killed all of F18
    is an artifact of smoothing in ACTIVATION space; smoothing in INPUT space is free of
    it. The only computable realization is conditioning on a subspace = PREINTEGRATION
    (exact Rao-Blackwellization + kink removal along the integrated direction). Frames the
    preintegration probe and predicts where its gain comes from.
  - Exact identity (b), the chaos spectrum is cheaply measurable: C(rho) =
    mean_i Cov(h_i(X), h_i(X_rho)) = sum_k V_k rho^k on coupled pairs (Mehler), and
    Poincare/Plancherel sum_k k*V_k = E||J||_F^2 via the F12 backprop probes. V_k tells us
    the variance ceiling of ANY degree-d polynomial surrogate CV (Hermite features have
    exactly known means): reducible variance <= sum_{k<=d} V_k. One cheap experiment
    bounds the whole fitted-surrogate class AND predicts preintegration's potential
    (high-order terms are MORE likely to involve any given direction).
  - Exact identity (c), Stein zero-mean families: E[x·g(x) - div g(x)] = 0 for any g
    (the paper's own b_k = E[phi^(k)(Y)] identity, Cor A.3, is its 1-D face). Fitted
    "control functionals" need NO known expectations — exactly zero mean by construction —
    and are the classic route to better-than-1/N MSE scaling. Candidate mechanism for the
    F15 compute-spenders; connects to Next item 3 (sample-fitted surrogates).
  - Bonus: Prop S.3.8 gives closed-form Hermite coefficients of ALL POWERS of ReLU at any
    (mu, sigma) — ammunition for any future Hermite-feature CV at layer 1 or analytic
    component. Extracted text cached at scratchpad paper_text.txt.

- F21 (2026-07-12) — CHAOS SPECTRUM OF h_L MEASURED (scratch/h5_chaos_probe.py, 6 mini
  MLPs, N=16384, GT-centered): the strongest mechanistic characterization of the frontier
  we have, via F20's identities (a)/(b).
  - C(rho) = sum_k V_k rho^k via OU-coupled pairs + NNLS: variance fraction at chaos order
    <=2: 0.46, <=4: 0.59, <=8: 0.68, <=12: 0.82. Poincare (E||J||_F^2 / Var): MEAN chaos
    order 25.2 (per-MLP 13-37) — an extremely heavy high-order tail (~18% of variance
    above order 12 carries most of the gradient energy).
  - CONSEQUENCES (each quantitative, write-up-grade):
    (1) Ceiling of ANY degree-d polynomial/Hermite surrogate CV, however fitted:
        d=4 -> at most 2.4x variance cut even if PERFECTLY fitted (and a perfect quartic
        fit in 256 dims is impossible from ~10^4 samples). Explains exactly why evaaaz's
        learned corrector (F16) stopped at 2.3x, why F8/F13 died, and why the fitted-
        surrogate class as a whole cannot reach the leaders' 3-6x.
    (2) Rao-Blackwell subspace conditioning ceiling: exactly integrating out the TOP
        gradient-eigendirection removes only 9.7% of variance (e1 or random dir: 4.6%;
        top-2: 17%; top-4: 29%; top-8 jointly: 44%). The deep noise is high-order AND
        spread isotropically across interaction subspaces — no low-dimensional subspace
        owns it. (Also retro-explains F12's rotation dud.)
    (3) The QMC-resistant ~50% (F13) decomposes as ~13% at orders 3-4, ~9% at 5-8,
        ~14% at 9-12, ~18% above 12: to reach 4x past Sobol one must kill orders up to
        ~10+, which no fittable low-order object can do. Whatever the F15 leaders do,
        it must act on the network structure (coupling, per-sample FLOPs), not on any
        low-order functional approximation of x -> h_L(x).

- F22 (2026-07-12) — PREINTEGRATION IS DEAD at L=32 (scratch/h5_preint_probe.py, 4 MLPs,
  R=12, equal total passes 8192): outer dshift-Sobol over the 255-dim complement + inner
  stratified-normal n_q-point rule along v (unbiased, replicate-variance methodology).
  Equal-FLOP gain vs plain dshift-Sobol: eig1/q2 0.64x, eig1/q4 0.35x, eig1/q8 0.21x,
  e1/q2 0.67x, e1/q4 0.28x — LOSES everywhere, all 4 MLPs. Decomposition: measured gains
  match the zero-bonus prediction 1/(n_q*(1-removal)) with a residual smoothness bonus of
  only ~1.15-1.5x (needed: >n_q x). The F20(a) frame held (unbiased input-space smoothing,
  bias-free as designed) but F21(2) is binding: one conditioned direction removes ~10% of
  variance and smooths only the kinks crossing it, while costing n_q x outer points. The
  last input-side idea standing is now closed with mechanism; H5's frontier moves fully to
  FLOP-side / network-coupling ideas.

- F23 (2026-07-12) — THERE IS NO CHEAP COUPLED FORWARD PASS AT DEPTH 32
  (scratch/h5_lowrank_probe.py, 4 MLPs, R=12): the two-level idea E[h] = E[h-h_cheap] +
  E[h_cheap] dies because He-init nets sit at the edge of chaos and depth exponentiates
  any perturbation of the computation:
  - Rank-r SVD truncation of every W_l: delta = Var(h-h_r)/Var(h) = 1.000 at r=32..96;
    even r=128 (cost ratio c=1.0, saves NOTHING since factorized matmul bills 2r/n per
    F5) has delta=0.92. Worse: at r<=64 the truncated net's activations COLLAPSE
    (Sobol[cheap] variance ~1e-20..1e-13) — truncation shrinks per-layer Jacobian norms
    and 32 layers drive the signal to zero. Two-level factors 0.85-2.8x, mean >1: LOSES.
  - Width-m sketch nets (W~ = P^T W P, random orthonormal P, c=(m/n)^2): delta = 2-7,
    i.e. the sketch is FURTHER from h than 0 is. Dead on arrival.
  - Affine tail (exact to layer l0, best-L2 per-neuron affine relu after; folds to one
    matmul): coupling GOOD (delta 0.02-0.13 at l0=24/28, and the diff Sobol-integrates
    ~15x better than h — late layers are where roughness lives, consistent with F18) but
    c = l0/32 ~ 0.8-0.9 saves almost nothing, and the cheap level's own Sobol error is
    ~= base, so the trade never pays (factors 0.74-1.84, mean >1).
  - Quantization/low-precision: dead a priori — flopscope bills FLOPs independent of
    dtype (F5), so there is no billed-cost discount to buy variance with.
  - Write-up-grade mechanism: coupling tightness delta is destroyed exponentially in L
    for weight-space perturbations (chaotic divergence), while computation-graph-
    preserving truncations (fewer layers) can't cut cost. The MLMC/two-level door at
    L=32 is closed structurally, not for lack of tuning.

- F24 (2026-07-12) — Number-operator Stein CVs are DEAD (scratch/h5_stein_probe.py,
  3 MLPs, R=8): the one exactly-zero-mean feature family that carries HIGH chaos order
  (F20c: N f = x.grad f - Lap f = sum_k k f_k, E[N f] = 0 exactly; estimated per sample
  by 2nd-order forward-mode through the smoothed net + Hutchinson probes, ~3.4x pass
  cost per probe):
  - Sobol gain raw 1.04-1.06x (noise level), FLOP-adjusted 0.07-0.14x. Dead.
  - Mechanism: (1) the Hutchinson estimator (w.x)*D_w h_sig - D^2_w h_sig has noise
    variance ~ d x signal (d=256): at K=2 probes the feature is essentially noise; the
    exact x.grad h needs a full Jacobian = 256 backward passes/sample. (2) Even exact,
    the spread spectrum (F21: orders 1..40+) caps a single k-weighted feature at
    ~<k>^2/<k^2> ~ 1.4x. Zero-mean-ness was verified in vivo (gt-chk unchanged).
  - The low-order Hermite reference battery (He2/He3 of top gradient directions) gave
    EXACTLY 1.00x on top of Sobol — direct in-vivo confirmation of F21(1)/F13.

- F25 (2026-07-12) — F15 REFRAME (compute-spender arithmetic corrected): a mult>0.1
  submission does NOT necessarily imply superlinear-in-N scaling. A fitted/expensive
  per-sample variance cut with gain g at per-sample cost (1+c) improves the adjusted
  score whenever g > 1+c if the team is not floor-pinned — and F16 (evaaaz at mult 0.42,
  leaving 4x on the table) proves top teams DO ship un-floor-optimized estimators.
  So the top-10 "compute-spenders" are consistent with plain strong variance reduction
  (or mere non-optimization), and the hunt should target "what cuts variance 2-6x at any
  cost profile", not exotic N-scaling. The floor-dwellers (raw 0.9-1.5e-6 at mult ~0.1)
  remain the informative archetype: 2-6x variance below us at OUR compute.

- F26 (2026-07-12) — flops.Module PACKING VERIFIED END-TO-END on the real grader
  (submission 316030, v5, GRADED 3.44e-7 adj / 2.667e-6 raw / mult 0.1289):
  - A .npz bundled in the submission folder (sobol_v5.npz, 8.39 MB) ships intact and loads
    via SobolTable.from_file() at 0 FLOPs; __file__-relative path resolution works in the
    grader sandbox. Setup drops to ~7ms — the F17 5.0s-setup-cap risk class is fully
    retired for any table-based design (and the door is open to bundling up to 50 MB of
    precomputed arrays: correctors, bigger point sets, per-layer tables).
  - Per-MLP predict-time scramble+ppf (~1.9e8 FLOPs, ~250ms local wall) costs the grader
    only ~0.0016 mult in wall residual (0.1289 total vs 0.12726 flops) — the v2.2 anchor
    held exactly. Local wall-lambda (mult 0.1408 for the same code) remains an artifact
    (F19); never tune wall-side design on local mult.
  - Draw-concentration evidence: v5's LB raw 2.667e-6 lands on the 3-draw family mean
    (v3 2.26, v3.1 2.90, v5 2.67e-6 — identical expected MSE by design), consistent with
    100 independent per-MLP scrambles concentrating the suite aggregate. v3's LB-best
    2.91e-7 adj is a favorable K=8-shared-table draw, not a better estimator. CONSEQUENCE
    for Phase 2: designate a v5-class submission (no setup risk, concentrated score), not
    a lottery winner; the LB rank difference between v3 and v5 is luck, not signal.
    Packaging gotcha (cost one dud tarball): `whest package --estimator <file>.py` ships
    ONLY that file; point it at the FOLDER to include sidecar files.

- F27 (2026-07-16) — EXACT COST of the reference factored kprop, measured with the repo's own
  flop_poly_fit tooling (fits total FLOPs as an exact polynomial in n, L from torch
  flop-counter runs; scratch/kprop_k4_cost.py). At the competition shape (n=256, L=32):
  - K=2 augment (unfactored): 1.87e9 = 0.007x B (cov-prop class, trivially cheap).
  - K=3 augment factored: 1.11e12 = 4.07x B (leading term 60 n^3 L^2). CORRECTS F7's
    "FLOPs only ~6% of budget" (that was our L^2 n^3 back-of-envelope with constant 1;
    the real constant is ~60): kprop-K3's honest adjusted score is 7.47e-6 x 4.07 ~ 3.0e-5,
    i.e. ~100x WORSE than our Sobol v5 — not "ties MC at the floor" as F7 suggested.
  - K=4 augment factored: 3.47e14 = 1277x B = ~12,800x the 0.1 floor (leading terms
    48 n^4 L^2 + 1027 n^4 L). The naive "K=4 costs 4.4e12 = 16x over budget" arithmetic
    (constant-1 L^2 n^4) was ~80x optimistic. Cost is spread across ~6 components
    (eval_part 22%, get_dslice 20%, contract_W 19%, nonlin absorption ~20%, ...) — no
    hotspot; constant-factor engineering cannot recover 4 orders of magnitude. Any viable
    K=4-quality estimator requires a DIFFERENT complexity class (e.g. a mean-targeted
    backward/adjoint formulation ~ L n^3 + L^2 n^2), not a faster port. Also note: the
    reference already uses diagonal-slice towers + factored top-degree tensors — 48 n^4 L^2
    IS the optimized version; rank of the factored 4-tensor grows ~n L with no compression.

- F28 (2026-07-16, partial — run stopped early) — THE CUMULANT EXPANSION DIVERGES OUTSIDE
  ITS VALIDITY DOMAIN, measured on synthetic He nets (scratch/kprop_k4_law.py, f64 MC GT):
  - At (n=64, L=32), i.e. n/L = 2: k2_simple stays sane (5e-4..2.8e-3) but the AUGMENT
    variants explode — k2_augment MSE 5e2..3e13, k3_augment_f 1e32..1e42, k4_augment_f NaN
    (all 3 seeds). The augment correction terms amplify through 32 layers when n/L is small:
    the series is ASYMPTOTIC in n/L with a divergence boundary somewhere in n/L in (2, 8).
  - Corroborating hint at ratio 8 but tiny scale (n=32, L=4, 1 seed, N_gt=2^18):
    K=2aug -> K=3 gains 4.1x (law predicts 8), K=3 -> K=4 gains only 1.35x. Consistent with
    an asymptotic series whose optimal truncation K* is near 3-4 at n/L = 8 — which would
    make the "K=4 at (256,32) ~ 8.3e-7, K=5 ~ 9.2e-8" geometric extrapolation invalid.
  - Wall-time reality check: k4_augment_f at (64,32) took ~630s/net (effective ~2.2e9
    flops/s, overhead-bound torch) -> (128,16) ~1h/net, (128,32) ~2.8h/net, and a real
    (256,32) K=4 run would be ~40h/MLP, not the naive ~6h. The decisive affordable test of
    the law remains the (n=128, L=16) cell (same n/L=8 as the competition; law predicts
    K=4 MSE ~ 8e-7 there if c_K is shape-stable).
  - (128,16) seed 0 (n/L = 8 == competition ratio): k2_simple 1.22e-4, k2_augment 6.95e-4
    (augment WORSE than simple at K=2 here), k3_augment_f 1.93e-5 — 2.6x higher than
    (256,32)'s 7.47e-6 despite the same ratio, so c_K is NOT shape-stable (single seed).
    k4_augment_f OOMed (2GB alloc failure): the factored 4-tensor rank grows ~ n L^2
    (from the contract_W share of the 48 n^4 L^2 cost), so reference K=4 at (256,32)
    needs HUNDREDS of GB — infeasible on the grader (64GB), locally, or any rentable
    64GB CPU pod. Third independent death sentence for the reference K=4 (FLOPs 1277x B,
    wall ~40h/MLP, memory ~600GB); only a reformulated (mean-targeted/adjoint) algorithm
    could ever field K=4 at competition shape.
  - Ratio-8 series completed -> settled by F29.
    Data: DATA: scratch/k4_law_results_{n64L32,n128L16}.json, k4_law_results.json.

- F29 (2026-07-16, SETTLES the K>=4 question, negatively, with mechanism) — THE CUMULANT
  EXPANSION AT DEPTH-32-CLASS SHAPES IS ASYMPTOTIC WITH OPTIMAL TRUNCATION K* = 3 AT
  n/L = 8; K=4 DEGRADES the estimator (scratch/kprop_k4_law.py, synthetic He nets, f64 MC
  GT at N=2^24..2^25, geometric means; all augment factored):
  | cell     | n/L | k2_simple | k3_aug  | k4_aug  | K3->K4        |
  | (32,4)   |  8  | 1.0e-3    | 1.0e-4  | 7.4e-5  | 1.35x better  |
  | (64,8)   |  8  | 6.3e-4    | 4.0e-5  | 7.8e-4  | 20x WORSE     |
  | (96,12)  |  8  | 2.0e-4    | 2.1e-5  | 3.5e-4  | 17x WORSE     |
  | (128,16) |  8  | 1.2e-4    | 1.9e-5  | OOM     | —             |
  | (64,16)  |  4  | 8.4e-4    | 2.9e-1  | 1e32    | diverged      |
  | (64,32)  |  2  | 2.8e-3    | 1e42    | NaN     | diverged      |
  | (256,32) |  8  | 6.8e-5    | 7.5e-6  | unrunnable (F27/F28) | ? |
  - Consistent picture: optimal truncation K* shrinks as L/n grows — K*=3 at n/L=8,
    K*=2 at n/L=4 (k3 is 340x WORSE than k2 there), full divergence at n/L=2. The
    K=2->K=3 gain at ratio 8 (16-37x here, 9.1x at (256,32)) is real, but the geometric
    "each +1 in K buys n/L" extrapolation (F7-based: K=4 ~ 8.3e-7, K=5 ~ 9.2e-8) is
    INVALID — the series has already turned at K=4. 5/5 nets at two ratio-8 shapes agree;
    no recovery trend in n at fixed ratio (the (32,4) 1.35x gain was the small-n outlier,
    reversed by n=64).
  - CONSEQUENCE 1: the "is there an adjoint/mean-targeted factorization ~100x cheaper
    than L^2 n^K" question is MOOT — even a FREE K=4 evaluation would lose to K=3.
    The analytic frontier at L=32 is a series-truncation wall, not a compute wall.
    Combined with F27 (K=3 = 4.07x budget -> adjusted ~3.0e-5, 100x behind our Sobol v5),
    the entire higher-K cumulant-propagation route is now closed PERMANENTLY at
    competition shape: D1 was righter than we knew.
  - CONSEQUENCE 2 (LB intelligence): the F15 floor-dwellers' 2-6x raw-variance edge
    cannot come from K>=4 analytics. Mechanism hunt narrows further.
  - CONSEQUENCE 3 (write-up): this partially answers the paper's own open problem
    (Sec 7.3, large-L asymptotics; F20) in the negative, with measurements: at L=32 the
    expansion is asymptotic in L/n and its usable depth is K*=3. Together with
    F21 (chaos spectrum) + F18/F22/F23 (closed frontier map), this is now our strongest
    "mechanistic" algorithmic-prize material — ARC explicitly favors such analysis (F16).
  - Caveats: augment variant only (F7 showed simple is worse/non-monotonic at L=32
    already); largest ratio-8 cell with a K=4 point is n=96 (K=4 at (128,16)+ OOMs, and
    at (256,32) is unrunnable by F27/F28 — this settle is as direct as physically
    possible); k2_augment is consistently 1.6-5x worse than k2_simple at these shapes.

- F30 (2026-07-16, THEORY — VALIDATED same day by F33: exact for n/L >= 32, deviation is
  the finite-width deficit measured there) — CLOSED-FORM CANDIDATE LAW FOR THE MEAN
  CHAOS ORDER: at kernel (infinite-width) order, for a fixed He-critical ReLU net of depth
  L, the centered OU-coupled covariance is C(rho) ~ f^oL(rho) - f^oL(0) with f the arccos
  correlation map f(c) = (sqrt(1-c^2) + (pi - arccos c) c)/pi, and since criticality makes
  the derivative chain at rho=1 exactly 1:
      kbar(L) = 1 / (1 - f^oL(0))  ~  2 L^2 / (9 pi^2)  ~  0.0225 L^2   (n-independent).
  - Exact at L=1 (1.467 = E[relu'^2]/Var, the known single-ReLU Hermite value). Predicted
    ladder: L=2: 1.98, L=4: 3.13, L=8: 6.04, L=16: 14.2, L=32: 39.6.
  - vs F21 at (256,32): predicted 39.6, measured 25.2 (per-MLP 13-37) — right order,
    but real n=256 nets sit ~1.6x BELOW the kernel curve; theory NNLS fractions on F21's
    own grid (V<=2: 0.20, <=4: 0.39, <=8: 0.41, <=12: 0.71) vs measured (0.46, 0.59,
    0.68, 0.82): real nets carry systematically MORE low-order mass. The deviation is
    presumably an L/n finite-width correction — the SAME expansion parameter that controls
    F29's truncation wall (K*=3 at n/L=8, K*=2 at 4, divergent at 2).
  - If a depth table (L in {2,4,8,16,32} at n=256 + ratio-matched cells) confirms the
    kernel shape with an L/n-dependent depression, F21 upgrades from measurement to LAW
    and unifies with F29: one dimensionless parameter (L/n at criticality) controls both
    the sampling-side chaos spectrum and the analytics-side series divergence. Strongest
    possible write-up spine. If it fails, F29 remains the spine and the failure map is
    itself Sec-7.3 material.
  - Estimator caveat for the test: F21's NNLS grid (max rho 0.98) cannot resolve order
    ~40 (even the PURE theory curve mis-fits: NNLS mean order 21 vs exact 39.6). Use
    Poincare kbar as primary; for NNLS compare like-with-like (same estimator applied to
    theory curve) or add rho = 0.99/0.995 points.

- F31 (2026-07-16, H6(ii) RESOLVED — gamma unstable, but a better GT-free criterion fell
  out) — TURNING-POINT DIAGNOSTIC MEASURED on all 13 F29-seed nets (scratch/
  kprop_gamma_check.py, gamma_check_results.json; t_K = RMS(m_K - m_{K-1}), augment
  family, k_max=1..4; fresh 2^24 GT for (32,4) which F29 never saved):
  - gamma = (t4/t3)/(t3/t2) is NOT shape-stable: 0.96-4.11 across healthy nets, drifting
    down with size at fixed ratio 8 ((32,4) 1.7-4.1 -> (64,8) 1.3-2.3 -> (96,12) 0.96).
    The two-step geometric extrapolation version of H6(ii) FAILS.
  - The naive term-shrinkage rule ALSO fails at order 4: t4 < t3 on 7/8 healthy nets,
    yet K4 hurts on 6/8. Term size alone does not locate the K=4 turn.
  - MECHANISM (verified, the real finding): a term helps iff its magnitude matches the
    current error. t3/e2 = 0.98-1.25 wherever K3 helps (e_K = RMS error vs GT), and 1.70
    at (64,16) where K3 hurts; t4/e3 = 1.3-1.6 at the marginal (32,4) cell vs 3.6-5.9 at
    every K4-hurts cell; after the turn the added term BECOMES the error (e4/t4 =
    0.91-1.03). The asymptotic turn is overshoot (t_{K+1} >> e_K), not term growth.
  - GT-FREE ORDERING: the single observable q = t3/t2 orders all regimes monotonically:
    q <= 0.30 -> only zone where K4 is even marginal (and there all label margins are
    1.3-1.5x = noise-class; NO strong K4-helps regime exists anywhere on the lattice);
    q = 0.47-0.83 -> K4 hurts 10x+; q > 1 -> K3 hurts (1.71 at ratio 4); q >> 1 ->
    divergence (ratio 2). Bonus: t3 is a GT-free error meter for K2 (e2 ~ t3 within 25%).
  - q DRIFTS UP with n at fixed ratio 8: 0.17-0.30 -> 0.47-0.51 -> 0.65-0.82 -> 0.83 at
    (128,16). Expected ~0.85-1.05 at (256,32) — deep in the K4-hurts zone. APPLY STEP
    (cheap, GT-free): run k1/k2/k3 on a few real mini MLPs at (256,32), report q; that
    states "K*=3 at competition shape" via a criterion validated on 13 labeled nets,
    without ever running K=4. F29's Consequence 3 upgrades accordingly.
  - CORRECTION (same day, F32): the q-thresholds and the size-drift above are CONFOUNDED
    by k2_augment's own depth-instability (its excess error over k2_simple grows
    1.2x -> 3x -> 5.8x -> ~3e4x at L=8/12/16/32, ratio 8). Overshoot mechanism and gamma
    verdict stand; the q-threshold criterion is superseded by F32's re-anchored ratio law.

- F32 (2026-07-16, APPLY STEP DONE at (256,32) + F31 threshold correction) — RE-ANCHORED
  DIAGNOSTIC q' = RMS(m3_aug - m2_SIMPLE)/RMS(m2_SIMPLE - m1) on the 13 calibration nets
  + 3 REAL mini MLPs at competition shape (scratch/kprop_q_recal.py, q_recal_results.json;
  first attempt with the augment anchor, scratch/kprop_q_at_256.py, came back degenerate
  q=1.00 because k2_augment's RMS error at (256,32) is 0.07-7.8e4 vs k2_simple's ~7e-3 —
  the discovery that forced the re-anchor):
  - RATIO LAW (new, H6-fusion-grade): q' depends on n/L ONLY. Ratio 8: q' = 0.35 +/- 0.13
    across n=32..256, a 64x size range INCLUDING the real (256,32) MLPs (0.14/0.25/0.59);
    no size drift. Ratio 4: q' = 8.4. Ratio 2: q' ~ 1e18-1e22. A GT-free observable of
    the series that is a function of the expansion parameter n/L alone; locates the
    K*>=3 boundary (q' crosses 1 between ratio 8 and 4) and the divergence wall.
  - IN-VIVO PRE-TURN CHECK at competition shape: t3'/e2 = 1.04/1.02/1.15 on the 3 real
    MLPs — the 3rd term still matches the K2 error exactly (F31's "term helps iff it
    matches the error"), so the series has NOT turned at K=3 at (256,32). K*>=3, GT-free.
    (Also in vivo: mse_k2s = 4.2e-5/5.0e-5/2.5e-4, mse_k3 = 8.9e-6/5.5e-6/2.3e-5 —
    consistent with F29's 6.8e-5 / 7.5e-6.)
  - LIMIT (honest form of the K*=3 claim): q' does NOT resolve the 3->4 turn within
    ratio 8 — helps/hurts labels there are q'-independent, and the only "helps" labels
    are the noise-class (32,4) margins (F31). "K*=3 at (256,32)" therefore rests on
    CLASS MEMBERSHIP, not a threshold: (a) K3 is pre-turn there (t3'/e2 ~ 1), (b) K4
    fails with the measured overshoot mechanism (t4/e3 = 3.6-5.9) on EVERY ratio-8 net
    where K4 is physically runnable (n=32..96), (c) (256,32) is indistinguishable from
    that class on every GT-free observable measured. This is the final upgraded form of
    F29 Consequence 3: turning-point analysis validated across the lattice, applied at
    competition shape without running K=4 — with the inference step stated explicitly.

- F33 (2026-07-16, H6(iii) DONE — F30 VALIDATED, fusion triangle complete, plus a new
  finding) — CHAOS DEPTH TABLE measured on 26 synthetic nets: L in {2,4,8,16,32} at
  n=256 (3 seeds each) + all ratio-matched F29/F32 cells, same seed convention
  (scratch/kprop_chaos_depth.py, chaos_depth_results.json; Poincare kbar primary,
  N=16384, M_PROBE=1024; NNLS secondary with extended rho grid 0.99/0.995):
  - F30 KERNEL LAW IS EXACT IN MEAN FIELD: deficit kbar_theory/kbar_meas at n=256 is
    1.00 / 1.01 / 1.01 / 1.08 / 2.25 at L = 2 / 4 / 8 / 16 / 32. Within 1% through L=8
    (n/L >= 32), first visible bend at L=16, breaks only at competition depth. The
    spectrum IS a law (answers Q1's conditional: the write-up spine does NOT have to
    fall back to F29 alone — the fused claim is available).
  - DEFICIT IS MONOTONE IN n/L (mean over all cells): 1.00 @ 128, 1.01 @ 64, 1.01 @ 32,
    1.08 @ 16, 1.43 +/- 0.58 @ 8, 1.61 @ 4, ~5.0 +/- 3.5 @ 2. Third measured quantity
    controlled by the expansion parameter — the H6 fusion triangle (K* ladder, q' ratio
    law, spectrum deficit) is now complete, all on shared seeds.
  - NEW FINDING — SELF-AVERAGING BREAKS AT SMALL n/L: per-seed kbar scatter explodes as
    the ratio drops: (256,32) kbar = 14.6 / 22.3 / 15.9 (deficit 1.8-2.7), (64,32)
    kbar = 4.0 / 11.2 / 26.0 (deficit 1.5-9.8), vs <2% seed scatter for n/L >= 32. At
    competition shape the chaos spectrum is a strongly PER-NET random quantity, not an
    ensemble constant. So the deficit law holds for the MEAN, with O(1) per-net
    fluctuations below ratio ~8 (the ratio-8 size sequence 1.1 / 1.4 / 1.0 / 0.8 /
    2.3 for n=32..256 is within that scatter, but constancy-in-n at fixed ratio is only
    established to the fluctuation level, stated honestly). Write-up angle: this is WHY
    per-net GT-free diagnostics (q', t3'/e2) beat ensemble prescriptions at (256,32) —
    and it matches the q' spread 0.14-0.59 on the real MLPs (F32).
  - Consistency: synthetic (256,32) kbar mean 17.6 vs F21's real-MLP 25.2 (per-MLP
    13-37) — same distribution within the now-measured per-net scatter; no discrepancy.
  - Estimator check: NNLS still under-resolves at high order even with rho = 0.99/0.995
    (nnls 10-15 vs Poincare 15-22 at (256,32)); Poincare-primary confirmed as the F21
    caveat predicted. Low-order mass table at n=256 (V<=2 fraction): 0.88 -> 0.72 ->
    0.54 -> 0.42 -> ~0.50 for L = 2..32 — the L=32 UPTICK is the finite-width excess
    low-order mass F30 flagged, now visible along the depth axis.

- F34 (2026-07-16, H6(i) DONE — boundary grid ratios 3/5/6, two n per ratio) — THE K*
  BOUNDARY IS A SHARP WALL, NOT A SMOOTH CROSSING (scratch/kprop_boundary_grid.py,
  boundary_grid_results.json; 16 nets, q' ladder + 2^23-sample MC labels, gt noise
  floor 4e-5..2e-4 = 50-100x below all finite errors, labels clean):
  - q' IS FLAT ACROSS THE ENTIRE K*=3 REGION, THEN JUMPS: q' = 0.44 +/- 0.15 at ratio 5,
    0.46 +/- 0.11 at ratio 6 — statistically identical to ratio 8's 0.35 +/- 0.13. Then
    8.4 at ratio 4 and 1.9e4..2.3e20 at ratio 3. The q'=1 crossing sits in the OPEN
    interval (4,5) and is a JUMP of ~20x, not a gradual drift. F32's ratio law refines
    to: q' ~ 0.4 (flat) for n/L >= 5, then a wall.
  - THE K*:3->2 FLIP COINCIDES WITH THE q' JUMP: K3 helps at ratios 5 and 6 (8/10 nets,
    margins e2s/e3 = 1.3-4.1, shrinking from ratio 8's ~10x), hurts at 4 (1.71x, F31),
    so the flip is ALSO in (4,5). GT-free q' jump and GT-labeled K* flip locate the same
    boundary — the fusion claim's cleanest single exhibit.
  - DIVERGENCE ONSET is between ratios 3 and 4: at ratio 4 K3 is finite-but-worse; at
    ratio 3 K3 blows up outright (e3 = 1.3e3..8.5e18 across 6/6 nets, both (48,16) and
    (96,32)) while K2_simple stays healthy there (e2s = 0.01-0.05). Full ladder:
    n/L >= 5 -> K*=3; n/L in [3,4] -> K*=2 (K3 finite-harmful at 4, divergent at 3);
    n/L <= 2 -> everything past K2 meaningless. NOTE: this kills H6(i)'s linear guess
    K* ~ 0.4-0.5 * n/L — the truncation ladder is a sequence of sharp walls in n/L,
    not a linear law.
  - K*(n/L) NOT K*(n,L): at both probed ratios the two sizes ((96,16) vs (192,32),
    (80,16) vs (160,32)) give the same q' band and the same majority label — the ratio
    is the whole story, consistent with F32/F33.
  - BOUNDARY-BAND STOCHASTICITY (matches F33's self-averaging breakdown): inside ratios
    5-6 the per-net outcome is random — 2/10 nets have K3 marginally hurt (margins 0.95,
    0.68), and those two are exactly the nets with the largest overshoot meter t3'/e2s
    (1.82, 2.09 vs 1.0-1.6 on every helps-net; same criterion that flagged (64,16) at
    1.70 in F31). The overshoot meter predicts per-net outcomes even inside a ratio
    band; q' alone does not separate them (0.43/0.56 vs 0.22-0.66). Near the wall, K*
    is a per-net random variable with ~1x-margin stakes — one more reason ensemble
    prescriptions are the right level for the write-up, and per-net adaptivity only
    pays at the boundary, which the competition shape (ratio 8) is safely inside.

- F65 (2026-09-01) — MEMORYLESS (2,1)-SLICE CLOSURE IS DEAD, with a clean mechanism: the
  fully-off-diagonal K3 content is essential for the NEXT layer's slices, and the lost
  part is neither low-rank nor predictable from the live state. Context: leaders run at
  0.22-0.26xB = ~14 dense n^3 units per LAYER total, i.e. a closure without per-source
  memory; F63 says the K3 gain flows through the slices D3/D21. Test (lean torch, P2
  dumps 0-7, exact = V16-class chain; scratch/slice_closure_probe.py, slice_fit_probe.py,
  slice_diag_probe.py; logs runs/slice_ladder.log, slice_fit_k0.log, slice_diag_k0.log;
  lean_k3.py flags slice_keep / d21_corr / d3_corr / collect_d21):
  * Surrogate: after each layer keep the `keep` youngest sources exact and replace all
    older content by ONE slice-supported tensor Sym(M_ij delta_jk), M = diag(D3)+3 D21^T
    (= the existing M-leg birth form; has exactly the old content's (3,) and (2,1)
    slices; transports in 4 units/layer). keep=0 is a pure slice state at ~6 units/layer
    (0.09xB -> floor 0.1). Unit = 2n^3 FLOPs = 2^31; B = 1024 units.
  * LADDER (mean raw mse over 8 dumps, exact 4.32e-8 @ 0.50xB, score 2.16e-8):
    keep=0 9.08e-7 @0.09xB (score 9.1e-8) | 1 6.72e-7 @0.145 (9.7e-8) | 2 4.25e-7
    @0.195 (8.3e-8) | 3 2.89e-7 @0.24 (7.0e-8) | 5 1.35e-7 @0.32 (4.4e-8) | 8 6.13e-8
    @0.42 (2.6e-8). Score falls monotonically toward exact: every collapse level is
    WORSE than V16b. (Pure slice state IS 5x better than covprop 4.4e-6 at the same
    floor price -- a cheap 0.1xB baseline of 9e-7, but no more.)
  * ANATOMY (dumps 5,6, keep=0): feeding the exact chain's D3 AND D21 pre-nonlin into the
    collapsed chain reproduces the exact mse EXACTLY (the two slices are the whole
    interface; nothing else is lost). Oracle D21 alone 4.4e-7, oracle D3 alone 4.7e-7
    (each channel ~half the loss in log terms); D3=0 is 3x WORSE than the wrong D3
    (2-3e-6) -> the (3,) slice matters here through the variance/covariance update, not
    "marginal skew"; F63's "skew worthless" is about the D3->mean path only.
  * Per layer the collapsed D21 explains only R^2 ~0.78-0.85 of the exact D21 (scalar
    gain ~0.93-1.0); the residual is 18-23% of the energy at EVERY layer (fresh per-layer
    loss, not compounding) and is HIGH-RANK (rank 32/128/256 hold 0.24-0.78 / 0.60-0.94
    / 0.83-0.98 of the residual energy). D3 relative error ~0.26-0.33 (corr 0.95-0.97).
  * FITTING: per-layer scalar regression of the exact D21 on [collapsed D21, C_off,
    C_off*var/mu row/col scalings, C_off^2, C_off*mu*mu] adds only +0.02-0.03 R^2 and
    changes the chain mse by 0% (train 9.5e-7 -> 9.5e-7, test 8.4e-7 -> 8.6e-7); a
    DAgger refit collapses the coefficients to identity. The missing content is
    weight-specific information (the transported off-diagonal K3), not a bias a fitted
    closure can supply. This is F47-UPDATE's "K3 content is nonlocal" one level deeper,
    and the concrete reason a Puffi-style fitted low-mode closure cannot be what the
    leaders run on the K3 channel.
  * WRITE-UP: the D3/D21 interface theorem (two slices carry the whole exact-vs-
    collapsed difference) + the 20%-per-layer high-rank residual is a crisp statement of
    what per-source memory buys in factorized cumulant propagation.

- F66 (2026-09-01) — THE HUB-SOURCE K3 STACK IS RIGID IN EVERY DIRECTION: age-gated shared
  low-rank basis for old sources holds accuracy only at rank >= n/2 where the honest port
  cost saves nothing, and dropping ANY single source (even the two cheapest, born at
  layers 13/14) costs 4x mse. Together with F61 (windowing), F55 (hub-index rank) and F65
  (slices) every compression axis of the family is now measured flat-to-negative; V16b
  (0.50xB) is its operating point, period. Lean torch, dumps 5,6 (exact 3.453e-8 @0.50xB,
  score 1.726e-8); scratch/leg_age_spectra.py, age_rank_probe.py, skip_birth_probe.py;
  logs runs/leg_age_spectra_d5.log, age_rank_ladder*.log, skip_birth.log; lean_k3.py flags
  age_rank=(age_min, r), skip_birth, k3_snapshots.
  * LEG SPECTRA BY AGE (exact chain, layers 8 and 15): transported P legs concentrate
    with age (age>=5: rank-128 holds >=0.92, rank-256 >=0.99 of the energy; age 1: 0.53 /
    0.81); X1/M legs are >=0.87 at rank 64 from birth -- but that is the rank-1
    mean-correlation mode of C_off (kernel collapse), so ENERGY is a misleading guide:
    the accuracy-relevant content sits in the small remainder (same trap as F46).
  * AGE-GATED SHARED BASIS (confine_old_: all legs of sources with age >= a projected on
    the top-r eigvecs of their summed leg Gram; young sources dense; compounding).
    mse / honest xB / score:  a=2 r=512: 3.56e-8 (+3%) / 0.465 / 1.66e-8;  a=2 r=384:
    4.48e-8 (+30%) / 0.376 / 1.68e-8;  a=3 r=384: 4.02e-8 / 0.394 / 1.58e-8 (-8%, best);
    a=2 r=256: 1.03e-7 / 0.293 / 3.0e-8;  a=1 r=512: 5.47e-8 / 0.459 / 2.5e-8;  a=3 r=128:
    2.1e-7 / 0.256 / 5.3e-8;  a=3 r=64: 4.1e-7.  Honest port cost per old source-layer =
    2(r/n)^2 (coefficient transports) + 2(r/n) (forming dense A,P from Q -- unavoidable
    because the dslices need Hadamard products) + 4(r/n) (two two-stage contractions);
    r=512 -> 3.5 units vs 4 dense. A first cost model (3r/n) that ignored the dense
    forming produced a mirage of 1.0e-8 at 0.28xB -- do not repeat it. Best realizable
    gain ~8% before QR/residual overheads: not worth a port.
  * SKIP-BIRTH (no K3 source injected at the listed layers): skip {0}: 1.37e-7 (4x)
    @0.44xB; {1}: 1.68e-7; {0,1}: 3.96e-7; {0,1,2}: 9.0e-7; {13,14}: 1.32e-7 @0.488xB.
    Every source of every age is essential; the freshest sources carry much of the
    final-layer D21 (consistent with F63: the D21 channel is the whole gain).
  * CONCLUSION: the leaders' 0.22-0.26xB tier cannot be a compressed hub-source K3. It is
    either (a) a different closure family, or (b) the same cumulant math under a cheaper
    exact algorithm -- the only untested lever from F48 is #3: symmetric-tensor
    contraction algorithms (7/18 of naive multiplications; Solomonik-Demmel), which at
    0.50xB would land ~0.3xB at IDENTICAL accuracy (~1.3e-8 local, top-10 border).
    That is a "genuinely different algorithm" question -> email the sponsor before
    building. Cheap baseline note for the record: the pure slice state (F65, keep=0)
    gives 9e-7 raw at the 0.1 floor = 9e-8, 5x better than covprop; not competitive.

- F67 (2026-09-02) — F48 LEVER #3 (SYMMETRIC-CONTRACTION ALGORITHMS) IS DEAD BY
  INSPECTION for V16b, so the "cheaper exact algorithm" branch of F66 closes without
  a build or a sponsor email. The paper's beta_{3,inf} = 7/18 (App. J) and the
  Solomonik-Demmel savings apply to contracting a DENSE symmetric order-3 kernel on
  all three indices (einsum ijk,ia,jb,kc->abc); V16b never materializes such an
  object. Its per-layer n^3 cost (F63) is 4k units of NON-symmetric matmuls:
  transports A_b <- WD A_b, P_b <- WD P_b and the (2,1)-slice contractions
  sum_b LA_b A_b^T + LP_b P_b^T (output kappa3_{iic} is not symmetric in (i,c)),
  where LA_b/LP_b are ELEMENTWISE functions of the legs (A*P, P*P, A*A), which
  blocks every associativity/telescoping rewrite (P_b = P_{b'} G_{b',b} nests, but
  the elementwise products do not). The only symmetric-output op per layer is the
  covariance congruence (0.5 of ~30 units/layer, <2%), and F64's probe_sym_cost.py
  already showed flopscope bills (A*c)@A.T at the full 2n^3. float16 is billed at
  the float32 rate (F44), so precision is not a lever either.
  LEADERBOARD READ (2026-09-02 13:23Z, reports/leaderboard_history.csv): TWO frontier
  tiers now, not one. (i) 0.20-0.23xB tier: J2W 2.23e-8 raw / 4.70e-9, Puffi 2.16e-8
  / 4.90e-9, oqaris 2.32e-8 / 5.30e-9, Activation-Regions 2.80e-8 / 5.50e-9 (0.196xB).
  (ii) OUR cost tier with ~2.3x better raw: marius_binner 1.78e-8 raw @0.41xB (7.30e-9,
  #6), jamespayor 1.96e-8 @0.48xB (9.40e-9, #8), fklassen 2.49e-8 @0.29xB (#5). Our
  V16b is 4.46e-8 @0.50xB (2.22e-8, #22, flat). Tier (ii) raw == the aug kappa4
  channel's raw (2.0-2.2e-8, F51/F64) at HALF the aug cost floor (0.95-1.05xB, F64):
  somebody propagates kappa4-grade accuracy at ~0.4xB. Top-10 line is 1.10e-8, so
  either raw 4.4e-8 @0.25xB or raw 2.2e-8 @0.50xB gets in; tier (ii) proves the
  second route exists at our cost. Reopens F64's closing condition on its own terms:
  "a representation that avoids per-source dense G transport" is worth ~2.3x score.

- F68 (2026-09-02) — MEMORYLESS KAPPA4 REGENERATION WORKS: the aug channel's
  off-diagonal kappa4 core is lambda_l * C_off (one scalar per layer), and with it
  the whole (2,1,1) channel becomes FREE on top of V16b -- aug-level raw accuracy
  (2.18e-8 vs V16b-class 4.15e-8, 1.9x) at V16b's 0.50xB. Lean torch, P2 dumps
  0/1 (0-7 for transfer), precise final_means GT; scratch/g_regen_probe.py, hooks
  added to scratch/lean_k3_aug.py (aug_no_k211, g_hook, g_lin_diag, g_use_hook,
  g_feed_hook); logs runs/g_regen_{1..10}.log.
  TRIGGER: Puffi's public SSC write-up (P1 #327725, scratch/writeups/puffi_algo.txt):
  they never carry kappa4 per source; they REGENERATE it after every ReLU from <=4
  live covariance-response modes B_a with a frozen fitted 4x4 coupling,
  kappa4 = sum_ab Lambda_ab Sym(B_a x B_b). Applied that lens to our own aug chain
  (F51/F64), where the kappa4 core G is produced each layer by the per-source hub
  contractions (k211, 2 dense units/source-layer) and fed forward as the dense leg
  X3 = P d(w1) G d(w1) (1 more unit/source-layer).
  MEASUREMENTS (mse_final, mean of dumps 0/1 unless noted):
  * dense aug reference 2.041e-8 (== F64). Drop the ENTIRE k211 channel (production,
    hub contractions, birth part): 4.196e-8 = V16b class -> the aug gain lives in the
    hub-contracted content of G, not in the n^2 slices K(4,), K(2,2), K(3,1).
  * Oracle projection of G_off on live modes (LS per layer, chain runs on the
    projected G): C_off alone R^2 0.66-0.97 (rising with depth) -> 2.230e-8; adding
    C*C, Sym(d(mu)K21^T), Sym(t mu^T), var var^T, Sym(K31), K22, Sym(d(mu)C), mu mu^T
    (9 modes) -> 2.205e-8. C_off IS the mode; nothing else carries signal.
  * REGEN (chain with NO k211 anywhere, G_off := lambda_l C_off with lambda_l from a
    dense pass): 2.230e-8 = identical to the projection. Drop the (3,1) production too
    (32 of the 36 aug-only n^2 terms; it only fed G_off): 2.217e-8. Drop the K4->K3
    feed instead (X3/Y3/M_t1): 9.23e-8 -> the FEED carries the gain, the use-side
    terms alone are harmful. Regenerating the DIAGONAL as well (G := lambda C):
    3.27e-7 -> diag(G) (the per-neuron radial kappa4, cf. vec+ww F49) must stay exact.
  * TRANSPORT (the port question): lean transports G <- W G W^T (2 units/layer).
    With G = diag(g) + lambda C_off that is lambda C_pre_off + W diag(g - lambda var)
    W^T. Keeping only the (W*W)-diagonal of the second term AND dropping lambda
    diag(W C_off W^T): 3.44e-8 (bad). But: exact transported diagonal + lambda
    C_pre_off in the USE side only: 2.184e-8; in the FEED only: 2.436e-8; in BOTH
    (= the free port): 2.179e-8. So the off-diagonal spill of W diag(.) W^T is
    irrelevant; the earlier loss was the wrong diagonal. Free-port recipe:
    dG_pre = (W*W) g + lambda_post (diag(C_pre) - (W*W) var_post)   [n^2],
    G_pre_off = lambda C_pre_off; feed X3 = lambda A d(w1) + P d(w1^2 dG_pre) = column
    scalings of the A/P legs V16b already transports; Y3 rank-1 -> (X3*Y3)P^T folds
    into the P-group (F64 trick a); M_t1 rank-1 -> one extra thin Z/L column.
    Extra n^3 cost: ZERO. Extra n^2: 4 use-side term-table entries with wk431 =
    0.5 m lambda C_pre_off (pairs (1,2),(3,1),(3,2),(1,1) all in V16b's table).
  * FROZEN TABLE: lambda_l fitted per dump agrees within ~10% across dumps at every
    layer (smooth: ~0.002 at L0, peak 0.0113 at L4-5, 0.0069 at L15 for the no31
    fit); replaying each dump with the MEAN of the OTHER dumps' lambdas costs 1-4%
    (xregen, 4 dumps: 2.22/2.28/2.35/2.31e-8). 16 frozen scalars fitted offline on
    public MLPs = legal data (same status as CORR_BETA).
  * BASELINE CAVEAT: the "V16b class" in this chain (nok211+no31, 4.15e-8) is the
    lean aug chain minus the channel, not V16b itself (V16b dumps 0/1: 4.10/3.57e-8
    with riders). Expect the port at ~2.2-2.4e-8 raw before rider refit.
  SCORE: 2.2e-8 x 0.50 = 1.1e-8 = the current top-10 line (#10 Camaro 1.10e-8);
  with V16b polish (R_RES=16, -0.01xB) and rider refit, ~1.0e-8 (#8-9). This is
  exactly tier (ii) of F67 (marius_binner/jamespayor: ~2e-8 raw at 0.4-0.5xB).
  [SUPERSEDED by F105: R^2 quoted below is the post-ReLU fit; see F105 for the pre-activation law]
  WRITE-UP: "the fourth-cumulant core of a deep ReLU chain is, to R^2 ~0.95 at
  depth, a scalar multiple of the covariance; the scalar is a smooth universal
  function of depth" is a clean mechanistic statement, and it explains WHY Puffi's
  covariance-response ansatz (B_1 = C) works. Also reframes F64: the dense G legs
  were carrying ~5% signal at 3 units/source-layer.
  NEXT: port = estimator_v17.py (V16b + regen channel), parity vs lean regen chain
  (test_v16b.py style), lambda table from 8 dumps, mini run, submit.
  PORT (same day, estimator_v17.py generated from V16b by scratch/make_v17.py; parity
  harness scratch/parity_v17.py; logs runs/v17_*.log):
  * Two port bugs found by layer-wise parity + unit checks (feed_formula_check.py,
    wick_pairs_check.py both exact): (1) the Y3 = y 1^T leg's y is a LEFT factor of
    the hub tensor and must be TRANSPORTED (y(l) = P(l) y) -- carried as one more thin
    Z column with a zero L column; treating it as static gave 2.2e-6 (50x worse than
    V16b). (2) The V16b CORR_BETA mean rider (F47, ridge-fitted on V16b's OWN
    trajectory) is anti-correlated with V16b's base error and HURTS on the better base:
    dump0 3.57e-8 with vs 2.51e-8 without; it is OFF in V17 (refit later). The
    residual per-layer V17-vs-lean rms is 1e-6 at L01 (f32) growing to 1.1e-4 at L15.
  * NOT the cause (measured): residual-leg rank (r=512 3.40e-8 with rider), the pruned
    term table (unpruned 94-term aug table: identical), V1.6 block ablations (lean with
    shadow_B2+no_wk22_fac+no_k3_feedback: 2.46e-8 vs 2.21e-8 -- this is the remaining
    port-vs-lean gap, known and cheap to leave).
  * RESULT (rider off, pruned table + 4 wk431 terms, real flopscope C): 8 dumps at
    r=32: mse 2.448e-8 (2.11-2.85e-8), C/B 0.5015, local score 1.228e-8. Rank ladder
    dumps 0/1: r=16 2.345e-8 @0.4921xB (score 1.154e-8), r=32 2.394e-8 @0.5015,
    r=64 2.412e-8 @0.5182 -> ship r=16. Unpruned table: 2.381e-8, +40 ms residual,
    no gain -> keep V16b's pruned table. Residual 0.28-0.29 s (cap 0.400), wall 14 s,
    smoke shapes 256x32 / 64x16 / 1024x3 finite (regen gated to the suite shape like
    the riders; off-suite = V16b path with zero-padded thin columns).
  * vs V16b (same dumps, riders on): 3.9e-8 raw -> 2.35-2.45e-8 raw at equal cost =
    the 1.65x raw gain predicted by the lean chain; local score 1.96e-8 -> ~1.16e-8.
  * SHIPPED as #329480 (r=16, rider off): mini 1.17e-8 (raw 2.37e-8, mult 0.4921, 0/100
    failed, residual 0.28 s/MLP); LB 1.11e-8 (raw 2.26e-8) = rank 22 -> 11/93, tied with
    huang_chung_yi, 3% behind #10 Camaro (1.08e-8). Our LB raw now equals the #1-3
    tier's raw (2.16-2.32e-8): the remaining gap to #1 (4.7e-9) is purely the
    multiplier (0.49 vs 0.21-0.23), i.e. the K3 source cost (F63 floor 4 units/
    source-layer) -- the F66 problem, now the ONLY problem.
  * RIDER REFIT on the V17 trajectory (scratch/k3_regen_corr.py, F47 protocol,
    dumps 0-4 train / 5-7 test): train 1.05x, test 1.01x -> the F47 online mean
    correction is DEAD on this base. Its 1.45x on V16 was the K3-truncation mean
    bias that the kappa4 channel now removes analytically. Lean regen chain on the
    same 8 dumps: 2.09e-8 (test dumps 5-7: 1.92e-8).

- F69 (2026-09-02) — V1.6 BLOCK ABLATIONS RE-MEASURED ON THE V17 (REGEN) BASE WITH A
  PORT-FAITHFUL LEAN CONFIG: only the D21 FEEDBACK into the birth factors matters
  (-10% raw, 2.36 -> 2.13e-8), it is VERY LOW-RANK (rank 8 keeps -8%, rank 16/32 -9%),
  B2 HURTS on this base and WK22 is 0%. Lean torch, P2 dumps 0/1; scratch/
  g_regen_probe.py (+ tokens sb2/nofb/nowk22/nobp, d21r:R, rres:R, fitfull);
  logs runs/v16blocks_ladder{,2,3,4_d1,5_d21r}.log, v16flags_semantics.log.
  * PORT-FAITHFUL LEAN CONFIG (new): regen replay with lambda fitted on the FULL dense
    chain (fitfull) + Rres rank 16 + the three V1.6 flags = 2.47/2.26e-8 on dumps 0/1
    vs the real V17 port 2.51/2.25e-8 (r=16). Parity of the ablation semantics is now
    established; the "2.46e-8 lean-with-flags" number in F68's port addendum came from
    a run whose lambda was fitted on the flagged chain and is superseded.
  * TWO PROBE TRAPS (both fixed, both would have killed the finding): (1) fitting the
    regen lambda on a chain that itself carries the ablation flags gives lambdas that
    go NEGATIVE at depth (dump1 L08-L14: -0.8e-3..-2.1e-3 vs +7e-3..+9e-3) and a 5.9e-7
    replay -- the flags perturb G_off enough to wreck a per-dump LS fit, so the lambda
    table must always come from the full chain (as the port's LAM does); (2) the
    driver popped the fitfull flag after the first dump (same per-dump-pop bug class
    as F64's), invalidating dump1 of ladders 2/3 until rerun.
  * LADDER (mean of dumps 0/1, rres 16, fitfull; port-equiv = shadow_B2 + no_k3_feedback
    + no_wk22_fac): port-equiv 2.36e-8 | +WK22 2.36 (0%) | +feedback dense 2.13 (-10%)
    | +feedback rank 32 2.14 | rank 16 2.14 | rank 8 2.17 (-8%) | +feedback+B2 2.31
    (-2%) | all three blocks 2.28 (-3%) | +feedback without the Y1 "bprime" term
    (0.5 d(w1) D21^T d(w3)): 2.55 on dump0 (WORSE than port-equiv) -> bprime essential.
    Base K3 chains for reference (flags mean the same in lean_k3.py and lean_k3_aug.py):
    base full 5.04e-8 / flags 6.43e-8 (dump0); nok211 full 4.43 / flags 5.87.
  * SIGN FLIP vs F45: on the V1.6 base B2 shadowed cost +11% and feedback 0%; on the
    kappa4-regen base feedback is the whole effect and B2 is negative. The regen feed
    (X3/Y3/M_t1) supplies the content B2 used to carry and over-counts with it.
  * PORT COST (V16b machinery, D21_off ~ U S V^T at rank r, diagonal part = P-column
    scaling, free): X1 = 3 a_b + 1.5 d(w2) D21 -> extra identity-born thin leg
    F1 = d(w2) U with static right factor S V^T; Y1 = X1 d(w2)/3 + 0.5 d(w1) D21^T d(w3)
    -> thin leg F2 = d(w1) V with static right factor S U^T d(w3). Both transport as
    extra Z columns (n^2 r each); in the dslice contractions a thin leg as LEFT
    Hadamard factor forms an n x n product (n^2 r) that folds into the fused LA/LP
    einsums (right factors A, P unchanged), and as RIGHT factor costs n^2 r. Total
    ~8-16 n^2 r per source-layer = 0.06-0.25 unit at r = 8-32 vs 4 dense units
    (+1.5-6% cost) + a D21 range finder per birth (~8 n^2 r). Net score ~-6% at r=8-16:
    LB 1.11e-8 -> ~1.03-1.05e-8, past #10 (1.08e-8). Half-day port (newborn extras +
    thin-leg terms in the fused contraction), parity target = this ladder's
    "+feedback rank 16" config: 2.195/2.091e-8 on dumps 0/1.
  * WRITE-UP: the (2,1)-slice feedback of the K3 content into the next layer's Wick
    factors is a rank<=8 object at n=1024 -- a concrete "what part of K3 the next layer
    needs" statement complementary to F63/F65 (which showed the fully off-diagonal
    content is NOT low-rank for the slices).
  * PORT (same day, estimator_v18.py generated from V17 by scratch/make_v18.py; smoke_v18.py,
    test_v18.py, parity_v18.py; logs runs/v18_*.log, mini_v18_r16.log): Option B -- the A leg
    is untouched, the feedback rides as two identity-born thin legs F1 = d(w2) Q, F2 = d(w1)
    Bm^T (D21 ~ Q Bm, rank R_FB, randomized range finder as for Rres) with static right
    factors 1.5 Bm and 0.5 Q^T d(w3), in their OWN transported stack Zf (a first layout that
    appended them to Z with zero L columns paid 0.125 n^3/source-layer inside the M-leg
    einsums for nothing: 0.5203 -> 0.5093xB after the split). Birth dslices in the general
    (xd, X1*Y1, yd) form; contraction: LA += Xt*P*w2/3 + P*Yt, LP += A*Yt + Xt*A*w2/3 +
    Xt*Yt/3, D3 += 3 rowsum(P*LPadd), D21 += (AP + Xt*P/3) R2^T F2^T + (AP*w2 + P*Yt)/3 R1^T
    F1^T. V18_NO_FB=1 reproduces V17 bit-for-bit (max diff 0).
    Parity dump0 (port LAM table, rres 16, d21 rank 16): lean 2.212e-8 vs V18 2.191e-8.
    Rank ladder dumps 0/1 (real flopscope): r=8 2.184e-8 @0.5013xB, r=16 2.157e-8 @0.5093,
    r=32 2.119e-8 @0.5473 (old layout) -> score-flat between 8 and 16, ship r=16.
    MINI (100 MLPs): adjusted 1.10e-8, raw 2.16e-8, mult 0.5093, 0/100 failed, residual
    0.328 s/MLP (V17: 1.17e-8 / 2.37e-8 / 0.4921 / 0.280 s) = -6% score, as the ladder
    predicted. Smoke shapes 64x6 / 256x32 / 64x16 / 4x2 / 128x20 / 1024x3 finite; whest
    validate + validate-package OK. SUBMITTED #329531 (2026-09-02 ~21:15 UTC).
    Residual note: +48 ms/MLP for ~30 extra client ops per layer (birth range finder +
    fresh n^2 products + thin einsums); 0.328 s has precedent (329212 shipped at 0.31-0.33 s,
    graded clean) but the cap margin is now 18%: any further residual growth needs op fusion
    (fold the birth products into the next layer's stacked ops) before adding features.

- F70 (2026-09-02) -- LITERATURE TRIAGE #2 (P2 lens, full reads of 4 PDFs in docs/papers/):
  ALL FOUR DISCARDED AS METHODS; none reaches past the Gaussian-closure (K=2) level that
  F60 capped at 4-5e-6 raw, while V18 sits at 2.08e-8 raw. Supersedes the F6 verdict on
  2403.16163 ("gets us to baseline"): at P2 it is simply DEAD. No experiment was run; the
  triage cost was reading only.
  * 2403.16163 (Wright/Nakahira/Moura 2024, analytic covariance propagation): Theorem 1 is
    the classical Mehler/Hermite series Cov(g(y_i),g(y_j)) = sum_k rho^k/k! (s^k d^k E[g]/dmu^k)^2
    for jointly Gaussian inputs, plugged into the plain mean+covariance recursion. For ReLU
    the closed form (arc-cosine kernel / bivariate-normal CDF) was already known, so the
    series only approximates what the cov-prop baseline computes exactly. No cost lever:
    it cheapens the elementwise ReLU-moment step, which is ~0% of our bill (F56: matmul
    91.6%). Their evidence is MC/analytic variance RATIOS of 1.01 +- 0.01 on 4-8-layer
    trained nets = 1e-2 relative, six orders coarser than our regime. Only reusable
    fragment: eq. 25, the k-th derivative of the ReLU mean as He_{k-2}(mu/s) phi(mu/s),
    already inside kprop's Hermite/Wick tables (F20).
  * 2202.04925 (Fischer/.../Helias 2022, networks as maps of correlation functions): a
    Feynman-diagram bookkeeping for how the nonlinearity mixes cumulant orders, then an
    immediate restriction of the inner layers to mean+covariance (eq. 17). Its only
    higher-order content is a DIAGONAL Gram-Charlier correction to the ReLU mean, linear
    in kappa3(i,i,i) (eq. C15) -- a strict subset of our K3 chain, and F65 showed the
    off-diagonal K3 is what the next layer needs. Their own Limitations paragraph defers
    order >= 3 propagation to future work (N^n cost). Regime: widths 10-200, depth <= 4,
    trained weights, normalized-KL ~1e-2. TWO CITATION-GRADE NOTES for the write-up:
    (a) App. B derives the width scaling of pre-activation cumulants -- order n is
    O(N^(1-n/2)) when the previous layer is weakly correlated at O(1/N), with a separate
    count for partially repeated indices (B10-B12): the same bookkeeping behind the
    companion paper's c_K (L/n)^K law (F2/F29) and prior art for "kappa4 is one order
    smaller than kappa3", which is why memoryless regeneration (F68) is plausible; it does
    NOT single out the lambda_l * C_off structure F68 found. (b) App. G recovers the
    Poole et al. depth scale xi = -1/ln(sigma_w^2 <phi'phi'>), which diverges exactly at
    the He-critical point our MLPs sit on -- consistent with the F30 chaos-order law.
  * 1805.11327 (Gast & Roth 2018, lightweight probabilistic deep nets): assumed density
    filtering with a FULLY FACTORIZED Gaussian per unit (mean vector + variance vector),
    variance transported as (W o W) v with no cross-covariance (eq. 32b), Frey-Hinton ReLU
    moments, Clark max-pool approximation; the rest is probabilistic output layers for
    training. One rung BELOW the cov-prop baseline (no arc-cosine kernel), i.e. the
    mean-prop tier (~2.2e-4). Only echo of our work: the (W o W) v diagonal transport is
    the trick F49 used for the anisotropic kappa4 vector -- a textbook identity, nothing to
    cite or port.
  * NetworkMoments.pdf (Bibi/Alfadly/Ghanem, CVPR 2018, moments of PL-DNNs under Gaussian
    input): EXACT mean and second moment for ONE (Affine, ReLU, Affine) block with
    Gaussian input; Lemma 3 is the arc-cosine kernel E[relu(x1)relu(x2)] = (1/2pi)(s12
    asin(rho) + s1 s2 sqrt(1-rho^2)) + s12/4, restricted to zero mean (their own Comments
    3.3 admit the 2nd moment needs mu = 0, c1 = 0); Lemma 2 is a multivariate Price
    theorem. For deeper nets they LINEARIZE (two-stage Taylor around one input point),
    which is meaningless for a random MLP over N(0, I), and their own MNIST/LeNet tables
    show variance ratios of 0.2-0.6. The one-layer identity is already the Gaussian part
    of our ReLU second moment.
  * TAKEAWAY: the whole "moment propagation for uncertainty quantification" literature
    (Frey-Hinton 1999 -> Gast-Roth 2018 -> Bibi 2018 -> Wu-DVI 2019 -> Wright 2024) stops
    at Gaussian closure because 1% accuracy is enough for its use cases; the only line
    that propagates order >= 3 through depth is ARC's own 2605.05179 (+ the P1 write-ups,
    F48/F68). Future paper triage: skip anything whose evidence is a variance RATIO or
    classification accuracy; only papers reporting moment ERROR vs depth at the 1e-6-or-
    better level can inform the score, and only ones with cost-per-layer identities for
    order-3 tensors can inform the multiplier (the only open gap, F66/F67).

- F71 (2026-09-03) — FINAL-LAYER TRIM: an EXACT 8% cost cut nobody had taken. The final
  layer emits only the mean, whose (1,) nonlin terms consume var = diag(C_pre), D3 and the
  K4 diagonal -- never C_off, the (2,1) dslice D21, nor the (1,1)/(2,1)/(2,2) term program.
  V18 computed all of them at layer 15 and discarded them: the two dense dslice
  contractions (2k = 30 n^3 units) + the full C_pre sandwich (only its diagonal is read).
  V19 = V18 with a diagonal-only covariance transport at the last layer (var =
  colsum(w32 * (C @ w32)), 1 unit) and the source machinery in D3-only mode (transports +
  n^2 contractions, D21 = None, d=2 program skipped). scratch/make_v19.py (asserted
  patches on V18; kill-switch V19_FULL_LAST=1 = exact V18 op stream), starterkit
  estimator_v19.py / test_v19.py / smoke_v19.py; logs runs/v19_trim_d01.log, mini_v19.log.
  * Dumps 0/1, real flopscope: V18 0.5203xB mse 2.1906/2.1230e-8 -> V19 0.4777xB mse
    2.1913/2.1227e-8 (f32 rounding only): local score 1.122e-8 -> 1.030e-8 (-8.2%).
    Smoke 64x6/256x32/64x16/4x2/128x20/1024x3/32x1: finite; FULL_LAST bit-identical to
    V18; trimmed path max |diff| 1.7e-6 on values ~3.
  * MINI (100 MLPs, v2-phase2): adjusted 1.03e-8 (V18 1.10e-8), raw 2.16e-8 (identical), mult
    0.4777 (0.5093), 0/100 failed, residual 0.315 s/MLP (0.328). SUBMITTED #329571 (2026-09-03
    ~02:10 UTC) -> GRADED 9.95e-9 adjusted (raw 2.083e-8 == V18's, mult 0.4777, 100/100, 0 failed):
    rank 10 -> 9/94, first sub-1e-8 score; #8 jamespayor 9.4e-9, #1 Puffi 4.5e-9.
  * PROBE (V19_NO_SRC_LAST=1): drop the last layer's source machinery entirely (D3(15)
    := 0, no transports; would save 30 more units -> 0.4478xB): mse 2.154/2.182e-7 = 10x
    WORSE -> DEAD. The marginal skew of the FINAL pre-activation is essential to the final
    mean (10x in mse); F63's "D3 -> mean worthless" was about D3's chain effect relative
    to the D21 -> covariance channel, not about the last layer's own skew term. The 30
    transports at layer 15 stay.
  * COST-FLOOR AUDIT of the exact hub-source chain (this session, by inspection; closes
    the "cheaper exact algorithm" question one level deeper than F63/F66/F67):
    (a) forming a leg from the propagator (A = P a, 1 unit) costs exactly what
    transporting it costs, so the shared-propagator nesting P_s(l) = P_s'(l) P_s(s')
    buys nothing: every old source still needs 2 dense n x n legs formed + 2 dense
    contractions = 4 units/source-layer; (b) the D21 contraction is a Hadamard-then-
    matmul (x_q o p_q) y_q^T, and Hadamard breaks matmul associativity: row-rank-r
    factored legs must be re-densified (n^2 r) before the Hadamard, Khatri-Rao
    alternatives cost n^2 r^2, Tucker cores cost n r^3 -> row compression is linear in
    r/n at best (~7 r/n units) and F66 needs r >= n/2, i.e. <= 12% gain; (c) hub-column
    (CP) truncation IS cost-linear but the hub weights w2_q vary only ~3x across neurons
    and the birth covariance fluctuation is a full-rank Wishart-like matrix, so columns are
    generic -- no heavy tail to cut; (d) a single-leg symmetric-CP form (2 units/source-
    layer: transport V + (V o V) diag(lam) V^T) needs symmetric rank 3n by polarization
    (x^2 y = [(x+y)^3 - (x-y)^3 - 2y^3]/6) -> 6 units, worse; (e) randomized estimation of
    the Hadamard contraction (cube-root-of-unity probes) has relative noise sqrt(n/N) ->
    N ~ 1000 n probes for the required precision; dead. (f) PRECISION REQUIREMENT
    (from F65/F66 numbers): extra mse ~ 4.2e-6 x eps^2 where eps = relative rms error of
    D21 (eps = 1 recovers cov-prop 4.4e-6; F65 keep=0 eps ~ 0.45 -> 9e-7; F66 a=3 r=384
    -> eps ~ 3.7%; skip source 0 -> eps ~ 15%). Staying within +10% of 2.1e-8 needs
    eps <= 2.2% = 99.95% D21 energy fidelity; that is why every "keep 99% of the energy"
    compression fails. CONCLUSION: the exact chain is at its floor (V19); the 0.21xB
    tier is not a compression of this representation. What remains: (i) F66 option C
    (a=3 r=384 shared basis, +8% at best), (ii) a different closure family, (iii) the
    leaders' P2 write-ups (Oct 24).
  * RESIDUAL-TIME FUSION (V20 = V19 + exact op-stream refactor, scratch/make_v20.py;
    probes starterkit probe_resid_v19.py, profile_v19.py/profile_v20.py --gaps; logs
    runs/v19_profile_d0.log, v20_profile_d0.log): the per-layer fnp.stack of ~50 dense
    term products costs 4 ms of residual (memcpy of 200 MB) and each fresh (n,n) multiply
    0.10 ms vs 0.01 ms with out=; strided qr/matmul input (w32[:, :r] views) 0.9 vs 0.1 ms.
    Writing the products with out= into one persistent (T,n,n) buffer + contiguous
    sketch copies: dump-0 residual 0.357 -> 0.289 s (-19%), values bit-identical to V19,
    C unchanged (0.4776xB). Remaining gaps: qr 2.3 ms x 12 (28 ms), vector reshapes 258 x
    0.09 ms (24 ms), dense multiply/add ~40 ms. V20 is the base for any next feature
    (cap 0.400 s; V18 shipped at 0.328 s).
  * WRITE-UP: the eps^2 law (extra mse ~ 4.2e-6 eps^2, eps = D21 relative error) is a
    one-line quantitative statement of the K3 channel's leverage on the mean of a deep
    ReLU net, and the final-layer skew ablation (10x) shows the (3,) slice's direct
    effect is not negligible even though its chain effect is dominated by (2,1).


- F72 (2026-09-03) — F66 OPTION C WORKS ON THE V19 CHAIN: an age-gated SHARED basis for old
  K3 sources cuts C 0.478 -> 0.383xB at +3.4% raw (8 dumps) = local score 1.046e-8 ->
  8.65e-9 (-17%), ported as V21/V22. F66's "flat" verdict was a V16b-chain artefact plus a
  double-counted cost model; on the regen+feedback chain the confinement error is a fraction
  of the closure error, and the trailing Q^T of the contraction is paid ONCE per layer.
  Lean: scratch/lean_k3_aug.py (+confine_old_, age_rank/age_eig kwargs, from lean_k3.py),
  g_regen_probe.py tokens agerank:a:r / ageeig:randq; logs runs/optc_ladder_d01.log,
  optc_ladder2_d01.log. Port: scratch/make_v21.py (V20 -> estimator_v21.py), make_v22.py
  (+ prefit LAM table); test_v21/test_v22.py (ranks arg = R_OLD, V21_AGE_OLD/V21_QPASS env);
  logs runs/v21_*_d01.log, v21_port_ladder_d01.log, v21_a4_r384_d0-7.log, v22_a4_r384_d0-7.log,
  v19_d0-7.log, mini_v22.log.
  * LEAN LADDER (port-faithful config regen:c+no31+feedc+usec+fitfull+rres:16+sb2+nowk22
    +d21r:16, mean dumps 0/1, control 2.1428e-8; a = age gate in transports, r = rank):
    a=3 r=384 2.322e-8 (+8.4%) | a=3 r=512 2.175 (+1.5%) | a=3 r=256 4.21 (+96%, DEAD) |
    a=2 r=512 2.320 (+8.3%) | a=4 r=384 2.164 (+1.0%) | a=4 r=320 2.286 (+6.7%) | a=4 r=256
    2.617 (+22%) | a=5 r=256 2.289 (+6.8%) | a=5 r=192 2.678 (+25%) | a=6 r=192 2.294 (+7%)
    | a=6 r=256 2.159 (+0.8%). Randomized basis instead of exact eigh at a=4 r=384: rand1
    2.179 (+1.7%), rand2 2.152 (+0.4%) -> the one-pass sketch costs <1% here. Rule of thumb:
    r/n ~ 3/8 at age 4, 1/4 at age 6; the cliff (F52's) sits one notch below each.
  * PORT (V21 = V20 + shared basis): old sources keep static factors FA/FP (r x n) and one
    transported basis Q (n x r); each layer their dense legs are FORMED from the factors
    (2 n^2 r, the Hadamard machinery is untouched) and the two dslice contractions run
    through the factors ([sum LA FA^T + LP FP^T] Q^T, 2 n^2 r/source + one n^2 r/layer).
    One source joins per layer: basis = one-pass randomized range finder on the weighted leg
    Gram (weights 9 + w2^2 + 9 e^2 on A-type legs, 1 + s^2 on P-type, i.e. X1/Y1/M/P as in
    the lean Gram; sketch = w32[:, :r]; post-wick legs like the lean; Gram core S (r x r)
    maintained in the factor basis so no old leg is re-densified). Thin legs Z/Zf and the K4
    feed columns stay exact. Join skipped at the last layer (D3-only use). Off-suite shapes
    with n <= r run the dense V20 path (smoke: bit-identical). Kill-switch V21_NO_CONFINE=1.
  * PORT LADDER (real flopscope, dumps 0/1; V19 = 2.157e-8 @0.4777 -> 1.030e-8):
    a=3 r=384 q1 2.576e-8 @0.3621 -> 9.33e-9 | q2 2.408 @0.3841 -> 9.25e-9 | a=4 r=384 q1
    2.201 @0.3834 -> 8.44e-9 | a=4 r=384 q2 2.153 @0.4034 -> 8.69e-9 | a=4 r=320 q2 2.282
    @0.3727 -> 8.51e-9 | a=5 r=256 q2 2.280 @0.3697 -> 8.43e-9. The frontier is flat at
    ~8.4-8.5e-9 for a in 4-5; a=3 pays +12-19% raw (young legs are not yet low-rank).
    SHIP a=4 r=384 q1 (lowest raw on the frontier). 8 dumps: V19 2.190e-8 vs V21 2.264e-8
    (ratio 1.034, per-dump 0.999-1.085) @0.3834xB. Cost anatomy (units of 2n^3): 54 dense
    young pairs x4 = 216, 66 old pairs x ~1.5 (form 0.75 + contract 0.75) = 99, 10 joins x
    ~4 = 40 (sketch 1.5, factor rotation 0.28/old source, joiner factors 0.75, S update
    0.3), C_pre 31, thin ~8 -> 0.383xB measured. A second range-finder pass costs +0.02xB
    for -2% raw: not worth it. Residual 0.288 s/MLP (idle machine; 0.7-0.8 s seen ONLY
    under 22-thread contention from the lean ladder -- measure residual alone).
  * LAMBDA PREFIT (F68 item, V22 = V21 + table): lam fitted at the PRE-activation
    (G_pre_off vs C_pre_off, the quantity the port uses) on the full dense chain, 8 dumps,
    per-layer rel. sd 2-10%: [4.99e-3, 8.09e-3, 9.83e-3, 1.055e-2, 1.085e-2, 1.083e-2,
    1.059e-2, 1.005e-2, 9.65e-3, 9.17e-3, 8.77e-3, 8.39e-3, 8.06e-3, 7.68e-3, 7.26e-3];
    differs from the post-ReLU table mainly at layer 0 (1.95e-3 -> 4.99e-3, R^2 0.51 there).
    Lean 8 dumps 2.0705e-8 vs 2.09e-8; port V22 vs V21: -0.3% on every dump (8.653e-9 vs
    8.682e-9). Probe trap fixed: EXTRA.pop("prefit") made dump >= 1 replay dump 0's table
    past its end (IndexError) -- the per-dump-pop bug class again.
  * WHY F66 SAID FLAT: (i) it was measured on V16b (raw 3.45e-8, D21-error-dominated), where
    the same absolute confinement error is a larger fraction; (ii) its cost model charged
    2(r/n)^2 + 6(r/n) per old pair with a per-source trailing Q^T; the shared basis makes it
    4(r/n) + one n^2 r per layer. The eps^2 law (F71) still holds: a=4 r=384 keeps D21 to
    ~1.5% rms (extra mse 7e-10 = 4.2e-6 x 0.015^2).
  * WRITE-UP: "old third-cumulant sources of a deep ReLU chain live in a shared subspace of
    dimension ~3n/8 after 4 transports and ~n/4 after 6, and the Hadamard structure of the
    (2,1) contraction is compatible with a factored basis if the dense legs are re-formed" is
    a clean cost identity; with F63/F68 it brings the K3+regen-K4 chain to 0.38xB.
  * MINI (100 MLPs): 8.52e-9 adjusted (V19 1.03e-8), raw 2.22e-8 (2.16e-8), mult 0.3834, 0/100
    failed, residual 0.29 s/MLP. SUBMITTED #329584 (2026-09-03 ~05:05 UTC). -> GRADED 8.10e-9 adjusted (raw 2.12e-8 vs V19's
    2.083e-8 = +1.8% raw on the LB, mult 0.383, 100/100, 0 failed): rank 9 -> 7/95. Gap to #6
    fklassen 7.2e-9 = 11%; to the 4.5e-9 tier = 1.8x (their mult 0.21-0.25 at raw 1.8-2.2e-8).
  * TIERS (lean, dumps 0/1, runs/optc_ladder3_tiers_d01.log; nested: sources of age >= 7 re-
    confined to a lower rank inside the age>=4/r=384 group): +7:256 2.155e-8 (FREE vs 2.164),
    +8:256 2.161, +6:256 2.179 (+0.7%), +7:192 2.192 (+1.3%). Port estimate: the 36 old-old
    pairs go 1.5 -> 1.0 unit and their joins get cheap (factored joiner) -> ~-5% C at r=256,
    ~-8% at r=192. lean_k3_aug.py age_rank now accepts a list of (age, rank) tiers;
    g_regen_probe.py: repeat the agerank token.
  NEXT: V23 = two-group port of the tiers (age>=4 r=384, age>=7 r=256); then tiered ranks (a second, lower-rank group at age >= 7: ~-4% C) and
  r=320 at a=4 (+5% raw for -3% C) are the remaining small knobs; the dense young pairs (0.21xB)
  are now the largest cost block.

- F73 (2026-09-03) -- NESTED SECOND TIER PORTED (V24): the oldest K3 sources (age >= 7)
  re-confined to a rank-224 sub-basis INSIDE the shared rank-384 basis cut C 0.3834 ->
  0.3667xB at +1.2% raw (8 dumps) = local score 8.653e-9 -> 8.377e-9 (-3.2%), the lean
  "free" tier confirmed in the port at a smaller gain than the -5% estimate. Two ports were
  built and measured; the lean-faithful NESTED form beats a second transported basis.
  Generators scratch/make_v23.py (separate bases, general tier list) and scratch/make_v24.py
  (nested, SHIPPED) from estimator_v22.py; tests test_v23/test_v24.py; logs runs/
  v23_tiers_d01.log, v24_nested_d01.log, v24_nested2_d01.log, v24_7_192_d0-7.log,
  v24_7_256_224_d0-7.log, mini_v24.log.
  * V23 = SEPARATE bases: group g owns its own transported Q_g (n x r_g), n-space range
    finder at each move, per-member factor rotations, its own trailing Q_g^T in the
    contraction. Ordered tier list (oldest first), each group takes one source per layer
    from the next-younger group without re-densifying (Gram term Q1p S_s Q1p^T, new factors
    (Qn^T Q1p) FA1, donor core S -= S_s). "4:384" alone reproduces V22 to the last bit
    (parity). 4:384 + 7:256: dumps 0/1 C 0.3834 -> 0.3745 (-2.3%), raw +0.2%, 8.413 ->
    8.234e-9.
  * V24 = NESTED (lean-faithful "youngest gate first, then the older gate on the confined
    legs"): tier-2 factors FA2/FP2 (kb, r2, n) relative to a sub-basis U (r1 x r2) of the
    tier-1 basis Qc, A_s = Qc U FA2_s. Legs formed from QU = Qc U (n r1 r2 once per layer,
    then r2/n per leg); the contraction [sum LA FA2^T + LP FP2^T] (n x r2) is lifted by U^T
    into the tier-1 inner product, so the single trailing Qc^T per layer is unchanged; U
    rotates with the tier-1 factors (U' = Tq U, r1^2 r2, no per-member rotation); the move
    happens right after the tier-1 rebuild in its orthonormal coordinates Qn: S_s = FA1 dA
    FA1^T + FP1 dP FP1^T (r1 x r1, 0.28 units), G2 = S_s + U S2 U^T, U_new = two-pass range
    finder of G2 in r1-space (sketch w32[:r1, :r2] -- no n-sized QR), FA2_j = U_new^T FA1,
    old FA2 rotated by U_new^T U (2 (r2/n)^2 per member), S2 = U_new^T G2 U_new, donor
    FAo = FAo[1:], Sg -= S_s. No second transport, no second trailing product.
    Bit-identical to V22 on all smoke shapes and with V24_AGE_OLD2=0.
  * LADDER dumps 0/1 (V22 8.413e-9 @0.3834; score = mean mse x C/B): 7:256 8.143e-9
    @0.3718 | 7:224 -- | 7:192 7.953e-9 @0.3616 | 7:160 8.080 @0.3566 | 7:128 8.857
    @0.3517 (CLIFF, raw +15%) | 6:256 8.212 @0.3676 | 6:192 8.731 @0.3547 (cliff) | 8:256
    8.234 @0.3754 | 8:160 8.124 @0.3637. Nested 7:256 (8.143) beats separate 7:256 (8.234)
    at -0.7% C, same raw: the cheaper mechanics are worth ~1%.
  * 8 DUMPS (the honest frontier; V22 raw 2.257e-8, 8.653e-9): 7:256 raw 2.268e-8 (+0.5%)
    @0.3718 -> 8.433e-9 | 7:224 2.285e-8 (+1.2%) @0.3667 -> 8.377e-9 (SHIP) | 7:192
    2.326e-8 (+3.0%) @0.3616 -> 8.409e-9. Dumps 0/1 FLATTERED r=192 (their raw +0.25%
    vs +4-7% on dumps 2/3/5): the frontier is flat within 0.7% for r in 192-256, so pick
    the point with the smaller raw loss. Rule of thumb extended: r/n ~ 3/8 at age 4, ~7/32
    at age 7, cliff at 1/8.
  * COST ANATOMY of the gain (units of 2n^3): 28 tier-2 pairs x (1.5 -> 2 x 224/1024 x
    2 = 0.875) = 17.5, plus 7 last-layer formings and 7 fewer tier-1 rotations per join
    (0.28 x (0+1+...+6) = 5.9), minus the moves (~0.6-1.0 each) and QU (0.08/layer):
    net ~17 units = 4.4% of 384 predicted, 4.4% measured (0.3834 -> 0.3667). Adding a
    THIRD nested tier (age >= 10 at r=128 inside 224) would touch 10 pairs for ~1.5 units
    (0.4%): not worth a port.
  * ARITHMETIC (no run needed): confining the age-4 source at r=512 (lean +1.5% raw)
    saves 2 units per layer on its pair but the dense join at r=512 costs ~4 (sketch 2,
    factors 1, S 0.5, rotation 0.5) vs ~3.4 at 384, and the extra factor-space move ~1.5
    per layer: net ~0 at +1.5% raw -> DEAD by inspection. The shared-basis family is
    exhausted at ~0.36xB; the 54 dense young pairs (0.21xB) remain the floor of this
    closure.
  * TRAP (again): heredoc stdin is NOT delivered to run_in_background Bash commands; a
    generator patch printed nothing and the shipped default stayed 256 until a foreground
    re-patch. Patch in the foreground, run in the background.
  * WRITE-UP: nested age tiers = the natural extension of F72's cost identity: "after 7
    transports the old third-cumulant sources live in ~7n/32 dimensions of the ~3n/8
    subspace they entered at age 4", and the nesting keeps all factor-space work O(r^2 n).
  * MINI (100 MLPs): 8.25e-9 adjusted (V22 8.52e-9, -3.2%), raw 2.25e-8 (2.22e-8, +1.4%),
    mult 0.3667, 0/100 failed, residual 0.277 s/MLP. whest validate + package OK. SUBMITTED
    #329633 (2026-09-03 ~13:45 UTC) -> GRADED 7.90e-9 (raw 2.15e-8, mult 0.3667, 100/100, 0 failed):
    adjusted -2.5% vs V22's 8.10e-9, NEW LIVE BEST, yet rank 7 -> 8/98 because hyojun_kwon
    moved 8.3e-9 -> 7.00e-9 (raw 3.17e-8 at ~0.22) the same hour. The #6-#7 line is now
    7.0-7.2e-9 (we need -9%); the 4.1-4.7e-9 tier needs -45%.

- F74 (2026-09-03) -- COMPANION PAPER RE-READ FOR AN O(L n^3) K3 FORM: NOT THERE. A full
  read of 2605.05179 S.4.3 (factorized K=3), S.8.5 (general K), App. C/D/J and Table 1
  (research agent, pp. 5-8, 18-19, 26-27, 47-52, 67-68) closes the F71 "re-read Wu S.4.3
  paired/shared-middle groups" item: those terms do not occur in the paper (zero hits in
  the full text; they came from our own notes/forum vocabulary).
  * The paper's ONLY factorized K3 representation is a concatenated factor list that grows
    by exactly 6n per layer (J^(l) = 6 n l, eq. 15-20): three n x J legs transported
    independently ("summing factored tensors sums the number of factors"), (2,1)-diagonal
    per factor block via eq. 16 (Hadamard of two legs, dense contraction against the
    third). Stated cost O(n^3 L^2); Table 1 leading term 30 n^3 L^2 (72 n^3 L^2
    augmented) = ~8.2e12 FLOPs at our shape = 3.7x the budget. Our 2-leg / 4-unit scheme
    (F63) is already ~7x tighter than the paper's algorithm.
  * It never merges, truncates or compresses old factors; it notes but does not exploit
    that all live legs are transported by the same operator (eq. 19, our A_s = P_s a_b);
    App. D and Sec. 7.3 list depth-independent error as an OPEN problem ("we leave this
    problem to future work").
  * VERDICT: the leaders' ~14 units/layer cannot come from anything in the paper. It is
    (a) a merging/compression of old sources not described there, (b) a memoryless closure
    (Puffi SSC lead, F65 says our slice version of it is DEAD), or (c) not cumulant
    propagation. Do not re-open the paper for cost ideas; the write-up angle is that F63 +
    F72/F73 already beat the published algorithm's cost by ~10x at equal content.

- F75 (2026-09-03) -- ADAPTIVE PER-MLP LAMBDA (V25): the regenerated kappa4 off-diagonal
  coefficient tracks an ONLINE observable, mean(dG)/mean(var) (transported kappa4 diagonal
  over pre-activation variance), with log-log slope ~1 at every layer; making the table
  adaptive removes most of the per-MLP lambda spread but buys only -0.3% raw (8 dumps
  2.2847 -> 2.2782e-8); a 0.95 global table scale adds another -0.3% (2.2704e-8). The frozen
  table is NOT the raw bottleneck. Probes: starterkit lam_obs_probe.py (V17_DEBUG hook on a
  copy estimator_v24s.py) -> scratch/lam_obs_d0-7.npz; scratch/lam_prefit_perdump.npy (the
  15 pre-activation lambdas per dump parsed from runs/lam_prefit_d0-7.log); port scratch/
  make_v25.py -> estimator_v25.py (env V25_BETA, 0 = V24 values; V17_LAM_SCALE default 0.95);
  logs runs/v24_lamscale_d0-7.log, v25_beta_d0-7.log, mini_v25.log.
  * STRUCTURE OF THE SPREAD: per-dump LS lambdas deviate 2-4% (layers 1-10) rising to 6-10%
    (layers 11-14) from the 8-dump mean; 64% of the deviation energy is ONE mode whose layer
    profile grows with depth (0 at L5 to -0.59 at L14); the low-mse dumps (4/5/6: 1.72-1.94e-8)
    carry the lowest deep-layer lambdas.
  * OBSERVABLES (per-layer correlation with the lambda deviation across 8 dumps, layers
    1..15): mean(dG) +0.84..+0.99 at EVERY layer (pooled +0.89); mean(var) +0.8..+0.96 from
    L5 on but ~0 early; rms(mu) weaker (+0.5..0.85); the normalized kurtosis dG/var^2 is
    NEGATIVE at depth (-0.7..-0.9) -> the mechanism is "more kappa4 diagonal at fixed
    variance = larger off-diagonal core", not a kurtosis ratio.
  * FIT (log lam/mean vs log obs/mean): dG alone slope 0.45 pooled (per-layer 2.1 -> 0.3);
    dG/var slope 1.08 pooled, per-layer 0.9-1.5 at depth, residual spread 1.0-2.2% (from
    2.3-10%), leave-one-dump-out identical -> rule lam_l = LAM[l] x ((mean dG/mean var)/
    REF_R[l])^beta with beta = 1 and a 15-entry REF_R table (8-dump mean ratio per layer).
    dG/var^1.5 is worse (slope unstable, residual 3-4%).
  * PORT: dG(lam) = WW g_prev + lam (var - WW var_prev) is affine in lam, so the rule costs
    one extra n^2 matvec + O(n): compute dG at the table value, form the ratio, rescale lam
    (clamped to [0.5, 2] x REF so an odd MLP cannot NaN into the zero fallback), recompute
    dG as a vector update; the adaptive value is then consumed by everything at that layer
    that used lam_prev (dG, wk431, the birth feed c1_b). Smoke shapes bit-identical to V24
    (regen is suite-gated).
  * LADDER 8 dumps (V24 2.2847e-8 @0.3667 = 8.377e-9): beta 1.0 2.2782e-8 | 1.2 2.2775 |
    0.8 2.2783 (flat in beta) | + LAM_SCALE 0.95 2.2704e-8 (8.325e-9, -0.6%, SHIP) | 0.92
    2.2727 | 0.90 2.2752. V24 alone: scale 0.9 2.2809, 1.1 2.3373 -> the table is centred
    to ~5%. In-sample caveat: REF_R/beta/scale fitted on the same 8 dumps; the 100-MLP mini
    (92 unseen) is the honest check.
  * WHY SO SMALL: the lean chain with per-dump in-sample lambdas is 2.07e-8 (8 dumps) vs the
    port's 2.28e-8; the 10% port gap is confinement (+4.3%, F72/F73), the rank-16 residual
    leg, f32 -- not the lambda table (F68's "1-4% transfer loss" was measured on the lean
    chain per dump, where other errors are smaller). The closure's intrinsic 2.07e-8 is the
    raw floor of K3 + memoryless kappa4 at L=16; the leaders' 1.86e-8 (marius_binner) needs
    content this chain does not carry.
  * WRITE-UP: "the kappa4 off-diagonal coefficient of a deep ReLU chain is proportional to
    the ratio of its kappa4 diagonal to its variance, per MLP" is a one-line closure law
    (beta = 1), the natural sharpening of F68's "lambda is a smooth universal function of
    depth".
  * MINI (100 MLPs, 92 unseen by the fit): 8.17e-9 adjusted (V24 8.25e-9, -1.0%), raw 2.23e-8
    (2.25e-8), mult 0.3667, 0/100 failed, residual 0.288 s/MLP. SUBMITTED #329644 (~15:25 UTC) -> GRADED
    7.80e-9 (raw 2.13e-8, 100/100, 0 failed): -1.3% vs V24's 7.90e-9, NEW LIVE BEST, rank 8/98
    unchanged (#7 fklassen 7.2e-9, #6 hyojun_kwon 6.9e-9).
  * RANK KNOBS on the V25 chain, 8 dumps (V25 2.2704e-8 @0.3667 = 8.325e-9): R_RES 16 -> 32
    raw 2.2434e-8 (-1.2%) @0.3745 (+2.1%) = 8.40e-9 (WORSE); R_FB 16 -> 32 raw 2.1932e-8 (-3.4%)
    @0.3895 (+6.2%) = 8.54e-9 (WORSE). No cheap knob is left on this chain
    (runs/v25_rres_rfb32_d0-7.log).

- F76 (2026-09-03) -- CAPPED HUB-COLUMN "ATOMS" ARE DEAD BY MEASUREMENT: carrying the
  accumulated K3 with a fixed budget of hub columns (Puffi P1 write-up: "caps that
  cross-family block to the four highest-scoring atoms"; F71 had rejected column truncation
  by inspection only) cannot reach the ~2% (2,1)-slice precision the chain needs (F71 eps^2
  law). Lean torch, exact K3-simple chain on P2 dump 0 (scratch/capped_atoms_probe.py, runs/
  capped_atoms_analyze_d0.log): every dense source is Sym sum_j X[:,j] (x) P[:,j] (x) Y[:,j]
  (+ M terms), one column = one atom; columns scored by their (2,1) contribution bound
  sum_terms |left_j| |right_j|, the top J kept globally across sources, the rest zeroed in
  every leg; one-shot D21/D3 relative error per layer (no compounding).
  * D21 rel error, cap 4n columns: L5 0.090 (mass kept 0.94) | L7 0.185 | L9 0.250 | L11
    0.316 | L13 0.359 | L15 0.387 (mass 0.73 of 15n live columns). Cap 8n: L9 0.031 |
    L11 0.071 | L13 0.109 | L15 0.140 (mass 0.93). Cap 2n: 0.40-0.61; cap n: 0.64-0.78.
    D3 errors identical to D21's. Random 4n: 0.31 (L5) -> 0.77 (L15): the ranking helps
    2x but the importance spectrum is FLAT (top quarter of the columns = 73% of the mass,
    a ~3x norm range, exactly F71's "generic" Wishart-column picture).
  * Cost arithmetic: a 4n-column state costs ~16 units/layer (0.25xB, the leaders' tier)
    and is 5-20x too inaccurate; 8n costs ~32 units/layer = today's chain and still leaves
    14% at depth. There is no cap that is both cheap and accurate: the hub index of the
    accumulated K3 does not compress, only its row space does (F72/F73).
  * Consequence for the leaders' 0.21xB tier: it is NOT a column-capped hub chain either.
    Together with F65 (memoryless slices dead), F66 (skip-birth 4x), F61 (windowing dead),
    F72/F73 (row-space compression exhausted at ~0.37xB) and F74 (nothing in the paper),
    every compression of the factorized-K3 representation is now measured or proven dead;
    the leaders run a DIFFERENT expansion whose D21-equivalent content is cheap by
    construction. Puffi's "atoms" cannot be hub columns of this kind.
  * Kill criterion of the 2026-09-03 plan step 1 ("prototype the leaders' closure family in
    the lean chain") is hit in one session: no candidate representation survives. The
    next research input must be external (P2 write-ups after close, forum) or a new
    expansion, not another compression experiment.
  * WRITE-UP: the flat column-importance spectrum (mass share ~ J/(kn) with a 2x ranking
    gain) is the quantitative form of "per-source memory is irreducible" in factorized
    cumulant propagation; with F65's 20%/layer high-rank slice residual and F72's row-space
    law it completes the picture of WHERE the K3 information lives.

- F77 (2026-09-05) -- V25 LEDGER AUDIT + FLOPSCOPE ALIAS PRICING: the symmetric-tag lever
  is EXHAUSTED on the V25 chain, and the Sep 4-5 board-wide cost drop points at the
  leaders carrying SYMMETRIC n x n objects (sandwich/gram-priced), not hub contractions.
  Scripts DATA starterkit audit_v25.py (op_log aggregation by op/subscripts/shapes) and
  probe_alias_pricing.py; logs runs/audit_v25_d0.log, runs/probe_alias_pricing.log.
  * BOARD (17:20 UTC): we are #11 at an unchanged 7.80e-9 (was #8 on 09-03). Every team
    that resubmitted since 09-04 04:00 UTC cut its multiplier 15-30% at FLAT raw: Puffi
    0.209 -> 0.157 (raw 2.15 -> 2.23e-8), hyojun_kwon 0.218 -> 0.155 (3.17 -> 2.96e-8),
    fklassen 0.289 -> 0.216 (2.49 -> 2.50e-8), Camaro 0.377 -> 0.319 (2.73 -> 2.73e-8),
    J2W 0.213 -> 0.186, marius_binner 0.220 -> 0.200, oqaris 0.219 -> 0.209; teams that did
    NOT resubmit (Activation-Regions 0.193, jamespayor 0.480, us 0.367) are unchanged.
    NOT a toolchain event: flopscope 0.12.1 / whestbench 0.16.1 shipped 08-29, we
    installed them 08-31, our 09-03 grade (0.3667) equals the local 0.12.1 price, and
    the kit's upstream (last commit 08-29) says 0.12.1 only makes fnp.full(array) pricier.
    Resubmitting V25 unchanged would NOT reprice. Forum has no method post (the 3 new
    threads: two organizer YouTube talks posted by Cipo/Puffi, a "please share, LLMs"
    joke, konstantin's residual-cap question).
  * V25 LEDGER (dump 0, flopscope's own op_log; 375.5 units, unit = 2n^3 = 2^31):
    einsum 220.6 u (58.8%) + matmul 148.5 u (39.6%) + qr 2.9 + elementwise ~3. The
    money: hub contractions 'kij,kcj->ic' (LA vs A_st, LP vs P_st) 100 u (27%); young
    transports WD @ (A,P,Z stacks) 'cij,cjk->cik' 80 u (21%); shared-basis contractions
    'kij,kqj->iq' ~30 u + formings 'iq,kqn->kin' ~33 u; W @ Q / WD @ Q 27 u (51+20
    calls); C_pre sandwich 'ij,ia,jb->ab' 22.5 u = 15 x 1.5 u, i.e. ALREADY at the 0.75
    symmetric rate (C is tagged at line 929); 16 x (n,n)@(n,n) matmuls 16 u (4.3%, one
    per layer, unidentified); joins/moves/QR ~20 u; term table 'tij,ti,tj,tg->gij' 2.5 u.
  * ALIAS PRICING (flopscope 0.12.1, n=1024): einsum with the SAME array on both sides
    ('ic,jc->ij' (A,A); 'ia,ib->ab' (W,W); weighted gram 'ia,i,ib->ab' (W,d,W);
    'ic,jc,ic,jc->ij' (A,A,P,P); stacked 'kic,kjc->ij' (Ak,Ak)) bills 0.5x dense and
    tags the output symmetric; the same contraction with distinct arrays bills 1.0x;
    matmul A @ A.T bills 1.0x (einsum only); the Hadamard-on-one-side form
    'ic,ic,jc->ij' (A,P,A) bills 1.0x (output genuinely non-symmetric). Quirks: the
    column-weighted alias 'ic,c,jc->ij' (A,d,A) and a sandwich on a doubly-tagged C
    raise SymmetryError "max deviation = inf" (flopscope validation bug, not ours).
  * CONSEQUENCE: our hub contraction is D21 = (2/3)(a o P) D12 a^T + (1/3)(a o a) D12 P^T
    (F45 leg algebra) -- a left-Hadamard leg against a plain leg, output NON-symmetric,
    priced 1.0x by construction; the transports are dense non-symmetric. No tag, alias
    or output-symmetry discount exists for the 180 u that dominate V25. Only crumbs:
    the shared-basis join grams F d F^T (lines 565/580, ~7 u as matmul) and the
    sketch grams could take the 0.5x alias form (~4 u), plus whatever the 16 one-unit
    matmuls are: ~3-5% total, rank-neutral on today's board (#10 = 7.10e-9, -9%).
  * READING OF THE LEADERS: a 25-30% cut at identical raw is exactly the sandwich
    (0.75x) / gram (0.5x) discount taken over a chain whose cost is dominated by
    symmetric n x n objects -- i.e. the 0.15-0.22xB tier transports SYMMETRIC matrices
    (covariance-response modes, Puffi SSC style), ~7 sandwiches or ~10 units per
    layer, and reaches raw 2.0-3.0e-8 with them. Our K3 hub chain cannot be re-expressed
    that way (F67, F76); reaching their tier needs the different expansion F76 named.
  * OPTION-1 VERDICT (same day, probe_gram_forms.py / probe_gram_forms2.py, runs/
    probe_gram_forms2.log): NOTHING TO TRIM. (i) The 16 one-unit matmuls are the 15
    newborn birth transports W @ a_b (line 628) + the trimmed last layer's C @ w32
    (line 512): all necessary. (ii) The weighted aliased gram einsum ('ia,i,ib->ab'
    (W,d,W), any operand order, contiguous or view) is computed by a path that is only
    symmetric to rounding, then validated by _check_generators (atol 1e-6 / rtol 1e-5,
    reported as "max deviation = inf"): it PASSES at operand scale <= 0.1 (billed 0.50 u,
    maxdev 1e-6) and RAISES SymmetryError at scale 1.0 -- a runtime exception = zero
    prediction for that MLP, so it is unshippable for the join/tier grams F d F^T whose
    entries are O(1-100). The exact alternative (sign-split sqrt(d): two unweighted
    aliased grams at 0.0705 u each) costs exactly today's 0.1407 u. (iii) The unweighted
    2-operand alias ('ia,ja->ij' (F,F)) is exact and passes at 0.5x, but V25 has no
    unweighted gram anywhere. (iv) C_pre via Cholesky + aliased gram = 1 + 0.5 = 1.5 u =
    the tagged sandwich already paid. Net: V25's ledger has no exact trim left; the
    frontier of this chain is 0.3667xB at raw 2.13e-8, and the next rank move must be a
    new expansion (option 2), not another ledger pass.

- F78 (2026-09-05) -- GAUSSIAN-MIXTURE CLOSURE IS DEAD (mean-shared scale mixture on the
  fluctuation part, the natural "symmetric n x n state at K sandwiches/layer" reading of
  F68's kappa4_off = lambda C_off law). scratch/mixture_probe.py (lean torch f64, gain-
  approx cov-prop per component, 3-node Gauss-Hermite split of each component's
  covariance scale with variance v_l per layer, greedy moment-matching merge back to K
  components), dumps 0/1, logs runs/mixture_probe_d01.log, mixture_probe_d01_k7_15.log.
  * Baseline K=1 (= cov-prop) 4.78e-6. K=3: v=1e-3 4.45e-6 | v=1e-2 2.52e-6 | v=LAM table
    2.59e-6. K=7: v=1e-2 2.33e-6 (best) | v=3e-2 1.16e-5 | v=1e-1 1.9e-4 | LAM 2.41e-6 |
    2xLAM 4.07e-6. K=15: 2.60e-6 (worse than K=7: the greedy merge proxy is not the
    bottleneck-optimal one, but the curve is flat). K=1 with any v = cov-prop (the merge
    right after the ReLU discards the split).
  * Kill rule (pre-registered: best raw > 1e-6 = dead) hit by 2.3x. A scale-mixture latent
    captures at most ~50% of the cov-prop error; the K3 chain's 2.1e-8 is 100x further.
    Consistent with F60 (the missing physics is the JOINT kurtosis structure that only
    the D21 slice / full cumulant tracking captures) and with F65/F76 (no low-dimensional
    surrogate of the K3 content exists). Note the homogeneity trap for future variants:
    a scale mixture that scales the MEAN as well is exactly invisible (ReLU nets are
    positively homogeneous), so only fluctuation-part mixtures are even candidates.
  * Intel sweep (docs/research/option2_intel_2026-09-05.md, agent): NO public P2 code or
    write-up of a symmetric-mode method; Puffi's P1 SSC (#327725) regenerates kappa4 from
    <= 4 symmetric response modes with a fitted 4x4 Lambda BUT still carries K3 atoms
    (cost 1.34x a K3 port at P1 shape) -> the 0.15-0.22xB tier is not a straight SSC
    port either. marius_binner's P1 PDF discloses a 5-level Strassen-Winograd recursion
    billed by flopscope "close to the analytical count" (-12.5% / -23.4% / -33% per 1/2/3
    levels) -> the Sep 4-5 board-wide drops (Camaro -15%, fklassen -25%, Puffi -25%,
    hyojun_kwon -29%, J2W -13%) fit 1-3 Strassen levels on matmul-dominated chains at
    least as well as symmetric tags. Winer's IPAM talk (Sep 2, "matching sampling on random
    models") is not yet online -- re-check https://www.youtube.com/@IPAMUCLA.

- F79 (2026-09-05) -- STRASSEN-WINOGRAD BLOCK PRODUCTS (V26): flopscope bills the
  recursion at its analytical count, and a pooled, batched, per-family implementation fits
  the 400 ms residual cap at THREE levels on the two dominant families: C 0.3667 -> 0.3125xB
  (-14.8%) at unchanged raw (8 dumps 2.2703e-8 vs V25 2.2704e-8), local 8-dump score
  8.32e-9 -> ~7.09e-9. Generator scratch/make_v26.py (asserted textual patch of V25 ->
  estimator_v26.py), harnesses test_v26.py / smoke_v26.py / audit_v26_resid.py / mem_v26.py,
  probes probe_strassen.py, probe_strassen2.py; logs runs/probe_strassen*.log,
  v26_L3_d0-7.log, mini_v26.log.
  * PRICING (probe_strassen.py, n=1024 f32, one (n,n)@(n,n)): dense 1.000 | L1 0.877 | L2
    0.773 | L3 0.686 of the dense count (7/8, 49/64, 343/512 plus the block adds at 1
    flop/elt); f32 relative error 7.7e-7 -> 1.35e-6 -> 2.4e-6 -> 3.4e-6. Naive recursion
    residual: 4 / 18 / 86 ms per product (28 / 217 / 1540 ops) = unusable; the pooled
    batched form (probe_strassen2.py): 0.7 ms per product FAMILY at L1 (36 ops), because
    every combo / product / assemble is one op over a whole (k, ...) stack written with
    out= into preallocated buffers (F53 discipline).
  * IMPLEMENTATION (make_v26.py, class _Strassen): recursive on batched operands with
    layouts plain X (bx,P,m,kd) @ Y (by,P,kd,w) (bx in {1,by}) and hub X (k,P,m,kd),
    Y (k,P,w,kd) -> sum_k X_k Y_k^T (the hub's (Y^T) blocks are read on the un-transposed
    Y, no transposes). Per level: 7 left combos + 7 right combos (5 adds + 2 copies each
    side) + 8 assemble ops + the recursion; leaf = ONE batched matmul (plain) or ONE
    batched matmul + ONE k-sum (hub; the einsum form 'kpij,kpcj->pic' takes a non-BLAS
    path at 0.5 s per call and doubled the wall time -- never use it). Pools shared by
    role and keyed by (role, level, P, h, q) with batch capacity 2(L-1). Two structural
    moves halve the op count: A and P legs interleaved in ONE (L-1, 2, n, n) ping-pong
    buffer so the young transports are ONE family per layer (b = 2 k'), and LA/LP
    interleaved the same way so the hub sum over (k, {A,P}) is ONE family (b = 2k).
    Dense fallback whenever a block side would not divide or falls under 128 (smoke
    shapes), so off-suite shapes are unaffected; with all knobs at 0 the op stream is
    V25's bit-for-bit (smoke: maxdiff 0 on all 7 shapes).
  * RESIDUAL ENGINEERING (audit_v25_resid.py / audit_v26_resid.py: the residual clock is
    the gap between ops, attributed to the op that follows; sum of gaps = 0.31 of 0.33 s):
    (i) first-call library init is ~50 ms per MLP (30 ms before the first qr, 24 ms before
    the first small matmul, 5 ms norm.pdf, 1-2 ms per first einsum subscript) -> one tiny
    call per op signature in setup() (off-budget, 0.03 s) = -48 ms measured; (ii) each
    fnp.reshape of a vector is a logged op at 0.055 ms while v[:, None] is a free view
    (0 ops) -> 34 sites converted, -8 ms; (iii) Strassen on the newborn transport (1 unit
    per layer) costs ~30 ms residual for 0.5% cost -> left dense (STRASSEN_NEW=0); (iv) the
    3-operand einsum 'kij,kij,kij->i' -> two pooled multiplies + one sum (no measurable
    gain; kept). Base chain residual: V25 0.284-0.288 s -> V26 L0 0.278 s.
  * LADDER dump 0 (levels on transports + hub; newborn dense): L1 C 0.3438 resid 0.296 s |
    L2 0.3274 / 0.309 s | L3 0.3125 / 0.318-0.319 s | wall 20 -> 26 s local (grader ran
    V25 at 31-37 s/MLP vs 20 s local -> ~45 s expected, cap 120 s). Memory peak 2.6 GB
    (cap 8 GB). SHIP = L3 on both families (V18 shipped at 0.328 s residual and graded
    100/100; V25 at 0.288). 8 dumps at L3: C/B 0.3081 with the newborn at L3 (0.3125
    without), mse 2.2703e-8, score 6.99e-9 (7.09e-9 at 0.3125).
  * REMAINING STRASSEN HEADROOM (not built): shared-basis formings 'iq,kqn->kin' (33 u)
    + contractions 'kij,kqj->iq' (32 u) + W@Q products (27 u) = 92 u; L1/L2 there would
    cut a further ~11-21 u (3-6%) for ~40-80 ms of residual, which the current base does
    not afford; the base chain's 596 fresh (n,n) multiplies + 327 adds (~35 ms) in the
    term/pk code are the next residual reserve. Bit-identical-order note: at L>0 the f32
    summation order changes; dump/mini raw are unchanged to 4 digits.
  * MINI #1 FAILED (2 of 100, MLPs 94 and 97 at residual 0.400 / 0.407 s; mean 0.372 s vs
    0.319 s on an isolated dump): two suite-level effects that single-dump tests never
    show. (a) The harness keeps every MLP's op log alive, so Python's cyclic GC scans a
    growing heap: measured 18-20 ms per predict for gen-0/1 plus 35-110 ms whenever a
    gen-2 collection lands inside predict, rising with the suite (test_v26_gc.py:
    gc.callbacks timing + tracked-object counts). FIX: predict() runs with gc.disable()
    and re-enables on exit (refcounting frees all temporaries; pure stdlib plumbing).
    (b) Allocating and freeing the ~2.5 GB of Strassen pools on every predict made the
    residual creep 0.344 -> 0.381 s over six MLPs in one process (V25's stream is flat
    at 0.306-0.315 in the same harness). FIX: pools persist on the estimator across
    predict() calls (rebuilt only if the depth changes the batch capacity). After both:
    L3 flat at 0.342-0.353 s in the op-log-retaining harness (= L0 + 37 ms), i.e. ~0.33 s
    expected mini mean. LESSON: validate residual on a multi-MLP single-process run that
    retains op logs, never on one dump.
  * MINI #2 (GC off + persistent pools, L3) STILL FAILED 7 of 100 (MLPs 15/37/39/51/75/
    76/99 at 0.404-0.417 s; mean 0.356 s): the growth is gone but the LEVEL is too high.
    Budget picture: V25 mini mean 0.288 s; Strassen L3 on the two families = +65-70 ms in
    the suite (~12 ms per level-family, ~1980 ops at ~0.02-0.03 ms; the leaf matmul/sum
    and the 1-D combos are the same price as any op); the suite tail is +45-60 ms above
    the mean on ~5-7% of MLPs (machine noise, scattered MLP ids). A clean 100 therefore
    needs a mini MEAN <= ~0.31 s: L1 (+~18 ms) fits, L2 (+~40) is marginal, L3 does not.
  * TWO BASE-CHAIN FUSIONS MEASURED USELESS (reverted): (a) gathering the 52 pairwise
    (n,n) term products per layer with fnp.take(out=) + ONE multiply: take bills 4 flops
    per element (+0.8% cost!) and the 6-MLP harness mean did not move (0.346 -> 0.356);
    (b) alpha/sigma power bases via fnp.power (exact) and the D3 einsum as pooled
    multiplies + sum: no residual change, and both break the level-0 bit-parity with V25.
    Removing ~880 cheap ops is invisible at the +-15 ms run-to-run noise, so the residual
    is NOT simply op count x 0.02 ms: the ~1000 einsum/matmul calls carry 0.1-0.3 ms of
    client-side gap each (half the residual), the ~5000 small elementwise ops ~0.02 ms.
    Next residual reserve = fewer heavy calls (merge the tier-1/tier-2 formings and
    contractions into one stacked call each, batch the 87 (n,n)@(n,16) feedback range-
    finder matmuls), not fewer vector ops.
  * SHIP DECISION: V26 defaults L1 on both families (C/B 0.3438, -6.2%, raw unchanged);
    level-0 stream bit-identical to V25 on all 7 smoke shapes; L2/L3 stay behind the env
    knob until the base residual is cut by ~40 ms.
  * MINI #3 (L1): 99 of 100, mean residual 0.325 s (V25 0.288 -> L1 costs ~+37 ms in the
    suite, twice the dump-0 estimate); the one failure is MLP 76 at 0.616 s, a one-off
    machine hiccup: in isolation (test_mlps_resid.py, dataset rows 76/15/37/0, twice
    each) it runs at 0.316-0.320 s like every other MLP (0.311-0.322). Mini score of the
    99: not reported by the harness (a failed MLP dominates the mean); mult 0.3455 on
    the mini MLPs. Shipped on the V18 precedent (mean 0.328 s graded 100/100).
  * RULES: Strassen is real arithmetic (fewer multiplications, every add metered), it
    was publicly disclosed by the current #1 (marius_binner, P1 write-up: a 5-level
    Strassen-Winograd recursion billed "close to the analytical count"), and a binding
    confirmation was requested from arc-whestbench@aicrowd.com on 2026-09-05 (draft in
    docs/sponsor_email_strassen_2026-09-05.md; the user sends it). Submitting without
    waiting, per the user's decision.

- F80 (2026-09-05) -- RESIDUAL ENGINEERING OF THE BASE CHAIN (V27) UNLOCKS DEEP STRASSEN
  (V28): the residual clock is dominated by RESULT ALLOCATION, not op count. Measured with
  the context's own residual clock (probe_resid_v27.py, 15 reps x 3, n=1024): a fresh
  (n,n) result costs 0.10 ms of residual, the same op written with out= into a pooled
  buffer 0.013 ms; diagflat 0.22 vs fill_diagonal(vec) 0.013 (bit-identical when the
  off-diagonal part has a zero diagonal); concatenating a (k,n,r) stack 0.25 vs a slot
  copyto 0.015; the PPL einsum 'kij,kjq->kiq' 0.34 vs matmul(out=) 0.037 (and 1.2 s ->
  8 ms of backend per call: the einsum forms 'kiq,kjq->kij' / 'kij,kjq->kiq' /
  'kij,kqj->iq' take non-BLAS paths 10-150x slower than matmul, same billing minus n^2);
  transposed / strided operands and strided out= views are FREE (0.02 ms); vector ops
  0.014 each. Generators scratch/make_v27.py (V26 -> V27) and make_v28.py (V27 -> V28);
  probes probe_ops_v27.py, probe_miss_v27.py, probe_gap_v27.py, probe_touch_v27.py,
  probe_resid_v27.py, probe_api_v27.py, probe_blocks_v27.py, probe_strided_v27.py,
  probe_first_v28.py, probe_qmm_v28.py; audits audit_v27_resid.py, audit_first_v28.py.
  * V27 (base chain, pooled): every fresh (n,n)/(k,n,r) result of the layer body goes
    out= into a persistent named buffer (class _Pool on the estimator, like the Strassen
    pools); thin stacks Z/Zf/L/R1T/R2T and the shared-basis factors FAo/FPo/FA2/FP2 live
    in ping-pong slot buffers (no per-layer concatenates); the PK->K matrix assembly is
    unrolled with out= in the generic loop's operation order (bit-identical); C = diagflat
    + sym -> add/multiply/fill_diagonal in place. Suite-like harness (6 MLPs, op logs
    retained, test_v27_gc.py) at L3: V26 0.331 s -> V27 0.170 s (-49%), wall 26 -> 21 s,
    C/B 0.3125 -> 0.3107 (matmul bills n^2 less than the einsum per product; the
    per-call fnp.reshape ops of the Strassen recursion were billed per element and are
    now one-time views), 8 dumps raw 2.2707e-8 = V26 2.2703e-8 (parity).
  * ACCOUNTING FACTS (flopscope 0.12.1): (i) the audit's "2.4 ms gap before every
    'kij,kij,kij->i' einsum" is an ARTIFACT -- on a path-cache miss the contract_path
    search runs before the op's recorded start and is booked to the free overhead
    bucket, so the next op shows a matching negative gap (explains F79's "no measurable
    gain" from splitting it); the path cache is keyed by operand shapes, so every layer
    (new k) misses -- harmless; (ii) einsum(..., out=) is compute-then-copy (2 ops, +n^2
    billed): no residual gain, prefer matmul forms or in-place adds; (iii) fnp.reshape /
    fnp.copyto / fnp.swapaxes are logged ops (reshape and copyto billed 1 flop/elt),
    x[None] / x[:, None] / slices are free views; (iv) copyto(buf, 0.0) is a metered
    page touch (backend time), used at pool allocation.
  * STRASSEN LADDER ON V27 (suite harness, C/B | steady residual): L3 0.3107 | 0.170 s;
    L4 (min block 64) 0.2973 | 0.187; L5 (min block 32) 0.2872 | 0.205; the newborn
    transport at L4 costs +32 ms for -1.9% (poor value, left dense). MSE unchanged to
    3 digits at every level (f32 error grows ~1.4x per level, far below the raw floor).
    The deeper levels' per-op residual grows with the number of sub-blocks (an add on
    8x2401 blocks of 32^2 costs ~0.07-0.28 ms vs 0.03 on 8x1 blocks of 512^2).
  * V28 = SHARED-BASIS STRASSEN: the tier-1 / tier-2 factor stacks live interleaved as
    (L-1, 2, r, n) buffers [slot, {A, P}] with one-time (2(L-1), 1, r, n) views, so the
    formings Qc @ [FA | FP] are ONE plain family per tier and the contractions
    sum_k LA_k FA_k^T + LP_k FP_k^T ONE hub family per tier (the hub kernel takes a
    rectangular right operand); levels capped per family by _Strassen.level() (largest
    level whose block sides divide and stay >= STRASSEN_MIN: r=384 -> L3, r2=224 -> L2 at
    min 32). L5 main + shared basis: C/B 0.2872 -> 0.2680 (-6.7%), residual +38 ms.
  * MEMORY ENGINEERING (peak working set, mem_v28.py): 6.9 GB at first -> 3.4 GB shipped.
    (a) pool keys carried the recursion's level counter, so identical block shapes reached
    from families with different top levels were pooled twice (-1 GB: key by block
    geometry); (b) the products of a level reuse its left-combo buffer when the callee
    recurses (L is dead once the next level's combos exist); (c) FUSED PER-PRODUCT LEAF:
    the leaf level held all 7 products of every block (3 x 525 MB + a 328 MB hub
    temporary at L5) -- forming one combo pair at a time and accumulating straight into
    the output quadrants keeps 7 matmul writes + 8 adds (same billing, the 4 raw-quadrant
    copies dropped: strided matmul inputs are free) with 7x smaller leaf buffers; applied
    only to deep leaves (>= 343 blocks; on every leaf it cost +40 ms of residual, on the
    deep ones +18 ms); (d) Strassen pools size to the batch actually seen (the suite's
    young-source batch <= 2 (AGE_OLD + 1) = 10, a third of the 2 (L-1) capacity).
  * FIRST-MLP EFFECT: with persistent pools the first predict() of a process pays the
    pool allocation/first-touch: +47/+88/+168 ms at L3/L4/L5 (0.41 s at L5 = over the
    cap), and a process-per-MLP grader would pay it on EVERY MLP. Metered page touches at
    allocation recover ~65 ms; the rest is the process growing to several GB during that
    predict (does not reproduce in isolation, probe_first_v28.py). FIX: the first predict
    of a process runs the main families one level shallower (STRASSEN_FIRST = 4; its
    block shapes are shared with the L5 run except the deepest), so no MLP pays the full
    first-time cost: first MLP 0.33 s at C/B 0.280, second 0.31 s, steady 0.26 s at
    0.2673. A setup()-time warm-up at full size was rejected (5 s setup cap = whole
    submission).
  * SHIP (V28 defaults: L5 / min block 32 / shared basis / staged first predict / fused
    deep leaves): suite harness C/B 0.2673, steady residual 0.26 s, wall 36 s/MLP local,
    peak 3.4 GB; expected score 2.27e-8 x 0.2673 ~ 6.1e-9 (V26 live 7.37e-9). Remaining
    Strassen headroom (not built): the joiner's (n,n)@(n,r) products (27 u, 7 per join;
    batched as 3-4 families ~0.07 u/ms), C_pre as two block products (-4.5 u), the
    newborn transport (-6 u for +32 ms), inner @ Qc^T (-1.5 u).
  * MINI (100 MLPs, whest run, in-process, residual cap enforced): 99/100, mean multiplier
    0.2748 (compute 0.2675), mean residual 0.306 s (V26 L1's mini mean was 0.325 s and
    graded 100/100); the one failure is again suite position 76 (0.718 s; V26's mini failed
    the same position at 0.616 s) -- in isolation MLP 76 runs at 0.33-0.36 s while paying
    the first-predict costs and its neighbours at 0.26-0.27 s steady (test_mlps_resid_v28.py,
    runs/isol_v28_76.log): a local, position-dependent hiccup of the retaining harness, not
    the MLP. Wall 36 s/MLP local (V26 L1: 21 s local -> 31-36 s graded; cap 120 s).
    Packaged submission_v28.tar.gz (snapshot DATA submissions/v28_deepstrassen/).
  * GRADED #329991 (2026-09-06 01:40 UTC): LB 5.70e-9 adjusted, raw 2.13e-8, mult 0.2676 =
    the steady local C/B (0 failed; the staged first MLP is invisible in the mean), rank
    11 -> 10/114 (#9 5.40e-9, #7 5.00e-9, #1 Puffi 3.00e-9). -22.7% vs V26 at identical raw.

- F81 (2026-09-06) -- THE LAST DENSE (n,n) PRODUCTS RIDE THE TRANSPORT FAMILY (V29):
  the newborn transport and the C_pre sandwich were the only n^3 work still billed at the
  full count after V28; both now come out Strassen-priced for ~+15 ms of residual, C/B
  0.2680 -> 0.2526 steady (-5.7%) at MSE parity. Generator scratch/make_v29.py (asserted
  patch of estimator_v28.py), probe DATA scratch/probe_cpre_v29.py (runs/probe_cpre_v29
  numbers below), harness test_v29_gc.py (runs/v29_gc.log), ledger audit_v29.py
  (runs/audit_v29_d0.log), mini runs/mini_v29.log, snapshot DATA submissions/v29_ridefamily/.
  * PRICING PROBE (n = 1024, flopscope 0.12.1): the tagged sandwich 'ij,ia,jb->ab' bills
    1.500 u at 0.05 ms residual; two dense matmuls 2.0 u; ONE lone Strassen product bills
    0.683 / 0.609 / 0.556 u at L3 / L4 / L5 for 1.24 / 1.54 / 1.96 ms of residual (f32 err
    3-6e-6). Adding a Y slot to an existing BROADCAST family (X = one (n,n) operand) costs
    +0.4 ms; making X per-slot instead costs +0.06-0.14 u per slot (the left combos stop
    being shared) -> keep X broadcast, fold products by choosing the RIGHT operand.
  * THE FOLD: the young legs' transport is WD @ A with WD = W diag(w1_prev). Pre-scaling
    the dense young legs by w1 (rows, one batched multiply on the interleaved view at the
    wick stage, 2k n^2 flops) makes the family's left operand the raw W, and then (a) the
    newborn's A leg a_b = w1 * C_off is born ALREADY scaled, so it sits in its stack slot at
    birth and the family transports it (W a_b, was 15 x 1 u dense + the 0.5 u last-layer
    C @ w32), and (b) the post-ReLU covariance C copied into the P position of the same
    slot comes out as W C, i.e. half of C_pre = W C W^T (the newborn's P leg W is copied
    over it once C_pre is formed). The joiner reads the scaled legs directly (its own w1c
    multiplies dropped). W (w1 A) vs (W w1) A differ only by rounding: MSE parity.
  * BLOCK-SYMMETRIC C_PRE: (W C) W^T is symmetric, so its 2x2 partition needs only the
    11, 12, 22 block products (h,n)@(n,h) = 0.75 of the dense count, run as ONE Strassen
    family of batch 3 (L4 at h = 512: 3 x 0.25 x 0.609 = 0.46 u), the 21 block copied as
    12^T. Per layer C_pre: 1.5 u -> ~0.56 (W C in the family) + 0.46 = 1.02 u. Layer 0 has
    C = I, so C_pre = w32^T w32 is the aliased 2-operand Gram (exact, 0.5 u; F77's one
    legitimate alias form). The trimmed last layer reads var = rowsum((W C) * W).
    NOT tagging C_pre symmetric costs nothing downstream (no tagged op consumes it).
  * LEDGER (dump 0, staged first predict L4): 286.4 -> 273.5 u; sandwich einsum 22.5 u gone
    (einsum total 31.1 -> 9.1 u), the 16 one-unit matmuls gone, the block family
    (3,343,32,64)@(3,343,64,32) 98 calls = 6.11 u, matmul total 241.9 -> 249.4 u (absorbs
    the folded products at Strassen prices). Steady (second MLP on): 0.2680 -> 0.2526.
  * HARNESS (test_v29_gc.py, 6 MLPs in one process, gc disabled): C/B 0.2671 first (staged
    L4) then 0.2540, 0.2526 x4; mse 2.2910 / 2.0856 / 2.6089 / 2.4998 / 2.2041 / 1.9624e-8
    (V28: 2.2897 / 2.0837 / 2.6091 / 2.4979 on dumps 0-3 = parity); residual 0.333 s first,
    0.319 s SECOND (the L5 pools are first touched on the second MLP, since the first
    predict is staged at L4: the F80 first-MLP trap now lands on MLP 2, still under the
    cap), then 0.254-0.261 s steady (V28 0.243-0.26 s: +~15 ms). Ops 11.4k -> 13.0k.
  * MEMORY (mem_v29.py, two predicts): peak 3.7 GB after the staged first predict, 5.5 GB
    after the second (level-5 pools + the 2 extra slots). F80's "peak 3.4 GB" for V28 was
    a ONE-predict reading (= the staged L4 pools); the V28 two-predict reference is
    runs/mem_v28_2.log. Grader limit 8 GB (advisory in --runner local).
  * MINI (100 MLPs, whest run --runner local): 99/100, mean compute utilization 0.2527,
    mean multiplier 0.2602 (the failure counts 1.0), total residual 29.4 s = 0.294 s/MLP
    (V28 mini 0.306 s). The failure is suite position 76 at 0.720 s AGAIN (V26 0.616 s,
    V28 0.718 s at the same position): the known local position-dependent hiccup, see
    runs/isol_v29_76.log for the isolation timing. Expected LB: 2.13e-8 x 0.2526 =
    5.38e-9 (V28 live 5.70e-9; #8 line 5.40e-9). Isolation (runs/isol_v29_76.log): MLP 76
    0.330 / 0.343 s as the first / second predict of a process, neighbours 0.253-0.264 s.
    V28 two-predict peak 4.7 GB (runs/mem_v28_2.log) vs V29 5.5 GB. SUBMITTED #330093.
  * GRADED #330093 (2026-09-06 20:40 UTC): LB 5.40e-9 adjusted, raw 2.13e-8 (= V28), 0 failed,
    mult = the steady local C/B 0.2526 (the 5.5 GB peak and the second-MLP pool touch are
    invisible on the grader, as for V28) = NEW LIVE BEST, rank 10 -> 8/119 in a three-way
    tie at 5.40e-9 (Activation-Regions, fklassen); #7 5.00e-9, #6 4.10e-9, #1 tie 3.00e-9.
    The Strassen line delivered 7.80e-9 (V25) -> 5.40e-9 in two days at identical raw.
  * WHAT IS LEFT DENSE (audit): joiner products (n,n)@(n,384) 51 calls 19.1 u +
    (384,n)@(n,n) 20 calls 7.5 u + inner @ Qc^T 10 calls 3.75 u: their 384 side caps Strassen
    at L3 (0.68), lone products cost 1.24 ms each and they cannot join a broadcast family
    (different left operands / widths): ~0.1 u per ms of residual -> not worth it. Deeper
    levels (L6) are dead by op count (the deep-level adds already dominate the residual).
    The Strassen line of this chain ends here at ~0.25xB.
  * FAIR ACCOUNTING: same class as the Strassen email of 09-05 (exact algorithms that
    compute the same quantities with fewer multiplications: Strassen recursion, computing
    only the upper triangle of a symmetric product, the Gram at the identity input); no
    declaration or tag trick, nothing computed outside flopscope.
  * WRITE-UP: cost engineering summary for the paper's "compute accounting" section: the
    K3 chain's n^3 work is 100% Strassen-priced now; the residual clock (op count), not
    the FLOP meter, is what bounds the recursion depth on the grader.

- F82 (2026-09-07) -- PUFFI'S A716 CLOSURE IS DEAD AT THE P2 SHAPE, and it lands exactly on
  F65's slice-state floor: its best form (MEMORYLESS merged (2,1) slice + paired D4, ~10
  units/layer = 0.156xB) gives raw 9.8e-7 on the 8 dumps (score 1.5e-7; V29 5.4e-9), the
  fitted 144-coefficient correction buys 1.07x (test 9.15e-7), the 8-generation history is
  WORSE than the merged state (1.24e-6), and the pre-registered kill line (raw 1e-7) is
  missed 9x. Reconstruction: DATA scratch/a716_probe.py (lean torch f64, eqs A19-A28 of
  Puffi's P1 score write-up, archived DATA scratch/writeups/puffi_score.txt); logs
  runs/a716_ladder.log, runs/a716_variants.log.
  * WHAT WAS BUILT: state m, full C, marginal kappa3/kappa4, paired (2,2) kappa4 matrix D4,
    up to 8 generations of the (2,1) slice P. Linear step: Lam = W C W^T; exact
    slice-supported transport of every carried object (P: the (i,i,j) arrangement
    (W o W) P W^T plus the two other arrangements 2 ((W o (W P^T)) W^T); the kappa3-diagonal
    term (W o W) diag(k3) W^T; D4: (W o W) D4 (W o W)^T + (W o W) diag(k4) (W o W)^T; the
    marginal d3/d4 read exactly from the same products). ReLU step: Gaussian response
    tables F_{k,q} = d^q M_k / d m^q in closed form (k!/(k-q)! M_{k-q} for q <= k,
    k! (-1)^(j-1) He_{j-1}(r) phi(r) / sigma^j beyond, j = q-k), marginal Edgeworth
    corrections clipped at 25% sigma^k (A23/A24), and the 9-term pair series (A25/A26)
    evaluated with the (1,1), (2c,1) and (2c,2c) response tables to produce C_off, the
    newborn and aged (2,1) slices and D4 (centered squared response f_a = (Y_a - m_a)^2,
    F^f = F2 - 2 m F1). Fit: s_L = Phi(r_L) o (W_L s_{L-1}) + H_L beta_L (A27/A28) never
    feeds back into the state, so the final s is LINEAR in all beta -> exact joint ridge
    on the final residual (9 features x 16 layers = 144 constants; Puffi's fifth channel is
    zero in their own estimator), train dumps 0-4, test 5-7.
  * SANITY: the Gaussian-only stage = cov-prop 4.06/4.72e-6 on dumps 0/1 (F78's K=1
    baseline 4.78e-6); marginal skew/kurtosis alone leaves the final layer unchanged
    (4.55e-6; layer 1 improves 4x) = F63's "skew worthless on the D3->mean path".
  * LADDER (8 dumps, mean final raw mse; unit = one n^3 matmul, B = 1024 u): (2,1)
    history only, 8 gens 1.52e-6 @0.55xB | D4 only 4.12e-6 @0.08 | full 8 gens 1.24e-6
    @0.59 | full 16 gens (no truncation) 9.87e-7 @1.03 | full MERGED (memoryless: the
    transport is linear and the ReLU response scaling is the same for every generation, so
    the sum of generations is one matrix) 9.80e-7 @0.156xB | thin transport ((i,i,j)
    arrangement only, 8 gens) 1.29e-6 @0.34 | literal Gaussian order 2 = order 4 (1.241 vs
    1.243e-6) | alpha_p 0 1.238e-6 | alpha_p 1 + no clip 1.38e-6 | no kappa3-diagonal
    term 1.55e-6. By layer (8 gens): 6.6e-9 at layer 2, 5.5e-8 at 3, 2.2e-7 at 5, 5.4e-7
    at 8, 1.05e-6 at 13 -- the depth law of F65's collapsed slice state.
  * FIT: ridge on the final residual gives test 9.15-9.9e-7 for EVERY variant (train
    8.6-9.2e-7), insensitive to lambda over 1e-3..1e-1: the 144 constants recover 7%
    (merged) to 25% (8 gens). Same ceiling class as F47's online correction and F65's
    "fitting adds nothing": the missing content is the transported fully off-diagonal K3,
    which no function of the live state predicts.
  * READING: A716 is F65's keep=0 slice closure rewritten as a low-order response series
    (9.8e-7 here vs 9.1e-7 with the exact Wick term table); Puffi's own figure only claims
    a 2x win over Wu K=3 through layer 9 at n/L = 8, where the K3 truncation is already
    asymptotic (F29). At 1024 x 16 it is 23x behind the exact K3 chain in raw (4.3e-8 at
    0.5xB) and 28x behind V29 in score (17x even at the 0.1 floor). The H2 cost estimate
    (docs/research/option2_intel_2026-09-05.md, "3-4 u/layer") was wrong: the merged
    closure needs ~10 u/layer (Lam 2 + slice-P 4 + kappa3-diag 1 + D4 3) = 0.156xB before
    Strassen/symmetric pricing; and the 8-generation window is a truncation artifact
    (worse than merged), not a memory mechanism.
  * DECISION: Puffi's P1 lineage is exhausted as a source of the 0.15-0.20 tier (SSC:
    F78 intel, capped atoms: F76, A716: here). Every slice-state closure measured at this
    shape sits at ~1e-6 (F65 9.1e-7, A716 9.8e-7, two independent ReLU treatments). Do not
    prototype another slice-only or fitted-low-mode closure. The next expansion must carry
    fully off-diagonal K3 information below 4 units per source-layer, or the climb continues
    on the V29 chain's raw error.
  * WRITE-UP: the response-table closed form and the "merged = all generations" linearity
    argument are reusable; the A716-vs-F65 coincidence is a second, independent measurement
    of the slice-state floor (the D3/D21 interface theorem of F65 predicted it).

- F83 (2026-09-07) -- LITERATURE TRIAGE #3, bucket 1 "finite-width corrections beyond the GP
  limit" (full reads of 3 PDFs in docs/papers/, notes in docs/papers/<id>-notes.md): TWO
  DISCARDED, ONE CITE-ONLY; NO experiment warranted. One structural fact kills the whole
  bucket at once: every paper averages over the WEIGHTS at a fixed finite set of inputs
  (annealed / NNGP-correction setting). In that ensemble the output is conditionally
  Gaussian given earlier layers, neurons are exchangeable, and ALL ODD CUMULANTS VANISH
  IDENTICALLY -- the third cumulant our whole chain is built on (F43/F63/F65) is exactly
  the object this literature sets to zero. Our problem is the quenched one (ONE fixed W,
  X ~ N(0,I), per-neuron, fully off-diagonal over neurons); nothing recurses, prices or
  measures that. Reading only; no code run.
  * 2508.11522 (Finite-width NTKs from Feynman diagrams, 2025; 53 pp.): DISCARD as method,
    CITE-ONLY. 1/n corrections to the weight-ensemble NNGP/NTK at <= 4 fixed inputs (NTK
    parametrization, App. B eq. 20). Under exchangeability every cumulant collapses to
    Kronecker deltas times a 4x4x4x4 INPUT-space tensor (eq. 23, pp. 12-13); odd cumulants
    vanish ("number of external lines is necessarily even", App. E.4 p. 24). The one
    "connected four-point" recursion (eq. 45, p. 16) is V^(l+1) = S(K^l) + T(K^l) V^l T(K^l)^T
    with S = single-neuron activation 4-pt minus its Gaussian factorization and T = the
    second-derivative response kernel <d^2 (sigma sigma - <sigma sigma>)/dz^2>_K: memoryless,
    born-then-transported = the same skeleton as our K3 sources, but no n-dependent tensor,
    no cost identity, and no lambda_l * C_off form (F68 stays ours). Depth law at
    criticality: susceptibility chi = 1 exactly (Thm 5.1, eqs. 79/82, App. H), so
    transported cumulants ACCUMULATE ADDITIVELY, V ~ l (measured exponents 1.19-1.55 at
    width 300 to depth 30, Fig. 12 p. 42); rank-k tensor O(n^{-(k-2)/2}) (p. 31), same law
    as Fischer 2022 App. B (F70). Evidence: 4-layer GeLU, widths 5-220, relative kernel
    deviations 1e-2..1e-1 (Figs. 1, 3-5, 22-25), ReLU noise floor ~4e-3: fails the F70
    1e-6 rule by 3+ orders. Only reusable identity is Price's theorem (eq. 106), already
    in kprop's Hermite/Wick tables (F20).
  * 2605.24072 (Optimal non-asymptotic Edgeworth expansions for multivariate NN outputs,
    2026; 34 pp.): CITE-ONLY (marginal). Annealed over Gaussian weights; conditionally
    Gaussian given earlier layers (Lemma 1, p. 4), so kappa3 = 0 and the series is driven
    by moments of the empirical-kernel fluctuation Q = K^{-1/2}(A-K)K^{-1/2} (eq. 2.7), with
    width bounds imported from Hanin 2024 (Remark 12, eq. B.1, p. 22). Bound: d_TV to the
    order-(4m-1) Edgeworth <= C E[||A-K||_HS^{4m}]^{1/2} (Thm 1, eq. 3.1), i.e. C1/n^m with a
    matching lower bound (Thm 3, p. 10) -- "optimal" = TV rate in WIDTH at FIXED depth; the
    joint (n, L) regime is explicitly deferred (Remark 8, p. 11), so nothing of the
    (L/n)^K form for F2/F29. No kappa3/kappa4 layer recursion; no corrected E[relu(z)]; no
    validity/clip rule (F38/F82 negatives stand). Evidence: pointwise density error
    1e-3..5e-2 vs an MC KDE, one hidden layer, scalar input (Fig. 2). Lemma 2 (eqs.
    4.8-4.9, p. 14) is the mismatched-covariance Wick identity already in kprop (F20).
    Notation trap: in Thm 1 n = number of output neurons, width is n_l.
  * 2608.20403 (Criticality in function space through Wilsonian fixed points, 2026; 30 pp.):
    DISCARD, not even cite-only. A position paper: the RG scale l is never defined for a
    network (eqs. 10-11, p. 9; admitted in Limitations p. 27), depth is a control
    parameter not the flow variable, beta(g) = yg - ag^2 (eq. 35, p. 22) is a drawn cartoon,
    Figs. 7-8 are schematics. Only model simulated: one-hidden-layer random-feature ReLU
    net, widths 20-2500, 24 inputs; kernel error N^{-1/2} at 1e-1..3e-2 (Fig. 1, p. 14),
    Q_4 = ||G_c^(4)||/||G_Wick^(4)|| ~ 1/N (Fig. 3, p. 17). Only finite-width formula:
    kappa4[f_N] = kappa4[z]/N for f_N = N^{-1/2} sum z_j (eqs. 24-25, p. 16). ReLU
    criticality chi = sigma_w^2/2, critical sigma_w = sqrt 2 (pp. 19-20, eqs. 29-31) is
    verbatim Poole 2016 / Schoenholz 2017. For a "finite width = interacting field theory"
    framing cite the primary source Halverson-Maiti-Stoner 2021 (arXiv:2008.08601).
  * WRITE-UP (three citable sentences, all demarcation, none a method): (a) the
    finite-width NNGP/NTK/Edgeworth literature is annealed over weights, where odd
    cumulants vanish; the quenched per-neuron third cumulant our estimator propagates is
    its exact complement (2508.11522 App. E.4; 2605.24072 Lemma 1). (b) At the He-critical
    point transported cumulants do not fade with depth (chi = 1, additive accumulation,
    2508.11522 Thm 5.1) -- prior art in spirit, different randomness, for why old K3
    sources cannot be dropped (F65/F66) and for the (L/n)^K depth law (F2/F29). (c) Finite-
    width Edgeworth series are asymptotic: the order-2 (H8) term is far worse than the
    Gaussian limit at width 20 and still hurts at 500 (2605.24072 Remark 9 / Fig. 2),
    an independent annealed-setting confirmation of F29's K* = 3 wall as a general
    phenomenon rather than our artifact.
  * F70 RULE SHARPENED for future triage: before reading past the abstract, check WHICH
    randomness the cumulants are over. Anything annealed over weights (NNGP/NTK 1/n
    corrections, "finite-width corrections", Edgeworth over inits, field-theory/RG framings)
    has kappa3 = 0 by construction and cannot inform a quenched per-neuron K3 chain; only
    the quenched literature (2605.05179 and the P1 write-ups, F48/F68) carries order >= 3
    through depth. Pointers logged but not read: Hanin 2024 (annealed additive-in-depth
    1/n kernel-variance hierarchy), Mansanarez-Poly-Swan 2510.14002 (Edgeworth on Wiener
    chaos, possible language for the F21/F30 chaos spectrum).

- F84 (2026-09-07) -- LITERATURE TRIAGE #3, bucket 2 "moment propagation through
  piecewise-linear networks with Gaussian input: did anyone go past covariance?" (full
  reads of 4 PDFs, notes in docs/papers/<id>-notes.md): ANSWER NO. ALL FOUR DISCARDED;
  NO experiment warranted. All four are in the RIGHT randomness (one fixed
  net, random input = our quenched setting), which is why they were worth reading after
  F83; none of them reaches ORDER 3. Two are exact-but-intractable representations of the
  pushforward (activation-pattern mixtures) with no moments at all; two are the
  uncertainty-quantification lineage F70 already closed, at or below Gaussian closure.
  The off-diagonal third cumulant among distinct neurons does not appear in any of them;
  the only quenched literature carrying order >= 3 through depth remains 2605.05179 and
  the P1 write-ups (F48/F68). Reading only; no code run.
  * 2503.22082 (ReLU networks as random functions, 2025; 22 pp.): DISCARD. Characterizes
    the EXACT discrete law over activation patterns and the resulting mixture of
    truncated Gaussians: conditional on the stacked mask zeta the net is affine,
    h_l = C_zeta x + d_zeta with C_l = W_l diag(z_{l-1}) C_{l-1} (eqs. 4-5, p. 4);
    P(zeta) is a Gaussian orthant probability in dimension sum_l n_l (Prop. 1, eq. 11,
    p. 6) = 15360 dims / 2^15360 patterns at our shape; every h_l is a mixture of
    truncated Gaussians, one per pattern (Prop. 2, eqs. 14-17, pp. 7-8). No E[relu],
    covariance, arc-cosine kernel, Hermite series or cross-cumulant anywhere; the only
    moment-type quantity is P(h > 0) = sum_k alpha_k Phi(mu/sigma) (eq. 21, p. 11), used
    to prune "low-entropy" neurons in a branch-and-bound support search that caps at 10
    branching neurons (p. 15) -- dead on He-random nets where every neuron sits at
    P ~ 0.5. Evidence: moons/MNIST, L=4, widths 4-32, tail probabilities to 2 digits.
    OUR READING (not the paper's): conditional on the pattern the map is affine, so all
    quenched cumulants of order >= 3 are MIXTURE cumulants induced by the mask
    randomness; kprop's Hermite treatment of the Heaviside factor is already the
    tractable form of that fact. One demarcation sentence for the write-up (Prop. 2:
    intermediate representations are mixtures of truncated Gaussians, not Gaussian).
  * 2502.11672 (Exact upper/lower bounds for the output distribution with random inputs,
    2025; 25 pp.): DISCARD. Output CDF of a net with INPUT DIMENSION 1-3, by enumerating
    ReLU activation polytopes, triangulating them and integrating a compactly supported
    piecewise-polynomial input pdf exactly on simplices (Thm 3.8, pp. 3-4, App. A.1);
    a Gaussian input must be truncated and sandwiched by piecewise constants (Thm 3.16,
    p. 7); region count O((max n_l)^(n_0 L)) (p. 3), triangulation capped at 50,000
    vertices (p. 7). No expectation, moment, cumulant or kernel identity of any kind.
    Evidence: Banana/Diabetes/Iris/Wine, widths 8-32, CDF bound gaps 1e-4 (n_0=1) to
    0.2 (n_0=3), Table 1 p. 9. Same 1e-2 verification/UQ family (Krapf 2024, Zhang-Shin
    2021, Weng 2019) the F70 takeaway dismissed.
  * 2503.21059 (Uncertainty propagation in feed-forward NN models, 2025; 30 pp.): DISCARD.
    First-order Taylor (delta method) of a fixed leaky-ReLU net around its MEAN input,
    z iid uniform (eqs. 4-5, pp. 3-4): g ~ m + J_L z with J_L the Jacobian at the mean
    (eqs. 31, 35-36, pp. 7-8), so E[relu(mu + s xi)] ~ relu(mu) and Cov ~ phi'(mu_i)
    phi'(mu_j) C_ij. One rung BELOW mean propagation, and UNDEFINED at our shape: zero
    biases put every linearization point exactly on the kink (phi'(0) undefined,
    predicted mean 0 instead of s/sqrt(2 pi)). Proven error bound is EXPONENTIAL in L
    (B.26-B.28, p. 26) while their Fig. 10 claims error decreasing with L on trained
    contracting nets. Joint law = Gaussian copula on linearized correlations (Sec. 6,
    pp. 9-11); no third cumulant. Evidence: width 64, L=1/5/20, output correlations to 2
    decimals vs 1e5 MC (Table 2, p. 14). The only exact identity, relu(h + Wz) =
    phi'(h) o (h + Wz) + Delta with Delta supported on the sign-flip set (B.4, B.13-14,
    pp. 20-22), is the trivial form of the H5/F13 kink-noise mechanism. Tangential
    pointer, not read: Venturi & Li 2023, Mori-Zwanzig formulation of deep learning
    (memory-kernel vocabulary for the F65/F66 "old K3 sources are history" picture).
  * 2601.16830 (Uncertainty propagation through trained MLPs: exact analytical results,
    2026; 12 pp.): DISCARD. ONE hidden layer, fixed trained weights, Gaussian input:
    closed-form mean and covariance of a rectified bivariate Gaussian. Thm 1 (p. 4) =
    Frey-Hinton E[relu z] = sigma phi(mu/sigma) + mu Phi(mu/sigma) and the (2,) power
    moment; Thm 2 (p. 4, App. A via Kan-Robotti 2017 truncated-MVN moments), with
    omega_ij = (mu_i sigma_j - rho mu_j sigma_i)/(sigma_j sqrt(1-rho^2)):
      E[relu z_i relu z_j] = mu_j sigma_i phi(mu_i/sigma_i) Phi(omega_ji)
        + mu_i sigma_j phi(mu_j/sigma_j) Phi(omega_ij)
        + sigma_i sigma_j (1-rho^2) phi_2(mu_i/sigma_i, mu_j/sigma_j; rho)
        + (mu_i mu_j + rho sigma_i sigma_j) Phi_2(mu_i/sigma_i, mu_j/sigma_j; rho),
    the general-mean completion of the arc-cosine kernel (Bibi 2018 Lemma 3 at mu = 0;
    what Wright 2024's Mehler series approximates and cov-prop computes). Multi-layer
    explicitly left open (Conclusions, p. 6); no cost, no n/L statements. Evidence: one
    trained 12-unit 2-input net, 40 test points, RMSE vs MC ~7e-5 at 1e6 samples
    (Table 1, p. 6) = a correctness check of a one-layer identity. Nothing beyond the
    (1,)/(2,) power-cumulant step kprop already has (F20); the ReLU-moment step is ~0%
    of our bill (F56) and not on the error path (F60).
  * TAKEAWAY for the write-up: the quenched literature splits into (i) exact
    pattern-mixture representations (2503.22082, 2502.11672, Krapf 2024) that are
    exponential in width x depth or input dimension and never take moments, and (ii)
    moment propagation (F70 lineage + 2503.21059 + 2601.16830) that stops at order 2 or
    below. The gap between them -- a tractable order-3 (and regenerated order-4)
    propagation of the per-neuron, fully off-diagonal cumulants of ONE fixed net through
    16 layers -- is occupied only by 2605.05179 and the WhestBench entries; that is the
    positioning sentence. F83's triage rule holds (check the randomness first); ADD: in
    the quenched setting, check next whether the paper ever writes a cross-moment of
    order >= 3 among distinct neurons -- if not, it cannot touch F65/F66/F68.

- F85 (2026-09-07) -- LITERATURE TRIAGE #3, last group (mixed lenses: cumulant-tensor cost,
  the annealed 4-pt primary source, and two annealed convergence-rate papers; full reads of
  the two short PDFs, main text + result appendices of the two long ones, proofs skimmed;
  notes in docs/papers/<id>-notes.md): TWO DISCARDED, TWO CITE-ONLY; NO experiment
  warranted. Closes triage #3 (F83 bucket 1, F84 bucket 2, F85 here): 11 PDFs, zero
  methods, one demarcation paragraph and a handful of primary-source citations for the
  write-up. Reading only; no code run.
  * 1701.05420 (Domino-Gawron-Pawela 2018, efficient computation of higher-order cumulant
    tensors, Cumulants.jl; 22 pp.): DISCARD. A SAMPLE-based cumulant estimator (t samples
    of n-dim data -> order-d tensor) with block-symmetric storage: b^d binom(n/b+d-1, d)
    ~ n^d/d! numbers (pp. 6, 12), (d-1) t multiplications per element (p. 13), cumulant
    correction eq. 34 (p. 10) over singleton-free partitions (N(4)=3, N(5)=10, N(6)=55,
    eq. 41); headline speedups 72x (order 4) / 900x (order 5) vs naive (p. 13). There is
    NO multilinear operation anywhere (Algorithm 1 fills unique blocks elementwise); the
    symmetric-tensor-times-matrix transport is its cited Schatz et al. SISC 2014 [49],
    a constant-factor (<= d!) saving on a DENSE n^4 op = 3072 n^3 units at n=1024 (~512
    with a full 3! saving) vs ~30 units/layer for our whole factorized chain and 4 units
    per source-layer for the off-diagonal part (F66/F67): cannot beat FACTORIZED cost.
    The partition calculus is the one kprop's diagram summation already uses cumulant-
    to-cumulant (F20); the (2,1)/(3,) power-cumulant slices are an object the paper does
    not even index. Its "compute only sorted-index blocks" mechanism IS the F5 principle
    for a symmetric saving under flopscope's executed-op billing, but our executed ops
    are non-symmetric matmuls on leg matrices (F67), so it does not apply. Evidence:
    timings only, n=30-70, t=1e5-5e5, orders 4-5 (Figs. 3, 5, 6). Nothing to cite.
  * 1910.00019 (Yaida 2019, non-Gaussian processes and NNs at finite widths; 22 pp.):
    CITE-ONLY, the PRIMARY SOURCE for the annealed recursion 2508.11522 re-derived (F83).
    Annealed over iid Gaussian weights and biases at fixed inputs (eqs. 1-4, p. 2); odd
    cumulants vanish identically (footnote 3, p. 3); exchangeable neurons put the 4-pt
    vertex in INPUT space with Kronecker deltas in neuron space (eq. V, p. 3). Recursion
    (Sec. 3.2, p. 5; T_l = (1/2) C_W <sigma sigma (zz - K)>_K):
      R1  K_{l+1} = C_b + C_W <s1 s2>_K                          (NNGP)
      R2  V_{l+1} = C_W^2 [<s1 s2 s3 s4>_K - <s1 s2>_K <s3 s4>_K] + (n_l/n_{l-1}) T_l V_l T_l
      R3  self-energy S_{l+1} from transported S and a 4th-derivative response on V.
    Memoryless, born-then-transported; the source is ONE neuron's connected activation
    4-pt function from the single repeated index (App. B, S26); four-distinct-neuron
    content is zero at O(1/n); V is a genuine input-space tensor per layer, not scalar
    times kernel (F68's lambda_l * C_off has no analogue; annealed growth is linear in l
    while our fitted lambda_l peaks at layers 4-5). CITATION-GRADE: hierarchy S12
    (p. 13) connected 2m-point = O(n^{1-m}); deep linear eq. 16, V_l/n = (sum_{l'<l}
    1/n_{l'}) [KK + KK], linear in depth; ReLU single input at C_W = 2 (eq. 17, S40-S45,
    p. 20): K layer-independent, V_{l+1}/K^2 = 5 + V_l/K^2 => annealed excess kurtosis
    15 (l-1)/n, the 5 being <relu^4>/<relu^2>^2 - 1; and the sentence p. 7 "finite-width
    corrections stay perturbative typically when depth/width << 1" = the primary
    statement of L/n as the expansion parameter behind F2/F29. Criticality (chi = 1) is
    deferred to future work (Sec. 6, p. 9): cite 2508.11522 Thm 5.1 for that only.
    Repeated-index handling (App. B, S21-S22: at O(1/n) only zero or one neuron
    repetition contributes) is the annealed mirror of F68's "diagonal kappa4 stays
    exact, off-diagonal is regenerable" -- a mirror, not the same statement. Evidence:
    one MNIST input, L=2-3, widths 10-100, output density plots (Fig. 1, S1).
  * 2205.08187 (Lee-Ayed-Jung-Lee-Yang-Caron 2022, DNNs with dependent weights: GP
    mixture limit, heavy tails, sparsity; 96 pp.; read fully pp. 1-52, 73-89, skimmed
    53-72 proofs): DISCARD, not citation-grade. Annealed over per-node scale-mixture
    weights, variances and biases at fixed inputs (eqs. 8-10, p. 11; char. function
    eq. 79, p. 70); output is a scale mixture of iid Gaussians given the previous layer
    (eq. 28, p. 17), so odd cumulants vanish. Iid Gaussian He init is the degenerate
    corner (lambda = c/p, trivial Levy measure, Sec. 6.1 p. 26) where every result
    switches off by their own Table 1 (p. 8). No finite-width rate at all (Thms 8, 16
    are convergence-in-distribution with sequential per-layer limits, Remark 17); the
    only depth-indexed quantities are heavy-tail-only (log^(l-2) tail factor, Prop. 10;
    geometric pruning constant, Cor. 13). Only order-2 recursions (eq. 23, eq. 40); the
    criticality condition sigma_v^2 C_phi (a + M_1) = 1 with C_phi = 1/2 for ReLU
    (p. 18) = He init, already cited via 2202.04925 App. G (F70). One formula touches
    our objects: the zero-mean order-2 arc-cosine kernel kappa_2(rho) = 3 rho
    sqrt(1-rho^2) + (pi/2 + arcsin rho)(1 + 2 rho^2) for E[relu^2 relu^2] (eqs. 41-42,
    p. 25; App. A.2 eq. 54) = the (2,2) power-cumulant slice kprop evaluates with general
    means via Hermite/Wick (F20, F68's K22); check value at most. Evidence: squared-
    output correlations at widths 100-2000, pruning accuracies, norm box plots (Tables
    3-9, App. F.1). Pointers, not read: Bracale-Favaro-Fortini-Peluchetti 2021
    (2102.10307), Jung et al. 2023 (alpha-stable), Tsuchida et al. 2019.
  * 2601.21539 (Celli 2026, wide NNs with general weights: convergence rate with explicit
    hyper-parameter dependence; 105 pp.; read fully pp. 1-80, skimmed 81-105 Stein-bound
    proofs): CITE-ONLY (marginal). Annealed over iid general weights at fixed inputs;
    conditionally a sum of iid terms (p. 50), odd cumulants vanish, no order >= 3 cross-
    moment among neurons anywhere, nothing quenched. THE RATES: general weights, single
    input, Thm 3 (p. 12) / Cor. 2 eq. 3.2 (p. 14): max(d_K, W_1) = O(sqrt(L/n) x
    2^{R L^2 log_2(...)}), i.e. sqrt(L/n) times 2^{O(L^3)}, convergence only for L <=
    ((1/2 - eps) log_2 n)^{1/3} (eq. 1.1, p. 4), admittedly non-sharp (p. 5, Remark 8);
    multi-input (Thm 4 / Cor. 3, pp. 15-17; ReLU Prop. 6 eq. 7.7 p. 71) still exponential
    in L. Gaussian weights, ReLU, single input, Example 3 (pp. 51-52): kernel-fluctuation
    recursion Q_2^(l) <= 96 C_W^2/n_l (...) + (C_W/2)^2 Q_2^(l-1), i.e. V^(l) <= a/n +
    chi^2 V^(l-1) with chi = C_W/2; at C_b = 0, C_W = 2 (OUR init, zero bias covered via
    eqs. 6.6/6.7 although the main theorems need C_b != 0, Remark 11): d_K, W_1 =
    O(sqrt(L/n)), explicit W_1 <= 16 sqrt(L/n), convergence iff L/n -> 0 (p. 52, last
    display: the sentence to cite). Remark 8 eq. 3.3: sqrt(sum_l 1/n_l) for identity/ReLU
    at critical init; exponent admittedly worse than the optimal 1/n of Favaro-Hanin-
    Marinucci-Nourdin-Peccati 2023 (fixed depth, non-explicit constants). All layer
    recursions are "1/n injected per layer, transported by chi^2", memoryless and
    additive at criticality = the F83 skeleton; scalar variances, no lambda_l * C_off
    analogue. Sec. 4.4 (pp. 25-26) names the primary source for "L/n is the expansion
    parameter": Hanin 2022 (2204.01058, effective depth; log-normal mixture limit as
    L/n -> alpha for 1-homogeneous activations, eq. 4.3) -- cite THAT for the effective-
    depth law, Celli only for explicit constants at the ReLU critical point. Evidence:
    NONE (no numerics).
  * TRIAGE #3 CLOSED (11 PDFs, F83-F85). Net yield for the SCORE: zero methods, zero
    experiments, consistent with F70's takeaway that only 2605.05179 and the P1 write-ups
    carry order >= 3 through depth for a fixed net. Net yield for the WRITE-UP, one
    demarcation paragraph: (i) the finite-width literature (Yaida 2019 -> 2508.11522,
    2605.24072, 2601.21539, 2205.08187, 2608.20403) is annealed over weights, where odd
    cumulants vanish and the O(1/n) object is an input-space 4-pt vertex born from one
    repeated neuron index and transported memorylessly; L/n is its expansion parameter
    (Yaida p. 7; Hanin 2022 effective depth; Celli O(sqrt(L/n)) at the ReLU critical
    point); at criticality it accumulates additively with depth (2508.11522 Thm 5.1).
    (ii) The quenched literature (F84) is either exact pattern mixtures that never take
    moments or moment propagation stopping at order 2. (iii) Our estimator propagates the
    quenched, per-neuron, fully off-diagonal third cumulant of ONE net through 16 layers
    with a regenerated fourth; its (L/n)^K truncation law (F2/F29) shares the parameter
    with (i) but is a different object. Primary-source map: annealed recursion -> Yaida
    2019 R2; O(n^{1-m}) hierarchy -> Yaida S12 / Fischer 2022 App. B; chi = 1 additive
    accumulation -> 2508.11522 Thm 5.1; L/n effective depth -> Hanin 2022; explicit
    sqrt(L/n) at He-ReLU -> Celli Example 3; Edgeworth series asymptotic at moderate
    width -> 2605.24072 Remark 9; intermediate laws are truncated-Gaussian mixtures ->
    2503.22082 Prop. 2; field-theory framing -> Halverson-Maiti-Stoner 2021.
  * TRIAGE RULE, final form (supersedes the F70 takeaway and the F83/F84 additions):
    (1) which randomness? annealed over weights -> kappa3 = 0 by construction -> at most
    a citation; (2) quenched -> is any cross-moment of order >= 3 among distinct neurons
    ever written? no -> cannot touch F65/F66/F68; (3) cost papers -> does the saving act
    on a FACTORIZED representation, or only on a dense n^d tensor we never materialize?
    dense only -> DISCARD (F67); (4) evidence -> moment error vs depth at <= 1e-6 or
    nothing. Unread pointers accumulated across F83-F85 (none expected to pass the rule):
    Hanin 2022 (2204.01058) and Hanin 2024, Favaro et al. 2023, Mansanarez-Poly-Swan
    2510.14002, Venturi & Li 2023, Schatz et al. 2014, Bracale et al. 2021, Krapf 2024.

- F86 (2026-09-07) -- V29 PER-FAMILY UNIT LEDGER: 95% OF THE BILL IS THE K3 SOURCE
  MACHINERY, split 116 u young dense tier / 107 u old shared-basis tier / 25 u thin +
  elementwise; everything else (covariance 7.1 u, closure + birth 5.7 u) is 5%. The
  engineering headroom INSIDE the ledger is ~10 u (3-4%); the 1.7x to the leaders is the
  unit COUNT of the old tier, and the leaders' multiplier equals our bill with the old
  tier removed to three digits. Answers the "better engineering instead of math?"
  question quantitatively. Tooling: DATA starterkit make_v29_ns.py (asserted textual
  patch of estimator_v29.py -> estimator_v29_ns.py, 33 flops.namespace() scopes around
  the semantic families + one per layer; the op stream is unchanged: call-1 C/B 0.2671
  / 273.5 u / mse 2.2910e-8 = the untagged audit_v29_d0.log to every digit),
  audit_v29_ns.py (aggregates op_log by last namespace segment = family and first
  segment = layer; ledger taken on the SECOND predict() of the process because the first
  runs STRASSEN_FIRST = 4 levels), log runs/audit_v29_ns_d0.log. Unit = 2n^3 = 2^31
  FLOPs, B = 1024 u. Dump 0 steady state 260.1 u = 0.2540xB (suite harness 0.2526, F81);
  0.0 u untagged; flopscope counts are shape-based so one dump is the ledger.
  * FAMILIES (units, share): young_transport 60.7 (23.3%) | hub 54.9 (21.1%) | shared
    27.9 (10.7%) | old_legs 27.9 (10.7%) | j_rf 20.8 (8.0%) | fb 11.5 (4.4%) | j_rot 9.1
    (3.5%) | j_proj 7.5 (2.9%) | cpre 7.1 (2.7%) | j_tier2 6.7 (2.6%) | thin_transport
    5.9 (2.3%) | j_qc 3.8 | thin 3.7 | elem 3.3 | nonlin 2.9 | j_core 2.8 | birth 2.5 |
    feed 0.5 | qc_transport 0.4 | pk2k/assemble/wick_scale/regen/wick/dslices/other
    0.3 total. GROUPED: K3 young (young_transport + hub) 115.6 u 44.5% | K3 old (join
    sub-families + old_legs + shared + qc_transport) 106.8 u 41.1% | K3 thin/elementwise
    (thin_transport, elem, thin, fb, feed) 24.8 u 9.5% | covariance 7.1 u 2.7% | closure
    + birth (n^2 families) 5.7 u 2.2%.
  * PER LAYER: 0.6 (L00) | 3.4 | 5.9 | 8.4 | 10.9 (L04, young tier filling: +2.3 u per
    new young source) | 15.2 (L05, first join) | 17.4 | 18.9 | 20.8 | 22.0 | 23.0 | 24.0
    | 25.0 | 26.0 | 27.0 (L14) | 11.8 (L15, mean-only trim). From L05 the young tier is
    FLAT at 4.34 + 4.36 = 8.7 u/layer (4 young sources x 2.17 u: 1.09 transport + 1.09
    hub = Strassen-L5 pricing of the dense 4-unit form of F66) and the old tier grows
    ~1.1 u per extra old source: at L14 (ka = 10 old, kb = 7 in tier 2) join 5.9 +
    shared 4.4 + old_legs 4.0 = 14.3 u, i.e. ~1.1 u per old source-layer after the
    fixed per-join overhead (~3.5 u/layer: range finder 2.08 = four dense
    (1024,1024)x(1024,384) products + one (1024,384) QR, projections FAj/FPj 0.75,
    Qc = W Qn 0.375, core 0.28). SO: the shared basis is a 2x per-source lever, not
    10x, because the F67 wall (elementwise products need dense legs) forces every old
    source to be RE-FORMED (old_legs, Qc FA -> (n,n), 0.35-0.4 u) and RE-CONTRACTED
    (shared, LA/LP against the factor slabs, 0.35-0.4 u) every layer; with 10 old
    sources vs 4 young by L14 the two tiers cost the same. j_rot (9.1 u) is the
    per-layer rotation of all old factors into the new basis (0.28 u per tier-1 slot,
    0.10 per tier-2 slot).
  * ENGINEERING HEADROOM, ranked (all exact, zero accuracy change unless stated):
    (1) the join's dense products are the LAST full-count n^3 work in the chain: j_rf
    4 x 0.375 + j_proj 2 x 0.375 = 2.25 u per join x 10 joins = 22.5 u billed at 1.0;
    lone-product Strassen pricing 0.556-0.683 u/u (F81 probe) -> save 7-10 u = 3-4%
    of C on paper. CORRECTION (same day): these are the "joiner/inner remnants" F81
    ALREADY priced and rejected -- 60 LONE products (6 per join: 4 range-finder + 2
    projections; different left operands / widths, so they cannot join a broadcast
    family), 1.24 / 1.54 / 1.96 ms of residual each at L3 / L4 / L5 for 0.119 / 0.147
    / 0.167 u saved each = ~0.1 u per ms: 7.1 u for +74 ms, 8.8 u for +92 ms, 10 u for
    +118 ms. On V29's residual (steady 0.254-0.261 s, second-predict peak 0.33-0.34 s,
    position-76 hiccup 0.72 s) that lands the peak at ~0.45 s > the 400 ms cap. NOT
    shippable at the current residual; it becomes a lever only if the base residual is
    cut by ~100 ms first (F80-class work). Cross-check with the price-per-unit knob:
    the V28 ladder (STRASSEN_LEVELS comment) is L3 0.3107 | L4 0.2973 | L5 0.2872 xB at
    0.170 / 0.187 / 0.205 s residual, i.e. an "L4 everywhere" safety variant of V29
    costs +3.4% of C for -18 ms of residual. Pricing puts V29 inside a +-4% band.
    (2) j_rot via a composite per-slot rotation instead of rotating the factors: dead by
    arithmetic -- the forming family would need a per-slot left operand, which F81
    priced at +0.06-0.14 u per slot, cancelling the 0.09 u/slot saved.
    (3) AGE_OLD 4 -> 3 (one more source confined): ~1.1 u/layer = ~12 u = 4.6% at the
    F72 ladder's accuracy cost at that gate (not re-measured on V29). A knob, not a lever.
    (4) fb (D21 feedback thin legs, R_FB = 16) 11.5 u = 4.4% buys -10% raw (F69):
    score-positive, keep. cpre 7.1 u and the whole n^2 closure 5.7 u: nothing to take.
  * THE LEADERS' MULTIPLIER IS OUR BILL WITHOUT THE OLD TIER: young 115.6 + thin 24.8
    + cov 7.1 + closure 5.7 = 153.2 u = 0.1496xB vs Puffi 0.149 / suliman 0.152 / J2W
    0.164 (board 2026-09-06 21:06Z, raw 1.97-2.02e-8 vs ours 2.13e-8). Whatever carries
    their old K3 content costs them ~0 units, while ours costs 107 u for a 6x raw gain
    (F65 ladder: keep 4-5 young sources only = 1.35-2.9e-7 raw). Two readings, both
    outside the current representation: (a) no old tier at all and young sources whose
    content is richer than ours (F61/F65 say a plain window at 4 loses 6x on our chain);
    (b) an old-content representation that never re-forms dense legs, i.e. one that
    does the elementwise products A*P, P*P, A*A in factor form -- exactly the F66/F67
    wall. Reformulates the F59/F60 open question as one number: 107 units.
  * DECISION: "better engineering" is worth <= 4% on V29 on paper (lever 1) and ~0%
    shippable at the current residual (F81's ~0.1 u per ms law); it is not the 1.7x;
    pricing polish on the young tier (already Strassen-L5), the covariance (2.7%) and the
    closure (2.2%) cannot move the multiplier. The multiplier is set by the number of
    dense (n,n) legs formed per layer: 8 young + 2 x 10 old re-formed at L14. Any V30
    that matters must cut that count, not the price per unit. Lever 1 is a cheap V30
    only if bundled with a raw gain (F75 headroom: leaders' raw 1.84-2.02e-8).
  * WRITE-UP: the ledger is the cost anatomy figure (family x layer table), and the
    "leaders = us minus the old tier" identity is a clean, checkable claim.

- F87 (2026-09-09) -- LITERATURE TRIAGE #4 (alphaXiv MCP semantic-search pass; screening only,
  no PDFs downloaded, no code run): the embedding search RE-FINDS the triage #3 corpus and
  surfaces FOUR new candidates, all DISCARDED by the F85 triage rule. Confirms triage #3
  was exhaustive; NO literature lever for the score. Do not run another literature pass
  without a new lens.
  * Method: alphaXiv MCP (api.alphaxiv.org/mcp/v1, API-key auth; called directly over
    HTTP JSON-RPC from a scratchpad script, NOT registered in the harness). Five
    discover_papers queries (third-cumulant propagation through ReLU; beyond-covariance
    moment propagation; cheap order-3 cumulant tensor propagation; finite-width 4-pt vs
    depth; white-box expected-activation estimation), difficulty 6-7. Hits already in
    docs/papers/ and covered by F83-F85: 2605.05179, 2605.24072, 2503.21059, 2503.22082,
    2601.16830, 2608.20403 -- i.e. the semantic index agrees with the hand-picked corpus.
  * New candidates, screened with answer_pdf_queries (page excerpts) against the rule:
    - 2601.22307 (Kuang-Lin 2026, exact Gaussian moment matching for residual nets,
      probit/GeLU/ReLU/Heaviside/sine, arbitrary (mu, Sigma)): ORDER 2 ONLY (mean +
      covariance under a Gaussian ansatz per layer) -> rule (2) fails. CITE-ONLY at most:
      its "moment matching removes the leading low-variance errors" bound (smooth-distance
      error bound, Sec. 1 contribution) is a possible reference for why covariance
      propagation is the right base of the hierarchy; its Table 1.1 is a compact survey
      of the order-2 moment-propagation literature and its assumptions.
    - 2608.20483 (Adams-Venturi 2026, UCSC, uncertainty propagation in auto-regressive
      random NNs): same group and same mechanism as 2503.21059 (F84 DISCARD): activation-
      pattern-preserving linearization, mean/covariance closed forms, extended to random
      parameters and iterated maps. Order 2, quenched-linearized -> DISCARD.
    - 2608.21588 (Ferko-Mutchler 2026, NN field theory at finite width; hep-th): which
      QFT axioms (reflection positivity, cluster decomposition) fail at finite N. No
      recursion, no algorithm, no cost -> DISCARD (framing citation only, and F85 already
      has Halverson-Maiti-Stoner for that).
    - 2608.10161 (Tang-Zhang-Han-Wu 2026, bandable cumulant tensors): SAMPLE-based tapered
      estimator for ORDERED coordinates (time/space/genome) where high-order interactions
      decay off the diagonal; O(p k^{d-1}) entries. Our neurons carry no ordering and a
      dense He weight map destroys any bandedness after one layer -> rule (3) fails
      (dense-tensor saving only) -> DISCARD.
  * TOOL VERDICT (for future sessions): the MCP is a better DISCOVERY tool than keyword
    arXiv/WebSearch (embedding search + page-level Q&A + raw full text + repo reading),
    and it needs no PDF hand-downloads; it does NOT change the content of the field. Use
    it only (a) if a NEW lens appears (a new object to search for), or (b) after Oct 17
    to pull exact full text of cited papers for the write-up. Never load its 19 tools
    into a score-work session.

- F88 (2026-09-09) -- THE V29 CHAIN IS AT ITS RAW FLOOR AND THE OLD TIER HAS NO FACTORED
  ESCAPE: the two untested raw levers measure 0.0% (f32 -> f64) and score-NEGATIVE (R_FB 32),
  and the three "open door" representations left by the F86/F71(b) audit close by arithmetic
  at the F72/F73 rank law. Track 2 (raw gains on this chain) is capped at the F75 closure
  floor 2.07e-8 (dumps) = ~1.94e-8 (LB) = score ~4.9e-9 = rank ~9; track 1 (a cheaper
  representation of OUR expansion) has no candidate that passes inspection. Plan doc:
  docs/plan_tracks_2026-09-09.md. Harness: DATA scratch/test_v29_knobs.py (env V18_R_FB,
  V17_R_RES; copy in starterkit/whest-starterkit/), DATA scratch/estimator_v29_f64.py +
  test_v29_f64.py (V29 with fnp.float64 everywhere, billing irrelevant).
  * E1, f32 HOT PATH IS LOSSLESS: V29 all-float64 on 6 dumps 2.2754e-8 vs f32 2.2753e-8
    (+0.004%; per dump -0.06% .. +0.09% = noise; runs/v29_f64_d0-7.log vs v29_gc.log).
    The "~2-3% unexplained f32 slice" of the F75 sum check was an artifact of multiplying
    independently measured percentages; there is no precision loss to buy back, and the
    selective-f64 idea (n^2 families at 2x billing, +2.2% C) is DEAD. Consistent with the
    V28 STRASSEN_MIN note (MSE unchanged to 4 digits at L5). Residual under f64: 0.30 s
    steady, 0.48 s first predict (informational only).
  * E2, R_FB 16 -> 32 ON V29 PRICING IS WORSE: 8-dump raw 2.2753 -> 2.2190e-8 (-2.47%;
    F75 measured -3.4% at V25), steady C/B 0.2526 -> 0.2676 (+5.94% = +15.4 u), score
    x1.033 (runs/v29_rfb32_d0-7.log). Resolves the F75 (+23 u at V25) vs F86 (fb family
    11.5 u, "linear in r") inconsistency in F75's favour: the fb rank also widens the
    feedback range finder and its transports, so the family is NOT linear in r at the
    ledger's 11.5 u. F75's verdict stands; residual +0.02-0.03 s (0.27-0.28 s steady).
  * TRACK 2 STATE: every measured knob of the chain is score-negative or flat at V29
    pricing (QPASS=2, R_RES, R_OLD2, LAM/BETA: F72/F73/F75; R_FB and f64: here). The only
    untested item is the F68 second regeneration mode (ORACLE -1.1% on dumps 0/1 with
    9 live modes; a fitted table captures less) -- optional, <= 1%. Ceiling arithmetic:
    the F75 closure floor 2.07e-8 on the dumps is ~1.94e-8 on the LB scale (dumps run
    ~6.5% high: V25 2.2704 vs 2.13, V19 2.190 vs 2.083), i.e. 1.94e-8 x 0.2526 = 4.9e-9
    = rank ~9 on the 2026-09-10 board; rank 5 (3.8e-9) at our multiplier needs raw
    1.5e-8, BELOW the floor. Raw work on this chain cannot reach the top 5.
  * TRACK 1, CLOSED BY ARITHMETIC (no code; the audit of F63/F65/F66/F67/F71/F72/F73/F76
    /F86 left exactly these doors, restated so they are not re-proposed):
    (1) Tucker core of the merged old tier in the shared basis Qc (r): the (2,1) slice
        needs sum_{m m' m''} Qc_im Qc_im' Qc_cm'' T_{m m' m''} = n r^3 flops = r^3/n^2 units
        for the WHOLE tier: r=128 2.0 u | 224 10.7 u | 256 16 u | 384 54 u. The rank law
        (F72/F73: r/n = 3/8 at age 4, 1/4 at 6, 7/32 at 7, cliff at 1/8) needs r = 224-384,
        where the core costs MORE than the ~6-14 u those sources cost today (F86: ~1.1 u
        per old source-layer). Wins only for r <~ 150, under the cliff. F71(b)'s "n r^3"
        now with the numbers. Rotating the core each join is cheap (3 r^4 = 0.1-0.4 u) and
        irrelevant.
    (2) A multiplicatively closed basis (Q_p * Q_q low-rank, Fourier-like): not choosable.
        Qc is the range of the weighted leg Gram of a random-matrix product (E:896-918),
        dense and data-determined; no structured Qc holds the F72 rank law.
    (3) Hub-side merge of the old tier (kn hub columns -> fewer): CP-rank reduction of the
        merged tensor = F76 (column importance flat: cap 4n -> D21 rel err 0.39). A low-rank
        birth factor a_b = w1*C_off (rank s) would cut only the A-leg forming to (s/n) u;
        the P leg is born as the identity and full rank (F46-UPDATE), and C_off's bulk is
        full rank at this shape (F66: the rank-64 energy is the rank-1 mean mode).
    (4) Adjoint (backward sensitivity of the final mean to D21 at each layer): still needs
        P_b(l) for every (b,l) pair and one n^3 contraction per pair (<Adj, D21^(b,l)> =
        sum_q w2_q sum_i A_iq P_iq (Adj A)_iq), and it linearizes the D21 -> covariance ->
        birth feedback that F69 measured at -10% raw. No cost drop.
  * WHAT THE BOARD SAYS (2026-09-10T01:13Z, reports/leaderboard_history.csv): marius_binner
    NEW #1 at 2.70e-9 with raw 1.71e-8 at ~0.158xB -- raw BELOW our closure floor at a
    multiplier near the Puffi tier; thegamers +7 to #5 at 3.80e-9 with raw 1.79e-8 at
    0.212xB (our cost tier). Two teams now carry content beyond K3 + memoryless kappa4
    at <= 0.21xB. Together with F86 (leaders' bill = ours minus the old tier), the
    consistent reading is an EXPANSION that is not per-source hub tensors: neither a
    compressed version of ours (this finding) nor a slice/mixture/atom closure (F65/F76/
    F78/F82). That is the open question in its final form; gate for any proposal in
    docs/plan_tracks_2026-09-09.md (F85 triage rule + < 1.1 u per old source-layer
    without dense legs + D21 eps <= 2.2% per F71f).
  * WRITE-UP: the f64 null result is a clean claim ("the estimator is precision-
    insensitive at f32; all error is truncation/closure"); the four arithmetic closures
    are the negative-result appendix of the cost anatomy (F86).

## Decisions

- D1 (2026-07-11) — Port-vs-reimplement RESOLVED: do NOT port the kprop machinery now
  (~4k lines torch, 8 modules; hazards: as_strided diagonal views, view-aliasing tricks in
  factor_k3, flopscope grants no symmetric discount, wall-time over cap). Evidence (F7-F9):
  best-known analytics (K=3 augment factored) at most TIES trivial MC after discounts, and
  contributes <8% in an optimal combination. The competitive frontier is sampling-class:
  leaders beat the MC invariant by only 3-6x. Plan: (1) ship batched-MC-at-floor as the new
  baseline (~5.9e-7 adjusted, 11x better than our current submission), (2) research
  variance/cost reduction in the 3-6x range, (3) revisit a LEAN kprop port only if a hybrid
  concretely needs an analytic component. Reference materials if we do port: repo's own
  factored-vs-unfactored tests + golden files (tests/goldens/, n=16 depth=3 kmax<=4).

## Open questions / hypotheses

- H1: ANSWERED 2026-07-11 (F7): the paper's methods reproduce on competition MLPs, but at
  L=32 even the best variant (K=3 augment factored) leaves a 12.7x gap to trivial MC's
  adjusted score. The paper is the wrong hill to climb at this depth; its value is the
  Wick/Hermite machinery, not the end-to-end estimator.
- H2: SHARPENED by F9: for pure MC the score is flat in N above the floor, so the optimal
  MC operating point is exactly AT the floor (minimize wall time at equal score). The
  question remains for non-MC estimators.
- H3: ANSWERED 2026-07-11 (F13): QMC gain by layer is 21x / 3.4x / 2.4x / ~1.6x at layers
  1/4/8/32 — the final-layer error is dominated by high-order "deep" noise injected by
  the ReLU kinks of layers 2+, which no input-side method reaches.
- H4: LARGELY ANSWERED (F8, F12, F13, all negative beyond plain QMC): affine CV,
  antithetic-on-CP, rotation, layer-1 exact-mean regression CV, moment matching and
  last-layer Gaussianization all fail or add nothing on top of scrambled Sobol. What
  remains open on the 3-6x frontier: estimators that attack the DEEP variance component
  (layer-wise smoothing/resampling with controlled bias?) and FLOP-cheaper forward passes.
  Near-term measured wins instead: dshift scrambling (1.6x vs iid), superlinear N-scaling
  (F14), antithetic-on-dshift (~1.18x).
- H5 (2026-07-11; CLOSED on the input side 2026-07-12 by F18/F19/F21/F22): the remaining
  error is "deep noise" — variance injected by ReLU kinks at layers 2+ that is high-order
  in x (F13), now MEASURED as a chaos spectrum with mean order ~25 and ~18% of variance
  above order 12 (F21). Measured DEAD: (a) smoothing with sampled/extrapolated/analytic
  bias control in every variant (F18), (b) MLMC telescoping over the smoothing parameter
  (F18), higher-order interlaced nets (F19), preintegration/Rao-Blackwell subspace
  conditioning (F22), and — by the F21 ceiling — the entire fitted low-order-surrogate
  class. Externally corroborated (F16). The paper re-read (F20) found no exact variance-
  reduction identity to rescue any of it. STILL OPEN, and now the ONLY standing frontier:
  (c) FLOP-side / network-coupling ideas — MEASURED DEAD same day: no cheap coupled
  forward pass exists at L=32 (F23: rank/sketch coupling destroyed by edge-of-chaos
  divergence, affine tail saves no FLOPs, dtype gives no billed discount), and the
  number-operator Stein CV family is dead too (F24: Hutchinson noise ~ d x signal;
  spectrum spread caps even the exact feature at ~1.4x).
  AFTER TODAY the open set is genuinely thin: (i) whatever the F15 floor-dwellers
  actually do (2-6x raw variance below us at equal compute — no candidate mechanism of
  ours survives measurement; F25 says it need not be exotic scaling, just a variance cut
  we haven't found); (ii) qualitatively better QMC point sets/scramblings for the
  low-order half (median-of-means over scrambles, tuned lattices — bounded by ~2x, F13);
  (iii) analytic-bias breakthroughs beyond the paper — MEASURED DEAD 2026-07-16 at the
  series level (F27/F28/F29): the cumulant expansion is asymptotic with optimal
  truncation K*=3 at n/L=8 (K=4 degrades error 10-20x even if computed for free, and
  the reference K=4 is anyway 1277x budget + ~600GB at competition shape). Higher-K
  analytics is closed permanently; any analytic breakthrough must be a different
  expansion, not more terms of this one. Keep harvesting the forum/LB for
  floor-dweller hints before spending more probe cycles.
- H6 (2026-07-16, program design; follows F30): the F29 truncation wall and the F30
  spectrum deficit are the SAME L/n physics and can be fused into one measured claim.
  (i) Boundary grid, K2/K3 only (cheap, GT for labels is the cost): divergence onset at
  ratio 3 ((48,16),(96,32)); K*:2->3 flip at ratios 5-6 ((80,16),(160,32),(96,16),
  (192,32)); two n per ratio tests K*(n/L) vs K*(n,L). Existing points suggest
  K* ~ c*(n/L), c ~ 0.4-0.5. The K*:3->4 boundary at n=256 is unreachable directly
  (F28 memory wall) — only (ii) reaches it. RAN 2026-07-16 -> F34: K*:3->2 flip AND
  the q' jump both sit in ratio (4,5) — a sharp wall, not a crossing; divergence onset
  in (3,4); K* is a function of n/L only; the linear K* ~ c*n/L guess is DEAD. DONE.
  (ii) GT-FREE turning-point diagnostic, computable at K=3 cost: t_K = RMS(m_K - m_{K-1})
  within the augment family. RAN 2026-07-16 -> F31: gamma is NOT shape-stable (0.96-4.11),
  the geometric extrapolation fails, but the raw observable q = t3/t2 orders all regimes
  monotonically with clean thresholds, and the turn mechanism is overshoot (t_{K+1} >>
  e_K), not term growth. APPLIED at (256,32) same day -> F32: re-anchored q' obeys a
  clean n/L ratio law (0.35 at ratio 8 incl. real MLPs / 8.4 at 4 / ~1e20 at 2), K3 is
  pre-turn at competition shape (t3'/e2 ~ 1), K*=3 stated via class membership. DONE.
  (iii) Fusion: run the chaos probe (F21/F30) on the SAME cells/seeds -> per-cell table
  (kbar_meas/kbar_kernel, t3/t2, K*). RAN 2026-07-16 -> F33: kernel law exact for
  n/L >= 32, deficit monotone in n/L (1.00 -> 1.08 -> 1.4 -> 1.6 -> ~5 at ratio
  128..2), self-averaging breaks below ratio ~8. Fusion triangle COMPLETE — single
  claim available: cumulant propagation is an asymptotic expansion in criticality-
  amplified L/n, K* ~ c*n/L, and the same parameter depresses (and randomizes) the
  spectrum below the F30 kernel law. DONE. Theory anchor:
  Roberts-Yaida/Hanin L/n effective coupling at criticality; our piece = kernel spectrum
  law + first optimal-truncation measurement of the hierarchy as an estimator.
  Per-layer ||kappa3|| amplification variant only if gamma proves stable (needs repo
  instrumentation; t_K version needs none).

- F35 (2026-08-04, flopscope v0.10.0 migration — dtype billing halved our score for one
  line of code): the Aug 3 grader update prices FLOPs by dtype (float32=1, float64/int64=2,
  see _dtype_billing.py/_weights.py in the installed package) and dataset MLP weights
  arrive FLOAT64. Our float32 z promoted to float64 at the first `h @ mlp.weights[l]`,
  so the ENTIRE 32-layer hot loop (matmuls, relu, sums) billed at 2x: 10-MLP mini mult
  0.135 -> 0.266, adjusted 4.93e-7 under v0.10.0. Fix: one-time per-MLP cast
  `w.astype(float32)` (~4.2M FLOPs billed at the heavier rate, negligible) -> mult back
  to 0.1346, adjusted 2.50e-7, raw MSE unchanged at 1.85e-6 (float32 accumulation noise
  ~1e-14 in squared terms, invisible above our MSE floor). Same MLPs/draws, so the 1.97x
  is pure billing. Implications: (a) every estimator variant we ever test must keep
  weights/hot-path float32 from now on — a silent 2x tax otherwise; (b) remaining
  float64 in the pipeline is only the Sobol->uniform->ppf path (~1% of billing; kept
  float64 deliberately, float32 would truncate 32-bit Sobol ints below the 24-bit
  mantissa); (c) data movement (copy/concat 1 FLOP/elt, gather/sort 4) now billed but
  immaterial for us (~2M elts/MLP); (d) einsum(out=) breaking change: we don't use it,
  validate passes, 0 failures; (e) NOT re-tested vs the old-stack anchor 5.09e-7 (Aug 1,
  CLI-default random MLPs) — different MLP set, compare mult not MSE. Upstream starterkit
  examples 03/covariance now fail their own tests (SymmetryLossWarning) — not our code.

- F36 (2026-08-04, forum intel from the flopscope-v0.10.0 thread comments, user thylinao;
  no staff replies yet — recheck for answers before acting on any of it):
  - LEADERS' MECHANISM HINT (feeds F15b, the open compute-spender axis): telemetry from
    ranked submissions shows top entries with instrumented FLOPs F "identical to three
    significant figures across all 50 MLPs" at ~1e-5 compute share — i.e. essentially ALL
    their compute runs OUTSIDE flopscope and is billed only as residual wall time at
    lambda=1e11 FLOP/s, which "remains cheaper than instrumented arithmetic by roughly 2x"
    EVEN AFTER the 1-core restriction. The 8.4e-9 class may be as much a billing-lane
    play as an algorithmic one. Also implies per-submission telemetry (F/residual split)
    is extractable from public submission pages — do this to the top-10 during the
    Aug 7-10 harvest.
  - Wall-time laxity: docs pin predict_timeout_s=30 / wall_time_limit_s=60 but graded
    submissions show per-MLP walls up to 64.8s with time_exhausted=false. Deployed limits
    are looser than documented. Do NOT rely on it (CLAUDE.md: no risky edge cases); our
    ~0.6s/MLP is 100x clear of every number in play.
  - Repricing history: the dtype/data-movement repricing actually landed in 0.9.0/0.9.1
    (op-weight + dtype-rate tables byte-identical 0.9.1 vs 0.10.0). Irrelevant to us
    (we jumped 0.8.0rc5 -> 0.10.0) but dates the change window.
  - float16 bills at rate 1.0, same as float32 — no half-precision discount. Sub-32-bit
    BIT-PACKING into uint32 lanes is a bounded ~32x discount on mask/predicate-heavy
    workloads — a potential FLOP-side lever for us (pairs with the F28-F34 sparsity axes).
  - Regrade mechanics (our API check, same day): old submissions were repriced from stored
    traces, NOT re-run — uncast v5 316030 moved 3.44e-7 -> 3.4654e-7 (+0.7%, data-movement
    charges only; stored traces evidently carry no dtype info so the 2x float64 tax did
    not apply), raw MSE bit-identical, 315894 untouched at 2.9157e-7. The 2x tax bites
    only on actual re-EXECUTION — i.e. new submissions and the Aug 11-17 private re-eval.

- F37 (2026-08-04, external write-up intel: radiant-allomancer sub 317660, Discourse 18085,
  Claude-assisted campaign, 4.47e-7 adj at mult 0.1009 floor / raw 4.21e-6 — BEHIND our
  3.44e-7 / raw 2.67e-6, but the methodology transfers):
  - MOST STEALABLE: Rao-Blackwell final layer + sample-based Edgeworth. They never sample
    ReLU at layer 32 — they estimate pre-activation moments (mu, sigma, kappa3, kappa4)
    from the SAME prefix samples and close analytically: E[ReLU(z)] = mu*Phi(a)+sigma*phi(a),
    a=mu/sigma, plus delta_skew = -(kappa3*mu/6sigma^3)*phi(a) and
    delta_kurt = (kappa4/24)((a^2-1)/sigma^3)*phi(a). Claimed to remove 98% of the
    non-Gaussian gap. The metric is FINAL-layer MSE only, so this targets exactly what is
    scored. Our v5 takes the raw sample mean at layer 32 — a direct upgrade candidate at
    ~zero FLOP cost. (Extends F16/evaaaz's RQMC+RB confirmation with the bias-correction
    half.)
  - Free: exact layer-1 mean (Gaussian input => sigma_1/sqrt(2pi) analytically).
  - DIAGNOSTIC worth running on v5: two-budget bias/variance split — fit m(N)=b^2+c*N^-p
    across two utilization levels; they find 59-73% of residual error is BIAS, not
    variance, at their floor operating point. If v5 is bias-dominated too, more samples
    buy nothing and the RB+Edgeworth closure is the right lever; if variance-dominated,
    it is not. Cheap, decisive experiment.
  - Their negative results INDEPENDENTLY CONFIRM our dead ends: input-anchored control
    variates decorrelate with depth (corr 0.86 L1 -> 0.25 L32; our F18/F19); transported
    third cumulant loses ~98% cross-cumulant info (our F29-F34 truncation walls, same
    physics); layer-wise analytic anchoring compounds bias 4-9x; median-of-means loses
    to plain means. Their 5%/10% moment-shrinkage toward the analytic chain barely
    helping = same conclusion from the other direction.
  - Also ran a 14-term ridge corrector trained on 200 local MLPs (+~10% raw) — learned,
    not mechanistic; note for score but weak for the algorithmic prize (ARC preference,
    see whest-algorithmic-prize-mechanics memory).

- F38 (2026-08-04, F37 follow-up RAN — RB/Edgeworth final layer is DEAD for v5; the
  two-budget probe says why in one number: OUR BIAS SHARE IS 0.9%, THEIRS WAS 59-73%.
  scratch/rb_edgeworth_probe.py, 8 mini MLPs x 4 shift replicates, GT noise floor 2.3e-9):
  - v5 at N=8192 is VARIANCE-dominated: m(N) = b^2 + c*N^-p fits b^2 = 2.1e-8 vs
    m(8192) = 2.4e-6, p = 1.06. Sub-317660's bias-dominated regime (their 59-73%) comes
    from their own pipeline (moment shrinkage, corrector, floor operating point), not
    from the problem. Their RB closure fixes a disease we do not have.
  - Rectified-Gaussian closure alone is WORSE than the sample mean (bias floor: 1.37e-6
    vs 5.4e-7 at N=32k) — the layer-32 pre-activations are genuinely non-Gaussian.
    Edgeworth skew recovers most of it, +kurtosis lands at parity: 1.00x at every N.
    Their "98% of non-Gaussian gap removed" claim REPLICATES; it just buys nothing when
    the plain mean is already unbiased and QMC has killed the variance. (Formulas
    verified by hand derivation before implementing — both match Hermite tail integrals.)
  - p = 1.06 at the final layer: the QMC advantage survives as a CONSTANT (F10's 1.4-2x),
    not as a rate improvement — 32 layers of ReLU mixing destroy the low-discrepancy
    rate, as expected.
  - Consequence (F9 arithmetic still rules): variance-dominated + score = MSE x C/B means
    score is ~N-invariant above the floor. The only remaining levers are (a) better
    variance reduction PER SAMPLE, (b) the F36 uninstrumented-compute lane, (c) genuinely
    new estimators. Free-but-tiny: exact layer-1 mean (unscored layer, skip).
  - Replicate-0 sanity: probe baseline at N=8192 = 2.40e-6 raw vs v5's graded 2.67e-6
    (different MLP subsets) — numpy replica of the sampling core is faithful.

- F39 (2026-08-18, CORRECTS F36's interpretation — the "uninstrumented-compute billing
  lane" is measured-EXPENSIVE by two independent strong Phase 1 entrants; it is NOT a
  cheap lane to exploit): two public Phase 1 write-ups posted to Discourse (deadline was
  Aug 17) directly quantify the metered-FLOP vs residual-wall-time trade-off, and both
  conclude the opposite of F36's free-lunch reading:
  - bin_yong_bong (topic 18184, "Measured: the metered-FLOP vs residual-wall-time
    trade-off"): tested Strassen-Winograd matmul to trade metered FLOPs for residual
    wall-time. Depth-5 Strassen saves ~24k metered FLOPs/sample but spends ~432k
    residual-equivalent FLOPs = an 18:1 LOSS (133:1 at chunk 8192). The lambda=1e11
    residual coefficient already makes the residual lane expensive; optimum is shallow
    (depth ~2). Their optimized submissions used 5-46 ms residual/MLP (0.2-1.7% of B).
    Conclusion: no hard cap needed because the lane self-penalizes; proposes ~250 ms/MLP
    as a Phase 2 floor to avoid punishing normal overhead. No exploitable 2x discount.
  - amalgonim (topic 18151, "[Phase 1 write-up] Where the closure family's floor actually
    is..."): (a) the distribution-propagation / cumulant-closure family has a hard MSE
    floor of 8.8e-7 at layer 31 EVEN WITH PERFECT MOMENTS — 40x worse than plain MC on
    identical draws, 3.7x worse than their deployed sampler; Edgeworth 3rd-cumulant makes
    it 3.8x worse. Independent external confirmation of our F18/F29-F34 (closure/higher-K
    is a dead end) and F38 (Edgeworth buys nothing). (b) Per-call residual cost model in
    float32 at (8192,256)@(256,256): matmul ~46 us (~4.6M FLOP-eq), concatenate ~56 us
    (~5.6M), ReLU ~20 us (~2.0M), add ~17 us (~1.7M), slices/views ~0.7 us and BILLED 0
    ("views are free both ways"). Optimizing billed FLOPs alone builds SLOWER code; row
    bucketing reaches practical optimum at 1.18x dense cost; Strassen doesn't combine.
  - CONSEQUENCE for our score-first push: F36's headline lever (leaders escape metering
    into a ~2x-cheaper residual lane) is contradicted by direct measurement. The residual
    lane is priced at lambda=1e11 and teams MINIMIZE it (matching our own v5: ~6 ms/MLP,
    residual = 1.3% of mult). Our score is FLOP-dominated + variance-limited (F9/F19/F38);
    cheaper matmul scheduling is real engineering but doesn't move a score that is flat in
    N and pinned near the 0.1-floor region already. The genuine remaining score frontier
    is unchanged and still unexplained: the floor-dwellers' 2-6x raw-variance edge (F15/
    F25). Notably, the public Phase 1 write-ups so far are from teams at ~our level
    (sampling/closure class) documenting engineering + negative results, NOT the top
    ~1e-7 teams — the floor-dweller mechanism remains publicly undisclosed.
  - ACTION: harvest the rest of the freshly-posted Phase 1 algorithmic write-ups (Aug 17
    deadline) for any floor-dweller variance mechanism before spending probe cycles;
    that is the highest-EV score activity while Phase 2 rules are still pending.

- F40 (2026-08-24, SETTLES F36/F39 by rule, not measurement — Phase 2 makes the flopscope
  meter the ONLY compute currency): P2 rules (v14, accepted) prohibit computation outside
  flopscope outright — no numpy/BLAS, no compiled kernels or runtime codegen, no FFI, no
  threads/subprocess/asyncio, no compute while a flopscope op is in flight; residual time
  is a 400 ms plumbing allowance and doing math there is disqualifiable EVEN under the cap
  (C_m = F_m, lambda = 0). The F36 lane debate is closed permanently.
  - New free channel instead: setup() is off-budget AND off-residual (5 s cap per run,
    runs 5-15x per submission, must be idempotent; overrun fails the whole submission).
    With C = F, shipped data files loaded in setup() are the cheapest resource in the
    game — offline precompute is unmetered, so anything expressible as a shipped table
    or weights is free at eval time. ("Not genuinely data" files can be DQ'd — sponsor
    discretion; borderline ideas can get BINDING written approval via
    arc-whestbench@aicrowd.com before building on them. Bit-packing explicitly banned.)
  - Meter mechanics worth exploiting (documented in the P2 kit's example 03, verified in
    our run): float32 bills at HALF the float64 rate (flops.stats.norm.pdf/cdf promote to
    float64 — cast back or the 2x rate re-infects downstream); symmetric-tagged einsum
    contractions (flops.as_symmetric) bill ~25% cheaper, and a diagonal write silently
    voids the tag — re-tag after every fill_diagonal (re-validation repays ~146x).

- F41 (2026-08-24, Phase 2 LB structure on day 3 — the top tier SPENDS the budget): our
  first P2 submission #328319 (kit cov-prop example verbatim: 5.17e10 FLOPs/MLP = 2.4% of
  B, floored 0.1x) graded 3.93e-7 adj = rank #25/40, 7-way tie of identical baselines.
  - Tier structure: top 5 (Puffi 1.50e-8 ... mliston 8.48e-8 adj) all run mult 0.42-0.97,
    i.e. they spend 42-97% of B for raw MSE 3.6-8.8e-8 (~100x better than cov-prop; leader
    "80x vs sampling"). Ranks 6-16 sit AT the 0.1 floor with raw ~1e-6 (a cheap-method
    tier ~4x better than baseline). Counterexample AXOLOTL #19: mult ~1.0 for raw 2.67e-7
    — budget spend without the right estimator does not pay.
  - ARC's stated P2 design intent (launch post): P1 winners were sampling variants; nets
    were made 4x wider / 2x shallower specifically to make cumulant-propagation-style
    methods competitive, with depth-dependence still the hard part at L=16. This aligns
    the score push and the algorithmic write-up (stretch goal) on the same axis.
  - Ladder implied: beat the stock baseline cheaply (breaks the 7-way tie), reach the
    floor tier (raw ~1e-6), then hybrid spend-the-budget toward raw ~4e-8.

- F42 (2026-08-24, P2 FLOP COST ANALYSIS for the hybrid port — the P1 hybrid premise is
  INVERTED at (1024,16): sampling lost its crown, analytics is now the engine and MC at
  most a floor-priced shrinkage sidecar. Measured, not assumed; scripts
  scratch/kprop_p2_cost.py (torch flop_poly_fit at P2 shape) and
  scratch/probe_flopscope_p2.py (flopscope 0.12.0 charges at width 1024)):
  - REFERENCE KPROP COST at (1024,16) vs B = 2^41 = 2.199e12 (exact polynomial fit,
    dtype-blind counts = flopscope float32 rate):
    K=2 simple 5.18e10 = 0.0236x B (== ex03 cov-prop's 5.17e10, same n^3 L term);
    K=2 AUGMENT 5.35e10 = 0.0243x B (augment is FREE at this shape — the n^3 L
    contract_W term dominates both; at P1 shape augment mattered for accuracy but here
    it costs nothing extra — cheap accuracy experiment pending);
    K=3 simple factored 4.91e12 = 2.23x B (get_dslice 47.6% + contract_W 47.2%);
    K=3 augment factored 1.86e13 = 8.48x B (pK_211 towers add ~4e12 each);
    K=4 augment factored 2.94e16 = 13,348x B — dead forever, as at P1.
    Leading terms: K3s 18n^3L^2, K3a 60n^3L^2, K4a 48n^4L^2.
  - FLOPSCOPE 0.12.0 CHARGE TABLE at width 1024 (float32 rate 1, float64 2x, MIXED
    f32@f64 bills the FULL 2x — cast weights first, F35 still binding):
    standard_normal 16/elt f32 (32 f64); uniform 2/elt; integers 1/elt;
    norm.ppf 166/elt (output ALWAYS f64 — cast back), pdf 54/elt, cdf 96/elt (both
    bill at the promoted-f64 rate); relu/xor/shift/copy 1/elt; astype 2/elt;
    matmul 2mnk exactly; eigh(1024) 9.7e9, qr 2.9e9, svd 2.8e10, chol 2.5e9 — all
    linalg <=1.3% of B, freely usable.
  - SYMMETRY DISCOUNTS (the big meter lever for a kprop port): pair-symmetric tagged
    einsum bills 0.750x naive; FULLY-SYMMETRIC 3-TENSOR (as_symmetric symmetry=(0,1,2))
    einsum ijk,ia,jb,kc->abc bills 0.391x — a 2.56x discount, accepted by flopscope
    0.12.0 today. Chained matmuls get NO discount even on tagged operands (single
    einsum only); untagged symmetric VALUES get nothing (tag or pay); fnp.eye arrives
    pre-tagged (my first probe's "untagged" 0.75 was that). Port consequence: K3-simple's
    billed cost lands in ~[1.9e12, 4.9e12] = 0.9-2.23x B depending on how much of
    get_dslice/contract_W can be expressed as single symmetric-tagged einsums.
  - MC SIDE at (1024,16): forward pass 3.357e7 FLOPs/sample f32 (65,504 samples = full
    B; 6,550 at the 0.1 floor). Sobol overhead (gen 1/elt + ppf 166/elt + casts) ~0.5%
    of a pass — the whole v5 Sobol machinery ports for free. BUT: measured mean
    per-neuron final-layer variance vbar = 7.36e-2 (3 He nets, N=8192, torch), so the
    PURE-MC INVARIANT SCORE = vbar*c_pass/B = 1.12e-6 — WORSE than the stock cov-prop
    baseline (3.93e-7 adj). At P1 pure MC beat cov-prop 12x; at P2 it LOSES 2.9x.
    Even with the ~2x QMC factor (F13 depth profile at L=16) Sobol-MC ~5.6e-7 still
    loses to plain cov-prop. ARC's re-shape did exactly what they said it would.
  - HYBRID ARITHMETIC (oracle per-neuron shrinkage MSE = b^2*v/(b^2+v), mult =
    max(0.1, s+0.024), v_sobol(s) = 5.6e-7/s at budget share s):
    (a) cov-prop + Sobol at the floor (s=0.0765, mult exactly 0.1): combined raw
        2.63e-6 -> adjusted ~2.6e-7 = 1.5x better than the baseline tie, reusing v5
        machinery + ex03 verbatim. The natural first non-trivial P2 submission.
        (iid instead of Sobol: 3.2e-7; spending above the floor NEVER pays here —
        score worsens monotonically once mult unpins.)
    (b) K=3-class base (projected raw ~6.4e-8 IF c_K transfers: 4.1e-6/64 per the
        (L/n)^K law; F29 warns c_K is not shape-stable, so range 3e-8..2e-7):
        MC correction on top adds <=6% MSE reduction while multiplying mult — at K3-level
        bias MC is WORTHLESS (v at any affordable N >> b^2). The endgame estimator is
        ANALYTIC-ONLY at the lowest achievable mult: K3-simple as-is 2.23x B -> score
        ~1.4e-7; with symmetry-tag engineering ~1.5x B -> ~1e-7 (floor-tier); a
        mean-targeted/lean reformulation at <=0.5x B -> ~3e-8 = leader territory.
    Priority order that falls out: (1) K2-augment accuracy check at P2 shape (free
    cost, maybe breaks the tie alone), (2) ship hybrid (a), (3) K3 accuracy
    measurement at (1024,16) with the torch reference (does simple hold at L=16? real
    b^2, settles c_K), (4) port K3-simple to flopscope with symmetric tags, engineering
    the 2.23x down. Wall-time flag for (4): 2e12+ FLOPs must clear the 120 s cap on the
    7-core flopscope backend (~sgemm-bound should be fine at ~1e11 flops/s, but the
    reference was overhead-bound in torch — F7's lesson stands).

- F43 (2026-08-24, K2-AUGMENT + K3 ACCURACY MEASURED on 8 REAL P2 mini MLPs — K3-SIMPLE
  DELIVERS TOP-5-CLASS RAW ERROR AND IS STABLE AT L=16; both augment variants are dead.
  scratch/kprop_bench_p2.py on scratch/mlp_dumps_p2/ (v2-phase2 shard 0, ids 0-7, GT from
  1e9 samples -> noise floor ~8e-11, 500x below everything measured; torch f64 reference,
  convention check layer-0 exact passed on all 8):
  - covprop(ex03 f64) == kprop K=2 simple to 13 digits on every MLP (as at P1): mean
    mse_final 4.39e-6 (per-MLP 2.5-5.9e-6) — consistent with the published 4.1e-6 over
    100 MLPs. Our 8-MLP subset is representative.
  - K=2 AUGMENT: 1.18-1.33e-5 = 2.6x WORSE than simple at the final layer. The F32
    k2-augment depth-instability is alive even at L=16/ratio 64 — the free-cost augment
    idea from F42 is DEAD, don't spend a submission on it.
  - K=3 SIMPLE factored: mean mse_final 4.32e-8, tight across all 8 MLPs (3.2-5.0e-8),
    mse_all 1.70e-8 — 102x better than K=2, slightly BEATING the (L/n)^K naive
    projection (6.4e-8). And unlike P1 (F7: simple-K3 non-monotonic, worse than K2 on
    3/8 at the final layer at L=32), at L=16 simple is uniformly excellent — the augment
    pieces are NOT needed at ratio 64. c_K transfers (even improves) at P2 shape.
  - K=3 AUGMENT factored: OOM at n=1024 (single 8.6 GB torch alloc failed; run-1 crash
    also took down the process during the next variant — results_run1.json). At 8.5x B
    billed cost it was score-irrelevant regardless. Augment is closed at P2.
  - Wall time: K3-simple 19.3 s/MLP in overhead-bound torch f64 (22 threads) — the
    120 s grader cap is comfortably in range for a flopscope port at this shape.
  - SCORE LADDER (raw 4.32e-8 with F42 costs): reference port as-is 2.23x B ->
    adjusted 9.7e-8 (floor-tier class, ~rank 6-16); symmetric-tag engineering ~1.7x ->
    ~7.3e-8; ANY port at <=1.0x B -> <=4.3e-8 = top-5; at 0.5x B -> 2.2e-8 = podium
    (Puffi leads at 1.50e-8 adj). The whole game is now COST ENGINEERING of K3-simple:
    get_dslice + contract_W (95% of 4.91e12) expressed in symmetric-tagged einsums,
    f32 hot path, and any mean-targeted slimming. MC is out of the estimator entirely
    (F42); Sobol machinery stays shelved unless a fallback needs it.)

- F44 (2026-08-25, K3 COST-ENGINEERING PROBES: approximation loses, exact restructuring
  wins ~22-25%; the v1 target is the EXACT K3-simple port at ~1.7x B.
  scratch/k3_window_probe.py + k3_shadow_probe.py on 4 real v2-phase2 MLPs, FLOPs
  measured with the repo's NamedFlopCounter at (1024,16); flopscope f16 rate checked in
  flopscope/data/default_weights.json):
  - COST ANATOMY (exact, verified against measured 4.93e12 ~ 4520 n^3): the entire
    2.24x B is rank growth of the factored K3. Each nonlin appends 3 rank-n blocks
    (two add_factors "hub" blocks B1,B2 with middle leg 3I; one from_dstensor DS block
    B3 with legs 2,3 = I), so r(l) = 3n*l, and per layer per source the meter pays
    contract_W 18n^3 + dslice(2,1) 18n^3 (3 Sym perms x hub matmul 2n^2 r).
    18n^3 L^2 total. All other work (94 eval_part terms, pK->K, harmonic proj) is n^2
    -class = 5%.
  - HARD WINDOWING (drop blocks older than W layers) FAILS: mse_final degrades faster
    than mult drops at every W (full adj 1.08e-7; best window adj 3.17e-7 at W=8;
    W=1..4 all ~1e-6 class). Old K3 blocks matter at full strength.
  - DS-SHADOW COMPRESSION (replace expired blocks by a rank-n from_dstensor block of
    their (3,)+(2,1) dslices, i.e. discard only their fully-off-diagonal content)
    beats hard windowing ~4x at equal cost (S=1: 6.9e-7 vs W=1 2.9e-6) but STILL loses
    to exact: S=4 adj 2.68e-7, S=2 3.93e-7, S=1 4.46e-7 vs full 1.08e-7. CONCLUSION:
    the (1,1,1) off-diagonal content of old blocks carries most of the K3 correction;
    approximate compression of the K3 tower is a dead end at P2 accuracy scales.
  - FLOAT16 BILLS AT 1.0 = SAME AS FLOAT32 (flopscope default_weights.json dtype_rates:
    f16/f32/c64 all 1.0, f64/i64 2.0, f96 3.0, f128 4.0). No half-precision discount
    lever. int8/int16 also 1.0 - no sub-1.0 rate exists at all.
  - EXACT SAVINGS THAT DO WORK (algebraically verified, to implement in the port):
    (1) SHARED-P INVARIANT: identity-born legs (B1/B2 middle 3I; B3 legs 2,3 = I) all
    evolve by the same map leg <- W @ (w1 * leg) (contract_wick multiplies all legs by
    the same w(1); new blocks join after wick). So per source layer propagate only
    {X1,Y1,X2,Y2,M} + one shared P (6 matmuls vs 9), P's birth step is free
    (row-scaled W), and B3=(M,P,P) needs only 2 distinct hub matmuls in the (2,1)
    dslice (perms {12},{13} coincide). Net ~22-25%: 2.24x B -> ~1.7x B.
    (2) WK[4] IS STRUCTURAL in simple mode with avg (diagonal) metric: HTensor
    (scalar core, r=2, metric c*I) has only slices (4,) = const vector and
    (2,2) = const*(ones-I); slices (3,1),(2,1,1),(1^4) vanish, killing all eval_part
    terms that touch them. K4 handling reduces to a few scalars per layer.
    (3) All partition/coefficient combinatorics (term tables, vec_part_coef,
    _pK_to_K_coef, proj_coef, relu-wick polys) is pure-Python scalar work ->
    precompute at estimator setup() off-budget; only n^2/n^3 array ops hit the meter.
  - SCORE PROJECTION for the exact port: raw 4.83e-8 (4-MLP mean; F43 8-MLP mean
    4.32e-8) x ~1.7 mult -> adj ~7-8e-8 = top-5 edge (leader Puffi 1.50e-8 adj).
    Straight unrestructured port = 2.24x -> adj 1.08e-7 still 3.6x better than our
    live cov-prop submission (3.93e-7). Wall-time flag: ~3.8e12 f32 FLOPs on the
    7-core flopscope backend must clear 120 s - test with local_engine early.)

- F45 (2026-08-25, V1.6 K3-SIMPLE PORTED TO PURE FLOPSCOPE: 0.833x B, local mse_final
  ~6.2e-8, projected adjusted ~5.1e-8; residual-time engineering findings.
  starterkit estimator.py (single file, tables embedded); lean parity via
  scratch/lean_k3.py + dumps; probes in this session's scratch scripts):
  - LEAN REIMPLEMENTATION (scratch/lean_k3.py) reproduces the reference factored
    kprop k=3 SIMPLE to machine precision (8.9e-16 at n=48 L=5) with the shared-P/
    atom structure of F44 and coefficient tables generated from the repo's own
    pure-python combinatorics. Reference cost 2.24x B -> shared-P + merged hub
    matmuls 1.43x B measured (torch counter), f32 drift 2.7e-6 (harmless).
  - BUDGET IS A HARD CAP (over-budget -> zero prediction). V1.6 ablation set
    (all DS-shadow-at-birth or measured-negligible drops):
    B2 sub-block shadowed (+11% mse), K3-dslice feedback out of new-block factors
    (-0%), WK22 struct term out of Y1 (+0.3%), terms pruned at rel>=1e-5 (-0%).
    Net: mse_final 6.1e-8 (vs 4.8e-8 exact), billed 1.83e12 = 0.833x B.
    Key extra exact trick: with Y1 = a*D(w2b) (struct-free), the two a-side hub
    perms COINCIDE (column scalings commute across hub legs), so the (2,1) dslice
    needs only 4 distinct contractions -> billing unchanged vs merged-acc form.
  - RESIDUAL-TIME (400 ms cap) FINDINGS for flopscope 0.12.0 clients:
    residual ~= wall - backend - overhead = OUR python between ops. It leaks
    ~90 us PER (n=1024)^2 RESULT BLOCK for pointwise ops/stack/concatenate
    (result alloc + wrap), ~25 us for views/reshape/fill_diagonal, and einsums
    leak only per OUTPUT block (internal temporaries are backend-attributed).
    => the discipline is: fuse elementwise chains into multi-operand einsums
    ('kij,kij,kcj->ic' bills exactly 2kn^3 and leaks ~130 us), batch per-source
    work as (k,n,n) stacks, prebuild all wick vectors as one (n, npairs) matrix
    via power-basis matmuls, evaluate term sums as stacked einsums.
    out= params: 20x less leak but 80x wall - unusable. A 7-operand einsum
    'tp,pij,tq,qij,ti,tj,tg->gij' takes a catastrophic path (6x overbilled,
    4.4 s wall) - keep operand counts <= 5. Same-object triple operands can
    crash flopscope's accumulation cost ("cycle size mismatch") - avoid.
    Port evolution: naive per-term/per-source 18.3k client calls = 1.22 s
    residual -> fused 3.1k calls but big stacks 0.84 s -> struct-free einsum
    hubs 0.31 s PASS. Grader margin ~25%; first-week free-failure window makes
    submitting now low-risk calibration.
  - Wall 14 s/MLP at 22 threads local (cap 120 s on 7-core backend - ample).
    whest validate PASS (setup 0.00s vs 5s cap). Deterministic, no RNG use.)

## F46 (2026-08-28) — Residual analysis kills learned-output-correction; the WIN is the
## 0.1x MULTIPLIER FLOOR via a low-rank (Tucker-R) K3, provisionally top-9 to #1.
Context: #328476 (K3-simple V1.6) is rank 10 at adjusted 4.47e-8 (raw 5.52e-8, mult
0.833). Investigated the logged "learned correction of K3 truncation bias" plan, which
redirected to a much higher-leverage cost play. All probes: 8 dumped v2-phase2 public
MLPs (GT = all_layer_means from the official parquet = the scored high-budget MC ref),
lean_k3.py float64. Scripts: scratch/k3_residual_probe{,2,3}.py, k3_d21_rank_probe.py,
k3_d21_trunc_probe.py, k3_tucker_scoping.py; lean_k3.py gained diag/force_mean/svd_d21/
d21_rank hooks (default behavior unchanged; parity preserved).

- RESIDUAL STRUCTURE (r = GT - K3pred): per-layer MSE grows monotonically 6.8e-10 (l0)
  -> 4.3e-8 (l15); final-layer MSE 4.32e-8 (f64, 8 MLPs). K3 systematically UNDER-predicts
  the mean (residual mean +ve every layer) but the pure per-MLP bias is only 0.5% of MSE;
  residual is ~99.7% zero-mean per-neuron fluctuation.
- LEARNED OUTPUT CORRECTION IS DEAD: final-layer residual has ~0 correlation with every
  cheap final-layer local feature (|corr|<0.09 for mu, var, sigma, alpha, Phi(alpha),
  pred). A memoryless output-layer correction cannot work -- by l15 the accumulated error
  has been mixed through 16 random W and decorrelated from local stats. Per-layer regression
  R^2 of ACCUMULATED residual on cheap features tops 0.24 (l1) -> 0.03 (l15).
- ERROR IS SELF-PROPAGATING MEAN ERROR: incoming-mean error (K3 mu_in vs the EXACT true
  pre-activation mean W_l @ GT_{l-1}, computable because means propagate linearly) equals
  -W_l @ r_{l-1}, i.e. just the previous residual carried forward; rms ratio 1.0-1.34x the
  residual. Teacher-forcing the true mean each layer (force_mean) roughly HALVES the final
  residual (rms 2.08e-4 -> 0.89e-4). FRESH (teacher-forced) per-layer error IS predictable:
  R^2 0.61 (l1), 0.27-0.43 deep, and the non-Gaussianity signals (D3/D21 dslices, K4 core)
  roughly DOUBLE R^2 over mu/sigma-only -> an ONLINE per-layer mean correction is viable
  (ceiling ~top-9, est 2-3e-8, stays at 0.833x). This is the fallback, not the main play.
- THE MAIN PLAY = THE 0.1x DISCOUNT FLOOR (score s = MSE * max(0.1, C/B)). We pay mult
  0.833; P1 was won at the floor (F14/F44). Break-even at the floor: match current if raw
  MSE < 4.6e-7 (8.3x worse OK), beat top-9 (2.6e-8) if raw < 2.6e-7, beat Puffi (9.5e-9)
  if raw < 9.5e-8. Running our CURRENT raw (5.5e-8) at the floor would be 5.5e-9 = #1 -- the
  only obstacle is cutting K3 cost 8.3x (0.833B -> 0.1B) without wrecking MSE. Plausibly the
  leaders are floor-dwelling with WORSE raw than us; we lose purely by paying full price.
- WINDOWING (drop old source-layers) IS DEAD for the floor: k3_window_probe already showed
  MSE rises FASTER than cost falls (W=8 3.8x MSE, W=1 60x MSE = 2.9e-6); every windowed
  point has worse adjusted score than full. The 3rd cumulant needs LONG memory; old blocks'
  off-diagonal content carries the correction (F44).
- BUT LOW-RANK FEEDBACK SURVIVES: rank-truncating the D21 (2,1)-dslice feedback matrix
  (n x n, zero diag) each layer degrades final MSE gracefully -- R=256: 8.2e-8 (1.7x),
  R=128: 1.45e-7 (3x), R=64: 2.5e-7 (5x), R=8: 5.9e-7 (12x). D21 spectrum: r90=7-20 deep
  (top-1 dir holds 82-87% energy) but a heavy tail r99=150-540. So the correction lives
  mostly in a low-dim subspace with a long low-energy tail.
- KEY MECHANISM INSIGHT: maintaining K[3] as a TUCKER rank-R tensor makes the linear step
  U_i <- W @ U_i cost O(n^2 R) NOT O(n^3) (the source-propagation cost I first assumed does
  not apply). Cheap scoping (k3_tucker_scoping.py: measured D21-trunc MSE proxy + explicit
  Tucker flop model): R=128 -> ~0.045B (mult floors at 0.1), adjusted ~1.45e-8 = TOP-9,
  3x better than current; R=256 -> ~0.042-0.16B, adjusted ~8-13e-9 ~ Puffi. GATING RISK:
  accuracy is the D21-TRUNCATION PROXY (truncates exact feedback, not a propagated-truncated
  3-tensor); real Tucker-R compounds truncation differently. NEXT: build a torch Tucker-K3
  prototype to measure true (MSE, cost) vs R and confirm the frontier before the pure-
  flopscope port. If confirmed: single biggest score jump available (rank 10 -> top-9/#1).

## F46-UPDATE (2026-08-28) — Compression floor play REFUTED; the D21-truncation proxy
## was a mirage. Best remaining path = online learned mean correction (~top-9).
The Tucker scoping (F46) said GO on a PROXY (rank-truncating the exact 2D D21 feedback).
Built the faithful compounding test (k3_tucker_proto.py + lean_k3 tucker_rank hook:
each layer confine all 3 K[3] legs to their dominant rank-R symmetric subspace, Q =
top-R eigvecs of the leg Gram, truncation compounds through depth). RESULT (n=1024,
2 MLPs) is far WORSE than the proxy: R=512 6.6e-7 (14x), R=256 1.49e-6 (31x; proxy said
8.2e-8/1.7x), R=128 2.27e-6 (48x). At the 0.1x floor every rank is WORSE than current
4.47e-8 (R=512 -> 6.6e-8). 
- WHY: the third cumulant's middle "P" leg is born as IDENTITY (full rank) and stays
  full rank under W-propagation, so the accumulated K[3] is NOT low symmetric-Tucker-rank.
  D21 (the contracted feedback) being low-rank (r90=7-20) is a DOWNSTREAM property of the
  contraction, not a compressible property of the propagated state. Truncating the state
  to rank R destroys it; truncating only the output feedback (the proxy) does not -- but
  the output feedback is not an implementable low-cost state.
- CONVERGENT EVIDENCE that K3 accuracy is irreducibly O(n^3 L^2): (1) windowing (drop old
  source layers) MSE rises faster than cost falls; (2) symmetric-Tucker confinement blows
  up MSE 14-48x; (3) the shared-P suffix structure (P_s = P_{s'} @ partial-product) is
  already exploited by V1.6. There is no low-rank state that reaches the 0.1x floor with
  usable accuracy. THE COMPRESSION FLOOR PLAY IS DEAD.
- METHOD LESSON (write-up): validate a compression on the PROPAGATED STATE with compounding,
  never on a single-shot truncation of the output -- output-feedback low-rank-ness does not
  transfer to state low-rank-ness. The cheap symmetric-Tucker prototype (project + eigh, no
  randomized-HOSVD) killed the moonshot in ~30 min vs a multi-day pure-flopscope build.
- BEST REMAINING PATH: online per-layer mean correction (F46: fresh-error R^2 0.61 l1,
  0.27-0.43 deep, non-Gaussianity signals double R^2). Stays at 0.833x; est ceiling
  ~top-9 (raw ~2.8e-8 if it ~halves the residual -> adjusted ~2.3e-8). Must be
  flopscope-cheap arithmetic on per-neuron features (mu, sigma, alpha, D3/D21/K4), fit
  offline on public MLPs (allowed data files), applied INSIDE the loop each layer so the
  correction propagates. Leaders (<2.6e-8) are likely NOT floor-dwelling (floor-K3 is
  impossible, P2-floor-MC too weak at width 1024) -> they run better raw MSE; we must
  improve raw MSE, and online correction is the only cheap handle found.

## F47 (2026-08-28) — Online per-layer mean correction WORKS (~1.45x, generalizes) but
## the ceiling is low (~rank 14-16); top-10 needs a better BASE method, not a correction.
Built the online mean correction (lean_k3.py mean_corr/collect_feats hooks + build_corr_feats;
13 per-neuron features {1,mu,var,sigma,alpha,|alpha|,phi,Phi,pred,D3,D21n,K4,sigma*phi};
delta = feats @ beta[l] added to mu INSIDE the loop so it propagates; scripts
scratch/k3_online_corr.py, beta saved k3_online_corr_beta.npy).
- FITTING LESSON: the naive fixed-point fit (fit beta on the free-running/accumulated
  residual, iterate) DIVERGES (round 0 helps 1.08x test, rounds 1-3 blow up to 0.01x).
  The accumulated residual is weakly predictable (R^2 0.03-0.24). The PRINCIPLED fit is
  ONE-SHOT on the TEACHER-FORCED trajectory (force_mean=GT -> features reflect the
  near-correct path, target = fresh per-layer error, R^2 0.3-0.6), then apply ONLINE.
  Stable, lam-insensitive (1e-1..1e-4 identical), no iteration.
- RESULT (8 MLPs, 5 train / 3 test, f64): baseline test 3.65e-8 -> corrected 2.52e-8 =
  1.45x (train 1.56x; train~test => generalizes, minimal overfit). Cost negligible (13
  dot-products/neuron/layer on already-computed features) -> stays at mult 0.833.
- HONEST RANK IMPACT: apply the 1.45x ratio to our PUBLIC 4.47e-8 -> ~3.0e-8 adjusted =
  ~rank 16 (field is dense: #15 2.49e-8, #16 2.65e-8, #18 4.89e-8). Theoretical ceiling of
  this approach ~2x (teacher-forcing halves the residual) -> 2.24e-8 = ~rank 14. So the
  correction is a bankable +1-2 places, cheap to port, but NOT a top-10 path.
- WHY LOW CEILING: it fixes only our raw-MSE axis (toward frontier levels) while our cost
  stays 0.833x; the field clusters at mult ~0.3 / raw ~2.5e-8 (F46-UPDATE / leaderboard
  2026-08-28). Top-10 requires a more FLOP-EFFICIENT + more accurate BASE method (test
  K3-full vs our SIMPLE mode; partial-K4), not a bolt-on correction. The correction can be
  layered on top of whatever base we choose.

## F47-UPDATE (2026-08-28) — Cheap-base + correction synthesis is DEAD; K3 content is
## fundamentally NONLOCAL. Only viable gain = K3 online correction (~rank 16).
Tested cov-prop (K2) base + the same online mean correction at the 0.1x floor
(scratch/cov_prop_corr.py: torch cov-prop with gain approx + correction hooks; correction
gets only local Gaussian features {1,mu,var,sigma,alpha,|alpha|,phi,Phi,pred} -- D3/D21/K4
are ABSENT without K3 tracking). RESULT: cov-prop raw 4.4e-6 -> corrected 4.4e-6 = 1.0x
(NO improvement, train and test). The local per-neuron correction captures ~nothing of
cov-prop's K3+ truncation error.
- WHY: the K3 content is a property of the propagated cross-neuron JOINT distribution (the
  off-diagonal 3rd cumulant) -- it is NOT visible in a single neuron's local Gaussian
  moments. Our K3-correction works (1.45x) ONLY because it has D3/D21/K4 features that carry
  the non-Gaussianity; strip those (cov-prop) and the correction is powerless. The correction
  only helps when the base already tracks the non-Gaussianity -- which is the expensive part.
- CONSEQUENCE: the 0.1x floor is UNREACHABLE for us with usable accuracy by ANY route tried
  (compression F46-UPDATE; cheap-base+correction here). The winners' mult-0.3/raw-2.5e-8
  operating point uses a method we do not have and cannot reverse-engineer (code sealed).
- STANDING after this session: only available improvement is the K3 ONLINE correction
  (F47, 1.45x -> ~rank 16 from #17). Everything else mapped is dead: learned OUTPUT
  correction, windowing, Tucker compression, cheap-base correction, and (per F42/F43)
  higher kprop modes (K3-simple is the best feasible; K4 is 1277x budget). TOP-10 requires
  a genuinely more FLOP-efficient 3rd-cumulant closure -- research-level, likely from
  studying the companion paper arXiv:2605.05179; not achievable by bolting onto K3-simple.

## F48 (2026-08-31) — Full read of companion paper 2605.05179 supplement: three unexplored
## levers (augmented-K3 h2(k4) tracking, score-driven term selection, symmetric FLOP
## accounting) + depth-robustness is an OPEN problem we are already attacking.
Read the entire technical supplement (S.1-S.8) + appendices C-J for the first time.
Where we stand: our K3-simple IS the paper's factorized K=3 basic algorithm (S.4.3) with a
truncated 94-term nonlin set. Exact factorized K3 costs 30n^3L^2+39n^3L ~ 8.9e12 FLOPs
(Table 1, symmetric-kernel counting) = 4.05x budget at n=1024 L=16 -> exact K3 would score
WORSE than our truncation (mult ~4 kills it). The frontier must also run a truncated/custom
closure. Unexplored items, ranked:
- (1) AUGMENTED K3 (S.4.2): basic K3 keeps only the SCALAR harmonic h0 of kappa4 (s_K:
  r=K+1 -> s=r/2). Augmented additionally tracks h2(kappa4) = the traceless-MATRIX harmonic,
  an n^2-sized state that contributes Omega(n^-K) MSE and propagates in O(n^K). Paper's
  factorized-augmented = 72n^3L^2 (2.4x cost, too rich), but an "augmented-lite" (h2(k4) as
  an explicit n x n matrix: linear step is one n^3 congruence like C, nonlin terms n^2 each)
  could be a CHEAP accuracy boost. This is the only paper mechanism between K3 (ours) and
  K4 (1277x budget, dead per F42/F43) — the natural candidate for frontier raw 2.3-2.5e-8.
- (2) SCORE-DRIVEN TERM SELECTION (eq. 14, k(nu)<=K grading): every diagram term has a
  provable n-growth grade; our 94-term SIMPLE set was inherited, never optimized for the
  metric MSE x FLOPs. Section 4.2 explicitly blesses adapting WHICH cumulant parts to track
  to the budget "beyond merely choosing a different value of K". Two-sided: prune terms
  whose MSE contribution < FLOP share; selectively add k(nu)=4-grade terms where cheap.
- (3) SYMMETRIC-TENSOR ACCOUNTING (App J): symmetric kernels need only beta_{3,inf}=7/18
  ~ 0.39 of naive FLOPs for order-3 contractions; frontier mult band 0.28-0.48 ~= 0.39-0.58
  x our 0.833. ACTION: audit the port — we use flops.as_symmetric for C but must verify
  every symmetric intermediate + einsum path is priced at distinct-entry cost. Also
  footnote 14: entrywise ops in the nonlin step are a big FLOP share -> shared-subexpression
  restructuring of the 94-term evaluation cuts C. Solomonik-Demmel 2015 (fewer mults for
  symmetric contractions) is cited as even better than 7/18 — if used, email sponsor first
  (fair-accounting rule; a genuinely different algorithm should be legit, get it in writing).
- (4) DEPTH-ROBUSTNESS IS OPEN (App D + section 7.3): paper proves/conjectures MSE ~
  c_K (L/n)^K, i.e. error grows L^K with depth, and explicitly leaves "error that does not
  increase with depth" to future work. Our F46/F47 self-propagating-mean-error analysis +
  online per-layer correction is a genuine attack on exactly this open problem ->
  WRITE-UP-WORTHY framing; also justifies pushing the correction beyond 1.45x.
Confirmed DEAD by the paper (no new effort): K4 in any form (Table 1: 24n^4L^2 ~ 3000x
budget); MC hybrids at our budget (MC noise floor ~1.5e-5 >> our bias 5e-8); polynomial
algorithm (superseded by Hermite version); GELU/tanh + low-prob + distillation machinery
(different tasks). Verify-cheap: confirm our port does the harmonic projection step (2d)
each layer (junk-component removal is free accuracy if missing).

## F49 (2026-08-31) — Audit: flopscope billing is TIGHT (no accounting lever exists);
## augmented-lite v0 (anisotropic K4 vector + (W*W) transport) WORKS: stack with
## correction = 1.44x on the ported V1.6 config at unchanged mult 0.833.
AUDIT (closes F48 item 3): per-op breakdown of the submitted port (audit_flops.py):
einsum 59.4% (1.089e12, dslices H/T2/T3/T4 ~ 2kn^3 each) + matmul 40.4% (7.41e11, stack
propagation 6kn^3) = 99.9%; entrywise ops 0.04% (paper footnote-14 worry does NOT apply).
Probing declarations vs theoretical floors (audit_flops2.py): every dominant op bills at
ratio 1.00 -- flopscope prices optimal contraction paths; C_pre already collects its
symmetric-C discount (~3.2n^3 vs 4n^3). The beta_{3,inf}=7/18 App-J hypothesis is REFUTED
for our FACTORIZED representation (symmetric-kernel savings apply to the unfactorized n^4
algorithm; factorization already removed that redundancy -- flopscope's 1/|G| discounts
can't touch unsymmetric batched matmuls). M-CHAIN ABLATION (43% of cost): DEAD -- MSE x45
for a 1.7x cost cut (DS-shadowing funnels essential content into the exact M chain).
CONSEQUENCE: no cost lever exists inside the current algorithm; frontier mult ~0.25-0.42
(2026-08-31 board) is a structurally cheaper method, not billing technique.
AUGMENTED-LITE (F48 item 1, first rung; k3_aug_lite.py, lean_k3.py flags no_k4 / k4_vec /
k4_prop / no_m_chain):
- no_k4: killing the K4 feedback channel = 7.6x worse -> channel has real headroom.
- vec (anisotropic birth, NO transport): 6.6x worse. Source-basis anisotropy used in the
  next layer's rotated basis is poison; isotropic averaging was doing real work.
- vec+ww: per-neuron K4 vector gv = n*P2*(c4*K4v + c22*K22.sum(0)) (mean-parity with the
  scalar), transported by (W*W)@gv/metric_c, fed as anisotropic wk4_row and
  wk4_mat=(g_i+g_j)/2: 1.069x BETTER, uniform 8/8 MLPs, cost ~n^2/layer (FREE).
- mat (full K22 matrix via (W*W) sandwich, +3.7% mult): 0.68x WORSE -- the omitted n^4
  cross-pairing terms act as isotropizers; raw off-diagonal transport is net harmful.
  Off-diagonal kappa4 needs the true (2,1,1) diagram terms (F48 item 1 rung 2), not a
  cheap transport hack.
STACK (k3_aug_corr.py, correction refit per config on teacher-forced V1.6 trajectory,
train 0-4 / test 5-7): V16 4.63e-8 -> +corr 3.35e-8 (1.38x) -> +vecww+corr 3.23e-8
(1.44x total; vecww adds 1.04x post-corr -- the corr's K4 feature already captured part).
Beta saved: scratch/k3_augcorr_beta.npy. Projected public: raw 5.37e-8/1.44 ~ 3.7e-8 ->
adj ~3.1e-8 ~ rank 21-22 (of 83). NEXT: port vec+ww (+~n^2/layer) + correction (13
dot-products/neuron/layer) into estimator.py, re-verify parity + 5s/120s caps, submit.
Top-10 still requires the structurally cheaper closure (F48 items 1-rung-2 / 2).

## F50 (2026-08-31) — GRADER SMOKE TEST USES NON-SUITE MLP SHAPES (depth 32 confirmed);
## shape-specific code must be gated. Port of vecww+corr shipped.
Context: first submission of V1.6+vecww+corr (#329207) FAILED at the grader smoke stage
with ESTIMATOR_EXCEPTION despite whest validate + parity + 100-MLP mini all passing
locally. Failed smoke = no quota cost (page: "nothing counted against your quota").
Traceback is organizer-only in schema 2.0 pages — diagnose by replaying the packaged
tarball locally on candidate shapes.
- ROOT CAUSE (reproduced exactly): the P2 grader smoke test runs a DEPTH-32 MLP (P1
  grader config, F11, still live). The submitted estimator indexed the embedded 16-row
  correction table by layer -> IndexError at layer 16. Local `whest validate` smokes at
  (2,4) only — depth-32 never exercised locally. Replaying the extracted tarball on
  32x256 raised the identical IndexError; 16x1024 ran fine even on flopscope 0.12.1.
- FIX (verified): gate BOTH riders (vec+ww transport + mean correction) on the exact
  fitted suite shape (n==1024 and L==16). Any other shape runs the shape-generic V1.6
  base path that already passed the smoke as #328476. Estimator now clean on 32x256,
  20x128, 48x512, 2x4.
- BONUS FINDING: V1.6 (old AND new) produces non-finite output at 16x64 (SymmetryError
  deviation=inf inside flopscope) — the K3 truncation diverges at small width x depth 16.
  Pre-existing, not smoke-relevant (smoke passed for #328476), but remember: NEVER assume
  the estimator is shape-robust below n~128.
- Same-object triple-operand einsum ("ij,ij,j->i", W, W, v) did NOT crash flopscope
  0.12.1 in isolation or at suite scale — F45's "avoid" stands as policy, but it was not
  the cause here; replaced with (W*W)@v anyway. fnp.sum(axis=) works but was swapped for
  matvec-with-ones forms to minimize novel-API surface against grader version skew.
- Toolchain: starterkit venv upgraded flopscope 0.12.0 -> 0.12.1, whestbench 0.16.0 ->
  0.16.1 (uv lock --upgrade-package; plain uv pip install gets reverted by uv run sync).
  Parity/cost identical under 0.12.1 (1.8323e12 = 0.8332x B, residual ~0.32 s).
- PORT RESULT (local, v2-phase2 public mini 100 MLPs): adjusted 3.50e-8, raw 4.20e-8,
  mult 0.8332, 0/100 failed = 1.31x better than #328476's 4.60e-8 local (LB 4.47e-8).
  Riders cost ~+1e8 FLOPs (invisible) and ~+10 ms residual.
- TRAP RE-CONFIRMED (was already in the 07-11 log): `whest run` WITHOUT --dataset makes
  synthetic MLPs with a 200k-sample MC reference -> noise floor ~4e-7 swamps any good
  estimator. Always pass --dataset hf://aicrowd/arc-whestbench-public-2026 --revision
  v2-phase2 --split mini.

## F51 (2026-08-31) — TRUE (2,1,1) KAPPA4 CHANNEL PORTED (augmented-K3, r=1 matrix
## core): raw MSE 2.2e-8 on P2 dumps = 2.1x better than V1.6 base, AT FRONTIER RAW
## LEVEL. Exact parity with reference AUGMENT. Cost is now the whole battle.
Context: rung-2 derivation session (F48 lever 1 / F49 conclusion "off-diag kappa4 needs
true (2,1,1) diagram terms"). Reference repo already implements the target as
kind=AUGMENT at k_max=3: get_r_x gives K4 = Sym(G x metric) with an (n,n) MATRIX core
(h2 channel) instead of SIMPLE's scalar radial core (r=2). factor_k3.py's augment path
was the derivation source.
- REFERENCE CANNOT RUN AT n=1024: kind=AUGMENT materializes the (2,1,1) pK slices as
  dense n^3 tensors (8.6 GB f64 each); OOMs at 64 GB even in f32. The lean port is the
  only way to run this channel at suite scale — it needs NO n^3 object anywhere:
  - USE side (dslices of Sym(G x m*I), verified exactly): (4,)=m*diag(G);
    (2,2)=(m/6)(G_aa+G_bb)  [only diag(G) — explains why vecww worked];
    (3,1)=(m/2)*G [the genuinely new off-diag channel]; (2,1,1)=(m/6) 1_a x G_bc
    (rank-structured); (1,1,1,1)=0.
  - PROD side: the r=1 harmonic projection only ever needs the (2,1,1) slice SUMMED
    OVER ITS 2-BLOCK AXIS (L^1 cap). Every (2,1,1) term = product of <=2 broadcast
    matrix factors x per-axis wick vectors -> capped sum is a single matmul (einsum
    over 'a' with a!=b, a!=c exclusion corrections, all n^2). Projection coefs:
    G += 6/(n+4)*(diag K(4,) + diag rowsum K(2,2) + K211_cap) + 12/(n+4)*Sym(K(3,1))
    - 3/((n+2)(n+4))*(sum K(4,) + sum K(2,2))*I, plus hub contractions pK211->K211
    (coef 6/(n+4)) and pK111->K211 (coef -12/(n+4)), plus K4->hub feed sources
    (reference augment terms 1&3; term 2 vanishes for diagonal avg metric).
- PARITY: lean_k3_aug.py (flag k4_aug) == reference AUGMENT to 1e-16 per layer on mu
  AND G (n=48). TRAP FOUND: the reference's proj_coef builds torch tensors at default
  dtype f32 -> its projection coefficients carry ~3e-8 relative noise (e.g. 0.5 comes
  out 0.5000000149 via f32(1/24)*12). Set torch.set_default_dtype(torch.float64)
  before comparing, else parity stalls at ~5e-9. Our lean uses exact f64 rationals
  (MORE precise than the reference).
- MSE (final layer): 16x256 local MC dumps: SIMPLE 6.3e-6/4.5e-6 -> AUG 1.4e-6/1.1e-6
  (4.3x). P2 dumps 16x1024: AUG raw 2.08e-8/2.00e-8/2.39e-8 vs V1.6 base ~4.6e-8 =
  2.1x, BEFORE any corr-rider refit. Frontier raw is 1.8-2.9e-8 -> the augment channel
  alone reaches frontier-level raw accuracy.
- CANCELLATION STRUCTURE: dropping both hub contractions (no_a+no_b) EXPLODES MSE to
  1e-1 — pk211_cap production and the (1,1,1)-block contractions carry large mutually
  cancelling content. They are one unit; pruning must respect the pairing.
- SOURCE TRUNCATION IS DEAD: contracting only the last-k hub sources (with consistent
  subset rep-correction — naive full-rep subtraction diverges outright) degrades badly
  even at k=8 of 16 (3.3e-5 vs 1.4e-6 at n=256). Old sources do NOT decay; the K211
  contraction is global. No cheap window.
- COST (flopscope estimate, n=1024): contractions (a)+(b) ~11 n^3-matmuls per source
  per layer x ~136 source-layers x2 ~ 5.7e12 FLOPs ~ 2.6xB on top of the existing
  0.83xB + capped-prod ~0.2xB -> ~3.7xB naive. Score would be 2.16e-8*3.7=8e-8: LOSES
  to the current ship (3.67e-8) despite 2x raw gain. Break-even needs mult <= 1.7.
  Identified levers: (1) EXACT merged-matmul restructure of k211_from_sources — the
  cap-X rotations share P: P @ (sum_pairs diag(c_i) Y_i^T), ditto cap-Y — ~2x, no MSE
  cost; (2) LOW-RANK FACTORED HUB LEGS (n x R factors): turns dslices AND contractions
  from n^3 into n^2 R — the structural bet that matches frontier mult ~0.25; (3)
  per-term MSE-vs-FLOP pruning of the 52 capped-211 production terms (F48 lever 2).
- Files: scratch/lean_k3_aug.py (port, flags k4_aug/aug_no_*/aug_recent_k/debug_state),
  scratch/verify_aug_formulas.py (dslice+projection coef verification),
  scratch/aug_bisect.py, aug_bisect2.py (piecewise parity vs reference internals),
  scratch/gen_midsize_dumps.py + mlp_dumps_mid/ (16x256 MC dumps, N=3.2e7),
  scratch/aug211_explore.py, aug_ref_dump.py.
- WRITE-UP ANGLE (stretch): the capped-slice evaluation scheme (never materializing
  the n^3 (2,1,1) slice; exclusion-corrected einsum contraction) is exactly the kind
  of algorithmic contribution ARC's P2 design hint points at, and it is what makes
  kind=AUGMENT feasible at width 1024 at all.

## F52 (2026-08-31) — AUG COST ATTACK: merged-matmul contraction (exact 12->5/source)
## + LOW-RANK FACTORED HUB SOURCES (gram basis, age-tiered ranks). geo_hirank
## schedule (2:768, 4:384, 8:192) matches dense-aug MSE (2.22e-8 vs 2.16e-8 on
## 3 P2 dumps) while factoring 13+ of 16 sources per layer.
Context: F51 levers 1-3 executed. All in scratch/lean_k3_aug.py (flags fac_schedule,
fac_basis, t211_drop); merged k211 verified exactly vs the naive triple loop
(test_k211_merge.py, 40 cases, ~1e-16).
- MERGED CONTRACTION (exact, free): in k211_from_sources the s-weighting factors out
  as diag(s) core diag(s), and per source all left-P rotation terms collapse into
  ONE matmul P @ R_acc, all right-P terms into L_acc @ P^T; only the 3 middle
  rotations Y_k diag(P^T u) X_k^T stay. 12 -> 5 matmuls per full source per call.
  End-to-end parity unchanged (reference AUGMENT 4.4e-16; P2 dump MSEs identical);
  local runtime 40 -> 30 s/MLP. Counted cost of dense aug (FlopCounterMode, 2*M*N*K
  convention): 7.07e12 vs base-full 3.13e12.
- FACTORED SOURCES (the structural bet): every leg of a source = T @ birth_leg with
  T = P the shared propagator, so ALL legs live in colspace(P). At age >= A project
  legs onto a rank-R basis Q: store Q (n x R) + factors Q^T leg (R x n). Transport
  becomes ONE n^2 R op per source (only Q moves; wick = row-scale of Q); dslices
  re-associate through the factors ((Xm*Pm) @ Fy^T) @ Q^T -> all n^2 R; the k211
  contraction collapses per source to an R x R core with a single Q G Q^T
  reconstruction. Exact for the factored rep; the only approximation is truncation.
- BASIS MATTERS: top eigenspace of the leg Gram (sum leg leg^T) >> top left singular
  space of P (A1:R128@n256: 1.53e-6 vs 2.52e-6; dense 1.44e-6). Always use gram.
- SCHEDULE (age-tiered, rank halves per age doubling; P2 dumps, dense = 2.08/2.00/
  2.39e-8): geo_hirank (2:768, 4:384, 8:192) -> 2.10/1.98/2.56e-8 = 1.03x dense
  (NOISE-LEVEL). geo_hi (2:512, 4:256, 8:128, 16:64) -> 1.34x. A4:R256 -> 1.19x.
  Tails below 128 hurt (8:64 tier -> 1.7x); first tier below 512 at age 2 hurts
  badly ((2,512),(4,128),(8,64) -> 3.5x). MSE cliff sits around R/n ~ 1/2 at age 2,
  ~1/8 at age 8.
- TRAP: the 16x256 midsize dumps CANNOT discriminate this lever — their aug bias
  floor (~1.4e-6) masks truncation error; geo_hi looked lossless at n=256 but is
  1.34x on real P2 dumps. Factorization sweeps must run on 16x1024 dumps.
- shadow_B2 is DEADLY in aug mode (6x MSE: 1.3e-7): the X2/Y2 hub block feeds the
  (2,1,1) contractions; opposite of the V1.6 lesson where shadowing B2 was the ship
  config. Do NOT reuse V1.6 shadow flags for the aug estimator.
- CAPPED-211 PRUNING (28 terms, 10 of them n^3-einsum): LOO on top of a fac schedule
  finds the wk4-operand terms droppable; combined drop of 12 terms {11,12,17,18,19,
  21,22,23,24,25,26,27} (6 n^3 terms) saves 60% of capped-production FLOPs at
  ~1.03x MSE (n=256) — P2 validation pending. Early terms are cancellation-locked
  (term 2 LOO = 6.6e5 x, term 0 = 91x): the drop set must be validated as a SET.
- COMBINED CONTRACTION (exact, halves the contraction phase): every rotation term
  is LINEAR in the cap vector u, and post-wick legs = diag(w1) pre-wick legs, so
  cA*contrib_a - 2cA*contrib_b(old sources) = cA diag(w1) core(u_a - 2 w1*u_b)
  diag(w1): ONE heavy contraction per layer instead of two. Birth source + rep/
  M_t1 corrections handled separately (all n^2). Parity 4.4e-16 vs reference.
- DEFERRED-Q DSLICES (exact): all factored-source sandwich terms share the same
  trailing @ Q^T -> accumulate (n, R) inner, one final matmul per source (5 -> 1).
- COST (FlopCounterMode 2*M*N*K, n=1024; CORRECTED calibration base = V1.6 flags
  2.0981e12 counted = 1.8323e12 fs -> scale 0.873, i.e. flopscope ~ 2*MNK with
  ~13% symmetry/discounts — the naive 0.586 scale from full-base was WRONG):
  aug dense 5.81e12 counted (est mult 2.31); hirank+prune+gram 4.19e12 (1.69);
  hirank+prune+gram4 3.95e12 (est mult 1.59); mid640+prune 3.73e12 (1.50);
  geo_hi+prune 3.29e12 (1.32).
- gram4 basis (Gram from X1/Y1/M/P only) == full gram MSE on P2 dumps, saves
  ~0.25e12 counted. rand1 (one-pass subspace iteration, fixed sketch) == exact
  eigh MSE within 3% -> the flopscope port needs only matmul + qr. rand2 buys
  ~1% more. age-1 first tier is DEAD (1.5x MSE).
- CORR RIDER IS NEARLY DEAD ON AUG (key negative result): refit on the aug+fac
  trajectory gives 1.05x train / 1.01x test (vs 1.38x it gave V1.6) — the true
  (2,1,1) channel already captures the bias the linear correction was mopping
  up. 8-dump raw mean (no corr) 1.99e-8 (train 2.10e-8 / test 1.80e-8).
- 16-tier is a no-op at L=16 (max source age = 15): schedules need no 16+ entry.
- PORT (estimator_aug.py in starterkit): full aug+fac path in fnp; tables from
  dump_aug_tables.py (126 dense + 16 capped specs, 33 wick pairs, unified wick
  form deg<=6 sigma^[-7,4]); gram4+rand1 truncation via fnp.linalg.qr with a
  setup-rng sketch; combined contraction; G tagged as_symmetric. Small-n parity
  vs lean = f32 drift (9e-7 @ scale 2.9). n=1024 parity/cost harness:
  parity_aug2.py (dense strict + ship MSE/flops/residual).
- PORT MEASURED (n=1024, real flopscope): dense config parity EXACT to f32
  (mse 2.044e-8 vs lean 2.043e-8, drift 3.2e-6). SHIP config: flops 3.8994e12
  = 1.7733xB (13% over the FlopCounter model — einsum-path/qr pricing overhead
  to audit), mse 2.129e-8 (== lean rand1 level; sketch differs, fine).
  Projected adjusted ~= 1.96e-8 * 1.773 ~ 3.5e-8 vs live ship 3.67e-8: +6% only
  as-is -> cost work (billing audit, rank-1 Y3 ~7%, mult target <=1.6) decides.
- RESIDUAL TIME IS THE BLOCKER: unfused port measures 2.9-5.3s vs the 0.400s
  cap (~19k op submissions at ~0.3 ms each; V1.6 ships ~1k ops = 0.31s). Fix
  plan (exact, next session): fnp.take-gathered term stacks (126-term dense
  production -> ~8 ops), tier-batched factored-source stacks via broadcast
  matmul (per-tier (k,n,R) Q-stacks), signature-grouped capped einsums,
  fused dense window. Corr beta embedded (gated n==1024 & L==16).
- Files: scratch/lean_k3_aug.py (all levers), test_k211_merge.py, fac_sweep.py,
  flop_count.py (FlopCounterMode cost model + truncation-event log), prune_211.py,
  k3_augfac_corr.py, dump_aug_tables.py, dump_lean_aug.py, dump_lean_small.py;
  starterkit: estimator_aug.py, k3aug_tables.py, parity_aug2.py, xcheck_aug.py,
  smoke_aug.py, api_probe.py.

## F53 (2026-09-01) — RESIDUAL BROKEN: flopscope residual time is dominated by
## RESULT-BUFFER ALLOC/FREE CHURN, not op count; out= into pooled buffers gives
## a flat ~0.02ms/op regardless of size. Fused aug port: 2.9-5.3s -> 0.28s
## (cap 0.400), 32.5k -> ~10.5k ops, ship mult 1.8116xB, MSE parity intact.
Context: estimator_aug2.py = full fused rewrite of the F52 aug+fac port.
- RESIDUAL MECHANICS (the key insight): residual = wall - backend - overhead
  (flopscope _budget.py). The numpy call itself (incl. first-touch of the
  result) is backend; what lands in residual is the python-side ALLOC/FREE of
  result buffers between ops. Measured per-op residual, clean inputs:
  fresh (n,n) 0.092ms, fresh (8,n,n) 0.71ms, fresh (140,n,n) 12ms -- but with
  out= into a preallocated buffer: 0.014-0.03ms FLAT at any size. So residual
  engineering == eliminating fresh allocations, not (only) fusing ops.
  TRAP: benchmark probes must build inputs OUTSIDE the BudgetContext -- raw
  numpy work (np.random etc) inside the context is counted as residual and
  fakes a size-dependent per-op cost.
- fnp.zeros/ones/eye/diagflat return SYMMETRY-TAGGED SymmetricTensors; a
  tagged out= buffer raises "out symmetry does not match result symmetry" on
  plain results. fnp.empty is untagged -> all pool buffers use empty (never
  do arithmetic on uninitialized data: 0.0*garbage can be NaN; zero with
  fnp.copyto(buf, 0.0)). as_symmetric costs 23.6ms WALL per call (0.73s over
  a run) vs ~1.4% flops saved on C_pre/G einsums -> dropped both.
- FUSION ARCHITECTURE (estimator_aug2.py): offline schedule compiler turns
  TERM_SPECS_AUG + capped substitutions + hub builds + PK2K into two stacked
  elemwise rounds per layer (R1 ~140 terms, R2 ~20; stack views -> 1 op at
  1 flop/elem, slice-sums per output); capped (2,1,1) main terms grouped by
  normalized index signature into batched k-einsums ((bc,ac) n^2-priced,
  (ac,ab) true n^3); a->b/a->c substituted terms become elemwise stack terms
  (zero-diag operands make most of them identically zero -> dropped at
  compile); factored tiers hold Q (k,n,R) + static F (k,R,8n) blocks in
  FIXED kcap-sized slot buffers (FIFO append/compact via copyto; transport
  + wick scale ping-pong between two slot buffers); dslices/k211 use
  concat-free block matmuls (left blocks written straight into a (k,n,8n)
  buffer, right = transpose view of a static leg rearrangement).
- Getitem/transpose/matrix_transpose are FREE (views, 0 ops); fnp.stack of a
  python list of views is 1 op at 1 flop/elem (vs take at 4 flops/elem);
  fill_diagonal accepts a VECTOR value -> diagflat(v)+M patterns become one
  in-place fill_diagonal on a zero-diag matrix (exact, no (n,n) temp).
- Exact freebies found: last layer only needs pk[(1,)] -> skip d2 production,
  capped channel, G transport (G's diagonal via t=G@w, einsum('ia,ia->a',w,t)
  = 2 ops); the diagflat(md) term in D21_new is DEAD (rep21 is zero-diagged);
  C = fill_diagonal(pk11, K2v) in place. Last-layer skip ~ -2% flops.
- Schedule compile (~10ms/mask python) runs in setup() -- setup time is
  outside the residual budget; warm all 8 masks there (protects the
  per-MLP-process grader case).
- CHAOS, not bug: deep configs amplify f32 sum-order drift ~x300 at the last
  layer (n=64 L=17: a 1e-7 weight perturbation -> 3.2e-4 output change in the
  REFERENCE port itself). A/B parity tolerances must scale with depth.
- MEASURED (mlp_0000/0001 P2 dumps): ship config flops 3.9838e12 = 1.8116xB
  (old port 1.7733xB: -2% last-layer skip, +1.4% no as_symmetric, +~0.5%
  fusion tax + K211 exact-dead-term drops), mse_final 2.040/2.088e-8 == lean,
  residual 0.282/0.275s (was 2.9-5.3s), wall 24.6s (was ~37s). Dense-parity
  config residual 0.43s (never ships). estimator.py now dispatches:
  1024x16 -> estimator_aug2, anything else -> estimator_v16 (= old
  estimator.py; grader smoke = depth-32, F50). 16x64 v16 blowup pre-existing.
- Files: starterkit estimator_aug2.py (fused), estimator.py (dispatcher),
  estimator_v16.py (fallback copy), profile_ops.py (op/gap profiler),
  probe_fusion.py, probe_fusion2.py, probe_residual.py, ab_check.py
  (A/B parity vs estimator_aug).

## F54 (2026-09-01) — THE FLOP BUDGET IS A HARD CAP: C_m > B_m = ZERO-prediction
## fallback for that MLP, NOT a >1x score multiplier. The F51/F52 cost planning
## ("mult 1.77", "need mult <= 1.7 to break even") misread the metric; the real
## mini run failed 100/100 MLPs on "exhausted FLOP budget" at 1.81xB.
- The rule was in our own docs all along (competition_overview.md: "Failure
  fallback: over-budget ... -> zero prediction substituted, no discount";
  CLAUDE.md failure-modes list). max(0.1, C/B) only DISCOUNTS below budget --
  which also explains the frontier: mult 0.25-0.42 entries are buying score
  linearly by being far UNDER budget, not merely being cheap.
- Consequence: the aug+fac ship config (raw ~2.0e-8 at 1.81xB) is UNSHIPPABLE
  until total C <= 2.199e12. If it fits at C ~ B it scores ~2.0-2.2e-8 = 1.7x
  better than our live 3.67e-8 (top-10 zone); every further cut multiplies.
- SAFETY: estimator.py REVERTED to the V1.6 submission code; the aug dispatcher
  parked as estimator_dispatch.py (whestignored) until the cost fits. NEVER
  submit the aug path before a real-mini pass shows 0 budget failures.
- Fused-estimator FLOP map at 1.812xB (mlp_0000, categorized op_log):
  tier legs/inner (Q@F recon + block matmuls) 1.27e12 (32%); dense-age source
  matmuls 8.7e11 (22%); batched W@ transports 6.0e11 (15%); 2d matmuls
  (factorize gram/proj + k211 dense) 4.7e11 (12%); tier (k,R,n) family 3.8e11
  (9.5%); C/G transport einsums 1.3e11; capped einsums 1.2e11; qr 5.2e10.
- Cost-attack levers for the ~1.8x cut (next session, lean-first on P2 dumps,
  score objective raw*max(0.1, C/B) with C<=B hard): (1) age-1 first
  factorization at HIGH rank (F52 only killed age-1 at low rank; dense phase
  + its transports ~ 1.4e12 = 35%); (2) Y3 is exactly rank-1 forever (c*1^T
  transports/scales to rank-1) -> drop a full leg from every block (~12% of
  source machinery); (3) rank ladder re-sweep under the corrected objective;
  (4) shared-Q per tier (power-iteration convergence argument; kills per-source
  reconstruction + transport); (5) more capped-term pruning; (6) einsum-path
  billing audit (real 3.98e12 vs FlopCounter model 3.45e12).

- F55 (2026-09-01) — Rank-ladder (FAC_SCHEDULE) is a DEAD lever for the cost cut:
  the mse-vs-C frontier is ~10:1 against us. Coarse sweep (cost_sweep.py, 2 P2
  dumps, uniform + age-1 ladders): ship (768/384/192)=1.77xB @ mse 2.06e-8; only
  configs under B are u-0.25 (192/96/48)=0.99xB @ mse 3.47e-7 and a1-erl
  (1,384..)=0.94xB @ mse 3.73e-7 -> score ~3.4e-7, i.e. 10x WORSE than live best
  3.67e-8. Cutting ranks trades ~17x accuracy for 1.8x compute. Age-1 confinement
  (Next lever 1) is parametric (pure FAC_SCHEDULE, no code change) but no better on
  the frontier. CONCLUSION: no parametric point beats our current submission;
  shipping any under-B rank config would LOWER our rank. Corrected ordering: do the
  EXACT/structural cuts first (Y3 rank-1 tracking ~12%, billing/declaration audit
  ~13% incl. norm.pdf/cdf float64 2x over-bill, shared-Q, dead-term pruning) to
  shift the whole curve left at ZERO accuracy cost, THEN a mild rank trim for the
  last mile. Target: C<=B at mse ~2-2.5e-8 (score < live 3.67e-8). Rank ladder is
  the LAST lever, not the first. (Frontier: runs/cost_sweep_coarse.log.)

- F56 (2026-09-01) — Billing audit: NO reclaimable over-billing in AUG. Per-op
  breakdown (audit_aug.py, ship 768/384/192, 3.90e12): matmul 91.6% (3.57e12,
  4839 calls), einsum 6.6%, qr 1.3%; norm.pdf/cdf 0.00% (float64 2x warning is a
  non-issue). Standard matmul/einsum/stack patterns bill at ratio 1.00 vs
  theoretical floor (audit_flops2.py) -> flopscope is NOT over-billing; F54's
  3.98 vs 3.45e12 gap was our FlopCounter UNDER-modeling, not waste. matmul shape
  profile (prof_matmul.py, matrix-matrix only): 670 full-dense (1024^2)@(1024^2)
  = 1.44e12 (age-0/1 dense source transports + cov prop, biggest lever); rest is
  rank-R (768/384/192) per-source reconstruction+transport ~1.7e12. Only semi-free
  cut: declare the ~32 symmetric cov/cumulant sandwiches (C'=WCW^T) as_symmetric
  (~3%). CONCLUSION: cost is real work; C cut must be ALGORITHMIC (fewer/smaller
  matmuls) = structural levers only: Y3 rank-1 (drop a leg, exact), shared-Q per
  tier (collapse per-source transports), shorten dense phase. No quick config win
  exists. (audit_aug.py, prof_matmul.py, runs breakdown.)

- F57 (2026-09-01) — Shared-Q per tier: ACCURACY HYPOTHESIS CONFIRMED. Replacing
  each source's own rank-R basis with ONE shared basis per age-tier (built from the
  summed leg-gram, newcomers projected onto the resident basis dropping out-of-span
  residual) holds mse: 2.34e-8/2.32e-8 on dumps 0/1 vs per-source baseline
  2.04e-8/2.09e-8 -- only +15%, well under the 3.67e-8 live-best win line
  (estimator_aug_sq.py). Aggressive floor (first/oldest member fixes the tier basis,
  no rebuild); a periodic rebuild would recover some of the +15% if needed. BUT
  transport-only sharing is net-neutral on C (1.81xB, +2%): per-join QR
  re-orthonormalization overhead cancels the collapsed W@Q transports. Real cost
  win requires (a) attacking the 670 full-dense n^3 matmuls (1.44e12=37% of C, the
  age-0/1 unconfined source transports) via EARLY (age-1) shared confinement --
  which per-source confinement couldn't afford (sweep a1-hi mse 3.7e-8) but shared-Q
  can; and (b) collapsing the linear final reconstructions (_k211 Q@(Gc@Q.T),
  _dslices inner@Q.T) per-tier. Intermediate Xm=Q@Fx inside Hadamard products in
  _dslices do NOT collapse. Next: minimize _factorize re-orth overhead, sweep
  early-confinement shared schedules for min(C/B) s.t. mse<=~3e-8.

- F58 (2026-09-01) — CONCLUSIVE: augmented-K3 is NOT shippable under 2^41; the
  budget gap is structural to the method, not fixable by cost engineering. After
  shared-Q (accuracy validated, F57), all accuracy-preserving levers exhausted:
  (a) reconstruction-collapse of the linear trailing Q@(Gc@Q.T)/inner@Q.T is EXACT
  but only ~3% (1.812->1.760xB); (b) early (age-1) shared confinement DEAD (raises C
  via tier overhead or mse->1.5e-7; young sources must stay dense); (c) lazy re-orth
  impossible (QR bounds basis norm, removing it -> inf over 16 transports). Trimmed
  shared-Q frontier floor = 1.293xB @ mse 1.1e-7. At mse tying live best (3.67e-8),
  C=1.47xB (47% over); forcing C=1.0xB -> mse 3-5e-7 -> score ~10x WORSE. Structural
  floor: 670 dense young-source n^3 transports alone ~0.65xB, unconfinable. aug's
  2.04e-8 accuracy exists ONLY at 1.77xB and can't be bought fractionally under B.
  DECISION: abandon the aug cost attack. #329212 (V16 K3, 3.6730e-8, 0.83xB) stays
  best. To climb, need a DIFFERENT lever -- most promising: spend V16's budget
  HEADROOM (0.83xB now, ~0.17xB spare to B, floor at 0.1xB) on CHEAP accuracy gains
  (partial kappa4 / variance reduction costing <0.17xB), since V16 is the only
  operating point that is under budget WITH room. Files: estimator_aug_sq.py,
  sq_sweep.py, runs/sq_sweep_l1.log, runs/sq_sweep_l2trim.log.

- F59 (2026-09-01) — LEADERBOARD reality check: we are #26/91, DOWN 3 (23->26) in
  ~20h with an unchanged 3.67e-8 while the board churned (Camaro +25 to #15 in 1h;
  J2W/oqaris submitted within 3h). Decisive finding from the top rows: leaders win
  on BOTH axes via a structurally CHEAP method. Implied multiplier = adjusted/raw:
  J2W 4.90e-9/2.27e-8 = 0.22xB; Puffi 5.90e-9/2.26e-8 = 0.26; marius 7.40e-9/
  1.86e-8 = 0.40; jamespayor 0.48. US: 3.67e-8/4.41e-8 = 0.83. They hit ~2e-8 raw
  mse at 0.2-0.4xB; we hit 4.4e-8 at 0.83xB. Our aug needed 1.77xB for 2e-8 -> we
  were on the WRONG cost tier entirely. The winners are NOT a tuned version of our
  Wick/kappa4 expansion; they run cheap analytic covariance/cumulant propagation
  (the family ARC said P2 favors) then iterate hard (J2W 88 entries). STRATEGIC
  PIVOT for next session: stop trying to cram an expensive term-expansion under
  budget (aug dead F58, V16 headroom-squeeze is polishing the wrong tier). Build/
  port a genuinely cheap propagation method targeting ~0.2-0.4xB at ~2e-8 raw mse.
  Re-read the P2 cov-propagation baseline (4.1e-6 in docs) and companion paper
  arXiv:2605.05179 for the efficient operating point. Board is live-competitive;
  every idle day costs rank.

- F60 (2026-09-01) — CHEAP COVARIANCE PROPAGATION IS CAPPED ~4-5e-6; there is NO
  cheap perturbative route to the frontier. Systematic test on the 8 P2 ground-truth
  dumps (torch f64; scratch/skew_covprop.py, covprop_diag2.py, var_ceiling.py):
  * baseline covprop (gain approx, ~0.024xB): final mse 4.4e-6 (matches ex03/K2).
  * marginal skew (W^3 @ s3 diagonal transport): NO help (co-skewness dominates,
    same failure mode as F58's cross-term argument).
  * leading co-skewness -> Gram-Charlier mean correction (1 extra n^3 matmul/layer,
    formula M3[i,j,k]~=g2_i g1_j g1_k C_ij C_ik +2 perms; s3'_a=3 sum_i (W o g2)
    R[a,i]^2, R=(W o g1)@C): only -4%.
  * EXACT off-diagonal ReLU covariance via Mehler/Hermite series (Cov_post[i,j]=
    sum_k C_ij^k/k! w_k[i]w_k[j]; w1=Phi,w2=phi/sig,w3=|a|phi/sig^2,w4=phi|a^2-1|/
    sig^3): only -6%, CONVERGES at order 2 (orders 3,4 add nothing). Gain approx was
    already fine; off-diag covariance is NOT the error.
  * learned per-layer ridge correction on pure-K2 state (cov_prop_corr.py): ~0%
    (K2 state carries no non-Gaussian info for the correction to latch onto).
  DIAGNOSTIC (covprop_diag2.py, MC ns=3e5 vs covprop per-layer pre-activation
  moments): dominant systematic error is VARIANCE UNDER-ESTIMATION growing
  monotonically with depth (-0.04% L1 -> -3.75% L16); skew NEGLIGIBLE (std-skew
  ~1e-3, why all skew work is dead); excess KURTOSIS grows to +6.6% L16 (the real
  non-Gaussianity). CEILING TEST (var_ceiling.py, force true MC var into covprop,
  measure final mean vs PRECISE dump gt): forcing true variance makes the mean
  WORSE (4.79e-6 -> 6.92e-6). The variance under-estimate is a BENEFICIAL ERROR-
  CANCELLATION against the leptokurtic marginal, not a fixable bug -> variance
  debiasing is DEAD too. CONCLUSION: Gaussian-closure covariance propagation is
  fundamentally ~4-5e-6; the 100-200x gap to the frontier (leaders raw 2.2e-8 @
  0.22xB; our aug 2.0e-8 @ 1.77xB) is irreducible marginal non-Gaussianity
  (kurtosis+) that only full cumulant tracking captures. The user's target
  "cheap covariance propagation @ 2e-8" is NOT achievable. Real levers remain:
  (1) cost-cut V16/K3 (raw 4.4e-8 @ 0.83xB, UNTESTED under aggressive cuts) toward
  ~0.25-0.3xB -> adjusted ~1.3e-8, top-10; (2) reverse-engineer the leaders'
  efficient kurtosis-aware closure (2.2e-8 @ 0.22xB).

- F61 (2026-09-01) — COST-CUTTING V16/K3 BY SOURCE-WINDOWING IS DEAD; the score is
  MINIMIZED at full V16. Direct measurement (estimator_v16_win.py + test_v16_win.py,
  4 P2 dumps, real flopscope C + HARD budget cap, mse vs precise final_means). The
  cost is entirely the (k,n,n) source-stack machinery: 3 transports matmul(WDb,.) +
  4 dslice einsums per layer ~= 7*k*n^3, k=live sources growing 1..15, sum_k~120 ->
  1.8e12 = 0.83xB. Capping live sources to a window W:
    full 0.833xB mse 3.88e-8 score 3.24e-8 (== shipped, harness validated)
    W=12 0.801xB 4.64e-8 3.72e-8   |  W=8 0.662xB 1.86e-7 1.23e-7
    W=6  0.552xB 4.35e-7 2.40e-7   |  W=4 0.414xB 1.02e-6 4.23e-7
    W=3  0.335xB 1.54e-6 5.14e-7   |  W=2 0.249xB 2.24e-6 5.58e-7
  Score rises MONOTONICALLY as cost falls: mse degrades faster than the C/B
  multiplier rewards, at EVERY window. Confirms the earlier k3_window_probe (on the
  un-cost-engineered reference) and F58's structural-floor finding, now on V16 itself
  with riders. DECISIVE COMPARISON: at W=2 (0.249xB ~= leaders' 0.22xB cost tier) our
  mse is 2.24e-6 vs leaders' 2.2e-8 -> a ~100x accuracy gap at EQUAL cost. The
  leaders are NOT running a cheaper version of our method; their closure is
  structurally ~2 orders more FLOP-efficient. Old sources do NOT decay enough to drop
  -- the third-cumulant memory is essential. CONCLUSION: both F60 (cheap covariance
  propagation) and F61 (cost-cut V16) are closed. The ONLY remaining lever to beat
  live 3.67e-8 is a structurally different efficient method (direction #2). Most
  plausible per P1 history (sampling variants won) + CLAUDE.md "hybrid analytic+MC"
  hint: covprop as a CONTROL VARIATE for a small MC estimate. covprop is ~0.999
  correlated with truth (raw 4e-6 error on O(0.4) means) -> variance-reduction factor
  1/(1-rho^2) ~ hundreds; ~1-3e4 antithetic samples (well under 0.22xB ~= 14k passes,
  but MC is the cheap-billing question) could reach 2e-8. NEXT: prototype
  analytic-base + MC-control-variate; measure the achievable (mse,C) frontier in
  flopscope (MC forward-pass billing is the crux -- one pass ~3.36e7 FLOPs).

- F62 (2026-09-01) — MC CONTROL-VARIATE HYBRID (direction #2 from F61) IS DEAD AT
  L=16, measured on 3 P2 dumps (scratch/cv_probe_p2.py, torch f32, N=32768, honest
  held-out ridge CVs; output in scratch/cv_probe_p2.out). The arithmetic:
  * per-neuron Var(h_L) = 0.079-0.096 (kernel theory: 1 - k_16(0) ~ 0.077, the
    ordered-phase collapse of iterated arc-cosine kernels). One forward pass bills
    3.355e7 FLOPs -> 6554 passes at the 0.1xB floor, 14418 at the leaders' 0.22xB.
  * pure MC at the floor: mse 1.2-1.5e-5 -> ADJUSTED 1.2-1.5e-6 (flat in N above
    the floor, F9). Variance reduction NEEDED: 33-40x just to tie live 3.67e-8,
    245-300x for the leaders' 4.9e-9. (P2 leaders beat the MC invariant by ~250x;
    P1's beat it by 3-6x (F9) -- P2's frontier is analytic, full stop.)
  * REALIZABLE levers (expectation exactly known): linear CV from x 1.18-1.20x,
    from h_1 1.31-1.35x, antithetic 1.10x, scrambled Sobol 1.37-1.48x, Sobol+CV(h_1)
    1.2-1.3x (does NOT stack, same mechanism as F13). Best case ~1.5x vs 33x needed.
  * ORACLE linear CVs from deeper layers (expectation NOT known -- would inherit
    the closure's mean error at that layer as bias): R^2 = 0.30 (h_2), 0.42 (h_4),
    0.64 (h_8), 0.84 (h_12), 0.92 (h_14), 0.96 (h_15) -> even the unrealizable
    h_15 oracle gives only 25-27x < 33x. And any nonlinear CV of h_l needs the
    law of h_l = the original problem.
  * WRITE-UP NUGGET: the R^2 ladder matches the linear-response kernel heuristic
    R^2(l) ~ prod_{j>l} kappa'(rho_j) * (1-rho_l)/(1-rho_L) with rho_l = k_l(0)
    of the iterated normalized arc-cosine kernel (predicted 0.17 from x, ~0.57 from
    h_8, ~0.74 from h_12; measured 0.15, 0.64, 0.84). The deep ReLU map is
    "mostly linear" in its late-layer fluctuations but never 99.6% linear, which is
    what a 250x CV would require. Unifies F8/F13/F22/F23/F24 at the new shape.
  CONCLUSION: no sampling-based or hybrid-sampling estimator can reach even our
  own live score under 2^41; sampling is closed for P2. The only route is a more
  FLOP-efficient analytic closure. First concrete lever found the same session:
  the V16 dslice algebra has an exact merge (T2 and T4 share right factor P):
  D21 = 2H + [(A*A*w2) + (2/3)(M*P)] P^T + T3/3 -> 3 einsums instead of 4, i.e.
  6k n^3 per layer instead of 7k (0.833 -> ~0.71xB at identical output). Ablation
  sweep of the D21 channel and the M leg running (estimator_v16_abl.py,
  test_v16_abl.py) -> see F63.

- F63 (2026-09-01) — V16 SOURCE MACHINERY RESTRUCTURED EXACTLY: 7k -> ~4k n^3 per
  layer (0.833xB -> 0.50xB) AT MSE PARITY, local score 3.24e-8 -> ~1.96e-8 (1.65x).
  Measured on 4 P2 dumps, real flopscope C, precise final_means GT
  (estimator_v16_abl.py + test_v16_abl.py for the ablations; estimator_v16b.py +
  test_v16b.py for the shippable form; logs runs/v16_abl*.log, runs/v16b_r*.log).
  MECHANISM (what the K3 legs actually carry):
  * Ablations (riders on): full 0.833xB 3.884e-8 (== shipped); drop the D21 (2,1)
    dslice channel 0.364xB 4.71e-6 (!); drop the M leg 0.481xB 3.46e-6; drop
    both 0.247xB 5.55e-6; M_b = diag(S3c) only (drop 3*S21^T) 0.716xB 2.10e-6.
    => The ENTIRE K3 gain over covprop (4.4e-6) flows through kappa3_{iic}
    feeding the covariance update; marginal skew (D3 -> mean) is worthless alone
    (consistent with F60: the closure error is a variance/kurtosis effect, and the
    (2,1) slices are what fix the variance propagation). The M leg is essential
    BECAUSE it carries S21 = K21 - rep21, the exact residual (2,1) content the
    struct-free B1 hub block misses (the DS-shadowed B2 content of V1.6).
  * Exact algebra #1 (MERGE): T2 = (A*A*w2)P^T and T4 = (M*P)P^T share right
    factor P -> one einsum: D21 = 2H + [(A*A*w2) + (2/3)(M*P)]P^T + T3/3.
    4 -> 3 dslice einsums, 0.833 -> 0.716xB, mse 3.8832e-8 vs 3.8838e-8.
  * Exact algebra #2 (the M leg): from the Wick tables, S21's leading term is
    the (c_off, ones2) term of pk21 minus the PK2K subtraction 2*pk1*pk11:
    S_sep = diag(w(1,2) - 2 pk1 w(1,1)) C_off diag(w(1,1)) = a column-scaled
    a_b^T. So 3 S_sep^T = 3 a_b diag(e) rides on the A leg (M = P diag(S3c)
    + 3 A diag(e) + ...), no transport. S_sep explains 99% of |S21| at the
    first birth, 84.5% at layer 14 (per-layer diag in runs/v16_abl_msep_diag.log).
    The remainder Rres = S21 - S_sep (D21-driven: 0.5 w(1,2)_i w(2,1)_c D21^T
    plus c_off*d21 cross terms) is NOT negligible (S_sep alone: 4.47e-7, 11x
    worse) but IS numerically low-rank: top-64 SVs hold 47-94% of its energy,
    top-256 75-99%. Keeping Rres to rank r via SVD: r=16 3.946e-8, r=32
    3.919e-8, r=64 3.899e-8, r=128 3.896e-8, r=256 3.884e-8 (dense 3.883e-8).
  * Shippable form (estimator_v16b.py): legs A, P dense (k,n,n) + Z = P Rr thin
    (k,n,r) transported, L static; M*P, leftA, leftP formed elementwise; D21 =
    leftA A^T + leftP P^T + (1/3)(PP L) Z^T -> 2 dense contractions + O(n^2 r).
    Rres factored by a randomized range finder (Omega = w32[:, :r], one power
    iteration, flopscope QR; ~4 n^2 r per birth, <0.1% of B). r=64 harness:
    C=0.5154xB, mse 4.08/3.54/3.94/4.09e-8 == SVD probe (4.05/3.56/3.92/4.06).
    r=32 chosen (0.50xB, +0.9% mse): local score ~1.96e-8 vs shipped 3.24e-8.
  * Cost per source-layer now: 2 transports + 2 contractions (was 3 + 4). The
    A leg (= P a_b, a_b = w1*C_off full-rank) cannot be folded further; the
    remaining floor of this representation is 4k n^3 with k = live sources.
  WRITE-UP: the D21-channel ablation is the cleanest mechanistic statement of
  WHY third cumulants matter for deep ReLU means (they correct covariance
  propagation, not the marginal skew), and the separable+low-rank structure of
  the residual (2,1) content is a genuine algorithmic contribution.
  NEXT: same lens on the kappa4 (2,1,1) channel (F51/F52 aug: 2.04e-8 raw at
  1.77xB, structural floor F58 was computed with 3+ dense legs/source) -- if its
  legs admit the same separable + low-rank treatment, aug may fit under B.

- F64 (2026-09-01) — F63 LENS APPLIED TO THE AUG KAPPA4 CHANNEL: it cuts the
  dense floor 21 -> 7 n^3-units per source-layer (1.77xB -> ~0.95-1.05xB) but the
  aug channel STILL cannot beat V16b under 2^41; the augmented-K3 family is CLOSED.
  Measured on P2 dumps 0/1 (lean torch, precise final_means GT, dense aug =
  2.0796/2.0033e-8, mean 2.0415e-8; scratch/aug_leg_probe.py, logs runs/aug_leg_*.log,
  spectra scratch/aug_leg_spectra_*.npz, cost model scratch/aug_cost_model.py).
  * LEG ANATOMY (exact, from the hub formulas): every aug leg is a sum of
    (i) column-scaled A = P a_b (a_b = w1*C_off, full rank), (ii) column-scaled P
    (all diagonal parts: diag(G) in X3, the -(m/3)diag(dG) of the zero-diagged wk4m,
    diag(S3) in M), (iii) D21-born parts diag(w2)D21 and diag(w1)D21^T (X1,Y1,X2,Y2),
    (iv) G-born part diag(w1)G_off diag(w1) in X3, (v) Rres in M (F63), (vi) rank<=2
    vectors (Y3, M_t1, wk4m off-diag = (m/6)(dG 1^T + 1 dG^T)). So transported dense
    objects are A, P and X3 = P d(w1) G d(w1); everything else is thin (n x r) with a
    static right factor. Dense-leg count 8 -> 3.
  * EXACT COST TRICKS FOUND: (a) X3's dslice contraction (P*Y3)@X3^T == (X3*Y3)@P^T
    because Y3 = y 1^T and Gs = d(w1) G d(w1) is symmetric -> folds into the P-group
    (X3 costs only its transport); (b) k211 uses only Sym(acc), so L_acc@P^T ->
    P@L_acc^T merges with P@R_acc: ONE matmul per source (parity 4.4e-16 vs reference
    AUGMENT; applied in lean_k3_aug.py); (c) right-factor grouping by dense object as
    in V16b; thin right factors two-stage (Left@F then @T^T) with the second stage
    grouped per thin object. Remaining dense floor per source-layer: transports A,P,X3
    (3) + dslice A-group, P-group (2) + k211 P-group (1) + k211 middle sandwich
    A d(c) A^T (1) = 7 units (V16b: 4). Thin cost (22 r_d + 4 r_m + 10)/n units.
    flopscope gives NO symmetric-output discount (probe_sym_cost.py: (A*c)@A.T
    bills 1.000 x 2n^3), so sandwiches cost full.
  * SPECTRA (energy captured at rank r, birth layer L01 -> L15): D21 r64 0.59 -> 0.98,
    r128 0.75 -> 0.99; G_off r64 0.61 -> 0.96; Rres r64 0.52 -> 0.97 (|Rres|/|S21|
    0.05 -> 0.18). Early-born (longest-lived, costliest) sources are the LEAST
    compressible.
  * FAITHFUL BIRTH-TRUNCATION (state, compounding; transport of a rank-r matrix is
    exact so this is NOT the F46 output proxy): d21=256 2.032e-8 (-0.5%), 64 2.149
    (+5%), 32 2.229 (+9%); rres=64 2.082 (+2%), 32 2.100 (+3%); g=256 2.113 (+3.5%),
    64 2.260 (+11%), 32 2.524 (+24%), g=0 8.1e-6 (G_off inside X3 is ESSENTIAL: 400x);
    combos 128/128/32 2.218 (+9%), 64/64/32 2.425 (+19%), 32/32/32 2.766 (+35%).
  * ABLATIONS: no K4->K3 feed (X3,Y3,M_t1 dropped) = 9.18e-6 on dump0, IDENTICAL to
    g=0 -> with diagonal-only G the term-1/term-3 feeds cancel exactly; the feed is a
    coupled unit and G_off is its whole content. Dropping the k211 middle rotations of
    old sources (saves the A-A sandwich, 1 unit): 2.352e-8 (+15%; dump1 +27%) -> NOT
    benign. (An earlier "+1.8%" read was a probe bug: per-dump flag pop; fixed.)
  * SCORE ARITHMETIC (dump-local; V16b r=32 reference = 3.93e-8 x 0.4986 = 1.96e-8;
    overhead 0.13xB = C/G transports + capped production + birth k211): with the
    middle rotation kept, r_d=32/r_m=32 -> 1.05xB (OVER, zero fallback); r_d=16 ->
    1.01xB. nomid r_d=32 -> 0.93xB at 2.58e-8 = score 2.40e-8 (WORSE than 1.96e-8);
    nomid r_d=64 -> 1.01xB (over). Even the THEORETICAL floor (7 units, zero thin
    cost, zero truncation loss) = 0.95xB x 2.04e-8 = 1.94e-8 == V16b. The aug raw
    advantage (1.9x over V16b) is smaller than its exact cost ratio (~2x): the score
    is flat at best, negative in every realizable configuration.
  * CONCLUSION: augmented-K3 CLOSED for P2 (F58 reaffirmed with the structural
    reason: the third dense leg X3 = P G and the k211 P-group are irreducible, and
    the D21-born thin parts cost ~1 unit at the ranks that hold accuracy). Do not
    reopen without a representation that avoids per-source dense G transport.
    V16b (4 units, 0.50xB) is the floor of the whole hub-source K3 family; the
    leaders' 0.22xB tier must come from a different closure (F59/F60 open question).
  * WRITE-UP: the leg anatomy (which K3/K4 content is separable, which is low-rank,
    which is irreducibly dense) + the two exact merges are a clean negative result
    with a mechanism; the P-group/A-group cost accounting generalizes F63.

- F105 (2026-09-22) -- F68 CLAIM CORRECTED AFTER AN OUTSIDE REPLICATION: THE R^2 WE QUOTED
  WAS THE POST-RELU FIT ON 2 DUMPS; THE SHIPPED (PRE-ACTIVATION) FIT IS 0.50 AT L0 -> 0.99
  AT DEPTH ON 8 DUMPS; THE (2,2) SLICE OF KAPPA4 IS A DIFFERENT OBJECT. Trigger: forum reply
  #4 on topic 18218 (AndreasHad04, 2026-09-22). They regressed the off-diagonal (2,2) slice
  of post-activation kappa4 on C_pre / C_post (through origin, 8 nets, their own chain):
  R^2 0.998 at L1 decaying monotonically to 0.045 at L15 (median L8-14: 0.12 vs C_pre, 0.07
  vs C_post; shuffled control ~0), lambda 0.32 -> 1.6e-3; C_pre the better regressor at
  14/15 layers. They asked which object our ~0.95 was measured on.
  CHECKED (no new runs, existing logs + code):
  * OBJECT: our G is the r=1 "harmonic projection" matrix core of the augmented chain
    (scratch/lean_k3_aug.py:1262): G_off = cA*(K211_cap + hub source contributions) +
    cS*Sym(K31). K22 enters ONLY through its row sums on the DIAGONAL, so the off-diagonal
    (2,2) slice is not part of G_off at all. Consistent with F68's own ablation (the gain
    lives in the hub-contracted content, not in the n^2 slices). Their measurement does not
    contradict F68; it is a measurement of a different object.
  * SAME REGRESSION: g_regen_probe.py FitHook / PreFit are through-origin, uncentered R^2 --
    identical to theirs.
  * POST-RELU fit (the F68 number, runs/g_regen_1.log, [fit:c], dumps 0/1): R^2 by layer
    0.87 0.66-0.69 0.75-0.77 ... 0.96-0.97 (L0..L15). This is the "0.66-0.97 rising with
    depth" quoted in the post, and the min is at L1, not L0.
  * PRE-ACTIVATION fit (G_pre_off vs C_pre_off = what V22+ ships, runs/lam_prefit_d0-7.log,
    8 dumps): 0.48-0.52 at L0, 0.83-0.86 L1, 0.92-0.93 L2, >= 0.95 from L3, 0.985-0.99 at
    L13-14. So the pre fit is BETTER than the post fit from L1 on -- the same ordering they
    found (C_pre > C_post), and the code's choice is the stronger one.
  * CORRECTION: the post/claims said "C_off" without stating pre vs post and quoted the
    post-ReLU R^2 range (2 dumps) next to a code that uses the pre-activation law. Correct
    statement: "the off-diagonal of the aug chain's kappa4 matrix core, at the
    PRE-activation, is lambda_l C_pre_off with R^2 ~0.5 at layer 0, >= 0.95 from layer 3
    and 0.99 at depth (8 dumps); the post-ReLU version of the fit is weaker (0.66-0.97, 2
    dumps)". F68's WRITE-UP line "R^2 ~0.95 at depth" is superseded by this entry.
  * NOT a claim about the (2,2) slice. Their finding (the (2,2) slice is covariance-shaped
    only at L1) is compatible and worth citing as the outside complement.
  ACTIONS: claims_evidence C1, derivation_code_map 2.4 + LLM-claim 1, community_post.md
  (repo copies) annotated; forum reply drafted (user posts). Offered to run their
  script on the aug core in the lean chain on the same 8 dumps if they send it.
