# ARC White-Box Estimation Challenge 2026 — Phase 2

Research repository for the [ARC White-Box Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
(WhestBench, Phase 2). The task: predict, for a fixed randomly initialised ReLU MLP (width 1024, depth 16, no biases,
He-initialised weights), the per-neuron mean of every post-activation layer under a standard Gaussian input — analytically,
within a hard budget of 2⁴¹ FLOPs per network, without sampling the network.

The score is `final_layer_mse × max(0.1, C/B)`, where `C` is the metered FLOP count and `B = 2⁴¹`, so accuracy and cost are
multiplied: a 1 % cost saving is worth exactly as much as a 1 % accuracy gain.

**Team Hack2Publish:** mdhamidhosen, esfer_sami, foyshal_sk. **Team best:** `#336377` (E299, md.hamidhosen) — adjusted
**3.3946e-9** (raw final-layer MSE 1.8824e-8, util 0.180), built on this repository's `#334870`. Public board 16 Oct 2026:
rank 46 (tied, displayed 3.4e-9). Work in this repository ended on 16 Oct 2026 (submissions close 17 Oct 2026 23:59 UTC); the final state is below.

---

## Status (final, 16 Oct 2026)

| Submission | Change | Adjusted score | Raw MSE | Util |
|---|---|---|---|---|
| #331681 | 504aldo's public V29 chain, verbatim (MIT, attributed) | 5.418e-9 | 2.133e-8 | 0.254 |
| #331787 | + Strassen floor, spherical-input reparameterisation (V31) | 5.137e-9 | 2.108e-8 | 0.244 |
| #332257 | + feedback scale 1.25, β 1.2, transport join | 4.9519e-9 | 2.1788e-8 | 0.227 |
| #332420 | + dead-neuron compaction, D21 column/row trims, basis-side Strassen | 4.0902e-9 | 2.1636e-8 | 0.189 |
| #334591 | + V55 κ₄-pattern corrector (A class) | 3.8582e-9 | — | — |
| #334870 | + V56 cubic star, map fitted on the synthetic net (A) — the chain the write-up analyses | 3.6521e-9 | 1.950e-8 | 0.187 |
| #336377 | md.hamidhosen's E299: 24-group and κ₄-corrector calibration + exact cost rewrites | **3.3946e-9** | **1.8824e-8** | **0.180** |
| #336440 | E299-B: E299 in the safe operation class (18.1k warm ops; exact rewrites) | 3.4719e-9 | 1.8825e-8 | 0.184 |

- **Write-up:** `writeup/WRITEUP_MAIN.pdf` (12 pages) with `writeup/ALGORITHM_TRACK_PAPER.pdf` as its appendix. It names
  one graded submission and is sent by e-mail to arc-whestbench@aicrowd.com by 24 Oct 2026 23:59 UTC.
- **Nominations** (up to two, chosen after the close on the AIcrowd submissions page): the best score plus a hedge with lower
  residual-time and memory risk; see the write-up header.
- **E-series** (md.hamidhosen): https://github.com/hamidhosen42/ARC-White-Box.

The starting point is the K = 3 factorised cumulant-propagation chain released under MIT by team **504aldo**
([whest-p2-cumulant-k3](https://github.com/504aldo/whest-p2-cumulant-k3)); all derived estimators keep that attribution.

## Principal findings

1. **A parity theorem.** Every third-cumulant source contributes an *odd* function of the current layer's weights, so its
   conditional expectation given any even statistic (covariance, Gram matrices, per-neuron variances) is exactly zero.
   Mean-field and "self-averaging" closures of the deep tier therefore lose the entire contribution, not merely some accuracy.
2. **The old-source tier is irreducible inside this formulation.** Rank reduction, pooled Tucker/TT cores, aggregate
   vector–metric states, sketches, birth-column screening, source-index coding, birth-operator compression, slice recursion
   and adjoint contraction were each measured and each is net negative. See `writeup/UNTESTED_IDEAS.md` for the board.
3. **The error is not where it was assumed to be.** An unconfined (exact) old tier differs from the shipped chain by an MSE of
   7.6e-10 — about 3 % of the chain's error — so ~97 % of the remaining error is the K3 + memoryless-κ₄ truncation itself.
4. **A free accuracy gain.** Because a bias-free ReLU network is positively homogeneous, the Gaussian input factorises exactly
   as `X = (R/√n)·Y` with `Y` uniform on the sphere and `R ~ χ_n` independent; the sphere's fourth cumulant can be injected at
   layer 0 through the chain's own κ₄ machinery and the output rescaled by `a₁ = E[R/√n]`. Cost +0.02 %, graded −1.1 % raw.
5. **V29's fitted λ has a derivation.** A Monte-Carlo κ₄ teacher (10⁷–10⁸ samples per layer, two independent networks) shows
   that the true (3,1) cumulant slice is weighted *per row* by `κ₄(i)/var(i)`, and that 504aldo's per-layer scalar λ equals the
   **mean** of that ratio to within 2 %. Only the row-to-row variation is unexploited.

The full experiment log — about forty measurements, each with its method and its falsifier — is
`docs/ARC_WhestBench_2026_Complete_Notes.pdf`, Part K.

## Repository layout

```
docs/              The 60-page handbook (PDF) and its builder; reference captures of the challenge pages
writeup/           Algorithmic-contribution write-up drafts and the idea board with per-idea verdicts
submissions/       One directory per submitted estimator (+ packaged .tar.gz archives)
experiments/
  probes/          Diagnostics that instrument the chain (dumps, sensitivity maps, rank studies, falsifiers)
  chain/           Runners and analyses (knob sweeps, memory census, κ₄ law fits, the oracle harness, the poller)
  variants/        Estimator variants built from a shipped submission by patching (angular, per-row λ, oracle, f64)
kaggle/            Kernel notebooks and metadata for every Kaggle run (runners, κ₄ teacher, dumps, one-step, kprop)
results/           Small JSON summaries and run logs
consultations/     Prompts sent to external models and their reports (rounds 1–5)
correspondence/    Security report to the organisers and forum post drafts
data/              Large artefacts: .npz dumps and downloaded Kaggle outputs  (git-ignored)
```

Three third-party working copies live at the repository root and are git-ignored; clone them if you want to reproduce:

```bash
git clone https://github.com/AIcrowd/whest-starterkit                                  # evaluation kit (uv environment)
git clone https://github.com/504aldo/whest-p2-cumulant-k3                              # the public K=3 chain (MIT)
git clone https://github.com/alignment-research-center/mlp_cumulant_propagation        # ARC's reference implementation
```

## Reproducing

```bash
# 1. Environment (from the starter kit, which provides the metered numpy replacement)
cd whest-starterkit && python -m uv sync
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1          # Windows consoles otherwise fail on the kit's output

# 2. Validate and score a submission locally
python -m uv run whest validate --estimator ../submissions/v31ang/estimator.py
python -m uv run whest run --estimator ../submissions/v31ang/estimator.py --split mini --n-mlps 10

# 3. Cost / accuracy knob sweep on one locally generated network
python -m uv run python ../experiments/chain/knob_test.py myvariant V26_STRASSEN_MIN=8

# 4. A diagnostic that needs the layer dumps (writes into ../data/dumps/)
python -m uv run python ../experiments/probes/probe_query.py 10,14
cd .. && python experiments/probes/probe_isdf.py
```

Scripts under `experiments/probes/` and `experiments/chain/` are invoked from the `whest-starterkit` directory when they need
the metered environment, and from the repository root when they only analyse saved dumps; each file's docstring states which.

Kaggle runs use the notebooks in `kaggle/`; each directory holds the notebook and its `kernel-metadata.json`, and the
accompanying dataset directories hold the estimator variants uploaded alongside them.

## Requirements

Analysis scripts need `numpy`, `scipy`, `torch` (CUDA for the Monte-Carlo teachers), `scikit-learn`, `teneva`, `pypdf` and
`reportlab`; the metered chain itself needs only `flopscope` and `whestbench` from the starter kit. See `requirements.txt`.

## Licensing and attribution

Original work in this repository is released under the MIT License (`LICENSE`). Estimators under `submissions/` derive from
504aldo's MIT-licensed `estimator_v29.py`; each keeps the upstream copyright header and ships `LICENSE-504aldo`. See `NOTICE`.

Third-party consultation reports under `consultations/` are the output of external language models, retained as research
records; they are not covered by this repository's licence.
