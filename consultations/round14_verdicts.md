# Round 14 verdicts (prompt: consultations/prompts/ROUND14_prompt_accuracy_mechanism.md)

## 1. "Response-Cashout Moment Propagator (RCMP)" — `Architecture Analysis/Round 14 - Response-Cashout Moment Propagator.pdf` (27 Sep 2026)

**Proposal.** Use every newborn κ₃ source once, at age 1; cash its effect into the post-ReLU mean and covariance through
the existing closure; discard the source. All later memory is carried by (m, C) only; no old κ₃ sources, no basis join, no
old tier. Claimed cost 91 dense products (util ~0.09), projected raw 1.65–1.85e-8.

**Verdict: DEAD (ceiling-type kill, fair under the new protocol).** The mechanism was already measured:

- RCMP lets only the age-1 source's slices enter the closures and carries older sources' effect through C alone. That is
  exactly the instrumented "keep old content only in the last h layers" experiment (writeup §5, Measurement B): extra
  final-layer MSE vs the full chain **4.85e-7 / 4.17e-7 / 3.16e-7 for h = 1 / 2 / 3** (the chain's own error vs truth is
  2.1e-8), consistent with an independent Kaggle "no-old" run (7.4e-7). RCMP ≈ h = 1: about 23× the chain's error, with the
  same closures as V29 — a structural gap that tuning cannot close.
- Its premise misreads K3m (notes K3m, probe_adjoint_val.py / probe_d21late.py): the "86% through the next one or two mean
  shifts" refers to covariance -> E[ReLU] -> mean paths in the response model, but the same study found that >= 90% of the old
  slices' effect at layer 14 travels through a NON-covariance channel, and that the old sources act mainly through their
  per-neuron κ₃ DIAGONAL (D3) in the mean closure: removing only the old D3 at layer 14 adds 7.8e-8 (about 4× the error
  budget). RCMP discards exactly this content, and diagonal-only κ₃ transport is itself dead (the next diagonal is fed 90–99%
  by off-diagonal content).
- Its Stage-0 falsifier (variance at L14–16 from a true (m, C) seed at L11) would not test the failing path: the loss is in
  the mean via the old κ₃ diagonal, not in the variance.

**Reply to the model (optional, paste as is):**

> Thank you — the design is clear, but its mechanism is already measured on our chain. Keeping only the age-1 κ₃ content in
> the closures (older sources' effect carried by C alone) adds 4.85e-7 final MSE vs the full chain (h = 1; 4.17e-7 at h = 2,
> 3.16e-7 at h = 3), about 23× the chain's own error of 2.1e-8. The "86%" figure was about covariance paths in a response
> model; the same study showed that >= 90% of the old sources' effect at layer 14 bypasses the covariance and acts through
> each neuron's own κ₃ diagonal in the mean closure (removing only the old diagonal at layer 14 adds 7.8e-8). Any successor
> must carry the old sources' per-neuron κ₃ diagonal (which itself needs their off-diagonal content) at every later layer.
> Given this, what representation would carry that diagonal cheaply?

## 2. "Exchange Vertex Closure (EVC)" — `Architecture Analysis/Round_14_Exchange_Vertex_Closure_and_Cost_Gap.pdf` (27-28 Sep 2026)

**Proposal.** Replace the κ₄ diagonal fed to the mean closure (on the most sensitive neurons) by the exchange term of the
connected 4-point function, κ₄ᵢᵢᵢᵢ ≈ 3 Σ_ab Yᵢₐ (C⁻¹)ₐb Yᵢb (Y = the κ₃ (2,1) slice incl. its diagonal), level-matched to the
chain; hypothesis: the independent vertex residual has little pattern. ~+15 u. The model itself did not claim it explains
the leaders.

**Verdict: DEAD (component test with teacher inputs, the proposal's own pre-registered gate, three regularisations).**
`experiments/probes/r14_exchange_oracle.py` (results/summaries/r14_exchange_oracle_tau*.json): with the TEACHER's C and Y,
centred pattern correlation with the teacher's standardised κ₄ diagonal on the decisive neurons (|μ/σ| < 1.5), layers 12-16:

| τ | MLP #0 | synthetic | G4-weighted error vs the chain (MLP #0) |
|---|---|---|---|
| 1e-3 (proposed) | −0.12 … −0.20 | −0.13 … −0.23 | ×69 … ×95 |
| 1e-2 | −0.11 … −0.19 | −0.11 … −0.22 | ×54 … ×75 |
| 1e-1 | −0.07 … −0.14 | −0.06 … −0.19 | ×26 … ×36 |

Gate: ≥ +0.7 and ≤ 0.5× the chain's error; the chain itself has +0.26 … +0.34. The exchange level is ~10× the true
standardised κ₄ (0.5-0.6 vs 0.05-0.06 at L12-16), i.e. it is almost cancelled by the independent vertex term, whose
pattern is therefore not negligible. Only at layers 4-5 is the correlation positive (+0.2 … +0.47).

**Reply to the model (optional):**

> We ran your falsifier exactly as specified (teacher C and Y, τ = 1e-3, no fitting) and with τ = 1e-2 and 1e-1. At
> layers 12-16 the exchange readout's pattern correlation with the true κ₄ diagonal is negative on both networks
> (−0.07 … −0.23), and its standardised level is ~10× the truth (0.5-0.6 vs 0.05-0.06): the exchange term is large and
> almost cancelled by the independent vertex, so the residual is the pattern. The sensitivity-weighted error is 26-95× the
> chain's. Closed. Thank you for the clean gate.

## 3. Follow-up proposal: joint oracle "cheap κ₃ chain × teacher κ₄ diagonal" (28 Sep 2026)

**Proposal (decision test, not an estimator).** Coupled rollouts from layer 1 on MLP #0: {full chain, previously tested
cheaper κ₃ chains} × {own κ₄, teacher κ₄ diagonal}. Close the path "cut κ₃, recover with κ₄" unless the optimistic score
(oracle cost zero, cheap chain's cost) reaches 2.1e-9 (e.g. raw ≤ 1.65e-8 at 130 u, ≤ 1.48e-8 at 145 u).

**Result** (`experiments/chain/k4_oracle_run2.py`, plain Gaussian chain, κ₄ diagonal injected 1:1; results/summaries/k4oracle2_joint_*.json):

| κ₃ configuration | util (unengineered) | own κ₄ | teacher κ₄ diagonal | recovery |
|---|---|---|---|---|
| full chain | 0.2476 | 2.290e-8 | **1.609e-8** | ×0.702 |
| AGE_OLD 2 | 0.2303 | 5.240e-8 | 4.590e-8 | ×0.876 |
| AGE_OLD 2, R_OLD 256, R_OLD2 128 | 0.1784 | 1.288e-7 | 1.244e-7 | ×0.966 |
| AGE_OLD 1, R_OLD 192, R_OLD2 96 | 0.1381 | 6.511e-7 | 6.665e-7 | ×1.024 |

**Verdict: the path "cheap κ₃ + κ₄ recovery" is CLOSED** — the κ₄ oracle helps less the cheaper κ₃ gets, and even the
optimistic scores are 3-40× short of the target. **Positive finding:** the FULL chain with the true κ₄ diagonal reaches raw
1.61e-8, the leaders' raw MSE (1.54-1.61e-8); at our engineered cost (~193 u) that would score ~3.0e-9. The leaders'
accuracy gap is fully accounted for by a better κ₄ diagonal on an unchanged κ₃ carrier.

**Reply to the model (optional):**

> We ran the joint oracle on MLP #0 (coupled from layer 1, κ₄ diagonal injected 1:1). Full chain: 2.29e-8 → 1.61e-8 with
> the true κ₄ diagonal (×0.70). Cheaper κ₃ chains: AGE_OLD 2 (util −7%) 5.24e-8 → 4.59e-8 (×0.88); + R_OLD 256 / R_OLD2 128
> (util −28%) 1.29e-7 → 1.24e-7 (×0.97); AGE_OLD 1 + 192 / 96 (util −44%) 6.51e-7 → 6.67e-7 (×1.02). So κ₄ does not buy
> back κ₃ cuts; that path is closed. But the full chain plus the true κ₄ diagonal lands exactly at the leaders' raw MSE
> (1.61e-8). The question therefore narrows to: a κ₄ diagonal estimator that reaches the true diagonal's benefit on the
> FULL κ₃ chain for ≤ ~20-30 u, given that the deep κ₄ pattern is carried memory (horizon ≥ 6 layers, linear carry of the
> true κ₄ gives pattern corr 0.77-0.80), is not a function of even statistics, and is not the exchange term K C⁻¹ K
> (measured: corr −0.07 … −0.23 at L12-16). What would you carry?
