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
