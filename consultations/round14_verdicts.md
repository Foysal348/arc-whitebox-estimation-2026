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
