# Round 11 — change WHAT is represented (one focused question)

Settled (do not re-propose):
- Any exact carrier of the total kappa3 costs what our chain costs (~249 u): source-forward and reverse-mode (lazy cubic circuit) are dual contraction orders of one tensor network. No linear compression closed under transport exists (Sym^3 irreducible). Tucker/TT/CP/ISDF/vector-metric/age modes/shared bases/short horizons/source-free trees are all measured dead.
- The kappa3 memory is long (h = 10 still leaves 16% at L15) and coherent (one age-mode ~88%), but that gives no new temporal law: the total is already carried exactly.

Measured facts about what the FINAL MEANS actually consume (MLP #0 / seed-7, 1024 wide, 16 layers):
- Score = final-layer mean MSE x cost. Channel oracles (true input injected every layer): variance -47%, kappa4 diagonal -30% (acts only through the mean closure), kappa3 -14%, all three -94%.
- Linear response inside our chain: the effect of the old sources' slices travels ~86% through next-layer MEAN shifts (variance -> E[relu] at layers l+1, l+2), near-range and full-rank; only ~14% through the low-rank final-variance path. Removing old D21 (n^3) or old d3 (diagonal, n^2) at one late layer each exceeds the error budget; the diagonal's weight grows toward the output.
- Cost ledger: young transport 54 u, young D21 contraction 50 u, join 41-46 u, old tier 41 u, thin legs 15 u. A leader shows the same raw MSE at half our cost.

Question: propose ONE deterministic scheme that computes the channels the next-layer means need (per-neuron variance, d3 = kappa3(z_i,z_i,z_i), kappa4 diagonal, and the covariance entries that feed them) WITHOUT carrying the kappa3 tensor or per-source dense legs, at <= 125 u, with a plausible path to the chain's accuracy. State exactly what state is carried per layer, its cost in n^3 products, and ONE falsifier we can run on stored data (teacher arrays: true pre/post mean, C, Y21 = kappa3(z_i,z_i,z_j), kappa4 diagonal at layers 3/7/11/14/15/16; dense source captures at 7/11/14/15). Short answer, one mechanism, no survey.
