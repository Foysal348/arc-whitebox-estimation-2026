# Round 10b — an exact contraction-order search for the young third-cumulant tier

*(Answer in English, at most 3 pages. This is a counting / search task, not a survey. Every FLOP count must be exact at the
real shapes; if you ran code, say so and show the counts it produced; never present an estimate as a measurement.)*

## Setting

Fixed bias-free ReLU MLP, width n = 1024, depth 16, weights N(0, 2/n), input x ~ N(0, I). We estimate the final
post-ReLU means with a deterministic cumulant chain (504aldo's factorised K = 3 chain: mean, covariance, κ₃ as a sum over
sources, memoryless κ₄, bivariate Wick/Edgeworth closure). Cost is metered in FLOPs: 1 u = 2³¹ FLOPs ≈ one dense
1024 × 1024 × 1024 product (2n³ FLOPs); a level-5 Strassen square product counts ≈ 0.51 u. Total ≈ 249 u per network.

**Sources.** Every ReLU layer b gives birth to one third-cumulant source. At a later layer l it carries two dense n × n legs:
- P_{b→l} = M_l · M_{l−1} ⋯ M_{b+1} (the propagator; M_t = W_t · diag(Φ_t), with Φ_t = P(z_t > 0) per neuron), updated
  each layer by one product P ← M · P;
- A_{b→l}: A is a second dense leg, born as W·R_b and transported by the SAME left map as P, so A = P·R_b with a fixed full n×n birth matrix R_b (verified on our captures: one common left map fits both legs to 1e-14; R_b is not diagonal); forming A entry-wise from P is still an n³ product, so both legs cost one product each.

The next closure needs, at every layer l, the (2,1) slice D21 (n × n) and its diagonal d3, summed over sources:

  D21_l = Σ_b [ (P∘P) · diag(w_b) · Aᵀ + 2 (P∘A) · diag(w_b) · Pᵀ ] / 3,     (P = P_{b→l}, A = A_{b→l})

(schematic; the code fuses the two right-factor contractions) where ∘ is the elementwise (Hadamard) product and w_b is a birth-weight vector. The Hadamard products are hyperedges of
the tensor network (a shared row index a appears twice in one term).

**Tiers.** Ages 1–4 (l − b ≤ 4) keep dense legs: the "young tier". Older sources move into a shared low-rank basis (not
part of this question).

## Measured cost ledger (per network)

| part | u |
|---|---|
| **young tier: transport M·[P, A] for ages 1–4** | **≈ 54** |
| **young tier: D21 contractions (Hadamard products × dense legs)** | **≈ 50** |
| join into the shared basis | ≈ 41–46 |
| old tier | ≈ 41 |
| thin feedback legs | ≈ 15 |
| covariance, closure, Strassen additions | ≈ 20 |

So the young tier is four dense n × n products per source-layer (two transports, two contractions), for four ages, at
every layer: ≈ 104 u.

## Already checked, no saving (do not re-propose without an exact count showing otherwise)

- Skipping A's transport: A = P·R_b, but forming A entry-wise for the Hadamard terms costs an n³ product anyway.
- Reusing P_old = P_new·M: forming it is itself an n³ product.
- Batching the four ages into wider products: identical FLOPs.
- Age-1 P = M_l (the weight map itself): already used.

## The question

Source-forward propagation (above) and reverse-mode evaluation (sweep the next-layer query vectors backwards through
the maps and meet each birth) are two contraction orders of one tensor network; both cost about the same (~4 square
products per (birth, target) pair). **Is there a mixed or hierarchical contraction order over the (birth b, target l)
pairs of the young tier — for example forward for ages 1–2 and shared reverse sweeps for ages 3–4, or precomputed
products of consecutive maps reused across several sources and targets — whose total FLOP count is below the current
≈ 104 u (target ≤ 55 u) while producing the same D21 and d3 exactly?**

Deliverable:
1. Write the young tier as a tensor network with hyperedges (the Hadamard products), listing the tensors and shapes.
2. Give the FLOP count of the current order at the real shapes and check it against ≈ 104 u.
3. Give the best order(s) you can find, each with an exact FLOP count, and say which intermediates are shared across
   sources or target layers.
4. Verdict: either an order that is ≥ 10% cheaper (with the exact op list, so we can implement and verify it on our
   stored legs to 1e-6), or an argument, with counts, for why the Hadamard hyperedge forces ≈ 4 square products per
   (birth, target) pair in every order.
