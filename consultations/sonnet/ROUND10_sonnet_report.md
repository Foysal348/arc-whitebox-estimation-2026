# Round 10 (Sonnet): exact contraction-order search for the young tier

**Question.** Source-forward propagation and reverse-mode evaluation are two contraction orders
of one tensor network. Is there a mixed/hierarchical order over (birth layer b, target layer l)
that beats the shipped young-tier FLOP count?

**Verdict: no.** Within the searched space (opt_einsum optimal-path search on the elementary
primitives, 3 explicit hand-designed candidate orders, and the one plausible algebraic shortcut
the task background suggested) nothing beats the shipped forward-incremental order. The shipped
scheme sits exactly at two independent, order-invariant lower bounds: the transport recurrence
needs 1 new dense product per (birth, target) edge, and the D21 contraction needs 2 dense
products per (source, layer) pair. Both bounds hold for *any* contraction order, so "forward" and
"reverse" are relabelings of the same unique minimum here — there is no third order to find.

## 1. Current young-tier op list, counted from a live run

Instrumented `submissions/v40fb/estimator.py` (read only, via `exec()` + `sys.settrace` on the
compiled module — no edits) while running `Estimator.predict()` once on a real width-1024,
depth-16 network under `flopscope.BudgetContext`
(`experiments/probes/sonnet10_young_tier_count.py`, `results/summaries/sonnet10_young_tier_count.json`).

Measured, per source line:

| line | op | measured u |
|---|---|---|
| L1019 | young-tier transport (`smm.mm`, batched over `2*(k-ka)+extra` legs) | **54.53 u** |
| L1680 | D21 hub, confined sources also present | 40.39 u |
| L1692 | D21 hub, all-young (no confined sources yet) | 10.10 u |
| **young tier total** | | **105.02 u** (background claim: ≈104 u — reproduced within 1%) |

`sys.settrace` on the same run recovered the exact per-layer `(k, ka, extra)` bookkeeping at the
transport gate (`L1018`) and at both `_hub2` call sites (`L1680`/`L1692`). From these, the
**number of dense (n,n,n) products** the young tier actually issues:

- transport: `sum_li (2*(k-ka) + extra)` = **108** dense products (li=2..15; young count ramps
  1,2,3 then plateaus at 3 unconfined sources for li=5..14, jumps to 4 at the untrimmed last
  layer; `extra=2` every layer for the newborn's A/C slots)
- D21 hub: `sum_li 2*(k-ka)` = **100** dense products (2 per young source per layer: one
  contraction against A, one against P)

Implied in-situ unit cost: `54.53/108 = 0.5049 u/product` for transport and
`50.49/100 = 0.5049 u/product` for the D21 hub — the **same constant to 4 significant figures**
from two independent lines of the same run, confirming the op-count model is exactly
proportional to the metered total (`results/summaries/sonnet10_unit_costs.json` measured this
unit cost in isolation at 0.55–0.63 u/product for the same Strassen-level-5 (n,n,n) product,
batch-size-independent from batch 2 to 12 — `mm_batch_scaling` is flat at 0.6334 u/matrix; the
~20% gap between the isolated and in-situ rate does not change the op-count argument, which is
unit-cost-independent).

## 2. Search

**(a) opt_einsum optimal-path search** on the elementary primitives, at real shapes (n=1024,
`shapes=True`, no arrays allocated) — `experiments/probes/sonnet10_opt_einsum_search.py`,
`results/summaries/sonnet10_opt_einsum_search.json`:

| check | opt_einsum optimal cost |
|---|---|
| one Hadamard-then-matmul D21 block (`ik,ik,jk->ij`) | 1.0005 u (= classical dense product; Hadamard-first-then-GEMM is forced, nothing cheaper) |
| 4 fully independent such blocks summed | 4.002 u (no spurious sharing found — correct, none exists) |
| 2 terms sharing ONE right factor | 1.002 u (opt_einsum *does* find the sum-before-matmul fusion — this is exactly the shipped `LA@A^T + LP@P^T` trick, already implemented) |
| 4-step same-size (n,n) matrix chain, single output | 4.0 u for every parenthesization (order-invariant: all factors are n×n) |

**(b) hand-designed mixed/hierarchical orders** for the transport recurrence
(`experiments/probes/sonnet10_candidate_orders.py`, `results/summaries/sonnet10_candidate_orders.json`),
counted in dense-product units per source's 4-age young lifetime:

| candidate | products | vs shipped |
|---|---|---|
| **C0 incremental (shipped)** — `P_l = W_l diag(Φ) P_{l-1}`, reuse previous age | **4** | — |
| C1 explicit-suffix precompute ("reverse-style": form `T_{l:b}` chain matrices first, then apply to `P_b`) | 7 | **+75%** |
| C2 independent-per-age (no reuse across ages) | 10 | **+150%** |
| C3 batching many legs into one wide Strassen family | same FLOPs | measured directly: 0.6334 u/matrix, flat across batch 1–12 |

**(c) the one plausible algebraic shortcut**: the task background describes the source's legs
as "P (propagator) and A = P·diag(a_b)". If that held under transport, `LA@A^T + LP@P^T` could
be rewritten via `A^T = diag(a_b)@P^T` as `(LA*a_b[None,:] + LP) @ P^T` — a **single** matmul
instead of two, i.e. a further 2× cut of the ≈50 u D21 line. Checked directly
(`experiments/probes/sonnet10_ap_relation_check.py`,
`results/summaries/sonnet10_ap_relation_check.json`) against both the birth code and the real
captured dense legs (`data/dumps/r7_cap_L15.npz`, all 14 live sources): **false**. At birth,
`A = diag(w1col) @ C_off` (a row-scaled covariance matrix) while `P = W` (the raw weight
matrix) — two a-priori unrelated dense matrices, not a column-diagonal rescaling of one
another. Measured relative column-nonconstancy of `A/P` is ≈9 (0 would mean the relation
holds); it does not hold at any age 1–14. The 1-matmul fusion is unavailable for this shipped
code.

## 3. Why no cheaper order exists (the structural argument)

- **Transport**: every leg touched is a full n×n matrix (never low-rank while young), so a
  chain of such products has **order-invariant cost** — opt_einsum's optimal search confirms
  4 same-size factors cost 4 u under every parenthesization. The young tier needs **all 4**
  intermediate ages, not just the last one (each feeds its own layer's D21), so the true
  requirement is "compute all 4 prefix products of a length-4 chain applied to `P_b`," whose
  minimum is `k-1`-relative-products = 4 total (shipped), strictly less than 10 (independent) or
  7 (explicit-suffix). No hierarchical grouping beats this because grouping only helps when some
  operand is smaller than n×n (that's exactly what the old-tier low-rank confinement at age>4
  already exploits — outside this round's scope).
- **D21 contraction**: the (X∘P) Hadamard hyperedge shares *both* legs between two operands,
  which forces elementwise materialization before any O(n^3) step — opt_einsum agrees there is
  no cheaper order for a single block. The shipped code already performs the maximal available
  common-subexpression elimination: what would naively be ~4 Hadamard-matmul terms (H, T2, T3,
  T4) collapses to exactly 2 dense contractions, one per independent right factor (A, P) —
  opt_einsum reproduces this exact fusion (2 shared-factor groups → 2 u) when given the terms
  with their true shared operands, and confirms 1 u per group is the floor. Merging the two
  groups into one would require `A = P·diag(a_b)`, which is false for this code (§2c).

## 4. Conclusion

No candidate — automatic (opt_einsum optimal search) or hand-designed (suffix precompute,
independent-per-age, batching, the A/P-diagonal shortcut) — beats the shipped forward-
incremental order, and several are markedly worse (+75%, +150%). The young tier's ≈105 u is the
count of 208 structurally-required dense (n,n,n) products (108 transport + 100 D21-hub), each
irreducible below one Strassen-level-5 product by the arguments in §3. Per the task's rule, no
implementation/verification step was needed (no candidate cleared the ≥10% bar).

## Scripts and outputs (all under `experiments/probes/sonnet10_*.py`, no other files touched)

- `sonnet10_young_tier_count.py` → `sonnet10_young_tier_count.json` (ledger reproduction + op-count model, §1)
- `sonnet10_unit_costs.py` → `sonnet10_unit_costs.json` (isolated Strassen-L5 unit costs, batch scaling, §1/§2b)
- `sonnet10_opt_einsum_search.py` → `sonnet10_opt_einsum_search.json` (§2a)
- `sonnet10_candidate_orders.py` → `sonnet10_candidate_orders.json` (§2b)
- `sonnet10_ap_relation_check.py` → `sonnet10_ap_relation_check.json` (§2c)
