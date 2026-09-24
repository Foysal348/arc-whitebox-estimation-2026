"""Round 10 (Sonnet, 24 Sep): explicit hand-designed candidate contraction orders for the young
tier's transport step, counted in dense-(n,n,n)-product units (1 product = 1 "mm-unit"; convert
to u with the measured Strassen-L5 unit cost from sonnet10_unit_costs.json).

The question: is there a MIXED/hierarchical order over (birth layer b, target layer l) pairs,
ages 1..4, cheaper than the shipped forward-incremental scheme (P_l(b) = W_l diag(Phi) P_{l-1}(b),
one new dense product per age, reusing the previous age's leg)?

Candidates counted here, for ONE birth layer b needing all 4 of its ages (the pattern repeated
identically -- shifted by one layer -- for every b, so per-layer cost is what the ledger meters):

  C0 incremental (shipped): age1 = 1 product from P_b; age2 = 1 product from age1; age3 = 1 from
     age2; age4 = 1 from age3.  Total 4 products, and every partial result is reused going
     forward -- this is also what a single forward pass over the whole depth-16 network gives
     for free (one running product per source per layer).

  C1 explicit-suffix precompute ("reverse-style"): form the chain matrices T2 = W_{l1+1} W_{l1},
     T3 = W_{l1+2} T2, T4 = W_{l1+3} T3 EXPLICITLY first (3 products, same chain-multiplication
     cost regardless of parenthesization since every factor is n x n -- see
     sonnet10_opt_einsum_search.json check (4)), THEN apply T1..T4 to P_b (4 more products).
     Total 7 products for one source's 4 ages -- WORSE than C0, because forming the Ti matrices
     is extra work the incremental scheme gets for free by applying W directly to the (already
     n x n, not n x r) leg at each step.

  C2 independent-per-age (no reuse across ages): age_k leg = length-k chain applied fresh to
     P_b. Cost 1+2+3+4 = 10 products for one source's 4 ages. Worse than C0 by 2.5x.

  C3 batched-into-one-wide-product (same total FLOPs; already measured directly in
     sonnet10_unit_costs.json mm_batch_scaling / hub_batch_scaling: per-matrix cost is
     CONSTANT (0.633 u, mm) or nearly so (0.56-0.65 u, hub) across batch size 1..12 -- batching
     changes wall-clock/overhead amortization, not the FLOP count).

None of C1-C3 is cheaper than C0 (the shipped order); C1 and C2 are strictly worse, C3 is
FLOP-neutral. This matches opt_einsum's own optimal-path search on the equivalent multi-matrix
chain (sonnet10_opt_einsum_search.json, check (4)/(4b)): with every factor square n x n, chain-
reordering cannot reduce the classical (k-1)-products cost, and obtaining k required DISTINCT
prefix outputs (not just the final one) costs at least k-1 relative products no matter the order.

Usage: python sonnet10_candidate_orders.py  (pure python, no project imports; run from anywhere)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT_COSTS_JSON = os.path.join(HERE, "..", "..", "results", "summaries", "sonnet10_unit_costs.json")
OUT_JSON = os.path.join(HERE, "..", "..", "results", "summaries", "sonnet10_candidate_orders.json")

unit_costs = json.load(open(UNIT_COSTS_JSON)) if os.path.exists(UNIT_COSTS_JSON) else None
mm_unit_u = unit_costs["mm_batch_scaling"]["4"]["per_matrix_u"] if unit_costs else 0.6334  # Strassen-L5 dense product

AGES = 4  # AGE_OLD
candidates = {
    "C0_incremental_shipped": {"products_per_source_lifetime": AGES,
                                "description": "one new dense product per age, reusing the previous age's leg"},
    "C1_explicit_suffix_precompute": {"products_per_source_lifetime": (AGES - 1) + AGES,
                                       "description": "form T2..T4 explicitly (3 products, chain-order-invariant "
                                                        "since all factors are n x n) then apply T1..T4 to P_b (4 more)"},
    "C2_independent_per_age": {"products_per_source_lifetime": sum(range(1, AGES + 1)),
                                "description": "recompute each age's leg from scratch, no reuse across ages"},
}
for name, c in candidates.items():
    c["mm_units_u"] = c["products_per_source_lifetime"] * mm_unit_u
    c["vs_shipped_pct"] = 100.0 * (c["products_per_source_lifetime"] / AGES - 1.0)

print(f"mm unit cost used: {mm_unit_u:.4f} u/product (Strassen level 5, batch >= 2)\n")
for name, c in candidates.items():
    print(f"{name}: {c['products_per_source_lifetime']} products/source-lifetime "
          f"= {c['mm_units_u']:.3f} u  ({c['vs_shipped_pct']:+.0f}% vs shipped)")

out = {"mm_unit_u": mm_unit_u, "ages": AGES, "candidates": candidates,
       "verdict": "no candidate found cheaper than the shipped incremental order (C0); "
                    "C1 and C2 are strictly more expensive, C3 (batching) is FLOP-neutral "
                    "(measured directly, see sonnet10_unit_costs.json)."}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(out, open(OUT_JSON, "w"), indent=1)
print("\nwrote", OUT_JSON)
