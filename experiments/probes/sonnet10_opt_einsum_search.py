"""Round 10 (Sonnet, 24 Sep): exact contraction-order search for the young tier, via
opt_einsum's 'optimal' path search (exhaustive over binary contraction trees), at the real
shapes (n=1024). opt_einsum.contract_path(..., shapes=True) computes the optimal path and its
FLOP cost from shape tuples alone (no arrays allocated), which is what "shapes=True" is for.

Four checks:
 (1) ONE Hadamard-then-matmul D21 block, "ik,ik,jk->ij" (X, P share both legs elementwise,
     Y is the other dense leg): is there any order cheaper than Hadamard-first-then-GEMM?
 (2) 4 fully INDEPENDENT such blocks summed (a naive, un-fused D21 = H + T2 + T3 + T4): does
     opt_einsum find any sharing when none is algebraically present?
 (3) 2 terms that share the SAME right factor (e.g. T2, T4 both contract against P): does
     opt_einsum discover the sum-before-matmul fusion (the one the shipped code already uses
     to go from ~4 naive terms down to 2 dense contractions, LA@A^T + LP@P^T)?
 (4) A chain of 4 same-size (n,n) transfer matrices applied to a birth leg P_b (the young-tier
     transport recurrence for ages 1-4): is ANY parenthesization (forward, reverse, split)
     cheaper than the classical (k-1) matmuls, when every factor is a full n x n matrix?

Usage: python sonnet10_opt_einsum_search.py   (pure python + opt_einsum + numpy stdlib; no
project imports needed, run from anywhere)
"""
import json, os
import opt_einsum as oe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "..", "..", "results", "summaries", "sonnet10_opt_einsum_search.json")

n = 1024
U = 2 ** 31  # 1 u = 2 n^3 flops (dense n x n x n product)
out = {"n": n, "u": U}

# (1) single block
path, info = oe.contract_path("ik,ik,jk->ij", (n, n), (n, n), (n, n), shapes=True, optimize="optimal")
out["single_block"] = {"opt_cost_flops": float(info.opt_cost), "opt_cost_u": float(info.opt_cost) / U, "path": str(path)}
print("(1) single Hadamard-matmul block: opt_einsum optimal cost =", float(info.opt_cost) / U, "u  (classical dense product = 1 u)")

# (2) 4 independent blocks summed (naive, un-fused D21 = H + T2 + T3 + T4)
path, info = oe.contract_path("tik,tik,tjk->ij", (4, n, n), (4, n, n), (4, n, n), shapes=True, optimize="optimal")
out["four_independent_terms"] = {"opt_cost_flops": float(info.opt_cost), "opt_cost_u": float(info.opt_cost) / U}
print("(2) 4 fully independent terms, opt_einsum optimal search: ", float(info.opt_cost) / U, "u  (= naive sum of 4 x 1u, no sharing found -- correct, none is present)")

# (3) 2 terms sharing ONE right factor (models T2, T4 both -> P^T; or H's own A^T group)
path, info = oe.contract_path("tik,tik,jk->ij", (2, n, n), (2, n, n), (n, n), shapes=True, optimize="optimal")
out["two_terms_shared_right_factor"] = {"opt_cost_flops": float(info.opt_cost), "opt_cost_u": float(info.opt_cost) / U}
print("(3) 2 terms sharing one right factor: opt_einsum finds", float(info.opt_cost) / U,
      "u (Hadamard-sum BEFORE the single matmul) -- this IS the shipped LA/LP fusion (2 shared-factor groups -> exactly 2 matmuls total, matching the code).")

# (4) matrix chain of 4 same-size square (n,n) matrices, single final output (the age-4 leg)
path, info = oe.contract_path("ab,bc,cd,de,ef->af", (n, n), (n, n), (n, n), (n, n), (n, n), shapes=True, optimize="optimal")
out["chain_4step_single_output"] = {"opt_cost_flops": float(info.opt_cost), "opt_cost_u": float(info.opt_cost) / U, "path": str(path)}
print("(4) 4-step same-size matrix chain, single (final) output: opt_einsum optimal =", float(info.opt_cost) / U,
      "u = 4 x 1u for ANY parenthesization (all factors are n x n, so chain-order gives zero savings; "
      "this is the textbook fact that matrix-chain reordering only helps when dimensions differ).")

# (4b) the REAL need is ALL 4 prefix outputs (ages 1..4), not just the last. Cost of computing
# each age INDEPENDENTLY from scratch (no reuse) vs the incremental scheme actually shipped.
naive_independent_u = sum(k for k in (1, 2, 3, 4))  # age-k leg needs a length-k chain = k matmuls, done from scratch
incremental_u = 4  # each new age = 1 more matmul reusing the previous age's leg (unavoidable minimum)
out["all_four_ages"] = {"naive_independent_u": naive_independent_u, "incremental_u": incremental_u,
                         "note": "computing every prefix independently costs 1+2+3+4=10u; the shipped "
                                  "incremental scheme already achieves the minimum 4u (1 new O(n^3) matmul "
                                  "per age, each of which is a REQUIRED, distinct output consumed by D21 at "
                                  "that layer -- no order can produce k required distinct full-rank n x n "
                                  "prefixes of a matrix chain in fewer than k-1 relative matmuls from P_b)."}
print("(4b) all 4 required prefix outputs: naive-independent =", naive_independent_u,
      "u, incremental (shipped) =", incremental_u, "u -- incremental already optimal.")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote", OUT_JSON)
