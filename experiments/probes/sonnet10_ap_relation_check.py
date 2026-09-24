"""Round 10 (Sonnet, 24 Sep): does A = P . diag(a_b) hold for the shipped v40fb young-tier
legs, at real captured layers? The task background describes the source's two dense legs as
"P (propagator) and A = P . diag(a_b)". IF that held exactly and PERSISTED under transport
(it would, by associativity: if A_birth = P_birth . diag(a_b) and both A, P are transported by
the SAME left-multiplication chain W_l diag(Phi) each layer, then A_l = P_l . diag(a_b) at
every later layer too), the young-tier D21 contraction

    D21 += LA @ A^T + LP @ P^T                         (2 dense n x n x n matmuls / source / layer)

could be rewritten, using A^T = diag(a_b) @ P^T, as

    D21 += (LA * a_b[None, :] + LP) @ P^T               (1 dense n x n x n matmul / source / layer)

i.e. a further 2x reduction of the "D21 contraction ~50 u" line -- well above the round's 10%
bar. This script checks the premise directly against the captured dense legs
(data/dumps/r7_cap_L15.npz, real teacher-network run) and against the estimator's own birth
code (submissions/v40fb/estimator.py, read only): for every live source, is A[s] proportional
to P[s] column-by-column (same scalar for every row in a given column)?

Usage (from whest-starterkit): python -m uv run python ../experiments/probes/sonnet10_ap_relation_check.py
"""
import os, json
import numpy as np

CAP = "../data/dumps"
f64 = lambda x: np.asarray(x, np.float64)

out = {"layers": {}}
for l in (15,):
    c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz"))
    A = f64(c["A_st"]); P = f64(c["P_st"])
    k = A.shape[0]
    rows = []
    for s in range(k):
        Asub, Psub = A[s], P[s]
        mask = np.abs(Psub) > 1e-6 * np.abs(Psub).max()
        ratio = np.where(mask, Asub / np.where(mask, Psub, 1.0), np.nan)
        col_std = np.nanstd(ratio, axis=0)
        col_absmean = np.nanmean(np.abs(ratio), axis=0)
        rel_nonconstancy = float(np.nanmean(col_std / (col_absmean + 1e-12)))
        rows.append({"s": s, "age": k - s, "rel_col_nonconstancy": rel_nonconstancy,
                      "frob_A": float(np.linalg.norm(Asub)), "frob_P": float(np.linalg.norm(Psub))})
        print(f"layer {l} s={s} age={k-s}: rel col-nonconstancy of A/P = {rel_nonconstancy:.3f} "
              f"(0 => A=P.diag(a) holds exactly; O(1) => it does not)")
    out["layers"][l] = rows

# ---- also confirm from the code text itself: A at birth is w1col * C_off (a full (n,n)
# elementwise row-scaled COVARIANCE matrix), while P at birth is copied from raw W -- two
# a-priori unrelated dense matrices, not a column-diagonal rescaling of one another.
SRC = "../submissions/v40fb/estimator.py"
lines = open(SRC, encoding="utf-8").read().splitlines()
birth_a = next((ln for ln in lines if "a_b = fnp.multiply(w1col, C_off" in ln), None)
birth_p = next((ln for ln in lines if 'fnp.copyto(legs["AP0"][k, 1], W)' in ln), None)
out["birth_code"] = {"A_birth_line": birth_a, "P_birth_line": birth_p,
                      "note": "A_birth = diag(w1col) @ C_off (row-scaled covariance); "
                               "P_birth = W (raw weight matrix). Unrelated matrices -> "
                               "A = P.diag(a_b) does not hold in this shipped code, so the "
                               "LA@A^T + LP@P^T -> 1-matmul fusion is NOT available."}
print("\nbirth code check:")
print(" A_birth:", birth_a)
print(" P_birth:", birth_p)

os.makedirs("../results/summaries", exist_ok=True)
json.dump(out, open("../results/summaries/sonnet10_ap_relation_check.json", "w"), indent=1)
