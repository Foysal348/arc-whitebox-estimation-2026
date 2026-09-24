"""Round 9 follow-up (GPT 5.6 Sol's shift-stable coherent carrier, 24 Sep): is the first Age-Gram mode transportable?

Sol's law: S_l = sum_a v_l(a) K_{l,a} with v_l the first eigenvector of the age Gram H at layer l; if
v_{l+1}(a+1) ~ rho v_l(a) then S_{l+1}^old ~ rho T_l S_l (one object carried across layers, infinite horizon).
Clean falsifier on the dense captures (L14 -> L15 are consecutive; the source of age a at L14 is age a+1 at L15):
    Y = sum_{a>=2} v15(a) D15_a   (actual old part of the L15 carrier)
    X = sum_{a>=1} v14(a) D15_{a+1}   (the L14 carrier transported exactly - D15_{a+1} IS the transported source)
    eps_shift = min_rho ||Y - rho X|| / ||Y||.   Gate: <= 2% important, 2-5% backbone, > 5% kill.
Control: uniform weights (the TOTAL itself) are shift-stable by construction, because transport is linear and
identical for every source; reported so the carrier is compared against the object the chain already carries.

Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/r9_shift_carrier.py
"""
import os, json, numpy as np
CAP = "../data/dumps"
src = open("../experiments/probes/r9_horizon_oracle.py", encoding="utf-8").read()
ns = {}; exec(src.split("out = []")[0], ns); blocks_of_source = ns["blocks_of_source"]

def per_age(l):
    c = np.load(os.path.join(CAP, f"r7_cap_L{l}.npz")); rres = int(c["rres"]); rfb = int(c["rfb"]); k = c["A_st"].shape[0]; n = c["A_st"].shape[1]
    off = ~np.eye(n, dtype=bool); Ds = []
    for s in range(k):
        sl = np.zeros((n, n))
        for (X, P, Y) in blocks_of_source(c, s, rres, rfb, n):
            sl += ((X * P) @ Y.T + (P * Y) @ X.T + (X * Y) @ P.T) / 3.0
        Ds.append(sl[off])
    return np.stack(Ds[::-1])                                   # row a-1 = age a

D14, D15 = per_age(14), per_age(15)
k14, k15 = D14.shape[0], D15.shape[0]
print(f"ages: L14 {k14}, L15 {k15}", flush=True)
def first_mode(D):
    lam, V = np.linalg.eigh(D @ D.T); v = V[:, -1]; return v * np.sign(v.sum())
v14, v15 = first_mode(D14), first_mode(D15)
m = min(k14, k15 - 1)
Y = v15[1:m + 1] @ D15[1:m + 1]                                 # ages 2..m+1 at L15
X = v14[:m] @ D15[1:m + 1]                                      # L14 weights on the same transported sources
rho = float(X @ Y / (X @ X)); eps = float(np.linalg.norm(Y - rho * X) / np.linalg.norm(Y))
coef_mis = float(np.linalg.norm(v15[1:m + 1] - rho * v14[:m]) / np.linalg.norm(v15[1:m + 1]))
# how well does the carrier (plus a fitted birth term) reproduce the TOTAL D21 at L15 (what the estimator needs)?
tot = D15.sum(0); B = np.stack([X, D15[0]]); cf = np.linalg.lstsq(B.T, tot, rcond=None)[0]
eps_tot = float(np.linalg.norm(tot - cf @ B) / np.linalg.norm(tot))
rec = {"k14": k14, "k15": k15, "rho": rho, "eps_shift": eps, "coef_mismatch": coef_mis, "eps_total_from_carrier_plus_birth": eps_tot,
       "v14": [float(x) for x in v14], "v15": [float(x) for x in v15], "control_uniform_eps": 0.0}
print(f"rho* = {rho:.4f}; eps_shift = {eps:.4f} (weight-vector mismatch {coef_mis:.4f})", flush=True)
print(f"TOTAL D21 at L15 from (carrier, birth) with 2 fitted scalars: rel err {eps_tot:.4f}  | uniform-weight carrier (= total): 0 by construction", flush=True)
print("v14: " + " ".join(f"{x:+.3f}" for x in v14)); print("v15: " + " ".join(f"{x:+.3f}" for x in v15))
json.dump(rec, open("../results/summaries/r9_shift_carrier.json", "w"), indent=1)
