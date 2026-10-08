"""Round 17 support (8 Oct): exact numerical check of the curvature-birth cumulant identities sent to Astra.
Uses an EMPIRICAL distribution (M weighted points, non-Gaussian, correlated), for which every joint cumulant is computed exactly
from the partition formula, so the Leonov-Shiryaev identities must hold to rounding. Checks:
  (1) k(u_a^2, u_b, u_c, u_d) = k5_aabcd + 2 [C_ab T_acd + C_ac T_abd + C_ad T_abc]
  (2) k(u_a^2, u_b^2, u_c, u_d) = k6_aabbcd + 2 C_ac K_abbd + 2 C_ad K_abbc + 2 C_bc K_aabd + 2 C_bd K_aabc + 4 C_ab K_abcd
                                  + 4 T_abc T_abd + 2 T_aab T_bcd + 2 T_abb T_acd + 4 C_ab (C_ac C_bd + C_ad C_bc)
  on all-distinct and on coinciding index tuples;
  (3) readout factors: sum_abcd W_ai W_bi W_ci W_di [Q1 C.T part] = 24 sum_a g_a W_ai (C U)_ai T(a; U_i, U_i), U = diag(Phi) W;
      and the Q2 Gaussian path part = 48 v_i^T C v_i with v_a = g_a W_ai (C U)_ai.
Usage: python experiments/probes/r17_curvature_identities.py
"""
import numpy as np, itertools, json

rng = np.random.default_rng(17)
n, M = 6, 40
G = rng.standard_normal((M, n)) @ rng.standard_normal((n, n))
X = np.maximum(G, 0) + 0.3 * G ** 2 * np.sign(G[:, ::-1]) + 0.2 * rng.standard_normal((M, n))   # skewed, heavy, correlated
p = rng.random(M); p /= p.sum()                                                                   # exact empirical measure
u = X - p @ X                                                                                      # centred


def partitions(s):
    if not s:
        yield []
        return
    first, rest = s[0], s[1:]
    for part in partitions(rest):
        for k in range(len(part)):
            yield part[:k] + [[first] + part[k]] + part[k + 1:]
        yield [[first]] + part


FACT = [1, 1, 2, 6, 24, 120, 720]


def joint_cum(cols):
    """Exact joint cumulant of the given list of (M,) arrays under the measure p."""
    k = len(cols); tot = 0.0
    for part in partitions(list(range(k))):
        m = len(part)
        term = (-1) ** (m - 1) * FACT[m - 1]
        for B in part:
            term *= p @ np.prod([cols[i] for i in B], axis=0)
        tot += term
    return tot


cum = lambda *idx: joint_cum([u[:, i] for i in idx])
C = np.array([[cum(a, b) for b in range(n)] for a in range(n)])
T = np.zeros((n,) * 3)
for a, b, c in itertools.product(range(n), repeat=3):
    T[a, b, c] = cum(a, b, c)
res = {"identity_Q1": [], "identity_Q2": [], "identity_Cu1": [], "identity_Cu1_He3": []}
tuples = [(0, 1, 2, 3), (2, 4, 1, 5), (0, 0, 1, 2), (1, 2, 2, 3), (3, 1, 3, 1), (4, 4, 4, 0)]
for a, b, c, d in tuples:
    lhs = joint_cum([u[:, a] ** 2, u[:, b], u[:, c], u[:, d]])
    rhs = cum(a, a, b, c, d) + 2 * (C[a, b] * T[a, c, d] + C[a, c] * T[a, b, d] + C[a, d] * T[a, b, c])
    res["identity_Q1"].append(dict(idx=[a, b, c, d], lhs=lhs, rhs=rhs, rel=abs(lhs - rhs) / max(abs(lhs), 1e-300)))
    K = lambda *i: cum(*i)
    lhs2 = joint_cum([u[:, a] ** 2, u[:, b] ** 2, u[:, c], u[:, d]])
    rhs2 = (cum(a, a, b, b, c, d)
            + 2 * C[a, c] * K(a, b, b, d) + 2 * C[a, d] * K(a, b, b, c) + 2 * C[b, c] * K(a, a, b, d) + 2 * C[b, d] * K(a, a, b, c)
            + 4 * C[a, b] * K(a, b, c, d)
            + 4 * T[a, b, c] * T[a, b, d] + 2 * T[a, a, b] * T[b, c, d] + 2 * T[a, b, b] * T[a, c, d]
            + 4 * C[a, b] * (C[a, c] * C[b, d] + C[a, d] * C[b, c]))
    res["identity_Q2"].append(dict(idx=[a, b, c, d], lhs=lhs2, rhs=rhs2, rel=abs(lhs2 - rhs2) / max(abs(lhs2), 1e-300)))
    # cubic piece (8 Oct): k(u_a^3, u_b, u_c, u_d) and the He3 form k(u_a^3 - 3 v_a u_a, u_b, u_c, u_d)
    va = C[a, a]
    lhs3 = joint_cum([u[:, a] ** 3, u[:, b], u[:, c], u[:, d]])
    star = 6 * C[a, b] * C[a, c] * C[a, d]
    ck = 3 * (C[a, b] * K(a, a, c, d) + C[a, c] * K(a, a, b, d) + C[a, d] * K(a, a, b, c))
    tt = 3 * (T[a, b, c] * T[a, a, d] + T[a, b, d] * T[a, a, c] + T[a, c, d] * T[a, a, b])
    rhs3 = cum(a, a, a, b, c, d) + 3 * va * K(a, b, c, d) + ck + tt + star
    res["identity_Cu1"].append(dict(idx=[a, b, c, d], lhs=lhs3, rhs=rhs3, rel=abs(lhs3 - rhs3) / max(abs(lhs3), 1e-300)))
    lhs4 = joint_cum([u[:, a] ** 3 - 3 * va * u[:, a], u[:, b], u[:, c], u[:, d]])
    rhs4 = cum(a, a, a, b, c, d) + ck + tt + star          # the -3 v_a u_a part cancels the 3 v_a K_abcd term exactly
    res["identity_Cu1_He3"].append(dict(idx=[a, b, c, d], lhs=lhs4, rhs=rhs4, rel=abs(lhs4 - rhs4) / max(abs(lhs4), 1e-300)))

# (3) readout factors on a full tensor
W = rng.standard_normal((n, n)) / np.sqrt(n)
Phi = rng.uniform(0.2, 0.9, n); g = rng.uniform(0.1, 0.6, n)
U = Phi[:, None] * W
# Q1 C.T part of kappa(h_a,h_b,h_c,h_d): quadratic piece at each of the 4 positions
base = 2 * (np.einsum("ab,acd->abcd", C, T) + np.einsum("ac,abd->abcd", C, T) + np.einsum("ad,abc->abcd", C, T))
q1 = np.einsum("a,b,c,d,abcd->abcd", g, Phi, Phi, Phi, base)
Q1 = sum(np.transpose(q1, perm) for perm in [(0, 1, 2, 3), (1, 0, 2, 3), (1, 2, 0, 3), (1, 2, 3, 0)])   # Q at position 1..4
lhs3 = np.einsum("ai,bi,ci,di,abcd->i", W, W, W, W, Q1)
CU = C @ U
rhs3 = 24 * np.einsum("a,ai,ai,acd,ci,di->i", g, W, CU, T, U, U)
# Q2 Gaussian path part: quadratic pieces at two of the four positions (6 choices)
path = 4 * np.einsum("ab,ac,bd->abcd", C, C, C) + 4 * np.einsum("ab,ad,bc->abcd", C, C, C)
q2 = np.einsum("a,b,c,d,abcd->abcd", g, g, Phi, Phi, path)
Q2 = np.zeros_like(q2)
for pos in itertools.combinations(range(4), 2):                # positions holding the two quadratic pieces
    rest = [k for k in range(4) if k not in pos]
    perm = np.argsort(list(pos) + rest)                          # axis order so that q2's (a,b,c,d) land on (pos, rest)
    Q2 += np.transpose(q2, perm)
lhs4 = np.einsum("ai,bi,ci,di,abcd->i", W, W, W, W, Q2)
V = g[:, None] * W * CU
rhs4 = 48 * np.einsum("ai,ab,bi->i", V, C, V)
cst = rng.uniform(-0.3, 0.3, n)
star_t = 6 * np.einsum("ab,ac,ad->abcd", C, C, C)
qs = np.einsum("a,b,c,d,abcd->abcd", cst, Phi, Phi, Phi, star_t)
QS = sum(np.transpose(qs, perm) for perm in [(0, 1, 2, 3), (1, 0, 2, 3), (1, 2, 0, 3), (1, 2, 3, 0)])
lhs5 = np.einsum("ai,bi,ci,di,abcd->i", W, W, W, W, QS)
rhs5 = 24 * np.einsum("a,ai,ai->i", cst, W, CU ** 3)
res["readout_star_24"] = dict(max_rel=float(np.max(np.abs(lhs5 - rhs5) / np.abs(lhs5))))
print("readout Gaussian star factor 24: max rel err", res["readout_star_24"]["max_rel"])
res["readout_CT_24"] = dict(max_rel=float(np.max(np.abs(lhs3 - rhs3) / np.abs(lhs3))))
res["readout_path_48"] = dict(max_rel=float(np.max(np.abs(lhs4 - rhs4) / np.abs(lhs4))))
for k in ("identity_Q1", "identity_Q2", "identity_Cu1", "identity_Cu1_He3"):
    print(k, "max rel err", max(r["rel"] for r in res[k]), [f"{r['lhs']:.4e}" for r in res[k]])
print("readout C.T factor 24: max rel err", res["readout_CT_24"]["max_rel"])
print("readout Gaussian path factor 48: max rel err", res["readout_path_48"]["max_rel"])
json.dump(res, open("results/summaries/r17_curvature_identities.json", "w"), indent=1)
