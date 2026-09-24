"""Moonshot D1 (24 Sep): make the shared-basis JOIN cheap. Builds experiments/variants/estimator_v41join.py from v40fb.

Ledger (m1_matmul_ledger.py): the join machinery costs ~41 u of the 250 u - per join (10 joins) a rank-384 range
finder over the joiner's dense legs (4 products n x n x r = 1.5 u) + the old-core term (0.3 u), projections (0.75 u),
rotations of the old factors (~0.9 u), cores (~0.8 u).

Structure used: an old source's propagator is the newer one times earlier maps, P_old = P_new T, so the column space of
the transported basis already contains most of what the joiner needs. Knob V41_JOIN:
  rf        - shipped (range finder)                                  [default: bit-identical to v40fb]
  transport - Qn = orth(w1 * Qc): the previous basis, transported; no range finder (the first join still uses rf)
  aug<k>    - transported basis plus k fresh directions of the joiner (sketch width k), truncated back to r by the
              eigen-decomposition of the full core in the (r + k)-dim span
"""
SRC = "submissions/v40fb/estimator.py"; DST = "experiments/variants/estimator_v41join.py"
s = open(SRC, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]; s = s.replace(old, new)
rep('FB_SCALE = float(', 'JOIN = _os.environ.get("V41_JOIN", "rf")   # V41: basis rebuild at a join (rf | transport | aug<k>)\nFB_SCALE = float(')
rep("""                    for _pass in range(QPASS):
                        # Yq = G Om with G = Aj dA Aj^T + Pj dP Pj^T + Qp Sg Qp^T""",
"""                    _jmode = JOIN if ka > 0 else "rf"
                    if _jmode == "transport":
                        Qn, _ = fnp.linalg.qr(Qp)          # V41: previous basis carried forward, no range finder
                    elif _jmode.startswith("aug"):
                        _k = int(_jmode[3:])
                        Qb, _ = fnp.linalg.qr(Qp)
                        Omk = pool.get(("omk", _k), (n, _k)); fnp.copyto(Omk, w32[:, :_k])
                        # joiner's own dominant directions from a width-k sketch, deflated against the carried basis
                        Yk = fnp.matmul(Aj, dAj * fnp.matmul(Aj.T, Omk)) + fnp.matmul(Pj, dPj * fnp.matmul(Pj.T, Omk))
                        Yk = Yk - Qb @ (Qb.T @ Yk)
                        Qk, _ = fnp.linalg.qr(Yk)
                        Z = fnp.concatenate([Qb, Qk], axis=1)                  # (n, r + k) orthonormal
                        ZA = Z.T @ Aj; ZP = Z.T @ Pj; ZQ = Z.T @ Qp
                        Sfull_ = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                        Gz = (ZA * dAj.T) @ ZA.T + (ZP * dPj.T) @ ZP.T + ZQ @ (Sfull_ @ ZQ.T)
                        _ev, _V = fnp.linalg.eigh(Gz)
                        _Vr = _V[:, ::-1][:, :r_old]
                        Qn = Z @ _Vr
                        _reuse = (_Vr.T @ ZA, _Vr.T @ ZP)   # V41: joiner projections without a second n x n x r pass
                    for _pass in range(QPASS if _jmode == "rf" else 0):
                        # Yq = G Om with G = Aj dA Aj^T + Pj dP Pj^T + Qp Sg Qp^T""")
rep("""                    _jmode = JOIN if ka > 0 else "rf"
""", """                    _jmode = JOIN if ka > 0 else "rf"
                    _reuse = None
""")
rep("""                    FAj = fnp.matmul(Qn.T, Aj, out=fap_new[m1, 0])
                    FPj = fnp.matmul(Qn.T, Pj, out=fap_new[m1, 1])
                    if ka > 0:
                        Tq = Qn.T @ Qp""", """                    if _reuse is not None:
                        fnp.copyto(fap_new[m1, 0], _reuse[0]); fnp.copyto(fap_new[m1, 1], _reuse[1])
                        FAj = fap_new[m1, 0]; FPj = fap_new[m1, 1]
                    else:
                        FAj = fnp.matmul(Qn.T, Aj, out=fap_new[m1, 0])
                        FPj = fnp.matmul(Qn.T, Pj, out=fap_new[m1, 1])
                    if ka > 0:
                        Tq = Qn.T @ Qp""")
open(DST, "w", encoding="utf-8").write(s)
import ast; ast.parse(s); print("wrote", DST)
