# Cost-first architecture request (10 Oct 2026): the same accuracy at 60 % of the FLOPs

## The problem in one paragraph
- **Task.** Estimate the final-layer post-ReLU means of a bias-free He-initialised ReLU MLP (width n = 1024, depth 16),
  input x ~ N(0, I).
- **Score.** Final MSE × max(0.1, FLOPs / 2^41). One u = 2^31 FLOPs, about one dense 1024³ product, so the budget is
  1024 u and the floor is 102 u.
- **Our estimator** is the public factorised third-cumulant chain (Wu et al., arXiv 2605.05179, §4.3; 504aldo's V29)
  plus our additions. It reaches raw MSE 1.97e-8 at 194 u (util 0.187), which ranks about #42.
- **The leaderboard says accuracy is not our gap; cost is:**

  | entry | raw MSE | util |
  |---|---|---|
  | #1 | 1.36e-8 | 0.107 |
  | #11 | 2.10e-8 | 0.102 (V29-level accuracy at the floor) |
  | top 25 | 1.1–2.0e-8 | 0.10–0.16 |

  At our MSE, util 0.10 would place about #9.

## Where our 194 u go (measured call-site ledger, one predict() on a public MLP)

| family | u |
|---|---|
| D21 slice readout: per-source hub contractions (X∘P)Yᵀ + (X∘Y)Pᵀ + (P∘Y)Xᵀ, Strassen level 5 | ≈ 70 |
| young-source leg transport: A, P ← (diag(w1) W)ᵀ ·, per source per layer, ages ≤ 5 | ≈ 63 |
| old sources in a shared rank-352 basis (basis rotation, factor transport, contractions) | ≈ 20 |
| sym products, joins, QR, einsums, covariance, closures | ≈ 40 |

Young sources: about 70 source-layers at about 1.9 u each (transport + readout). The young tier sits at its enumerated
floor for this representation.

## The representation (for reference)
- Each ReLU at layer b creates a source with dense legs [neuron, hub]:
  - the propagator P (identity at birth);
  - A = P diag(w1_b) C_off(b), which has a zero diagonal and a flat spectrum: 99 % of its energy needs 310–809
    directions.
- κ3 = Σ_s Sym(X1, P, Y1) + Sym(M, P, P) + Sym(X3, P, Y3), with X/Y built from A, P and thin legs.
- The next layer needs D21_ic = κ3(i, i, c) (n² numbers) and d3. D21 feeds the bivariate covariance closure, the
  newborn's repeated-index compensation, the thin feedback legs and the κ4 pattern features. It cannot be dropped.
- Sources older than 5 layers live in one shared basis.

## Measured dead ends for cost (do not re-propose)
- **One pool for the whole κ3** (young + old in a single basis of C's eigenvectors):
  - D21 error at L11–15: rank 384 → 7–11 %, rank 512 → 3.4–6.3 %, rank 768 → 0.2–1 %; the tolerance is about 2 %.
  - The readout needs neuron-space Hadamards, so legs must be lifted back to n and nothing is saved.
- **Old-only pooling at rank 384:** 1.3 %, which is what we already ship.
- **Rank or age truncation of young legs, birth-operator compression, query sketches (TensorSketch, ISDF, adjoint),
  slice-to-slice recursions, CP/Tucker/TT aggregate states, particles:** all fail by 5–50× at matched cost.
- **Cutting cost with accuracy knobs:** within this representation, every knob trades about 1 % FLOPs for 1 % MSE.
- **float16:** bills the same as float32.
- **Local channel corrections:** they repeatedly worsen the final mean because the chain's errors partly cancel; only
  joint input oracles gave large gains.

## Question
The leaders obtain 1.1–2.1e-8 at 0.10–0.14 util. 504aldo's guess is that their method "is not a per-source hub-tensor
expansion at all". Propose ONE representation of the κ3 information the mean closure actually needs that removes the
per-source n³ terms (transport and/or readout) while keeping the D21/d3 inputs within about 2 % at layers 9–16 (that is
where κ3 accuracy matters: perfect κ3 at L1–4 is worth ×1.00, at L13–16 ×0.81).

Give the representation, its per-layer FLOP count in u, the identity or approximation it rests on, and a cheap falsifier
we can run on stored data:
- teacher mean, C, κ3 slice and κ4 diagonal at all 16 layers of one public MLP;
- the chain's dense source legs at layers 3, 7, 11, 14 and 15.

Candidates welcome, if you can show the arithmetic:
- carrying D21 itself with an exact-up-to-order recursion plus a cheap correction;
- sharing one propagator across sources through the transport recursion;
- representing the young tier in hub-compressed form, given that the hub side is static after birth;
- reading out D21 without forming per-source Hadamards.

Mark everything as derived, estimated or speculative; we will run the measurements.
