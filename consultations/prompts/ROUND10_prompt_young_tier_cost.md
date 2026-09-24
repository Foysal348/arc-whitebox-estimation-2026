# Round 10 — ONE focused question: halve the cost of the young third-cumulant sources

*(Answer in English, as a PDF, at most 4 pages. One question only. Give one concrete method (or a proof that none of the
kind exists), its exact FLOP count, and one falsifier. Please do not re-propose anything in the "measured dead" list.)*

## Setting (short)

Fixed bias-free ReLU MLP, width n = 1024, depth 16, weights N(0, 2/n), input x ~ N(0, I). Estimate the final post-ReLU
means. Score = MSE × max(0.1, FLOPs/2⁴¹). 1 u = 2³¹ FLOPs = one dense n³ product; a level-5 Strassen square product
counts ≈ 0.51 u. Our estimator: 504aldo's factorised K = 3 cumulant chain (mean, covariance, κ₃ as a sum over sources,
memoryless κ₄, bivariate Wick/Edgeworth closure). A source born at a ReLU carries two n × n legs P (the propagator,
P ← W·diag(Φ)·P each layer) and a second leg A (A is a second dense leg, born as W·X_b (X_b an n×n birth matrix built from the covariance) and transported alongside P; measured on our captures, A and P are NOT related by any fixed right factor (P⁻¹A changes by 100–970% between consecutive layers), so both legs must be transported); the next closure needs
D21 = Σ_s [(P_s∘P_s)·diag(w_s)·A_sᵀ + 2(P_s∘A_s)·diag(w_s)·P_sᵀ]/3 and its diagonal. Sources older than 4 layers are
moved into a shared rank-384 basis (rank-224 sub-basis after age 7).

## The measured cost ledger (metered, per network; total ≈ 250 u, 88 % in matmul)

| family | u |
|---|---|
| **young sources (ages 1–4, dense legs): transport W·diag(Φ)·[P, A]** | **≈ 54** |
| **young sources: D21 contractions (Hadamard products × dense legs)** | **≈ 50** |
| joining a source to the shared basis (range finder, projections, rotations) | ≈ 41–46 |
| old sources in the basis (forming legs from factors + contractions) | ≈ 41 |
| thin feedback legs | ≈ 15 |
| covariance, closure, Strassen additions | ≈ 20 |

## Facts you must respect (all measured on real networks against Monte-Carlo teachers)

1. **Leaders' reach.** One leaderboard team reaches raw MSE 2.0e-8 at 0.126 × budget (≈ 129 u) — our accuracy (2.1e-8) at
   half our cost. So a ~2× cheaper method of equal accuracy exists; we do not know it.
2. **Every knob trades one-for-one.** On a deterministic 100-network harness (paired SE 0.01–0.15 %), every change inside
   the chain buys accuracy with cost at ~1 % per 1 % (age gates, basis ranks, subspace passes, birth-column screening,
   later confinement). Truncating the young legs (they are near full rank: 99 % energy needs 229–504 of 1024
   directions, and their Hadamard products are full rank with a flat spectrum) loses accuracy faster than it saves cost.
3. **The join is not the lever.** Rebuilding the shared basis from the transported old basis (no range finder) saves 6.6 %
   cost but loses 1–6 % accuracy; an eigen-truncated rebuild is 1 % more accurate but not cheaper.
4. **Where the error is.** The closures are exact given true inputs; the whole error is fresh per-layer error in the
   κ₃/κ₄ inputs that accumulates (the κ₃ slice D21 is 4–7 % wrong, from the all-distinct entries of the post-ReLU κ₃
   tensor). So a cheaper young tier must keep D21 within ~2 % relative error (tolerance law: extra final MSE ≈
   4.2e-6·ε² for a relative D21 error ε).
5. **Measured dead** (do not propose): rank / Tucker / TT / CP compression of the source state; sketching or sampling
   the birth index (Horvitz–Thompson included); aggregate or vector–metric states; slice recursions; low-rank covariance
   corrections; any Monte-Carlo or control-variate estimator (coupled surrogates correlate 0.19–0.73 with the needed
   statistics); learned corrections; mixed-coordinate re-ordering (same FLOPs); Strassen levels ≥ 6 (memory cap 8 GB).

## Scope (clarified 24 Sep)

Only the young tier changes: the young sources' (ages 1–4) contribution to D21 and its diagonal d3 must cost **≤ 55 u
in total over the network** (today ≈ 104 u), at ≤ 2% relative D21 error. Everything else stays as it is (join ≈ 41–46 u,
old tier ≈ 41 u, thin legs ≈ 15 u, covariance + closure ≈ 8 u, Strassen additions ≈ 12 u; κ4 is memoryless, no transport
cost). Success brings the total from ~249 u to ~200 u.

Already checked by us and found to save nothing (do not re-propose without an exact count that says otherwise):
- skipping A's transport: A is not a fixed function of P (measured), and forming it entry-wise costs n³ anyway;
- reusing P_old = P_new·M: forming it is itself an n³ product;
- batching the four ages into wider products: identical FLOPs;
- age-1 P = W: already used.
An exact contraction-order search (24 Sep) found NO cheaper order: the shipped forward order is at the floor (4 relative products per source lifetime for transport; the Hadamard hyperedge forces materialisation, and the code already fuses to 2 right-factor contractions); reverse-style suffix precompute costs +75%, per-age independent +150%. A cheaper young tier must change the representation, not the order.

Also measured (so a proposal cannot rely on them): the next layer's d3 is fed 90–99% by off-diagonal κ3; diagonal-only
transport, low-rank covariance, rank-1 κ3, slice-state recursions, deterministic particles and aggregate CP are all dead.

## The question

The young tier costs ≈ 104 u: four dense n×n products per source-layer (two transports, two contractions) for four
source ages at every layer. **Is there an algebraic reorganisation or a different representation of the ages-1–4 content
that produces the same D21 (to ≤ 2 % relative error) and its diagonal at ≤ 55 u in total?** For example (take or reject
with a reason): exploiting that consecutive young sources share propagator factors (P_{b} = P_{b+1}·M_{b+1}); computing
the age-1 and age-2 contributions in closed form from W, C and a_b without transporting their legs; batching the
contractions of all four ages into fewer products.

Deliverable: the method, its exact FLOP count per layer (count every product at its real shape; Strassen only for
square products), why its D21 error stays ≤ 2 %, and one falsifier we can run on stored legs (we have dense legs,
births a_b, weights and teacher slices at every layer).
