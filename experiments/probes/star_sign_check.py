"""Sign of the Gaussian cubic star against the sampled kappa4 births (11 Oct 2026; settles the [OPEN] of paper §6.5 (c)).

Question: the star-fed kappa3 feed (star_J6) correlates -0.87 to -0.92 with the sampled born feed (bornJ6). Is the star's
sign a convention error, or is it physical?

Data: the 9 Oct kappa4 ceiling run (fable_ceiling_2026-10-09.py, 2e6 exact samples, MLP #0, targets L2-16, ages 1-6).
It stores per (target t, age j) the kappa4-diagonal readouts of the closed-form star and of the sampled one-vertex births:
- Cu1 = 4 k(C,A,A,A), the cubic-vertex birth (the star is its Gaussian part);
- Q2 = 6 k(B,B,A,A), the quadratic-vertex-squared birth (Gaussian part 6*8 g^2 v Ybar^2 X^2 >= 0 on the diagonal);
- births_exact, every birth.

Checks, summed over ages 1-6 at L9-16:
- corr(star, Cu1) > 0: the closed form has the sign of the sampled cubic birth, so there is no convention error;
- corr(star, Q2) < 0 and |Q2| > |star|: the quadratic birth has the opposite sign and dominates;
- corr(star + Q2, born) > 0 while corr(star, born) < 0.

Usage: python experiments/probes/star_sign_check.py [path to fable_ceiling_all.npz]
"""
import sys, json
import numpy as np

Z = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/kaggle_out/fable_ceiling_all/fable_ceiling_all.npz")


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


out = {}
for t in range(9, 17):
    S = lambda k: sum(Z[f"L{t}_j{j}_{k}"] for j in range(1, 7))
    s, cu, q2, be = S("star"), S("Cu1"), S("Q2"), S("births_exact")
    r = dict(corr_star_Cu1=corr(s, cu), corr_star_born=corr(s, be), corr_Q2_born=corr(q2, be), corr_starQ2_born=corr(s + q2, be),
             corr_star_Q2=corr(s, q2), norm_Q2_over_star=float(np.linalg.norm(q2) / np.linalg.norm(s)),
             mean_star=float(s.mean()), mean_Q2=float(q2.mean()), mean_born=float(be.mean()), frac_star_neg=float((s < 0).mean()))
    out[t] = r
    print(f"L{t}: corr(star,Cu1) {r['corr_star_Cu1']:+.3f} | corr(star,born) {r['corr_star_born']:+.3f} | corr(Q2,born) {r['corr_Q2_born']:+.3f} | "
          f"corr(star+Q2,born) {r['corr_starQ2_born']:+.3f} | corr(star,Q2) {r['corr_star_Q2']:+.3f} | |Q2|/|star| {r['norm_Q2_over_star']:.2f} | "
          f"star<0 on {100 * r['frac_star_neg']:.0f} % of neurons")
json.dump(out, open("results/summaries/star_sign_check.json", "w"), indent=1)
print("wrote results/summaries/star_sign_check.json")
