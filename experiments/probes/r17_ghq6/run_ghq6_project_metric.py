"""Runs the author's GHQ-6 carrier (round17_ghq6_falsifier.py, unmodified, imported) on the stored teacher and scores it with
(i) the author's metric (raw kappa4, decisive neurons) and (ii) the project metric (kappa4 / v^2 on decisive neurons), plus the
same V55-features combination test as for HTQB (leave-one-layer-out over the three targets is too small, so the map is fitted on
the other two targets). Usage (project root): python experiments/probes/r17_ghq6/run_ghq6_project_metric.py
"""
import importlib.util, json, numpy as np
spec = importlib.util.spec_from_file_location("ghq", "experiments/probes/r17_ghq6/round17_ghq6_falsifier.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
W, _ = G.load_weights(); T = G.load_teachers()
pa, p6 = G.build_carrier_predictions(W, T)
S = np.load("data/dumps/k4c_feats.npz")
c = lambda a, b, m: float(np.corrcoef(a[m], b[m])[0, 1])
out = {}
for l in G.TARGETS:
    v = np.diag(T[l]["C"]); dec = np.abs(T[l]["mean"] / np.sqrt(v)) < 1.5; k4 = T[l]["k4"]
    out[l] = {"author_raw_all_history": c(pa[l], k4, dec), "author_raw_age6": c(p6[l], k4, dec),
              "project_all_history": c(pa[l] / v ** 2, k4 / v ** 2, dec), "project_age6": c(p6[l] / v ** 2, k4 / v ** 2, dec)}
    print(f"L{l}: " + "  ".join(f"{k} {x:+.4f}" for k, x in out[l].items()), flush=True)
json.dump(out, open("results/summaries/r17_ghq6.json", "w"), indent=1)
