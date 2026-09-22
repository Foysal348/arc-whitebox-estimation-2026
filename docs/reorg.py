"""Reorganize the project into a professional repository layout. Moves only (nothing is deleted), then rewrites the
relative paths inside the moved scripts so they keep working from their documented working directories.
Usage: python reorg.py <project_root> [--apply]
"""
import os, sys, shutil, re, json
ROOT = sys.argv[1]; APPLY = "--apply" in sys.argv
os.chdir(ROOT)

DIRS = ["docs", "docs/reference", "writeup", "submissions", "submissions/packages", "experiments", "experiments/probes",
        "experiments/chain", "experiments/variants", "kaggle", "kaggle/runners", "kaggle/estimators", "kaggle/k4",
        "kaggle/dumps", "kaggle/onestep", "kaggle/kprop", "results", "results/summaries", "results/logs",
        "consultations", "consultations/prompts", "correspondence", "data", "data/dumps", "data/kaggle_out"]

PROBES = ["probe_rank.py", "probe_query.py", "probe_query_full.py", "probe_query_analyze.py", "probe_agg.py", "probe_gem.py",
          "probe_loopholes.py", "probe_columns.py", "probe_colmask.py", "probe_young_dump.py", "probe_srccode.py",
          "probe_birth_dump.py", "probe_birthcomp.py", "probe_isdf.py", "probe_aap_mpp.py", "probe_adjoint_rank.py",
          "probe_adjoint_val.py", "probe_jac_val.py", "probe_d21late.py", "probe_tt_rank.py", "sens_probe.py"]
CHAIN = ["knob_test.py", "pool_census.py", "resid_ab.py", "run_v29_phase1_256.py", "residual_fit.py", "make_v31ang.py",
         "k4_law_fit.py", "k4_law_fit2.py", "k4_law_coef.py", "k4_oracle_run.py", "poll_submission.py"]
SUBMISSIONS = {"submission_v29": "submissions/v29", "submission_v29m8": "submissions/v29m8", "submission_v29m8f5": "submissions/v29m8f5",
               "submission_v31ang": "submissions/v31ang", "submission_v31b": "submissions/v31b"}
KAGGLE = {"kaggle_runner": "kaggle/runners/runner_a", "kaggle_runner_b": "kaggle/runners/runner_b", "kaggle_runner_c": "kaggle/runners/runner_c",
          "kaggle_estimators": "kaggle/estimators/set_a", "kaggle_estimators_b": "kaggle/estimators/set_b", "kaggle_estimators_c": "kaggle/estimators/set_c",
          "kaggle_k4code": "kaggle/k4/code", "kaggle_k4run": "kaggle/k4/run_synthetic", "kaggle_k4run2": "kaggle/k4/run_public", "kaggle_k4teach": "kaggle/k4/local",
          "kaggle_dump": "kaggle/dumps/driver", "kaggle_dumpcode": "kaggle/dumps/code", "kaggle_whest-dump-a": "kaggle/dumps/kernel_a",
          "kaggle_whest-dump-b": "kaggle/dumps/kernel_b", "kaggle_onestep": "kaggle/onestep/kernel", "kaggle_onestepcode": "kaggle/onestep/code",
          "kaggle_kprop": "kaggle/kprop"}
CONSULT = {"AI Assistant Suggession": "consultations/ai-assistant-suggestions"}
PROMPTS = ["ASTRA6_prompt_questionA.md", "ROUND2_prompt_new_formulations.md", "ROUND3_prompt_propagator_structure.md"]
CORRESP = ["K2_security_report_email.md", "K3_forum_reply_draft.md", "K3_forum_reply_plaintext.md"]
DOCS = {"ARC_WhestBench_2026_Complete_Notes.pdf": "docs/ARC_WhestBench_2026_Complete_Notes.pdf",
        "Research Paper/ARC Research Paper.pdf": "docs/reference/ARC_Research_Paper.pdf",
        "lb.html": "docs/reference/leaderboard_snapshot.html", "page_.html": "docs/reference/challenge_page.html",
        "page_insights.html": "docs/reference/challenge_insights.html"}
moves = []
def plan(src, dst):
    if os.path.exists(src): moves.append((src, dst))
for d in DIRS: os.makedirs(d, exist_ok=True) if APPLY else None
for f in PROBES: plan(f, f"experiments/probes/{f}")
for f in CHAIN: plan(f, f"experiments/chain/{f}")
for s, d in SUBMISSIONS.items(): plan(s, d)
for s, d in KAGGLE.items(): plan(s, d)
for s, d in CONSULT.items(): plan(s, d)
for f in PROMPTS: plan(f, f"consultations/prompts/{f}")
for f in CORRESP: plan(f, f"correspondence/{f}")
for s, d in DOCS.items(): plan(s, d)
plan("UNTESTED_IDEAS.md", "writeup/UNTESTED_IDEAS.md")
plan("submission_v25_estimator.py", "submissions/packages/estimator_v25_reference.py")
plan("scratch_est", "experiments/variants")
# packaged submissions
for f in os.listdir("."):
    if f.endswith(".tar.gz"): plan(f, f"submissions/packages/{f}")
    elif f.endswith(".npz"): plan(f, f"data/dumps/{f}")
    elif f.startswith("kaggle_out"): plan(f, f"data/kaggle_out/{f}")
    elif f.endswith(".log"): plan(f, f"results/logs/{f}")
    elif f.endswith(".json") and f not in ("variants.json",): plan(f, f"results/summaries/{f}")
    elif f in ("_dumptest", "_dumptest2", "_dumptest3"): plan(f, f"data/{f}")
    elif f.endswith(".html"): plan(f, f"docs/reference/{f}")
    elif f.endswith(".npy"): plan(f, f"data/dumps/{f}")
print(f"{len(moves)} moves planned")
if APPLY:
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print("SKIP (exists):", dst); continue
        shutil.move(src, dst)
    print("moves done")
    # ---- path rewrites inside the moved scripts ----
    TOKENS = ["query_dump_", "young_dump_", "gem_dump_", "gem_weights", "birth_dump", "adjoint_val_preds", "jac_val",
              "knob_", "colmask_results", "sens_results", "d21late_results", "k4oracle_", "phase1_256_", "mlps_used"]
    KOUT = ["kaggle_out_dumps", "kaggle_out_k4pub", "kaggle_out_k4", "kaggle_out_ang", "kaggle_out_row", "kaggle_out_gamma",
            "kaggle_out_age2", "kaggle_out_angtune", "kaggle_out_onestep", "kaggle_out_a1", "kaggle_out"]
    changed = []
    for base, _, files in os.walk("experiments"):
        for fn in files:
            if not fn.endswith(".py"): continue
            p = os.path.join(base, fn); s = open(p, encoding="utf-8").read(); o = s
            for old, new in [("../submission_v29m8f5/", "../submissions/v29m8f5/"), ("../submission_v29m8/", "../submissions/v29m8/"),
                             ("../submission_v29/", "../submissions/v29/"), ("../submission_v31ang/", "../submissions/v31ang/"),
                             ("../scratch_est/", "../experiments/variants/"), ("scratch_est/", "experiments/variants/"),
                             ("submission_v29m8f5/", "submissions/v29m8f5/"), ("submission_v29m8/", "submissions/v29m8/"),
                             ("submission_v31ang/", "submissions/v31ang/")]:
                s = s.replace(old, new)
            for t in TOKENS:
                s = s.replace(f'"../{t}', f'"../data/dumps/{t}').replace(f"'../{t}", f"'../data/dumps/{t}")
                s = s.replace(f'f"../{t}', f'f"../data/dumps/{t}')
                s = re.sub(r'(["\'])(' + re.escape(t) + r')', r'\1data/dumps/\2', s) if base.endswith("probes") or base.endswith("chain") else s
            for k in KOUT:
                s = s.replace(f'"{k}', f'"data/kaggle_out/{k}').replace(f"'{k}", f"'data/kaggle_out/{k}")
            s = s.replace('data/dumps/data/dumps/', 'data/dumps/').replace('../data/dumps/data/dumps/', '../data/dumps/')
            s = s.replace('data/kaggle_out/data/kaggle_out/', 'data/kaggle_out/')
            if s != o:
                open(p, "w", encoding="utf-8").write(s); changed.append(p)
    print(f"path rewrites in {len(changed)} files")
    json.dump({"moves": moves, "rewritten": changed}, open("docs/reorg_manifest.json", "w"), indent=1)
else:
    for s, d in moves[:15]: print(" ", s, "->", d)
