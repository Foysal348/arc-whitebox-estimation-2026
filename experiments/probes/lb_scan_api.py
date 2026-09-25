"""Scan public per-submission results of Phase 2 through the whestbench status API (organiser, 24 Sep: the same data is
public on the Submissions List page). One request per second, resumable; writes data/lb_scan/phase2_api.csv.
Usage (from whest-starterkit):  python -m uv run python ../experiments/probes/lb_scan_api.py 328000 332300
"""
import csv, os, sys, time
from whestbench.aicrowd_client import AIcrowdClient
from whestbench.aicrowd_config import load_api_key

lo, hi = int(sys.argv[1]), int(sys.argv[2])
out = "../data/lb_scan/phase2_api.csv"; os.makedirs(os.path.dirname(out), exist_ok=True)
done = set()
if os.path.exists(out):
    with open(out, newline="") as f:
        done = {int(r["id"]) for r in csv.DictReader(f)}
c = AIcrowdClient(api_key=load_api_key())
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["id", "round", "created_at", "status", "score", "raw", "message"])
    for sid in range(lo, hi):
        if sid in done: continue
        for attempt in range(5):
            try:
                s = c.get_submission_status(sid); break
            except Exception as e:
                s = {"error": repr(e)[:80]}
                if "429" in repr(e) or "Too Many" in repr(e): time.sleep(30 * (attempt + 1)); continue
                break
        w.writerow([sid, s.get("challenge_round_id"), s.get("created_at"), s.get("grading_status_cd"), s.get("score"),
                    s.get("score_secondary"), (str(s.get("grading_message") or s.get("error") or "")[:60]).replace("\n", " ")])
        f.flush()
        if sid % 200 == 0: print(time.strftime("%H:%M:%S"), sid, flush=True)
        time.sleep(1.0)
print("done", flush=True)
