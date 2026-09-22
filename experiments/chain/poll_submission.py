"""Poll an AIcrowd submission until graded/failed. Usage: python poll_submission.py <id> [max_minutes]"""
import json, sys, time
from whestbench.aicrowd_client import AIcrowdClient
from whestbench.aicrowd_config import load_api_key

sid = int(sys.argv[1]); max_min = float(sys.argv[2]) if len(sys.argv) > 2 else 120
c = AIcrowdClient(api_key=load_api_key())
t0 = time.time(); last = None
while time.time() - t0 < max_min * 60:
    try:
        s = c.get_submission_status(sid)
        st = s.get("grading_status_cd")
        line = f"{time.strftime('%H:%M:%S')} status={st} score={s.get('score')} secondary={s.get('score_secondary')} msg={str(s.get('grading_message'))[:200]}"
        if line[9:] != (last or "")[9:]:
            print(line, flush=True); last = line
        if st in ("graded", "failed"):
            print(json.dumps({k: v for k, v in s.items() if k in ('id','grading_status_cd','score','score_secondary','grading_message','created_at','meta')}, default=str, indent=1))
            break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), "poll error:", repr(e)[:200], flush=True)
    time.sleep(60)
