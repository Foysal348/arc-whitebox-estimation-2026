To: arc-whestbench@aicrowd.com
Subject: [Security] Submission-status API returns other participants' scores and grading messages (ARC White-Box Estimation Challenge 2026)

Hello AIcrowd / ARC team,

I am a Phase 2 participant (AIcrowd username: foyshal_sk, participant id 524767). While polling the grading status of my own submission through the endpoint the whestbench CLI uses, I found that the endpoint returns data for submissions that are not mine. Reporting privately rather than on the forum because it is an access-control issue.

What I observed (20 Sep 2026, ~20:55 UTC)
- Endpoint: GET https://www.aicrowd.com/api/v1/submissions/{id}
- Without an Authorization header: 401 "HTTP Token: Access denied" (correct).
- With my own participant token (Authorization: Token <my key>): 200 for submission ids that belong to other participants, returning: id, challenge_id, challenge_round_id, created_at, grading_status_cd, grading_message, score, score_secondary.
- Example (not my submission): id 331685 -> grading_status_cd "failed", grading_message "<p>Error : Prohibited file in submission</p>". Ids 331682-331684 (also not mine) returned "graded" with their score and score_secondary values.
- whestbench's aicrowd_client.py docstring states this endpoint is "participant-token auth, authorized to the caller's own submission", so the server-side check appears to be missing or scoped only to authentication, not ownership.

Why it matters
- grading_message leaks error text of other teams' failed submissions (the same class of leak reported on the forum on 9 Sep 2026, "Errors are leaking information", for the global submission feed, which you said was RBAC-protected on the details page).
- score / score_secondary expose every graded submission of every team, not only the best-per-team values shown on the public leaderboard, and ids are sequential, so a participant can enumerate all submissions, their timing and their per-attempt scores.

Scope of my access
I fetched five ids adjacent to my own (331682-331686) once, to identify which id was mine after `whest submit --watch` timed out, and one repeat of id 331685 to confirm the fields for this report. I did not store, share or use any of that data beyond this email.

Minimal reproduction (uses the key stored by `whest login`, prints no secrets)
```python
import httpx
from whestbench.aicrowd_config import load_api_key
url = "https://www.aicrowd.com/api/v1/submissions/331685"   # any submission id that is not the caller's
print(httpx.get(url).status_code)                                                   # 401
r = httpx.get(url, headers={"Authorization": f"Token {load_api_key()}"})
print(r.status_code, sorted(r.json().keys()), r.json().get("grading_message"))     # 200, [...], other team's error text
```

Suggested fix
Return 403/404 unless the submission belongs to the caller's participant or team (the same ownership check the submission details page applies), and consider whether the leaderboard's public best-score rows should be the only per-submission scores readable cross-team.

Happy to re-test once a fix is deployed. Thank you for running the challenge.

Best regards,
<Foysal Emon Shanto>
AIcrowd: foyshal_sk
