"""Build the experiment queue PDF (Bengali + English) from queue.json.

Every idea worth testing from any AI consultation (Astra, Fable, GPT, ...) or from our own work goes into queue.json with
a priority (P1 = cheap and large potential, do now; P2 = after P1 or moderate cost; P3 = expensive or conditional) and a
status (running / open / success / dead). This script renders it to HTML with a Bengali-capable font and prints it to
PDF with headless Microsoft Edge (ReportLab cannot shape Bengali script).

Usage (from PowerShell - the Bash tool sandbox blocks the headless Edge print):  python docs/experiment_queue/build_queue.py
"""
import json, os, html, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
Q = json.load(open(os.path.join(HERE, "queue.json"), encoding="utf-8"))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
STATUS = {"running": ("Running", "#1565c0"), "open": ("Open", "#6d4c41"), "success": ("Success", "#2e7d32"), "dead": ("Dead", "#9e9e9e")}
PRIO = {"P1": ("P1 — Now (cheap, high potential)", "#c62828"), "P2": ("P2 — After P1", "#ef6c00"), "P3": ("P3 — Conditional / expensive", "#5d4037")}
e = lambda s: html.escape(str(s))

def card(it):
    st, stc = STATUS.get(it["status"], (it["status"], "#000"))
    rows = [("What to do", it["what"]), ("Evidence / so far", it["evidence"]), ("Cost", it["cost"]),
            ("Gate to pass", it["gate"]), ("Next step", it["next"]), ("File", it["files"])]
    body = "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in rows if v and v != "-")
    return (f'<div class="card"><div class="hd"><span class="id">{e(it["id"])}</span> {e(it["title"])}'
            f'<span class="st" style="background:{stc}">{e(st)}</span></div>'
            f'<div class="src">Source: {e(it["source"])}</div><table>{body}</table></div>')

active = [i for i in Q["items"] if i["status"] in ("running", "open")]
done = [i for i in Q["items"] if i["status"] in ("success", "dead")]
secs = []
for p in ("P1", "P2", "P3"):
    its = [i for i in active if i["priority"] == p]
    its.sort(key=lambda i: (i["status"] != "running", i["id"]))
    if its:
        secs.append(f'<h2 style="border-color:{PRIO[p][1]}">{e(PRIO[p][0])}</h2>' + "".join(card(i) for i in its))
summary = "".join(f'<tr><td>{e(i["id"])}</td><td>{e(i["priority"])}</td><td>{e(STATUS[i["status"]][0])}</td><td>{e(i["title"])}</td><td>{e(i["cost"])}</td></tr>'
                  for i in sorted(active, key=lambda i: (i["priority"], i["status"] != "running", i["id"])))
doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Experiment Queue</title><style>
@page {{ size: A4; margin: 14mm 12mm; }}
body {{ font-family: 'Nirmala UI', 'Segoe UI', sans-serif; font-size: 10.5pt; color: #222; }}
h1 {{ font-size: 18pt; margin: 0 0 4px 0; }} h2 {{ font-size: 13pt; border-left: 6px solid; padding-left: 8px; margin-top: 18px; }}
.meta {{ color: #555; font-size: 9.5pt; margin-bottom: 8px; }}
.card {{ border: 1px solid #ccc; border-radius: 6px; padding: 8px 10px; margin: 8px 0; page-break-inside: avoid; }}
.hd {{ font-weight: 700; font-size: 11pt; }} .id {{ color: #fff; background: #37474f; padding: 1px 6px; border-radius: 4px; margin-right: 4px; }}
.st {{ float: right; color: #fff; padding: 1px 8px; border-radius: 10px; font-size: 9pt; }}
.src {{ color: #666; font-size: 9pt; margin: 2px 0 4px 0; }}
table {{ border-collapse: collapse; width: 100%; }} th {{ text-align: left; vertical-align: top; width: 22%; color: #444; font-weight: 600; padding: 2px 6px 2px 0; }}
td {{ padding: 2px 0; vertical-align: top; }}
.sum td, .sum th {{ border-bottom: 1px solid #ddd; padding: 3px 4px; font-size: 9.5pt; }} .sum th {{ width: auto; background: #eceff1; }}
.dead .card {{ opacity: 0.75; }}
</style></head><body>
<h1>Experiment Queue — ARC WhestBench 2026 (Phase 2)</h1>
<div class="meta">Updated: {e(Q["updated"])} · {e(Q["baseline"])}<br>{e(Q["headroom"])}</div>
<p>All the ideas worth testing, from any AI model (Astra, Fable, GPT…) or from our own work, are sorted here by priority.
P1 = cheap and high potential, do now; P2 = after P1; P3 = expensive or conditional. Dead ideas (tested and failed) are listed last, so time isn't wasted testing them again.</p>
<h2 style="border-color:#37474f">Summary (open and running)</h2>
<table class="sum"><tr><th>ID</th><th>Priority</th><th>Status</th><th>Topic</th><th>Cost</th></tr>{summary}</table>
{''.join(secs)}
<h2 style="border-color:#9e9e9e">Finished (success / dead)</h2><div class="dead">{''.join(card(i) for i in done)}</div>
</body></html>"""
hp = os.path.join(HERE, "Experiment_Queue.html"); pdf = os.path.join(HERE, "Experiment_Queue.pdf")
open(hp, "w", encoding="utf-8").write(doc)
import tempfile
prof = tempfile.mkdtemp(prefix="edgeq_")
r = subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file:" + __import__("urllib.request").request.pathname2url(hp)],
                   capture_output=True, text=True, timeout=120)
if not os.path.exists(pdf): sys.exit("PDF not written: " + r.stderr[-500:])
print("wrote", pdf, os.path.getsize(pdf), "bytes")
