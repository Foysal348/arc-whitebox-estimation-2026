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
STATUS = {"running": ("চলছে", "#1565c0"), "open": ("বাকি", "#6d4c41"), "success": ("সফল", "#2e7d32"), "dead": ("মৃত", "#9e9e9e")}
PRIO = {"P1": ("P1 — এখনই (সস্তা, বড় সম্ভাবনা)", "#c62828"), "P2": ("P2 — P1-এর পরে", "#ef6c00"), "P3": ("P3 — শর্তসাপেক্ষ / দামি", "#5d4037")}
e = lambda s: html.escape(str(s))

def card(it):
    st, stc = STATUS.get(it["status"], (it["status"], "#000"))
    rows = [("কী করতে হবে", it["what"]), ("প্রমাণ / এখন পর্যন্ত", it["evidence"]), ("খরচ", it["cost"]),
            ("পাস করার শর্ত", it["gate"]), ("পরের ধাপ", it["next"]), ("File", it["files"])]
    body = "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in rows if v and v != "-")
    return (f'<div class="card"><div class="hd"><span class="id">{e(it["id"])}</span> {e(it["title"])}'
            f'<span class="st" style="background:{stc}">{e(st)}</span></div>'
            f'<div class="src">উৎস: {e(it["source"])}</div><table>{body}</table></div>')

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
doc = f"""<!doctype html><html lang="bn"><head><meta charset="utf-8"><title>Experiment Queue</title><style>
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
<div class="meta">হালনাগাদ: {e(Q["updated"])} · {e(Q["baseline"])}<br>{e(Q["headroom"])}</div>
<p>সব AI model-এর (Astra, Fable, GPT…) আর নিজেদের যে ধারণাগুলো পরীক্ষা করার মতো, সেগুলো এখানে priority অনুযায়ী সাজানো।
P1 = সস্তা আর বড় সম্ভাবনা, এখনই করতে হবে; P2 = P1-এর পরে; P3 = দামি বা শর্তসাপেক্ষ। মৃত (পরীক্ষা করে ব্যর্থ) ধারণাগুলো শেষে আছে, যাতে আবার সময় নষ্ট না হয়।</p>
<h2 style="border-color:#37474f">সংক্ষেপ (বাকি ও চলমান)</h2>
<table class="sum"><tr><th>ID</th><th>Priority</th><th>Status</th><th>বিষয়</th><th>খরচ</th></tr>{summary}</table>
{''.join(secs)}
<h2 style="border-color:#9e9e9e">শেষ হয়েছে (সফল / মৃত)</h2><div class="dead">{''.join(card(i) for i in done)}</div>
</body></html>"""
hp = os.path.join(HERE, "Experiment_Queue.html"); pdf = os.path.join(HERE, "Experiment_Queue.pdf")
open(hp, "w", encoding="utf-8").write(doc)
import tempfile
prof = tempfile.mkdtemp(prefix="edgeq_")
r = subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file:" + __import__("urllib.request").request.pathname2url(hp)],
                   capture_output=True, text=True, timeout=120)
if not os.path.exists(pdf): sys.exit("PDF not written: " + r.stderr[-500:])
print("wrote", pdf, os.path.getsize(pdf), "bytes")
