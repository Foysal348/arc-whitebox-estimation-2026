"""Render docs/moonshot/Moonshot_Blueprint.html to PDF with headless Edge (run from PowerShell; Edge is blocked in the Bash sandbox)."""
import os, sys, subprocess, tempfile, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
hp = os.path.join(HERE, "Moonshot_Blueprint.html"); pdf = os.path.join(HERE, "Moonshot_Blueprint.pdf")
if os.path.exists(pdf): os.remove(pdf)
prof = tempfile.mkdtemp(prefix="edgem_")
r = subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", "file:" + urllib.request.pathname2url(hp)], capture_output=True, text=True, timeout=120)
if not os.path.exists(pdf): sys.exit("PDF not written: " + r.stderr[-500:])
print("wrote", pdf, os.path.getsize(pdf), "bytes")
