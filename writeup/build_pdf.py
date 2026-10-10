"""Build writeup/ALGORITHM_TRACK_PAPER.pdf from the Markdown source (python-markdown + Edge headless print).
Usage: python writeup/build_pdf.py"""
import re, subprocess, os, markdown
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "ALGORITHM_TRACK_PAPER.md"), encoding="utf-8").read().splitlines()
item = re.compile(r"^\s*([-*+]|\d+\.)\s")
out, in_code = [], False
for line in src:
    if line.strip().startswith("```"): in_code = not in_code
    # python-markdown needs a blank line between a paragraph and the list that follows it
    if not in_code and item.match(line) and out and out[-1].strip() and not item.match(out[-1]) and not out[-1].startswith((" ", "\t", "|")):
        out.append("")
    if not in_code and re.match(r"^ {4}[-*] ", line):
        line = line[1:]            # 4-space sub-items under two-digit numbers -> 3 spaces (truly_sane nesting)
    out.append(line)
body = markdown.markdown("\n".join(out), extensions=["tables", "fenced_code", "toc", "mdx_truly_sane_lists"],
                         extension_configs={"mdx_truly_sane_lists": {"nested_indent": 2, "truly_sane": True}})
css = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'Cambria', 'Segoe UI', 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; margin: 0 0 6pt; } h2 { font-size: 14.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; page-break-after: avoid; } h4 { font-size: 11pt; margin: 10pt 0 4pt; }
p, li { text-align: justify; } ul, ol { padding-left: 18pt; margin: 3pt 0; } li { margin: 1.5pt 0; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #888; padding: 2.5pt 5pt; vertical-align: top; } th { background: #e8e8e8; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f5f5f5; border: 1px solid #ddd; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
"""
html_path = os.path.join(HERE, "ALGORITHM_TRACK_PAPER.html"); pdf_path = os.path.join(HERE, "ALGORITHM_TRACK_PAPER.pdf")
open(html_path, "w", encoding="utf-8").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Algorithm Track Paper - Hack2Publish</title><style>{css}</style></head><body>{body}</body></html>")
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", "file:///" + html_path.replace("\\", "/")], capture_output=True)
print("wrote", pdf_path, os.path.getsize(pdf_path))
