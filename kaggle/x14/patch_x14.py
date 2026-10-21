"""Builds experiments/variants/e30k_x14.py = e30k_x12.py + X14 patches (kaggle/x14/x14_patches.py): N4 modulators
(zr, zc, rc2, ast, pdl, pst; computed only if the table uses them), dump keys for them, knob X14_T21 (default off)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import x14_patches
src = open(os.path.join(ROOT, "experiments/variants/e30k_x12.py"), encoding="utf-8").read()
src = x14_patches.apply(src, dump=True, t21_default="0")
src = src.replace("# e30k_x12 (20 Oct):", "# e30k_x14 (21 Oct): e30k_x12 + X14 (N4 modulators zr/zc/rc2/ast/pdl/pst, X14_T21 knob; kaggle/x14/patch_x14.py)\n# e30k_x12 (20 Oct):", 1)
open(os.path.join(ROOT, "experiments/variants/e30k_x14.py"), "w", encoding="utf-8", newline="\n").write(src)
print("wrote e30k_x14.py", len(src))
