"""Dump the TRUE birth matrices a_b = diag(w1) C_off of V29 at every layer and report their spectra:
energy fraction captured by rank k (16/32/64/128/256) and r99. Usage (from whest-starterkit): python -m uv run python ../experiments/probes/probe_birth_dump.py
"""
import os, sys, types
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from local_engine import build_mlp
import flopscope as flops
SRC = "../submissions/v29m8f5/estimator.py"
src = open(SRC, encoding="utf-8").read()
anchor = "            a_b = fnp.multiply(w1col, C_off, out=legs[\"AP0\"][k_b, 0])\n"
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + "            _BIRTH(li, a_b, C_off, w1col)\n")
OUT = {}
def _BIRTH(li, a_b, C_off, w1col):
    a = np.asarray(a_b, dtype=np.float64); c = np.asarray(C_off, dtype=np.float64)
    sv = np.linalg.svd(a, compute_uv=False); e = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    svc = np.linalg.svd(c, compute_uv=False); ec = np.cumsum(svc ** 2) / np.sum(svc ** 2)
    OUT[f"a{li}"] = a.astype(np.float32)
    print(f"birth at layer {li}: |a_b| {np.linalg.norm(a):.3e} |diag| {np.linalg.norm(np.diag(a)):.2e}  energy captured by rank 16/32/64/128/256: "
          f"{e[15]:.3f}/{e[31]:.3f}/{e[63]:.3f}/{e[127]:.3f}/{e[255]:.3f}  r99 {int(np.searchsorted(e, 0.99))+1}   (C_off alone: rank32 {ec[31]:.3f}, r99 {int(np.searchsorted(ec, 0.99))+1})", flush=True)
mod = types.ModuleType("est_birth"); mod.__dict__["_BIRTH"] = _BIRTH
exec(compile(src, SRC, "exec"), mod.__dict__)
class _Ctx: seed = 0; width = 1024; depth = 16; flop_budget = 2**41; submission_dir = None; scratch_dir = None; api_version = "x"
est = mod.Estimator(); est.setup(_Ctx())
mlp = build_mlp(width=1024, depth=16, seed=7)
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    est.predict(mlp, 2**41)
np.savez_compressed("../data/dumps/birth_dump.npz", **OUT); print("saved birth_dump.npz")
