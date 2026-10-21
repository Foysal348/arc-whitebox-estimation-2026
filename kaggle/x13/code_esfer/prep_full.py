"""X13 (20 Oct): prepare public FULL-split MLPs (aicrowd/arc-whestbench-public-2026 @ v2-phase2, config 'full') for the
x13 driver, without downloading the whole 70 GB split.
Usage: python prep_full.py <shards e.g. 0-4 or 40,41> <out_dir>
 - reads the mini split's mlp_seed / mlp_name / final layer truth column-only (7 small remote reads) and EXCLUDES every
   full row whose seed or name equals a mini row's, or whose last-layer truth equals a mini net's (mini 70-99 are our
   held-out nets; on 20 Oct the seeds are disjoint and one name, 'ashley-williams', repeats with a different seed)
 - per shard: hf_hub_download data/full-000kk-of-00067.parquet (16 MLPs, 1.1 GB), write per MLP
     out_dir/f<mlp_id:04d>_w.npy  (16,1024,1024) float32 weights
     out_dir/f<mlp_id:04d>_t.npy  (16,1024) float32 all_layer_means (the truth)
     out_dir/f<mlp_id:04d>.json   {mlp_id, name, mlp_seed, shard}
   then delete the shard. Net id used downstream = 1000 + mlp_id.
 - out_dir/seed_ctx.json : seed protocol version + salt (hex) from metadata.json (estimator never reads mlp.seed; kept
   for MLP.from_row fidelity).
"""
import os, sys, json, time
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "aicrowd/arc-whestbench-public-2026"; REV = "v2-phase2"
BASE = f"datasets/{REPO}@{REV}"
spec, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
shards = []
for part in spec.split(","):
    a, _, b = part.partition("-")
    shards += list(range(int(a), int(b or a) + 1))
T0 = time.time()
fs = HfFileSystem()


def flat_np(col):
    st = col.storage if hasattr(col, "storage") else col
    while pa.types.is_list(st.type) or pa.types.is_large_list(st.type) or pa.types.is_fixed_size_list(st.type):
        st = st.flatten()
    return np.asarray(st.to_numpy(zero_copy_only=False), dtype=np.float32)


# ---- seed context
ctx = {"version": None, "salt_hex": None}
try:
    md = json.loads(fs.cat(BASE + "/metadata.json").decode())
    sp = md.get("seed_protocol", {}) or {}
    ctx["version"] = sp.get("version", "2.0")
    if ctx["version"] == "4.0":
        from whestbench.seeds import resolve_salt
        s = resolve_salt(sp)
        ctx["salt_hex"] = s.hex() if isinstance(s, (bytes, bytearray)) else None
except Exception as e:
    print("seed ctx failed:", repr(e)[:300], flush=True)
json.dump(ctx, open(os.path.join(OUT, "seed_ctx.json"), "w"))
print("seed ctx", ctx["version"], "salt", "yes" if ctx["salt_hex"] else "no", flush=True)

# ---- mini identities
mini_files = sorted(p for p in fs.ls(BASE + "/data", detail=False) if p.split("/")[-1].startswith("mini-"))
m_seed, m_name, m_last = set(), set(), []
for p in mini_files:
    with fs.open(p, "rb", block_size=2**20) as f:
        t = pq.ParquetFile(f).read(columns=["mlp_seed", "mlp_name", "final_means"])
    m_seed |= set(int(x) for x in t.column("mlp_seed").to_pylist())
    m_name |= set(str(x) for x in t.column("mlp_name").to_pylist())
    fm = flat_np(t.column("final_means").combine_chunks()).reshape(t.num_rows, 1024)
    m_last += [r for r in fm]
m_last = np.stack(m_last)
print("mini rows", len(m_seed), "names", len(m_name), "final_means", m_last.shape, flush=True)
assert len(m_seed) == 100

# ---- full shards
excl = []
kept = []
for k in shards:
    t0 = time.time()
    fn = f"data/full-{k:05d}-of-00067.parquet"
    path = hf_hub_download(REPO, fn, repo_type="dataset", revision=REV, local_dir="/tmp/x13_hf")
    t = pq.ParquetFile(path).read(columns=["mlp_id", "mlp_name", "mlp_seed", "weights", "all_layer_means"])
    n = t.num_rows
    W = flat_np(t.column("weights").combine_chunks()).reshape(n, 16, 1024, 1024)
    TR = flat_np(t.column("all_layer_means").combine_chunks()).reshape(n, 16, 1024)
    ids = [int(x) for x in t.column("mlp_id").to_pylist()]
    names = [str(x) for x in t.column("mlp_name").to_pylist()]
    seeds = [int(x) for x in t.column("mlp_seed").to_pylist()]
    del t
    for r in range(n):
        why = []
        if seeds[r] in m_seed:
            why.append("seed")
        if names[r] in m_name:
            why.append("name")
        if np.any(np.all(np.abs(m_last - TR[r, 15][None]) < 1e-9, axis=1)):
            why.append("truth")
        if why:
            excl.append({"mlp_id": ids[r], "name": names[r], "why": why}); print("EXCLUDE", ids[r], names[r], why, flush=True)
            continue
        b = os.path.join(OUT, "f%04d" % ids[r])
        np.save(b + "_w.npy", np.ascontiguousarray(W[r]))
        np.save(b + "_t.npy", TR[r])
        json.dump({"mlp_id": ids[r], "name": names[r], "mlp_seed": seeds[r], "shard": k}, open(b + ".json", "w"))
        kept.append(ids[r])
    del W, TR
    os.remove(path)
    print("shard %d: %d rows (ids %d-%d) %.0fs, total kept %d" % (k, n, ids[0], ids[-1], time.time() - t0, len(kept)), flush=True)
json.dump({"shards": shards, "kept": kept, "excluded": excl}, open(os.path.join(OUT, "prep_report.json"), "w"), indent=1)
print("prep done: kept", len(kept), "excluded", excl, "%.0fs" % (time.time() - T0), flush=True)
