"""Round 3: bounded algebra/provenance checks, not a prediction benchmark.
Run: python round3_checks.py --root PROJECT --out results.json
No project modules are imported or executed; NPZ loads use allow_pickle=False.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, hashlib, itertools, json, time
from pathlib import Path
import numpy as np

def relative(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-300))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); start = time.perf_counter()
    rng = np.random.default_rng(301)
    result = {'scope': 'algebra checks and saved seed-7 L14 dump; no truth MSE rerun'}
    q = rng.normal(size=(23, 9)); x = rng.normal(size=(9, 17))
    target = (q @ x)**2
    diag = (q*q) @ (x*x)
    full = diag.copy()
    for i in range(9):
        for j in range(i+1, 9):
            full += 2*np.outer(q[:, i]*q[:, j], x[i]*x[j])
    result['pair_lift'] = {'exact_relative_error': relative(target, full),
                           'diagonal_only_relative_error': relative(target, diag)}
    p = rng.normal(size=(8, 8)); h = rng.normal(size=(8, 8))
    tensor = np.einsum('ir,jr,kr->ijk', p, p, h)
    sym = sum(tensor.transpose(perm) for perm in itertools.permutations(range(3)))/6
    direct = sym[np.arange(8), np.arange(8), :]
    formula = ((p*p) @ h.T + 2*(p*h) @ p.T)/3
    result['d21_formula_error'] = relative(direct, formula)
    # At birth, B3 vanishes on all-distinct triples; B1 need not.
    eye = np.eye(8); covleg = rng.normal(size=(8, 8)); np.fill_diagonal(covleg, 0)
    b1 = np.einsum('ir,jr,kr->ijk', 3*covleg, eye, covleg)
    b1 = sum(b1.transpose(t) for t in itertools.permutations(range(3)))/6
    b3 = np.einsum('ir,jr,kr->ijk', h, eye, eye)
    b3 = sum(b3.transpose(t) for t in itertools.permutations(range(3)))/6
    distinct = np.ones((8, 8, 8), dtype=bool)
    for i,j,k in np.ndindex(distinct.shape):
        distinct[i,j,k] = len({i,j,k}) == 3
    result['birth_support'] = {'B1_distinct_norm': float(np.linalg.norm(b1[distinct])),
                              'B3_distinct_norm': float(np.linalg.norm(b3[distinct]))}
    # Gaussian-chaos identity for one mixed-row and one same-row query.
    k = sym; wi = rng.normal(size=8); wk = rng.normal(size=8); sigma2 = 0.7
    trace = np.einsum('aac->c', k)
    exact = np.einsum('abc,a,b,c', k, wi, wi, wk)
    lin = sigma2*trace@wk
    third = np.einsum('abc,ab,c', k, np.outer(wi, wi)-sigma2*np.eye(8), wk)
    same = np.einsum('abc,a,b,c', k, wi, wi, wi)
    H3 = np.einsum('a,b,c->abc', wi,wi,wi)
    H3 -= sigma2*(np.einsum('ab,c->abc', np.eye(8),wi)+
                   np.einsum('ac,b->abc', np.eye(8),wi)+
                   np.einsum('bc,a->abc', np.eye(8),wi))
    result['chaos_identity'] = {'offdiag_abs_error': float(abs(exact-lin-third)),
                               'diagonal_abs_error': float(abs(same-3*sigma2*trace@wi-np.sum(k*H3)))}
    # Finite size Frobenius fluctuation, using real network propagator legs.
    dump = a.root/'query_dump_L14.npz'
    with np.load(dump, allow_pickle=False) as d:
        q, _ = np.linalg.qr(d['Qc'])
        stats = []
        for s in [0, int(d['ka'])-1]:
            p = d[f'P{s}'].astype(np.float64)
            fp = q.T @ p
            pp = p*p
            centered = pp-pp.mean(axis=0, keepdims=True)
            diag = (q*q) @ (fp*fp)
            stats.append({'source_slot': s, 'P_shape': list(p.shape),
                          'leg_reconstruction_error': relative(p, q@fp),
                          'centered_square_fraction': float(np.linalg.norm(centered)/np.linalg.norm(pp)),
                          'diagonal_pair_lift_error': relative(pp, diag)})
        result['saved_network'] = {'file': dump.name, 'n': int(q.shape[0]),
                                    'basis_rank': int(q.shape[1]), 'old_sources': int(d['ka']),
                                    'checks': stats}
    n=1024
    result['costs'] = []
    for r in (160,224,250,256,384,1024):
        s=r*(r+1)//2
        result['costs'].append({'r':r,'pair_features':s,
          'pair_feature_MiB_float32': 4*n*s/2**20,
          'direct_QX_u':r/n,
          'square_lift_Q2X2_u':s/n,
          'dense_core_MiB_float32':4*r**3/2**20,
          'prompt_optimistic_slice_u':r**3/(4*n*n),
          'prompt_optimistic_rotation_u':3*r**4/(2*n**3),
          'prompt_optimistic_slice_join_rotation_u':r**3/(2*n*n)+3*r**4/(2*n**3)})
    paths=['mlp_cumulant_propagation/src/mlp_kprop/factor_k3.py',
           'mlp_cumulant_propagation/src/mlp_kprop/kprop_harmonic.py',
           'submission_v29m8f5/estimator.py','probe_query.py','probe_agg.py']
    result['sha256']={p:hashlib.sha256((a.root/p).read_bytes()).hexdigest() for p in paths}
    result['elapsed_seconds']=time.perf_counter()-start
    a.out.write_text(json.dumps(result,indent=2), encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
