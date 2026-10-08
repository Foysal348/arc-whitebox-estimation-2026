#!/usr/bin/env python3
"""
Round 18 falsifier: Hermite Tree Quartic Bank (HTQB-6/3)

Primary test:
  - exact Gaussian-tree carrier: Q2 path + Cu1 star, both carried for 6 ages
  - shippable truncation: Cu1 star for 6 ages, Q2 path for 3 ages
  - project metric at L12/L15/L16:
      dec = abs(mu/sqrt(v)) < 1.5
      corr = pearson((pred/v**2)[dec], (K4d/v**2)[dec])

The carrier uses the leading connected Hermite diagrams of the degree-3 ReLU response:
  Star  (3,1,1,1): 24 sum_a c_a P_ai X_ai^3
  Path  (2,2,1,1): 48 v_i^T C v_i,
        X = C @ (Phi[:,None] * P),
        v = (g[:,None] * P) * X.
Here P is the linearized propagator from post-ReLU layer b to preactivation target t.

The script also performs a fresh true-network Monte Carlo (default N=2e6, float32 forward,
float64 accumulation) at L12/L15/L16 and reports carrier correlations against both the packed
teacher and the fresh MC estimate. The pre-registered gate uses the packed teacher because it
has much lower noise; the MC result is an independent sanity check.

Usage (project root):
  python round18_htqb_falsifier.py \
      --data kaggle/r17/data/r17_data.npz \
      --N 2000000 --batch 4096 --backend auto \
      --out results/summaries/round18_htqb.json

Fast algebra check only:
  python round18_htqb_falsifier.py --self-test

Gate:
  PASS if exact h6 tree carrier corr >= 0.70 at L12/L15/L16.
  KILL if exact h6 tree carrier corr < 0.55 at all three.
  The shippable h6-star/h3-path carrier is reported separately.
"""

import argparse, json, math, os, time
import numpy as np

TARGETS = (12, 15, 16)


def _pick(d, *names):
    for k in names:
        if k in d:
            return d[k]
    raise KeyError(f"none of keys {names} found; available={list(d.keys())}")


def phi_pdf(alpha):
    # vectorized standard normal CDF/PDF without scipy dependency
    erfv = np.vectorize(math.erf, otypes=[np.float64])
    Phi = 0.5 * (1.0 + erfv(alpha / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    return Phi, pdf


def pearson(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    x = x - x.mean(); y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 0 else float('nan')


def project_metric(pred, truth, mu, var):
    dec = np.abs(mu / np.sqrt(np.maximum(var, 1e-30))) < 1.5
    x = (pred / np.maximum(var * var, 1e-60))[dec]
    y = (truth / np.maximum(var * var, 1e-60))[dec]
    return pearson(x, y), int(dec.sum())


def slopes_from_teacher(mean, C):
    L = mean.shape[0] - 1
    Phi = np.zeros_like(mean, dtype=np.float64)
    g = np.zeros_like(mean, dtype=np.float64)
    c3 = np.zeros_like(mean, dtype=np.float64)
    for l in range(1, L + 1):
        var = np.diag(C[l]).astype(np.float64)
        sig = np.sqrt(np.maximum(var, 1e-30))
        alpha = mean[l].astype(np.float64) / sig
        ph, pdf = phi_pdf(alpha)
        Phi[l] = ph
        g[l] = pdf / (2.0 * sig)
        c3[l] = -alpha * pdf / (6.0 * sig * sig)
    return Phi, g, c3


def propagator(W, Phi, b, t):
    """Row convention. Birth at post-ReLU layer b -> preactivation target t, b < t.
       W[l] maps h_l -> z_{l+1}; hence first map is W[b].
    """
    P = W[b].astype(np.float64, copy=True)
    for r in range(b + 1, t):
        U = Phi[r][:, None] * W[r].astype(np.float64, copy=False)
        P = P @ U
    return P


def source_tree_readout(Cb, phib, gb, cb, P, want_path=True):
    """Exact diagonal contraction of the two retained connected Gaussian tree diagrams."""
    X = Cb @ (phib[:, None] * P)
    star = 24.0 * np.sum((cb[:, None] * P) * (X * X * X), axis=0)
    if not want_path:
        return star, np.zeros_like(star)
    V = (gb[:, None] * P) * X
    path = 48.0 * np.sum(V * (Cb @ V), axis=0)
    return star, path


def carrier_predictions(W, mean, C):
    Phi, g, c3 = slopes_from_teacher(mean, C)
    out = {}
    for t in TARGETS:
        n = W.shape[1]
        star6 = np.zeros(n, np.float64)
        path6 = np.zeros(n, np.float64)
        path3 = np.zeros(n, np.float64)
        per_birth = []
        for age in range(1, 7):
            b = t - age
            if b < 1:
                continue
            P = propagator(W, Phi, b, t)
            Cb = C[b].astype(np.float64, copy=False)
            s, q = source_tree_readout(Cb, Phi[b], g[b], c3[b], P, want_path=True)
            star6 += s
            path6 += q
            if age <= 3:
                path3 += q
            per_birth.append({"birth": b, "age": age,
                              "star_rms": float(np.sqrt(np.mean(s*s))),
                              "path_rms": float(np.sqrt(np.mean(q*q)))})
        out[t] = {
            "ceiling_h6": star6 + path6,
            "ship_h6star_h3path": star6 + path3,
            "star_h6": star6,
            "path_h6": path6,
            "per_birth": per_birth,
        }
    return out


def _xp_backend(name):
    if name == "numpy":
        return np, "numpy"
    if name in ("auto", "cupy"):
        try:
            import cupy as cp
            # force context creation
            _ = cp.zeros(1)
            return cp, "cupy"
        except Exception:
            if name == "cupy":
                raise
    return np, "numpy"


def fresh_mc(W, mean, N, batch, backend, seed=20261008):
    """True MLP MC; float32 forward, float64 accumulation around packed-teacher means."""
    xp, backend_used = _xp_backend(backend)
    Wdev = xp.asarray(W, dtype=xp.float32)
    n = W.shape[1]
    sums2 = {t: np.zeros(n, np.float64) for t in TARGETS}
    sums4 = {t: np.zeros(n, np.float64) for t in TARGETS}
    count = 0
    t0 = time.time()
    if backend_used == "cupy":
        rs = xp.random.RandomState(seed)
    else:
        rs = np.random.default_rng(seed)
    while count < N:
        m = min(batch, N - count)
        if backend_used == "cupy":
            h = rs.standard_normal((m, n), dtype=xp.float32)
        else:
            h = rs.standard_normal((m, n), dtype=np.float32)
        for l in range(1, 17):
            z = h @ Wdev[l-1]
            if l in TARGETS:
                mu = xp.asarray(mean[l], dtype=xp.float32)
                u = z - mu[None, :]
                s2 = xp.sum(u*u, axis=0, dtype=xp.float64)
                s4 = xp.sum((u*u)*(u*u), axis=0, dtype=xp.float64)
                if backend_used == "cupy":
                    s2 = xp.asnumpy(s2); s4 = xp.asnumpy(s4)
                sums2[l] += np.asarray(s2, np.float64)
                sums4[l] += np.asarray(s4, np.float64)
            h = xp.maximum(z, xp.float32(0.0))
        count += m
        if count % max(batch, (N//10//batch)*batch) == 0 or count == N:
            print(f"MC {count}/{N} ({100*count/N:.1f}%) elapsed={time.time()-t0:.1f}s", flush=True)
    k4 = {}
    var = {}
    for t in TARGETS:
        v = sums2[t] / N
        k = sums4[t] / N - 3.0 * v * v
        var[t] = v; k4[t] = k
    return k4, var, backend_used, time.time()-t0


def self_test():
    import itertools
    rng = np.random.default_rng(7)
    n, m = 4, 3
    A = rng.normal(size=(n,n)); C = A @ A.T
    ph = rng.uniform(0.2,0.9,size=n)
    g = rng.normal(size=n); c = rng.normal(size=n)
    P = rng.normal(size=(n,m))
    Bstar = np.zeros((n,n,n,n), np.float64)
    Bpath = np.zeros_like(Bstar)
    for a,b,d,e in itertools.product(range(n), repeat=4):
        inds = [a,b,d,e]
        ss = 0.0
        for hp in range(4):
            h = inds[hp]
            oth = [inds[j] for j in range(4) if j != hp]
            ss += 6.0*c[h]*ph[oth[0]]*ph[oth[1]]*ph[oth[2]]*C[h,oth[0]]*C[h,oth[1]]*C[h,oth[2]]
        Bstar[a,b,d,e] = ss
        sp = 0.0
        for i,j in itertools.combinations(range(4),2):
            k,l = [x for x in range(4) if x not in (i,j)]
            ii,jj,kk,ll = inds[i],inds[j],inds[k],inds[l]
            sp += 4.0*g[ii]*g[jj]*ph[kk]*ph[ll]*C[ii,jj]*(C[ii,kk]*C[jj,ll] + C[ii,ll]*C[jj,kk])
        Bpath[a,b,d,e] = sp
    exs=[]; exp=[]
    for o in range(m):
        q=P[:,o]
        exs.append(np.einsum('abcd,a,b,c,d->',Bstar,q,q,q,q))
        exp.append(np.einsum('abcd,a,b,c,d->',Bpath,q,q,q,q))
    X=C@(ph[:,None]*P)
    fs=24*np.sum((c[:,None]*P)*(X**3),axis=0)
    V=(g[:,None]*P)*X
    fp=48*np.sum(V*(C@V),axis=0)
    rs=float(np.linalg.norm(np.asarray(exs)-fs)/max(np.linalg.norm(exs),1e-30))
    rp=float(np.linalg.norm(np.asarray(exp)-fp)/max(np.linalg.norm(exp),1e-30))
    print(json.dumps({"self_test_star_relerr":rs,"self_test_path_relerr":rp}, indent=2))
    assert rs < 1e-12 and rp < 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='kaggle/r17/data/r17_data.npz')
    ap.add_argument('--N', type=int, default=2_000_000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--backend', choices=['auto','cupy','numpy'], default='auto')
    ap.add_argument('--seed', type=int, default=20261008)
    ap.add_argument('--skip-mc', action='store_true')
    ap.add_argument('--out', default='results/summaries/round18_htqb.json')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    d = np.load(args.data)
    W = _pick(d, 'W', 'weights').astype(np.float32)
    mean = _pick(d, 'mean', 'mu').astype(np.float64)
    C = _pick(d, 'C', 'cov').astype(np.float64)
    K4d = _pick(d, 'K4d', 'k4d', 'K4').astype(np.float64)
    assert W.shape == (16,1024,1024), W.shape
    assert mean.shape[0] >= 17 and C.shape[0] >= 17 and K4d.shape[0] >= 17

    print('computing analytic HTQB carrier...', flush=True)
    pred = carrier_predictions(W, mean, C)
    result = {
        'design': 'HTQB-6/3',
        'data': args.data,
        'gate': 'PASS if ceiling_h6 teacher corr >=0.70 at L12,L15,L16; KILL if <0.55 at all three',
        'targets': {},
        'mc': None,
    }
    for t in TARGETS:
        var = np.diag(C[t]).astype(np.float64)
        rec = {'n_decisive': None, 'teacher': {}, 'per_birth': pred[t]['per_birth']}
        for name in ('ceiling_h6','ship_h6star_h3path','star_h6','path_h6'):
            cc, nd = project_metric(pred[t][name], K4d[t], mean[t], var)
            rec['teacher'][name] = cc; rec['n_decisive'] = nd
        result['targets'][str(t)] = rec
        print(f"L{t} teacher: ceiling_h6={rec['teacher']['ceiling_h6']:.4f}  ship={rec['teacher']['ship_h6star_h3path']:.4f}  star={rec['teacher']['star_h6']:.4f}  path={rec['teacher']['path_h6']:.4f}  ndec={nd}")

    if not args.skip_mc:
        print(f"running fresh true-MLP MC N={args.N} batch={args.batch} backend={args.backend}...", flush=True)
        k4mc, vmc, backend_used, secs = fresh_mc(W, mean, args.N, args.batch, args.backend, args.seed)
        result['mc'] = {'N': args.N, 'batch': args.batch, 'backend': backend_used, 'seconds': secs, 'targets': {}}
        for t in TARGETS:
            rr={}
            # project metric still uses packed teacher mu/variance mask and denominator, exactly as project definition
            vteach=np.diag(C[t]).astype(np.float64)
            mc_truth=k4mc[t]
            for name in ('ceiling_h6','ship_h6star_h3path'):
                cc, nd = project_metric(pred[t][name], mc_truth, mean[t], vteach)
                rr[name]=cc
            teacher_vs_mc,_=project_metric(K4d[t],mc_truth,mean[t],vteach)
            rr['packed_teacher_vs_fresh_mc']=teacher_vs_mc
            result['mc']['targets'][str(t)]=rr
            print(f"L{t} freshMC: ceiling_h6={rr['ceiling_h6']:.4f}  ship={rr['ship_h6star_h3path']:.4f}  packedTeacher-vs-MC={teacher_vs_mc:.4f}")

    ceiling = [result['targets'][str(t)]['teacher']['ceiling_h6'] for t in TARGETS]
    if all(x >= 0.70 for x in ceiling): verdict='PASS'
    elif all(x < 0.55 for x in ceiling): verdict='KILL'
    else: verdict='BORDERLINE'
    result['verdict']=verdict
    print('VERDICT', verdict, ceiling)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print('wrote',args.out)


if __name__ == '__main__':
    main()
