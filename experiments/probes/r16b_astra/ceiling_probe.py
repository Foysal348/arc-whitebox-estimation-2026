"""Round 16: exact dense six-age pair-innovation quartic carrier ceiling.
Project data are read-only. Teacher arrays are used ONLY for scoring.
Single predeclared model; no fitted parameters, rank cuts, or source screening.
"""
from pathlib import Path
import json,time,sys
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
ROOT=Path(r'D:\Machine Learning 1 Year Challenge\Kaggle\ARC White-Box Estimation Challenge 2026')
H=Path(__file__).resolve().parent
PRE=ROOT/'data/kaggle_out/kaggle_out_k4pub/k4teach'
t0=time.time()
horizon=15 if '--full' in sys.argv else 6
stem='full_history' if horizon==15 else 'ceiling'
W=np.load(PRE/'weights_seed0.npz')['W'].astype(np.float64)
BASE=np.load(ROOT/'data/dumps/errbudget_v29plain.npz')
cs={}; report={'model':'full pair-innovation core at every ReLU; exact dense transport; selected age window','network':'public MLP #0','precision':'float64 (stored K22/wk4m originate in float32)','horizon':horizon,'threshold':0.7,'layers':{},'checks':{}}
for l in range(1,17):
    with np.load(ROOT/f'data/dumps/r7_cap_L{l}.npz') as f:
        keys=['mu','var','W_all','K4','K22','g4row','wk4m']
        cs[l]={k:np.asarray(f[k],np.float64) for k in keys if k in f.files}
    cs[l]['d']=cs[l]['W_all'][:,4]
report['checks']['max_wick_phi_error']=max(float(np.max(np.abs(c['d']-ndtr(c['mu']/np.sqrt(c['var']))))) for c in cs.values())
# G is exactly the polynomial core: off-diagonal 3*delta_K22; diagonal delta_K4.
def birth(c):
    d2=c['d']**2
    delta22=c['K22']-d2[:,None]*c.get('wk4m',np.zeros((1024,1024)))*d2[None,:]
    G=1.5*(delta22+delta22.T)
    diag=c['K4']-d2*d2*c.get('g4row',np.zeros(1024))
    np.fill_diagonal(G,diag)
    return G
# Algebra check independent of network: explicitly form a small symmetric four-tensor.
rng=np.random.default_rng(1601); m=5
G=rng.normal(size=(m,m));G=(G+G.T)/2;P=rng.normal(size=(m,m))
K=np.zeros((m,)*4)
import itertools
for a in range(m):
    K[a,a,a,a]=G[a,a]
    for b in range(a+1,m):
        for idx in set(itertools.permutations([a,a,b,b])): K[idx]=G[a,b]/3
pred=np.sum(((P*P)@G)*(P*P),axis=1)
ref=np.einsum('ia,ib,ic,id,abcd->i',P,P,P,P,K,optimize=True)
report['checks']['small_tensor_relative_error']=float(np.linalg.norm(pred-ref)/np.linalg.norm(ref))
active=[]; snapshots={}
with threadpool_limits(limits=1):
    for l in range(2,17):
        active=[(b,W[l-1].T@(cs[l-1]['d'][:,None]*p),g) for b,p,g in active if l-b<=horizon]
        active.append((l-1,W[l-1].T.copy(),birth(cs[l-1])))
        if l in [3,7,11,14,15]:
            with np.load(ROOT/f'data/dumps/r7_cap_L{l}.npz') as cap:
                if 'P_st' in cap.files:
                    stored=cap['P_st']
                    report['checks'][f'P_relative_errors_L{l}']={str(b):float(np.linalg.norm(p-stored[b-1])/np.linalg.norm(p)) for b,p,g in active}
        if l<8: continue
        B=np.stack([np.sum(((p*p)@g)*(p*p),axis=1) for b,p,g in active])
        q=B.sum(axis=0)
        old=np.asarray(BASE[f'L{l}_ck4'],np.float64)
        raw=q.copy(); q=q*(old.mean()/q.mean())
        with np.load(PRE/f'layer{l}_seed0.npz') as t:
            vt=np.diag(t['C']).astype(np.float64); mu=t['mean'].astype(np.float64); truth=t['K4d'].astype(np.float64)
        alpha=mu/np.sqrt(vt);mask=np.abs(alpha)<1.5
        def corr(a,b): return float(np.corrcoef(a[mask],b[mask])[0,1])
        lever=np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)*(alpha*alpha-1)/(24*vt**1.5)
        row={'decisive_neurons':int(mask.sum()),'births':[b for b,_,_ in active],
             'corr_standardized':corr(q/vt**2,truth/vt**2),'corr_chain_standardized':corr(old/vt**2,truth/vt**2),
             'corr_raw':corr(q,truth),'corr_chain_raw':corr(old,truth),'normalization':float(old.mean()/raw.mean()),
             'raw_mean':float(raw.mean()),'chain_mean':float(old.mean()),'teacher_mean':float(truth.mean()),
             'mean_closure_weighted_error_ratio':float(np.linalg.norm(lever*(q-truth))/np.linalg.norm(lever*(old-truth)))}
        report['layers'][l]=row
        snapshots[f'L{l}_q']=q;snapshots[f'L{l}_source_readouts']=B
        print('L',l,json.dumps(row),flush=True)
report['pass_all_L12_16']=all(report['layers'][l]['corr_standardized']>=0.7 for l in range(12,17))
report['elapsed_seconds']=time.time()-t0
(H/f'{stem}_result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez(H/f'{stem}_readouts.npz',**snapshots)
print('DONE',report['pass_all_L12_16'],'seconds',report['elapsed_seconds'],flush=True)


