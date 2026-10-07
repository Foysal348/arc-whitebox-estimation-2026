"""Round 15: source-resolved skew energy. Read-only to project; outputs beside script.
No repository imports execute; only the audited factor reconstruction function is extracted.
"""
from pathlib import Path
import ast, json, time, sys
import numpy as np
from scipy.special import ndtr

ROOT=Path(r'D:\Machine Learning 1 Year Challenge\Kaggle\ARC White-Box Estimation Challenge 2026')
HERE=Path(__file__).resolve().parent
PRE=ROOT/'data/kaggle_out/kaggle_out_k4pub/k4teach'
D=np.load(ROOT/'data/dumps/errbudget_v29plain.npz')
ns={'np':np,'n':1024,'f32':lambda x:np.asarray(x,np.float32)}
tree=ast.parse((ROOT/'experiments/probes/r7_srcgain.py').read_text(encoding='utf-8'))
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='blocks_of_source')
exec(compile(ast.Module(body=[node],type_ignores=[]),'factor_reconstruction','exec'),ns)
blocks=ns['blocks_of_source']
all_data={}; report={'network':'public MLP #0','mode':'stored dense all-source chain captures; no teacher feedback; ungraded offline test','layers':{}}
USE_CACHE='--cached' in sys.argv
if USE_CACHE:
    report=json.loads((HERE/'probe_birth_energy.json').read_text(encoding='utf-8'))
t0=time.time()
for L in [7,11,14,15]:
    if USE_CACHE:
        with np.load(HERE/f'birth_features_L{L}.npz') as a:
            all_data[L]={key:a[key] for key in a.files if key!='B'}
        continue
    with np.load(ROOT/f'data/dumps/r7_cap_L{L}.npz') as archive:
        c={key:archive[key] for key in archive.files}
    k=c['P_st'].shape[0]
    ds=[]
    for s in range(k):
        d=np.zeros(1024)
        for X,P,Y in blocks(c,s,int(c['rres']),int(c['rfb'])):
            d+=np.sum(X.astype(np.float64)*P*Y,axis=1)
        ds.append(d)
    B=np.asarray(ds).T
    v=c['var'].astype(np.float64); sig=np.sqrt(v); mu=c['mu'].astype(np.float64)
    Y=c['D21'].astype(np.float64).copy(); np.fill_diagonal(Y,c['D3'])
    C=c['C_off'].astype(np.float64).copy(); np.fill_diagonal(C,v)
    Z=Y/v[:,None]/sig[None,:]; R=C/sig[:,None]/sig[None,:]
    old=D[f'L{L}_ck4'].astype(np.float64)
    X=np.stack([(c['D3']/v**1.5)**2,(Z*Z).sum(1),(Z*Z).sum(0),(R*R).sum(1),np.abs(mu/sig),old/v**2,np.ones(1024)],axis=1)
    q=np.sum(B*B,axis=1)/v**3
    T=np.load(PRE/f'layer{L}_seed0.npz'); vt=np.diag(T['C']); at=T['mean']/np.sqrt(vt); dec=np.abs(at)<1.5
    target=T['K4d'].astype(np.float64); t=target/vt**2
    g4=(at*at-1)*np.exp(-at*at/2)/np.sqrt(2*np.pi)/(24*vt**1.5)
    chainmask=np.abs(mu/sig)<1.5
    all_data[L]={'X':X,'q':q,'v':v,'vt':vt,'old':old,'target':target,'t':t,'g4':g4,'dec':dec,'chainmask':chainmask}
    corr=lambda a,b:float(np.corrcoef(a[dec],b[dec])[0,1])
    report['layers'][L]={'n_sources':k,'n_decisive':int(dec.sum()),'source_sum_rel_error':float(np.linalg.norm(B.sum(1)-c['D3'])/np.linalg.norm(c['D3'])),
       'corr_energy':corr(q,t),'corr_total_skew_squared':corr(X[:,0],t),'corr_chain':corr(old/vt**2,t)}
    print('source layer',L,json.dumps(report['layers'][L]),flush=True)
    np.savez(HERE/f'birth_features_L{L}.npz',B=B,**all_data[L])
    del c

def fit(ls,extra):
    Xs=[];ts=[]
    for L in ls:
        a=all_data[L]; X=a['X']; X=np.c_[X,a['q']] if extra else X
        Xs.append(X[a['dec']]); ts.append((a['target']/a['v']**2)[a['dec']])
    X=np.concatenate(Xs); y=np.concatenate(ts); scale=np.linalg.norm(X,axis=0)
    return np.linalg.lstsq(X/scale,y,rcond=1e-10)[0]/scale

def score(L,beta,extra):
    a=all_data[L]; X=a['X']; X=np.c_[X,a['q']] if extra else X
    q=X@beta
    pred=a['v']**2*q
    pred=pred*(a['old'].mean()/pred.mean())
    num=np.linalg.norm(a['g4']*(pred-a['target'])); den=np.linalg.norm(a['g4']*(a['old']-a['target']))
    return {'corr':float(np.corrcoef((pred/a['vt']**2)[a['dec']],a['t'][a['dec']])[0,1]),'weighted_norm_ratio_vs_original_chain':float(num/den)}

report['tests']={}
for label,ls in [('fit_early',[7,11]),('fit_all',[7,11,14,15])]:
    out={}
    for extra in [False,True]:
        beta=fit(ls,extra)
        out['base_plus_energy' if extra else 'base']={'beta':beta.tolist(),'scores':{L:score(L,beta,extra) for L in all_data}}
    report['tests'][label]=out
    print(label,json.dumps(out),flush=True)
report['leave_one_layer_out']={}
for L in all_data:
    ls=[m for m in all_data if m!=L]
    report['leave_one_layer_out'][L]={str(extra):score(L,fit(ls,extra),extra) for extra in [False,True]}
report['scoring_elapsed_s']=time.time()-t0
(HERE/'probe_birth_energy.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('saved',HERE/'probe_birth_energy.json','elapsed',report['scoring_elapsed_s'],flush=True)
