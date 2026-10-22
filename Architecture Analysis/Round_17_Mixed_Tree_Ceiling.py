#!/usr/bin/env python3
"""Round 17 -- full mixed-tree quartic carrier: INPUT-ORACLE ceiling.

Self-contained: NumPy required, PyTorch optional for CUDA. No project Python imports.
Use: python Round_17_Mixed_Tree_Ceiling.py --root PROJECT --device auto
     python Round_17_Mixed_Tree_Ceiling.py --self-test
Kaggle: add --out /kaggle/working/r17.json.

C and Gaussian Wick coefficients come from pre-teachers; full K3 is estimated
from new samples through the TRUE MLP. No teacher K4 enters any birth.
This tests a specified tree closure with oracle lower moments. A PASS is NOT
validation of chain-supplied moments, compression, final MSE or deployment cost.

Default N=2,000,000; targets L12/L15/L16; float32 forwards/projections;
float64 moment products AND accumulation; independent stored K4 teacher as target.
Gate: PASS if all correlations >= .70; KILL if all < .55; else INCONCLUSIVE.
Partial runs never get a gate. Full history is primary; six-age readout is secondary.
"""
from pathlib import Path
import argparse,contextlib,itertools,json,math,time,sys
import numpy as np
TARGETS=(12,15,16)

def phi(x): return np.exp(-.5*x*x)/np.sqrt(2*np.pi)
def Phi(x): return np.fromiter((.5*(1+math.erf(float(v)/np.sqrt(2))) for v in x.flat),float,count=x.size).reshape(x.shape)
def derivatives(mu,var):
    s=np.sqrt(var);a=mu/s;p=phi(a)
    return Phi(a),p/s,-a*p/(s*s)
def tree_direction(C,K3,d1,d2,d3,V):
    X=d1[:,None]*V;A=C@X;T=d2[:,None]*V*A
    return (4*np.sum(d3[:,None]*V*A**3,0)+12*np.sum(T*(C@T),0)
            +12*np.einsum('ijk,ia,ja,ka->a',K3,T,X,X,optimize=True))
def mixed_k3(sums,N):
    ma,mc,a2,ac,ca2=sums/N
    return (ca2-mc*a2-2*ma*ac+2*mc*ma*ma)*(N*N/((N-1)*(N-2)))
def k4(sums,N):
    m1,m2,m3,m4=sums/N;c2=m2-m1*m1
    c4=m4-4*m1*m3+6*m1*m1*m2-3*m1**4
    return N*N*((N+1)*c4-3*(N-1)*c2*c2)/((N-1)*(N-2)*(N-3))
def corr(a,b,mask):
    a=np.asarray(a)[mask];b=np.asarray(b)[mask]
    if a.size<2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])
def gate(rows):
    v=[r['corr'] for r in rows.values()]
    if any(x is None or not np.isfinite(x) for x in v):return 'INCONCLUSIVE'
    if all(x>=.70 for x in v):return 'PASS'
    if all(x<.55 for x in v):return 'KILL'
    return 'INCONCLUSIVE'

def source_cofactors(A,P,Z,Zf,w2,s,e,c1,c2,L,R1,R2):
    """Exact contraction cofactors for the project's three cubic CP source blocks.
    Z: [n,rres+2], Zf: [n,2*rfb], L: [n,rres+2], R1/R2: [n,rfb].
    Returns FA,FP,FZ,FZf: evaluate mixed birth by 4*sum((J@B_birth)*F).
    A/P/Z/Zf supplied here are their current transported query evaluations.
    """
    rf=R1.shape[1];X=3*A+Zf[:,:rf]@R1.T
    Y=A*w2[None,:]+Zf[:,rf:]@R2.T
    M=P*s[None,:]+3*A*e[None,:]+Z@L.T
    X3=A*c1[None,:]+P*c2[None,:];yy=Z[:,-1,None];pp=P*P
    FA=3*P*Y+X*P*w2[None,:]+3*pp*e[None,:]+P*yy*c1[None,:]
    FP=X*Y+pp*s[None,:]+2*M*P+P*yy*c2[None,:]+X3*yy
    FZ=pp@L;np.add(FZ[:,-1],np.sum(X3*P,1),out=FZ[:,-1])
    FZf=np.concatenate(((P*Y)@R1,(X*P)@R2),1)
    return FA,FP,FZ,FZf

def self_test():
    r=np.random.default_rng(1701);n=5;R=7
    Q=r.normal(size=(n,n));C=Q@Q.T+.4*np.eye(n);mu=r.normal(size=n)
    d1,d2,d3=derivatives(mu,np.diag(C));fac=[r.normal(size=(n,R)) for _ in range(3)]
    raw=np.einsum('ir,jr,kr->ijk',*fac)
    K=sum(raw.transpose(p) for p in itertools.permutations(range(3)))/6
    I=np.eye(n);A=d1[:,None]*C
    raw4=(4*np.einsum('a,ia,ja,ka,la->ijkl',d3,I,A,A,A)
         +12*np.einsum('a,b,ab,ia,ja,kb,lb->ijkl',d2,d2,C,I,A,I,A)
         +12*np.einsum('a,ia,ja,akl,k,l->ijkl',d2,I,A,K,d1,d1))
    K4=sum(raw4.transpose(p) for p in itertools.permutations(range(4)))/24
    V=r.normal(size=(n,n));dense=np.einsum('ijkl,ia,ja,ka,la->a',K4,V,V,V,V,optimize=True)
    pred=tree_direction(C,K,d1,d2,d3,V);err=float(np.linalg.norm(dense-pred)/np.linalg.norm(dense))
    rz,rf=4,2;A0,P0=[r.normal(size=(n,n)) for _ in range(2)]
    Z0=r.normal(size=(n,rz));Zf0=r.normal(size=(n,2*rf))
    w2,ss,ee,c1,c2=[r.normal(size=n) for _ in range(5)]
    L=r.normal(size=(n,rz));R1,R2=[r.normal(size=(n,rf)) for _ in range(2)]
    T=r.normal(size=(n,n));J=r.normal(size=(n,n))
    Fs=source_cofactors(T@A0,T@P0,T@Z0,T@Zf0,w2,ss,ee,c1,c2,L,R1,R2)
    opt=4*sum(np.sum((J@B)*F,1) for B,F in zip([A0,P0,Z0,Zf0],Fs))
    X0=3*A0+Zf0[:,:rf]@R1.T;Y0=A0*w2+Zf0[:,rf:]@R2.T
    M0=P0*ss+3*A0*ee+Z0@L.T;X30=A0*c1+P0*c2
    Y30=np.repeat(Z0[:,-1,None],n,axis=1);direct=np.zeros(n)
    for X,Y,Z in [(X0,P0,Y0),(M0,P0,P0),(X30,P0,Y30)]:
        XL,YL,ZZ=T@X,T@Y,T@Z
        direct+=4*np.sum((J@X)*YL*ZZ+(J@Y)*XL*ZZ+(J@Z)*XL*YL,1)
    er2=float(np.linalg.norm(opt-direct)/np.linalg.norm(direct))
    z=r.normal(size=(101,n));z+=.2*(z*z-1);zz=z-z.mean(0)
    U,V=r.normal(size=(n,3)),r.normal(size=(n,3));aa=z@U;cc=z@V
    acc=np.stack([aa.sum(0),cc.sum(0),(aa*aa).sum(0),(aa*cc).sum(0),(cc*aa*aa).sum(0)])
    val=mixed_k3(acc,len(z));ref=np.mean((zz@V)*(zz@U)**2,0)*len(z)**2/((len(z)-1)*(len(z)-2))
    er3=float(np.max(np.abs(val-ref)))
    assert err<1e-12 and er2<1e-12 and er3<1e-12,(err,er2,er3)
    assert gate({'a':{'corr':.70},'b':{'corr':.8},'c':{'corr':.71}})=='PASS'
    assert gate({'a':{'corr':.54},'b':{'corr':.1},'c':{'corr':.0}})=='KILL'
    assert gate({'a':{'corr':.55},'b':{'corr':.1},'c':{'corr':.0}})=='INCONCLUSIVE'
    print(json.dumps({'self_test':'PASS','scope':'synthetic algebra checks, NOT MLP #0 accuracy',
      'dense_quartic_relative_error':err,'optimized_cofactor_relative_error':er2,
      'mixed_cumulant_absolute_error':er3},indent=2))

class Backend:
    def __init__(self,device,seed,threads):
        self.torch=None;self.device='numpy'
        if device!='numpy':
            try:import torch
            except ImportError:
                if device in ['cuda','cpu']:raise RuntimeError('PyTorch required for --device cuda/cpu; use numpy otherwise.')
            else:
                if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
                if device in ['cuda','cpu'] or (device=='auto' and torch.cuda.is_available()):
                    self.torch=torch;self.device='cuda' if device=='cuda' or (device=='auto' and torch.cuda.is_available()) else 'cpu'
                    if threads:torch.set_num_threads(threads)
                    if self.device=='cuda':torch.backends.cuda.matmul.allow_tf32=False
                    torch.set_float32_matmul_precision('highest')
                    self.rng=torch.Generator(device=self.device);self.rng.manual_seed(seed)
        if self.torch is None:self.rng=np.random.default_rng(seed)
    def arr(self,x):
        x=np.asarray(x,np.float32)
        return self.torch.as_tensor(x,device=self.device) if self.torch else x
    def zeros64(self,shape):return self.torch.zeros(shape,dtype=self.torch.float64,device=self.device) if self.torch else np.zeros(shape,np.float64)
    def rand(self,shape):return self.torch.randn(shape,dtype=self.torch.float32,device=self.device,generator=self.rng) if self.torch else self.rng.standard_normal(shape,dtype=np.float32)
    def relu(self,z):return self.torch.clamp_min(z,0) if self.torch else np.maximum(z,0)
    def host(self,x):return x.detach().cpu().numpy() if self.torch else np.asarray(x)
    def mixed_stats(self,a,c):
        a=a.to(self.torch.float64) if self.torch else a.astype(np.float64)
        c=c.to(self.torch.float64) if self.torch else c.astype(np.float64);a2=a*a
        v=[a.sum(0),c.sum(0),a2.sum(0),(a*c).sum(0),(a2*c).sum(0)]
        return self.torch.stack(v) if self.torch else np.stack(v)
    def fourth_stats(self,a):
        a=a.to(self.torch.float64) if self.torch else a.astype(np.float64);a2=a*a
        v=[a.sum(0),a2.sum(0),(a2*a).sum(0),(a2*a2).sum(0)]
        return self.torch.stack(v) if self.torch else np.stack(v)
    def sync(self):
        if self.torch and self.device=='cuda':self.torch.cuda.synchronize()

def find_root(value):
    if value:return Path(value).resolve()
    for p in [Path.cwd(),*Path(__file__).resolve().parents]:
        if (p/'data/kaggle_out/kaggle_out_k4pub/k4teach').is_dir():return p
    raise FileNotFoundError('Use --root PATH containing data/kaggle_out/kaggle_out_k4pub/k4teach.')

def main(args):
    if args.samples<16 or args.batch<1:raise ValueError('Need samples >= 16 and batch >= 1.')
    root=find_root(args.root);pre=root/'data/kaggle_out/kaggle_out_k4pub/k4teach'
    W=np.load(pre/'weights_seed0.npz')['W'].astype(np.float32)
    if W.ndim!=3 or W.shape[0]<16 or W.shape[1]!=W.shape[2]:raise ValueError('Expected W[depth,n,n], depth >= 16.')
    n=W.shape[1];C={};mu={};var={};true4={}
    for l in range(1,17):
        with np.load(pre/f'layer{l}_seed0.npz') as f:
            mu[l]=np.asarray(f['mean'],np.float64);C[l]=np.asarray(f['C'],np.float64)
            if l in TARGETS:true4[l]=np.asarray(f['K4d'],np.float64)  # SCORE ONLY
        C[l]=(C[l]+C[l].T)/2;var[l]=np.diag(C[l]).copy()
        if np.any(var[l]<=0):raise ValueError(f'Nonpositive variance at layer {l}.')
    C[1]=W[0].astype(np.float64).T@W[0].astype(np.float64);mu[1]=np.zeros(n);var[1]=np.diag(C[1]).copy()
    ds={l:derivatives(mu[l],var[l]) for l in range(1,17)}
    with np.load(root/'data/dumps/errbudget_v29plain.npz') as f:old={l:np.asarray(f[f'L{l}_ck4'],np.float64) for l in TARGETS}
    be=Backend(args.device,args.seed,args.threads);start=time.perf_counter()
    print(f'Preparing full-history tree births: device={be.device}, n={n}, N={args.samples:,}; teacher K4 is score-only.',flush=True)
    pairs=[];by_birth={l:[] for l in range(1,16)}
    for target in TARGETS:
        V=W[target-1].astype(np.float64)
        for b in range(target-1,0,-1):
            d1,d2,d3=ds[b];X=d1[:,None]*V;A=C[b]@X;T=d2[:,None]*V*A
            det=4*np.sum(d3[:,None]*V*A**3,0)+12*np.sum(T*(C[b]@T),0)
            p={'birth':b,'target':target,'det':det,'X':None,'T':None}
            if b>1:p.update(X=be.arr(X),T=be.arr(T));by_birth[b].append(len(pairs))
            pairs.append(p)
            if b>1:V=W[b-1].astype(np.float64)@(d1[:,None]*V)
    m=len(pairs);acc=be.zeros64((2,m,5,n));truth_acc=be.zeros64((2,3,4,n))
    weights=[be.arr(w) for w in W[:16]];mus={l:be.arr(mu[l]) for l in range(1,17)}
    del C,ds
    half=args.samples//2;counts=[0,0];done=0;batch_no=0;last_report=0.
    eval_start=time.perf_counter();context=be.torch.inference_mode() if be.torch else contextlib.nullcontext()
    output={'scope':'input-oracle tree-closure ceiling, NOT deployment validation','births':'Gaussian star + Gaussian path + full K3 x C; no true K4 births',
      'network':'public MLP #0','targets':list(TARGETS),'requested_N':args.samples,'seed':args.seed,'backend':be.device,'n':n,
      'pre_registered_gate':{'PASS':'all rho >= 0.70','KILL':'all rho < 0.55','otherwise':'INCONCLUSIVE'},
      'labels':{'correlations':'measured by this execution','runtime':'measured','dominant_flops':'derived'},
      'dominant_sampling_flops':int(90*args.samples*n*(2*n-1))}
    with context:
        while done<args.samples:
            if time.perf_counter()-start>=args.time_limit:break
            split=0 if done<half else 1
            bs=min(args.batch,args.samples-done,half-done if split==0 else args.samples-done);h=be.rand((bs,n))
            for l in range(1,17):
                z=h@weights[l-1]
                if l in TARGETS:truth_acc[split,TARGETS.index(l)]+=be.fourth_stats(z-mus[l])
                if by_birth.get(l):
                    dz=z-mus[l]
                    for j in by_birth[l]:
                        p=pairs[j];acc[split,j]+=be.mixed_stats(dz@p['X'],dz@p['T'])
                h=be.relu(z)
            done+=bs;counts[split]+=bs;batch_no+=1;now=time.perf_counter()
            if batch_no==1 or now-last_report>=30 or done==args.samples:
                be.sync();elapsed=time.perf_counter()-eval_start;eta=elapsed*(args.samples-done)/done
                print(f'N={done:,}/{args.samples:,}; sampling elapsed={elapsed:.1f}s; remaining ETA={eta:.1f}s [estimate]',flush=True)
                if batch_no==1 and eta+elapsed>args.time_limit:print('Estimated runtime exceeds time limit; CUDA recommended. A partial run cannot PASS/KILL.',flush=True)
                last_report=time.perf_counter()
    be.sync();output.update(completed_N=done,elapsed_seconds=time.perf_counter()-start,half_counts=counts)
    if done<args.samples:output.update(status='INCOMPLETE',gate='NOT_EVALUATED')
    else:
        sums=be.host(acc);truth_sums=be.host(truth_acc);stats={}
        for scope,sm,nt in [('all',sums.sum(0),done),('half1',sums[0],counts[0]),('half2',sums[1],counts[1])]:
            stats[scope]=[p['det']+(12*mixed_k3(sm[j],nt) if p['birth']>1 else 0) for j,p in enumerate(pairs)]
        output.update(status='COMPLETE',full_history={},six_age={})
        for ti,l in enumerate(TARGETS):
            mask=np.abs(mu[l]/np.sqrt(var[l]))<1.5;target=true4[l]/var[l]**2;tmc=k4(truth_sums[:,ti].sum(0),done)
            for horizon,name in [(0,'full_history'),(6,'six_age')]:
                ix=[j for j,p in enumerate(pairs) if p['target']==l and (not horizon or l-p['birth']<=horizon)]
                row={'births':[pairs[j]['birth'] for j in ix],'n_decisive':int(mask.sum()),'MC_truth_vs_independent_teacher':corr(tmc/var[l]**2,target,mask),'chain_vs_teacher':corr(old[l]/var[l]**2,target,mask)}
                for scope in ['all','half1','half2']:
                    q=sum(stats[scope][j] for j in ix);level=float(np.mean(q));pred=q*(float(np.mean(old[l]))/level) if level!=0 else q*np.nan
                    row['corr' if scope=='all' else scope+'_corr']=corr(pred/var[l]**2,target,mask)
                    if scope=='all':row['raw_predicted_mean']=level;row['chain_mean']=float(np.mean(old[l]))
                output[name][str(l)]=row
                print(f'{name} L{l}: rho={row["corr"]}; halves={row["half1_corr"]}, {row["half2_corr"]}; MC truth rho={row["MC_truth_vs_independent_teacher"]}',flush=True)
        output['gate']=gate(output['full_history']);output['six_age_gate']=gate(output['six_age'])
        print(f'UNTRUNCATED INPUT-ORACLE GATE: {output["gate"]}; SIX-AGE DIAGNOSTIC: {output["six_age_gate"]}',flush=True)
        print('PASS does not validate chain inputs, rank compression, final MSE or cost.',flush=True)
    out=Path(args.out).expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8');print('Saved',out,flush=True)
    return 0 if output['status']=='COMPLETE' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root');p.add_argument('--samples',type=int,default=2_000_000);p.add_argument('--batch',type=int,default=2048)
    p.add_argument('--seed',type=int,default=777);p.add_argument('--device',choices=['auto','numpy','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=0,help='Diagnostic CPU threads; 0 preserves defaults.')
    p.add_argument('--time-limit',type=float,default=7200,help='Seconds; incomplete runs get no gate.')
    p.add_argument('--out',default='round17_tree_ceiling_results.json');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:
        with contextlib.ExitStack() as stack:
            if a.threads:
                try:from threadpoolctl import threadpool_limits
                except ImportError:pass
                else:stack.enter_context(threadpool_limits(limits=a.threads))
            sys.exit(main(a))
