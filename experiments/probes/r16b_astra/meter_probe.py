"""Round 16 actual flopscope costs; no estimator changes or predict() calls.
Meter the proposed full pair-core construction + full six-source readout.
Warm/cold scratch pools distinguished. Diagnostics run single-threaded.
"""
from pathlib import Path
import ast,math,json,gc
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits
ROOT=Path(r'D:\Machine Learning 1 Year Challenge\Kaggle\ARC White-Box Estimation Challenge 2026')
H=Path(__file__).resolve().parent
src=ROOT/'submissions/v55k4c/estimator.py'
tree=ast.parse(src.read_text(encoding='utf-8'))
node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='_Strassen')
ns={'math':math,'fnp':fnp,'STRASSEN_MIN':8,'STRASSEN_FUSE_P':343}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
SMM=ns['_Strassen']; n=1024; k=6; rng=np.random.default_rng(1602)
def arr(shape): return fnp.asarray(rng.standard_normal(shape,dtype=np.float32)*np.float32(.03))
def rec(ctx): return {'flops':int(ctx.flops_used),'u':float(ctx.flops_used/2**31),'ops':len(ctx._op_log)}
P,G=[arr((k,1,n,n)) for _ in range(2)]
old=arr((n,)); d=arr((n,)); pre4=arr((n,)); post4=arr((n,)); pre22=arr((n,n)); post22=arr((n,n))
report={'meter_version':flops.__version__,'shape':[k,1,n,n],'births':15,'corrected_layers':list(range(8,17)),'cases':{}}
with flops.BudgetContext(flop_budget=10**14,quiet=True) as ctx:
    S=fnp.empty((k,1,n,n),dtype=np.float32)
    R=fnp.empty((k,1,n,n),dtype=np.float32)
    slots=fnp.empty((15,n,n),dtype=np.float32)
report['setup_buffers']=rec(ctx)
# Write one birth into its chronological slab slot. No symmetry tag / rank assumption.
# 3*delta22 has the same quartic polynomial as its symmetric part.
with flops.BudgetContext(flop_budget=10**14,quiet=True) as ctx:
    d2=d*d
    fnp.multiply(d2[:,None],pre22,out=slots[0])
    fnp.multiply(slots[0],d2[None,:],out=slots[0])
    fnp.subtract(post22,slots[0],out=slots[0])
    fnp.multiply(slots[0],np.float32(3),out=slots[0])
    diagonal=post4-(d2*d2)*pre4
    fnp.fill_diagonal(slots[0],diagonal)
report['birth']=rec(ctx)
with flops.BudgetContext(flop_budget=10**14,quiet=True) as first:
    fnp.multiply(post22,np.float32(3),out=slots[0])
    fnp.fill_diagonal(slots[0],post4)
report['first_gaussian_birth']=rec(first)
with threadpool_limits(limits=1):
    for lev in [0,3,4,5]:
        smm=SMM(np.float32,k)
        def query():
            fnp.multiply(P,P,out=S)
            smm.mm(S,G,R,lev)
            q=fnp.einsum('bpij,bpij->i',R,S)
            q=q*(fnp.mean(old)/fnp.mean(q))
            return q
        with flops.BudgetContext(flop_budget=10**14,quiet=True) as c0: q=query()
        with flops.BudgetContext(flop_budget=10**14,quiet=True) as c1: q=query()
        warm=rec(c1);cold=rec(c0);birth=report['birth'];setup=report['setup_buffers']
        total={key:setup[key]+14*birth[key]+report['first_gaussian_birth'][key]+cold[key]+8*warm[key] for key in ['flops','ops']}
        total['u']=total['flops']/2**31
        shares={key:setup[key]+14*birth[key]+report['first_gaussian_birth'][key]+9*warm[key] for key in ['flops','ops']};shares['u']=shares['flops']/2**31
        pools=sum(int(x.nbytes) for x in smm.flat.values())
        report['cases'][str(lev)]={'query_cold':cold,'query_warm':warm,'network_dedicated_pool':total,'network_if_pool_already_warm':shares,'pool_bytes':pools}
        print('LEVEL',lev,json.dumps(report['cases'][str(lev)]),flush=True)
        del smm;gc.collect()
(H/'meter_result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('BIRTH',json.dumps(report['birth']),'SETUP',json.dumps(report['setup_buffers']),flush=True)


