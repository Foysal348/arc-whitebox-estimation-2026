from pathlib import Path
import numpy as np, json
import flopscope as flops
import flopscope.numpy as fnp
H=Path(__file__).resolve().parent
rng=np.random.default_rng(19); n=1024
res={'meter_version':flops.__version__,'n':n,'cases':[]}
for k in [7,15]:
    A,P,T,MP,F=[fnp.asarray(rng.standard_normal((k,n,n),dtype=np.float32)) for _ in range(5)]
    R,Y=[fnp.asarray(rng.standard_normal((k,n),dtype=np.float32)) for _ in range(2)]
    v=fnp.asarray(1+np.abs(rng.standard_normal(n,dtype=np.float32)))
    p0=fnp.asarray(rng.standard_normal(n,dtype=np.float32)); theta=np.float32(0.1)
    with flops.BudgetContext(flop_budget=10**14,quiet=True) as old:
        d=fnp.einsum('kij,kij,kij->i',A,A,T)*np.float32(3)+fnp.einsum('kij,kij->i',MP,P)
        d=d+fnp.einsum('kij,kij->i',P,F)*np.float32(3)
        d=d+fnp.einsum('ki,ki->i',R,Y)
    with flops.BudgetContext(flop_budget=10**14,quiet=True) as new:
        B=fnp.einsum('kij,kij,kij->ki',A,A,T)*np.float32(3)+fnp.einsum('kij,kij->ki',MP,P)
        B=B+fnp.einsum('kij,kij->ki',P,F)*np.float32(3)
        B=B+fnp.multiply(R,Y)
        dn=fnp.sum(B,axis=0)
        e=fnp.sum(B*B,axis=0)/(v*v*v)
        pred=p0+theta*e
    row={'k':k,'old_flops':old.flops_used,'new_flops':new.flops_used,'delta_flops':new.flops_used-old.flops_used,
         'old_ops':len(old._op_log),'new_ops':len(new._op_log),'delta_ops':len(new._op_log)-len(old._op_log),
         'diag_relative_difference':float(np.linalg.norm(np.asarray(d)-np.asarray(dn))/np.linalg.norm(np.asarray(d)))}
    res['cases'].append(row);print(json.dumps(row),flush=True)
    del A,P,T,MP,F,R,Y,B,d,dn
(H/'meter_birth_energy.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
