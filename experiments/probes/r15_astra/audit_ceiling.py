from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
out={}
for L in [7,11,14,15]:
    a=np.load(HERE/f'birth_features_L{L}.npz'); dec=a['dec']; v=a['v']; vt=a['vt']; g=a['g4']
    F=v[:,None]**2*a['X']; Fa=np.c_[F,v*v*a['q']]
    def projection(X,y):
        scale=np.linalg.norm(X,axis=0); scale[scale==0]=1
        return X@(np.linalg.lstsq(X/scale,y,rcond=1e-12)[0]/scale)
    def bounds(X):
        # Relaxation: extra standardized intercept, arbitrary per-layer coefficients,
        # true target on every tested neuron, and no level-preservation constraint.
        T=np.c_[X/vt[:,None]**2,np.ones(len(v))][dec]
        pred=projection(T,a['t'][dec])
        wtarget=g*a['target']; wpred=projection(g[:,None]*X,wtarget)
        e=np.linalg.norm(wpred-wtarget)
        return {'pattern_corr_oracle':float(np.corrcoef(pred,a['t'][dec])[0,1]),
          'G4_norm_oracle_vs_original_chain':float(e/np.linalg.norm(g*(a['old']-a['target']))),
          'G4_squared_error_oracle':float(e*e)}
    b=bounds(F); h=bounds(Fa)
    out[L]={'base':b,'plus_birth_energy':h,'G4_norm_ratio_oracles':float(np.sqrt(h['G4_squared_error_oracle']/b['G4_squared_error_oracle']))}
(HERE/'audit_ceiling.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2),flush=True)
