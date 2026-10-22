#!/usr/bin/env python3
"""
fable_ct_agecollapsed_rank64.py

Prototype for the Fable 2026-10-09 C.T component.

Core identity
-------------
For target t and quartic-birth window m=t-6,...,t-1, define

    G[m,t] = g_m[:,None] * Yhub[m,t] * (C_m @ Ylin[m,t])

and for one symmetric K3 term with anchor leg A_s(m) and the two
target legs B_s(t), D_s(t),

    ct_{s,m,t}/8 = rowsum( (A_s(m)^T G[m,t])^T * (B_s(t)*D_s(t)) ).

Because A_s(m)=L[a->m] A_s(a), a=max(birth_s,t-6),

    sum_m A_s(m)^T G[m,t] = A_s(a)^T Gamma[a,t].

We never form Gamma. Let S_a be an r-dimensional sketch transported by
the SAME left map as the source leg:

    S_{k+1} = L_k S_k.

Then

    Pi[a,t] = S_a^T Gamma[a,t]
            = sum_{m=a}^{t-1} S_m^T G[m,t].

Approximate Gamma by its projection onto span(S_a):

    A^T Gamma ~= (A^T S_a (S_a^T S_a)^-1) Pi[a,t].

The final diagonal functional is evaluated without an n x n source product:

    B = A^T S_tilde              # k x r
    Z = (Btarget * Dtarget) @ B  # n x r
    out += rowsum(Z * Pi.T)

Two thin O(k*n*r) / O(n*k*r) GEMMs replace every dense n^3 Q product.

Project-specific hook
---------------------
I cannot reach the project machine in this chat. source_triplet() first tries
helpers in experiments/probes/fable_component_2026-10-09.py and then falls
back to blocks_of_source() in r7_srcgain.py. If your exact Fable loop uses a
different helper, edit ONLY TripletAdapter.get() so it returns the same
(X,P,Y) triple that the current exact Q_m(t) code uses.
"""

import argparse, importlib.util, inspect, json, re
from pathlib import Path
import numpy as np

try:
    from scipy.special import ndtr
except Exception:
    from math import erf
    def ndtr(x):
        x=np.asarray(x)
        return 0.5*(1.0+np.vectorize(erf)(x/np.sqrt(2.0)))


def pearson(a,b):
    a=np.asarray(a,dtype=np.float64).ravel(); b=np.asarray(b,dtype=np.float64).ravel()
    a=a-a.mean(); b=b-b.mean(); den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a.dot(b)/den) if den>0 else np.nan


def load_npz(path):
    z=np.load(path,allow_pickle=True)
    return {k:z[k] for k in z.files}


def load_py(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def norm_blocks(obj):
    if isinstance(obj,dict):
        if all(k in obj for k in ("X","P","Y")):
            return [(np.asarray(obj["X"]),np.asarray(obj["P"]),np.asarray(obj["Y"]))]
        for k in ("blocks","triples","terms"):
            if k in obj: return norm_blocks(obj[k])
    if isinstance(obj,tuple) and len(obj)==3 and all(hasattr(x,"shape") for x in obj):
        return [(np.asarray(obj[0]),np.asarray(obj[1]),np.asarray(obj[2]))]
    if isinstance(obj,(list,tuple)):
        out=[]
        for q in obj:
            if isinstance(q,(tuple,list)) and len(q)>=3 and all(hasattr(x,"shape") for x in q[:3]):
                out.append((np.asarray(q[0]),np.asarray(q[1]),np.asarray(q[2])))
            elif isinstance(q,dict) and all(k in q for k in ("X","P","Y")):
                out.append((np.asarray(q["X"]),np.asarray(q["P"]),np.asarray(q["Y"])))
        if out: return out
    raise TypeError(f"Cannot normalize helper output of type {type(obj)}")


class TripletAdapter:
    def __init__(self,root,block_mode="concat"):
        self.root=Path(root); self.block_mode=block_mode; self.helper=None; self.helper_name=None
        fp=self.root/"experiments/probes/fable_component_2026-10-09.py"
        if fp.exists():
            try:
                mod=load_py(fp,"_fable_component_probe")
                for name in ("source_triplet","get_source_triplet","build_source_triplet","triplet_of_source","blocks_of_source"):
                    fn=getattr(mod,name,None)
                    if callable(fn):
                        self.helper=fn; self.helper_name=f"{fp}:{name}"; break
            except Exception as e:
                print("WARN importing Fable component:",repr(e))
        if self.helper is None:
            rp=self.root/"experiments/probes/r7_srcgain.py"
            mod=load_py(rp,"_r7_srcgain_probe")
            self.helper=getattr(mod,"blocks_of_source")
            self.helper_name=f"{rp}:blocks_of_source"
        try: sig=str(inspect.signature(self.helper))
        except Exception: sig="<?>"
        print("Triplet helper:",self.helper_name,sig)

    def _call(self,cap,s):
        errs=[]
        for args in ((cap,s),(s,cap)):
            try: return self.helper(*args)
            except Exception as e: errs.append(repr(e))
        for kw in ({"cap":cap,"s":s},{"data":cap,"s":s},{"cap":cap,"source":s}):
            try: return self.helper(**kw)
            except Exception as e: errs.append(repr(e))
        raise RuntimeError("Could not call triplet helper; edit TripletAdapter.get(). First errors: "+" | ".join(errs[:3]))

    def get(self,cap,s):
        blocks=norm_blocks(self._call(cap,s))
        if self.block_mode=="first": return blocks[0],1
        X=np.concatenate([b[0] for b in blocks],axis=1)
        P=np.concatenate([b[1] for b in blocks],axis=1)
        Y=np.concatenate([b[2] for b in blocks],axis=1)
        return (X,P,Y),len(blocks)


def cap_C(cap):
    if "C" in cap: return np.asarray(cap["C"],dtype=np.float64)
    if "C_pre" in cap: return np.asarray(cap["C_pre"],dtype=np.float64)
    if "C_off" in cap and "var" in cap:
        C=np.asarray(cap["C_off"],dtype=np.float64).copy(); d=np.asarray(cap["var"],dtype=np.float64)
        C.flat[::C.shape[0]+1]+=d; return C
    raise KeyError("Need C or (C_off,var)")


def cap_mu_var(cap):
    mk=next((k for k in ("mu","mean","mu_pre") if k in cap),None)
    vk=next((k for k in ("var","variance","v") if k in cap),None)
    if mk is None or vk is None: raise KeyError(f"Need mu/var; keys={list(cap)[:30]}")
    return np.asarray(cap[mk],np.float64),np.asarray(cap[vk],np.float64)


def get_W(caps,root):
    for l in sorted(caps):
        for k in ("W_all","W","weights"):
            if k in caps[l] and np.asarray(caps[l][k]).ndim==3:
                return np.asarray(caps[l][k],dtype=np.float64)
    wp=Path(root)/"weights_seed0.npz"
    if wp.exists():
        z=load_npz(wp)
        for v in z.values():
            if np.asarray(v).ndim==3: return np.asarray(v,dtype=np.float64)
    raise KeyError("Could not find W_all/weights_seed0.npz")


def make_U(caps,W):
    U={}; g={}
    for l,cap in caps.items():
        mu,var=cap_mu_var(cap); sig=np.sqrt(np.maximum(var,1e-30)); alpha=mu/sig
        ph=ndtr(alpha); pdf=np.exp(-0.5*alpha*alpha)/np.sqrt(2*np.pi)
        g[l]=pdf/(2.0*sig)
        if l < W.shape[0]: U[l]=ph[:,None]*W[l]
    return U,g


def transports(m,t,U,W):
    # Row convention. Linear K3 legs include Phi_m; quadratic hub omits Phi_m on first step.
    Ylin=U[m].copy(); Yhub=W[m].copy()
    for k in range(m+1,t):
        Ylin=Ylin@U[k]; Yhub=Yhub@U[k]
    return Ylin,Yhub


def source_counts(caps):
    out={}
    for l,c in caps.items():
        if "P_st" in c: out[l]=int(np.asarray(c["P_st"]).shape[0])
        elif "A_st" in c: out[l]=int(np.asarray(c["A_st"]).shape[0])
        else: out[l]=0
    return out


def birth_layers(caps):
    cnt=source_counts(caps); maxs=max(cnt.values()); birth={}
    for s in range(maxs):
        ls=[l for l,n in cnt.items() if s<n]
        if ls: birth[s]=min(ls)
    return birth


def exact_ct_from_ceiling(ceil,t):
    # Prefer j-resolved exact CT keys.
    found=[]
    for k,v in ceil.items():
        if not k.startswith(f"L{t}"): continue
        kl=k.lower()
        if "ct_r" in kl or not kl.endswith("_ct"): continue
        m=re.search(r"_j(\d+)_",k)
        if m and np.asarray(v).ndim==1:
            found.append((int(m.group(1)),k,np.asarray(v,dtype=np.float64)))
    if found:
        found.sort(); return sum(x[2] for x in found),found
    cands=[(k,np.asarray(v,dtype=np.float64)) for k,v in ceil.items()
           if k.startswith(f"L{t}") and k.lower().endswith("_ct") and "ct_r" not in k.lower() and np.asarray(v).ndim==1]
    if len(cands)==1: return cands[0][1],[(0,cands[0][0],cands[0][1])]
    raise KeyError(f"Cannot identify exact CT keys for L{t}; got {[k for k in ceil if k.startswith('L'+str(t)) and 'ct' in k.lower()][:40]}")


def age_ablation(parts):
    total=sum(x[2] for x in parts); out=[]
    for j,k,v in parts:
        out.append({"age":int(j),"key":k,"corr_component_to_total":pearson(v,total),
                    "rms_fraction":float(np.linalg.norm(v)/max(np.linalg.norm(total),1e-30)),
                    "corr_leave_one_out_to_total":pearson(total-v,total)})
    return out


def choose_omega(n,r,seed):
    rng=np.random.default_rng(seed)
    return rng.choice(np.array([-1.0,1.0]),size=(n,r))/np.sqrt(r)


def approx_target(t,caps,W,U,g,birth,adapter,r=64,h=6,seed=1709):
    m0=max(1,t-h); n=W.shape[1]
    G={}
    for m in range(m0,t):
        Ylin,Yhub=transports(m,t,U,W)
        X=cap_C(caps[m])@Ylin  # DEPLOYMENT: reuse existing Fable X=C@Ylin.
        G[m]=g[m][:,None]*Yhub*X

    # Source legs use left map L=U.T in the current row-vector convention.
    Om=choose_omega(n,r,seed+t)
    S={m0:cap_C(caps[m0])@Om}
    for m in range(m0,t-1): S[m+1]=U[m].T@S[m]

    SG={m:S[m].T@G[m] for m in range(m0,t)}
    Pi={}; run=np.zeros((r,n),dtype=np.float64)
    for a in range(t-1,m0-1,-1):
        run=run+SG[a]; Pi[a]=run.copy()

    Stilde={}
    for a in range(m0,t):
        M=S[a].T@S[a]; ridge=1e-10*np.trace(M)/max(r,1)+1e-30
        Stilde[a]=S[a]@np.linalg.inv(M+ridge*np.eye(r))

    nsrc=source_counts(caps)[t]
    role_out=[np.zeros(n,dtype=np.float64) for _ in range(3)]
    interactions=0; width_factor=0.0
    for s in range(nsrc):
        a=max(birth[s],m0)
        if a>=t: continue
        (Xa,Pa,Ya),_ = adapter.get(caps[a],s)
        (Xt,Pt,Yt),_ = adapter.get(caps[t],s)
        anchors=(Xa,Pa,Ya); Hs=(Pt*Yt, Xt*Yt, Xt*Pt)
        width_factor=max(width_factor, Xa.shape[1]/n)
        for role in range(3):
            A=anchors[role]; H=Hs[role]
            B=A.T@Stilde[a]   # k x r
            Z=H@B             # n x r
            role_out[role]+=np.sum(Z*Pi[a].T,axis=1)
            interactions+=1
    role_out=[8.0*x for x in role_out]  # 24*(1/3 sym)
    return sum(role_out),role_out,{"m0":m0,"nsrc":nsrc,"interactions":interactions,"width_factor":width_factor}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--rank",type=int,default=64)
    ap.add_argument("--horizon",type=int,default=6)
    ap.add_argument("--targets",default="12,15,16")
    ap.add_argument("--ceiling",default="data/kaggle_out/fable_ceiling_all/fable_ceiling_all.npz")
    ap.add_argument("--block-mode",choices=["concat","first"],default="concat")
    ap.add_argument("--out",default="results/summaries/fable_ct_agecollapsed_r64.json")
    args=ap.parse_args(); root=Path(args.root); targets=[int(x) for x in args.targets.split(",") if x.strip()]

    caps={}
    for l in range(1,17):
        p=root/f"data/dumps/r7_cap_L{l}.npz"
        if p.exists(): caps[l]=load_npz(p)
    if len(caps)<16: raise FileNotFoundError(f"Expected r7_cap_L1..16; found {sorted(caps)}")
    W=get_W(caps,root); U,g=make_U(caps,W); birth=birth_layers(caps)
    adapter=TripletAdapter(root,args.block_mode); ceil=load_npz(root/args.ceiling)

    report={"rank":args.rank,"horizon":args.horizon,"targets":{}}
    for t in targets:
        exact,parts=exact_ct_from_ceiling(ceil,t)
        approx,roles,stats=approx_target(t,caps,W,U,g,birth,adapter,args.rank,args.horizon)
        item={"corr_approx_vs_exact_ct":pearson(approx,exact),
              "role_corr_vs_exact_ct":[pearson(x,exact) for x in roles],
              "age_ablation_exact":age_ablation(parts),**stats}
        report["targets"][str(t)]=item
        print(f"L{t}: corr approx/exact CT={item['corr_approx_vs_exact_ct']:.6f}; roles={item['role_corr_vs_exact_ct']}; {stats}")
        print(" exact age ablation:")
        for row in item["age_ablation_exact"]: print("  ",row)

    n=W.shape[1]; thin_u=args.rank/n
    # Worst-case deployment count for the Fable one-triplet model: 13 sources x 3 roles x 10 targets.
    n_targets=10; nsrc=13; roles=3
    shared=12*n_targets                 # C*Omega + 5 sketch transports + 6 S^T G per target
    per_source=2*nsrc*roles*n_targets  # A^T S and H@B
    report["derived_cost_one_triplet_worstcase"]={
        "thin_gemms":shared+per_source,
        "u_thin_gemms":(shared+per_source)*thin_u,
        "shared_thin_gemms":shared,"source_thin_gemms":per_source,
        "note":"Assumes X=C@Y and Yhub are reused from existing Fable/star path. If adapter concatenates k blocks, source cost scales by k."
    }
    print("DERIVED COST:",report["derived_cost_one_triplet_worstcase"])
    out=root/args.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print("Wrote",out)

if __name__=="__main__": main()
