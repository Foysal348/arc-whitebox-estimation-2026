from pathlib import Path
import argparse, collections, hashlib, json, math, statistics
p=argparse.ArgumentParser(description='Offline oracle audit only; no estimator runs or policy training.')
p.add_argument('--root',default=r'D:\Machine Learning 1 Year Challenge\Kaggle\ARC White-Box Estimation Challenge 2026')
p.add_argument('--out',required=True)
a=p.parse_args(); root=Path(a.root)
refpath=root/'data/kaggle_out/v56t_1/run_v56_ref.json'
ref=json.loads(refpath.read_text(encoding='utf-8-sig'))
names=sorted(x['mlp_name'] for x in ref['results']['per_mlp'])
sha=ref['run_config']['dataset']['sha256']; runs=[]; rejected=collections.Counter()
for path in sorted((root/'data/kaggle_out').rglob('run_*.json')):
    try:
        blob=path.read_bytes(); d=json.loads(blob.decode('utf-8-sig'))
        cfg,res=d['run_config'],d['results']; rows=res['per_mlp']
        if cfg['width']!=1024 or cfg['depth']!=16 or cfg['dataset']['sha256']!=sha:
            rejected['different_network_set']+=1; continue
        if len(rows)!=len(names) or sorted(r['mlp_name'] for r in rows)!=names or len(set(names))!=len(names):
            rejected['unmatched_names']+=1; continue
        flags=('budget_exhausted','time_exhausted','residual_wall_time_exhausted','combined_budget_exhausted')
        if res['n_failed_mlps'] or any(any(r.get(k,False) for k in flags) or r.get('traceback') for r in rows):
            rejected['failed_run']+=1; continue
        rows=sorted(rows,key=lambda r:r['mlp_name'])
        losses=[float(r['adjusted_final_layer_score']) for r in rows]
        recomputed=[r['final_layer_mse']*max(.1,r['effective_compute']/cfg['flop_budget']) for r in rows]
        if not all(math.isfinite(x) and math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-18) for x,y in zip(losses,recomputed)):
            rejected['score_formula_mismatch']+=1; continue
        runs.append(dict(path=path.relative_to(root).as_posix(),sha256=hashlib.sha256(blob).hexdigest(),losses=losses,
            mean_score=statistics.mean(losses),mean_flop_u=statistics.mean(r['flops_used']/2**31 for r in rows),
            mean_effective_u=statistics.mean(r['effective_compute']/2**31 for r in rows),lambda_flops_per_second=cfg.get('lambda_flops_per_second')))
    except (KeyError,TypeError,ValueError): rejected['invalid_schema']+=1
refname=refpath.relative_to(root).as_posix(); refscore=next(r['mean_score'] for r in runs if r['path']==refname)
def summary(bank):
    best=min(bank,key=lambda r:r['mean_score'])
    choices=[min(range(len(bank)),key=lambda j:bank[j]['losses'][i]) for i in range(len(names))]
    oracle=statistics.mean(bank[j]['losses'][i] for i,j in enumerate(choices))
    return dict(n_experts=len(bank),best_fixed=best['path'],best_fixed_score=best['mean_score'],oracle_score=oracle,
        oracle_over_reference=oracle/refscore,oracle_over_best_fixed=oracle/best['mean_score'],
        gain_percent_over_best_fixed=100*(1-oracle/best['mean_score']),minimum_mean_flop_u=min(r['mean_flop_u'] for r in bank),
        winner_counts=dict(collections.Counter(bank[j]['path'] for j in choices)))
out=dict(status='OFFLINE OUTCOME ORACLE ONLY; no routing policy trained or deployed',pilot_cost_included=False,
    n_networks=len(names),dataset_sha256=sha,reference=refname,reference_score=refscore,
    groups={'v56':summary([r for r in runs if Path(r['path']).parent.name.startswith('v56t_')]),'all':summary(runs)},
    rejected=dict(rejected),runs=[{k:v for k,v in r.items() if k!='losses'} for r in runs])
Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='runs'},indent=2))
