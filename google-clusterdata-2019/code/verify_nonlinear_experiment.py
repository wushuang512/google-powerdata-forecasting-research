"""Read-only reconstruction of frozen selection, features and delivered arrays."""
import ast,datetime,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round11'
sys.path.insert(0,str(ROOT/'code'))
import nonlinear_residual_experiment as e
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    checks={};s=read(R/'results/selection.json');q=read(R/'results/summary.json');protocol=read(R/'docs/protocol.json')
    checks['selection_hash']=sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
    checks['protocol_hash']=sha(R/'docs/protocol.json')==s['protocol_sha256']
    checks['implementation_hash']=sha(ROOT/'code/nonlinear_residual_experiment.py')==protocol['script_sha256']
    for relative,h in protocol['input_fingerprints'].items():checks['source_'+relative]=sha(B/relative)==h
    checks['dev_prediction_hash']=sha(R/'results/dev_predictions.npz')==s['dev_predictions_sha256']
    checks['dev_summary_hash']=sha(R/'results/dev_summary.json')==s['dev_summary_sha256']
    checks['selected_dev_model_hash']=sha(R/f'models/{s["candidate"]}_dev.pkl')==s['selected_dev_model_sha256']
    checks['final_model_hash']=sha(R/'models/selected_candidate_final.pkl')==q['final_model_sha256']
    checks['freeze_before_test']=datetime.datetime.fromisoformat(s['frozen_utc'])<datetime.datetime.fromisoformat(q['completed_utc'])
    d=e.load();b=e.bank(['train','dev','test']);dv=np.load(R/'results/dev_predictions.npz');ty,tm=e.truth(d,e.TEST);dy,dm=e.truth(d,e.DEV)
    # Compile only the original feature function, without heavyweight imports.
    tree=ast.parse((B/'round5/code/calibration.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='features')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'original_features','exec'),ns)
    for o in [7,13,14,21,22,30]:
        actual=e.feature(d,o,b);expected=ns['features'](d,o,b)
        checks[f'original_feature_equivalence_{o}']=bool(np.array_equal(actual[0],expected[0]) and actual[2]==expected[2])
    chosen=min(e.CONFIGS,key=lambda k:e.metrics(dy[:,:4],dv[k][:,:4],dm[:,:4])['wape'])
    checks['early_only_family_choice']=chosen==s['candidate']
    weights=[]
    for a in e.ALPHAS:
        p=np.clip(dv['prior'][:,4:]+a*dv['delta_'+chosen][:,4:],0,1)
        w=dict(alpha=a,wape=e.metrics(dy[:,4:],p,dm[:,4:])['wape'],**e.paired_se(dy[:,4:],dm[:,4:],dv['prior'][:,4:],p,d['cells']))
        weights.append(w)
    win=min(weights,key=lambda z:z['wape']);pick=min([z for z in weights if z['wape']<=win['wape']+win['selection_se']+1e-12],key=lambda z:z['alpha'])
    checks['late_only_alpha_choice']=pick==s['chosen'] and weights==s['weights']
    checks['development_prior_matches_r8b']=bool(np.array_equal(dv['prior'],e.prior_dev(b)))
    for name in e.CONFIGS:
        with (R/f'models/{name}_dev.pkl').open('rb') as f:model=pickle.load(f)
        delta,_=e.correction(model,e.ridge('dev'),d,b,e.DEV)
        checks[f'dev_reconstruct_{name}']=bool(np.array_equal(delta,dv['delta_'+name]) and np.array_equal(np.clip(dv['prior']+delta,0,1),dv[name]))
    with (R/'models/selected_candidate_final.pkl').open('rb') as f:model=pickle.load(f)
    delta,_=e.correction(model,e.ridge('final'),d,b,e.TEST);z=np.load(R/'results/test_predictions.npz')
    checks['test_delta_reconstructed']=bool(np.array_equal(delta,z['delta']))
    checks['test_candidate_reconstructed']=bool(np.array_equal(np.clip(z['prior']+delta,0,1),z['candidate']))
    checks['test_selected_frozen_blend']=bool(np.array_equal(np.clip(z['prior']+s['chosen']['alpha']*delta,0,1),z['selected']))
    checks['test_prior_unchanged']=bool(np.array_equal(z['prior'],np.load(B/'round8b/results/test_predictions.npz')['selected']))
    a5=d['cells']=='a'
    for name in ['prior','candidate','selected']:
        for scope,ix in [('all57',np.ones(57,bool)),('a5',a5)]:
            actual=e.metrics(ty[ix],z[name][ix],tm[ix])
            checks[f'score_{name}_{scope}']=all(abs(actual[k]-q['scores'][name][scope][k])<1e-12 for k in actual)
    old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv');new=pd.read_csv(R/'results/selected_deployment_origin31_1day.csv')
    dep=np.load(R/'results/deployment_components.npz');b[31]=np.load(B/'round4/results/deployment_origin31_1day.npz')['best_mix'][:,0]
    ddelta,_=e.correction(model,e.ridge('final'),d,b,[31])
    checks['deploy_delta_reconstructed']=bool(np.array_equal(ddelta,dep['delta']))
    checks['deploy_prior_from_r8b']=bool(np.max(np.abs(old.predicted_util.to_numpy()-dep['prior'].ravel()))<1e-15)
    checks['deploy_selected_reconstructed']=bool(np.array_equal(np.clip(dep['prior']+s['chosen']['alpha']*ddelta,0,1),dep['selected']))
    checks['deploy_csv_matches']=bool(np.max(np.abs(new.predicted_util.to_numpy()-dep['selected'].ravel()))<1e-15)
    checks['rows_and_series']=len(new)==5472 and new.groupby('series').size().eq(96).all() and new.series.nunique()==57
    checks['timestamp_alignment']=bool(new[['series','cell','local_trace_time_PT']].equals(old[['series','cell','local_trace_time_PT']]))
    checks['deployment_report']=all(read(R/'results/deployment_verification.json')['checks'].values())
    levels=np.stack([e.levels(d,o) for o in e.DEV],1)[:,:,None]
    width=float(np.quantile((np.abs(dy-dv['selected'])/levels)[dm],.8,method='higher'))
    checks['dev_only_interval_width']=width==s['empirical80_halfwidth_relative_to_past_mean']
    checks['deploy_interval_bounds']=bool(((new.lower80_empirical>=0)&(new.upper80_empirical<=1)&(new.lower80_empirical<=new.predicted_util)&(new.predicted_util<=new.upper80_empirical)).all())
    run=read(R/'results/guarded_pipeline_run.json')
    checks['resource_safe_completion']=run['exit_code']==0 and not run['resource_pressure_stop'] and run['minimum_available_physical_bytes']>=1.5*2**30 and run['minimum_available_commit_bytes']>=2*2**30
    checks['limited_registered_search']=len(e.CONFIGS)==4 and protocol['development_reused'] and protocol['test_reused'] and not protocol['independent_confirmation']
    checks.update(e.causal_check(d,b,[7,14,21,22,30,31]))
    out=dict(checked_utc=e.utc(),checks=checks,check_count=len(checks),all_passed=bool(all(checks.values())))
    print(json.dumps(out,ensure_ascii=False,indent=2))
    if not out['all_passed']:sys.exit(1)
if __name__=='__main__':main()
