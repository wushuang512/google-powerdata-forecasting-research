"""Read-only independent formula reconstruction, including rounding disclosure."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round12'
sys.path.insert(0,str(ROOT/'code'))
import robust_fusion_experiment as e
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def canonical_dev_anchor():
    c=np.load(B/'round3/results/chronos_dev.npz')['native']
    t=np.load(B/'round3/results/tirex_dev.npz')['native']
    f=np.load(B/'round3/results/tf3_dev.npz')['tf3_cellprod']
    return np.clip(.25*c+.25*t+.5*f,0,1)
def reference_delta(d,origins,x,name,canonical=False):
    avg=canonical_dev_anchor() if canonical else np.clip(np.sum(x*np.array([.25,.25,.5])[:,None,None,None],axis=0),0,1)
    mid=np.sort(x,axis=0)[1];level=np.stack([np.maximum(np.mean(d['y'][:,o-7:o],axis=(1,2)),.05) for o in origins],axis=1)[:,:,None]
    if name=='point_median':consensus=mid
    elif name=='daily_consensus':
        xm=x.mean(axis=-1);consensus=avg+(np.sort(xm,axis=0)[1]-np.sum(xm*np.array([.25,.25,.5])[:,None,None],axis=0))[:,:,None]
    else:
        radius=(.01 if name=='winsor01' else .03)*level
        clipped=np.minimum(np.maximum(x,mid[None]-radius[None]),mid[None]+radius[None])
        consensus=np.sum(clipped*np.array([.25,.25,.5])[:,None,None,None],axis=0)
    return np.minimum(np.maximum(consensus-avg,-.01*level),.01*level)
def main():
    checks={};p=read(R/'docs/protocol.json');s=read(R/'results/selection.json');q=read(R/'results/summary.json')
    checks['script_hash']=sha(ROOT/'code/robust_fusion_experiment.py')==p['script_sha256']
    checks['protocol_hash']=sha(R/'docs/protocol.json')==s['protocol_sha256']
    checks['selection_hash']=sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
    checks['dev_predictions_hash']=sha(R/'results/dev_predictions.npz')==s['dev_predictions_sha256']
    checks['dev_summary_hash']=sha(R/'results/dev_summary.json')==s['dev_summary_sha256']
    for key,value in p['input_fingerprints'].items():checks['input_'+key]=sha(B/key)==value
    checks['freeze_before_test']=datetime.datetime.fromisoformat(s['frozen_utc'])<datetime.datetime.fromisoformat(q['completed_utc'])
    d=e.load();x=e.experts('dev');v=np.load(R/'results/dev_predictions.npz');y,m=e.truth(d,e.DEV)
    # Preserve actual registered arithmetic; audit the historical arithmetic
    # separately. Never replace or reselect the frozen candidate after TEST.
    avg=canonical_dev_anchor()
    old=.5*avg+.5*np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000']
    old_selection=read(B/'round8b/results/selection.json');candidate=np.load(B/'round8b/results/dev_predictions.npz')[old_selection['config']]
    canonical_prior=np.clip((1-old_selection['chosen']['alpha'])*old+old_selection['chosen']['alpha']*candidate,0,1)
    checks['canonical_dev_prior_matches_previous']=bool(np.array_equal(canonical_prior,np.load(B/'round11/results/dev_predictions.npz')['prior']))
    rounding=float(np.max(np.abs(canonical_prior-v['prior'])))
    checks['registered_dev_prior_reconstructed']=bool(np.array_equal(e.prior_dev(),v['prior']))
    for name in e.CONFIGS:
        dd=reference_delta(d,e.DEV,x,name)
        checks['dev_delta_'+name]=bool(np.array_equal(dd,v['delta_'+name]))
        checks['dev_prediction_'+name]=bool(np.array_equal(np.clip(v['prior']+dd,0,1),v[name]))
    choice=min(e.CONFIGS,key=lambda k:e.metrics(y[:,:4],v[k][:,:4],m[:,:4])['wape'])
    checks['early_only_family_selection']=choice==s['candidate']
    weights=[]
    for a in e.ALPHAS:
        pp=np.clip(v['prior'][:,4:]+a*v['delta_'+choice][:,4:],0,1)
        weights.append(dict(alpha=a,wape=e.metrics(y[:,4:],pp,m[:,4:])['wape'],**e.se(y[:,4:],m[:,4:],v['prior'][:,4:],pp,d['cells'])))
    win=min(weights,key=lambda z:z['wape']);pick=min([z for z in weights if z['wape']<=win['wape']+win['selection_se']+1e-12],key=lambda z:z['alpha'])
    checks['frozen_weight_reconstruction']=weights==s['weights'] and pick==s['chosen']
    canonical_predictions={k:np.clip(canonical_prior+reference_delta(d,e.DEV,x,k,True),0,1) for k in e.CONFIGS}
    canonical_choice=min(e.CONFIGS,key=lambda k:e.metrics(y[:,:4],canonical_predictions[k][:,:4],m[:,:4])['wape'])
    cw=[]
    cd=reference_delta(d,e.DEV,x,choice,True)
    for a in e.ALPHAS:
        pp=np.clip(canonical_prior[:,4:]+a*cd[:,4:],0,1)
        cw.append(dict(alpha=a,wape=e.metrics(y[:,4:],pp,m[:,4:])['wape'],**e.se(y[:,4:],m[:,4:],canonical_prior[:,4:],pp,d['cells'])))
    cwin=min(cw,key=lambda z:z['wape']);cpick=min([z for z in cw if z['wape']<=cwin['wape']+cwin['selection_se']+1e-12],key=lambda z:z['alpha'])
    checks['rounding_cannot_change_recorded_decision']=canonical_choice==choice and cpick['alpha']==s['chosen']['alpha']
    t=np.load(R/'results/test_predictions.npz');tx=e.experts('test');ty,tm=e.truth(d,e.TEST)
    dd=reference_delta(d,e.TEST,tx,choice)
    checks['test_delta_reconstructed']=bool(np.array_equal(dd,t['delta']))
    checks['test_prior_bitwise_previous']=bool(np.array_equal(t['prior'],np.load(B/'round8b/results/test_predictions.npz')['selected']))
    checks['test_candidate_reconstructed']=bool(np.array_equal(np.clip(t['prior']+dd,0,1),t['candidate']))
    checks['test_frozen_selected']=bool(np.array_equal(np.clip(t['prior']+s['chosen']['alpha']*dd,0,1),t['selected']))
    for name in ['prior','candidate','selected']:
        for scope,ix in [('all57',np.ones(57,bool)),('a5',d['cells']=='a')]:
            actual=e.metrics(ty[ix],t[name][ix],tm[ix]);checks[f'score_{name}_{scope}']=all(abs(actual[k]-q['scores'][name][scope][k])<1e-12 for k in actual)
    checks['selected_bootstrap_reconstructed']=e.bootstrap(ty,tm,t['selected'],t['prior'])==q['selected_day_bootstrap95_pp']
    checks['candidate_bootstrap_reconstructed']=e.bootstrap(ty,tm,t['candidate'],t['prior'])==q['candidate_day_bootstrap95_pp']
    dep=np.load(R/'results/deployment_components.npz');dx,gate=e.deployment_experts();new=pd.read_csv(R/'results/selected_deployment_origin31_1day.csv');oldcsv=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv')
    dd=reference_delta(d,[31],dx,choice)
    checks['deployment_delta_reconstructed']=bool(np.array_equal(dd,dep['delta']))
    checks['deployment_expert_mean_control']=gate['passed'] and gate['original_ensemble_max_difference']<1e-12
    checks['deployment_frozen_blend']=bool(np.array_equal(np.clip(dep['prior']+s['chosen']['alpha']*dd,0,1),dep['selected']))
    checks['deployment_csv_matches']=bool(np.max(np.abs(new.predicted_util.to_numpy()-dep['selected'].ravel()))<1e-15)
    checks['deployment_5472_57x96']=len(new)==5472 and new.series.nunique()==57 and new.groupby('series').size().eq(96).all()
    checks['deployment_time_alignment']=new[['series','cell','local_trace_time_PT']].equals(oldcsv[['series','cell','local_trace_time_PT']])
    checks['stored_deployment_checks']=all(read(R/'results/deployment_verification.json')['checks'].values())
    checks['no_learned_parameters']=read(R/'models/robust_fusion.json')['learned_parameters']==0 and p['model_parameters_learned']==0
    checks['resource_completion']=read(R/'results/guarded_pipeline_run.json')['exit_code']==0 and not read(R/'results/guarded_pipeline_run.json')['resource_pressure_stop']
    checks.update(e.causal(d,[14,21],x[:,:,[0,7]]));checks.update(e.causal(d,[22,30],tx[:,:,[0,8]]));checks.update(e.causal(d,[31],dx))
    checks={k:bool(v) for k,v in checks.items()}
    out=dict(checked_utc=e.utc(),check_count=len(checks),checks=checks,all_passed=bool(all(checks.values())),
        dev_arithmetic_audit=dict(max_power_util_difference_vs_canonical_round8b=rounding,
            reason='Stacking promotes original Chronos and TiRex float32 arrays to float64 before their partial sum; original source adds those two terms before TimesFM float64',
            original_frozen_selection_preserved=True,canonical_early_choice=canonical_choice,canonical_alpha=cpick['alpha'],
            canonical_prior_late4_wape=e.metrics(y[:,4:],canonical_prior[:,4:],m[:,4:])['wape'],
            canonical_late4_weights=cw,not_new_model_selection=True))
    print(json.dumps(out,ensure_ascii=False,indent=2))
    if not out['all_passed']:sys.exit(1)
if __name__=='__main__':main()
