"""Bounded rank/robust-consensus correction around the fixed best predictor."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round12'
sys.path.insert(0,str(B/'code'))
from common import load,truth,metrics
TRAIN=list(range(7,14));DEV=list(range(14,22));TEST=list(range(22,31))
CONFIGS={'point_median':dict(kind='median'),
         'daily_consensus':dict(kind='daily'),
         'winsor01':dict(kind='winsor',radius=.01),
         'winsor03':dict(kind='winsor',radius=.03)}
ALPHAS=[0,.1,.25,.5,1]
W=np.array([.25,.25,.5])
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def experts(split):
    return np.stack([np.load(B/f'round3/results/{k}_{split}.npz')[key] for k,key in [('chronos','native'),('tirex','native'),('tf3','tf3_cellprod')]])
def anchor(x):return np.clip((W[:,None,None,None]*x).sum(0),0,1)
def levels(d,origins):return np.stack([d['y'][:,o-7:o].mean((1,2)) for o in origins],1)[:,:,None].clip(.05)
def delta(d,origins,x,name):
    cfg=CONFIGS[name];base=anchor(x);med=np.median(x,axis=0);level=levels(d,origins)
    if cfg['kind']=='median':q=med
    elif cfg['kind']=='daily':
        # Daily level consensus changes no forecast shape before final clipping.
        shift=np.median(x.mean(-1),axis=0)-(W[:,None,None]*x.mean(-1)).sum(0)
        q=base+shift[:,:,None]
    else:
        radius=cfg['radius']*level
        bounded=np.clip(x,med[None]-radius[None],med[None]+radius[None])
        q=(W[:,None,None,None]*bounded).sum(0)
    return np.clip(q-base,-.01*level,.01*level)
def prior_dev():
    x=experts('dev');base=anchor(x)
    p=np.clip(.5*base+.5*np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000'],0,1)
    s=read(B/'round8b/results/selection.json');ft=np.load(B/'round8b/results/dev_predictions.npz')[s['config']]
    return np.clip((1-s['chosen']['alpha'])*p+s['chosen']['alpha']*ft,0,1)
def register():
    for sub in ['docs','results','models']:(R/sub).mkdir(parents=True,exist_ok=True)
    files=[B/'results/data.npz',B/'code/common.py',B/'round8b/results/selection.json',B/'round8b/results/selection.sha256',
           B/'round8b/results/dev_predictions.npz',B/'round8b/results/test_predictions.npz',
           B/'round8b/results/selected_deployment_origin31_1day.csv',B/'round5/results/calibration_dev.npz',
           B/'round2/results/tirex_legacy_31_1day.npz',B/'round2/results/legacy_deployment_origin31_1day.csv',
           B/'round4/results/deployment_origin31_1day.npz',B/'round11/results/summary.json']
    for split in ['train','dev','test']:
        files.extend(B/f'round3/results/{k}_{split}.npz' for k in ['chronos','tirex','tf3'])
    protocol=dict(created_utc=utc(),method='robust forecast consensus correction around fixed round8b',
        registry=CONFIGS,alphas=ALPHAS,training_origins=TRAIN,early_dev_origins=DEV[:4],late_dev_origins=DEV[4:],test_origins=TEST,
        expert_definition='Original frozen univariate Chronos2, TiRex2 and cell-production TimesFM3 forecast caches; round8b adaptation remains in prior',
        correction='clip(prior + alpha * clip(robust_original_experts - original_.25_.25_.5_ensemble, +/-0.01*past7_mean),0,1)',
        selection='early4 choose family by macro all57 WAPE; late4 one-standard-error rule max(paired date SE,paired cell SE), prefer alpha0/smaller; hash freeze before TEST',
        model_parameters_learned=0,foundation_weights_changed=False,new_downloads=0,
        deployment_chronos='Reconstructed from existing legacy equal-mean CSV and stored univariate TiRex; validate original ensemble against round4 best_mix to 1e-12',
        independent_confirmation=False,development_reused=True,test_reused=True,
        script_sha256=sha(__file__),input_fingerprints={str(p.relative_to(B)):sha(p) for p in files})
    path=R/'docs/protocol.json'
    if path.exists():
        old=read(path)
        for k,v in protocol.items():
            if k!='created_utc':assert old[k]==v,f'Frozen protocol drift {k}'
    else:save(path,protocol)
    return read(path)
def dw(y,p,m):return 100*(np.abs(y-p)*m).sum((1,2))/(np.abs(y)*m).sum((1,2))
def se(y,m,prior,p,cells):
    dif=dw(y,prior,m)-dw(y,p,m)
    bycell=np.array([dif[cells==c].mean() for c in np.unique(cells)])
    byday=np.array([float(np.mean(dw(y[:,j:j+1],prior[:,j:j+1],m[:,j:j+1])-dw(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1]))) for j in range(y.shape[1])])
    a=float(bycell.std(ddof=1)/len(bycell)**.5);b=float(byday.std(ddof=1)/len(byday)**.5)
    return dict(paired_cell_se=a,paired_day_se=b,selection_se=max(a,b),daily_gain_pp=byday.tolist())
def causal(d,origins,x):
    checks={}
    for j,o in enumerate(origins):
        dd={k:v.copy() for k,v in d.items()};dd['y'][:,o:]=.917;dd['prod'][:,o:]=.138;dd['valid'][:,o:]=False;dd['pvalid'][:,o:]=False
        for name in CONFIGS:
            a=delta(d,[o],x[:,:,j:j+1],name);b=delta(dd,[o],x[:,:,j:j+1],name)
            checks[f'future_perturbation_{o}_{name}']=bool(np.array_equal(a,b))
    assert all(checks.values());return checks
def train_diagnostic(d):
    x=experts('train');y,m=truth(d,TRAIN);base=anchor(x)
    rawmed=np.median(x,axis=0)
    rows={k:metrics(y,np.clip(base+delta(d,TRAIN,x,k),0,1),m) for k in CONFIGS}
    return dict(scope='Training origins7..13 only; original unadapted backbone ensemble, not current best training performance',
        prior=metrics(y,base,m),unbounded_median=metrics(y,rawmed,m),bounded_candidates=rows,
        prediction_difference_std=float(np.std(np.diff(base,axis=-1))),power_difference_std=float(np.std(np.diff(y,axis=-1))))
def dev():
    register()
    if (R/'results/selection.json').exists():
        assert sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
        print('Retain existing frozen selection; no repeated DEV',flush=True);return
    d=load();d={k:(v[:,:22].copy() if k in ['y','prod','valid','pvalid'] else v) for k,v in d.items()}
    save(R/'results/training_diagnostic.json',train_diagnostic(d))
    x=experts('dev');base=prior_dev();y,m=truth(d,DEV);pred={};deltas={};rows=[]
    for name in CONFIGS:
        dd=delta(d,DEV,x,name);p=np.clip(base+dd,0,1);pred[name]=p;deltas[name]=dd
        row=dict(model=name,early4=metrics(y[:,:4],p[:,:4],m[:,:4]),late4=metrics(y[:,4:],p[:,4:],m[:,4:]),all8=metrics(y,p,m))
        rows.append(row);print('DEV',json.dumps(row),flush=True)
    name=min(CONFIGS,key=lambda k:metrics(y[:,:4],pred[k][:,:4],m[:,:4])['wape']);weights=[]
    for a in ALPHAS:
        p=np.clip(base[:,4:]+a*deltas[name][:,4:],0,1)
        weights.append(dict(alpha=a,wape=metrics(y[:,4:],p,m[:,4:])['wape'],**se(y[:,4:],m[:,4:],base[:,4:],p,d['cells'])))
    win=min(weights,key=lambda z:z['wape']);chosen=min([z for z in weights if z['wape']<=win['wape']+win['selection_se']+1e-12],key=lambda z:z['alpha'])
    selected=np.clip(base+chosen['alpha']*deltas[name],0,1)
    width=float(np.quantile((np.abs(y-selected)/levels(d,DEV))[m],.8,method='higher'))
    np.savez_compressed(R/'results/dev_predictions.npz',prior=base,selected=selected,**pred,**{'delta_'+k:v for k,v in deltas.items()})
    save(R/'results/dev_summary.json',dict(candidates=rows,prior=metrics(y,base,m),prior_early4=metrics(y[:,:4],base[:,:4],m[:,:4]),prior_late4=metrics(y[:,4:],base[:,4:],m[:,4:]),causal_checks=causal(d,[14,21],x[:, :, [0,7]])))
    s=dict(frozen_utc=utc(),candidate=name,chosen=chosen,weights=weights,protocol_sha256=sha(R/'docs/protocol.json'),
        dev_predictions_sha256=sha(R/'results/dev_predictions.npz'),dev_summary_sha256=sha(R/'results/dev_summary.json'),
        empirical80_halfwidth_relative_to_past_mean=width,independent_confirmation=False,development_reused=True,test_reused=True)
    save(R/'results/selection.json',s);(R/'results/selection.sha256').write_text(sha(R/'results/selection.json'),encoding='utf-8')
    save(R/'models/robust_fusion.json',dict(selection_sha256=sha(R/'results/selection.json'),prior_model='round8b',candidate=name,alpha=chosen['alpha'],learned_parameters=0,method=CONFIGS[name]))
    print('FROZEN',json.dumps(s),flush=True)
def frozen():
    protocol=register();s=read(R/'results/selection.json')
    assert sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
    assert s['protocol_sha256']==sha(R/'docs/protocol.json')
    for relative,h in protocol['input_fingerprints'].items():assert sha(B/relative)==h,relative
    return s
def bootstrap(y,m,p,prior):
    # Sampling dates jointly preserves cross-PDU co-movement.
    pe=(np.abs(y-p)*m).sum(-1);be=(np.abs(y-prior)*m).sum(-1);den=(np.abs(y)*m).sum(-1)
    rng=np.random.default_rng(20261003);out=[]
    for _ in range(5000):
        ix=rng.integers(0,y.shape[1],y.shape[1]);out.append(float(np.mean(100*(be[:,ix].sum(1)-pe[:,ix].sum(1))/den[:,ix].sum(1))))
    return np.quantile(out,[.025,.975]).tolist()
def test():
    s=frozen()
    if (R/'results/summary.json').exists():print('TEST scored once; no repeat selection',flush=True);return
    d=load();x=experts('test');base=np.load(B/'round8b/results/test_predictions.npz')['selected']
    dd=delta(d,TEST,x,s['candidate']);candidate=np.clip(base+dd,0,1);selected=np.clip(base+s['chosen']['alpha']*dd,0,1)
    y,m=truth(d,TEST);a5=d['cells']=='a';pred=dict(prior=base,candidate=candidate,selected=selected)
    scores={k:dict(all57=metrics(y,p,m),a5=metrics(y[a5],p[a5],m[a5])) for k,p in pred.items()}
    gain=scores['prior']['all57']['wape']-scores['selected']['all57']['wape'];rel=100*gain/scores['prior']['all57']['wape'];ci=bootstrap(y,m,selected,base)
    np.savez_compressed(R/'results/test_predictions.npz',**pred,delta=dd)
    rows=[]
    for j,o in enumerate(TEST):
        for name,p in pred.items():rows.append(dict(origin=o,target_date=f'2019-05-{o+1:02d}',model=name,**metrics(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1])))
    pd.DataFrame(rows).to_csv(R/'results/per_day_metrics.csv',index=False)
    pd.DataFrame(dict(domain=d['names'],cell=d['cells'],**{k:dw(y,p,m) for k,p in pred.items()})).to_csv(R/'results/per_domain_metrics.csv',index=False)
    out=dict(completed_utc=utc(),selection=s,scores=scores,selected_gain_pp=gain,selected_relative_reduction_percent=rel,
        selected_day_bootstrap95_pp=ci,candidate_day_bootstrap95_pp=bootstrap(y,m,candidate,base),
        large_gain_rule_met=bool(rel>=5 and ci[0]>0),causal_checks=causal(d,[22,30],x[:,:,[0,8]]),
        independent_confirmation=False,test_reused=True,development_reused=True,foundation_models_rerun=False)
    save(R/'results/summary.json',out);print('TEST',json.dumps(scores),flush=True)
def deployment_experts():
    t=np.load(B/'round2/results/tirex_legacy_31_1day.npz')['tirex_uni']
    csv=pd.read_csv(B/'round2/results/legacy_deployment_origin31_1day.csv')
    old=csv.predicted_util.to_numpy().reshape(57,1,96)
    c=2*old-t;f=np.load(B/'round4/results/deployment_origin31_1day.npz')['tf3_cellprod']
    x=np.stack([c,t,f]);reference=np.load(B/'round4/results/deployment_origin31_1day.npz')['best_mix']
    difference=float(np.max(np.abs(anchor(x)-reference)));assert difference<1e-12,difference
    return x,dict(original_ensemble_max_difference=difference,tolerance=1e-12,passed=True,chronos_reconstructed_from_equal_mean=True)
def deploy():
    s=frozen();d=load();x,control=deployment_experts();old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv')
    base=old.predicted_util.to_numpy().reshape(57,1,96);dd=delta(d,[31],x,s['candidate'])
    candidate=np.clip(base+dd,0,1);selected=np.clip(base+s['chosen']['alpha']*dd,0,1)
    old['round8b_prior_util']=base.ravel();old['robust_candidate_util']=candidate.ravel();old['predicted_util']=selected.ravel()
    half=(s['empirical80_halfwidth_relative_to_past_mean']*levels(d,[31])).ravel();half=np.repeat(half,96)
    old['lower80_empirical']=np.maximum(selected.ravel()-half,0);old['upper80_empirical']=np.minimum(selected.ravel()+half,1)
    old.to_csv(R/'results/selected_deployment_origin31_1day.csv',index=False)
    np.savez_compressed(R/'results/deployment_components.npz',prior=base,candidate=candidate,selected=selected,delta=dd,experts=x)
    checks=dict(rows_5472=len(old)==5472,series57=old.series.nunique()==57,finite=bool(np.isfinite(old.select_dtypes('number')).all().all()),
        bounds=bool(old.predicted_util.between(0,1).all()),intervals_ordered=bool(((old.lower80_empirical<=old.predicted_util)&(old.predicted_util<=old.upper80_empirical)).all()),
        june1_timestamps=bool(pd.to_datetime(old.local_trace_time_PT).min()==pd.Timestamp('2019-06-01') and pd.to_datetime(old.local_trace_time_PT).max()==pd.Timestamp('2019-06-01 23:45')),
        alpha0_prior_identity=bool(s['chosen']['alpha']!=0 or np.array_equal(base,selected)),deployment_control=control['passed'])
    checks.update(causal(d,[31],x));assert all(checks.values())
    save(R/'results/deployment_verification.json',dict(checks=checks,all_passed=True,control=control,not_live_forecast=True,no_june_truth=True))
    print('DEPLOY',checks,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','test','deploy','all'],default='all');a=ap.parse_args()
    if a.phase in ['dev','all']:dev()
    if a.phase in ['test','all']:test()
    if a.phase in ['deploy','all']:deploy()
if __name__=='__main__':main()
