"""Registered bounded nonlinear replacement of the existing Ridge residual.

No foundation model is loaded or downloaded. Model choice uses chronological
development slices only; previous May test reuse is explicitly disclosed.
"""
import argparse, datetime, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
B = ROOT.parent / 'experiment'
R = B / 'round11'
sys.path.insert(0, str(B / 'code'))
from common import load, truth, metrics
TRAIN = list(range(7, 14))
DEV = list(range(14, 22))
TEST = list(range(22, 31))
CONFIGS = {
    'l1_leaves7': dict(objective='regression_l1', num_leaves=7),
    'l1_leaves15': dict(objective='regression_l1', num_leaves=15),
    'l2_leaves7': dict(objective='regression', num_leaves=7),
    'l2_leaves15': dict(objective='regression', num_leaves=15),
}
FIXED = dict(n_estimators=80, learning_rate=.03, max_depth=4,
             min_child_samples=1024, reg_lambda=100, n_jobs=2,
             verbosity=-1, random_state=20261003, deterministic=True,
             force_col_wise=True, max_bin=63)
ALPHAS = [0, .1, .25, .5, 1]

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def anchor(split):
    with np.load(B/f'round3/results/tf3_{split}.npz') as z: f=z['tf3_cellprod']
    with np.load(B/f'round3/results/chronos_{split}.npz') as z: c=z['native']
    with np.load(B/f'round3/results/tirex_{split}.npz') as z: t=z['native']
    return np.clip(.25*c+.25*t+.5*f, 0, 1)

def bank(splits):
    result = {}
    for split in splits:
        a=anchor(split)
        result.update({o:a[:,j] for j,o in enumerate(dict(train=TRAIN,dev=DEV,test=TEST)[split])})
    return result

def feature(d,o,b):
    # Exact same inputs/order as the existing calibration.py. Only the
    # regression family changes, enabling a matched residual replacement.
    base=b[o]; y=d['y']; prod=d['prod']; pv=d['pvalid']; items=[]; names=[]
    def add(v,name):
        if v.ndim==1: v=np.repeat(v[:,None],96,axis=1)
        items.append(v.reshape(-1,1)); names.append(name)
    add(base,'base'); add(base-base.mean(1)[:,None],'predicted_shape')
    add(y[:,o-1]-base,'last_minus_base'); add(y[:,o-7]-base,'week_minus_base')
    add(y[:,o-3:o].mean(1)-base,'mean3_minus_base')
    for k in [1,3,7]:
        add(y[:,o-k:o].mean((1,2))-base.mean(1),f'level{k}_minus_base')
        add(y[:,o-k:o].std((1,2)),f'power_std{k}')
    count=pv[:,o-7:o].sum((1,2))
    pmean=(prod[:,o-7:o]*pv[:,o-7:o]).sum((1,2))/np.maximum(count,1)
    add(pmean,'past_proxy_mean'); add(count/(7*96),'proxy_availability7')
    add(pv[:,o-1].astype(float),'proxy_slot_available')
    add(np.where(pv[:,o-1],prod[:,o-1],pmean[:,None]),'last_proxy')
    avail=[k for k in range(max(7,o-7),o) if k in b]
    if avail:
        errs=np.stack([y[:,k]-b[k] for k in avail],1)
        mask=np.stack([d['valid'][:,k] for k in avail],1)
        daily=(errs*mask).sum(2)/mask.sum(2).clip(1)
        bias=(errs*mask).sum((1,2))/mask.sum((1,2)).clip(1)
        shape=(errs*mask).sum(1)/mask.sum(1).clip(1); last=daily[:,-1]
    else: bias=np.zeros(57); shape=np.zeros((57,96)); last=np.zeros(57)
    add(bias,'observed_bias7'); add(last,'observed_bias1'); add(shape,'observed_error_shape7')
    slot=np.arange(96)
    add(np.broadcast_to(np.sin(2*np.pi*slot/96),(57,96)),'hour_sin')
    add(np.broadcast_to(np.cos(2*np.pi*slot/96),(57,96)),'hour_cos')
    for c in np.unique(d['cells']): add((d['cells']==c).astype(float),f'cell_{c}')
    x=np.concatenate(items,1)
    assert np.isfinite(x).all()
    return x,base,names

def levels(d,o): return d['y'][:,o-7:o].mean((1,2)).clip(.05)
def prior_dev(b):
    s=read(B/'round8b/results/selection.json')
    old=np.clip(.5*anchor('dev')+.5*np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000'],0,1)
    with np.load(B/'round8b/results/dev_predictions.npz') as z: candidate=z[s['config']]
    return np.clip((1-s['chosen']['alpha'])*old+s['chosen']['alpha']*candidate,0,1)

def register():
    for sub in ['docs','results','models','code']: (R/sub).mkdir(parents=True,exist_ok=True)
    files=[B/'results/data.npz',B/'code/common.py',B/'round5/code/calibration.py',
           B/'round8b/results/selection.json',B/'round8b/results/selection.sha256',
           B/'round8b/results/dev_predictions.npz',B/'round8b/results/test_predictions.npz',
           B/'round8b/results/selected_deployment_origin31_1day.csv',
           B/'round5/results/calibration_dev.npz',B/'round5/results/test_candidates.npz',
           B/'round5/models/residual_ridge100000_dev.pkl',B/'round5/models/residual_ridge100000_final.pkl',
           B/'round4/results/deployment_origin31_1day.npz',
           B/'round5/results/selected_deployment_origin31_1day.csv']
    for split in ['train','dev','test']:
        files.extend([B/f'round3/results/{k}_{split}.npz' for k in ['tf3','chronos','tirex']])
    protocol=dict(created_utc=utc(),method='shallow nonlinear residual replacement inside fixed round8b',
        candidates=CONFIGS,fixed_parameters=FIXED,training_origins=TRAIN,
        development_family_origins=DEV[:4],development_weight_origins=DEV[4:],
        final_training_origins=TRAIN+DEV,test_origins=TEST,
        feature_source='exact existing historical Ridge features; no new future covariates',
        target='(observed power minus fixed backbone ensemble) / past7-day mean',
        sample_weight='past7 mean / training valid target sum per domain',
        correction='clip(best + alpha * 0.5 * clip(tree_residual - fixed_ridge_residual, +/-0.02*past7mean),0,1)',
        ridge_control='existing frozen round5 dev/final pickle, independently regressed against saved predictions',
        alphas=ALPHAS,
        selection='early4 choose family by all57 macro WAPE; late4 one standard error rule with max(paired date SE, paired cell SE), favor alpha0/smaller; freeze before TEST',
        large_gain_rule='>=5 percent relative WAPE reduction and positive paired-day bootstrap lower bound; exploratory only',
        training_days_short=True,development_reused=True,test_reused=True,independent_confirmation=False,
        model_downloads=0,foundation_weights_changed=False,script_sha256=sha(__file__),
        input_fingerprints={str(p.relative_to(B)):sha(p) for p in files})
    path=R/'docs/protocol.json'
    if path.exists():
        old=read(path)
        for k,v in protocol.items():
            if k!='created_utc': assert old[k]==v, f'Protocol drift {k}'
    else: save(path,protocol)
    return read(path)

def causal_check(d,b,origins):
    checks={}
    for o in origins:
        dd={k:v.copy() for k,v in d.items()}
        for k in ['y','prod']: dd[k][:,o:]=.913
        for k in ['valid','pvalid']: dd[k][:,o:]=False
        bb={k:v.copy() for k,v in b.items()}
        for k in bb:
            if k>o: bb[k][:]=.127
        checks[f'future_perturbation_origin{o}']=bool(np.array_equal(feature(d,o,b)[0],feature(dd,o,bb)[0]))
    assert all(checks.values()); return checks

def fit_tree(d,b,origins,name):
    xs=[];ys=[];ws=[]
    den=np.stack([np.abs(d['y'][:,o])*d['valid'][:,o] for o in origins],1).sum((1,2)).clip(1e-6)
    for o in origins:
        x,base,names=feature(d,o,b); level=levels(d,o)
        mask=d['valid'][:,o].ravel()
        xs.append(x[mask]); ys.append(((d['y'][:,o]-base)/level[:,None]).ravel()[mask])
        ws.append(np.repeat(level/den,96)[mask])
    x=np.concatenate(xs); y=np.concatenate(ys); w=np.concatenate(ws)
    w=w/w.mean()
    model=LGBMRegressor(**FIXED,**CONFIGS[name]); model.fit(x,y,sample_weight=w,feature_name=names)
    history=dict(origins=origins,valid_rows=len(y),features=names,actual_trees=model.n_estimators_,
                 total_leaves=sum(t['num_leaves'] for t in model.booster_.dump_model()['tree_info']))
    return model,history

def ridge(split):
    with (B/f'round5/models/residual_ridge100000_{split}.pkl').open('rb') as f: return pickle.load(f)
def correction(model,control,d,b,origins):
    ds=[];raw=[]
    for o in origins:
        x,_,_=feature(d,o,b); level=levels(d,o)[:,None]
        resid=model.predict(x).reshape(57,96)*level
        r=control.predict(x).reshape(57,96)
        raw.append(resid);ds.append(.5*np.clip(resid-r,-.02*level,.02*level))
    return np.stack(ds,1),np.stack(raw,1)
def control_gate(control,d,b,origins,reference):
    p=np.clip(np.stack([b[o]+control.predict(feature(d,o,b)[0]).reshape(57,96) for o in origins],1),0,1)
    diff=float(np.max(np.abs(p-reference))); assert diff<1e-12,f'Ridge control mismatch {diff}'
    return dict(max_absolute_difference=diff,tolerance=1e-12,passed=True)

def domain_wape(y,p,m): return 100*(np.abs(y-p)*m).sum((1,2))/(np.abs(y)*m).sum((1,2))
def paired_se(y,m,prior,p,cells):
    diff=domain_wape(y,prior,m)-domain_wape(y,p,m)
    bycell=np.array([diff[cells==c].mean() for c in np.unique(cells)])
    byday=np.array([float(np.mean(domain_wape(y[:,j:j+1],prior[:,j:j+1],m[:,j:j+1])-domain_wape(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1]))) for j in range(y.shape[1])])
    cellse=float(bycell.std(ddof=1)/len(bycell)**.5); dayse=float(byday.std(ddof=1)/len(byday)**.5)
    return dict(paired_cell_se=cellse,paired_day_se=dayse,selection_se=max(cellse,dayse),daily_gain_pp=byday.tolist())

def dev():
    register()
    if (R/'results/selection.json').exists():
        assert sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
        print('Existing frozen selection retained; no repeated DEV search',flush=True); return
    d=load();d={k:(v[:,:22].copy() if k in ['y','prod','valid','pvalid'] else v) for k,v in d.items()}
    b=bank(['train','dev']); base=prior_dev(b); control=ridge('dev')
    gate=control_gate(control,d,b,DEV,np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000'])
    causal=causal_check(d,b,[7,14,21]); y,m=truth(d,DEV)
    corrections={}; candidates={}; rows=[]
    for name in CONFIGS:
        model,history=fit_tree(d,b,TRAIN,name)
        delta,_=correction(model,control,d,b,DEV);p=np.clip(base+delta,0,1)
        corrections[name]=delta;candidates[name]=p
        row=dict(model=name,early4=metrics(y[:,:4],p[:,:4],m[:,:4]),late4=metrics(y[:,4:],p[:,4:],m[:,4:]),all8=metrics(y,p,m),training=history)
        rows.append(row)
        with (R/f'models/{name}_dev.pkl').open('wb') as f:pickle.dump(model,f)
        save(R/f'results/{name}_training.json',history);print('DEV',json.dumps(row),flush=True)
    best=min(CONFIGS,key=lambda k:metrics(y[:,:4],candidates[k][:,:4],m[:,:4])['wape'])
    weights=[]
    for a in ALPHAS:
        p=np.clip(base[:,4:]+a*corrections[best][:,4:],0,1)
        weights.append(dict(alpha=a,wape=metrics(y[:,4:],p,m[:,4:])['wape'],**paired_se(y[:,4:],m[:,4:],base[:,4:],p,d['cells'])))
    win=min(weights,key=lambda z:z['wape']);chosen=min([z for z in weights if z['wape']<=win['wape']+win['selection_se']+1e-12],key=lambda z:z['alpha'])
    selected=np.clip(base+chosen['alpha']*corrections[best],0,1)
    interval=float(np.quantile((np.abs(y-selected)/np.stack([levels(d,o) for o in DEV],1)[:,:,None])[m],.8,method='higher'))
    np.savez_compressed(R/'results/dev_predictions.npz',prior=base,selected=selected,**candidates,**{f'delta_{k}':v for k,v in corrections.items()})
    save(R/'results/dev_summary.json',dict(candidates=rows,prior=metrics(y,base,m),prior_early4=metrics(y[:,:4],base[:,:4],m[:,:4]),prior_late4=metrics(y[:,4:],base[:,4:],m[:,4:]),control=gate,causal_checks=causal))
    s=dict(frozen_utc=utc(),candidate=best,chosen=chosen,weights=weights,protocol_sha256=sha(R/'docs/protocol.json'),
           dev_predictions_sha256=sha(R/'results/dev_predictions.npz'),dev_summary_sha256=sha(R/'results/dev_summary.json'),
           selected_dev_model_sha256=sha(R/f'models/{best}_dev.pkl'),empirical80_halfwidth_relative_to_past_mean=interval,
           independent_confirmation=False,development_reused=True,test_reused=True)
    save(R/'results/selection.json',s);(R/'results/selection.sha256').write_text(sha(R/'results/selection.json'),encoding='utf-8')
    print('FROZEN',json.dumps(s),flush=True)

def frozen():
    protocol=register();s=read(R/'results/selection.json')
    assert sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
    assert s['protocol_sha256']==sha(R/'docs/protocol.json')
    for relative,h in protocol['input_fingerprints'].items():assert sha(B/relative)==h,relative
    return s

def bootstrap(y,m,p,base):
    rng=np.random.default_rng(20261003); values=[]
    for _ in range(5000):
        ix=rng.integers(0,len(TEST),len(TEST)); values.append(float(np.mean(domain_wape(y[:,ix],base[:,ix],m[:,ix])-domain_wape(y[:,ix],p[:,ix],m[:,ix]))))
    return np.quantile(values,[.025,.975]).tolist()

def test():
    s=frozen()
    if (R/'results/summary.json').exists():print('Frozen TEST already scored; retaining result',flush=True);return
    d=load();b=bank(['train','dev','test']);control=ridge('final')
    prior=np.load(B/'round8b/results/test_predictions.npz')['selected']
    gate=control_gate(control,d,b,TEST,np.load(B/'round5/results/test_candidates.npz')['calibration'])
    model,history=fit_tree(d,b,TRAIN+DEV,s['candidate'])
    with (R/'models/selected_candidate_final.pkl').open('wb') as f:pickle.dump(model,f)
    delta,_=correction(model,control,d,b,TEST)
    candidate=np.clip(prior+delta,0,1);selected=np.clip(prior+s['chosen']['alpha']*delta,0,1)
    y,m=truth(d,TEST);a5=np.array([str(c)=='a' for c in d['cells']]);assert a5.sum()==5
    predictions=dict(prior=prior,candidate=candidate,selected=selected)
    scores={k:dict(all57=metrics(y,p,m),a5=metrics(y[a5],p[a5],m[a5])) for k,p in predictions.items()}
    gain=scores['prior']['all57']['wape']-scores['selected']['all57']['wape'];rel=100*gain/scores['prior']['all57']['wape']
    ci=bootstrap(y,m,selected,prior);candidate_ci=bootstrap(y,m,candidate,prior)
    np.savez_compressed(R/'results/test_predictions.npz',**predictions,delta=delta)
    byday=[]
    for j,o in enumerate(TEST):
        for name,p in predictions.items():byday.append(dict(origin=o,target_date=f'2019-05-{o+1:02d}',model=name,**metrics(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1])))
    pd.DataFrame(byday).to_csv(R/'results/per_day_metrics.csv',index=False)
    pd.DataFrame(dict(domain=d['names'],cell=d['cells'],prior=domain_wape(y,prior,m),candidate=domain_wape(y,candidate,m),selected=domain_wape(y,selected,m))).to_csv(R/'results/per_domain_metrics.csv',index=False)
    summary=dict(completed_utc=utc(),selection=s,scores=scores,selected_gain_pp=gain,selected_relative_reduction_percent=rel,
                 selected_day_bootstrap95_pp=ci,candidate_day_bootstrap95_pp=candidate_ci,
                 large_gain_rule_met=bool(rel>=5 and ci[0]>0),control=gate,causal_checks=causal_check(d,b,[22,30]),
                 final_training=history,final_model_sha256=sha(R/'models/selected_candidate_final.pkl'),
                 foundation_models_rerun=False,independent_confirmation=False,development_reused=True,test_reused=True)
    save(R/'results/summary.json',summary);print('TEST',json.dumps(scores),flush=True)

def deploy():
    s=frozen();summary=read(R/'results/summary.json');d=load();b=bank(['train','dev','test'])
    old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv')
    raw=np.load(B/'round4/results/deployment_origin31_1day.npz')
    b[31]=raw['best_mix'][:,0]
    with (R/'models/selected_candidate_final.pkl').open('rb') as f:model=pickle.load(f)
    control=ridge('final')
    r=control.predict(feature(d,31,b)[0]).reshape(57,96)
    old5=pd.read_csv(B/'round5/results/selected_deployment_origin31_1day.csv').predicted_util.to_numpy().reshape(57,96)
    expected5=np.clip(.5*b[31]+.5*np.clip(b[31]+r,0,1),0,1)
    control_diff=float(np.max(np.abs(expected5-old5)));assert control_diff<1e-12,control_diff
    delta,_=correction(model,control,d,b,[31]);prior=old.predicted_util.to_numpy().reshape(57,1,96)
    candidate=np.clip(prior+delta,0,1);selected=np.clip(prior+s['chosen']['alpha']*delta,0,1)
    old['round8b_prior_util']=prior.ravel();old['nonlinear_candidate_util']=candidate.ravel();old['predicted_util']=selected.ravel()
    half=s['empirical80_halfwidth_relative_to_past_mean']*np.repeat(levels(d,31),96)
    old['lower80_empirical']=np.maximum(selected.ravel()-half,0);old['upper80_empirical']=np.minimum(selected.ravel()+half,1)
    old.to_csv(R/'results/selected_deployment_origin31_1day.csv',index=False)
    np.savez_compressed(R/'results/deployment_components.npz',prior=prior,candidate=candidate,selected=selected,delta=delta,anchor=b[31])
    checks=dict(rows_5472=len(old)==5472,finite=bool(np.isfinite(old.select_dtypes('number')).all().all()),bounds=bool(old.predicted_util.between(0,1).all()),
                intervals_ordered=bool(((old.lower80_empirical<=old.predicted_util)&(old.predicted_util<=old.upper80_empirical)).all()),
                alpha0_exactly_prior=bool(s['chosen']['alpha']!=0 or np.array_equal(prior,selected)),
                causal_origin31=causal_check(d,b,[31])['future_perturbation_origin31'],
                ridge_deployment_control=control_diff<1e-12,
                timestamps_june1=bool(pd.to_datetime(old.local_trace_time_PT).min()==pd.Timestamp('2019-06-01') and pd.to_datetime(old.local_trace_time_PT).max()==pd.Timestamp('2019-06-01 23:45')),
                series57=old.series.nunique()==57)
    assert all(checks.values());save(R/'results/deployment_verification.json',dict(checks=checks,all_passed=True,forecast_date='2019-06-01',historical_inference_not_live=True,independent_interval_guarantee=False))
    print('DEPLOY',checks,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','test','deploy','all'],default='all');a=ap.parse_args()
    if a.phase in ['dev','all']:dev()
    if a.phase in ['test','all']:test()
    if a.phase in ['deploy','all']:deploy()
if __name__=='__main__':main()
