"""Chronologically selected convex, recency-weighted residual replacement.

The test month has already been inspected. This is an exploratory experiment,
not independent confirmation. No foundation weights or new data are loaded.
"""
import argparse, ast, datetime, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
B=ROOT.parent/'experiment'; R=B/'round13'
sys.path.insert(0,str(B/'code'))
from common import load,truth,metrics
TRAIN=list(range(7,14)); DEV=list(range(14,22)); TEST=list(range(22,31))
CONFIGS={
    'recent_ridge_h3':dict(half_life=3,loss='l2'),
    'recent_ridge_h7':dict(half_life=7,loss='l2'),
    'smoothL1_h3_p05':dict(half_life=3,loss='smooth_absolute',penalty_factor=.5),
    'smoothL1_h3_p2':dict(half_life=3,loss='smooth_absolute',penalty_factor=2),
    'smoothL1_h7_p05':dict(half_life=7,loss='smooth_absolute',penalty_factor=.5),
    'smoothL1_h7_p2':dict(half_life=7,loss='smooth_absolute',penalty_factor=2),
}
ALPHAS=[0,.1,.25,.5,1]
FIXED=dict(ridge_alpha=100000,epsilon_fraction_past7mean=.005,max_iterations=80,
           objective_relative_tolerance=1e-8,delta_cap_fraction_past7mean=.01,
           recency_total_weight_renormalized=True,cpu_threads=2)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

# Compile only the audited original historical feature function. Avoid importing
# its lib5 module, which loads Torch and foundation-model helpers unnecessarily.
src=B/'round5/code/calibration.py'
tree=ast.parse(src.read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='features')
namespace={'np':np}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),namespace)
feature=namespace['features']

def anchor(split):
    with np.load(B/f'round3/results/chronos_{split}.npz') as z:c=z['native']
    with np.load(B/f'round3/results/tirex_{split}.npz') as z:t=z['native']
    with np.load(B/f'round3/results/tf3_{split}.npz') as z:f=z['tf3_cellprod']
    # Preserve the arithmetic order and source dtypes of the old ensemble.
    return np.clip(.25*c+.25*t+.5*f,0,1)
def bank(splits):
    out={}
    for split in splits:
        a=anchor(split)
        out.update({o:a[:,j] for j,o in enumerate(dict(train=TRAIN,dev=DEV,test=TEST)[split])})
    return out
def levels(d,o):return d['y'][:,o-7:o].mean((1,2)).clip(.05)

def register():
    for sub in ['docs','results','models']:(R/sub).mkdir(parents=True,exist_ok=True)
    files=[B/'results/data.npz',B/'code/common.py',src,
           B/'round11/results/dev_predictions.npz',B/'round8b/results/selection.json',
           B/'round8b/results/test_predictions.npz',B/'round8b/results/selected_deployment_origin31_1day.csv',
           B/'round5/results/calibration_dev.npz',B/'round5/results/test_candidates.npz',
           B/'round4/results/deployment_origin31_1day.npz',B/'round5/results/selected_deployment_origin31_1day.csv']
    for split in ['train','dev','test']:
        files += [B/f'round3/results/{model}_{split}.npz' for model in ['chronos','tirex','tf3']]
    p=dict(created_utc=utc(),method='low-capacity convex residual retraining with recency weights and smooth absolute loss',
           configs=CONFIGS,fixed_parameters=FIXED,training_origins=TRAIN,
           family_selection_origins=DEV[:4],weight_selection_origins=DEV[4:],final_training_origins=TRAIN+DEV,test_origins=TEST,
           feature_source='unmodified AST-extracted 30 historical features from round5/calibration.py',
           target='observed normalized power minus original three-backbone anchor',
           original_sample_weight='1/max(mean observed power strictly before origin,0.05)',
           recency_weight='2**(-(latest training origin-origin)/half_life), renormalized to original sum of weights',
           scaling='unweighted training-only StandardScaler-compatible mean/std; zero or numerically constant features have scale 1',
           ridge_loss='0.5 sum(w*(y-Xb-intercept)**2)+0.5*100000*sum(b**2)',
           robust_loss='sum(w*sqrt((y-Xb-intercept)**2+(.005*past7mean)**2))+0.5*lambda*sum(b**2)',
           robust_lambda='100000/max(median(abs(training residual of matching recency ridge)),1e-4)*penalty_factor',
           optimizer='IRLS initialized from matching ridge; objective must decrease and converge within 80 iterations; no test-dependent stopping',
           correction='clip(best+alpha*clip(0.5*(clip(anchor+new_residual,0,1)-old_clipped_calibration), +/-.01*past7mean),0,1)',
           alphas=ALPHAS,selection='early4 select converged family by all57 macro WAPE; late4 one-SE with max paired date and cell SE, favor zero/smaller; freeze before TEST',
           model_dimensions=31,intercept_unpenalized=True,large_gain_rule='>=5% relative WAPE reduction and positive paired-date bootstrap lower bound; exploratory only',
           foundation_weights_changed=False,model_downloads=0,development_reused=True,test_reused=True,independent_confirmation=False,
           script_sha256=sha(__file__),input_fingerprints={str(p.relative_to(B)):sha(p) for p in files})
    path=R/'docs/protocol.json'
    if path.exists():
        old=read(path)
        for k,v in p.items():
            if k!='created_utc':assert old[k]==v,f'Protocol drift: {k}'
    else:save(path,p)
    return read(path)

def training_arrays(d,b,origins,half_life):
    xs=[];ys=[];ws=[];rs=[];eps=[];roworig=[]
    for o in origins:
        x,base,names=feature(d,o,b);mask=d['valid'][:,o].ravel()
        xs.append(x[mask]);ys.append((d['y'][:,o]-base).ravel()[mask])
        w=np.repeat(1/np.maximum(d['y'][:,:o].mean((1,2)),.05),96)[mask]
        ws.append(w);rs.append(np.full(len(w),1 if half_life is None else 2**(-(max(origins)-o)/half_life)))
        eps.append(np.repeat(.005*levels(d,o),96)[mask]);roworig.extend([o]*len(w))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);recent=np.concatenate(rs)
    weight=w*recent;weight*=w.sum()/weight.sum()
    mean=x.mean(0);var=x.var(0);scale=np.sqrt(var)
    constant=var<=len(x)*np.finfo(float).eps*var+(len(x)*mean*np.finfo(float).eps)**2
    scale[constant]=1
    z=(x-mean)/scale;z=np.column_stack([np.ones(len(z)),z])
    assert np.isfinite(z).all() and np.isfinite(y).all() and (weight>0).all()
    return z,y,weight,np.concatenate(eps),mean,scale,names,np.array(roworig)
def solve(z,y,w,penalty):
    gram=z.T@(w[:,None]*z);gram.flat[::len(gram)+1]+=np.r_[0,np.full(z.shape[1]-1,penalty)]
    return np.linalg.solve(gram,z.T@(w*y))
def fit(d,b,origins,config):
    z,y,w,eps,mean,scale,names,roworig=training_arrays(d,b,origins,config.get('half_life'))
    beta=solve(z,y,w,100000);scale_resid=max(float(np.median(np.abs(y-z@beta))),1e-4)
    history=dict(origins=origins,rows=len(y),features=names,train_residual_scale=scale_resid,
                 feature_count=len(names),learned_parameters=len(beta),training_weight_sum=float(w.sum()),
                 weight_by_origin={str(o):float(w[roworig==o].sum()) for o in origins},config=config)
    if config['loss']=='smooth_absolute':
        penalty=100000/scale_resid*config['penalty_factor']
        def objective(v):return float((w*np.sqrt((y-z@v)**2+eps**2)).sum()+.5*penalty*np.dot(v[1:],v[1:]))
        values=[objective(beta)];converged=False
        for i in range(80):
            updated=solve(z,y,w/np.sqrt((y-z@beta)**2+eps**2),penalty)
            value=objective(updated)
            assert value<=values[-1]+1e-8*max(1,values[-1]),'IRLS objective increased'
            relative=abs(value-values[-1])/max(1,values[-1]);values.append(value);beta=updated
            if relative<1e-8:converged=True;break
        history.update(penalty=penalty,objective_values=values,iterations=len(values)-1,converged=converged,final_relative_change=relative)
    else:
        gradient=z.T@(w*(z@beta-y))+np.r_[0,100000*beta[1:]]
        history.update(penalty=100000,iterations=1,converged=True,normal_equation_gradient_max=float(np.max(np.abs(gradient))))
    assert np.isfinite(beta).all()
    return dict(mean=mean,scale=scale,beta=beta),history
def predict(model,x):return np.column_stack([np.ones(len(x)),(x-model['mean'])/model['scale']])@model['beta']
def save_model(path,model):np.savez_compressed(path,**model)
def load_model(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def calibrated(model,d,b,origins):
    return np.stack([np.clip(b[o]+predict(model,feature(d,o,b)[0]).reshape(57,96),0,1) for o in origins],1)
def control_gate(d,b,trainorig,origins,reference):
    model,h=fit(d,b,trainorig,dict(half_life=None,loss='l2'))
    actual=calibrated(model,d,b,origins);difference=float(np.max(np.abs(actual-reference)))
    assert difference<1e-10,f'Original Ridge control mismatch {difference}'
    return dict(max_absolute_difference=difference,tolerance=1e-10,passed=True,training=h)
def correction(model,d,b,origins,oldcal):
    new=calibrated(model,d,b,origins)
    level=np.stack([levels(d,o) for o in origins],1)[:,:,None]
    return np.clip(.5*(new-oldcal),-.01*level,.01*level),new
def causal_check(d,b,origins):
    checks={}
    for o in origins:
        dd={k:v.copy() for k,v in d.items()}
        for k in ['y','prod']:dd[k][:,o:]=.913
        for k in ['valid','pvalid']:dd[k][:,o:]=False
        bb={k:v.copy() for k,v in b.items()}
        for k in bb:
            if k>o:bb[k][:]=.127
        checks[f'future_perturbation_origin{o}']=bool(np.array_equal(feature(d,o,b)[0],feature(dd,o,bb)[0]))
    assert all(checks.values());return checks
def domain_wape(y,p,m):return 100*(np.abs(y-p)*m).sum((1,2))/(np.abs(y)*m).sum((1,2))
def paired_se(y,m,base,p,cells):
    diff=domain_wape(y,base,m)-domain_wape(y,p,m)
    bycell=np.array([diff[cells==c].mean() for c in np.unique(cells)])
    byday=np.array([float(np.mean(domain_wape(y[:,j:j+1],base[:,j:j+1],m[:,j:j+1])-domain_wape(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1]))) for j in range(y.shape[1])])
    cs=float(bycell.std(ddof=1)/len(bycell)**.5);ds=float(byday.std(ddof=1)/len(byday)**.5)
    return dict(paired_cell_se=cs,paired_day_se=ds,selection_se=max(cs,ds),daily_gain_pp=byday.tolist())
def weight_selection(y,m,base,delta,cells):
    rows=[]
    for a in ALPHAS:
        p=np.clip(base+a*delta,0,1)
        rows.append(dict(alpha=a,wape=metrics(y,p,m)['wape'],**paired_se(y,m,base,p,cells)))
    win=min(rows,key=lambda z:z['wape'])
    choice=min([z for z in rows if z['wape']<=win['wape']+win['selection_se']+1e-12],key=lambda z:z['alpha'])
    return rows,choice
def dev():
    register()
    if (R/'results/selection.json').exists():
        assert sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
        print('Existing frozen DEV selection retained',flush=True);return
    d=load();d={k:(v[:,:22].copy() if k in ['y','prod','valid','pvalid'] else v) for k,v in d.items()}
    b=bank(['train','dev']);base=np.load(B/'round11/results/dev_predictions.npz')['prior']
    old=np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000']
    gate=control_gate(d,b,TRAIN,DEV,old);causal=causal_check(d,b,[7,14,21]);y,m=truth(d,DEV)
    rows=[];candidates={};deltas={}
    for name,config in CONFIGS.items():
        model,h=fit(d,b,TRAIN,config);delta,newcal=correction(model,d,b,DEV,old)
        p=np.clip(base+delta,0,1);deltas[name]=delta;candidates[name]=p
        save_model(R/f'models/{name}_dev.npz',model);save(R/f'results/{name}_training.json',h)
        row=dict(model=name,early4=metrics(y[:,:4],p[:,:4],m[:,:4]),late4=metrics(y[:,4:],p[:,4:],m[:,4:]),all8=metrics(y,p,m),training=h)
        rows.append(row);print('DEV',json.dumps(row),flush=True)
    eligible=[r['model'] for r in rows if r['training']['converged']]
    assert eligible,'No converged registered candidate'
    chosen=min(eligible,key=lambda k:metrics(y[:,:4],candidates[k][:,:4],m[:,:4])['wape'])
    weights,pick=weight_selection(y[:,4:],m[:,4:],base[:,4:],deltas[chosen][:,4:],d['cells'])
    selected=np.clip(base+pick['alpha']*deltas[chosen],0,1)
    half=float(np.quantile((np.abs(y-selected)/np.stack([levels(d,o) for o in DEV],1)[:,:,None])[m],.8,method='higher'))
    np.savez_compressed(R/'results/dev_predictions.npz',prior=base,selected=selected,**candidates,**{f'delta_{k}':v for k,v in deltas.items()})
    save(R/'results/dev_summary.json',dict(candidates=rows,prior=metrics(y,base,m),prior_early4=metrics(y[:,:4],base[:,:4],m[:,:4]),prior_late4=metrics(y[:,4:],base[:,4:],m[:,4:]),control=gate,causal_checks=causal))
    s=dict(frozen_utc=utc(),candidate=chosen,chosen=pick,weights=weights,protocol_sha256=sha(R/'docs/protocol.json'),
           dev_predictions_sha256=sha(R/'results/dev_predictions.npz'),dev_summary_sha256=sha(R/'results/dev_summary.json'),
           selected_dev_model_sha256=sha(R/f'models/{chosen}_dev.npz'),empirical80_halfwidth_relative_to_past_mean=half,
           independent_confirmation=False,development_reused=True,test_reused=True)
    save(R/'results/selection.json',s);(R/'results/selection.sha256').write_text(sha(R/'results/selection.json'),encoding='utf-8')
    print('FROZEN',json.dumps(s),flush=True)
def frozen():
    p=register();s=read(R/'results/selection.json')
    assert sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
    assert s['protocol_sha256']==sha(R/'docs/protocol.json')
    assert s['dev_predictions_sha256']==sha(R/'results/dev_predictions.npz')
    assert s['selected_dev_model_sha256']==sha(R/f"models/{s['candidate']}_dev.npz")
    for relative,h in p['input_fingerprints'].items():assert sha(B/relative)==h,relative
    return s
def bootstrap(y,m,p,base):
    rng=np.random.default_rng(20261004);v=[]
    for _ in range(5000):
        ix=rng.integers(0,len(TEST),len(TEST));v.append(float(np.mean(domain_wape(y[:,ix],base[:,ix],m[:,ix])-domain_wape(y[:,ix],p[:,ix],m[:,ix]))))
    return np.quantile(v,[.025,.975]).tolist()
def test():
    s=frozen()
    if (R/'results/summary.json').exists():print('Frozen TEST already scored; retaining original',flush=True);return
    d=load();b=bank(['train','dev','test']);old=np.load(B/'round5/results/test_candidates.npz')['calibration']
    gate=control_gate(d,b,TRAIN+DEV,TEST,old)
    model,h=fit(d,b,TRAIN+DEV,CONFIGS[s['candidate']]);assert h['converged'],'Selected final fit did not converge; stop without reselection'
    save_model(R/'models/selected_candidate_final.npz',model);save(R/'results/final_training.json',h)
    # Record final training before first test scoring; no test-derived refit.
    save(R/'results/final_model_frozen.json',dict(at_utc=utc(),selection_sha256=sha(R/'results/selection.json'),model_sha256=sha(R/'models/selected_candidate_final.npz'),training=h))
    base=np.load(B/'round8b/results/test_predictions.npz')['selected'];delta,_=correction(model,d,b,TEST,old)
    ps=dict(prior=base,candidate=np.clip(base+delta,0,1),selected=np.clip(base+s['chosen']['alpha']*delta,0,1))
    y,m=truth(d,TEST);a5=d['cells']=='a';assert a5.sum()==5
    scores={k:dict(all57=metrics(y,p,m),a5=metrics(y[a5],p[a5],m[a5])) for k,p in ps.items()}
    np.savez_compressed(R/'results/test_predictions.npz',**ps,delta=delta)
    rows=[]
    for j,o in enumerate(TEST):
        for name,p in ps.items():rows.append(dict(origin=o,target_date=f'2019-05-{o+1:02d}',model=name,**metrics(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1])))
    pd.DataFrame(rows).to_csv(R/'results/per_day_metrics.csv',index=False)
    pd.DataFrame(dict(domain=d['names'],cell=d['cells'],**{k:domain_wape(y,p,m) for k,p in ps.items()})).to_csv(R/'results/per_domain_metrics.csv',index=False)
    gain=scores['prior']['all57']['wape']-scores['selected']['all57']['wape'];rel=100*gain/scores['prior']['all57']['wape']
    ci=bootstrap(y,m,ps['selected'],base)
    q=dict(completed_utc=utc(),selection=s,scores=scores,selected_gain_pp=gain,selected_relative_reduction_percent=rel,
           selected_day_bootstrap95_pp=ci,candidate_day_bootstrap95_pp=bootstrap(y,m,ps['candidate'],base),
           large_gain_rule_met=bool(rel>=5 and ci[0]>0),control=gate,causal_checks=causal_check(d,b,[22,30]),
           final_training=h,final_model_sha256=sha(R/'models/selected_candidate_final.npz'),
           foundation_models_rerun=False,independent_confirmation=False,development_reused=True,test_reused=True)
    save(R/'results/summary.json',q);print('TEST',json.dumps(scores),flush=True)
def deploy():
    s=frozen();q=read(R/'results/summary.json');d=load();b=bank(['train','dev','test'])
    b[31]=np.load(B/'round4/results/deployment_origin31_1day.npz')['best_mix'][:,0]
    old5=pd.read_csv(B/'round5/results/selected_deployment_origin31_1day.csv').predicted_util.to_numpy().reshape(57,96)
    oldcal=np.clip(2*old5-b[31],0,1)[:,None,:]
    gate=control_gate(d,b,TRAIN+DEV,[31],oldcal)
    model=load_model(R/'models/selected_candidate_final.npz');assert sha(R/'models/selected_candidate_final.npz')==q['final_model_sha256']
    delta,_=correction(model,d,b,[31],oldcal);out=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv')
    base=out.predicted_util.to_numpy().reshape(57,1,96);candidate=np.clip(base+delta,0,1);selected=np.clip(base+s['chosen']['alpha']*delta,0,1)
    out['round8b_prior_util']=base.ravel();out['recency_robust_candidate_util']=candidate.ravel();out['predicted_util']=selected.ravel()
    half=s['empirical80_halfwidth_relative_to_past_mean']*np.repeat(levels(d,31),96)
    out['lower80_empirical']=np.maximum(selected.ravel()-half,0);out['upper80_empirical']=np.minimum(selected.ravel()+half,1)
    out.to_csv(R/'results/selected_deployment_origin31_1day.csv',index=False)
    np.savez_compressed(R/'results/deployment_components.npz',prior=base,candidate=candidate,selected=selected,delta=delta,anchor=b[31],old_calibration=oldcal)
    checks=dict(rows_5472=len(out)==5472,series57x96=out.series.nunique()==57 and bool(out.groupby('series').size().eq(96).all()),
                finite=bool(np.isfinite(out.select_dtypes('number')).all().all()),bounds=bool(out.predicted_util.between(0,1).all()),
                intervals_ordered=bool(((out.lower80_empirical<=out.predicted_util)&(out.predicted_util<=out.upper80_empirical)).all()),
                alpha0_exactly_prior=bool(s['chosen']['alpha']!=0 or np.array_equal(base,selected)),
                timestamps_june1=bool(pd.to_datetime(out.local_trace_time_PT).min()==pd.Timestamp('2019-06-01') and pd.to_datetime(out.local_trace_time_PT).max()==pd.Timestamp('2019-06-01 23:45')),
                ridge_deployment_control=gate['passed'],causal_origin31=causal_check(d,b,[31])['future_perturbation_origin31'])
    assert all(checks.values())
    save(R/'results/deployment_verification.json',dict(checks=checks,all_passed=True,control=gate,forecast_date='2019-06-01',historical_inference_not_live=True,independent_interval_guarantee=False))
    print('DEPLOY',checks,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['register','dev','test','deploy','all'],default='all');a=ap.parse_args()
    if a.phase=='register':register()
    if a.phase in ['dev','all']:dev()
    if a.phase in ['test','all']:test()
    if a.phase in ['deploy','all']:deploy()
if __name__=='__main__':main()
