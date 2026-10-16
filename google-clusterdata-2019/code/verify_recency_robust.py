"""Read-only reconstruction of chronology, fitting stationarity and forecasts."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round13'
sys.path.insert(0,str(ROOT/'code'))
import recency_robust_experiment as e
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def predict(model,x):return model['beta'][0]+((x-model['mean'])/model['scale'])@model['beta'][1:]
def delta(model,d,b,origins,oldcal):
    power=np.stack([np.clip(b[o]+predict(model,e.feature(d,o,b)[0]).reshape(57,96),0,1) for o in origins],1)
    cap=.01*np.stack([np.maximum(d['y'][:,o-7:o].mean((1,2)),.05) for o in origins],1)[:,:,None]
    return np.minimum(np.maximum(.5*(power-oldcal),-cap),cap)
def score(y,p,m):
    err=np.abs(y-p);wape=float(np.mean(100*(err*m).sum((1,2))/(np.abs(y)*m).sum((1,2))))
    count=m.sum(-1);a=(y*m).sum(-1)/count.clip(1);pred=(p*m).sum(-1)/count.clip(1);ok=count>0
    daily=float(np.mean(100*(np.abs(a-pred)*ok).sum(1)/(np.abs(a)*ok).sum(1)))
    high=np.where(m,y,-np.inf).max(-1);ph=np.where(m,p,-np.inf).max(-1)
    peak=float(100*np.mean((np.abs(high-ph)/np.maximum(high,1e-8))[ok]))
    return dict(wape=wape,daily_wape=daily,peak_mape=peak)
def stationarity(model,d,b,origins,config):
    z,y,w,eps,mean,scale,names,roworig=e.training_arrays(d,b,origins,config['half_life'])
    beta=model['beta'];res=z@beta-y
    if config['loss']=='l2':penalty=100000;gradient=z.T@(w*res)
    else:
        ridge=e.solve(z,y,w,100000);train_scale=max(float(np.median(np.abs(y-z@ridge))),1e-4)
        penalty=100000/train_scale*config['penalty_factor']
        gradient=z.T@(w*res/np.sqrt(res**2+eps**2))
    gradient[1:]+=penalty*beta[1:]
    relative=float(np.max(np.abs(gradient))/max(1,np.max(np.abs(z.T@(w*y))),np.max(np.abs(penalty*beta[1:]))))
    return relative, bool(np.array_equal(model['mean'],mean) and np.array_equal(model['scale'],scale)),bool(np.isfinite(beta).all())
def main():
    p=read(R/'docs/protocol.json');s=read(R/'results/selection.json');q=read(R/'results/summary.json');v=np.load(R/'results/dev_predictions.npz')
    checks={};details={}
    checks['script_hash']=sha(ROOT/'code/recency_robust_experiment.py')==p['script_sha256']
    checks['protocol_hash']=sha(R/'docs/protocol.json')==s['protocol_sha256']
    checks['selection_hash']=sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
    for k,h in p['input_fingerprints'].items():checks['input_'+k]=sha(B/k)==h
    for k in ['dev_predictions','dev_summary']:
        suffix='npz' if k=='dev_predictions' else 'json'
        checks[k+'_hash']=sha(R/f'results/{k}.{suffix}')==s[k+'_sha256']
    checks['selected_dev_model_hash']=sha(R/f"models/{s['candidate']}_dev.npz")==s['selected_dev_model_sha256']
    freeze=read(R/'results/final_model_frozen.json')
    checks['final_model_hash']=sha(R/'models/selected_candidate_final.npz')==q['final_model_sha256']==freeze['model_sha256']
    checks['chronological_freeze']=datetime.datetime.fromisoformat(s['frozen_utc'])<datetime.datetime.fromisoformat(freeze['at_utc'])<datetime.datetime.fromisoformat(q['completed_utc'])
    checks['registered_six_configs']=len(p['configs'])==6 and p['configs']==e.CONFIGS
    checks['registered_dates']=p['training_origins']==list(range(7,14)) and p['family_selection_origins']==[14,15,16,17] and p['weight_selection_origins']==[18,19,20,21] and p['final_training_origins']==list(range(7,22))
    d=e.load();b=e.bank(['train','dev','test']);dy,dm=e.truth(d,e.DEV)
    oldcal=np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000']
    checks['dev_prior_bitwise']=bool(np.array_equal(v['prior'],np.load(B/'round11/results/dev_predictions.npz')['prior']))
    eligible=[]
    for name,config in e.CONFIGS.items():
        model=e.load_model(R/f'models/{name}_dev.npz');h=read(R/f'results/{name}_training.json')
        dd=delta(model,d,b,e.DEV,oldcal)
        checks['dev_delta_'+name]=bool(np.max(np.abs(dd-v['delta_'+name]))<2e-15)
        checks['dev_pred_'+name]=bool(np.max(np.abs(np.clip(v['prior']+dd,0,1)-v[name]))<2e-15)
        rel,scaled,finite=stationarity(model,d,b,e.TRAIN,config)
        details[name]=dict(relative_stationarity=rel)
        checks['stationarity_'+name]=rel<.005
        checks['training_scaling_'+name]=scaled;checks['finite_model_'+name]=finite
        checks['training_origins_'+name]=h['origins']==list(range(7,14)) and h['rows']==int(sum(d['valid'][:,o].sum() for o in e.TRAIN))
        checks['train_only_convergence_'+name]=h['converged'] and h['iterations']<=80
        if 'objective_values' in h:checks['decreasing_objective_'+name]=bool(np.all(np.diff(h['objective_values'])<=1e-8))
        if h['converged']:eligible.append(name)
    choice=min(eligible,key=lambda name:score(dy[:,:4],v[name][:,:4],dm[:,:4])['wape'])
    checks['early_only_family']=choice==s['candidate']
    weights,pick=e.weight_selection(dy[:,4:],dm[:,4:],v['prior'][:,4:],v['delta_'+choice][:,4:],d['cells'])
    checks['late_only_weight']=weights==s['weights'] and pick==s['chosen']
    model=e.load_model(R/'models/selected_candidate_final.npz');h=read(R/'results/final_training.json')
    checks['final_training_dates']=h['origins']==list(range(7,22)) and h['converged']
    rel,scaled,finite=stationarity(model,d,b,e.TRAIN+e.DEV,e.CONFIGS[choice]);details['final']=dict(relative_stationarity=rel)
    checks['final_stationarity']=rel<.005;checks['final_training_scaling']=scaled;checks['final_finite_model']=finite
    t=np.load(R/'results/test_predictions.npz');y,m=e.truth(d,e.TEST)
    oldtest=np.load(B/'round5/results/test_candidates.npz')['calibration'];dd=delta(model,d,b,e.TEST,oldtest)
    checks['test_delta_rebuilt']=bool(np.max(np.abs(dd-t['delta']))<2e-15)
    checks['test_candidate_rebuilt']=bool(np.max(np.abs(np.clip(t['prior']+dd,0,1)-t['candidate']))<2e-15)
    checks['test_prior_bitwise']=bool(np.array_equal(t['prior'],np.load(B/'round8b/results/test_predictions.npz')['selected']))
    checks['test_selected_frozen']=bool(np.array_equal(np.clip(t['prior']+s['chosen']['alpha']*t['delta'],0,1),t['selected']))
    for name in ['prior','candidate','selected']:
        for scope,ix in [('all57',np.ones(57,bool)),('a5',d['cells']=='a')]:
            actual=score(y[ix],t[name][ix],m[ix]);checks[f'score_{name}_{scope}']=all(abs(actual[k]-q['scores'][name][scope][k])<1e-12 for k in actual)
    checks['candidate_bootstrap']=e.bootstrap(y,m,t['candidate'],t['prior'])==q['candidate_day_bootstrap95_pp']
    checks['selected_bootstrap']=e.bootstrap(y,m,t['selected'],t['prior'])==q['selected_day_bootstrap95_pp']
    dep=np.load(R/'results/deployment_components.npz');new=pd.read_csv(R/'results/selected_deployment_origin31_1day.csv');old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv')
    b[31]=np.load(B/'round4/results/deployment_origin31_1day.npz')['best_mix'][:,0]
    dd=delta(model,d,b,[31],dep['old_calibration'])
    checks['deployment_delta_rebuilt']=bool(np.max(np.abs(dd-dep['delta']))<2e-15)
    checks['deployment_selected_frozen']=bool(np.array_equal(np.clip(dep['prior']+s['chosen']['alpha']*dep['delta'],0,1),dep['selected']))
    checks['deployment_csv_matches']=bool(np.max(np.abs(new.predicted_util.to_numpy()-dep['selected'].ravel()))<2e-15)
    checks['deployment_57x96']=len(new)==5472 and new.series.nunique()==57 and bool(new.groupby('series').size().eq(96).all())
    checks['deployment_time_alignment']=new[['series','cell','local_trace_time_PT']].equals(old[['series','cell','local_trace_time_PT']])
    checks['stored_deployment_checks']=read(R/'results/deployment_verification.json')['all_passed'] and all(read(R/'results/deployment_verification.json')['checks'].values())
    checks.update(e.causal_check(d,b,[7,14,21,22,30,31]))
    run=read(R/'results/guarded_pipeline_run.json');checks['guard_complete']=run['exit_code']==0 and not run['resource_pressure_stop']
    checks['truthful_limits']=not q['independent_confirmation'] and q['test_reused'] and q['development_reused']
    checks={k:bool(v) for k,v in checks.items()}
    out=dict(checked_utc=e.utc(),check_count=len(checks),checks=checks,all_passed=all(checks.values()),optimizer_checks=details)
    print(json.dumps(out,ensure_ascii=False,indent=2))
    if not out['all_passed']:sys.exit(1)
if __name__=='__main__':main()
