"""Independent chronological selection and stored network reconstruction."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];TOP=ROOT.parent;B=TOP/'experiment';R=B/'round14'
sys.path.insert(0,str(ROOT/'code'))
import direct_shared_experiment as e
from direct_networks import DirectNetwork
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
e.TORCH=torch;e.DEVICE='cpu'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def score(y,p,m):
    err=np.abs(y-p);wape=float(np.mean(100*(err*m).sum((1,2))/(np.abs(y)*m).sum((1,2))))
    count=m.sum(-1);a=(y*m).sum(-1)/count.clip(1);pred=(p*m).sum(-1)/count.clip(1);ok=count>0
    daily=float(np.mean(100*(np.abs(a-pred)*ok).sum(1)/(np.abs(a)*ok).sum(1)))
    high=np.where(m,y,-np.inf).max(-1);ph=np.where(m,p,-np.inf).max(-1)
    peak=float(100*np.mean((np.abs(high-ph)/np.maximum(high,1e-8))[ok]))
    return dict(wape=wape,daily_wape=daily,peak_mape=peak)
def network_forecast(path,d,origins):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['protocol_sha256']==e.fingerprint()
    net=DirectNetwork(**e.CONFIGS[cp['name']]);net.load_state_dict(cp['state']);net.eval()
    arrays=e.inputs(d,np.array(origins)*96);parts=[]
    with torch.inference_mode():
        for start in range(0,len(arrays[0]),128):
            args=[torch.from_numpy(a[start:start+128]) for a in arrays[:6]]
            parts.append(net(*args).clamp(0,1).numpy())
    return np.concatenate(parts).reshape(len(origins),57,96).transpose(1,0,2).astype('float64'),cp
def main():
    p=read(R/'docs/protocol.json');s=read(R/'results/selection.json');q=read(R/'results/summary.json');f=read(R/'results/final_model_frozen.json')
    checks={};differences={}
    checks['main_script_hash']=sha(ROOT/'code/direct_shared_experiment.py')==p['script_sha256']
    checks['protocol_hash']=sha(R/'docs/protocol.json')==s['protocol_sha256']
    checks['selection_hash']=sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
    for k,v in p['file_fingerprints'].items():checks['source_'+k]=sha(TOP/k)==v
    for k in ['dev_predictions','dev_summary']:
        suffix='npz' if k=='dev_predictions' else 'json';checks[k+'_hash']=sha(R/f'results/{k}.{suffix}')==s[k+'_sha256']
    checks['freeze_chronology']=datetime.datetime.fromisoformat(s['frozen_utc'])<datetime.datetime.fromisoformat(f['at_utc'])<datetime.datetime.fromisoformat(q['completed_utc'])
    checks['six_configs_two_seeds_three_checkpoints']=len(p['configs'])==6 and p['seeds']==[20261004,20261005] and p['checkpoints']==[20,60,120]
    checks['no_small_correction_cap']=p['fixed_parameters']['no_residual_cap'] and 1 in e.ALPHAS
    d=e.load();y,m=e.truth(d,e.DEV);v=np.load(R/'results/dev_predictions.npz');rows=[]
    for name in e.CONFIGS:
        for seed in e.SEEDS:
            meta=read(R/f'results/dev_{name}_seed{seed}_training_meta.json')
            checks[f'train_boundary_{name}_{seed}']=meta['cutoff_days']==14 and meta['maximum_target_end']<=1344 and meta['protocol_sha256']==e.fingerprint()
        for ep in e.CHECKPOINTS:
            key=f'{name}_e{ep}';parts=[]
            for seed in e.SEEDS:
                with np.load(R/f'results/{key}_dev_seed{seed}.npz') as a:
                    checks[f'cache_binding_{key}_{seed}']=str(a['protocol_sha256'])==e.fingerprint();parts.append(a['prediction'])
            checks['seed_average_'+key]=bool(np.array_equal((parts[0]+parts[1])/2,v[key]))
            rows.append(dict(model=key,early4=score(y[:,:4],v[key][:,:4],m[:,:4])['wape']))
    choice=min(rows,key=lambda z:z['early4'])['model'];checks['early_only_selection']=choice==s['candidate']
    for local,shared in [('dense_local64','dense_shared64'),('mixer_local32','mixer_shared32'),('mixer_local64','mixer_shared64')]:
        a=read(R/f'results/dev_{local}_seed{e.SEEDS[0]}_training_meta.json');b=read(R/f'results/dev_{shared}_seed{e.SEEDS[0]}_training_meta.json')
        checks['matched_control_size_'+shared]=a['parameters']==b['parameters'] and a['samples']==b['samples']
    weights=[]
    for a in e.ALPHAS:
        pp=np.clip((1-a)*v['prior'][:,4:]+a*v[choice][:,4:],0,1)
        weights.append(dict(alpha=a,wape=e.metrics(y[:,4:],pp,m[:,4:])['wape'],**e.previous.paired_se(y[:,4:],m[:,4:],v['prior'][:,4:],pp,d['cells'])))
    win=min(weights,key=lambda z:z['wape']);pick=min([z for z in weights if z['wape']<=win['wape']+win['selection_se']+1e-12],key=lambda z:z['alpha'])
    checks['late_only_weight']=weights==s['weights'] and pick==s['chosen']
    checks['dev_prior_bitwise']=bool(np.array_equal(v['prior'],np.load(B/'round11/results/dev_predictions.npz')['prior']))
    for seed in e.SEEDS:
        path=R/f"models/{s['candidate']}_dev_seed{seed}.pt"
        checks[f'dev_model_hash_{seed}']=sha(path)==s['selected_dev_model_sha256'][str(seed)]
        prediction,cp=network_forecast(path,d,e.DEV);ref=np.load(R/f"results/{s['candidate']}_dev_seed{seed}.npz")['prediction']
        difference=float(np.max(np.abs(prediction-ref)));differences[f'dev_seed{seed}']=difference
        checks[f'actual_dev_model_reconstruction_{seed}']=difference<1e-6 and cp['epoch']==s['epoch'] and cp['cutoff']==14
    t=np.load(R/'results/test_predictions.npz');ty,tm=e.truth(d,e.TEST);final=[];deploy=[]
    for seed in e.SEEDS:
        path=R/f'models/selected_final_seed{seed}.pt';h=sha(path)
        checks[f'final_model_hash_{seed}']=h==q['final_model_sha256'][str(seed)]==f['models'][str(seed)]
        pp,cp=network_forecast(path,d,e.TEST);final.append(pp)
        dp,_=network_forecast(path,d,[31]);deploy.append(dp)
        meta=cp['training_meta'];checks[f'final_train_boundary_{seed}']=cp['cutoff']==22 and cp['epoch']==s['epoch'] and meta['maximum_target_end']<=2112
    rebuilt=(final[0]+final[1])/2;difference=float(np.max(np.abs(rebuilt-t['candidate'])));differences['test_cpu_vs_training_device']=difference
    checks['actual_final_model_reconstruction']=difference<1e-6
    checks['test_prior_bitwise']=bool(np.array_equal(t['prior'],np.load(B/'round8b/results/test_predictions.npz')['selected']))
    checks['test_selected_frozen']=bool(np.array_equal(np.clip((1-s['chosen']['alpha'])*t['prior']+s['chosen']['alpha']*t['candidate'],0,1),t['selected']))
    for name in ['prior','candidate','selected']:
        for scope,ix in [('all57',np.ones(57,bool)),('a5',d['cells']=='a')]:
            actual=score(ty[ix],t[name][ix],tm[ix]);checks[f'actual_metric_{name}_{scope}']=all(abs(actual[k]-q['scores'][name][scope][k])<1e-12 for k in actual)
    checks['candidate_interval']=e.previous.bootstrap(ty,tm,t['candidate'],t['prior'])==q['candidate_day_bootstrap95_pp']
    checks['selected_interval']=e.previous.bootstrap(ty,tm,t['selected'],t['prior'])==q['selected_day_bootstrap95_pp']
    dep=np.load(R/'results/deployment_components.npz');new=pd.read_csv(R/'results/selected_deployment_origin31_1day.csv');old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv')
    difference=float(np.max(np.abs((deploy[0]+deploy[1])/2-dep['candidate'])));differences['deployment_cpu_vs_training_device']=difference
    checks['actual_deployment_model_reconstruction']=difference<1e-6
    checks['deployment_frozen_blend']=bool(np.array_equal(np.clip((1-s['chosen']['alpha'])*dep['prior']+s['chosen']['alpha']*dep['candidate'],0,1),dep['selected']))
    checks['deployment_csv_matches']=bool(np.max(np.abs(new.predicted_util.to_numpy()-dep['selected'].ravel()))<2e-15)
    checks['deployment_57x96']=len(new)==5472 and new.series.nunique()==57 and bool(new.groupby('series').size().eq(96).all())
    checks['deployment_keys_timestamps']=new[['series','cell','local_trace_time_PT']].equals(old[['series','cell','local_trace_time_PT']])
    checks['deployment_stored_checks']=read(R/'results/deployment_verification.json')['all_passed'] and all(read(R/'results/deployment_verification.json')['checks'].values())
    checks.update(e.causal(d,[672,1344,2016,2112,2880,2976]))
    base=e.inputs(d,[1344])[0];changed={k:v.copy() for k,v in d.items()};changed['y'][0,7:14,::8]+=.03
    altered=e.inputs(changed,[1344])[0]
    checks['leave_one_out_global_state']=bool(np.max(np.abs(base[0,:,4]-altered[0,:,4]))<1e-6)
    checks['leave_one_out_cell_state']=bool(np.max(np.abs(base[0,:,5]-altered[0,:,5]))<1e-6)
    checks['changed_other_history_reaches_shared_state']=bool(np.max(np.abs(base[1,:,4]-altered[1,:,4]))>1e-4)
    run=read(R/'results/guarded_pipeline_run.json');checks['guard_completion']=run['exit_code']==0 and not run['resource_pressure_stop']
    checks['honest_limits']=q['test_reused'] and q['development_reused'] and not q['independent_confirmation']
    checks={k:bool(v) for k,v in checks.items()};out=dict(checked_utc=e.utc(),check_count=len(checks),checks=checks,all_passed=all(checks.values()),
          network_reconstruction=dict(tolerance=1e-6,max_absolute_differences=differences,reason='CPU re-evaluation versus original deterministic training device; selection was not recomputed from CPU predictions'))
    print(json.dumps(out,ensure_ascii=False,indent=2))
    if not out['all_passed']:sys.exit(1)
if __name__=='__main__':main()
