"""Registered wider search over direct all-domain forecasting with shared state."""
import argparse,datetime,gc,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];TOP=ROOT.parent;B=TOP/'experiment';R=B/'round14'
sys.path.insert(0,str(B/'code'))
from common import load,truth,metrics
import recency_robust_experiment as previous
CONFIGS={
    'dense_local64':dict(family='dense',width=64,shared=False),
    'dense_shared64':dict(family='dense',width=64,shared=True),
    'mixer_local32':dict(family='mixer',width=32,shared=False),
    'mixer_shared32':dict(family='mixer',width=32,shared=True),
    'mixer_local64':dict(family='mixer',width=64,shared=False),
    'mixer_shared64':dict(family='mixer',width=64,shared=True),
}
SEEDS=[20261004,20261005];CHECKPOINTS=[20,60,120];ALPHAS=[0,.25,.5,.75,1]
DEV=list(range(14,22));TEST=list(range(22,31))
FIXED=dict(history_points=672,horizon_points=96,patch_points=4,training_stride_points=4,
           batch_size=512,learning_rate=.0005,weight_decay=.1,dropout=.1,
           main_loss='per-window valid power WAPE',daily_loss_weight=.1,peak_loss_weight=.02,bounds_loss_weight=.01,
           max_gradient_norm=1,cpu_threads=2,deterministic=True,amp=False,foundation_models_loaded=False,
           no_residual_cap=True,no_teacher_anchor_loss=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def register():
    for sub in ['models','docs','results']:(R/sub).mkdir(parents=True,exist_ok=True)
    files=[B/'results/data.npz',B/'code/common.py',B/'round11/results/dev_predictions.npz',
           B/'round8b/results/test_predictions.npz',B/'round8b/results/selected_deployment_origin31_1day.csv',
           ROOT/'code/direct_networks.py',ROOT/'code/recency_robust_experiment.py']
    p=dict(created_utc=utc(),method='full daily-curve direct dense and temporal-mixer models with leave-one-out shared historical PDU state',
           configs=CONFIGS,seeds=SEEDS,checkpoints=CHECKPOINTS,fixed_parameters=FIXED,
           development_training_cutoff_completed_days=14,final_training_cutoff_completed_days=22,
           family_selection_origins=DEV[:4],weight_selection_origins=DEV[4:],test_origins=TEST,
           windows='Hourly shifted rolling windows, all 96 target slots must end at or before training cutoff; windows overlap and are not independent',
           input='7-day own power/valid/production-proxy/mask, past-only normalized other-PDU and same-cell leave-one-out states, known origin calendar, site and cell IDs',
           preprocessing='Own window valid mean/std with std>=0.01; proxy uses only own window valid mean/std; shared states use only the same completed history, exclude target PDU',
           architecture='Task-specific TiDE-inspired dense residual encoder and TSMixer-inspired temporal/channel mixing; not official reproductions',
           sources=['https://arxiv.org/abs/2304.08424','https://arxiv.org/abs/2303.06053'],
           selection='Equal two-seed average for 18 architecture/checkpoint candidates; first4 DEV choose by all57 WAPE; later4 one-SE max paired date/cell SE preferring smaller alpha; freeze before test',
           blending='clip((1-alpha)*old_best+alpha*direct_candidate,0,1); alpha1 full replacement; no +/-1% correction restriction',
           constraints='Inference power bounds [0,1]; training bounds penalty; neither is an energy-conservation model',
           large_gain_rule='>=5% relative WAPE reduction and positive paired-date bootstrap lower bound, exploratory only',
           development_reused=True,test_reused=True,independent_confirmation=False,new_downloads=False,
           script_sha256=sha(__file__),file_fingerprints={str(p.relative_to(TOP)):sha(p) for p in files})
    path=R/'docs/protocol.json'
    if path.exists():
        old=read(path)
        for k,v in p.items():
            if k!='created_utc':assert old[k]==v,f'Protocol drift {k}'
    else:save(path,p)
    return read(path)
def fingerprint():return sha(R/'docs/protocol.json')
def inputs(d,ends):
    yf=d['y'].reshape(57,-1);vf=d['valid'].reshape(57,-1)
    pf=d['prod'].reshape(57,-1);pv=d['pvalid'].reshape(57,-1)
    cells=d['cells'];cellnames=sorted(set(cells));cellids=np.array([cellnames.index(c) for c in cells])
    rows=[[] for _ in range(7)]
    for e in ends:
        assert e>=672 and e<=yf.shape[1]
        y=yf[:,e-672:e];m=vf[:,e-672:e];p=pf[:,e-672:e];pm=pv[:,e-672:e]
        count=m.sum(1,keepdims=True).clip(1);mean=(y*m).sum(1,keepdims=True)/count
        std=np.sqrt(((y-mean)**2*m).sum(1,keepdims=True)/count).clip(.01)
        z=(y-mean)/std*m
        pc=pm.sum(1,keepdims=True).clip(1);pmean=(p*pm).sum(1,keepdims=True)/pc
        psd=np.sqrt(((p-pmean)**2*pm).sum(1,keepdims=True)/pc).clip(.01)
        pz=(p-pmean)/psd*pm
        total=z.sum(0,keepdims=True);nc=m.sum(0,keepdims=True)
        global_state=(total-z)/np.maximum(nc-m,1)
        cell_state=np.zeros_like(z)
        for c in cellnames:
            ix=cells==c;cell_state[ix]=(z[ix].sum(0,keepdims=True)-z[ix])/np.maximum(m[ix].sum(0,keepdims=True)-m[ix],1)
        xx=np.stack([z,pz,m.astype(float),pm.astype(float),global_state,cell_state],-1)
        xx=xx.reshape(57,168,4,6).mean(2).astype('float32')
        phase=np.array([np.sin(2*np.pi*e/96),np.cos(2*np.pi*e/96),np.sin(2*np.pi*(e/96+2)/7),np.cos(2*np.pi*(e/96+2)/7)],dtype='float32')
        calendar=np.tile(phase,(57,1));last=np.where(m[:,-96:],y[:,-96:],mean).astype('float32')
        for target,item in zip(rows,[xx,calendar,np.arange(57,dtype='int64'),cellids,last,std.astype('float32'),np.repeat(e,57)]):target.append(item)
    out=tuple(np.concatenate(a) for a in rows)
    assert all(np.isfinite(a).all() for a in out)
    return out
def causal(d,ends):
    checks={}
    for e in ends:
        altered={k:v.copy() for k,v in d.items()}
        for key in ['y','prod','valid','pvalid']:
            flat=altered[key].reshape(57,-1);flat[:,e:]=.123 if key in ['y','prod'] else False
        checks[f'future_perturbation_point{e}']=all(np.array_equal(a,b) for a,b in zip(inputs(d,[e]),inputs(altered,[e])))
    assert all(checks.values());return checks
def training_data(d,cutoff):
    # Strictly truncate the source before forming features or target windows.
    dd={k:(v[:,:cutoff].copy() if k in ['y','prod','valid','pvalid'] else v) for k,v in d.items()}
    ends=np.arange(672,cutoff*96-96+1,4);a=inputs(dd,ends)
    y=dd['y'].reshape(57,-1);m=dd['valid'].reshape(57,-1)
    target=np.concatenate([y[:,e:e+96] for e in ends]).astype('float32')
    mask=np.concatenate([m[:,e:e+96] for e in ends]).astype('float32')
    keep=mask.sum(1)>0
    arrays=tuple(v[keep] for v in a[:6])+(target[keep],mask[keep])
    meta=dict(cutoff_days=cutoff,minimum_history_end=int(ends.min()),maximum_history_end=int(ends.max()),maximum_target_end=int(ends.max()+96),
              hourly_windows=len(ends),samples=int(keep.sum()),overlapping_windows=True,not_independent_samples=True)
    assert meta['maximum_target_end']<=cutoff*96
    return arrays,meta
TORCH=None;DEVICE=None
def setup():
    global TORCH,DEVICE
    if TORCH is not None:return TORCH
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    TORCH=torch;torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    save(R/'results/runtime.json',dict(device=DEVICE,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if DEVICE=='cuda' else None,cpu_threads=2,foundation_models_loaded=False))
    return torch
def atomic_torch(path,obj):
    torch=setup();path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);tmp.replace(path)
def newnet(name):
    from direct_networks import DirectNetwork
    return DirectNetwork(**CONFIGS[name]).to(DEVICE)
def predict(net,d,origins):
    torch=setup();arrays=inputs(d,np.array(origins)*96);out=[];net.eval()
    with torch.inference_mode():
        for start in range(0,len(arrays[0]),512):
            part=[torch.as_tensor(v[start:start+512],device=DEVICE) for v in arrays[:6]]
            p=net(*part).clamp(0,1);out.append(p.cpu().numpy())
    return np.concatenate(out).reshape(len(origins),57,96).transpose(1,0,2).astype('float64')
def train(d,cutoff,name,seed,epochs,prefix,callback=None):
    torch=setup();torch.manual_seed(seed);np.random.seed(seed)
    if DEVICE=='cuda':torch.cuda.manual_seed_all(seed)
    net=newnet(name);opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.1)
    data,meta=training_data(d,cutoff);data=tuple(torch.as_tensor(a,device=DEVICE) for a in data)
    history=[];first=1;cache=R/f'models/resume_{prefix}_{name}_seed{seed}.pt'
    if cache.exists():
        state=torch.load(cache,map_location=DEVICE,weights_only=False)
        assert state['protocol_sha256']==fingerprint() and state['name']==name and state['seed']==seed and state['cutoff']==cutoff
        net.load_state_dict(state['state']);opt.load_state_dict(state['optimizer']);history=state['history'];first=state['epoch']+1
        torch.set_rng_state(state['rng'].cpu())
        if DEVICE=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        print('RESUME',prefix,name,seed,first,flush=True)
    meta.update(parameters=sum(p.numel() for p in net.parameters()),seed=seed,architecture=name,device=DEVICE,protocol_sha256=fingerprint())
    save(R/f'results/{prefix}_{name}_seed{seed}_training_meta.json',meta)
    for ep in range(first,epochs+1):
        net.train();losses=[]
        for ix in torch.randperm(len(data[0]),device=DEVICE).split(512):
            x,cal,site,cell,last,std,y,m=(v[ix] for v in data)
            raw=net(x,cal,site,cell,last,std);den=(y.abs()*m).sum(1).clamp_min(1e-5);count=m.sum(1).clamp_min(1)
            point=((raw-y).abs()*m).sum(1)/den
            daily=((raw*m).sum(1)-(y*m).sum(1)).abs()/den
            actual_max=y.masked_fill(m==0,-10).max(1).values;pred_max=raw.masked_fill(m==0,-10).max(1).values
            peak=(actual_max-pred_max).abs()/(den/count)
            bounds=(torch.relu(-raw)+torch.relu(raw-1)).mean(1)
            loss=(point+.1*daily+.02*peak+.01*bounds).mean()
            assert torch.isfinite(loss),'Nonfinite model loss'
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=ep,loss=float(np.mean(losses))))
        # Complete model and prediction checkpoint before advancing the resume
        # position. If interrupted here, restart from the previous committed
        # training epoch rather than skipping a missing development checkpoint.
        if callback and ep in CHECKPOINTS:
            path=R/f'models/{name}_e{ep}_{prefix}_seed{seed}.pt'
            atomic_torch(path,dict(state=net.state_dict(),epoch=ep,name=name,seed=seed,cutoff=cutoff,protocol_sha256=fingerprint()))
            callback(net,ep)
        if ep%10==0 or ep==epochs:
            checkpoint=dict(state=net.state_dict(),optimizer=opt.state_dict(),epoch=ep,history=history,name=name,seed=seed,cutoff=cutoff,
                            protocol_sha256=fingerprint(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if DEVICE=='cuda' else [])
            atomic_torch(cache,checkpoint);save(R/f'results/{prefix}_{name}_seed{seed}_history.json',history)
            print('TRAIN',prefix,name,seed,ep,history[-1]['loss'],flush=True)
    return net,meta
def dev():
    register()
    if (R/'results/selection.json').exists():
        assert sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
        print('Existing frozen selection retained',flush=True);return
    d=load();d={k:(v[:,:22].copy() if k in ['y','prod','valid','pvalid'] else v) for k,v in d.items()}
    checks=causal(d,[672,1344,2016]);save(R/'results/causal_dev.json',checks)
    for name in CONFIGS:
        for seed in SEEDS:
            paths=[R/f'results/{name}_e{ep}_dev_seed{seed}.npz' for ep in CHECKPOINTS]
            if all(p.exists() for p in paths):print('Cached complete DEV seed',name,seed,flush=True);continue
            def callback(net,ep):
                out=predict(net,d,DEV);np.savez_compressed(R/f'results/{name}_e{ep}_dev_seed{seed}.npz',prediction=out,protocol_sha256=fingerprint())
                print('DEV_CHECKPOINT',name,seed,ep,flush=True)
            net,meta=train(d,14,name,seed,max(CHECKPOINTS),'dev',callback)
            # A resource stop after writing resume but before prediction must not
            # lose a checkpoint; reconstruct saved development checkpoints.
            for ep,path in zip(CHECKPOINTS,paths):
                if not path.exists():
                    cp=setup().load(R/f'models/{name}_e{ep}_dev_seed{seed}.pt',map_location=DEVICE,weights_only=False)
                    assert cp['protocol_sha256']==fingerprint();net.load_state_dict(cp['state']);callback(net,ep)
            del net;gc.collect();setup().cuda.empty_cache()
    prior=np.load(B/'round11/results/dev_predictions.npz')['prior'];y,m=truth(d,DEV);preds={};rows=[]
    for name in CONFIGS:
        for ep in CHECKPOINTS:
            key=f'{name}_e{ep}';parts=[]
            for seed in SEEDS:
                with np.load(R/f'results/{key}_dev_seed{seed}.npz') as z:
                    assert str(z['protocol_sha256'])==fingerprint();parts.append(z['prediction'])
            pp=(parts[0]+parts[1])/2;preds[key]=pp
            rows.append(dict(model=key,early4=metrics(y[:,:4],pp[:,:4],m[:,:4]),late4=metrics(y[:,4:],pp[:,4:],m[:,4:]),all8=metrics(y,pp,m)))
    key=min(preds,key=lambda k:metrics(y[:,:4],preds[k][:,:4],m[:,:4])['wape'])
    weights=[]
    for a in ALPHAS:
        p=np.clip((1-a)*prior[:,4:]+a*preds[key][:,4:],0,1)
        weights.append(dict(alpha=a,wape=metrics(y[:,4:],p,m[:,4:])['wape'],**previous.paired_se(y[:,4:],m[:,4:],prior[:,4:],p,d['cells'])))
    win=min(weights,key=lambda x:x['wape']);chosen=min([x for x in weights if x['wape']<=win['wape']+win['selection_se']+1e-12],key=lambda x:x['alpha'])
    selected=np.clip((1-chosen['alpha'])*prior+chosen['alpha']*preds[key],0,1)
    level=np.stack([d['y'][:,o-7:o].mean((1,2)).clip(.05) for o in DEV],1)[:,:,None]
    interval=float(np.quantile((np.abs(y-selected)/level)[m],.8,method='higher'))
    np.savez_compressed(R/'results/dev_predictions.npz',prior=prior,selected=selected,**preds)
    save(R/'results/dev_summary.json',dict(candidates=rows,prior=metrics(y,prior,m),prior_early4=metrics(y[:,:4],prior[:,:4],m[:,:4]),prior_late4=metrics(y[:,4:],prior[:,4:],m[:,4:]),causal_checks=checks))
    name,ep=key.rsplit('_e',1)
    s=dict(frozen_utc=utc(),candidate=key,architecture=name,epoch=int(ep),seeds=SEEDS,chosen=chosen,weights=weights,
           protocol_sha256=fingerprint(),dev_predictions_sha256=sha(R/'results/dev_predictions.npz'),dev_summary_sha256=sha(R/'results/dev_summary.json'),
           selected_dev_model_sha256={str(seed):sha(R/f'models/{key}_dev_seed{seed}.pt') for seed in SEEDS},
           empirical80_halfwidth_relative_to_past_mean=interval,independent_confirmation=False,development_reused=True,test_reused=True)
    save(R/'results/selection.json',s);(R/'results/selection.sha256').write_text(sha(R/'results/selection.json'),encoding='utf-8')
    print('FROZEN',json.dumps(s),flush=True)
def frozen():
    p=register();s=read(R/'results/selection.json')
    assert sha(R/'results/selection.json')==(R/'results/selection.sha256').read_text().strip()
    assert s['protocol_sha256']==fingerprint()
    for k,v in p['file_fingerprints'].items():assert sha(TOP/k)==v,k
    for seed,h in s['selected_dev_model_sha256'].items():assert sha(R/f"models/{s['candidate']}_dev_seed{seed}.pt")==h
    return s
def final_models(d,s):
    parts=[];metadata=[];models={}
    for seed in SEEDS:
        path=R/f'models/selected_final_seed{seed}.pt'
        if path.exists():
            setup();cp=TORCH.load(path,map_location=DEVICE,weights_only=False);assert cp['protocol_sha256']==fingerprint() and cp['epoch']==s['epoch'] and cp['name']==s['architecture']
            net=newnet(s['architecture']);net.load_state_dict(cp['state']);meta=cp['training_meta']
        else:
            net,meta=train(d,22,s['architecture'],seed,s['epoch'],'final')
            atomic_torch(path,dict(state=net.state_dict(),protocol_sha256=fingerprint(),epoch=s['epoch'],name=s['architecture'],seed=seed,cutoff=22,training_meta=meta))
        models[str(seed)]=sha(path);metadata.append(meta);del net;gc.collect();setup().cuda.empty_cache()
    save(R/'results/final_model_frozen.json',dict(at_utc=utc(),models=models,selection_sha256=sha(R/'results/selection.json'),training=metadata))
    return models,metadata
def candidate(d,s,origins):
    parts=[];setup()
    for seed in SEEDS:
        cp=TORCH.load(R/f'models/selected_final_seed{seed}.pt',map_location=DEVICE,weights_only=False)
        assert cp['protocol_sha256']==fingerprint() and cp['name']==s['architecture'] and cp['epoch']==s['epoch']
        net=newnet(s['architecture']);net.load_state_dict(cp['state']);parts.append(predict(net,d,origins));del net;gc.collect();TORCH.cuda.empty_cache()
    return (parts[0]+parts[1])/2
def test():
    s=frozen()
    if (R/'results/summary.json').exists():print('Existing frozen TEST retained',flush=True);return
    d=load();models,meta=final_models(d,s)
    direct=candidate(d,s,TEST);prior=np.load(B/'round8b/results/test_predictions.npz')['selected']
    chosen=np.clip((1-s['chosen']['alpha'])*prior+s['chosen']['alpha']*direct,0,1)
    y,m=truth(d,TEST);ps=dict(prior=prior,candidate=direct,selected=chosen);a5=d['cells']=='a'
    scores={k:dict(all57=metrics(y,p,m),a5=metrics(y[a5],p[a5],m[a5])) for k,p in ps.items()}
    np.savez_compressed(R/'results/test_predictions.npz',**ps)
    byday=[]
    for j,o in enumerate(TEST):
        for k,p in ps.items():byday.append(dict(origin=o,target_date=f'2019-05-{o+1:02d}',model=k,**metrics(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1])))
    pd.DataFrame(byday).to_csv(R/'results/per_day_metrics.csv',index=False)
    pd.DataFrame(dict(domain=d['names'],cell=d['cells'],**{k:previous.domain_wape(y,p,m) for k,p in ps.items()})).to_csv(R/'results/per_domain_metrics.csv',index=False)
    ci=previous.bootstrap(y,m,chosen,prior);gain=scores['prior']['all57']['wape']-scores['selected']['all57']['wape'];rel=100*gain/scores['prior']['all57']['wape']
    q=dict(completed_utc=utc(),selection=s,scores=scores,selected_gain_pp=gain,selected_relative_reduction_percent=rel,
           selected_day_bootstrap95_pp=ci,candidate_day_bootstrap95_pp=previous.bootstrap(y,m,direct,prior),large_gain_rule_met=bool(rel>=5 and ci[0]>0),
           final_model_sha256=models,final_training=meta,causal_checks=causal(d,[22*96,30*96]),independent_confirmation=False,development_reused=True,test_reused=True)
    save(R/'results/summary.json',q);print('TEST',json.dumps(scores),flush=True)
def deploy():
    s=frozen();summary=read(R/'results/summary.json');d=load()
    for seed,h in summary['final_model_sha256'].items():assert sha(R/f'models/selected_final_seed{seed}.pt')==h
    direct=candidate(d,s,[31]);old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv')
    prior=old.predicted_util.to_numpy().reshape(57,1,96);selected=np.clip((1-s['chosen']['alpha'])*prior+s['chosen']['alpha']*direct,0,1)
    old['round8b_prior_util']=prior.ravel();old['direct_shared_candidate_util']=direct.ravel();old['predicted_util']=selected.ravel()
    level=d['y'][:,-7:].mean((1,2)).clip(.05);half=s['empirical80_halfwidth_relative_to_past_mean']*np.repeat(level,96)
    old['lower80_empirical']=np.maximum(selected.ravel()-half,0);old['upper80_empirical']=np.minimum(selected.ravel()+half,1)
    old.to_csv(R/'results/selected_deployment_origin31_1day.csv',index=False)
    np.savez_compressed(R/'results/deployment_components.npz',prior=prior,candidate=direct,selected=selected)
    checks=dict(rows5472=len(old)==5472,series57x96=old.series.nunique()==57 and bool(old.groupby('series').size().eq(96).all()),
                finite=bool(np.isfinite(old.select_dtypes('number')).all().all()),bounds=bool(old.predicted_util.between(0,1).all()),
                intervals_ordered=bool(((old.lower80_empirical<=old.predicted_util)&(old.predicted_util<=old.upper80_empirical)).all()),
                timestamps_june1=bool(pd.to_datetime(old.local_trace_time_PT).min()==pd.Timestamp('2019-06-01') and pd.to_datetime(old.local_trace_time_PT).max()==pd.Timestamp('2019-06-01 23:45')),
                alpha0_exactly_prior=bool(s['chosen']['alpha']!=0 or np.array_equal(prior,selected)),causal_origin31=causal(d,[31*96])[f'future_perturbation_point{31*96}'])
    assert all(checks.values());save(R/'results/deployment_verification.json',dict(checks=checks,all_passed=True,historical_inference_not_live=True,independent_interval_guarantee=False))
    print('DEPLOY',checks,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['register','preflight','dev','test','deploy','all'],default='all');a=ap.parse_args()
    if a.phase=='register':register()
    if a.phase=='preflight':
        register();d=load();arrays,meta=training_data(d,14);checks=causal(d,[672,1344,21*96]);save(R/'results/preflight.json',dict(all_passed=True,training=meta,causal_checks=checks,feature_shape=list(arrays[0].shape)));print('Causal preflight',meta,flush=True)
    if a.phase in ['dev','all']:dev()
    if a.phase in ['test','all']:test()
    if a.phase in ['deploy','all']:deploy()
if __name__=='__main__':main()
