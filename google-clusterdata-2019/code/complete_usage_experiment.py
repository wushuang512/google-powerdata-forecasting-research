"""Predeclared matched controls and honest selection for full cell-a JOB usage."""
import complete_usage_models as ex
import json,datetime,pickle,hashlib,copy
import numpy as np,pandas as pd,torch
from assess import comparisons
REGISTRY={f'{g}_ridge{a}':dict(kind='ridge',group=g,alpha=a) for g in ['control','history','forecast'] for a in [10000,100000,1000000]}
REGISTRY.update({f'{g}_alpha{a}':dict(kind=g,alpha=a) for g in ['monotone','physical_control'] for a in [10,100,1000]})
REGISTRY.update({name:dict(kind='fusion',cfg=dict(width=16,use_cpu=cpu,cross=cross,freq=freq,sparse=True)) for name,cpu,cross,freq in [('fusion_cpu',True,True,True),('fusion_control',False,True,True),('fusion_no_cross',True,False,True),('fusion_no_frequency',True,True,False)]})
SEEDS=[20261001,20261002];CHECKPOINTS=[4,10,20];ALPHAS=[0,.1,.25,.5,1]

def register():
    for sub in ['code','docs','models','results']:(ex.R/sub).mkdir(parents=True,exist_ok=True)
    p=ex.R/'docs/protocol.json'
    if p.exists():
        assert json.loads(p.read_text())['registry']==REGISTRY,'Do not silently change predeclared search'
        z=json.loads(p.read_text())
        if 'final_attribution' not in z:
            assert not (ex.R/'results/dev_predictions.npz').exists(),'Do not change evaluation after it begins'
            (ex.R/'docs/protocol_v1_before_attribution_clarification.json').write_bytes(p.read_bytes());z['clarified_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();z['final_attribution']='Always final-fit all three Ridge feature groups at the early-development best Ridge alpha and all four neural ablations at one common epoch. If overall winner is neural, its epoch is common; otherwise choose full-CPU neural epoch on early development. Retain matched tests without using test ranks to change selection.';ex.save_json(p,z)
        return
    ex.save_json(p,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),registry=REGISTRY,seeds=SEEDS,neural_checkpoints=CHECKPOINTS,blend_weights=ALPHAS,training_origins=ex.TRAIN,development_origins=ex.DEV,final_training_origins=ex.TRAIN+ex.DEV,test_origins=ex.TEST,prerequisite='All 12085 official cell-a Parquet files projected and audited; never train from the pilot as though complete',forecasting='Only completed JOB CPU/memory records in past bins. Future workload is forecast by frozen TimesFM3, never actual future CPU. Capacity is a historical scheduler-available feature.',selection='First four development dates choose family/checkpoint, two fixed neural seeds averaged. Last four dates choose blend by paired day one-standard-error preferring lower new weight. One cell has real CPU; cell-based SE would always reject a single-cell correction, so day units are declared before this experiment.',scope='Ridge candidates change cell-a PDUs only; all other domains exactly equal frozen round8b prior. Neural candidates may change all 57 domains. Report both 57-domain and 5-domain scores with matched controls.',initialization='Fusion encoder initializes from corresponding round5 seed and fitting cutoff. Power residual output reinitialized to zero around existing calibrated ensemble. Extra resource embedding initializes at zero.',constraints='Power quantiles in [0,1], CPU NCU quantiles >=0 and ordered. No CPU<=Power or CPU<=scheduler-capacity constraint.',independent_confirmation=False,limitations='Same May2019 test repeatedly inspected in past rounds; freezing this selection prevents new test-driven changes but cannot restore independence. Full export coverage does not imply complete physical telemetry.'))
    print('REGISTERED',p,flush=True)

def score(d,p,origins):
    y,m=ex.truth(d,origins);ix=d['cells']=='a';return dict(all57=ex.metrics(y,p,m),cell_a_5=ex.metrics(y[ix],p[ix],m[ix]))

def frozen_selection(d,p,base):
    y,m=ex.truth(d,ex.DEV);best=min(p,key=lambda k:ex.metrics(y[:,:4],p[k][:,:4],m[:,:4])['wape']);cand=[]
    for a in ALPHAS:
        z=(1-a)*base[:,4:]+a*p[best][:,4:];den=(y[:,4:]*m[:,4:]).sum(2).clip(1e-9);per=100*(np.abs(y[:,4:]-z)*m[:,4:]).sum(2)/den;ref=100*(np.abs(y[:,4:]-base[:,4:])*m[:,4:]).sum(2)/den;dif=(ref-per).mean(0);se=dif.std(ddof=1)/np.sqrt(4);cand.append(dict(alpha=a,wape=ex.metrics(y[:,4:],z,m[:,4:])['wape'],paired_day_se=float(se)))
    win=min(cand,key=lambda z:z['wape']);pick=min((z for z in cand if z['wape']<=win['wape']+win['paired_day_se']+1e-12),key=lambda z:z['alpha']);best_ridge=min((k for k in p if 'ridge' in k),key=lambda k:ex.metrics(y[:,:4],p[k][:,:4],m[:,:4])['wape']);best_cpu=min((k for k in p if k.startswith('fusion_cpu_')),key=lambda k:ex.metrics(y[:,:4],p[k][:,:4],m[:,:4])['wape']);s=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=best,chosen=pick,validation=cand,attribution=dict(ridge_alpha=REGISTRY[best_ridge]['alpha'],neural_epoch=int(best.rsplit('_e',1)[1]) if best.startswith('fusion') else int(best_cpu.rsplit('_e',1)[1])),protocol_sha256=hashlib.sha256((ex.R/'docs/protocol.json').read_bytes()).hexdigest());path=ex.R/'results/selection.json'
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==(ex.R/'results/selection.sha256').read_text();original=json.loads(path.read_text());assert original['protocol_sha256']==s['protocol_sha256'];ex.save_json(ex.R/'results/selection_replay_audit.json',dict(original=original,recomputed_development_only=s,action='Preserve original frozen choice after implementation repair or restart; never reselect with test knowledge'));s=original
    else:ex.save_json(path,s);(ex.R/'results/selection.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    print('FROZEN',s,flush=True);return s

def main():
    register();d,v=ex.inputs();wf=ex.forecasts(d,v);base=ex.previous('dev');p={};quant={};wp={};rows=[];orig=ex.DEV;bb={o:base[:,j] for j,o in enumerate(orig)}
    for name,cfg in REGISTRY.items():
        if cfg['kind']!='fusion':
            from complete_usage_light_stage import cached_development_ridge
            cached=cached_development_ridge(name,cfg)
            if cached is not None:
                p[name]=cached;print('REUSE_VERIFIED_DEVELOPMENT',name,flush=True)
            else:
                model,features=ex.fit_candidate(d,v,ex.TRAIN,cfg,wf);p[name]=ex.predict_candidate(model,d,v,orig,cfg,wf,base)
                with (ex.R/'models'/f'{name}_dev.pkl').open('wb') as f:pickle.dump(model,f)
                ex.save_json(ex.R/'docs'/f'{name}_features.json',features)
        else:
            seedq={};seedw={}
            for seed in SEEDS:
                def ck(net,epoch):
                    q,w,g=ex.neural_predict(net,d,v,orig,bb);seedq.setdefault(epoch,[]).append(q);seedw.setdefault(epoch,[]).append(w);torch.save(dict(state=net.state_dict(),config=cfg['cfg'],cutoff=14,epoch=epoch,seed=seed),ex.R/'models'/f'{name}_e{epoch}_dev_seed{seed}.pt')
                net,hist=ex.train_neural(d,v,ex.TRAIN,cfg['cfg'],seed,20,ck);ex.save_json(ex.R/'results'/f'{name}_dev_seed{seed}_history.json',hist);del net;torch.cuda.empty_cache()
            for ep in CHECKPOINTS:
                key=f'{name}_e{ep}';quant[key]=np.mean(seedq[ep],0);wp[key]=np.mean(seedw[ep],0);p[key]=quant[key][:,:,:,1]
        for key in ([name] if cfg['kind']!='fusion' else [f'{name}_e{ep}' for ep in CHECKPOINTS]):
            row=dict(model=key,**score(d,p[key],orig));rows.append(row);print('DEV',row,flush=True)
        np.savez_compressed(ex.R/'results/dev_predictions.npz',**p,prior=base);ex.save_json(ex.R/'results/dev_metrics.json',rows)
    s=frozen_selection(d,p,base);key=s['config'];name,ep=(key.rsplit('_e',1) if key.startswith('fusion') else (key,None));cfg=REGISTRY[name];test={};tquant={};twp={};testbase=ex.previous('test');tbb={o:testbase[:,j] for j,o in enumerate(ex.TEST)}
    # Attribution checks are fixed by the selected configuration, not test rankings.
    ridge_alpha=s['attribution']['ridge_alpha'];neural_epoch=s['attribution']['neural_epoch'];needed={f'{g}_ridge{ridge_alpha}' for g in ['control','history','forecast']};needed.update(['fusion_cpu','fusion_control','fusion_no_cross','fusion_no_frequency']);needed.update(k for k,c in REGISTRY.items() if c['kind'] in ['monotone','physical_control'])
    for n in sorted(needed):
        c=REGISTRY[n]
        if c['kind']!='fusion':
            model,_=ex.fit_candidate(d,v,ex.TRAIN+ex.DEV,c,wf);test[n]=ex.predict_candidate(model,d,v,ex.TEST,c,wf,testbase)
            with (ex.R/'models'/f'{n}_final.pkl').open('wb') as f:pickle.dump(model,f)
        else:
            qq=[];ww=[]
            for seed in SEEDS:
                net,hist=ex.train_neural(d,v,ex.TRAIN+ex.DEV,c['cfg'],seed,neural_epoch);q,w,g=ex.neural_predict(net,d,v,ex.TEST,tbb);qq.append(q);ww.append(w);torch.save(dict(state=net.state_dict(),config=c['cfg'],cutoff=22,epoch=neural_epoch,seed=seed),ex.R/'models'/f'{n}_e{neural_epoch}_final_seed{seed}.pt');ex.save_json(ex.R/'results'/f'{n}_final_seed{seed}_history.json',hist);del net;torch.cuda.empty_cache()
            test[f'{n}_e{neural_epoch}']=np.mean(qq,0)[:,:,:,1];tquant[f'{n}_e{neural_epoch}']=np.mean(qq,0);twp[f'{n}_e{neural_epoch}']=np.mean(ww,0)
    test['prior']=testbase;a=s['chosen']['alpha'];test['selected']=np.clip((1-a)*testbase+a*test[key],0,1);ty,tm=ex.truth(d,ex.TEST);scores={k:score(d,z,ex.TEST) for k,z in test.items()};np.savez_compressed(ex.R/'results/test_predictions.npz',**test);np.savez_compressed(ex.R/'results/test_quantiles.npz',**tquant);np.savez_compressed(ex.R/'results/test_cpu_quantiles.npz',**twp);np.savez_compressed(ex.R/'results/dev_quantiles.npz',**quant);np.savez_compressed(ex.R/'results/dev_cpu_quantiles.npz',**wp)
    checks={}
    for o in [14,22,30]:
        dd=copy.deepcopy(d);vv={k:x.copy() for k,x in v.items()};dd['y'][:,o:]=999;dd['prod'][:,o:]=999
        for k in vv:vv[k][:,o:]=999 if vv[k].dtype!=bool else ~vv[k][:,o:]
        b=ex.bank(['train','dev','test']);checks[f'causal_cpu_resource_features_o{o}']=bool(np.array_equal(ex.residual_features(d,v,o,b,'forecast',wf)[0],ex.residual_features(dd,vv,o,b,'forecast',wf)[0]));pb={o:ex.previous('dev')[:,0]};xx=ex.neural_inputs(d,v,[o],pb);zz=ex.neural_inputs(dd,vv,[o],pb);checks[f'causal_fusion_features_o{o}']=all(torch.equal(x,z) for x,z in zip(xx,zz))
    checks['selection_hash']=hashlib.sha256((ex.R/'results/selection.json').read_bytes()).hexdigest()==(ex.R/'results/selection.sha256').read_text();checks['power_finite_bounds']=all(np.isfinite(z).all() and z.min()>=0 and z.max()<=1 for z in test.values());checks['cpu_quantiles_nonnegative_ordered']=all(np.isfinite(z).all() and z.min()>=0 and (np.diff(z,axis=-1)>=-1e-7).all() for z in twp.values())
    dy,dm=ex.truth(d,ex.DEV);dsel=np.clip((1-a)*base+a*p[key],0,1);devlevel=np.stack([d['y'][:,o-7:o].mean((1,2)) for o in ex.DEV],1).clip(.05);testlevel=np.stack([d['y'][:,o-7:o].mean((1,2)) for o in ex.TEST],1).clip(.05);width=float(np.quantile((np.abs(dy-dsel)/devlevel[:,:,None])[dm],.8));lo=(test['selected']-width*testlevel[:,:,None]).clip(0,1);hi=(test['selected']+width*testlevel[:,:,None]).clip(0,1);interval=dict(nominal=.8,method='Development residual quantile of error divided by past7-day power mean; no test fitting',relative_halfwidth=width,test_coverage=float((((ty>=lo)&(ty<=hi))*tm).sum()/tm.sum()),mean_width=float(((hi-lo)*tm).sum()/tm.sum()));np.savez_compressed(ex.R/'results/selected_intervals_test.npz',lower=lo,upper=hi)
    checks['ridge_and_monotone_other52_exact_prior']=all(np.array_equal(z[d['cells']!='a'],testbase[d['cells']!='a']) for k,z in test.items() if not k.startswith('fusion') and k not in ['selected','prior']);checks['physical_slopes_nonnegative']=True
    for k,c in REGISTRY.items():
        if c['kind'] in ['monotone','physical_control']:
            with (ex.R/'models'/f'{k}_final.pkl').open('rb') as f:models=pickle.load(f)
            checks['physical_slopes_nonnegative'] &= all((m.coef_>=0).all() for m in models.values())
    aix=d['cells']=='a';wy=v['u'][aix,:, :,0][:,ex.TEST];wm=v['present'][aix][:,ex.TEST];cpu_predictions=dict(last_day=v['u'][aix,:,:,0][:,[o-1 for o in ex.TEST]],foundation=np.stack([wf[str(o)][aix,:,0] for o in ex.TEST],1),fusion=twp[f'fusion_cpu_e{neural_epoch}'][aix,:,:,1]);cpu_scores={k:ex.metrics(wy,z,wm) for k,z in cpu_predictions.items()};wq=twp[f'fusion_cpu_e{neural_epoch}'][aix];cpu_coverage=float((((wy>=wq[:,:,:,0])&(wy<=wq[:,:,:,2]))*wm).sum()/wm.sum())
    summary=dict(scores=scores,selection=s,final_matched_ridge_alpha=ridge_alpha,final_matched_neural_epoch=neural_epoch,comparison_vs_previous=comparisons(ty,tm,test['selected'],testbase,d['cells']),uncertainty=interval,cpu_scores=cpu_scores,cpu_raw_nominal80_coverage=cpu_coverage,checks=checks,all_passed=all(checks.values()),complete_cell_a_projection=True,complete_all_eight_cells=False,actual_cpu_used=True,independent_confirmation=False,large_improvement=False);ex.save_json(ex.R/'results/summary.json',summary);print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--register-only',action='store_true');a=ap.parse_args();register() if a.register_only else main()
