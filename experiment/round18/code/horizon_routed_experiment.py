"""Known-horizon routing with genuinely native seven-day residual calibration."""
from pathlib import Path
import sys,os,json,hashlib,datetime,argparse,pickle,ast
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round18';Q=R/'results'
sys.path.insert(0,str(B/'code'));import common
src=B/'round17/code/multihorizon_residual_experiment.py';e={'__file__':str(src),'__name__':'immutable_round17_utilities'}
tree=ast.parse(src.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets))
exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(src),'exec'),e)
METRICS=e['METRICS'];score=e['score'];envelope=e['envelope'];features=e['features'];FOURIER=e['FOURIER']
CONFIGS=[dict(id='prior',kind='prior')]+[dict(id='legacy_'+c['id'],kind='legacy',legacy=c) for c in e['CONFIGS'] if c['id']!='prior']
for ridge in [1000,10000,100000]:
 for shape in [0.,1.]:
  for gate in [False,True]:CONFIGS.append(dict(id=f'zero_centered_a{ridge}_shape{int(shape)}_agree{int(gate)}',kind='zero_centered',ridge=ridge,shape=shape,gate=gate))
ALPHAS=[0.,.1,.25,.5,1.]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.with_suffix('.tmp');f.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');f.replace(p)
def z(file,key):
 with np.load(B/file,allow_pickle=False) as a:return a[key].astype('float64')
def prepare():
 for a in ['code','docs','models','results']:(R/a).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert read(R/'docs/protocol.json')['script_sha256']==sha(__file__);return
 files=['results/data.npz','round17/code/multihorizon_residual_experiment.py','round16/code/joint_local_experiment.py','round17/results/native_train7.npz','round17/results/native_train7_verification.json','round17/results/independent_verification.json','round17/models/dev_heads.pkl','round17/models/final_heads.pkl','round11/results/dev_predictions.npz','round8b/results/test_predictions.npz','round8b/results/component_deployment_origin31.npz','round8b/results/selected_deployment_origin31_1day.csv','round15/results/tf3_dev7.npz','round15/results/tf3_dev7_verification.json','results/dev7.npz','seven_day_completion_20261005/results/test7_predictions.npz','model_comparison_20261005/results/task_revision/summary.json','round15/results/test_predictions.npz','round16/results/test_predictions.npz','round3/results/tf3_dev.npz','round3/code/timesfm3_run.py','models/timesfm3-base/config.json','models/timesfm3-base/model.safetensors']
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),model_name='Known-horizon Chronos-2 TiRex-2 TimesFM3 routing with Ridge calibration and optional zero-centered Fourier residual gate',configs=CONFIGS,metrics=METRICS,alphas=ALPHAS,new_downloads=False,route='Known H=1 uses original round8b; known H=7 uses native TimesFM3 uni21 plus only DEV-selected seven-day residual. This is one routing procedure with two branches, not one identical neural network. Routing never inspects future truth.',training='Reuse audited native672 forecasts origins7..21. DEV fitting only target dates<14; final only target dates<22, inverse repeated-target-date weights. Existing full-intercept heads reused under same cutoffs; twelve new zero-intercept Ridge configurations fit only native seven-day residuals.',family_selection='DEV origin14 full7days, fixed candidate blend .25. Prefer nonprior candidates satisfying primary WAPE and daily WAPE<=prior+.001pp, RMSE/P95<=1.005prior, peak<=1.02prior, then most metric-best-control leads and smallest equally normalized primary loss. If none, take minimum-primary-loss nonprior as rejected diagnostic, not adoption.',weight_selection='DEV origin15 and aggregate origins14/15, both overlapping. For the single family, evaluate common seven-day strengths0/.1/.25/.5/1. Require primary no regression beyond .001pp separately at both origins, RMSE/P95/peak guards, primary improvement>=.001pp on at least one aggregate metric, and aggregate leading metric count>=prior. Select most leads then minimum primary loss, choose smaller weight within one paired-date SE of minimum among same-leading-count eligible strengths. Otherwise keep0. Dates overlap; this is stability screening, not independent confirmation.',frozen_diagnostic='Always freeze best nonprior family before TEST; diagnostic weight equals selected weight if positive else .25. Only this one diagnostic and frozen selected scheme are TEST-scored. Never reselect after TEST.',development7=[14,15],test24=list(range(22,31)),test7=[22,23,24],primary_metrics=['wape','daily_wape'],ranking='All ten metrics separately, combined with unchanged24 branch. TEST rank controls include original18 plus both frozen prior round15/16 diagnostic candidates. Best count is descriptive, not significance.',independent_confirmation=False,test_seen_before=True,overlapping_seven_windows=True,base7_previously_test_identified=True,resource_policy=dict(admission_physical_gib=3,admission_commit_gib=6,runtime_physical_gib=1.5,runtime_commit_gib=2),source_hashes={f:sha(B/f) for f in files}))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii');print('PROTOCOL FROZEN',sha(R/'docs/protocol.json'),flush=True)
def fit_zero(d,cutoff,alpha):
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 bank=z('round17/results/native_train7.npz','prediction');xs=[];ys=[];dates=[]
 for o in range(7,cutoff):
  for h in range(7):
   end=o+h
   if end>=cutoff:continue
   b=bank[:,o-7,h*96:(h+1)*96];x,scale,_=features(d,o,h,b);mask=d['valid'][:,end];keep=mask.sum(-1)>=48;res=d['y'][:,end]-b;mean=(res*mask).sum(-1)/mask.sum(-1).clip(1);target=[]
   for i in range(57):
    co=np.linalg.lstsq(FOURIER[:,mask[i]].T,res[i,mask[i]],rcond=None)[0] if keep[i] else np.zeros(7);peak=np.max(d['y'][i,end,mask[i]])-np.max(b[i,mask[i]]) if keep[i] else 0;target.append(np.r_[mean[i],co[1:],peak]/scale[i])
   xs.append(x[keep]);ys.append(np.stack(target)[keep]);dates.append(np.full(keep.sum(),end))
 dates=np.concatenate(dates);w=1/np.bincount(dates)[dates];model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,fit_intercept=False));model.fit(np.concatenate(xs),np.concatenate(ys),ridge__sample_weight=w*len(w)/w.sum());assert np.isfinite(model[-1].coef_).all();return model
def heads(d,cutoff):
 path=B/f'round17/models/{"dev" if cutoff==14 else "final"}_heads.pkl'
 return dict(legacy=pickle.load(path.open('rb'))[7],zero={a:fit_zero(d,cutoff,a) for a in [1000,10000,100000]})
def candidate(d,origins,base,cfg,models):
 if cfg['kind']=='prior':return base.copy()
 if cfg['kind']=='legacy':return e['candidate'](d,origins,base,cfg['legacy'],models['legacy'],7)
 out=[]
 for j,o in enumerate(origins):
  daily=[]
  for h in range(7):
   b=base[:,j,h*96:(h+1)*96];x,scale,_=features(d,o,h,b);a=np.clip(models['zero'][1000].predict(x),-.05,.05);c=np.clip(models['zero'][100000].predict(x),-.05,.05);q=np.clip(models['zero'][cfg['ridge']].predict(x),-.05,.05)
   gate=np.where(a*c>0,np.minimum(abs(a),abs(c))/np.maximum(np.maximum(abs(a),abs(c)),.002),0) if cfg['gate'] else np.ones_like(q);q=q*gate*scale[:,None];shape=b+cfg['shape']*(q[:,1:7]@FOURIER[1:]);mean=b.mean(-1)+q[:,0];daily.append(np.clip(shape+(mean-shape.mean(-1))[:,None],0,1))
  out.append(np.concatenate(daily,-1))
 return np.stack(out,1)
def primary(s,base):return .5*(s['wape']/base['wape']+s['daily_wape']/base['daily_wape'])
def guard(s,base):return all(s[k]<=base[k]+.001 for k in ['wape','daily_wape']) and all(s[k]<=base[k]*1.005 for k in ['rmse','p95_ape']) and s['peak_mape']<=base['peak_mape']*1.02
def leads(s,ev):return [k for k in METRICS if s[k]<=ev[k]+1e-10]
def paired_se(d,p,base):
 values=[]
 for h in range(7):
  a=score(d,[15+h],p[:,1:2,h*96:(h+1)*96],1);b=score(d,[15+h],base[:,1:2,h*96:(h+1)*96],1);values.append(primary(a,b)-1)
 return float(np.std(values,ddof=1)/np.sqrt(len(values)))
def develop(d):
 assert not (Q/'selection.json').exists(),'Do not alter frozen selection'
 hs=heads(d,14);pickle.dump(hs,(R/'models/dev_heads.pkl').open('wb'));base=z('round15/results/tf3_dev7.npz','prediction');refs=e['refs7'](d);s0=score(d,[14],base[:,:1],7);s1=score(d,[15],base[:,1:],7);sall=score(d,[14,15],base,7);ev0=envelope(d,[14],{k:v[:,:1] for k,v in refs.items()},7);eva=envelope(d,[14,15],refs,7);preds={};rows=[]
 for cfg in CONFIGS:
  cp=candidate(d,[14,15],base,cfg,hs);preds[cfg['id']]=cp;p=.75*base+.25*cp;sc=score(d,[14],p[:,:1],7);rows.append(dict(id=cfg['id'],scores=sc,primary_loss=primary(sc,s0),guard_passed=guard(sc,s0),leading_metrics=leads(sc,ev0),leading_count=len(leads(sc,ev0))));print('DEV7 FAMILY',cfg['id'],rows[-1]['primary_loss'],flush=True)
 choices=[a for a in rows if a['id']!='prior'];eligible=[a for a in choices if a['guard_passed']];best=min(eligible,key=lambda a:(-a['leading_count'],a['primary_loss'])) if eligible else min(choices,key=lambda a:a['primary_loss']);cfg=next(c for c in CONFIGS if c['id']==best['id']);cp=preds[cfg['id']];weights=[]
 for alpha in ALPHAS:
  p=(1-alpha)*base+alpha*cp;s14=score(d,[14],p[:,:1],7);s15=score(d,[15],p[:,1:],7);sc=score(d,[14,15],p,7);lead=leads(sc,eva);ok=alpha>0 and guard(s14,s0) and guard(s15,s1) and len(lead)>=len(leads(sall,eva)) and any(sc[k]<=sall[k]-.001 for k in ['wape','daily_wape']);weights.append(dict(alpha=alpha,scores14=s14,scores15=s15,scores=sc,leading_metrics=lead,leading_count=len(lead),primary_loss=primary(sc,sall),paired_date_se=paired_se(d,p,base),eligible=ok))
 eligible=[a for a in weights if a['eligible']]
 if eligible:
  count=max(a['leading_count'] for a in eligible);pool=[a for a in eligible if a['leading_count']==count];winner=min(pool,key=lambda a:a['primary_loss']);pick=min((a for a in pool if a['primary_loss']<=winner['primary_loss']+winner['paired_date_se']),key=lambda a:a['alpha'])
 else:pick=weights[0]
 np.savez_compressed(Q/'dev_predictions.npz',**preds);save(Q/'dev_summary.json',dict(configs=rows,weights=weights,baseline_scores=sall,baseline_leading_metrics=leads(sall,eva),metric_best_controls=eva,selected_family_early=best,dev_windows_overlap=True,testing_not_loaded=True))
 s=dict(frozen_utc=utc(),candidate=cfg,chosen_alpha=pick['alpha'],diagnostic_alpha=pick['alpha'] if pick['alpha']>0 else .25,development_supported=bool(eligible),protocol_sha256=sha(R/'docs/protocol.json'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),dev_heads_sha256=sha(R/'models/dev_heads.pkl'),independent_confirmation=False);save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();assert sha(R/'docs/protocol.json')==s['protocol_sha256'] and sha(Q/'dev_summary.json')==s['dev_summary_sha256'] and sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256']
 hs=heads(d,22);pickle.dump(hs,(R/'models/final_heads.pkl').open('wb'));base=z('seven_day_completion_20261005/results/test7_predictions.npz','tf3_uni21_7');raw=candidate(d,[22,23,24],base,s['candidate'],hs);diag=(1-s['diagnostic_alpha'])*base+s['diagnostic_alpha']*raw;selected=(1-s['chosen_alpha'])*base+s['chosen_alpha']*raw;p24=z('round8b/results/test_predictions.npz','selected');np.savez_compressed(Q/'test_predictions.npz',prior7=base,raw7=raw,diagnostic7=diag,selected7=selected,selected24=p24)
 old=read(B/'model_comparison_20261005/results/task_revision/summary.json');refs24={v['model']:z(v['source'],v['array']) for v in old['models']};refs7={v['model']:z(v['source'],v['array']) for v in old['seven_day']}
 for n in [15,16]:refs24[f'round{n}_candidate']=z(f'round{n}/results/test_predictions.npz','candidate24');refs7[f'round{n}_candidate']=z(f'round{n}/results/test_predictions.npz','candidate7')
 ev24=envelope(d,list(range(22,31)),refs24,1);ev7=envelope(d,[22,23,24],refs7,7);s24=score(d,list(range(22,31)),p24,1);s7={k:score(d,[22,23,24],p,7) for k,p in dict(prior=base,diagnostic=diag,selected=selected).items()};wins24=leads(s24,ev24);wins7=leads(s7['selected'],ev7);save(Q/'summary.json',dict(selection=s,scores24=s24,scores7=s7,leading_metrics24=wins24,leading_metrics7=wins7,leading_count=len(wins24)+len(wins7),control_best24=ev24,control_best7=ev7,control_count_each=20,multiple_metric_best_descriptive=len(wins24)+len(wins7)>=2,no_post_test_selection=True,independent_confirmation=False,fitted_final_label_cutoff=22));print('ACTUAL TEST',json.dumps(dict(scores24=s24,scores7=s7,leading24=wins24,leading7=wins7)),flush=True)
def deploy7(d):
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);sys.path.insert(0,str(B/'round3/code'));sys.path.insert(0,str(B/'round2/code'))
 from timesfm3 import TimesFM3Forecaster
 from timesfm3_run import forecast,CONFIGS as TFCONFIG
 assert torch.cuda.is_available()
 from timesfm3.torch.timesfm3_forecaster import ModelConfig
 from timesfm3.torch import configs as c
 rawcfg=read(B/'models/timesfm3-base/config.json');tc=dict(rawcfg['transformer_config']);tc['transformer']=c.TransformerConfig(**tc['transformer']);cfg=ModelConfig(checkpoint_path=str(B/'models/timesfm3-base/model.safetensors'),device='cuda',per_core_batch_size=1,input_patch_length=rawcfg['input_patch_len'],output_patch_length=rawcfg['output_patch_len'],quantiles=rawcfg['quantiles'],residual_block_config=c.ResidualBlockConfig(**rawcfg['residual_block_config']),transformer_config=c.StackedTransformersConfig(**tc),**{k:rawcfg[k] for k in ['input_transform','linear_detrending_threshold','use_frozen_running_stats','use_iterative_cpm_revin','use_linear_detrending','use_stitching','use_variate_attention','value_clip']})
 with torch.device('cuda'):model=TimesFM3Forecaster(config=cfg)
 control=np.clip(forecast(model,d,[14],TFCONFIG['tf3_uni_context2016'],h=96),0,1);delta=float(abs(control-z('round3/results/tf3_dev.npz','tf3_uni_context2016')[:,:1]).max());assert delta<1e-5
 base=np.clip(forecast(model,d,[31],TFCONFIG['tf3_uni_context2016'],h=672),0,1);s=read(Q/'selection.json');hs=pickle.load((R/'models/final_heads.pkl').open('rb'));raw=candidate(d,[31],base,s['candidate'],hs);a=s['chosen_alpha'];p=(1-a)*base+a*raw;np.savez_compressed(Q/'deployment7_origin31.npz',prior=base,raw=raw,selected=p)
 import pandas as pd
 stamps=pd.date_range('2019-06-01',periods=672,freq='15min',tz='UTC');rows=[]
 for i in range(57):
  for j,t in enumerate(stamps):rows.append(dict(pdu=str(d['names'][i]),cell=str(d['cells'][i]),timestamp=t.isoformat(),forecast_normalized_power=float(p[i,0,j]),horizon_days=7,method='Known-horizon Chronos-2 TiRex-2 TimesFM3 routing',seven_correction_alpha=a,historical_replay=True))
 pd.DataFrame(rows).to_csv(Q/'selected_deployment_origin31_7day.csv',index=False,encoding='utf-8-sig');save(Q/'deployment7_verification.json',dict(rows=len(rows),pdus=57,start=str(stamps[0]),end=str(stamps[-1]),finite_bounded=bool(np.isfinite(p).all() and p.min()>=0 and p.max()<=1),control96_max_abs_difference=delta,sha256=sha(Q/'selected_deployment_origin31_7day.csv'),selection_sha256=sha(Q/'selection.json'),no_june_truth=True))
 print('ACTUAL DEPLOYMENT7',len(rows),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(R/'docs/protocol.json')==(R/'docs/protocol.sha256').read_text() and read(R/'docs/protocol.json')['script_sha256']==sha(__file__);d=common.load()
 if a.stage=='develop':develop(d)
 elif a.stage=='test':test(d)
 elif a.stage=='deploy7':deploy7(d)
 else:raise ValueError(a.stage)
