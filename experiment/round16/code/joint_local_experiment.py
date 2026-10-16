"""Causal daily-mean/peak heads with a bounded daily curve projection."""
from pathlib import Path
import sys,os,json,hashlib,datetime,argparse,pickle
import numpy as np,pandas as pd
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round16';Q=R/'results'
sys.path.insert(0,str(B/'code'));import common
METRICS=['wape','mae','rmse','daily_wape','peak_mape','p95_ape','peak_time_mae_minutes','absolute_bias_percent','cella5_wape','other52_wape']
CONFIGS=[dict(id='prior',mean=0.,peak=0.,alpha=100,peak_kind='history',shape=0.)]
for peak in [.25,.5,1.]:CONFIGS.append(dict(id=f'history_peak{peak}',mean=0.,peak=peak,alpha=100,peak_kind='history',shape=0.))
for alpha in [10,100]:
 for mean,peak in [(.25,.25),(.25,.5),(.5,.5)]:CONFIGS.append(dict(id=f'learned_a{alpha}_mean{mean}_peak{peak}',mean=mean,peak=peak,alpha=alpha,peak_kind='learned',shape=0.))
CONFIGS += [dict(id='learned_history_shape25',mean=.25,peak=.5,alpha=100,peak_kind='learned',shape=.25),dict(id='history_peak_shape25',mean=0.,peak=.5,alpha=100,peak_kind='history',shape=.25)]
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
 if (R/'docs/protocol.json').exists():
  p=read(R/'docs/protocol.json');assert p['script_sha256']==sha(__file__);return
 files=['results/data.npz','round11/results/dev_predictions.npz','round8b/results/selection.json','round8b/results/test_predictions.npz','round3/results/tf3_dev.npz','round3/results/tf3_train.npz','round2/results/development_selected.npz','results/dev7.npz','seven_day_completion_20261005/results/test7_predictions.npz','model_comparison_20261005/results/task_revision/summary.json','models/timesfm3-base/config.json','models/timesfm3-base/model.safetensors','round15/results/tf3_dev7.npz','round15/results/tf3_dev7_verification.json','round15/docs/protocol.json']
 save(R/'docs/protocol.json',dict(created_utc=utc(),model_name='TimesFM3 Chronos2 TiRex2 with causal daily-mean/peak heads and bounded curve projection',script_sha256=sha(__file__),configs=CONFIGS,metrics=METRICS,training_cutoff_dev=14,training_cutoff_final=22,training_origins='7..cutoff-1, only target days < cutoff; weight duplicate target dates inversely',development24=list(range(14,22)),development7=[14,15],test24=list(range(22,31)),test7=[22,23,24],selection='First four DEV24 dates choose one family by minimax ratio against each metric best retained control; later four DEV24 and both overlapping DEV7 windows choose common shrinkage 0/.25/.5/1. Adopt only if every registered metric at both horizons is <= best control. Freeze selection before TEST score. No post-test changes.',base24='Canonical original round8b development prior (trained labels before May15) and frozen final test prior',base7='Original zero-shot TimesFM3 21day context, same pretraining weights; reuse independently checked round15 DEV672 cache; identical origins and original base weights. This backbone was identified from already viewed TEST, so no independent confirmation claim.',targets='One-day and seven-day curve; daily means derived from same curves',projection='Minimal squared change to each daily curve with one causally chosen maximum pinned at target peak, other points projected by a shared offset into [0,target_peak]; bisection enforces feasible full96 daily mean. It may alter peak timing after capping, which is evaluated. Shape consistency, not energy conservation.',independent_confirmation=False,test_seen_before=True,overlapping_seven_windows=True,new_downloads=False,resource_policy=dict(admission_physical_gib=3,admission_commit_gib=6,runtime_physical_gib=1.5,runtime_commit_gib=2),source_hashes={f:sha(B/f) for f in files}))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii')
 print('PROTOCOL FROZEN',sha(R/'docs/protocol.json'),flush=True)
def features(d,o,h):
 a=d['y'][:,o-7:o];m=d['valid'][:,o-7:o];count=m.sum(-1).clip(1);mu=(a*m).sum(-1)/count;pk=np.where(m,a,-np.inf).max(-1);pk=np.where(np.isfinite(pk),pk,mu)
 lo=np.where(m,a,np.inf).min(-1);lo=np.where(np.isfinite(lo),lo,mu);std=np.sqrt((((a-mu[:,:,None])**2)*m).sum(-1)/count)
 scale=mu.mean(-1).clip(.05);v=[]
 for k in [1,3,7]:
  for stat in [mu,pk,lo,std]:v.append(stat[:,-k:].mean(-1)/scale)
 v.extend([(mu[:,-1]-mu[:,-3:].mean(-1))/scale,(pk[:,-1]-pk[:,-3:].mean(-1))/scale,(mu[:,-1]-mu[:,0])/scale,(pk[:,-1]-pk[:,0])/scale])
 day=(2+o+h)%7;v += [np.full(57,np.sin(2*np.pi*day/7)),np.full(57,np.cos(2*np.pi*day/7)),np.full(57,h/6),count.mean(-1)/96]
 v += [(d['cells']==c).astype(float) for c in sorted(set(d['cells']))]
 return np.stack(v,-1),scale,np.stack([mu[:,-1],pk[:,-1]],-1)
def fit_heads(d,cutoff,alpha):
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 xs=[];ys=[];weights=[];ends=[]
 for o in range(7,cutoff):
  for h in range(7):
   e=o+h
   if e>=cutoff:continue
   x,scale,anchor=features(d,o,h);m=d['valid'][:,e];cnt=m.sum(-1);mu=(d['y'][:,e]*m).sum(-1)/cnt.clip(1);pk=np.where(m,d['y'][:,e],-np.inf).max(-1);keep=cnt>0
   xs.append(x[keep]);ys.append((np.stack([mu,pk],-1)-anchor)[keep]/scale[keep,None]);ends.append(np.full(keep.sum(),e))
 dates=np.concatenate(ends);weight=1/np.bincount(dates)[dates]
 model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(np.concatenate(xs),np.concatenate(ys),ridge__sample_weight=weight*len(dates)/weight.sum())
 assert np.isfinite(model[-1].coef_).all();return model
def project(shape,mean,peak,history):
 mean=np.clip(mean,0,1);peak=np.clip(np.maximum(peak,mean),0,1);mean=np.maximum(mean,peak/96)
 rows=np.arange(len(mean));index=np.argmax(shape,-1);flat=np.ptp(shape,axis=-1)<1e-12;index[flat]=np.argmax(history[flat],-1)
 low=np.full(len(mean),-2.);high=np.full(len(mean),2.)
 for _ in range(55):
  mid=(low+high)/2;v=np.clip(shape+mid[:,None],0,peak[:,None]);v[rows,index]=peak;bad=v.mean(-1)>mean
  high=np.where(bad,mid,high);low=np.where(bad,low,mid)
 out=np.clip(shape+(.5*(low+high))[:,None],0,peak[:,None]);out[rows,index]=peak
 assert abs(out.mean(-1)-mean).max()<1e-9 and abs(out.max(-1)-peak).max()<1e-9
 return out
def candidate(d,origins,base,cfg,model):
 if cfg['id']=='prior':return base.copy()
 out=[];H=base.shape[-1]//96
 for j,o in enumerate(origins):
  days=[]
  for h in range(H):
   b=base[:,j,h*96:(h+1)*96];x,scale,anchor=features(d,o,h);head=np.clip(anchor+scale[:,None]*model.predict(x),0,1)
   hist=d['y'][:,o-1];shape=(1-cfg['shape'])*b+cfg['shape']*(hist-hist.mean(-1)[:,None]+b.mean(-1)[:,None]);pm=anchor[:,1] if cfg['peak_kind']=='history' else head[:,1]
   mean=(1-cfg['mean'])*b.mean(-1)+cfg['mean']*head[:,0];peak=(1-cfg['peak'])*b.max(-1)+cfg['peak']*pm
   days.append(project(shape,mean,peak,hist))
  out.append(np.concatenate(days,-1))
 return np.stack(out,1)
def score(d,origins,p,h):
 y,m=common.truth(d,origins,h);a=common.metrics(y,p,m);err=p-y;a['rmse']=float(np.sqrt((err[m]**2).mean()));a['absolute_bias_percent']=float(abs(100*err[m].sum()/y[m].sum()));per=100*(abs(err)*m).sum((1,2))/(abs(y)*m).sum((1,2))
 for name,ix in [('cella5_wape',d['cells']=='a'),('other52_wape',d['cells']!='a')]:a[name]=float(per[ix].mean())
 return {k:a[k] for k in METRICS}
def refs24(d):
 p=z('round11/results/dev_predictions.npz','prior');a=z('round3/results/chronos_dev.npz','native');t=z('round3/results/tirex_dev.npz','native');f=z('round3/results/tf3_dev.npz','tf3_cellprod')
 return dict(prior=p,tf3=z('round3/results/tf3_dev.npz','tf3_uni_context2016'),tirex=t,chronos=a,chronos_cov=z('round2/results/development_selected.npz','zero_cov'),equal=(a+t)/2,three_fixed=np.clip(.25*a+.25*t+.5*f,0,1),curve_three=z('round2/results/development_selected.npz','new_curve'),peak_three=z('round2/results/development_selected.npz','new_peak'),last=common.baseline(d,list(range(14,22))))
def refs7(d):
 with np.load(B/'results/dev7.npz') as a:r={k:a[k].astype('float64') for k in ['last','week','ridge_1000','lgb_p0_l15_n200','ensemble7','chronos_zero','chronos_cov']}
 r['tf3']=z('round15/results/tf3_dev7.npz','prediction');return r
def envelope(d,origins,refs,h):return {k:min(score(d,origins,v,h)[k] for v in refs.values()) for k in METRICS}
def worst(values,best):return max(values[k]/max(best[k],1e-8) for k in METRICS)
def dev7(d):
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);sys.path.insert(0,str(B/'round3/code'));sys.path.insert(0,str(B/'round2/code'))
 from timesfm3 import TimesFM3Forecaster
 from timesfm3_run import forecast,CONFIGS as TFCONFIG
 if torch.cuda.is_available():
  from timesfm3.torch.timesfm3_forecaster import ModelConfig
  from timesfm3.torch import configs as c
  raw=read(B/'models/timesfm3-base/config.json');tc=dict(raw['transformer_config']);tc['transformer']=c.TransformerConfig(**tc['transformer'])
  cfg=ModelConfig(checkpoint_path=str(B/'models/timesfm3-base/model.safetensors'),device='cuda',per_core_batch_size=1,input_patch_length=raw['input_patch_len'],output_patch_length=raw['output_patch_len'],quantiles=raw['quantiles'],residual_block_config=c.ResidualBlockConfig(**raw['residual_block_config']),transformer_config=c.StackedTransformersConfig(**tc),**{k:raw[k] for k in ['input_transform','linear_detrending_threshold','use_frozen_running_stats','use_iterative_cpm_revin','use_linear_detrending','use_stitching','use_variate_attention','value_clip']})
  with torch.device('cuda'):model=TimesFM3Forecaster(config=cfg)
 else:model=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device='cpu',per_core_batch_size=1,local_files_only=True)
 control=np.clip(forecast(model,d,[14],TFCONFIG['tf3_uni_context2016'],h=96),0,1);delta=float(abs(control-z('round3/results/tf3_dev.npz','tf3_uni_context2016')[:,:1]).max());assert delta<1e-5
 preds=[]
 for o in [14,15]:preds.append(np.clip(forecast(model,d,[o],TFCONFIG['tf3_uni_context2016'],h=672),0,1))
 np.savez_compressed(Q/'tf3_dev7.npz',prediction=np.concatenate(preds,1));save(Q/'tf3_dev7_verification.json',dict(max_abs_control=delta,origins=[14,15],horizon=672,protocol_sha256=sha(R/'docs/protocol.json'),script_sha256=sha(__file__),model_weights_sha256=sha(B/'models/timesfm3-base/model.safetensors')))
def develop(d):
 assert not (Q/'selection.json').exists(),'Do not overwrite frozen selection'
 refs=refs24(d);base=refs['prior'];r7=refs7(d);b7=r7['tf3'];early=list(range(14,18));late=list(range(18,22));bestearly=envelope(d,early,{k:v[:,:4] for k,v in refs.items()},1);heads={a:fit_heads(d,14,a) for a in [10,100]};preds={};rows=[]
 for cfg in CONFIGS:
  p=candidate(d,list(range(14,22)),base,cfg,heads[cfg['alpha']]);preds[cfg['id']]=p;sc=score(d,early,p[:,:4],1);rows.append(dict(id=cfg['id'],worst_metric_ratio=worst(sc,bestearly),scores=sc));print('DEV CANDIDATE',cfg['id'],rows[-1]['worst_metric_ratio'],flush=True)
 winner=min(rows,key=lambda x:(x['worst_metric_ratio'],next(i for i,c in enumerate(CONFIGS) if c['id']==x['id'])));cfg=next(c for c in CONFIGS if c['id']==winner['id']);new7=candidate(d,[14,15],b7,cfg,heads[cfg['alpha']]);e24=envelope(d,late,{k:v[:,4:] for k,v in refs.items()},1);e7=envelope(d,[14,15],r7,7);valid=[]
 for a in [0.,.25,.5,1.]:
  p=(1-a)*base[:,4:]+a*preds[cfg['id']][:,4:];q=(1-a)*b7+a*new7;s24=score(d,late,p,1);s7=score(d,[14,15],q,7);rat=max(worst(s24,e24),worst(s7,e7));valid.append(dict(alpha=a,worst_ratio=rat,scores24=s24,scores7=s7,all_metrics_at_least_as_good=rat<=1+1e-10))
 eligible=[a for a in valid if a['alpha']>0 and a['all_metrics_at_least_as_good']];pick=min(eligible,key=lambda a:(a['worst_ratio'],a['alpha'])) if eligible else valid[0]
 # Future observations cannot alter current features, including seven-day horizon.
 for o in [14,18]:
  altered={k:v.copy() for k,v in d.items()};altered['y'][:,o:]=.123;altered['valid'][:,o:]=False;altered['prod'][:,o:]=.987
  for h in range(7):assert all(np.array_equal(a,b) for a,b in zip(features(d,o,h),features(altered,o,h)))
 np.savez_compressed(Q/'dev_predictions.npz',**preds,seven_candidate=new7,seven_prior=b7)
 save(Q/'dev_summary.json',dict(candidates=rows,confirmation=valid,envelope24=e24,envelope7=e7,future_feature_check=True,all_metric_winner_found=bool(eligible),base_model='round8b24 / TimesFM3uni21_7',testing_not_loaded=True))
 s=dict(frozen_utc=utc(),candidate=cfg,chosen_alpha=pick['alpha'],all_metric_dev_eligible=bool(eligible),protocol_sha256=sha(R/'docs/protocol.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),dev_summary_sha256=sha(Q/'dev_summary.json'),independent_confirmation=False)
 save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();assert sha(R/'docs/protocol.json')==s['protocol_sha256'];assert sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256'] and sha(Q/'dev_summary.json')==s['dev_summary_sha256']
 cfg=s['candidate'];head=fit_heads(d,22,cfg['alpha']);pickle.dump(head,(R/'models/joint_heads_final.pkl').open('wb'));p=z('round8b/results/test_predictions.npz','selected');p7=z('seven_day_completion_20261005/results/test7_predictions.npz','tf3_uni21_7');cp=candidate(d,list(range(22,31)),p,cfg,head);cq=candidate(d,[22,23,24],p7,cfg,head);a=s['chosen_alpha'];sp=(1-a)*p+a*cp;sq=(1-a)*p7+a*cq
 np.savez_compressed(Q/'test_predictions.npz',candidate24=cp,selected24=sp,prior24=p,candidate7=cq,selected7=sq,prior7=p7)
 old=read(B/'model_comparison_20261005/results/task_revision/summary.json');refs1={v['model']:z(v['source'],v['array']) for v in old['models']};refs_7={v['model']:z(v['source'],v['array']) for v in old['seven_day']};ev1=envelope(d,list(range(22,31)),refs1,1);ev7=envelope(d,[22,23,24],refs_7,7)
 scores24={k:score(d,list(range(22,31)),v,1) for k,v in dict(prior=p,candidate=cp,selected=sp).items()};scores7={k:score(d,[22,23,24],v,7) for k,v in dict(prior=p7,candidate=cq,selected=sq).items()};rat=max(worst(scores24['candidate'],ev1),worst(scores7['candidate'],ev7))
 save(Q/'summary.json',dict(selection=s,scores24=scores24,scores7=scores7,test_best_control_per_metric24=ev1,test_best_control_per_metric7=ev7,all_metric_test_candidate_dominance=rat<=1+1e-10,test_worst_ratio=rat,no_post_test_selection=True,independent_confirmation=False,overlapping_windows=True,fitted_final_label_cutoff=22))
 save(Q/'implementation_verification.json',dict(all_passed=True,training_targets_before_cutoff=True,future_feature_perturbation_passed=True,mean_peak_projection_passed=True,selected24_unchanged_if_alpha0=bool(a!=0 or np.array_equal(sp,p)),selected7_unchanged_if_alpha0=bool(a!=0 or np.array_equal(sq,p7)),selection_sha256=sha(Q/'selection.json')))
 print('ACTUAL JOINT TEST',json.dumps(dict(scores24=scores24,scores7=scores7,all_metric_dominance=rat<=1+1e-10)),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(R/'docs/protocol.json')==(R/'docs/protocol.sha256').read_text();assert read(R/'docs/protocol.json')['script_sha256']==sha(__file__);d=common.load()
 if a.stage=='dev7':dev7(d)
 elif a.stage=='develop':develop(d)
 elif a.stage=='test':test(d)
 else:raise ValueError(a.stage)
