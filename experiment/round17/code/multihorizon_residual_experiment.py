"""Frozen-backbone multi-horizon Fourier residual calibration, without TEST selection."""
from pathlib import Path
import sys,os,json,hashlib,datetime,argparse,pickle
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round17';Q=R/'results'
sys.path.insert(0,str(B/'code'));import common
# Pure scoring/projection/control utilities from the immutable prior implementation.
_p=B/'round16/code/joint_local_experiment.py';_ns={'__file__':str(_p),'__name__':'prior_pure_utilities'}
exec(compile(_p.read_text(encoding='utf-8').split('ap=argparse.ArgumentParser()')[0],str(_p),'exec'),_ns)
METRICS=_ns['METRICS'];score=_ns['score'];envelope=_ns['envelope'];worst=_ns['worst'];project=_ns['project'];refs24=_ns['refs24'];refs7=_ns['refs7']
CONFIGS=[dict(id='prior',ridge=1000,mean=0.,shape=0.,peak=0.,gate=False)]
for ridge in [100,1000]:
 for mode,mean,shape,peak in [('mean',1.,0.,0.),('mean_shape',1.,1.,0.),('mean_peak',1.,0.,1.),('mean_shape_peak',1.,1.,1.)]:
  for gate in [False,True]:CONFIGS.append(dict(id=f'ridge{ridge}_{mode}_agree{int(gate)}',ridge=ridge,mean=mean,shape=shape,peak=peak,gate=gate))
# Two conservative alternatives retain an observed previous-day peak anchor.
for gate in [False,True]:CONFIGS.append(dict(id=f'ridge1000_mean_shape_history50_agree{int(gate)}',ridge=1000,mean=1.,shape=1.,peak=.5,gate=gate,history_peak=True))
phase=np.arange(96)*2*np.pi/96
FOURIER=np.stack([np.ones(96)]+[f(k*phase) for k in [1,2,3] for f in [np.cos,np.sin]],axis=0)
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
def bank24():
 banks=[]
 for split in ['train','dev']:
  c=z(f'round3/results/chronos_{split}.npz','native');t=z(f'round3/results/tirex_{split}.npz','native');f=z(f'round3/results/tf3_{split}.npz','tf3_cellprod');banks.append(np.clip(.25*c+.25*t+.5*f,0,1))
 return np.concatenate(banks,1)
def prepare():
 for a in ['code','docs','models','results','cache']:(R/a).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert read(R/'docs/protocol.json')['script_sha256']==sha(__file__);return
 files=['results/data.npz','round16/code/joint_local_experiment.py','round11/results/dev_predictions.npz','round8b/results/selection.json','round8b/results/test_predictions.npz','round2/results/development_selected.npz','results/dev7.npz','seven_day_completion_20261005/results/test7_predictions.npz','model_comparison_20261005/results/task_revision/summary.json','models/timesfm3-base/config.json','models/timesfm3-base/model.safetensors','round15/results/tf3_dev7.npz','round15/results/tf3_dev7_verification.json','round3/code/timesfm3_run.py']
 for s in ['train','dev']:
  for m in ['chronos','tirex','tf3']:files.append(f'round3/results/{m}_{s}.npz')
 save(R/'docs/protocol.json',dict(created_utc=utc(),model_name='Chronos-2 + TiRex-2 + TimesFM3 with forecast-conditioned multi-horizon Ridge Fourier residual heads and regularization-agreement gating',script_sha256=sha(__file__),configs=CONFIGS,metrics=METRICS,training_cutoff_dev=14,training_cutoff_final=22,training24='Origins 7..cutoff-1; fixed .25Chronos2+.25TiRex2+.5TimesFM3-cellprod forecast residuals. This frozen zero-shot bank is a proxy for the refined round8b base, not in-sample fine-tuned residuals.',training7='Native original TimesFM3 direct672 forecasts from origins7..21, retain targets o+h<cutoff; inverse target-date frequency weighting. No single-step residual extrapolation.',features='Forecast daily statistics and first3 Fourier harmonics, observed past7 power daily statistics/harmonics, known future weekday and forecast horizon, historical production proxy level/trend, cell onehot. All observed inputs strictly before origin.',targets='Normalized residual of valid-slot daily mean, six fitted sine/cosine shape coefficients, and valid-slot daily peak relative to each corresponding frozen forecast. Full96 output evaluated on future valid mask only at scoring.',bounds='Each predicted residual coefficient clipped to +/-5% of observed past7 power mean; daily peak pin with minimal-L2 bounded projection. Mean-only additive correction is clipped into[0,1]. Shape constraint is not energy conservation.',gate='Optional multiplier per channel: zero when Ridge100 and1000 residual signs disagree; otherwise abs(smaller)/max(abs(larger),.002). Same gate and parameters at both horizons.',development24=list(range(14,22)),development7=[14,15],test24=list(range(22,31)),test7=[22,23,24],selection='19 predeclared configs including prior; first four DEV24 dates choose one family by minimax ratios to metric-best controls; later four DEV24 plus both DEV7 windows choose one common shrinkage 0/.25/.5/1. Adoption requires all10 metrics at both horizons<=metric-best controls. Freeze before one TEST evaluation; never reselect after TEST.',base24='round8b canonical DEV14 training cutoff and final frozen selected prediction',base7='TimesFM3 uni21 direct672, previously TEST-identified baseline, hence exploratory',independent_confirmation=False,test_seen_before=True,overlapping_seven_windows=True,new_downloads=False,resource_policy=dict(admission_physical_gib=3,admission_commit_gib=6,runtime_physical_gib=1.5,runtime_commit_gib=2),source_hashes={f:sha(B/f) for f in files}))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii');print('PROTOCOL FROZEN',sha(R/'docs/protocol.json'),flush=True)
def forecast_train7(d):
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);sys.path.insert(0,str(B/'round3/code'));sys.path.insert(0,str(B/'round2/code'))
 from timesfm3 import TimesFM3Forecaster
 from timesfm3_run import forecast,CONFIGS as TFCONFIG
 assert torch.cuda.is_available(),'Preserve memory reserves: direct GPU load required on this machine'
 from timesfm3.torch.timesfm3_forecaster import ModelConfig
 from timesfm3.torch import configs as c
 raw=read(B/'models/timesfm3-base/config.json');tc=dict(raw['transformer_config']);tc['transformer']=c.TransformerConfig(**tc['transformer'])
 cfg=ModelConfig(checkpoint_path=str(B/'models/timesfm3-base/model.safetensors'),device='cuda',per_core_batch_size=1,input_patch_length=raw['input_patch_len'],output_patch_length=raw['output_patch_len'],quantiles=raw['quantiles'],residual_block_config=c.ResidualBlockConfig(**raw['residual_block_config']),transformer_config=c.StackedTransformersConfig(**tc),**{k:raw[k] for k in ['input_transform','linear_detrending_threshold','use_frozen_running_stats','use_iterative_cpm_revin','use_linear_detrending','use_stitching','use_variate_attention','value_clip']})
 with torch.device('cuda'):model=TimesFM3Forecaster(config=cfg)
 control=np.clip(forecast(model,d,[14],TFCONFIG['tf3_uni_context2016'],h=96),0,1);delta=float(abs(control-z('round3/results/tf3_dev.npz','tf3_uni_context2016')[:,:1]).max());assert delta<1e-5
 hashes={};ps=[];dev=z('round15/results/tf3_dev7.npz','prediction');protocol_sha=sha(R/'docs/protocol.json')
 for o in range(7,22):
  path=R/f'cache/origin{o}.npz';meta=path.with_suffix('.json')
  if path.exists():assert read(meta)['protocol_sha256']==protocol_sha and read(meta)['sha256']==sha(path)
  else:
   p=dev[:,o-14:o-13] if o in [14,15] else np.clip(forecast(model,d,[o],TFCONFIG['tf3_uni_context2016'],h=672),0,1)
   assert p.shape==(57,1,672) and np.isfinite(p).all();np.savez_compressed(path,prediction=p)
   save(meta,dict(origin=o,horizon=672,protocol_sha256=protocol_sha,sha256=sha(path),reused_checked_dev=o in [14,15]))
  with np.load(path) as a:ps.append(a['prediction'].astype('float64'))
  hashes[str(o)]=sha(path);print('NATIVE TRAIN672 CACHE',o,flush=True)
 np.savez_compressed(Q/'native_train7.npz',prediction=np.concatenate(ps,1))
 save(Q/'native_train7_verification.json',dict(origins=list(range(7,22)),shape=[57,15,672],control96_max_abs_difference=delta,protocol_sha256=protocol_sha,source_hashes=hashes,weights_sha256=sha(B/'models/timesfm3-base/model.safetensors'),aggregate_sha256=sha(Q/'native_train7.npz')))
def features(d,o,h,b):
 hist=d['y'][:,o-7:o];m=d['valid'][:,o-7:o];cnt=m.sum(-1).clip(1);mu=(hist*m).sum(-1)/cnt;scale=mu.mean(-1).clip(.05);hist_fill=np.where(m,hist,mu[:,:,None]);pk=hist_fill.max(-1);std=hist_fill.std(-1);lo=hist_fill.min(-1);v=[]
 for k in [1,3,7]:
  for a in [mu,pk,lo,std]:v.append(a[:,-k:].mean(-1)/scale)
 v.extend([(mu[:,-1]-mu[:,-3:].mean(-1))/scale,(pk[:,-1]-pk[:,-3:].mean(-1))/scale,(mu[:,-1]-mu[:,0])/scale])
 for a in [b.mean(-1),b.max(-1),b.min(-1),b.std(-1)]:v.append(a/scale)
 coeff=b@FOURIER.T/96;last=hist_fill[:,-1]@FOURIER.T/96
 for j in range(1,7):v += [coeff[:,j]/scale,last[:,j]/scale]
 v += [(b.mean(-1)-mu[:,-1])/scale,(b.max(-1)-pk[:,-1])/scale]
 prod=d['prod'][:,o-7:o];pm=d['pvalid'][:,o-7:o];pc=pm.sum((1,2)).clip(1);pl=(prod*pm).sum((1,2))/pc;p1=(prod[:,-1]*pm[:,-1]).sum(-1)/pm[:,-1].sum(-1).clip(1)
 v += [pl,p1-pl,pm.mean((1,2)),cnt.mean(-1)/96]
 day=(2+o+h)%7;v += [np.full(57,np.sin(day*2*np.pi/7)),np.full(57,np.cos(day*2*np.pi/7)),np.full(57,h/6)]
 v += [(d['cells']==c).astype(float) for c in sorted(set(d['cells']))]
 return np.stack(v,-1),scale,hist_fill[:,-1]
def fit_heads(d,cutoff,alpha,H):
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 bank=bank24() if H==1 else z('round17/results/native_train7.npz','prediction');xs=[];ys=[];dates=[]
 for o in range(7,cutoff):
  for h in range(H):
   e=o+h
   if e>=cutoff:continue
   b=bank[:,o-7,h*96:(h+1)*96];x,scale,_=features(d,o,h,b);m=d['valid'][:,e];keep=m.sum(-1)>=48
   residual=d['y'][:,e]-b;mean=(residual*m).sum(-1)/m.sum(-1).clip(1);target=[]
   # Fit harmonics on observed residual slots, rather than treating missing truth as zeros.
   for i in range(57):
    co=np.linalg.lstsq(FOURIER[:,m[i]].T,residual[i,m[i]],rcond=None)[0] if keep[i] else np.zeros(7)
    peak=np.max(d['y'][i,e,m[i]])-np.max(b[i,m[i]]) if keep[i] else 0
    target.append(np.r_[mean[i],co[1:],peak]/scale[i])
   xs.append(x[keep]);ys.append(np.stack(target)[keep]);dates.append(np.full(keep.sum(),e))
 dates=np.concatenate(dates);w=1/np.bincount(dates)[dates];model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(np.concatenate(xs),np.concatenate(ys),ridge__sample_weight=w*len(w)/w.sum());assert np.isfinite(model[-1].coef_).all();return model
def candidate(d,origins,base,cfg,heads,H):
 if cfg['id']=='prior':return base.copy()
 out=[]
 for j,o in enumerate(origins):
  days=[]
  for h in range(H):
   b=base[:,j,h*96:(h+1)*96];x,scale,history=features(d,o,h,b);a=np.clip(heads[100].predict(x),-.05,.05);c=np.clip(heads[1000].predict(x),-.05,.05);q=a if cfg['ridge']==100 else c
   gate=np.where(a*c>0,np.minimum(abs(a),abs(c))/np.maximum(np.maximum(abs(a),abs(c)),.002),0) if cfg['gate'] else np.ones_like(q);q=q*gate*scale[:,None]
   mean=b.mean(-1)+cfg['mean']*q[:,0];shape=b+cfg['shape']*(q[:,1:7]@FOURIER[1:])
   if cfg['peak']:
    change=history.max(-1)-b.max(-1) if cfg.get('history_peak') else q[:,7]
    peak=b.max(-1)+cfg['peak']*change;v=project(shape,mean,peak,history)
   else:v=np.clip(shape+(mean-shape.mean(-1))[:,None],0,1)
   days.append(v)
  out.append(np.concatenate(days,-1))
 return np.stack(out,1)
def develop(d):
 assert not (Q/'selection.json').exists(),'Frozen selection must never be overwritten'
 heads={H:{a:fit_heads(d,14,a,H) for a in [100,1000]} for H in [1,7]};pickle.dump(heads,(R/'models/dev_heads.pkl').open('wb'));refs=refs24(d);r7=refs7(d);base=refs['prior'];b7=r7['tf3'];early=list(range(14,18));late=list(range(18,22));ev=envelope(d,early,{k:v[:,:4] for k,v in refs.items()},1);preds={};rows=[]
 for cfg in CONFIGS:
  p=candidate(d,list(range(14,22)),base,cfg,heads[1],1);preds[cfg['id']]=p;s=score(d,early,p[:,:4],1);rows.append(dict(id=cfg['id'],worst_metric_ratio=worst(s,ev),scores=s));print('DEV',cfg['id'],rows[-1]['worst_metric_ratio'],flush=True)
 winner=min(rows,key=lambda a:a['worst_metric_ratio']);cfg=next(c for c in CONFIGS if c['id']==winner['id']);cp=preds[cfg['id']];cq=candidate(d,[14,15],b7,cfg,heads[7],7);e24=envelope(d,late,{k:v[:,4:] for k,v in refs.items()},1);e7=envelope(d,[14,15],r7,7);valid=[]
 for alpha in [0.,.25,.5,1.]:
  s24=score(d,late,(1-alpha)*base[:,4:]+alpha*cp[:,4:],1);s7=score(d,[14,15],(1-alpha)*b7+alpha*cq,7);ratio=max(worst(s24,e24),worst(s7,e7));valid.append(dict(alpha=alpha,worst_ratio=ratio,scores24=s24,scores7=s7,all_metrics_at_least_as_good=ratio<=1+1e-10))
 eligible=[a for a in valid if a['alpha']>0 and a['all_metrics_at_least_as_good']];pick=min(eligible,key=lambda a:(a['worst_ratio'],a['alpha'])) if eligible else valid[0]
 np.savez_compressed(Q/'dev_predictions.npz',**preds,seven_candidate=cq,seven_prior=b7);save(Q/'dev_summary.json',dict(candidates=rows,confirmation=valid,envelope24=e24,envelope7=e7,all_metric_winner_found=bool(eligible),testing_not_loaded=True))
 s=dict(frozen_utc=utc(),candidate=cfg,chosen_alpha=pick['alpha'],all_metric_dev_eligible=bool(eligible),protocol_sha256=sha(R/'docs/protocol.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),dev_summary_sha256=sha(Q/'dev_summary.json'),train7_sha256=sha(Q/'native_train7.npz'),dev_heads_sha256=sha(R/'models/dev_heads.pkl'),independent_confirmation=False)
 save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();assert sha(R/'docs/protocol.json')==s['protocol_sha256'] and sha(Q/'dev_summary.json')==s['dev_summary_sha256'] and sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256'] and sha(Q/'native_train7.npz')==s['train7_sha256']
 heads={H:{a:fit_heads(d,22,a,H) for a in [100,1000]} for H in [1,7]};pickle.dump(heads,(R/'models/final_heads.pkl').open('wb'));cfg=s['candidate'];p=z('round8b/results/test_predictions.npz','selected');p7=z('seven_day_completion_20261005/results/test7_predictions.npz','tf3_uni21_7');cp=candidate(d,list(range(22,31)),p,cfg,heads[1],1);cq=candidate(d,[22,23,24],p7,cfg,heads[7],7);a=s['chosen_alpha'];sp=(1-a)*p+a*cp;sq=(1-a)*p7+a*cq
 np.savez_compressed(Q/'test_predictions.npz',candidate24=cp,selected24=sp,prior24=p,candidate7=cq,selected7=sq,prior7=p7)
 old=read(B/'model_comparison_20261005/results/task_revision/summary.json');ev24=envelope(d,list(range(22,31)),{v['model']:z(v['source'],v['array']) for v in old['models']},1);ev7=envelope(d,[22,23,24],{v['model']:z(v['source'],v['array']) for v in old['seven_day']},7)
 s24={k:score(d,list(range(22,31)),v,1) for k,v in dict(prior=p,candidate=cp,selected=sp).items()};s7={k:score(d,[22,23,24],v,7) for k,v in dict(prior=p7,candidate=cq,selected=sq).items()};ratio=max(worst(s24['candidate'],ev24),worst(s7['candidate'],ev7))
 save(Q/'summary.json',dict(selection=s,scores24=s24,scores7=s7,test_best_control_per_metric24=ev24,test_best_control_per_metric7=ev7,test_worst_ratio=ratio,all_metric_test_candidate_dominance=ratio<=1+1e-10,no_post_test_selection=True,independent_confirmation=False,overlapping_windows=True,fitted_final_label_cutoff=22));print('ACTUAL TEST',json.dumps(dict(scores24=s24,scores7=s7,all_metric_dominance=ratio<=1+1e-10)),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(R/'docs/protocol.json')==(R/'docs/protocol.sha256').read_text() and read(R/'docs/protocol.json')['script_sha256']==sha(__file__);d=common.load()
 if a.stage=='forecast_train7':forecast_train7(d)
 elif a.stage=='develop':develop(d)
 elif a.stage=='test':test(d)
 else:raise ValueError(a.stage)
