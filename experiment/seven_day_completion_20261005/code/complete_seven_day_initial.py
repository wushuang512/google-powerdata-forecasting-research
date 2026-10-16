"""Frozen-weight seven-day inference; no fitting or future observations."""
from pathlib import Path
import sys,os,json,hashlib,ast,pickle,datetime,argparse
import numpy as np
TOP=Path(sys.argv[1]).resolve();B=TOP/'experiment';R=B/'seven_day_completion_20261005'
sys.path.insert(0,str(B/'code'));import common
P=R/'protocol.json';RESULTS=R/'results';CACHE=R/'cache'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
 return h.hexdigest()
def save(p,v):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def signature():return sha(P)+':'+sha(Path(__file__))+':'+json.loads(P.read_text())['source_digest']
def z(rel,key):
 with np.load(B/rel,allow_pickle=False) as f:return f[key].copy()
def get_model_fn(rel,name,namespace):
 tree=ast.parse((B/rel).read_text(encoding='utf-8'))
 fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(B/rel),'exec'),namespace)
 return namespace[name]
def bank5():
 b={}
 for split,origins in [('train',list(range(7,14))),('dev',list(range(14,22))),('test',list(range(22,31)))]:
  c=z(f'round3/results/chronos_{split}.npz','native');t=z(f'round3/results/tirex_{split}.npz','native');f=z(f'round3/results/tf3_{split}.npz','tf3_cellprod');a=np.clip(.25*c+.25*t+.5*f,0,1)
  for j,o in enumerate(origins):b[o]=a[:,j]
 return b
def bank3():
 b={}
 for split,origins in [('train',list(range(7,14))),('dev',list(range(14,22))),('test',list(range(22,31)))]:
  c=z(f'round3/results/chronos_{split}.npz','native');t=z(f'round3/results/tirex_{split}.npz','native')
  for j,o in enumerate(origins):b[o]=(c[:,j],t[:,j])
 return b
def cache_get(key):
 p=CACHE/(key+'.npz');meta=CACHE/(key+'.json')
 if not p.exists():return None
 a=json.loads(meta.read_text());assert a['signature']==signature() and a['sha256']==sha(p),'Incompatible cache'
 with np.load(p,allow_pickle=False) as f:return f['prediction'].copy()
def cache_put(key,p,checks=None):
 assert np.isfinite(p).all();f=CACHE/(key+'.npz');tmp=CACHE/(key+'.tmp.npz');np.savez_compressed(tmp,prediction=p);tmp.replace(f)
 save(CACHE/(key+'.json'),dict(signature=signature(),sha256=sha(f),checks=checks or {},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 print('SAVED',key,tuple(p.shape),flush=True)
def prepare():
 for p in [R,RESULTS,CACHE]:p.mkdir(parents=True,exist_ok=True)
 if P.exists():print('Existing immutable protocol retained');return
 files=['results/data.npz','round2/results/test7_predictions.npz','round4/results/test7.npz','round3/results/selection.json','round4/results/selection.json','round5/results/selection.json','round8b/results/selection.json','round3/models/ridge100000_final.pkl','round5/models/residual_ridge100000_final.pkl','round8b/models/best_tail_lr1e6_teacher01_s100_final.pt','round3/code/residual.py','round5/code/calibration.py','code/classical.py']
 for split in ['train','dev','test']:
  files += [f'round3/results/{kind}_{split}.npz' for kind in ['chronos','tirex','tf3']]
 files+=['round3/results/residual_test.npz','round5/results/test_predictions.npz','round8b/results/test_predictions.npz','round3/code/timesfm3_run.py','round2/code/round2_lib.py','round3/code/multiscale.py']
 for model in ['timesfm3-base','timesfm25-base','chronos2-base']:
  files += [str(p.relative_to(B)).replace('\\','/') for p in sorted((B/'models'/model).glob('*.safetensors'))]
 hashes={p:sha(B/p) for p in files};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
 save(P,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),origins=[22,23,24],horizon=672,no_refit=True,no_test_selection=True,new_downloads=False,
  inherited_weights=dict(round3='frozen selection.json weights, components below 1e-12 are numerical zero',round5_alpha=.5,round8b_alpha=.5,round8b_component_delta=.5),
  scope='Complete missing seven-day tasks for existing frozen models; report transferred residual inference as adapted version, not a new trained seven-day model',
  method='TimesFM3 uni21 and fine-tuned cellprod14, Chronos2 mirror14 and TimesFM2.5 context1344 flip1 predict672 directly. R5 and R3 residual regressors are applied separately to each forecast day using the same fixed-origin observed power/proxy/error history, day-specific base forecast and known future weekday for R3; never append observed future power or workload. Weights and calibration coefficients unchanged.',
  future_observation_policy='No features use days >= origin. Saved historical forecast bank entries k<origin only. Day templates use original last known day and week, not simulated truths.',
  controls='Each heavy model first reproduces frozen 96-point origin22 cache; max absolute difference <=1e-5 normalized utilization. R5 residual extraction must reproduce all saved24 forecasts <=1e-10; future perturbation must leave feature arrays identical.',
  candidate_ids=['r5_transfer7','best_transfer7','r3_transfer7','tf3_uni21_7','tf25_7'],
  missing_tasks_not_to_be_filled_with_24h_scores=True,source_hashes=hashes,source_digest=digest,independent_confirmation=False,overlapping_windows=True,
  resource_policy=dict(admission_physical_gib=3,admission_commit_gib=6,runtime_physical_gib=1.5,runtime_commit_gib=2)))
 (R/'protocol.sha256').write_text(sha(P),encoding='ascii');print('PROTOCOL FROZEN',sha(P),flush=True)
def calibration(d):
 base=z('round4/results/test7.npz','best_mix');b=bank5();fn=get_model_fn('round5/code/calibration.py','features',{'np':np});model=pickle.load((B/'round5/models/residual_ridge100000_final.pkl').open('rb'))
 # Existing 24h score-independent reproduction check across all nine origins.
 base24=z('round5/results/test_predictions.npz','anchor');out=[]
 for j,o in enumerate(range(22,31)):
  bb=dict(b);bb[o]=base24[:,j];x,a,_=fn(d,o,bb);out.append(np.clip(.5*a+.5*np.clip(a+model.predict(x).reshape(57,96),0,1),0,1))
 reproduced=np.stack(out,1);ref=z('round5/results/test_predictions.npz','selected');delta=float(abs(reproduced-ref).max());assert delta<=1e-10,delta
 preds=[]
 for j,o in enumerate([22,23,24]):
  days=[]
  for h in range(7):
   bb=dict(b);bb[o]=base[:,j,h*96:(h+1)*96];x,a,_=fn(d,o,bb)
   altered={k:v.copy() for k,v in d.items()};altered['y'][:,o:]=.123;altered['prod'][:,o:]=.987;altered['valid'][:,o:]=False;altered['pvalid'][:,o:]=False
   xx,_,_=fn(altered,o,bb);assert np.array_equal(x,xx),'Future-dependent feature'
   days.append(np.clip(.5*a+.5*np.clip(a+model.predict(x).reshape(57,96),0,1),0,1))
  preds.append(np.concatenate(days,-1))
 cache_put('r5_transfer7',np.stack(preds,1),dict(reproduction_max_abs=delta,future_feature_perturbation=True))
def torch_setup():
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 sys.path.insert(0,str(B/'round2/code'));sys.path.insert(0,str(B/'round3/code'));sys.path.insert(0,str(B/'round5/code'))
 return torch
def heavy(stage,d):
 torch=torch_setup()
 if stage in ['tf3_uni21_7','tf3_finetuned7']:
  from timesfm3 import TimesFM3Forecaster
  from timesfm3_run import forecast,CONFIGS
  model=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device='cuda' if torch.cuda.is_available() else 'cpu',per_core_batch_size=1,local_files_only=True)
  key='tf3_uni_context2016' if stage=='tf3_uni21_7' else 'tf3_cellprod'
  if stage=='tf3_finetuned7':
   checkpoint=torch.load(B/'round8b/models/best_tail_lr1e6_teacher01_s100_final.pt',map_location='cpu',weights_only=True);check=model.model.load_state_dict(checkpoint['state'],strict=False);assert not check.unexpected_keys;del checkpoint
  control=cache_get(stage+'_control')
  if control is None:
   control=np.clip(forecast(model,d,[22],CONFIGS[key],h=96),0,1)
   if stage=='tf3_uni21_7':ref=z('round3/results/tf3_test.npz',key)[:,:1]
   else:
    prior=z('round8b/results/test_predictions.npz','prior')[:,:1];orig=z('round3/results/tf3_test.npz','tf3_cellprod')[:,:1];ref=z('round8b/results/test_predictions.npz','selected')[:,:1]
    control=np.clip(.5*prior+.5*np.clip(prior+.5*(control-orig),0,1),0,1)
   delta=float(abs(control-ref).max());assert delta<=1e-5,(stage,delta);cache_put(stage+'_control',control,dict(max_abs=delta))
  for o in [22,23,24]:
   if cache_get(stage+'_o'+str(o)) is not None:continue
   p=np.clip(forecast(model,d,[o],CONFIGS[key],h=672),0,1);cache_put(stage+'_o'+str(o),p)
 elif stage=='chronos_mirror7':
  from deep_models import base_chronos
  model=base_chronos()
  def predict(o,h):
   a=(1-d['y'][:,max(0,o-14):o].reshape(57,-1)).astype('float32');q=model.predict_quantiles([x for x in a],prediction_length=h,quantile_levels=[.5],context_length=1344,batch_size=8)[0];return 1-np.stack([x[0,:,0].cpu().numpy() for x in q])[:,None]
  if cache_get(stage+'_control') is None:
   p=predict(22,96);ref=z('round3/results/chronos_test.npz','mirror')[:,:1];delta=float(abs(p-ref).max());assert delta<=1e-5,delta;cache_put(stage+'_control',p,dict(max_abs=delta))
  for o in [22,23,24]:
   if cache_get(stage+'_o'+str(o)) is None:cache_put(stage+'_o'+str(o),np.clip(predict(o,672),0,1))
 elif stage=='tf25_7':
  import timesfm
  model=timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(B/'models/timesfm25-base'),torch_compile=False,local_files_only=True)
  def compile(h):model.compile(timesfm.ForecastConfig(max_context=1344,max_horizon=h,normalize_inputs=True,per_core_batch_size=1,use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True))
  def predict(o,h):
   a=d['y'].reshape(57,-1)[:,max(0,o*96-1344):o*96];p,_=model.forecast(horizon=h,inputs=[x for x in a]);return np.clip(p[:,None],0,1)
  if cache_get(stage+'_control') is None:
   compile(128);p=predict(22,96);ref=z('round4/results/test_predictions.npz','tf25_c1344_flip1')[:,:1];delta=float(abs(p-ref).max());assert delta<=1e-5,delta;cache_put(stage+'_control',p,dict(max_abs=delta))
  compile(768)
  for o in [22,23,24]:
   if cache_get(stage+'_o'+str(o)) is None:cache_put(stage+'_o'+str(o),predict(o,672))
def finish(d):
 import pandas as pd
 def full(key):return np.concatenate([cache_get(key+'_o'+str(o)) for o in [22,23,24]],1)
 p5=cache_get('r5_transfer7');ff=full('tf3_finetuned7');orig=z('round4/results/test7.npz','tf3_cellprod');best=np.clip(.5*p5+.5*np.clip(p5+.5*(ff-orig),0,1),0,1)
 c=z('round2/results/test7_predictions.npz','zero_uni');t=z('round2/results/test7_predictions.npz','tirex_uni');eq=(c+t)/2;mir=full('chronos_mirror7');tf3=full('tf3_uni21_7')
 from classical import features
 fn=get_model_fn('round3/code/residual.py','xfeatures',{'np':np,'features':features,'smooth':common.smooth})
 bank=bank3();model=pickle.load((B/'round3/models/ridge100000_final.pkl').open('rb'));res=[];online=[]
 # Verify that the extracted original residual implementation reproduces 24h.
 ref=z('round3/results/residual_test.npz','residual_ridge100000');ctrl=[]
 for o in range(22,31):
  x,_,a=fn(d,o,bank);ctrl.append(a+model.predict(x).reshape(57,96))
 residual_control_difference=float(abs(np.stack(ctrl,1)-ref).max());assert residual_control_difference<=1e-10,residual_control_difference
 for j,o in enumerate([22,23,24]):
  rr=[];oo=[]
  for h in range(7):
   bb=dict(bank);bb[o]=(c[:,j,h*96:(h+1)*96],t[:,j,h*96:(h+1)*96]);x,names,a=fn(d,o,bb)
   # Future weekday is known at origin; all observations remain fixed at o.
   altered={k:v.copy() for k,v in d.items()};altered['y'][:,o:]=.123;altered['prod'][:,o:]=.987;altered['valid'][:,o:]=False;altered['pvalid'][:,o:]=False
   xx,_,_=fn(altered,o,bb);assert np.array_equal(x,xx),'Future-dependent R3 feature'
   dow=(2+o+h)%7;x[:,names.index('day_of_week')]=dow;x[:,names.index('weekend')]=int(dow>=5)
   rr.append(a+model.predict(x).reshape(57,96));past=o-1;bias=(d['y'][:,past]-(bank[past][0]+bank[past][1])/2).mean(1)[:,None];oo.append(np.clip(a+.25*bias,0,1))
  res.append(np.concatenate(rr,-1));online.append(np.concatenate(oo,-1))
 rs=np.stack(res,1);on=np.stack(online,1);weights=json.loads((B/'round3/results/selection.json').read_text())['weight_map']
 components={'old_equal_mix':eq,'chronos_mirror':mir,'residual_ridge100000':rs,'online_w1_a0.25_shape0':on,'tf3_uni_context2016':tf3}
 r3=np.clip(sum(weights[k]*v for k,v in components.items()),0,1)
 preds={'r5_transfer7':p5,'best_transfer7':best,'r3_transfer7':r3,'tf3_uni21_7':tf3,'tf25_7':full('tf25_7')}
 y,m=common.truth(d,[22,23,24],7);scores={k:common.metrics(y,v.astype('float64'),m) for k,v in preds.items()}
 np.savez_compressed(RESULTS/'test7_predictions.npz',**preds)
 save(RESULTS/'summary.json',dict(scores=scores,protocol_sha256=sha(P),fixed_weights=True,new_fit=False,future_truth_in_features=False,feature_future_perturbation_passed=True,residual_control_max_abs=residual_control_difference,adapter_methods=['r5_transfer7','best_transfer7','r3_transfer7'],independent_confirmation=False,overlapping_windows=True))
 pd.DataFrame([dict(model=k,**v) for k,v in scores.items()]).to_csv(RESULTS/'test7_scores.csv',index=False,encoding='utf-8-sig')
 print('ACTUAL SEVEN DAY SCORES',json.dumps(scores),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(P)==(R/'protocol.sha256').read_text();d=common.load()
 if a.stage=='calibration':calibration(d)
 elif a.stage=='finalize':finish(d)
 else:heavy(a.stage,d)
