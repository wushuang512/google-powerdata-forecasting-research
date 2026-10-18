"""Rebuild Google inputs, HDSF members, selected gradient updates and 9.28 metrics."""
from pathlib import Path
import gzip,hashlib,importlib,sys,gc
from .support import REPO,read,save,sha,source_path,module,export_csv

def setup(work):
 import numpy as np,joblib
 source_path(work)
 sys.path[:0]=[str(work/'experiment/round46/code'),str(work/'experiment/round47/code'),str(work/'experiment/code')]
 from power_forecast.data import load_dataset
 data=load_dataset(work/'google-powerdata-2019/raw')
 config=read(REPO/'repro/google_frozen.json');order=config['canonical_order']
 canon=np.array([data.series_keys.index(n) for n in order])
 with np.load(work/'experiment/results/data.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 assert np.array_equal(d['valid'],data.target_valid[canon])
 assert np.max(abs(d['y'][d['valid']]-data.target[canon][d['valid']]))<1e-12
 return data,d,canon,config

def prepare(work):
 import numpy as np,pandas as pd,joblib
 source_path(work)
 from power_forecast.data import load_dataset
 lock=read(REPO/'repro/assets.lock.json');ys=[];ps=[];ms=[];pms=[];names=[];cells=[]
 for row in lock['google_data']:
  p=work/row['path'];assert sha(p)==row['sha256'],p.name
  if p.name.startswith('machine'):continue
  frame=pd.read_csv(p).sort_values('time');assert len(frame)==8928
  assert np.array_equal(frame.time,600000000+np.arange(8928)*300000000)
  raw=frame.measured_power_util.to_numpy(float);prod=frame.production_power_util.to_numpy(float)
  valid=~frame.bad_measurement_data.to_numpy(bool)&np.isfinite(raw)&(raw>=0)&(raw<=1)
  pv=~frame.bad_production_power_data.to_numpy(bool)&np.isfinite(prod)&(prod>=0)&(prod<=1)
  y=pd.Series(np.where(valid,raw,np.nan)).ffill().to_numpy()
  pp=pd.Series(np.where(pv,prod,np.nan)).ffill().fillna(0).to_numpy()
  assert np.isfinite(y).all() and np.isfinite(pp).all()
  ys.append(y.reshape(31,96,3).mean(-1));ps.append(pp.reshape(31,96,3).mean(-1));ms.append(valid.reshape(31,96,3).all(-1));pms.append(pv.reshape(31,96,3).all(-1))
  names.append(p.name[:-7]);cells.append(str(frame.cell.iloc[0]))
 np.savez_compressed(work/'experiment/results/data.npz',y=ys,prod=ps,valid=ms,pvalid=pms,names=names,cells=cells)
 data=load_dataset(work/'google-powerdata-2019/raw');joblib.dump(data,work/'experiment/round44/results/source_dataset.joblib')
 _,d,canon,cfg=setup(work)
 report={'input_shape':list(d['y'].shape),'valid24':int(d['valid'][:,22:].sum()),'source_and_canonical_valid_targets_equal':True,'original_raw_files':58,'data_files_uploaded':False}
 assert report['valid24']==49243
 save(work/'reproduction/results/google_data_audit.json',report)
 return report

def evaluator(work):
 common=module('reproduction_common',work/'experiment/code/common.py')
 aligned=module('reproduction_aligned_metrics',work/'experiment/round29/code/aligned_metrics.py')
 return common,aligned

def baselines(work):
 import numpy as np
 with np.load(work/'experiment/results/data.npz') as z:d={k:z[k] for k in z.files}
 common,ev=evaluator(work);rows=[]
 for h,oo in [(1,list(range(22,31))),(7,[22,23,24])]:
  for kind in ['last','week','mean3']:
   p=common.baseline(d,oo,h,kind);rows.append(dict(horizon_days=h,model=kind,**ev.evaluate(common,d,oo,p,h)))
 export_csv(work/'reproduction/results/google_baselines.csv',rows)
 assert abs(rows[0]['wape']-3.199450324772723)<1e-11
 assert abs(rows[-1]['wape']-3.2119187773783344)<1e-11
 return rows

def member(work,name,phase):
 import numpy as np,joblib,torch
 data,d,canon,cfg=setup(work)
 from power_forecast.hdsf_experiment import HDSFCollector,HDSF_MEMBERS
 from power_forecast.attainment_local import fit_local24
 from power_forecast.hdsf_patchtst import checkpoint,model_class
 # Exactly the original source CPU path, with the recorded batch8 memory fallback.
 class Collector(HDSFCollector):
  def _predict_member(self,n,task,origin):
   if n!='patchtstfm_long':return super()._predict_member(n,task,origin)
   if self.patch_predictor is None:self.patch_predictor=model_class().from_pretrained(str(checkpoint(self.cache_dir)),local_files_only=True).to('cpu').eval()
   h=1 if task=='24h' else 7;ctx=np.ascontiguousarray(data.target[:,max(0,origin-21):origin].reshape(57,-1),dtype=np.float32);out=[]
   with torch.no_grad():
    for j in range(0,57,8):
     x=torch.tensor(ctx[j:j+8]);z=self.patch_predictor(past_values=x,past_observed_mask=torch.ones_like(x,dtype=torch.bool),prediction_length=h*96)
     out.append(z.prediction_outputs[...,0].cpu().numpy().astype(np.float32))
   return np.concatenate(out).astype(float)
 tasks=[t for t,mm in HDSF_MEMBERS.items() if name in mm]
 if not tasks:raise ValueError('Unknown HDSF member: '+name)
 col=Collector(data,tasks,cache_dir=work/'experiment/round44/model-cache',cache_predictions=False,batch_size=16,jobs=8,device='cpu')
 col.members={t:(name,) for t in tasks}
 if name=='local24':
  cut=14 if phase=='dev' else 22;local=fit_local24(data,cut,cache_dir=None,jobs=8)
  if phase=='dev':col.development_local=local
  else:col.frozen_local=local
  joblib.dump(local,work/'experiment/round44/results'/f'local_cut{cut}.joblib')
 frames=col.collect(development=phase=='dev');out={}
 for task,frame in frames.items():
  h=1 if task=='24h' else 7;origins=sorted(map(int,frame.origin_day.unique()));a=np.zeros((57,len(origins),h*96))
  for j,o in enumerate(origins):
   f=frame.loc[frame.origin_day==o];step=(f.target_day.to_numpy(int)-o-1)*96+f.slot.to_numpy(int);a[f.series_id.to_numpy(int),j,step]=f[name]
  out[name+f'_H{h}']=a[canon]
 dest=work/'reproduction'/phase;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/(name+'.npz'),**out)
 save(dest/(name+'.json'),{'source':'actual_model_inference_or_local_fit','phase':phase,'member':name,'npz_sha256':sha(dest/(name+'.npz'))})

def weights(cfg,h):
 import numpy as np
 c=cfg['hdsf']['24h' if h==1 else '7d'];n=str(c['segments'][0]);a=np.asarray(c['component_a'][n]);b=np.asarray(c['component_b'][n])
 return c['methods'],(1-c['mix'])*np.column_stack([a,np.zeros(len(a))])+c['mix']*b

def base(work,phase,h):
 import numpy as np
 cfg=read(REPO/'repro/google_frozen.json');names,w=weights(cfg,h)
 bank={}
 for n in names:
  with np.load(work/'reproduction'/phase/(n+'.npz')) as z:bank[n]=z[n+f'_H{h}'].copy()
 p=np.stack([bank[n] for n in names],-1);idx=np.minimum(np.arange(h*96)//48,len(w)-1)
 return np.einsum('solm,lm->sol',p,w[idx],optimize=True)

def train_or_infer(work,family,recipe,phase,state=None):
 import numpy as np,torch
 data,d,canon,cfg=setup(work)
 import hdsf_foundation_stage as stage
 from hdsf_foundation_adapters import Adapter,load_loss
 stage.data=data
 cut=14 if phase=='dev' else 22
 stage.data=data;stage.canon=canon
 is47=recipe=='r47curve';horizons=[1] if is47 else [7]
 target=work/'reproduction'/phase;target.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(20261012);adapter=Adapter(work,family,'cuda');adapter.set_scope('tail')
 def predict():
  arrays={}
  members=['patchtstfm_long'] if family=='patchtst' else ['timesfm30_long','timesfm30_uni'] if family=='timesfm3' else ['toto2_joint']
  for h in horizons:
   oo=([14,16,18,20] if h==1 else [14,15]) if phase=='dev' else (list(range(22,31)) if h==1 else [22,23,24])
   for n in members:arrays[n+f'_H{h}']=np.stack([stage.predict_member(adapter,n,o,h)[canon] for o in oo],1)
  return arrays
 controls=predict();initial=adapter.trainable_state()
 if state:
  loaded=torch.load(state,map_location='cpu',weights_only=True);adapter.load_trainable_state(loaded.get('state',loaded))
  training='provided_local_checkpoint_not_retrained_in_this_run'
 else:
  steps=160 if is47 else 40;lr=1.5e-6 if is47 else 3e-6;schedule=160 if is47 else 80
  if is47:
   import hdsf_shape_training as tr
   tr.P=read(REPO/'experiment/round47/docs/protocol.json');seq=tr.batches(cut,family);lossfn=lambda q,p,y,m,t:tr.load_loss(adapter,q,p,y,m,t,'curve')
  else:seq=stage.sample_batches(cut,family,80)[:steps];lossfn=lambda q,p,y,m,t:load_loss(adapter,q,p,y,m,t,teacher_weight=.1)
  refs={n:v.to(adapter.device) for n,v in initial.items()};params=[p for p in adapter.model.parameters() if p.requires_grad]
  opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01);flat=data.target[:,:cut].reshape(57,-1);valid=data.target_valid[:,:cut].reshape(57,-1);teachers=[]
  for b in seq:
   ids,o,ctx,h=b['series_ids'],b['origin_tick'],b['context_ticks'],b['horizon_ticks'];assert o+h<=cut*96
   with torch.no_grad():_,p=adapter.tensor_forecast(flat[ids,o-ctx:o],h,b.get('mode','uni'),False)
   teachers.append(p.detach().float().cpu())
  history=[]
  for i,b in enumerate(seq,1):
   ids,o,ctx,h=b['series_ids'],b['origin_tick'],b['context_ticks'],b['horizon_ticks'];q,p=adapter.tensor_forecast(flat[ids,o-ctx:o],h,b.get('mode','uni'),True)
   loss,info=lossfn(q.float(),p.float(),flat[ids,o:o+h],valid[ids,o:o+h],teachers[i-1].to(adapter.device))
   prox=torch.stack([(dict(adapter.model.named_parameters())[n]-v).square().mean()/v.square().mean().clamp_min(1e-4) for n,v in refs.items()]).mean()
   loss=loss+.05*prox;assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
   for group in opt.param_groups:group['lr']=lr*(.2+.8*(1-i/schedule))
   history.append({'step':i,**info})
   if i%10==0:print('REAL_GRADIENT',family,recipe,phase,i,flush=True)
  filename=work/'experiment'/('round47' if is47 else 'round46')/'models'/(f'patchtst_curve_cut{cut}.pt' if is47 else f'{family}_tail_s40_cut{cut}.pt')
  torch.save(adapter.trainable_state(),filename);save(filename.with_suffix('.reproduction.json'),{'actual_optimizer_steps':steps,'cutoff':cut,'seed':20261012,'no_test_reselection':True,'history':history})
  training='actual_local_gradient_updates'
 updated=predict();np.savez_compressed(target/(family+'_'+recipe+'_delta.npz'),**{k:updated[k]-controls[k] for k in updated})
 save(target/(family+'_'+recipe+'_delta.json'),{'training':training,'family':family,'recipe':recipe,'phase':phase,'state_sha256':adapter.hash_trainable(),'matched_native_original_control':True})

def bounded(prior,delta):
 import numpy as np
 shape=prior.shape;p=prior.reshape(57,-1,96);v=delta.reshape(57,-1,96);raw=p+v;wanted=np.clip(p.mean(-1,keepdims=True)+v.mean(-1,keepdims=True),0,1)
 if (raw<0).any() or (raw>1).any():
  left=np.full_like(wanted,-2.);right=np.full_like(wanted,2.)
  for _ in range(60):
   mid=(left+right)/2;means=np.clip(raw+mid,0,1).mean(-1,keepdims=True);left=np.where(means<wanted,mid,left);right=np.where(means>=wanted,mid,right)
  raw=np.clip(raw+(left+right)/2,0,1)
 return raw.reshape(shape)

def corrected(work,phase,h):
 import numpy as np
 data,d,canon,cfg=setup(work);prior=base(work,phase,h);names,w=weights(cfg,h)
 if h==1:
  with np.load(work/'reproduction'/phase/'patchtst_r47curve_delta.npz') as z:delta=z['patchtstfm_long_H1']
  mu=delta.reshape(57,-1,1,96).mean(-1,keepdims=True);pd=prior.reshape(57,-1,1,96);scale=pd.mean(-1,keepdims=True)
  groups=list(zip(data.meta.iloc[canon].cell.astype(str),data.meta.iloc[canon].is_mvpp.astype(bool)));out=mu.copy();rel=mu/np.maximum(scale,1e-8)
  for g in dict.fromkeys(groups):
   ix=np.array([i for i,v in enumerate(groups) if v==g])
   if len(ix)>=3:out[ix]=.5*mu[ix]+.5*np.median(rel[ix],axis=0)*scale[ix]
  delta=np.broadcast_to(out,pd.shape).reshape(prior.shape)
 else:
  delta=np.zeros_like(prior)
  for family,mm in [('timesfm3',['timesfm30_long','timesfm30_uni']),('toto2',['toto2_joint']),('patchtst',['patchtstfm_long'])]:
   with np.load(work/'reproduction'/phase/(family+'_r46_delta.npz')) as z:
    for n in mm:delta+=z[n+'_H7']*w[np.arange(672)//48,names.index(n)][None,None,:]
  vd=delta.reshape(57,-1,96);delta=(vd-vd.mean(-1,keepdims=True)).reshape(prior.shape)
 return bounded(prior,delta)

def score(work):
 import numpy as np
 data,d,canon,cfg=setup(work);common,ev=evaluator(work);iv=module('repro_intervals',work/'experiment/round28/code/source_metrics.py');rows=[]
 for h,oo,dev_o in [(1,list(range(22,31)),[14,16,18,20]),(7,[22,23,24],[14,15])]:
  p=corrected(work,'test',h);dp=corrected(work,'dev',h);metrics=ev.evaluate(common,d,oo,p,h);y,m=common.truth(d,oo,h);dy,dm=common.truth(d,dev_o,h)
  low=np.array([np.quantile((dy-dp)[i][dm[i]],.1) for i in range(57)]);high=np.array([np.quantile((dy-dp)[i][dm[i]],.9) for i in range(57)])
  lo=np.clip(p+low[:,None,None],0,1);hi=np.clip(p+high[:,None,None],0,1);interval=iv.interval_metrics(y,p,m,lo,hi)
  yd,pd,md=y.reshape(57,-1,96),p.reshape(57,-1,96),m.reshape(57,-1,96);ok=md.any(-1)
  peak_under=float(100*(np.where(md,pd,-np.inf).max(-1)[ok]<np.where(md,yd,-np.inf).max(-1)[ok]).mean())
  expected=cfg['robust'][str(h)]['metrics'];differences={k:metrics[k]-expected[k] for k in metrics if isinstance(metrics[k],(float,int)) and isinstance(expected.get(k),(float,int))}
  rows.append(dict(horizon_days=h,model='Google_R48_reconstruction',**metrics,peak_underestimate_percent=peak_under,**{'interval80_'+k:v for k,v in interval.items()}))
  np.savez_compressed(work/'reproduction/results'/f'google_R48_H{h}.npz',prediction=p,truth=y,valid=m,lower80=lo,upper80=hi)
  save(work/'reproduction/results'/f'google_R48_H{h}_comparison.json',{'expected_source':'original_published_frozen_R48','metric_differences':differences,'exact_bitwise_match_claimed':False,'previous_test_exposure':True})
 export_csv(work/'reproduction/results/google_R48_all_metrics.csv',rows);return rows
