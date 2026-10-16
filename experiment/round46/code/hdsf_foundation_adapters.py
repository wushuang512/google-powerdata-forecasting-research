"""Round46 differentiable adapters; native checkpoints, trainable model parameters.

No external labels, data downloads, or edits to vendor source.
"""
from pathlib import Path
from contextlib import nullcontext
import dataclasses, gc, hashlib, json, math, os, sys
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
sys.dont_write_bytecode=True
import numpy as np
import torch

FAMILIES={
 'chronos2':['chronos2_long'], 'tirex2':['tirex2'], 'timesfm25':['timesfm25'],
 'timesfm3':['timesfm30','timesfm30_long','timesfm30_uni'],
 'toto2':['toto2_joint','toto2_single'], 'patchtst':['patchtstfm_long'],
}
class Adapter:
 def __init__(self,root,family,device='cuda'):
  self.root=Path(root);self.family=family;self.device=torch.device(device);self.wrapper=None
  B=self.root/'experiment';A=B/'round44';self.old=A
  torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  from safetensors.torch import load_file
  if family=='chronos2':
   from chronos import Chronos2Pipeline
   self.wrapper=Chronos2Pipeline.from_pretrained(str(B/'models/chronos2-base'),device_map=device,dtype=torch.float32,local_files_only=True)
   self.model=self.wrapper.model;self.levels=list(self.model.chronos_config.quantiles)
  elif family=='timesfm25':
   import timesfm
   with torch.device(device):self.wrapper=timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
   self.model=self.wrapper.model
   self.model.load_state_dict(load_file(str(B/'models/timesfm25-base/model.safetensors'),device=device),strict=True,assign=True)
   self.model.device=self.device;self.model.device_count=1;self.levels=[i/10 for i in range(1,10)]
  elif family=='timesfm3':
   from timesfm3 import TimesFM3Forecaster
   with torch.device(device):
    self.wrapper=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device=device,per_core_batch_size=4,local_files_only=True)
   self.model=self.wrapper.model;self.levels=list(self.model.quantiles)
  elif family=='toto2':
   from toto2 import Toto2Model
   from toto2.model import Toto2ModelConfig
   self.path=A/'model-cache/models--Datadog--Toto-2.0-313m/snapshots/a7bab288f5e95f8606f8306f86659357e1c001ef'
   raw=json.loads((self.path/'config.json').read_text());known={f.name for f in dataclasses.fields(Toto2ModelConfig)}
   with torch.device(device):self.model=Toto2Model(Toto2ModelConfig(**{k:v for k,v in raw.items() if k in known}))
   self.model.load_state_dict(load_file(str(self.path/'model.safetensors'),device=device),strict=True,assign=True)
   self.levels=list(self.model.output_head.knots)
  elif family=='patchtst':
   from power_forecast.hdsf_patchtst import model_class
   PatchTSTFMForPrediction=model_class()
   self.path=A/'model-cache/models--ibm-granite--granite-timeseries-patchtst-fm-r2/snapshots/b125275f9204d37cbb81fe47b9cf5e08a521e829'
   self.model=PatchTSTFMForPrediction.from_pretrained(str(self.path),local_files_only=True,torch_dtype=torch.float32,device_map=device,low_cpu_mem_usage=True)
   self.levels=list(self.model.config.quantile_levels)
  elif family=='tirex2':
   from tirex2 import load_model
   # Native PyTorch recurrent kernels avoid the Windows CUDA-extension compiler.
   self.wrapper=load_model(B/'models/tirex2-base',device='cpu',use_flex_attention=False,compile=False)
   self.model=self.wrapper.model.to(self.device);self.levels=[float(x) for x in self.model.quantiles.detach().cpu()]
  else:raise ValueError(family)
  self.model.to(self.device).eval();self.median_index=int(np.argmin(abs(np.asarray(self.levels)-.5)))
  self.compiled_eager=torch.compiler.set_stance('force_eager') if family=='tirex2' else nullcontext()
  self.set_scope('head')

 def set_scope(self,scope):
  heads={'chronos2':['output_patch_embedding.'],'tirex2':['output_patch_embedding.','stack_out_norm.'],
         'timesfm25':['output_projection_point.','output_projection_quantiles.'],
         'timesfm3':['output_head.'],'toto2':['output_head.'],
         'patchtst':['backbone.out_layer.','backbone.prehead_norm.']}[self.family]
  names=list(dict(self.model.named_parameters()));tail=[]
  if scope=='tail':
   patterns={'chronos2':'encoder.block.11.','tirex2':'stack.11.',
             'timesfm25':'stacked_xf.19.','timesfm3':'transformer_stack.layers.19.',
             'toto2':None,'patchtst':None}
   pat=patterns[self.family]
   if pat:tail=[pat]
   else:
    # Resolve official final block from actual parameter paths, not a guessed module.
    import re
    candidates=[]
    for n in names:
     for match in re.finditer(r'(.*(?:layers|blocks)\.)(\d+)\.',n):
      candidates.append((int(match.group(2)),match.group(1)))
    if candidates:
     idx,prefix=max(candidates);tail=[prefix+str(idx)+'.']
  self.train_names=[]
  for name,param in self.model.named_parameters():
   enabled=any(name.startswith(h) for h in heads+tail);param.requires_grad_(enabled)
   if enabled:self.train_names.append(name)
  assert self.train_names,('No trainable official model parameters',self.family,names[-10:])
  if scope=='tail':assert any(any(n.startswith(t) for t in tail) for n in self.train_names),('Final block scope unresolved',self.family)
  self.scope=scope;return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

 def _tf25_grad(self,x,H):
  from timesfm.torch import util
  from timesfm.torch.util import revin
  # Follow native input normalization and running per-patch normalization.
  external_mu=x.mean(-1,keepdim=True);external_sigma=x.std(-1,keepdim=True).clamp_min(1e-6)
  xx=revin(x,external_mu,external_sigma,reverse=False)
  patched=xx.reshape(len(x),-1,32);mask=torch.zeros_like(patched,dtype=torch.bool)
  n=torch.zeros(len(x),device=x.device);mu=n.clone();sigma=n.clone();mus=[];sigmas=[]
  for i in range(patched.shape[1]):
   (n,mu,sigma),_=util.update_running_stats(n,mu,sigma,patched[:,i],mask[:,i]);mus.append(mu);sigmas.append(sigma)
  mu=torch.stack(mus,1);sigma=torch.stack(sigmas,1)
  normed=revin(patched,mu,sigma,reverse=False)
  (_,_,point,spread),_=self.model(normed,mask)
  point=revin(point,mu,sigma,reverse=True).reshape(len(x),-1,128,10)[:,-1,:H]
  spread=revin(spread,mu,sigma,reverse=True).reshape(len(x),-1,1024,10)[:,-1,:H]
  quantiles=torch.stack([spread[:,:,j]-spread[:,:,5]+point[:,:,5] for j in range(1,10)],-1)
  quantiles=revin(quantiles,external_mu[:,:,None],external_sigma[:,:,None],reverse=True)
  return quantiles,quantiles[:,:,4]

 def tensor_forecast(self,context,H,mode='uni',grad=True):
  x=torch.as_tensor(np.ascontiguousarray(context,dtype=np.float32),device=self.device)
  family=self.family
  with (nullcontext() if grad else torch.no_grad()):
   if family=='chronos2':
    result=self.model(context=x,group_ids=torch.arange(len(x),device=x.device),num_output_patches=math.ceil(H/self.model.chronos_config.output_patch_size))
    q=result.quantile_preds[:,:,:H].transpose(1,2);p=q[:,:,self.median_index]
   elif family=='timesfm25':q,p=self._tf25_grad(x,H)
   elif family=='timesfm3':
    from timesfm3.torch.timesfm3_forecaster import linear_interpolation
    values=linear_interpolation(x.detach().cpu().numpy())
    values=torch.tensor(values,device=x.device,dtype=torch.float32)
    target=values[:,None] if mode=='uni' else values[None]
    if grad:q=type(self.model).decode.__wrapped__(self.model,target,horizon=H)
    else:q=self.model.decode(target,horizon=H)
    q=q[:,0] if mode=='uni' else q[0];p=q[:,:,self.median_index]
   elif family=='toto2':
    target=x[:,None] if mode=='uni' else x[None]
    inp={'target':target,'target_mask':torch.ones_like(target,dtype=torch.bool),
         'series_ids':torch.zeros(target.shape[:-1],device=x.device,dtype=torch.long)}
    if grad:q=type(self.model).forecast.__wrapped__(self.model,inp,horizon=H,decode_block_size=None,has_missing_values=False)
    else:q=self.model.forecast(inp,horizon=H,decode_block_size=None,has_missing_values=False)
    q=q[:, :,0] if mode=='uni' else q[:,0];q=q.permute(1,2,0);p=q[:,:,self.median_index]
   elif family=='patchtst':
    out=self.model(past_values=x,past_observed_mask=torch.ones_like(x,dtype=torch.bool),prediction_length=H)
    q=out.quantile_outputs[...,0].transpose(1,2)
    levels=torch.tensor(self.levels,device=q.device,dtype=q.dtype)
    p=torch.trapz(q,levels,dim=-1)+q[:,:,0]*levels[0]+q[:,:,-1]*(1-levels[-1])
   elif family=='tirex2':
    from tirex2 import TimeseriesType
    inputs=[TimeseriesType(target=row[None],past_covariates=None,future_covariates=None) for row in x]
    with torch.compiler.set_stance('force_eager'):
     outputs=self.model._predict_once(inputs,H,preserve_grad=grad,tta_diff=True,pad_context=True)
    q=torch.stack([a[0] for a in outputs]).transpose(1,2);p=q[:,:,self.median_index]
   else:raise ValueError(family)
  assert q.shape==(len(x),H,len(self.levels)),(family,q.shape)
  assert p.shape==(len(x),H) and torch.isfinite(q).all(),family
  return q,p

 def predict(self,context,H,mode='uni'):
  context=np.ascontiguousarray(context,dtype=np.float32)
  # Use the original inference APIs for exact matched-before/after delta controls.
  with torch.inference_mode():
   if self.family=='timesfm25':
    import timesfm
    self.wrapper.compile(timesfm.ForecastConfig(max_context=context.shape[1],max_horizon=H,
       normalize_inputs=True,per_core_batch_size=4,use_continuous_quantile_head=True,
       force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True))
    p,_=self.wrapper.forecast(horizon=H,inputs=[r.copy() for r in context]);return np.asarray(p,dtype=float)
   if self.family=='timesfm3':
    flags=dict(return_quantiles=False,use_symmetric_averaging=False,make_positive=False,use_znorm=False,padding_mode='none')
    if mode=='joint':return np.asarray(self.wrapper.predict(context=context,horizon=H,**flags).forecast,dtype=float)
    out=list(self.wrapper.predict_batch(contexts=[r.copy() for r in context],horizon=H,**flags))
    return np.stack([o.forecast for o in out]).astype(float)
   if self.family=='tirex2':
    from tirex2 import TimeseriesType
    parts=[]
    for start in range(0,len(context),4):
     inputs=[TimeseriesType(target=torch.tensor(r,device=self.device)[None],past_covariates=None,future_covariates=None) for r in context[start:start+4]]
     with torch.compiler.set_stance('force_eager'):out=self.model.predict(inputs,H)
     parts.extend([o[0,self.median_index].detach().cpu().numpy() for o in out])
    return np.array(parts,dtype=float)
   q,p=self.tensor_forecast(context,H,mode,grad=False)
   return p.detach().float().cpu().numpy().astype(float)

 def trainable_state(self):return {n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}
 def load_trainable_state(self,state):
  params=dict(self.model.named_parameters())
  with torch.no_grad():
   for n,v in state.items():params[n].copy_(v.to(device=params[n].device,dtype=params[n].dtype))
 def hash_trainable(self):
  digest=hashlib.sha256()
  for n,p in self.model.named_parameters():
   if p.requires_grad:digest.update(n.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()

def load_training_data(root):
 A=Path(root)/'experiment/round44'
 sys.path.insert(0,str(A/'reference/PowerData_Prediction'))
 import joblib
 return joblib.load(A/'results/source_dataset.joblib')

def load_loss(adapter,q,p,target,mask,teacher,teacher_weight=.05):
 y=torch.as_tensor(np.ascontiguousarray(target,dtype=np.float32),device=p.device)
 m=torch.as_tensor(np.ascontiguousarray(mask),device=p.device);scale=(abs(y)*m).sum(-1).clamp_min(.05)
 point=(abs(p-y)*m).sum(-1)/scale
 dayp=p.reshape(len(p),-1,96);dayy=y.reshape(len(y),-1,96);daym=m.reshape(len(y),-1,96);complete=daym.all(-1)
 day=((abs(dayp.mean(-1)-dayy.mean(-1))*complete).sum(-1)/(abs(dayy.mean(-1))*complete).sum(-1).clamp_min(.05)).mean()
 peak=(abs(dayp.max(-1).values-dayy.max(-1).values)/dayy.max(-1).values.clamp_min(.05)*complete).sum()/complete.sum().clamp_min(1)
 levels=torch.as_tensor(adapter.levels,device=p.device,dtype=p.dtype);error=y[:,:,None]-q
 pin=(torch.maximum(levels*error,(levels-1)*error)*m[:,:,None]).sum((1,2))/scale/len(levels)
 slope=(abs(torch.diff(p,dim=1)-torch.diff(y,dim=1))*(m[:,1:]&m[:,:-1])).sum(-1)/scale
 anchor=((p-teacher)**2).mean()/((teacher.abs().mean()).clamp_min(.05)**2)
 bounds=(torch.relu(-p)+torch.relu(p-1)).mean()
 loss=point.mean()+.25*day+.1*peak+.05*pin.mean()+.05*slope.mean()+teacher_weight*anchor+.1*bounds
 return loss,{'point':float(point.mean().detach()),'daily':float(day.detach()),'peak':float(peak.detach()),'pinball':float(pin.mean().detach()),'loss':float(loss.detach())}
