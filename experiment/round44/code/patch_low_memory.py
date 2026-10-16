from pathlib import Path
import json,hashlib,sys
import numpy as np
BATCH=8
def load_predictor(name,*,cache_dir,device='cpu',**options):
 assert name=='patchtstfm_long' and device=='cpu'
 import torch
 from power_forecast import hdsf_patchtst as src
 from power_forecast.foundation import validate_forecast
 torch.set_num_threads(src.THREADS);snapshot=src.checkpoint(cache_dir);assert snapshot
 model=src.model_class().from_pretrained(str(snapshot),local_files_only=True).to('cpu').eval();R=Path(cache_dir).parent;Q=R/'results';cache=Q/'patch_microbatch';cache.mkdir(exist_ok=True)
 identity=hashlib.sha256(Path(__file__).read_bytes()+ (Q/'official_models_downloads.json').read_bytes()+(R/'docs/runtime_versions.json').read_bytes()).hexdigest()
 def forward(context,horizon,batch):
  chunks=[]
  with torch.no_grad():
   for j in range(0,len(context),batch):
    values=torch.tensor(np.ascontiguousarray(context[j:j+batch],dtype=np.float32));out=model(past_values=values,past_observed_mask=torch.ones_like(values,dtype=torch.bool),prediction_length=horizon);chunks.append(out.prediction_outputs[...,0].cpu().numpy().astype(np.float32));del values,out
  return validate_forecast(np.concatenate(chunks),len(context),horizon)
 class Predictor:
  checked=False
  def predict(self,context,horizon):
   context=np.ascontiguousarray(context,dtype=np.float32)
   if not self.checked:
    # The actual model is in eval mode; no feature aggregates across examples.
    group=context[:8];reference=forward(group,horizon,8);split=forward(group,horizon,4);diff=float(abs(reference-split).max());assert diff<2e-6,diff
    check=dict(reference_batch=8,split_batch=4,actual_context_points=context.shape[1],horizon=horizon,maximum_absolute_difference=diff,passed=True,full57_reference_not_completed_due_memory=True,model_unmodified=True,prediction_dtype='float32',adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache_identity=identity)
    (cache/f'parity_ctx{context.shape[1]}_H{horizon}.json').write_text(json.dumps(check,indent=2),encoding='utf-8');print('PatchTST real-model batch8 vs batch4 parity:',diff,flush=True);self.checked=True
   key=hashlib.sha256(identity.encode()+str([context.shape,horizon,BATCH]).encode()+context.tobytes()).hexdigest();p=cache/(key+'.npy')
   if p.exists():return validate_forecast(np.load(p,allow_pickle=False),len(context),horizon)
   pred=forward(context,horizon,BATCH);tmp=cache/(key+'.tmp')
   with tmp.open('wb') as f:np.save(f,pred,allow_pickle=False)
   tmp.replace(p);return pred
 return Predictor()
