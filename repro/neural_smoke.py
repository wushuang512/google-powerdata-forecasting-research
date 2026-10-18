"""One real gradient update; explicitly not a full fit or accuracy measurement."""
from .support import save
from .google import setup
def run(work,family):
 import torch,numpy as np
 setup(work)
 import hdsf_foundation_stage as stage
 from hdsf_foundation_adapters import Adapter,load_loss
 data,_,_,_=setup(work);stage.data=data;torch.manual_seed(20261012)
 if not torch.cuda.is_available():raise RuntimeError('This selected training recipe requires CUDA; CPU HDSF members use a separate environment')
 a=Adapter(work,family,'cuda');a.set_scope('tail');before=a.hash_trainable();b=stage.sample_batches(22,family,80)[0]
 ids,o,ctx,h=b['series_ids'],b['origin_tick'],b['context_ticks'],b['horizon_ticks'];assert o+h<=22*96
 flat=data.target[:,:22].reshape(57,-1);mask=data.target_valid[:,:22].reshape(57,-1);x=flat[ids,o-ctx:o]
 with torch.no_grad():_,teacher=a.tensor_forecast(x,h,b['mode'],False)
 q,p=a.tensor_forecast(x,h,b['mode'],True)
 loss,info=load_loss(a,q.float(),p.float(),flat[ids,o:o+h],mask[ids,o:o+h],teacher.detach(),teacher_weight=.1)
 assert loss.requires_grad and torch.isfinite(loss)
 params=[p for p in a.model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=3e-6,weight_decay=.01);opt.zero_grad(set_to_none=True);loss.backward()
 gradients=[p.grad for p in params if p.grad is not None]
 assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().max()>0 for g in gradients)
 torch.nn.utils.clip_grad_norm_(params,1);opt.step();after=a.hash_trainable();assert after!=before
 result={'family':family,'actual_cuda':True,'torch':torch.__version__,'real_optimizer_steps':1,'scope':'head and final encoder block','trainable_tensor_hash_changed':True,'finite_nonzero_gradients':True,'batch_context_points':ctx,'forecast_points':h,'last_label_tick':o+h-1,'train_cutoff_tick':22*96,'completed_full_training':False,'accuracy_evaluated':False,'official_weight_files_modified':False}
 save(work/'reproduction/results'/f'neural_smoke_{family}.json',result);print(result)
