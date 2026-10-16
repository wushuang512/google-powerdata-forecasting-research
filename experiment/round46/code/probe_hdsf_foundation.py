from pathlib import Path
import argparse, datetime, json, traceback, sys
sys.dont_write_bytecode=True
import numpy as np,torch
from hdsf_foundation_adapters import Adapter,load_training_data,load_loss
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('family');ap.add_argument('--device',default='cuda');args=ap.parse_args()
R=Path(args.root)/'experiment/round46';Q=R/'results/preflight';Q.mkdir(parents=True,exist_ok=True)
try:
 data=load_training_data(args.root);adapter=Adapter(args.root,args.family,args.device)
 x=data.target[:2,:7].reshape(2,-1);y=data.target[:2,7].reshape(2,-1);m=data.target_valid[:2,7].reshape(2,-1)
 mode='joint' if args.family in ['timesfm3','toto2'] else'uni'
 with torch.no_grad():tq,tp=adapter.tensor_forecast(x,96,mode,False)
 before=adapter.hash_trainable();q,p=adapter.tensor_forecast(x,96,mode,True)
 assert p.requires_grad and q.requires_grad
 loss,info=load_loss(adapter,q,p,y,m,tp);opt=torch.optim.AdamW([v for v in adapter.model.parameters() if v.requires_grad],lr=1e-5,weight_decay=.01)
 loss.backward();grads={n:float(v.grad.detach().abs().max()) for n,v in adapter.model.named_parameters() if v.requires_grad and v.grad is not None}
 assert grads and max(grads.values())>0 and all(np.isfinite(v) for v in grads.values())
 torch.nn.utils.clip_grad_norm_([v for v in adapter.model.parameters() if v.requires_grad],1);opt.step();after=adapter.hash_trainable();assert before!=after
 tail=None
 try:tail=adapter.set_scope('tail')
 except Exception as e:tail=repr(e)
 result={'family':args.family,'stage':'one_step_training_interface_probe_not_completed_finetune',
         'all_passed':True,'actual_GPU':next(adapter.model.parameters()).is_cuda,'head_parameters':sum(v.numel() for n,v in adapter.model.named_parameters() if n in grads),
         'head_tensor_names':list(grads),'tail_scope':tail,'point_requires_grad':p.requires_grad,'quantiles_require_grad':q.requires_grad,
         'nonzero_gradient_max':max(grads.values()),'before_sha256':before,'after_sha256':after,
         'point_loss':info,'output_shape':list(q.shape),'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,
         'no_TEST_data_used':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
except BaseException as e:
 result={'family':args.family,'stage':'one_step_training_interface_probe','all_passed':False,'error':repr(e),'traceback':traceback.format_exc(),'no_TEST_data_used':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(Q/(args.family+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True),flush=True)
if not result['all_passed']:sys.exit(1)
