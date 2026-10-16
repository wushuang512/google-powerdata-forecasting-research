from pathlib import Path
import sys,numpy as np,json,hashlib,datetime
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round35';Q=R/'results';sys.path[:0]=[str(R/'runtime'),str(R/'code'),str(R/'code/official_uni2ts/src'),str(B/'code')]
import torch,common
from moirai_native_adapter import build,load_forecast_class,repair_joint_initial_write
assert torch.cuda.is_available();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;d=common.load();x=torch.tensor(d['y'][:5,7:14].reshape(5,-1).T[None],dtype=torch.float32,device='cuda');v=torch.tensor(d['valid'][:5,7:14].reshape(5,-1).T[None],device='cuda');pad=torch.zeros((1,672),dtype=torch.bool,device='cuda')
m=build(R,672,64,5).to('cuda');Cls=load_forecast_class(R,type(m.module));original=Cls(64,5,0,0,672,module=m.module).eval()
with torch.inference_mode():a=m(x,v,pad);b=original(x,v,pad)
assert a.shape==(1,9,64,5) and torch.equal(a,b)
Cls=repair_joint_initial_write(Cls,R);repaired=Cls(96,5,0,0,672,module=m.module).eval()
with torch.inference_mode():c=repaired(x,v,pad)
assert c.shape==(1,9,96,5) and torch.isfinite(c).all()
z=json.loads((Q/'adapter_verification.json').read_text(encoding='utf-8'));z.update(adapter_sha256=hashlib.sha256((R/'code/moirai_native_adapter.py').read_bytes()).hexdigest(),joint64_unmodified_official_max_difference=float((a-b).abs().max()),joint96_recursive_shape=list(c.shape),joint_recursive_finite=True,joint_fix_scope='Custom adapter fixes first recursive write flattened token count only for target_dim>1. Upstream original source unchanged; multivariate extension remains exploratory, not untouched official long-horizon inference.',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(Q/'adapter_verification.json').write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding='utf-8');print('Joint64 matches upstream exactly; repaired joint96 finite andcorrectshape',flush=True)
