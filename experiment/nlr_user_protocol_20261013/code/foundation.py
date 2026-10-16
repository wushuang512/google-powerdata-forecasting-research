import argparse,gc,os,sys,time,hashlib
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
import numpy as np,torch
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('family',choices=['timesfm3','toto2','moirai2']);ap.add_argument('phase',choices=['validation','test']);a=ap.parse_args();R=Path(a.root);F,cfg,d,q=load(R)
if a.phase=='test':assert sha(F/'results/selection.json')==(F/'results/selection.sha256').read_text().strip()
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available();device='cuda'
sys.path[:0]=[str(R/'experiment/round46/code'),str(R/'experiment/round44/reference/PowerData_Prediction')]
for p in ['experiment/round44/.venv/Lib/site-packages','experiment/round35/runtime','experiment/round35/code','experiment/round35/code/official_uni2ts/src','experiment/round29/cpu_runtime']:sys.path.append(str(R/p))
if a.family=='moirai2':
    from moirai_native_adapter import build
    model=build(R/'experiment/round35',1344,96,1).to(device).eval();weight=R/'experiment/round35/models/moirai2/model.safetensors'
    def predict(ctx,H):
        x=torch.as_tensor(np.ascontiguousarray(ctx,dtype=np.float32),device=device);model.hparams.context_length=1344;model.hparams.prediction_length=H
        obs=torch.ones_like(x[:,:,None],dtype=torch.bool);pad=torch.zeros(x.shape,dtype=torch.bool,device=device)
        with torch.inference_mode():p=model(x[:,:,None],obs,pad)[:,4,:]
        assert p.shape==(len(ctx),H);return p.float().cpu().numpy().astype(float)
else:
    from hdsf_foundation_adapters import Adapter
    adapter=Adapter(R,a.family,device);adapter.set_scope('head');predict=lambda ctx,H:adapter.predict(ctx,H,'uni')
    weight=R/'experiment/models/timesfm3-base/model.safetensors' if a.family=='timesfm3' else R/'experiment/round44/model-cache/models--Datadog--Toto-2.0-313m/snapshots/a7bab288f5e95f8606f8306f86659357e1c001ef/model.safetensors'
cut=q['split']['train'][1] if a.phase=='validation' else q['split']['validation'][1];known=d['y'][:cut];scale=float(np.quantile(known[np.isfinite(known)],.95)/.8);assert scale>0
weight_sha=sha(weight);save(F/'models'/f'{a.family}_{a.phase}_scale.json',dict(fitted_to='train' if a.phase=='validation' else'train plus validation',end_exclusive=cut,scale=scale,method='training-only95percentile divided by0.8, not capacity',neural_parameters_updated=False,weight_file=str(weight),weight_sha256=weight_sha))
for H in [1,7]:
    oo=origins(q,a.phase,H);parts=[];digests=[]
    for k,o in enumerate(oo):
        ctx=np.ascontiguousarray(d['y'][o-14:o].reshape(1,1344)/scale,dtype=np.float32);digest=hashlib.sha256(ctx.tobytes()).hexdigest();cp=F/'cache'/f'{a.phase}_{a.family}_H{H}_o{o}.npz'
        identity=hashlib.sha256((sha(F/'docs/protocol.json')+sha(Path(__file__))+weight_sha+str(scale)+digest).encode()).hexdigest()
        if cp.exists():
            with np.load(cp) as z:assert str(z['fingerprint'])==identity;point=z['p'].copy()
        else:
            point=np.maximum(0,predict(ctx,H*96)[0]*scale);assert point.shape==(H*96,) and np.isfinite(point).all();np_save(cp,p=point,fingerprint=identity,input_digest=digest,origin=o,horizon=H)
        parts.append(point);digests.append(dict(origin=o,input_last_interval=o*96-1,input_digest=digest));
        if k%10==0:
            save(F/'results/foundation_status.json',dict(state='running',family=a.family,phase=a.phase,horizon_days=H,complete=k+1,total=len(oo),pid=os.getpid(),utc=utc()));print('FOUNDATION',a.family,a.phase,H,k+1,len(oo),flush=True)
    save_forecasts(F,cfg,d,q,a.phase,H,a.family,np.stack(parts),dict(history_points=1344,neural_parameters_updated=False,weight_sha256=weight_sha,scale_end_exclusive=cut,scale=scale,inputs=digests))
save(F/'results'/f'{a.family}_{a.phase}_complete.json',dict(all_passed=True,utc=utc(),real_forecasts=True,weight_sha256=weight_sha,neural_parameters_updated=False))
