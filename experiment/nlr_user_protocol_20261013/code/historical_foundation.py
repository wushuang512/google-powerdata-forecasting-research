import argparse,gc,os,sys,time,hashlib,traceback
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
import numpy as np,torch
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('family',choices=['chronos2','tirex2','timesfm25','patchtst','timesfm3','toto2']);ap.add_argument('phase',choices=['validation','test']);a=ap.parse_args();R=Path(a.root);F,cfg,d,q=load(R);reg=read(F/'docs/historical_controls.json');assert sha(F/'docs/historical_controls.json')==(F/'docs/historical_controls.sha256').read_text().strip();assert sha(F/'results/selection.json')==reg['original_target_selection_sha256']
torch.set_num_threads(2);sys.path[:0]=[str(R/'experiment/round46/code'),str(R/'experiment/round44/reference/PowerData_Prediction')];sys.path.append(str(R/'experiment/round44/.venv/Lib/site-packages'))
from hdsf_foundation_adapters import Adapter
cut=q['split']['train'][1] if a.phase=='validation' else q['split']['validation'][1];known=d['y'][:cut];scale=float(np.quantile(known[np.isfinite(known)],.95)/.8)
device='cuda'
if a.family=='tirex2':
    # xlstm's unrelated optional sLSTM import searches for a CUDA compiler.
    # Import the explicitly selected native CPU implementation, restore CUDA
    # availability immediately, then verify native tensor computation on GPU.
    original_cuda_available=torch.cuda.is_available
    try:
        torch.cuda.is_available=lambda:False
        adapter=Adapter(R,a.family,'cpu')
    finally:torch.cuda.is_available=original_cuda_available
    adapter.set_scope('tail');initial=adapter.trainable_state()
    o=origins(q,a.phase,1)[0];ctx=np.ascontiguousarray(d['y'][o-14:o].reshape(1,1344)/scale,dtype=np.float32)
    try:
        cpu_probe=adapter.predict(ctx,320,'uni');adapter.model.to('cuda');adapter.device=torch.device('cuda');probe=adapter.predict(ctx,320,'uni')
        assert probe.shape==(1,320) and np.isfinite(probe).all() and np.allclose(probe,cpu_probe,rtol=1e-5,atol=1e-6)
        save(F/'results/tirex_gpu_probe.json',dict(passed=True,device='cuda',native_recurrent_configuration='pure PyTorch CPU reference implementation moved as tensors toGPU',cpu_gpu_same_input_comparison_passed=True,max_absolute_normalized_difference=float(abs(probe-cpu_probe).max()),no_labels_or_scores_used_for_device_choice=True,optional_sLSTM_cuda_compiler_import_skipped_only=True))
    except Exception as error:
        save(F/'results/tirex_gpu_probe.json',dict(passed=False,error=repr(error),traceback=traceback.format_exc(),fallback='native cpu',no_labels_or_scores_used_for_device_choice=True));adapter.model.to('cpu');adapter.device=torch.device('cpu');device='cpu'
else:adapter=Adapter(R,a.family,device);adapter.set_scope('tail');initial=adapter.trainable_state()
def infer(ctx,H):
    if a.family!='tirex2' or H<=320:return adapter.predict(ctx,H,'uni')
    parts=[];history=ctx.copy();remaining=H
    while remaining:
        n=min(320,remaining);p=adapter.predict(history,n,'uni');parts.append(p);history=np.concatenate([history,p],1)[:,-1344:];remaining-=n
    return np.concatenate(parts,1)
if a.family in ['chronos2','tirex2','timesfm25']:modes=[('original',None,[1,7])]
elif a.family=='patchtst':modes=[('original',None,[1,7]),('r47curve',R/'experiment/round47/models/patchtst_curve_cut22.pt',[1]),('r46',Path(reg['r46_states']['patchtst']),[7])]
elif a.family=='timesfm3':modes=[('r8b',Path(reg['r8_weight_file']),[1,7]),('r46',Path(reg['r46_states']['timesfm3']),[7])]
else:modes=[('r46',Path(reg['r46_states']['toto2']),[7])]
for label,statepath,HH in modes:
    adapter.load_trainable_state(initial)
    if statepath:
        assert reg['bindings'][str(statepath.relative_to(R))]==sha(statepath);state=torch.load(statepath,map_location='cpu',weights_only=True);adapter.load_trainable_state(state['state'] if 'state' in state else state)
    weight_digest=adapter.hash_trainable()
    for H in HH:
        oo=origins(q,a.phase,H);out=[]
        for k,o in enumerate(oo):
            ctx=np.ascontiguousarray(d['y'][o-14:o].reshape(1,1344)/scale,dtype=np.float32);finger=hashlib.sha256((sha(F/'docs/historical_controls.json')+sha(Path(__file__))+weight_digest+str(scale)+hashlib.sha256(ctx.tobytes()).hexdigest()).encode()).hexdigest();path=F/'cache'/f'historical_{a.phase}_{a.family}_{label}_H{H}_o{o}.npz'
            if path.exists():
                with np.load(path) as z:assert str(z['fingerprint'])==finger;point=z['p'].copy()
            else:point=np.maximum(0,infer(ctx,H*96)[0]*scale);assert point.shape==(H*96,) and np.isfinite(point).all();np_save(path,p=point,fingerprint=finger,origin=o,horizon=H)
            out.append(point)
            if k%20==0:save(F/'results/historical_foundation_status.json',dict(family=a.family,phase=a.phase,label=label,horizon_days=H,complete=k+1,total=len(oo),device=device,pid=os.getpid(),utc=utc()));print('SOURCE',a.family,a.phase,label,H,k+1,len(oo),flush=True)
        name='source_'+a.family+'_'+label;save_forecasts(F,cfg,d,q,a.phase,H,name,np.stack(out),dict(source_weights_frozen=True,target_gradient_training=False,scale_end_exclusive=cut,history_points=1344,input_adaptation=True,state_file=str(statepath) if statepath else'original pretrained',state_sha256=sha(statepath) if statepath else None,source_head_digest=weight_digest,device=device))
