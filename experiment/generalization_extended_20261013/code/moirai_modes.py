"""Refit previously promising joint coarse Moirai modes; score on original grid."""
from pathlib import Path
import argparse, hashlib, sys, os
sys.dont_write_bytecode=True;os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('case');ap.add_argument('--rate',type=int,choices=[4,12],required=True);ap.add_argument('--cut',type=int);ap.add_argument('--stage',choices=['train','infer'],default='train');ap.add_argument('--label',default='s0');ap.add_argument('--phase',default='dev');a=ap.parse_args()
ROOT=Path(a.root);F=ROOT/'experiment/generalization_extended_20261013';OLD=ROOT/'experiment/generalization_validation_20261012'
sys.path[:0]=[str(OLD/'code'),str(ROOT/'experiment/round35/code'),str(ROOT/'experiment/round35/code/official_uni2ts/src')];sys.path.append(str(ROOT/'experiment/round35/runtime'))
from core import read,save,sha,utc,np_save,baseline
from moirai_native_adapter import build
proto=read(F/'docs/protocol.json');cfg=proto['cases'][a.case];assert a.case!='Sea2022';cut=a.cut or cfg['cutoff'];rate=a.rate
z=np.load(F/'data'/f'{a.case}.npz');scale=z['scales'];x=z['x']/scale[:,None,None];y=z['y']/scale[:,None,None];mask=z['valid'];names=z['names'];dates=z['dates'];ppdc=96//rate
xx=x.reshape(len(x),-1,rate).mean(-1);yy=y.reshape(len(y),-1,rate).mean(-1);mm=mask.reshape(len(mask),-1,rate).all(-1)
groups=[np.arange(len(x))] if a.case.startswith('NLR') else[np.where(np.array([str(n).split('_')[0] for n in names])==g)[0] for g in sorted(set(str(n).split('_')[0] for n in names))]
torch.set_num_threads(2);torch.manual_seed(20261013);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device='cuda';assert torch.cuda.is_available()
status=F/'results'/f'{a.case}_moirai_rate{rate}_cut{cut}_{a.stage}_{a.phase}_status.json';save(status,dict(state='running',pid=os.getpid(),utc=utc()))
def tensors(values):
    v=torch.tensor(values.T[None],dtype=torch.float32,device=device);return v,torch.ones_like(v,dtype=torch.bool),torch.zeros(v.shape[:2],dtype=torch.bool,device=device)
def model(ids,context,horizon,state=None):
    m=build(ROOT/'experiment/round35',context,horizon,len(ids)).to(device).eval()
    for n,p in m.module.named_parameters():p.requires_grad_(n.startswith('out_proj.') or n.startswith('encoder.layers.5.'))
    if state is not None:m.module.load_state_dict(state,strict=False)
    return m
def train():
    final=F/'models'/f'{a.case}_moirai_rate{rate}_cut{cut}_s48.pt'
    if final.exists():assert read(final.with_suffix('.json'))['model_sha256']==sha(final);return
    # Joint module parameters do not depend on the number of variates.
    context_days=min(7,max(3,cut//2))
    ids=groups[0];m=model(ids,context_days*ppdc,64);params=[p for p in m.module.parameters() if p.requires_grad]
    initial={n:p.detach().clone() for n,p in m.module.named_parameters() if p.requires_grad};opt=torch.optim.AdamW(params,lr=3e-6,weight_decay=.01);rng=np.random.default_rng(20261013);history=[];windows=[];skips=[]
    for attempt in range(1,241):
        step=len(history)+1
        if step>48:break
        ids=groups[int(rng.integers(len(groups)))];m.hparams.target_dim=len(ids);m.hparams.context_length=context_days*ppdc;m.hparams.prediction_length=64
        for draw in range(100):
            s=int(rng.integers(context_days*ppdc,cut*ppdc-4));length=min(64,cut*ppdc-s)
            if mm[ids,s-context_days*ppdc:s].mean()>=.9 and mm[ids,s:s+length].mean()>=.8:break
        else:raise RuntimeError('No eligible coarse training window')
        values=xx[ids,s-context_days*ppdc:s];v,obs,pad=tensors(values);packed=m._convert(16,v,obs,pad);raw=m.module(*packed,training_mode=False);ct=(context_days*ppdc+15)//16;ix=torch.arange(ct-1,ct*len(ids),ct,device=device)
        q=raw[:,ix].reshape(1,len(ids),4,9,16).permute(0,1,3,2,4).reshape(len(ids),9,64).transpose(1,2);p=q[:,:,4]
        target=np.zeros((len(ids),64),dtype=np.float32);valid=np.zeros((len(ids),64),bool);target[:,:length]=yy[ids,s:s+length];valid[:,:length]=mm[ids,s:s+length]
        t=torch.tensor(target,device=device);ok=torch.tensor(valid,device=device);den=(t.abs()*ok).sum(-1).clamp_min(.03);curve=((p-t).abs()*ok).sum(-1)/den
        levels=torch.arange(1,10,device=device)/10;err=t[:,:,None]-q;pin=(torch.maximum(levels*err,(levels-1)*err)*ok[:,:,None]).sum((1,2))/den/9
        prox=torch.stack([((dict(m.module.named_parameters())[n]-ref)**2).mean()/ref.square().mean().clamp_min(1e-4) for n,ref in initial.items()]).mean()
        loss=curve.mean()+.05*pin.mean()+.01*prox+.1*torch.relu(-p).mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
        gradients=[float(p.grad.abs().max()) for p in params if p.grad is not None]
        if not gradients or not np.isfinite(gradients).all() or max(gradients)==0:
            skips.append(dict(attempt=attempt,next_update=step,origin_coarse=s));opt.zero_grad(set_to_none=True);continue
        torch.nn.utils.clip_grad_norm_(params,1);opt.step();history.append(dict(step=step,loss=float(loss.detach())));windows.append(dict(series=ids.tolist(),origin_fine=s*rate,label_end_fine_exclusive=(s+length)*rate,cutoff_fine=cut*96))
        if step%8==0:print('TRAIN joint coarse Moirai',a.case,rate,cut,step,history[-1],flush=True)
        if step in [16,48]:
            out=F/'models'/f'{a.case}_moirai_rate{rate}_cut{cut}_s{step}.pt';state={n:p.detach().cpu() for n,p in m.module.named_parameters() if p.requires_grad};torch.save(state,out)
            save(out.with_suffix('.json'),dict(real_optimizer_updates=step,model_sha256=sha(out),script_sha256=sha(Path(__file__)),rate=rate,cutoff=cut,training_windows=windows.copy(),loss_history=history.copy(),skipped_gradients=skips.copy(),source_weight_sha256=sha(ROOT/'experiment/round35/models/moirai2/model.safetensors')))
    assert len(history)==48
def origins(H):
    if a.case.startswith('NLR'):return cfg['origins'][a.phase]
    if a.phase=='oof':return [7] if H==7 else list(range(7,14))
    if a.phase=='dev':return list(range(14,22)) if H==1 else[14,15]
    return cfg['origins'][a.phase+str(H)]
def infer():
    state=None;weight='original'
    if a.label!='s0':
        path=F/'models'/f'{a.case}_moirai_rate{rate}_cut{cut}_{a.label}.pt';meta=read(path.with_suffix('.json'));assert meta['model_sha256']==sha(path);state=torch.load(path,map_location='cpu',weights_only=True);weight=sha(path)
    m=model(groups[0],7*ppdc,ppdc,state)
    for H in [1,7]:
        days=7 if H==1 else 14;oo=origins(H);base=baseline(x,oo,H,'last' if H==1 else'mean3')
        for k,o in enumerate(oo):
            path=F/'cache'/f'{a.case}_moirai_rate{rate}_cut{cut}_{a.label}_{a.phase}_H{H}_o{o}.npz';context=xx[:,max(0,o-days)*ppdc:o*ppdc]
            fingerprint=hashlib.sha256((sha(Path(__file__))+weight+sha(F/'docs/protocol.json')).encode()+np.ascontiguousarray(context).tobytes()).hexdigest()
            if path.exists():assert str(np.load(path)['fingerprint'])==fingerprint;continue
            coarse=np.zeros((len(x),H*ppdc))
            with torch.inference_mode():
                for ids in groups:
                    m.hparams.target_dim=len(ids);m.hparams.context_length=context.shape[1];m.hparams.prediction_length=H*ppdc
                    v,obs,pad=tensors(context[ids]);q=m(v,obs,pad).cpu().numpy()
                    if len(ids)==1:q=q[:,:,:,None]
                    assert q.shape==(1,9,H*ppdc,len(ids));coarse[ids]=q[0,4].T
            target=np.maximum(0,coarse);b=base[:,k].reshape(len(x),H*ppdc,rate)
            if a.case=='Google2019':
                target=np.clip(target,0,1);lo=np.full(target.shape,-1.);hi=np.full(target.shape,1.)
                for _ in range(50):
                    shift=(lo+hi)/2;over=np.clip(b+shift[:,:,None],0,1).mean(-1)>target;hi=np.where(over,shift,hi);lo=np.where(over,lo,shift)
                point=np.clip(b+((lo+hi)/2)[:,:,None],0,1)
            else:
                mean=b.mean(-1);point=np.where(mean[:,:,None]>1e-8,b*target[:,:,None]/np.maximum(mean[:,:,None],1e-8),np.broadcast_to(target[:,:,None],b.shape));point=np.maximum(0,point)
            assert np.max(abs(point.mean(-1)-target))<1e-7
            point=point.reshape(len(x),H*96);np_save(path,p=point,origin=o,horizon=H,fingerprint=fingerprint,rate=rate,original_evaluation_grid_points_per_day=96,protocol_sha256=sha(F/'docs/protocol.json'))
            print('INFER joint coarse Moirai',a.case,rate,cut,a.label,a.phase,H,str(dates[o]),flush=True)
try:
    train() if a.stage=='train' else infer();save(status,dict(state='complete',utc=utc(),case=a.case,rate=rate,cut=cut,stage=a.stage,phase=a.phase))
except BaseException as e:
    import traceback
    save(status,dict(state='failed',error=repr(e),traceback=traceback.format_exc(),utc=utc()));raise
