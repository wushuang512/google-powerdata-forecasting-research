"""Actual target training of native pretrained parameters; isolated family job."""
from pathlib import Path
import argparse, gc, hashlib, json, math, os, sys, time, traceback
sys.dont_write_bytecode=True
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg')
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('case');ap.add_argument('family');ap.add_argument('--cut',type=int);ap.add_argument('--stage',choices=['train','infer'],default='train');ap.add_argument('--label',default='s0');ap.add_argument('--phase',default='dev');a=ap.parse_args()
ROOT=Path(a.root); F=ROOT/'experiment/generalization_extended_20261013';old=ROOT/'experiment/generalization_validation_20261012'
sys.path.insert(0,str(old/'code'));from core import read,save,sha,utc,np_save
proto=read(F/'docs/protocol.json');assert sha(F/'docs/protocol.json')==(F/'docs/protocol.sha256').read_text().strip()
cfg=proto['cases'][a.case];z=np.load(F/'data'/f'{a.case}.npz');ppd=int(z['ppd']);sc=z['scales'].copy();xn=z['x']/sc[:,None,None];yn=z['y']/sc[:,None,None];valid=z['valid'];dates=z['dates'];flat=xn.reshape(len(xn),-1);truth=yn.reshape(len(yn),-1);mask=valid.reshape(len(valid),-1)
cut=a.cut or cfg['cutoff'];assert ppd==96 or a.stage=='infer','Small daily dataset uses target residual fitting, not high-dimensional weight training'
status=F/'results'/f'{a.case}_{a.family}_cut{cut}_{a.stage}_{a.phase}_status.json'
save(status,dict(state='starting',pid=os.getpid(),utc=utc(),case=a.case,family=a.family,cut=cut))
import torch
torch.set_num_threads(2);torch.manual_seed(20261013);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device='cpu' if a.family=='tirex2' else 'cuda';assert device=='cpu' or torch.cuda.is_available()
sys.path[:0]=[str(ROOT/'experiment/round46/code'),str(ROOT/'experiment/round44/reference/PowerData_Prediction')]
sys.path.append(str(ROOT/'experiment/round44/.venv/Lib/site-packages'))
for p in ['experiment/round35/runtime','experiment/round35/code','experiment/round35/code/official_uni2ts/src','experiment/round29/cpu_runtime']:
    sys.path.append(str(ROOT/p))

class ExtraAdapter:
    def __init__(self,family):
        self.family=family;self.device=torch.device(device)
        if family=='moirai2':
            from moirai_native_adapter import build
            self.model=build(ROOT/'experiment/round35',672,64,1).to(device).eval();self.levels=[i/10 for i in range(1,10)]
            self.basepath=ROOT/'experiment/round35/models/moirai2/model.safetensors'
        else:
            from ttm_official.modeling_tinytimemixer import TinyTimeMixerForPrediction
            self.context=1024 if family=='ttm1024' else 512
            self.basepath=ROOT/f'experiment/round29/models/{self.context}-720-r2/model.safetensors'
            self.model=TinyTimeMixerForPrediction.from_pretrained(str(self.basepath.parent),local_files_only=True).to(device).eval();self.levels=[.5]
        self.train_names=[]
        for n,p in self.model.named_parameters():
            enabled=(n.startswith('module.out_proj.') or n.startswith('module.encoder.layers.5.')) if family=='moirai2' else(n.startswith('head.') or n.startswith('decoder.'))
            p.requires_grad_(enabled)
            if enabled:self.train_names.append(n)
        assert self.train_names
    def trainable_state(self):return {n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}
    def load_trainable_state(self,state):
        params=dict(self.model.named_parameters())
        with torch.no_grad():
            for n,v in state.items():params[n].copy_(v.to(self.device))
    def hash_trainable(self):
        h=hashlib.sha256()
        for n,p in self.model.named_parameters():
            if p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def tensor_forecast(self,context,H,mode='uni',grad=True):
        x=torch.as_tensor(np.ascontiguousarray(context,dtype=np.float32),device=self.device)
        if self.family=='moirai2':
            m=self.model;m.hparams.context_length=x.shape[-1];m.hparams.prediction_length=H
            obs=torch.ones_like(x[:,:,None],dtype=torch.bool);pad=torch.zeros(x.shape,dtype=torch.bool,device=x.device)
            if grad:
                assert H==64
                packed=m._convert(16,x[:,:,None],obs,pad);raw=m.module(*packed,training_mode=False)
                ct=math.ceil(x.shape[-1]/16);q=raw[:,ct-1].reshape(len(x),4,9,16).permute(0,2,1,3).reshape(len(x),9,64).transpose(1,2)
            else:q=m(x[:,:,None],obs,pad).transpose(1,2)
            p=q[:,:,4]
        else:
            if x.shape[-1]<self.context:x=torch.cat([x[:,:1].expand(-1,self.context-x.shape[-1]),x],1)
            x=x[:,-self.context:];p=self.model(past_values=x[:,:,None]).prediction_outputs[:,:H,0];q=p[:,:,None]
        assert p.shape==(len(x),H) and q.shape==(len(x),H,len(self.levels)) and torch.isfinite(q).all(),(self.family,p.shape,q.shape,H)
        return q,p
    def predict(self,context,H,mode='uni'):
        with torch.inference_mode():return self.tensor_forecast(context,H,mode,False)[1].float().cpu().numpy().astype(float)

def make_adapter():
    if a.family in ['moirai2','ttm','ttm1024']:return ExtraAdapter(a.family)
    from hdsf_foundation_adapters import Adapter
    result=Adapter(ROOT,a.family,device);result.set_scope('tail');return result

def context(ids,o,H):
    days=28 if a.family=='chronos2' else 21 if a.family in ['timesfm3','patchtst'] else 14 if a.family in ['tirex2','timesfm25','toto2'] else 7 if H==1 else 14
    left=max(0,o-days);x=flat[np.array(ids),left*ppd:o*ppd].copy()
    if a.family=='timesfm25':
        n=max(32,math.ceil(x.shape[-1]/32)*32)
        if x.shape[-1]<n:x=np.concatenate([np.repeat(x[:,:1],n-x.shape[-1],axis=1),x],1)
    return np.ascontiguousarray(x,dtype=np.float32)

def origin_list(phase,H):
    if a.case.startswith('NLR'):return cfg['origins'][phase]
    if phase=='oof':return [7] if H==7 else list(range(7,14))
    if phase=='dev':
        if a.case=='Google2019':return list(range(14,22)) if H==1 else[14,15]
        return cfg['origins']['dev'+str(H)]
    return cfg['origins'][phase+str(H)]

def collect(adapter,label,phase):
    weight=adapter.hash_trainable();batch=1 if a.family in ['patchtst','tirex2','moirai2'] else 2
    for H in [1,7]:
        oo=origin_list(phase,H)
        for o in oo:
            path=F/'cache'/f'{a.case}_{a.family}_cut{cut}_{label}_{phase}_H{H}_o{o}.npz'
            full=context(range(len(xn)),o,H);digest=hashlib.sha256(full.tobytes()).hexdigest()
            finger=hashlib.sha256((sha(F/'docs/protocol.json')+sha(Path(__file__))+weight+digest).encode()).hexdigest()
            if path.exists():
                q=np.load(path);assert str(q['fingerprint'])==finger and np.isfinite(q['p']).all();continue
            started=time.perf_counter();chunks=[]
            for first in range(0,len(xn),batch):chunks.append(adapter.predict(full[first:first+batch],H*ppd,'uni'))
            p=np.concatenate(chunks);assert p.shape==(len(xn),H*ppd) and np.isfinite(p).all()
            p=np.maximum(0,p)
            if a.case=='Google2019':p=np.clip(p,0,1)
            np_save(path,p=p,fingerprint=finger,input_digest=digest,weight_digest=weight,origin=o,horizon=H,protocol_sha256=sha(F/'docs/protocol.json'))
            save(status,dict(state='inference',pid=os.getpid(),utc=utc(),case=a.case,family=a.family,cut=cut,label=label,phase=phase,horizon=H,origin_date=str(dates[o]),seconds=time.perf_counter()-started))
            print('INFER',a.case,a.family,cut,label,phase,H,str(dates[o]),round(time.perf_counter()-started,2),flush=True)

def loss_fn(adapter,q,p,y,m,teacher,initial):
    yy=torch.as_tensor(y,dtype=torch.float32,device=p.device);mm=torch.as_tensor(m,device=p.device);den=(yy.abs()*mm).sum(-1).clamp_min(.03)
    curve=((p-yy).abs()*mm).sum(-1)/den
    levels=torch.tensor(adapter.levels,device=p.device);err=yy[:,:,None]-q
    pin=(torch.maximum(levels*err,(levels-1)*err)*mm[:,:,None]).sum((1,2))/den/len(levels)
    day=peak=p.new_tensor(0.)
    if p.shape[1]%96==0:
        pd=p.reshape(len(p),-1,96);yd=yy.reshape(len(p),-1,96);good=mm.reshape(len(p),-1,96).all(-1)
        day=((pd.mean(-1)-yd.mean(-1)).abs()*good).sum()/(yd.mean(-1).abs()*good).sum().clamp_min(.03)
        peak=(((pd.max(-1).values-yd.max(-1).values).abs()/yd.max(-1).values.clamp_min(.03))*good).sum()/good.sum().clamp_min(1)
    slope=(torch.diff(p,dim=1)-torch.diff(yy,dim=1)).abs()*(mm[:,1:]&mm[:,:-1]);slope=(slope.sum(-1)/den).mean()
    anchor=((p-teacher)**2).mean()/teacher.abs().mean().square().clamp_min(.03)
    params=dict(adapter.model.named_parameters());prox=torch.stack([((params[n]-v)**2).mean()/v.square().mean().clamp_min(1e-4) for n,v in initial.items()]).mean()
    # Nonnegative power constraint only. NLR training scale is not a capacity.
    loss=curve.mean()+.25*day+.1*peak+.05*pin.mean()+.05*slope+.05*anchor+.01*prox+.1*torch.relu(-p).mean()
    return loss,dict(curve=float(curve.mean().detach()),daily=float(day.detach()),peak=float(peak.detach()),total=float(loss.detach()))

def train(adapter):
    final=F/'models'/f'{a.case}_{a.family}_cut{cut}_s48.pt';meta=final.with_suffix('.json')
    if final.exists():
        info=read(meta);assert info['weight_file_sha256']==sha(final) and info['script_sha256']==sha(Path(__file__));return
    initial=adapter.trainable_state();initial_hash=adapter.hash_trainable();initial_device={n:v.to(adapter.device) for n,v in initial.items()}
    params=[p for p in adapter.model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=3e-6,weight_decay=.01)
    rng=np.random.default_rng(20261013);history=[];windows=[];maximum_label=0
    for step in range(1,49):
        h=64 if a.family=='moirai2' else 96 if a.family in ['chronos2','tirex2','timesfm25'] or step%2 else 672
        minimum=3*96
        possible=np.arange(minimum,cut*96-h+1,48)
        if len(possible)==0:h=96;possible=np.arange(minimum,cut*96-h+1,48)
        assert len(possible)>0,(a.case,cut,h)
        ids=[];starts=[]
        for attempt in range(100):
            j=int(rng.integers(len(xn)));s=int(rng.choice(possible))
            if mask[j,max(0,s-7*96):s].mean()>=.9 and mask[j,s:s+h].mean()>=.8:
                ids=[j];starts=[s];break
        assert ids,'Insufficient eligible training windows'
        j,s=ids[0],starts[0];n=min(7*96,s);n=n//32*32 if a.family=='timesfm25' else n
        x=np.ascontiguousarray(flat[j:j+1,s-n:s],dtype=np.float32);yy=truth[j:j+1,s:s+h];mm=mask[j:j+1,s:s+h]
        assert s+h<=cut*96
        with torch.no_grad():_,teacher=adapter.tensor_forecast(x,h,'uni',False)
        q,p=adapter.tensor_forecast(x,h,'uni',True);assert p.requires_grad and q.requires_grad
        loss,values=loss_fn(adapter,q.float(),p.float(),yy,mm,teacher.detach().float(),initial_device)
        assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=max(float(v.grad.abs().max()) for v in params if v.grad is not None)
        assert gradient>0 and np.isfinite(gradient);torch.nn.utils.clip_grad_norm_(params,1);optimizer.step()
        maximum_label=max(maximum_label,s+h);windows.append(dict(series=j,input_start=s-n,input_end_exclusive=s,label_end_exclusive=s+h,horizon=h))
        history.append(dict(step=step,**values,max_gradient=gradient,utc=utc()))
        save(status,dict(state='training',pid=os.getpid(),utc=utc(),case=a.case,family=a.family,cut=cut,step=step,total=48,last_loss=values))
        if step%8==0:print('TRAIN',a.case,a.family,cut,step,values,flush=True)
        if step in [16,48]:
            out=F/'models'/f'{a.case}_{a.family}_cut{cut}_s{step}.pt';state=adapter.trainable_state();assert adapter.hash_trainable()!=initial_hash;torch.save(state,out)
            info=dict(real_optimizer_steps=step,weight_file_sha256=sha(out),script_sha256=sha(Path(__file__)),protocol_sha256=sha(F/'docs/protocol.json'),
                training_last_label_exclusive=maximum_label,training_cutoff_exclusive=cut*96,trainable_parameters=sum(p.numel() for p in params),
                trainable_tensor_names=adapter.train_names,initial_trainable_sha256=initial_hash,trained_trainable_sha256=adapter.hash_trainable(),
                data_sha256=sha(F/'data'/f'{a.case}.npz'),training_windows=windows.copy(),loss_history=history.copy(),no_future_true_inputs=True,device=device)
            assert maximum_label<=cut*96;save(out.with_suffix('.json'),info)
    save(F/'results'/f'{a.case}_{a.family}_cut{cut}_training_complete.json',dict(all_passed=True,real_training=True,steps=48,utc=utc(),script_sha256=sha(Path(__file__))))

try:
    adapter=make_adapter()
    if a.stage=='train':train(adapter)
    else:
        if a.label!='s0':
            path=F/'models'/f'{a.case}_{a.family}_cut{cut}_{a.label}.pt';info=read(path.with_suffix('.json'))
            assert info['weight_file_sha256']==sha(path) and info['protocol_sha256']==sha(F/'docs/protocol.json')
            adapter.load_trainable_state(torch.load(path,map_location='cpu',weights_only=True))
        collect(adapter,a.label,a.phase)
    save(status,dict(state='complete',utc=utc(),case=a.case,family=a.family,cut=cut,stage=a.stage,phase=a.phase,label=a.label))
except BaseException as error:
    save(status,dict(state='failed',utc=utc(),error=repr(error),traceback=traceback.format_exc(),case=a.case,family=a.family,cut=cut));raise
