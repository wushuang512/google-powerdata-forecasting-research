import sys,os,json,gc,copy,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'round2/code'));sys.path.insert(0,str(B/'code'))
import numpy as np
import torch
import pandas as pd
from round2_lib import predict,DEV,TEST
from common import load,truth,metrics
from deep_models import base_chronos
torch.set_num_threads(4)

CONFIGS={
 'head_uni':dict(scope='head',kind='uni',lr=1e-5,anchor=.5,physical=0),
 'head_cov':dict(scope='head',kind='cov',lr=1e-5,anchor=.5,physical=.1),
 'tail_cov':dict(scope='tail',kind='cov',lr=1e-6,anchor=.5,physical=.1),
 'full_cov':dict(scope='all',kind='cov',lr=5e-7,anchor=.5,physical=.1)
}
CHECKPOINTS=[50,150,300]

def sample(d,cutoff,rng,cov,batch=16,context=672):
    y=d['y'][:,:cutoff].reshape(57,-1);v=d['valid'][:,:cutoff].reshape(57,-1)
    prod=np.where(d['pvalid'][:,:cutoff],d['prod'][:,:cutoff],np.nan).reshape(57,-1)
    xx=[];yy=[];mask=[]
    for _ in range(batch):
        i=int(rng.integers(57));end=int(rng.integers(288,cutoff*96-96+1));start=max(0,end-context)
        a=y[i,start:end];xx.append(np.pad(a,(context-len(a),0),constant_values=np.nan));yy.append(y[i,end:end+96]);mask.append(v[i,end:end+96])
        if cov:
            a=prod[i,start:end];xx.append(np.pad(a,(context-len(a),0),constant_values=np.nan));yy.append(prod[i,end:end+96]);mask.append(np.isfinite(prod[i,end:end+96]))
    group=np.repeat(np.arange(batch),2) if cov else np.arange(batch)
    x=torch.tensor(np.stack(xx),dtype=torch.float32,device='cuda');target=torch.tensor(np.nan_to_num(np.stack(yy)),dtype=torch.float32,device='cuda');valid=torch.tensor(np.stack(mask),device='cuda')
    return x,torch.tensor(group,device='cuda'),target,valid

def train_one(d,cutoff,name,cfg,teacher,steps=300,callback=None):
    torch.manual_seed(20260930);rng=np.random.default_rng(20260930)
    pipe=base_chronos();model=pipe.model;model.eval()
    for p in model.parameters():p.requires_grad_(False)
    if cfg['scope']=='all':
        for p in model.parameters():p.requires_grad_(True)
    else:
        for p in model.output_patch_embedding.parameters():p.requires_grad_(True)
        if cfg['scope']=='tail':
            for p in model.encoder.block[-1].parameters():p.requires_grad_(True)
    params=[p for p in model.parameters() if p.requires_grad];print('PARAMETERS',name,sum(p.numel() for p in params),flush=True)
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01);mid=10;cov=cfg['kind']=='cov';history=[]
    for step in range(1,steps+1):
        x,groups,target,valid=sample(d,cutoff,rng,cov)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher_q=teacher.model(context=x,group_ids=groups,num_output_patches=6).quantile_preds.float()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):q=model(context=x,group_ids=groups,num_output_patches=6).quantile_preds.float()
        ids=torch.arange(0,len(x),2 if cov else 1,device='cuda');median=q[ids,mid];labels=target[ids];vm=valid[ids]
        den=(labels.abs()*vm).sum(-1).clamp_min(1e-3);point=((median-labels).abs()*vm).sum(-1)/den
        anchor=(q[ids]-teacher_q[ids]).abs().mean((1,2))/(den/vm.sum(-1).clamp_min(1))
        loss=point.mean()+cfg['anchor']*anchor.mean()
        bounds=(torch.relu(-median)+torch.relu(median-1)).mean()
        if cov:
            pm=q[ids+1,mid];pv=valid[ids+1]&vm
            aux=((pm-target[ids+1]).abs()*pv).sum(-1)/den
            order=(torch.relu(pm-median)*pv).sum(-1)/den
            loss=loss+.1*aux.mean()+cfg['physical']*order.mean()
        loss=loss+.1*bounds
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
        for g in opt.param_groups:g['lr']=cfg['lr']*(.2+.8*(1-step/300))
        if step%25==0:
            row=dict(step=step,point_loss=float(point.mean().detach()),anchor=float(anchor.mean().detach()),total=float(loss.detach()));history.append(row);print('TRAIN',name,cutoff,row,flush=True)
        if callback and step in CHECKPOINTS:callback(pipe,step)
    return pipe,history

def save_delta(pipe,path,cfg,cutoff,step):
    path.mkdir(parents=True,exist_ok=True)
    state={k:p.detach().cpu().clone() for k,p in pipe.model.named_parameters() if p.requires_grad}
    torch.save(state,path/'trainable_weights.pt')
    (path/'config.json').write_text(json.dumps(dict(configuration=cfg,cutoff=cutoff,step=step,base_model='amazon/chronos-2',base_revision='29ec3766d36d6f73f0696f85560a422f50e8498c',weights='full values of trainable parameters, loaded over unchanged base'),indent=2))

def load_delta(path):
    pipe=base_chronos();state=torch.load(path/'trainable_weights.pt',map_location='cpu',weights_only=True);result=pipe.model.load_state_dict(state,strict=False);assert not result.unexpected_keys;pipe.model.eval();return pipe

def main():
    d=load();teacher=base_chronos();teacher.model.eval()
    for p in teacher.model.parameters():p.requires_grad_(False)
    y,m=truth(d,DEV);path=R/'results/development.npz';pred=dict(np.load(path)) if path.exists() else {};rows=[]
    (R/'docs/protocol.json').write_text(json.dumps(dict(configurations=CONFIGS,checkpoints=CHECKPOINTS,cutoff=14,target='Direct masked point WAPE surrogate, teacher anchoring, production auxiliary and ordering penalties',inference='Median only; no calibrated uncertainty claim',selection='first four development days choose checkpoint, later four check before test',test_used_for_tuning=False,already_examined_month=True),indent=2))
    for name,cfg in CONFIGS.items():
        if all(f'{name}_s{k}' in pred for k in CHECKPOINTS):continue
        def checkpoint(pipe,step):
            key=f'{name}_s{step}';pred[key]=np.clip(predict(pipe,d,DEV,cfg['kind']),0,1)
            row=dict(model=key,**metrics(y,pred[key],m));rows.append(row);np.savez_compressed(path,**pred)
            pd.DataFrame([dict(model=k,**metrics(y,z,m)) for k,z in pred.items()]).to_csv(R/'results/development.csv',index=False)
            save_delta(pipe,R/'models'/f'{key}_d14',cfg,14,step);print('RESULT',json.dumps(row),flush=True)
        ft,hist=train_one(d,14,name,cfg,teacher,callback=checkpoint)
        (R/'results'/f'{name}_history.json').write_text(json.dumps(hist,indent=2));del ft;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
