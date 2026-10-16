import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import json,time,gc, argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from common import *

torch.set_num_threads(4)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

class PatchModel(nn.Module):
    """Compact PatchTST-inspired channel-independent encoder; not official reproduction."""
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(24,64)
        self.pos=nn.Parameter(torch.zeros(1,55,64))
        layer=nn.TransformerEncoderLayer(64,4,128,dropout=.15,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.15),nn.Linear(55*64,96))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        mean=x.mean(-1,keepdim=True);std=x.std(-1,keepdim=True).clamp_min(.01)
        z=(x-mean)/std
        e=self.embed(z.unfold(-1,24,12))+self.pos
        # Residual around the last-day profile; all inputs precede the origin.
        return x[:,-96:]+self.head(self.encoder(e))*std

def train_patch(d,cutoff,epochs=24,constrained=False,checkpoints=None):
    torch.manual_seed(20260929);np.random.seed(20260929)
    yf=d['y'][:,:cutoff].reshape(57,-1);vf=d['valid'][:,:cutoff].reshape(57,-1)
    x=[];y=[];m=[]
    for t in range(672,cutoff*96-96+1,12):
        x.append(yf[:,t-672:t]);y.append(yf[:,t:t+96]);m.append(vf[:,t:t+96])
    xt=torch.tensor(np.concatenate(x),dtype=torch.float32,device=DEVICE)
    yt=torch.tensor(np.concatenate(y),dtype=torch.float32,device=DEVICE)
    mt=torch.tensor(np.concatenate(m),dtype=torch.float32,device=DEVICE)
    net=PatchModel().to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=.02)
    results={};history=[]
    for ep in range(1,epochs+1):
        net.train();perm=torch.randperm(len(xt),device=DEVICE);losses=[]
        for ix in perm.split(64):
            xx,yy,mm=xt[ix],yt[ix],mt[ix]
            pp=net(xx)
            scale=xx.mean(-1,keepdim=True).clamp_min(.05)
            loss=(torch.abs(pp-yy)/scale*mm).sum()/mm.sum()
            if constrained:
                # Physical bounds and multi-scale consistency targets, not an HVAC law.
                loss=loss+2*(torch.relu(-pp)+torch.relu(pp-1)).mean()
                loss=loss+.15*((pp*mm).sum(-1)/mm.sum(-1)-(yy*mm).sum(-1)/mm.sum(-1)).abs().mean()/scale.mean()
                loss=loss+.1*(pp.masked_fill(mm==0,-1).max(-1).values-yy.masked_fill(mm==0,-1).max(-1).values).abs().mean()/scale.mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(loss.item())
        history.append(dict(epoch=ep,loss=float(np.mean(losses))))
        if checkpoints and ep in checkpoints:
            p=predict_patch(net,d,DEV24)
            score=metrics(*_metric_args(d,p))
            name=f'patch_{"physical" if constrained else "plain"}_e{ep}'
            results[name]=p
            torch.save(net.state_dict(),ROOT/f'models/{name}_dev.pt')
            print(name,json.dumps(score),flush=True)
        elif ep%8==0:print('patch train',cutoff,constrained,ep,float(np.mean(losses)),flush=True)
    return net,results,history

def _metric_args(d,p):
    y,m=truth(d,DEV24);return y,p,m

@torch.inference_mode()
def predict_patch(net,d,origins,h=1):
    net.eval();out=[]
    for o in origins:
        context=torch.tensor(d['y'][:,o-7:o].reshape(57,672),dtype=torch.float32,device=DEVICE)
        pred=[]
        for j in range(h):
            p=net(context).clamp(0,1);pred.append(p.cpu().numpy());context=torch.cat([context[:,96:],p],-1)
        out.append(np.concatenate(pred,axis=-1))
    return np.stack(out,axis=1)

def develop_patch():
    d=load();preds={};hist={}
    for physical in [False,True]:
        net,p,h=train_patch(d,14,epochs=24,constrained=physical,checkpoints=[4,8,16,24]);preds.update(p);hist[str(physical)]=h
        del net;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(ROOT/'results/dev_patch.npz',**preds)
    y,m=truth(d,DEV24)
    pd.DataFrame([dict(model=k,**metrics(y,v,m)) for k,v in preds.items()]).to_csv(ROOT/'results/dev_patch.csv',index=False)
    (ROOT/'results/patch_training_history.json').write_text(json.dumps(hist,indent=2))

def chronos_inputs(d,o,cov=False,full=False):
    a=0 if full else max(0,o-14)
    targets=d['y'][:,a:o].reshape(57,-1).astype(np.float32)
    if not cov:return [row for row in targets]
    prod=np.where(d['pvalid'][:,a:o],d['prod'][:,a:o],np.nan).reshape(57,-1).astype(np.float32)
    return [dict(target=targets[i],past_covariates={'production':prod[i]}) for i in range(57)]

def predict_chronos(pipe,d,origins,h=1,cov=False):
    out=[];quant=[]
    for o in origins:
        q,_=pipe.predict_quantiles(chronos_inputs(d,o,cov),prediction_length=96*h,quantile_levels=[.1,.5,.9],context_length=1344,batch_size=32)
        a=np.stack([v[0].float().cpu().numpy() for v in q])
        quant.append(a);out.append(a[:,:,1])
        print('Chronos forecast',o,h,'cov',cov,flush=True)
    return np.stack(out,axis=1),np.stack(quant,axis=1)

def base_chronos():
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(str(ROOT/'models/chronos2-base'),device_map=DEVICE)

def project_components(total,production,remainder):
    """Euclidean reconciliation with total=production+remainder, components>=0, total<=1."""
    p=(2*production+total-remainder)/3
    r=(2*remainder+total-production)/3
    # Enumerate the interior and three edges of the feasible triangle.
    candidates=[(p,r),(np.zeros_like(p),np.clip((total+remainder)/2,0,1)),(np.clip((total+production)/2,0,1),np.zeros_like(p))]
    a=np.clip((1+production-remainder)/2,0,1);candidates.append((a,1-a))
    costs=[]
    for a,b in candidates:
        cost=(a+b-total)**2+(a-production)**2+(b-remainder)**2
        cost=np.where((a>=0)&(b>=0)&(a+b<=1+1e-7),cost,np.inf)
        costs.append(cost)
    best=np.argmin(np.stack(costs),axis=0)
    ap=np.take_along_axis(np.stack([a for a,b in candidates]),best[None],axis=0)[0]
    ar=np.take_along_axis(np.stack([b for a,b in candidates]),best[None],axis=0)[0]
    return ap+ar,ap,ar

def predict_joint(pipe,d,origins,h=1):
    raw=[];physical=[];diagnostics=[]
    for o in origins:
        a=max(0,o-14);t=d['y'][:,a:o].reshape(57,-1).astype(np.float32)
        good=(d['valid'][:,a:o]&d['pvalid'][:,a:o]).reshape(57,-1)
        prod=d['prod'][:,a:o].reshape(57,-1).astype(np.float32)
        enough=good.mean(1)>=.2
        inp=[np.stack([t[i],np.where(good[i],prod[i],np.nan),np.where(good[i],t[i]-prod[i],np.nan)]) if enough[i] else t[i] for i in range(57)]
        q,_=pipe.predict_quantiles(inp,prediction_length=96*h,quantile_levels=[.5],context_length=1344,batch_size=24)
        rp=[];cp=[];gap=[]
        for i,v in enumerate(q):
            x=v[:,:,0].float().cpu().numpy()
            rp.append(x[0])
            if enough[i]:
                total,p,r=project_components(x[0],x[1],x[2]);cp.append(total);gap.append(float(np.mean(np.abs(x[0]-x[1]-x[2]))))
            else:cp.append(np.clip(x[0],0,1))
        raw.append(np.stack(rp));physical.append(np.stack(cp));diagnostics.append(dict(origin=o,joint_domains=int(enough.sum()),mean_raw_balance_gap=float(np.mean(gap))))
        print('Chronos joint',o,h,diagnostics[-1],flush=True)
    return np.stack(raw,1),np.stack(physical,1),diagnostics

def fit_chronos(base,d,cutoff,steps=100):
    # Invalid targets are NaN rather than imputed supervision for this model.
    targets=np.where(d['valid'][:,:cutoff],d['y'][:,:cutoff],np.nan).reshape(57,-1).astype(np.float32)
    return base.fit([row for row in targets],prediction_length=96,finetune_mode='lora',context_length=672,learning_rate=1e-4,num_steps=steps,batch_size=16,min_past=288,
        output_dir=str(ROOT/f'models/chronos_lora_d{cutoff}_s{steps}'),logging_steps=25,report_to='none',seed=20260929,data_seed=20260929,save_strategy='no',disable_tqdm=True,dataloader_num_workers=0)

def develop_chronos():
    d=load();pipe=base_chronos();preds={};qall={};y,m=truth(d,DEV24)
    for cov in [False,True]:
        name='chronos_cov' if cov else 'chronos_zero'
        p,q=predict_chronos(pipe,d,DEV24,cov=cov);preds[name]=p;qall[name]=q
        print(name,json.dumps(metrics(y,p,m)),flush=True)
    rp,cp,jointdiag=predict_joint(pipe,d,DEV24)
    preds['chronos_joint_raw']=rp;preds['chronos_joint_physical']=cp
    (ROOT/'results/chronos_joint_dev_audit.json').write_text(json.dumps(jointdiag,indent=2))
    print('chronos_joint_raw',json.dumps(metrics(y,rp,m)),flush=True)
    print('chronos_joint_physical',json.dumps(metrics(y,cp,m)),flush=True)
    for steps in [100,300]:
        ft=fit_chronos(pipe,d,14,steps)
        p,q=predict_chronos(ft,d,DEV24);name=f'chronos_lora_{steps}';preds[name]=p;qall[name]=q
        print(name,json.dumps(metrics(y,p,m)),flush=True)
        del ft;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(ROOT/'results/dev_chronos.npz',**preds)
    np.savez_compressed(ROOT/'results/dev_chronos_quantiles.npz',**qall)
    pd.DataFrame([dict(model=k,**metrics(y,v,m)) for k,v in preds.items()]).to_csv(ROOT/'results/dev_chronos.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['patch','chronos','all']);args=parser.parse_args()
    print('DEVICE',DEVICE,torch.__version__,flush=True)
    if args.mode in ['patch','all']:develop_patch()
    if args.mode in ['chronos','all']:develop_chronos()
