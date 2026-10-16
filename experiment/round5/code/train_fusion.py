import argparse,gc,time
from lib5 import *
from fusion_model import FusionMoE
CONFIGS={'fusion32':dict(width=32,cross=True,freq=True,sparse=True),'fusion16':dict(width=16,cross=True,freq=True,sparse=True),'no_cross':dict(width=32,cross=False,freq=True,sparse=True),'no_frequency':dict(width=32,cross=True,freq=False,sparse=True),'dense_moe':dict(width=32,cross=True,freq=True,sparse=False)}
CHECKPOINTS=[4,10,20]
def training_data(d,cutoff):
    ends=np.arange(672,cutoff*96-96+1,12);xx,cc,ci=history(d,ends);ix=np.tile(np.arange(57),len(ends));ee=np.repeat(ends,57)
    arrays=[]
    for key in ['y','prod','valid','pvalid']:
        a=d[key].reshape(57,-1);arrays.append(np.stack([a[i,e:e+96] for i,e in zip(ix,ee)]).astype('float32'))
    return tuple(tensor(a) for a in (xx,cc,ci,*arrays))
@torch.inference_mode()
def predict(net,d,origins):
    net.eval();x,c,ci=history(d,np.array(origins)*96);allp=[];allw=[];gates=[]
    for start in range(0,len(x),128):
        pq,wq,_,g=net(tensor(x[start:start+128]),tensor(c[start:start+128]),tensor(ci[start:start+128]));allp.append(pq.cpu().numpy());allw.append(wq.cpu().numpy());gates.append(g.cpu().numpy())
    pp=np.concatenate(allp).reshape(len(origins),57,96,3).transpose(1,0,2,3);ww=np.concatenate(allw).reshape(len(origins),57,96,3).transpose(1,0,2,3)
    return pp,ww,np.concatenate(gates)
def train(d,cutoff,cfg,seed,epochs,callback=None):
    torch.manual_seed(seed);np.random.seed(seed);net=FusionMoE(**cfg).to(DEVICE);data=training_data(d,cutoff);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.05);hist=[];qs=torch.tensor([.1,.5,.9],device=DEVICE)
    print('PARAMETERS',sum(p.numel() for p in net.parameters()),'samples',len(data[0]),flush=True)
    for ep in range(1,epochs+1):
        net.train();losses=[]
        for ix in torch.randperm(len(data[0]),device=DEVICE).split(64):
            x,c,ci,y,w,vm,wm=(a[ix] for a in data);pq,wq,balance,g=net(x,c,ci);den=(y.abs()*vm).sum(-1).clamp_min(1e-4)
            main=((pq[:,:,1]-y).abs()*vm).sum(-1)/den
            error=y[:,:,None]-pq;pin=(torch.maximum(qs*error,(qs-1)*error)*vm[:,:,None]).sum((1,2))/den/3
            werr=w[:,:,None]-wq;wpin=(torch.maximum(qs*werr,(qs-1)*werr)*wm[:,:,None]).sum((1,2))/den/3
            aux=((wq[:,:,1]-w).abs()*wm).sum(-1)/den
            # Shrink residuals toward the causal last-day profile; no foundation future labels used as inputs.
            anchor_loss=(pq[:,:,1]-x[:,-96:,0]).abs().sum(-1)/den
            loss=main.mean()+.2*pin.mean()+.15*aux.mean()+.1*wpin.mean()+.1*anchor_loss.mean()+.002*balance
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(float(loss.detach()))
        hist.append(dict(epoch=ep,loss=float(np.mean(losses))));print('TRAIN',cutoff,seed,hist[-1],flush=True)
        if callback and ep in CHECKPOINTS:callback(net,ep)
    return net,hist
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='dev');ap.add_argument('--config');ap.add_argument('--seed',type=int,default=20261001);a=ap.parse_args();d=load();origins=DEV if a.split=='dev' else TEST;cutoff=14 if a.split=='dev' else 22;prefix='dev' if a.split=='dev' else 'test'
    configs=CONFIGS if a.config is None else {a.config:CONFIGS[a.config]};p={};w={};quant={};rows=[];y,m=truth(d,origins)
    if a.split=='test':
        sel=json.loads((R/'results/selection.json').read_text());chosen=sel['chosen']['fusion'];name,ep=chosen.rsplit('_e',1);configs={name:CONFIGS[name]};maxep=int(ep)
    else:maxep=20
    suffix='' if a.seed==20261001 else f'_seed{a.seed}'
    for name,cfg in configs.items():
        def checkpoint(net,ep):
            key=f'{name}_e{ep}';pp,ww,g=predict(net,d,origins);p[key]=pp[:,:,:,1];w[key]=ww[:,:,:,1];quant[key]=pp
            row=dict(model=key,**metrics(y,p[key],m));rows.append(row);torch.save(dict(state=net.state_dict(),config=cfg,cutoff=cutoff,epoch=ep,seed=a.seed),R/'models'/f'{key}_{prefix}{suffix}.pt');np.savez_compressed(R/f'results/fusion_{prefix}{suffix}.npz',**p);np.savez_compressed(R/f'results/fusion_quantiles_{prefix}{suffix}.npz',**quant);np.savez_compressed(R/f'results/proxy_{prefix}{suffix}.npz',**w);pd.DataFrame(rows).to_csv(R/f'results/fusion_{prefix}{suffix}.csv',index=False);print('RESULT',row,flush=True)
            save_json(R/'results'/f'{key}_gate_{prefix}{suffix}.json',dict(mean_probability=g.mean(0).tolist(),route_counts=np.bincount(np.argsort(g,axis=1)[:,-2:].ravel(),minlength=3).tolist()))
        net,hist=train(d,cutoff,cfg,a.seed,maxep,checkpoint if a.split=='dev' else None)
        if a.split=='test':checkpoint(net,maxep)
        save_json(R/'results'/f'{name}_history_{prefix}{suffix}.json',hist);del net;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
