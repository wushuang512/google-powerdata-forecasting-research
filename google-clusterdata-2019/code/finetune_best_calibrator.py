"""Fine tune the existing best fixed-ensemble residual coefficients against macro WAPE."""
import sys,json,copy,pickle,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round8'
sys.path.insert(0,str(B/'round5/code'))
from lib5 import load,truth,metrics,TRAIN,DEV,TEST,anchor,save_json,DEVICE
from calibration import bank,features,fit
from assess import comparisons
CONFIGS={'lr3e5_anchor5':dict(lr=3e-5,penalty=5),'lr3e5_anchor50':dict(lr=3e-5,penalty=50),'lr1e4_anchor50':dict(lr=1e-4,penalty=50)}
STEPS=[25,100,300]
def train(d,b,origins,cfg,callback=None,steps=300):
    model,names=fit(d,b,origins,100000);scaler,reg=model.steps[0][1],model.steps[1][1];xs=[];ys=[];bs=[];ws=[]
    den=np.stack([d['y'][:,o]*d['valid'][:,o] for o in origins],1).sum((1,2)).clip(1e-6)
    for o in origins:
        x,base,_=features(d,o,b);m=d['valid'][:,o].ravel();xs.append(scaler.transform(x)[m]);ys.append(d['y'][:,o].ravel()[m]);bs.append(base.ravel()[m]);ws.append(np.repeat(1/den,96)[m]/57)
    tensors=[torch.as_tensor(np.concatenate(v),dtype=torch.float32,device=DEVICE) for v in [xs,ys,bs,ws]];x,y,base,w=tensors;coeff=torch.nn.Parameter(torch.as_tensor(reg.coef_,dtype=torch.float32,device=DEVICE).clone());inter=torch.nn.Parameter(torch.tensor(reg.intercept_,dtype=torch.float32,device=DEVICE));c0=coeff.detach().clone();i0=inter.detach().clone();opt=torch.optim.Adam([coeff,inter],lr=cfg['lr']);hist=[]
    for step in range(1,steps+1):
        p=(base+.5*(x@coeff+inter)).clamp(0,1);loss=((p-y).abs()*w).sum()+cfg['penalty']*((coeff-c0).square().sum()+(inter-i0).square());opt.zero_grad();loss.backward();opt.step()
        if step%25==0:hist.append(dict(step=step,objective=float(loss.detach()),coefficient_change_l2=float((coeff-c0).norm().detach())));print('TRAIN',cfg,hist[-1],flush=True)
        if callback and step in STEPS:
            z=copy.deepcopy(model);z.steps[1][1].coef_=coeff.detach().cpu().numpy().astype(float);z.steps[1][1].intercept_=float(inter.detach());callback(z,step)
    reg.coef_=coeff.detach().cpu().numpy().astype(float);reg.intercept_=float(inter.detach());return model,hist
def predict(model,d,b,origins):return np.clip(np.stack([base+.5*model.predict(x).reshape(57,96) for x,base,_ in [features(d,o,b) for o in origins]],1),0,1)
def main():
    for sub in ['code','docs','models','results']:(R/sub).mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,checkpoints=STEPS,training_origins=TRAIN,development_origins=DEV,final_training_origins=TRAIN+DEV,test_origins=TEST,objective='Exact domain-macro WAPE for existing 50% Ridge residual + frozen three-model foundation ensemble. Quadratic anchor around fitted Ridge coefficients.',rule='Early4 development choose checkpoint; late4 choose alpha by paired-cell one-standard-error rule preferring alpha0. Test scores loaded only after hash freeze.',independent_confirmation=False,complete_cpu_input=False)
    save_json(R/'docs/protocol.json',protocol);d=load();b=bank(['train','dev']);y,m=truth(d,DEV);prior=.5*anchor('dev')+.5*np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000'];p={};rows=[]
    for name,cfg in CONFIGS.items():
        def ck(model,step):
            key=f'{name}_s{step}';p[key]=predict(model,d,b,DEV);row=dict(model=key,**metrics(y,p[key],m));rows.append(row)
            with (R/'models'/f'{key}_dev.pkl').open('wb') as f:pickle.dump(model,f)
            print('DEV',row,flush=True)
        _,hist=train(d,b,TRAIN,cfg,ck);save_json(R/'results'/f'{name}_history.json',hist)
    best=min(p,key=lambda k:metrics(y[:,:4],p[k][:,:4],m[:,:4])['wape']);candidates=[]
    for a in [0,.1,.25,.5,1]:
        z=(1-a)*prior[:,4:]+a*p[best][:,4:];per=100*(np.abs(y[:,4:]-z)*m[:,4:]).sum((1,2))/(y[:,4:]*m[:,4:]).sum((1,2));ref=100*(np.abs(y[:,4:]-prior[:,4:])*m[:,4:]).sum((1,2))/(y[:,4:]*m[:,4:]).sum((1,2));v=pd.DataFrame(dict(cell=d['cells'],difference=ref-per)).groupby('cell').difference.mean().to_numpy();candidates.append(dict(alpha=a,wape=float(per.mean()),paired_cell_se=float(v.std(ddof=1)/np.sqrt(len(v)))))
    win=min(candidates,key=lambda z:z['wape']);pick=min((z for z in candidates if z['wape']<=win['wape']+win['paired_cell_se']+1e-12),key=lambda z:z['alpha']);s=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=best,chosen=pick,validation=candidates,protocol_sha256=hashlib.sha256((R/'docs/protocol.json').read_bytes()).hexdigest());save_json(R/'results/selection.json',s);(R/'results/selection.sha256').write_text(hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest());np.savez_compressed(R/'results/dev_predictions.npz',**p,prior=prior);pd.DataFrame(rows).to_csv(R/'results/dev_metrics.csv',index=False);print('FROZEN',s,flush=True)
    name,step=best.rsplit('_s',1);b=bank(['train','dev','test']);model,hist=train(d,b,TRAIN+DEV,CONFIGS[name],steps=int(step));ft=predict(model,d,b,TEST);prior=np.load(B/'round5/results/test_candidates.npz')['selected'];selected=np.clip((1-pick['alpha'])*prior+pick['alpha']*ft,0,1);ty,tm=truth(d,TEST);scores={k:metrics(ty,v,tm) for k,v in dict(prior=prior,finetuned=ft,selected=selected).items()};np.savez_compressed(R/'results/test_predictions.npz',prior=prior,finetuned=ft,selected=selected);save_json(R/'results/summary.json',dict(scores=scores,selection=s,comparison_vs_previous=comparisons(ty,tm,selected,prior,d['cells']),independent_confirmation=False));save_json(R/'results/final_training.json',hist)
    with (R/'models'/f'{best}_final.pkl').open('wb') as f:pickle.dump(model,f)
    print('TEST',scores,flush=True)
if __name__=='__main__':main()
