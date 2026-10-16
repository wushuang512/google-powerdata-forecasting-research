"""Adapt TimesFM component inside the actual best calibrated three-model ensemble."""
import sys,copy,gc,json,pickle,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'experiment';R=B/'round8b'
sys.path.insert(0,str(B/'round5/code'))
from lib5 import load,truth,metrics,TRAIN,DEV,TEST,save_json,DEVICE,anchor,tensor
from calibration import bank,features,fit
import finetune_tf3 as ft
from timesfm3 import TimesFM3Forecaster
from timesfm3.torch.timesfm3_forecaster import linear_interpolation
from assess import comparisons
CONFIGS={'best_head_lr3e5_teacher005':dict(scope='head',lr=3e-5,teacher=.05),'best_tail_lr1e6_teacher01':dict(scope='tail',lr=1e-6,teacher=.1)}
STEPS=[100,300,600]
def prior(split):return .5*anchor('dev')+.5*np.load(B/'round5/results/calibration_dev.npz')['residual_ridge100000'] if split=='dev' else np.load(B/'round5/results/test_candidates.npz')['selected']
def shifts(d,cutoff):
    origins=TRAIN if cutoff==14 else TRAIN+DEV;b=bank(['train','dev']);model,_=fit(d,b,origins,100000);result={}
    for split,oo in [('train',TRAIN),('dev',DEV)]:
        if split=='dev' and cutoff==14:continue
        c=np.load(B/f'round3/results/chronos_{split}.npz')['native'];t=np.load(B/f'round3/results/tirex_{split}.npz')['native']
        for j,o in enumerate(oo):
            x,_,_=features(d,o,b);result[o]=.25*c[:,j]+.25*t[:,j]+.5*model.predict(x).reshape(57,96)
    return result
def train(d,cutoff,cfg,steps=600,callback=None):
    torch.manual_seed(20261001);rng=np.random.default_rng(20261001);shiftbank=shifts(d,cutoff);f=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device=DEVICE,per_core_batch_size=4,local_files_only=True);model=f.model;teacher=copy.deepcopy(model).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    for p in model.parameters():p.requires_grad_(False)
    for p in model.output_head.parameters():p.requires_grad_(True)
    if cfg['scope']=='tail':
        for p in model.transformer_stack.layers[-1].parameters():p.requires_grad_(True)
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01);model.eval();qs=tensor(np.array(model.quantiles,dtype='float32'));hist=[]
    for step in range(1,steps+1):
        o=int(rng.choice(list(shiftbank)));cells=np.unique(d['cells']);sizes=np.array([(d['cells']==c).sum() for c in cells]);ci=int(rng.choice(len(cells),p=sizes/57));ix=np.where(d['cells']==cells[ci])[0];inp=ft.inputs(d,o,'cellprod',context=1344)[ci];target=tensor(linear_interpolation(inp['target'])[None]);pc=inp.get('past_covariates',{});cov=tensor(linear_interpolation(np.stack(list(pc.values())))[None]) if pc else None;y=tensor(d['y'][ix,o].astype('float32'));m=tensor(d['valid'][ix,o]);shift=tensor(shiftbank[o][ix].astype('float32'))
        with torch.no_grad():tq=teacher.decode(target,horizon=96,past_only_covariates=cov)[:,:len(ix)][0]
        q=type(model).decode.__wrapped__(model,target,horizon=96,past_only_covariates=cov)[:,:len(ix)][0];mix=q.sort(-1).values*.5+shift[:,:,None];den=(y.abs()*m).sum(-1).clamp_min(1e-4);point=((mix[:,:,4].clamp(0,1)-y).abs()*m).sum(-1)/den;error=y[:,:,None]-mix;pin=(torch.maximum(qs*error,(qs-1)*error)*m[:,:,None]).sum((1,2))/den/9;anch=(q-tq).abs().mean((1,2))/(den/m.sum(-1).clamp_min(1));bounds=(torch.relu(-mix[:,:,4])+torch.relu(mix[:,:,4]-1)).mean();loss=point.mean()+.05*pin.mean()+cfg['teacher']*anch.mean()+.1*bounds+.05*torch.relu(q[:,:,:-1]-q[:,:,1:]).mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params);torch.nn.utils.clip_grad_norm_(params,1);opt.step()
        for group in opt.param_groups:group['lr']=cfg['lr']*(.2+.8*(1-step/600))
        if step%50==0:hist.append(dict(step=step,point=float(point.mean().detach()),teacher=float(anch.mean().detach()),loss=float(loss.detach())));print('TRAIN',cutoff,cfg,hist[-1],flush=True)
        if callback and step in STEPS:callback(f,step)
    del teacher;gc.collect();torch.cuda.empty_cache();return f,hist
def main():
    for sub in ['code','docs','models','results']:(R/sub).mkdir(parents=True,exist_ok=True)
    save_json(R/'docs/protocol.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),registry=CONFIGS,checkpoints=STEPS,training_origins=TRAIN,development_origins=DEV,test_origins=TEST,objective='Optimize actual best calibrated three-model ensemble with only TimesFM component adapted. Calibration is fixed for a split; teacher shift uses supervised training data only.',selection='Early4 choose member, late4 one-standard-error choose alpha0/.1/.25/.5/1. Freeze before new test score.',independent_confirmation=False,cpu_input=False,license='TimesFM Non-Commercial v1.0, research only'));d=load();orig=np.load(B/'round3/results/tf3_dev.npz')['tf3_cellprod'];base=prior('dev');y,m=truth(d,DEV);p={};rows=[]
    for name,cfg in CONFIGS.items():
        def ck(f,step):
            key=f'{name}_s{step}';raw=np.clip(ft.forecast(f,d,DEV,ft.BASECFG['tf3_cellprod']),0,1);p[key]=np.clip(base+.5*(raw-orig),0,1);ft.save_delta(f,R/'models'/f'{key}_dev.pt',cfg,14,step);row=dict(model=key,**metrics(y,p[key],m));rows.append(row);np.savez_compressed(R/'results/dev_predictions.npz',**p);pd.DataFrame(rows).to_csv(R/'results/dev_metrics.csv',index=False);print('DEV',row,flush=True)
        f,h=train(d,14,cfg,callback=ck);save_json(R/'results'/f'{name}_history.json',h);del f;gc.collect();torch.cuda.empty_cache()
    best=min(p,key=lambda k:metrics(y[:,:4],p[k][:,:4],m[:,:4])['wape']);cand=[]
    for a in [0,.1,.25,.5,1]:
        z=(1-a)*base[:,4:]+a*p[best][:,4:];per=100*(np.abs(y[:,4:]-z)*m[:,4:]).sum((1,2))/(y[:,4:]*m[:,4:]).sum((1,2));ref=100*(np.abs(y[:,4:]-base[:,4:])*m[:,4:]).sum((1,2))/(y[:,4:]*m[:,4:]).sum((1,2));v=pd.DataFrame(dict(cell=d['cells'],difference=ref-per)).groupby('cell').difference.mean().to_numpy();cand.append(dict(alpha=a,wape=float(per.mean()),paired_cell_se=float(v.std(ddof=1)/np.sqrt(len(v)))))
    win=min(cand,key=lambda z:z['wape']);pick=min((z for z in cand if z['wape']<=win['wape']+win['paired_cell_se']+1e-12),key=lambda z:z['alpha']);s=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=best,chosen=pick,validation=cand,protocol_sha256=hashlib.sha256((R/'docs/protocol.json').read_bytes()).hexdigest());save_json(R/'results/selection.json',s);(R/'results/selection.sha256').write_text(hashlib.sha256((R/'results/selection.json').read_bytes()).hexdigest());print('FROZEN',s,flush=True)
    name,step=best.rsplit('_s',1);f,h=train(d,22,CONFIGS[name],steps=int(step));raw=np.clip(ft.forecast(f,d,TEST,ft.BASECFG['tf3_cellprod']),0,1);original=np.load(B/'round3/results/tf3_test.npz')['tf3_cellprod'];base=prior('test');candidate=np.clip(base+.5*(raw-original),0,1);selected=np.clip((1-pick['alpha'])*base+pick['alpha']*candidate,0,1);ty,tm=truth(d,TEST);scores={k:metrics(ty,z,tm) for k,z in dict(prior=base,candidate=candidate,selected=selected).items()};np.savez_compressed(R/'results/test_predictions.npz',prior=base,candidate=candidate,selected=selected);ft.save_delta(f,R/'models'/f'{best}_final.pt',CONFIGS[name],22,int(step));save_json(R/'results/summary.json',dict(scores=scores,selection=s,comparison_vs_previous=comparisons(ty,tm,selected,base,d['cells']),independent_confirmation=False));save_json(R/'results/final_training.json',h);print('TEST',scores,flush=True)
if __name__=='__main__':main()
