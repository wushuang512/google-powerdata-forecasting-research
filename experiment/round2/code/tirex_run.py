import os
# Official CPU kernels avoid CUDA-toolkit and compiler requirements on Windows.
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TORCHDYNAMO_DISABLE']='1'
import json,time,argparse
from round2_lib import *
from tirex2 import load_model,TimeseriesType
import pandas as pd

def tirex_inputs(d,o,kind,h,context):
    src=inputs(d,o,kind,h=h,context=context);out=[]
    for a in src:
        if isinstance(a,np.ndarray):a={'target':a}
        target=torch.tensor(a['target']).reshape(-1,len(a['target'][-1]) if np.ndim(a['target'])==2 else len(a['target']))
        cov=a.get('past_covariates',{});future=a.get('future_covariates',{})
        past=[v for k,v in cov.items() if k not in future]
        fut=[np.concatenate([cov[k],v]) for k,v in future.items()]
        out.append(TimeseriesType(target=target,past_covariates=torch.tensor(np.stack(past)) if past else None,future_covariates=torch.tensor(np.stack(fut)) if fut else None))
    return out

def predict_tirex(model,d,origins,kind='uni',h=96,context=1344):
    out=[]
    for o in origins:
        ts=tirex_inputs(d,o,kind,h,context)
        # Official checkpoint supports at most 320 steps. Extend causally by
        # appending median forecasts, never observed future target/covariate data.
        work=ts;parts=[[] for _ in ts];done=0
        while done<h:
            chunk=min(320,h-done)
            zz=model.forecast(work,prediction_length=chunk,output_type='numpy',batch_size=8)
            assert all(q.shape[-1]==chunk for q in zz)
            new=[];done+=chunk
            for i,(item,q) in enumerate(zip(work,zz)):
                med=q[:,4,:];parts[i].append(med)
                length=ts[i].target.shape[-1]
                target=torch.cat([item.target,torch.tensor(med)],dim=-1)[:,-length:]
                past=None if item.past_covariates is None else torch.cat([item.past_covariates,torch.full((item.past_covariates.shape[0],chunk),float('nan'))],dim=-1)[:,-length:]
                future=None if ts[i].future_covariates is None else ts[i].future_covariates[:,done:]
                new.append(TimeseriesType(target=target,past_covariates=past,future_covariates=future))
            work=new
        z=[np.concatenate(pp,axis=-1) for pp in parts]
        if kind in ['cell','cellprod','all']:
            groups=[np.arange(57)] if kind=='all' else [np.where(d['cells']==c)[0] for c in np.unique(d['cells'])]
            a=np.empty((57,h))
            for ix,q in zip(groups,z):a[ix]=q
        else:a=np.stack([q[0] for q in z])
        out.append(a);print('TIREX',kind,'origin',o,flush=True)
    return np.stack(out,axis=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='dev');ap.add_argument('--origin',type=int,default=31);ap.add_argument('--days',type=int,default=1);args=ap.parse_args()
    d=load();model=load_model(str(B/'models/tirex2-base'),device='cpu',use_flex_attention=False,compile=False)
    origins=DEV if args.split=='dev' else ([22,23,24] if args.split=='test7' else TEST);h=672 if args.split=='test7' else 96
    if args.split in ['deploy','legacy']:origins=[args.origin];h=args.days*96
    path=R/f'results/tirex_{args.split}.npz';preds=dict(np.load(path)) if path.exists() else {}
    if args.split in ['deploy','legacy']:path=R/f'results/tirex_{args.split}_{args.origin}_{args.days}day.npz';preds={}
    configs={'tirex_uni':('uni',1344),'tirex_cov':('cov',1344),'tirex_cell':('cell',1344),'tirex_cellprod':('cellprod',1344),'tirex_calendar':('calendar',1344),'tirex_uni_ctx672':('uni',672),'tirex_uni_ctx2048':('uni',2048)}
    if args.split!='dev':
        selected=json.loads((R/'results/selection.json').read_text())
        configs={k:v for k,v in configs.items() if k in selected['needed'] or k=='tirex_uni'}
    if args.split=='legacy':configs={'tirex_uni':('uni',1344)}
    rows=[]
    if args.split not in ['deploy','legacy']:y,m=truth(d,origins,h//96)
    for name,(kind,ctx) in configs.items():
        if name not in preds or preds[name].shape!=(57,len(origins),h):
            started=time.time();preds[name]=predict_tirex(model,d,origins,kind,h,ctx)
            np.savez_compressed(path,**preds);print('SECONDS',name,time.time()-started,flush=True)
        if args.split in ['deploy','legacy']:continue
        score=dict(model=name,**metrics(y,preds[name],m));rows.append(score);print('RESULT',json.dumps(score),flush=True)
        pd.DataFrame(rows).to_csv(R/f'results/tirex_{args.split}.csv',index=False)

if __name__=='__main__':main()
