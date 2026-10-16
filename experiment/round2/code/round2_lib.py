import os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'code'))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
from common import load,truth,metrics,baseline,smooth
from deep_models import base_chronos,project_components
torch.set_num_threads(4)
DEV=list(range(14,22));TEST=list(range(22,31))

def inputs(d,o,kind='uni',full=False,h=96,context=1344):
    y=d['y'].reshape(57,-1);v=d['valid'].reshape(57,-1)
    end=o*96;start=0 if full else max(0,end-context)
    yy=np.where(v[:,start:end],y[:,start:end],np.nan).astype('float32')
    # Inference contexts use causal forward filling, training labels remain masked.
    if not full:yy=y[:,start:end].astype('float32')
    if kind=='uni':return [a for a in yy]
    prod=np.where(d['pvalid'].reshape(57,-1)[:,start:end],d['prod'].reshape(57,-1)[:,start:end],np.nan).astype('float32')
    if kind in ['peer','peerprod','parts']:
        out=[]
        for i in range(57):
            if kind=='parts':
                item={'target':np.stack([yy[i],prod[i],np.where(np.isfinite(prod[i]),yy[i]-prod[i],np.nan)])}
            else:
                ix=np.where((d['cells']==d['cells'][i])&(np.arange(57)!=i))[0]
                peer=np.nanmean(yy[ix],axis=0) if len(ix) else yy[i]
                peer=peer/max(np.nanmean(peer),1e-5)*np.nanmean(yy[i])
                item={'target':np.stack([yy[i],peer])}
                if kind=='peerprod':item['past_covariates']={'production':prod[i]}
            out.append(item)
        return out
    if kind in ['cell','cellprod','all']:
        groups=[np.arange(57)] if kind=='all' else [np.where(d['cells']==c)[0] for c in np.unique(d['cells'])]
        out=[]
        for ix in groups:
            item={'target':yy[ix]}
            if kind=='cellprod':item['past_covariates']={f'production_{j}':prod[j] for j in ix if np.isfinite(prod[j]).mean()>=.2}
            out.append(item)
        return out
    out=[]
    for i in range(57):
        item={'target':yy[i]};cov={}
        if 'cov' in kind:cov['production']=prod[i]
        if 'calendar' in kind:
            ti=np.arange(start,end+h)
            cal={'hour_sin':np.sin(2*np.pi*ti/96),'hour_cos':np.cos(2*np.pi*ti/96),'week_sin':np.sin(2*np.pi*(ti/96+2)/7),'week_cos':np.cos(2*np.pi*(ti/96+2)/7)}
            cov.update({k:a[:end-start].astype('float32') for k,a in cal.items()})
            item['future_covariates']={k:a[end-start:].astype('float32') for k,a in cal.items()}
        item['past_covariates']=cov;out.append(item)
    return out

def predict(pipe,d,origins,kind='uni',h=96,context=1344):
    out=[]
    for o in origins:
        inp=inputs(d,o,kind,h=h,context=context)
        if kind in ['cellprod','cell']:
            # Different cells carry different covariate schemas.
            batches=[pipe.predict_quantiles([item],prediction_length=h,quantile_levels=[.5],batch_size=64,context_length=context)[0][0] for item in inp]
        else:batches=pipe.predict_quantiles(inp,prediction_length=h,quantile_levels=[.5],batch_size=64,context_length=context)[0]
        if kind in ['cell','cellprod','all']:
            groups=[np.arange(57)] if kind=='all' else [np.where(d['cells']==c)[0] for c in np.unique(d['cells'])]
            a=np.empty((57,h))
            for ix,z in zip(groups,batches):a[ix]=z[:,:,0].cpu().numpy()
        elif kind=='parts':
            raw=np.stack([z[:,:,0].cpu().numpy() for z in batches])
            a=project_components(raw[:,0],raw[:,1],raw[:,2])[0]
            enough=d['pvalid'].reshape(57,-1)[:,max(0,o*96-context):o*96].mean(1)>=.2
            a[~enough]=np.clip(raw[~enough,0],0,1)
        else:a=np.stack([z[0,:,0].cpu().numpy() for z in batches])
        out.append(a)
    return np.stack(out,axis=1)

def train(base,d,cutoff,cfg,tag):
    kind=cfg['kind'];inp=inputs(d,cutoff,kind,full=True)
    kwargs=dict(prediction_length=96,finetune_mode=cfg['mode'],context_length=cfg['context'],learning_rate=cfg['lr'],num_steps=cfg['steps'],batch_size=cfg.get('batch',8),min_past=288,
        output_dir=str(R/'models'/f'{tag}_d{cutoff}'),logging_steps=100,report_to='none',seed=20260930,data_seed=20260930,save_strategy='no',disable_tqdm=True,dataloader_num_workers=0,bf16=True)
    return base.fit(inp,**kwargs)

CONFIGS={
 'lora_uni_lr1e5_s300':dict(mode='lora',kind='uni',lr=1e-5,steps=300,context=1344),
 'lora_uni_lr3e5_s300':dict(mode='lora',kind='uni',lr=3e-5,steps=300,context=1344),
 'lora_cov_lr1e5_s300':dict(mode='lora',kind='cov',lr=1e-5,steps=300,context=1344),
 'lora_cov_lr3e5_s300':dict(mode='lora',kind='cov',lr=3e-5,steps=300,context=1344),
 'lora_cov_lr1e4_s100':dict(mode='lora',kind='cov',lr=1e-4,steps=100,context=672),
 'lora_cell_lr3e5_s300':dict(mode='lora',kind='cell',lr=3e-5,steps=300,context=672,batch=16),
 'full_uni_lr1e6_s100':dict(mode='full',kind='uni',lr=1e-6,steps=100,context=1344),
 'full_uni_lr3e6_s100':dict(mode='full',kind='uni',lr=3e-6,steps=100,context=1344),
 'full_uni_lr1e6_s300':dict(mode='full',kind='uni',lr=1e-6,steps=300,context=1344),
 'full_cov_lr1e6_s100':dict(mode='full',kind='cov',lr=1e-6,steps=100,context=1344),
 'full_cov_lr3e6_s100':dict(mode='full',kind='cov',lr=3e-6,steps=100,context=1344),
 'full_cov_lr1e6_s300':dict(mode='full',kind='cov',lr=1e-6,steps=300,context=1344)
}
