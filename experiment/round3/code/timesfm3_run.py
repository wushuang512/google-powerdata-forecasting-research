import argparse,time,traceback
from lib import *
from round2_lib import inputs
from deep_models import project_components
from timesfm3 import TimesFM3Forecaster
import pandas as pd

CONFIGS={
 'tf3_uni':dict(kind='uni',context=1344,symmetric=False),
 'tf3_uni_symmetric':dict(kind='uni',context=1344,symmetric=True),
 'tf3_uni_context2016':dict(kind='uni',context=2016,symmetric=False),
 'tf3_cov':dict(kind='cov',context=1344,symmetric=False),
 'tf3_cell':dict(kind='cell',context=1344,symmetric=False),
 'tf3_cellprod':dict(kind='cellprod',context=1344,symmetric=False),
 'tf3_peerprod':dict(kind='peerprod',context=1344,symmetric=False),
 'tf3_parts':dict(kind='parts',context=1344,symmetric=False)
}

def forecast(model,d,origins,cfg,h=96):
    out=[];kind=cfg['kind'];ctx=cfg['context']
    for o in origins:
        src=inputs(d,o,kind,h=h,context=ctx);targets=[];covs=[]
        for a in src:
            if isinstance(a,np.ndarray):targets.append(a);covs.append(None)
            else:
                targets.append(a['target']);cov=a.get('past_covariates',{});covs.append(np.stack(list(cov.values())) if cov else None)
        if kind in ['cell','cellprod']:
            q=[model.predict(target,horizon=h,past_only_covariates=cov,use_symmetric_averaging=cfg['symmetric']) for target,cov in zip(targets,covs)]
        else:q=list(model.predict_batch(contexts=targets,horizon=h,past_only_covariates=covs,use_symmetric_averaging=cfg['symmetric']))
        if kind in ['cell','cellprod']:
            p=np.empty((57,h));groups=[np.where(d['cells']==c)[0] for c in np.unique(d['cells'])]
            for ix,output in zip(groups,q):p[ix]=output.forecast
        elif kind=='parts':
            raw=np.stack([x.forecast for x in q]);p=project_components(raw[:,0],raw[:,1],raw[:,2])[0]
            enough=d['pvalid'].reshape(57,-1)[:,max(0,o*96-ctx):o*96].mean(1)>=.2
            p[~enough]=np.clip(raw[~enough,0],0,1)
        else:p=np.stack([x.forecast.reshape(-1,h)[0] for x in q])
        out.append(p);print('TF3',kind,o,flush=True)
    return np.stack(out,axis=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='dev');args=ap.parse_args()
    d=load();model=TimesFM3Forecaster.from_pretrained(str(B/'models/timesfm3-base'),device='cuda',per_core_batch_size=4,local_files_only=True)
    origins=DEV if args.split=='dev' else TEST;path=R/f'results/tf3_{args.split}.npz';p=dict(np.load(path)) if path.exists() else {};rows=[];fails=[]
    cfgs=CONFIGS.copy()
    if args.split=='test':
        s=json.loads((R/'results/selection.json').read_text());needed=s['needed']+[json.loads((R/'results/secondary.json').read_text())['single_model']];cfgs={k:v for k,v in cfgs.items() if k in needed}
    y,m=truth(d,origins)
    for name,cfg in cfgs.items():
        try:
            if name not in p:p[name]=forecast(model,d,origins,cfg)
            row=dict(model=name,**metrics(y,p[name],m));rows.append(row);np.savez_compressed(path,**p)
            pd.DataFrame(rows).to_csv(R/f'results/tf3_{args.split}.csv',index=False);print('RESULT',json.dumps(row),flush=True)
        except Exception as e:traceback.print_exc();fails.append(dict(model=name,error=str(e)));torch.cuda.empty_cache()
    (R/'docs/tf3_configs.json').write_text(json.dumps(cfgs,indent=2));(R/f'results/tf3_{args.split}_failures.json').write_text(json.dumps(fails,indent=2))

if __name__=='__main__':main()
