import os,sys
if '--tirex' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TORCHDYNAMO_DISABLE']='1'
import argparse
from lib import *
import pandas as pd

def run(model,d,origins,old,tirex=False):
    factorized=[];reconciled=[];indices=[];scales=[]
    for j,o in enumerate(origins):
        y=d['y'][:,max(0,o-14):o].reshape(57,-1);mu=np.maximum(y.mean(1),1e-5);norm=y/mu[:,None]
        cells=np.unique(d['cells']);g=np.stack([norm[d['cells']==c].mean(0) for c in cells]);resid=norm.copy()
        for k,c in enumerate(cells):resid[d['cells']==c]-=g[k]
        inp=np.concatenate([g,resid]).astype('float32')
        if tirex:
            from tirex2 import TimeseriesType
            q=model.forecast([TimeseriesType(target=torch.tensor(a[None]),past_covariates=None,future_covariates=None) for a in inp],prediction_length=96,output_type='numpy',batch_size=8)
            p=np.stack([a[0,4] for a in q])
        else:
            q=model.predict_quantiles([a for a in inp],prediction_length=96,quantile_levels=[.5],context_length=1344,batch_size=64)[0]
            p=np.stack([a[0,:,0].cpu().numpy() for a in q])
        part=p[10:].copy();adjusted=old[:,j].copy()
        for k,c in enumerate(cells):
            ix=d['cells']==c;part[ix]=(part[ix]-part[ix].mean(0)+p[k])*mu[ix,None]
            adjusted[ix]+=(p[k]-(old[ix,j]/mu[ix,None]).mean(0))*mu[ix,None]
        factorized.append(part);reconciled.append(adjusted);indices.append(p[:10]);scales.append(mu)
        print('HIERARCHY',o,flush=True)
    return np.stack(factorized,1),np.stack(reconciled,1),np.stack(indices,1),np.stack(scales,1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tirex',action='store_true');ap.add_argument('--split',default='dev');args=ap.parse_args()
    kind='tirex' if args.tirex else 'chronos';d=load();origins=DEV if args.split=='dev' else TEST
    old=(np.load(R/f'results/chronos_{args.split}.npz')['native']+np.load(R/f'results/tirex_{args.split}.npz')['native'])/2
    if args.tirex:
        from tirex2 import load_model
        model=load_model(str(B/'models/tirex2-base'),device='cpu',use_flex_attention=False,compile=False)
    else:
        from deep_models import base_chronos
        model=base_chronos()
    fact,recon,agg,scales=run(model,d,origins,old,args.tirex);pred={};y,m=truth(d,origins);rows=[]
    for method,v in [('factor',fact),('recon',recon)]:
        for alpha in [.25,.5,1]:
            name=f'hier_{kind}_{method}_a{alpha}';pred[name]=np.clip((1-alpha)*old+alpha*v,0,1);rows.append(dict(model=name,**metrics(y,pred[name],m)))
    np.savez_compressed(R/f'results/hierarchy_{kind}_{args.split}.npz',**pred)
    np.savez_compressed(R/f'results/hierarchy_{kind}_{args.split}_components.npz',factorized=fact,reconciled=recon,cell_index=agg,past_scale=scales)
    pd.DataFrame(rows).sort_values('wape').to_csv(R/f'results/hierarchy_{kind}_{args.split}.csv',index=False)
    print(pd.DataFrame(rows).sort_values('wape').to_string(index=False),flush=True)

if __name__=='__main__':main()
