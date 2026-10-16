import os,sys
# Separate processes let Chronos use GPU and TiRex use native CPU kernels.
if '--tirex' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TORCHDYNAMO_DISABLE']='1'
import argparse,time
from lib import *
import pandas as pd

CONFIGS={
 'native':dict(factor=1,days=14,flip=False,diff=True,sign=False),
 'coarse2':dict(factor=2,days=14,flip=False,diff=True,sign=False),
 'coarse4':dict(factor=4,days=14,flip=False,diff=True,sign=False),
 'coarse8':dict(factor=8,days=14,flip=False,diff=True,sign=False),
 'context7':dict(factor=1,days=7,flip=False,diff=True,sign=False),
 'context21':dict(factor=1,days=21,flip=False,diff=True,sign=False),
 'mirror':dict(factor=1,days=14,flip=True,diff=True,sign=False),
 'no_diff':dict(factor=1,days=14,flip=False,diff=False,sign=False),
 'sign_average':dict(factor=1,days=14,flip=False,diff=True,sign=True)
}

def forecast(model,d,origins,cfg,tirex=False):
    out=[];factor=cfg['factor'];h=96//factor
    for o in origins:
        a=coarse_inputs(d,o,factor,cfg['days'],cfg['flip'])
        if tirex:
            from tirex2 import TimeseriesType
            inp=[TimeseriesType(target=torch.tensor(row[None]),past_covariates=None,future_covariates=None) for row in a]
            q=model.forecast(inp,prediction_length=h,output_type='numpy',batch_size=8,tta_diff=cfg['diff'],tta_sign_flip=cfg['sign'])
            p=np.stack([x[0,4] for x in q])
        else:
            q=model.predict_quantiles([x for x in a],prediction_length=h,quantile_levels=[.5],context_length=max(a.shape[-1],288//factor),batch_size=64)[0]
            p=np.stack([x[0,:,0].cpu().numpy() for x in q])
        if cfg['flip']:p=1-p
        out.append(expand(p,factor))
        print('FORECAST',o,cfg,flush=True)
    return np.stack(out,axis=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tirex',action='store_true');ap.add_argument('--split',default='dev');args=ap.parse_args()
    kind='tirex' if args.tirex else 'chronos';d=load();origins={'train':TRAIN,'dev':DEV,'test':TEST}[args.split]
    path=R/f'results/{kind}_{args.split}.npz';pred=dict(np.load(path)) if path.exists() else cached_native(args.split,kind)
    if args.tirex:
        from tirex2 import load_model
        model=load_model(str(B/'models/tirex2-base'),device='cpu',use_flex_attention=False,compile=False)
    else:
        from deep_models import base_chronos
        model=base_chronos()
    cfgs=CONFIGS.copy()
    if not args.tirex:
        for name in ['no_diff','sign_average']:cfgs.pop(name)
    if args.split=='train':cfgs={k:v for k,v in cfgs.items() if k=='native'}
    if args.split=='test':
        sel=json.loads((R/'results/selection.json').read_text());needed=sel['needed']
        cfgs={k:v for k,v in cfgs.items() if f'{kind}_{k}' in needed or k=='native'}
    rows=[];y,m=truth(d,origins)
    for name,cfg in cfgs.items():
        if name not in pred:pred[name]=forecast(model,d,origins,cfg,args.tirex)
        np.savez_compressed(path,**pred)
        row=dict(model=f'{kind}_{name}',**metrics(y,pred[name],m));rows.append(row);print('RESULT',json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(R/f'results/{kind}_{args.split}.csv',index=False)
    (R/'docs'/f'{kind}_configs.json').write_text(json.dumps(cfgs,indent=2))

if __name__=='__main__':main()
