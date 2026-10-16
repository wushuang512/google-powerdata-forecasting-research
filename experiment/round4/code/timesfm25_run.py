import sys,json,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'round3/code'))
from lib import load,truth,metrics,DEV,TEST,np,torch
import timesfm,pandas as pd
CONFIGS={f'tf25_c{c}_flip{int(f)}':dict(context=c,flip=f) for c in [672,1344,2016] for f in [False,True]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='dev');a=ap.parse_args();d=load();origins=DEV if a.split=='dev' else TEST
    model=timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(B/'models/timesfm25-base'),torch_compile=False,local_files_only=True)
    configs=CONFIGS
    if a.split=='test':
        s=json.loads((R/'results/selection.json').read_text());configs={k:v for k,v in CONFIGS.items() if k in s['needed']}
    path=R/f'results/tf25_{a.split}.npz';p=dict(np.load(path)) if path.exists() else {};y,m=truth(d,origins)
    for name,cfg in configs.items():
        model.compile(timesfm.ForecastConfig(max_context=cfg['context'],max_horizon=128,normalize_inputs=True,per_core_batch_size=8,use_continuous_quantile_head=True,force_flip_invariance=cfg['flip'],infer_is_positive=True,fix_quantile_crossing=True))
        if name not in p:
            out=[]
            for o in origins:
                x=d['y'].reshape(57,-1)[:,max(0,o*96-cfg['context']):o*96]
                pred,_=model.forecast(horizon=96,inputs=[z for z in x]);out.append(pred);print(name,o,flush=True)
            p[name]=np.clip(np.stack(out,axis=1),0,1);np.savez_compressed(path,**p)
        print('RESULT',name,metrics(y,p[name],m),flush=True)
    pd.DataFrame([dict(model=k,**metrics(y,v,m)) for k,v in p.items()]).to_csv(R/f'results/tf25_{a.split}.csv',index=False)
if __name__=='__main__':main()
