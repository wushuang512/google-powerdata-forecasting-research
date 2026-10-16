"""Run saved, frozen models without retraining. Default origin is the dataset end."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse,json,pickle,gc
import numpy as np
import pandas as pd
import torch
from common import *
from classical import predict_tree
from deep_models import base_chronos,predict_chronos,predict_joint,PatchModel,predict_patch,DEVICE
from evaluate import history_predict
from selection import peak_adjust

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--days',type=int,choices=[1,7],default=1);parser.add_argument('--origin',type=int,default=31);a=parser.parse_args()
    d=load();assert 14<=a.origin<=d['y'].shape[1]
    s=json.loads((ROOT/'results/selection24.json').read_text());cfg=json.loads((ROOT/'results/classical_configs.json').read_text())
    if a.days==1:primary=s['primary24'];weights=s['weights']
    else:
        s7=json.loads((ROOT/'results/selection7.json').read_text());primary=s7['primary7'];weights=s7['weights']
    needed=[k for k,w in weights.items() if w>1e-8] if primary.startswith('ensemble') else [primary]
    preds={};pipe=None
    for name in needed:
        if name in ['last','week','mean3','mean7','level']:p=baseline(d,[a.origin],a.days,name)
        elif name.startswith('history'):p=history_predict(d,[a.origin],cfg[name],a.days)
        elif name.startswith(('lgb','ridge')):
            with (ROOT/f'models/{name}_final.pkl').open('rb') as f:model=pickle.load(f)
            p=predict_tree(model,d,[a.origin],cfg[name],a.days)
        elif name.startswith('patch'):
            model=PatchModel().to(DEVICE);model.load_state_dict(torch.load(ROOT/f'models/{name}_final.pt',map_location=DEVICE,weights_only=True));p=predict_patch(model,d,[a.origin],a.days)
            del model;gc.collect();torch.cuda.empty_cache()
        else:
            if name.startswith('chronos_lora'):
                from chronos import Chronos2Pipeline
                steps=int(name.rsplit('_',1)[-1]);model=Chronos2Pipeline.from_pretrained(str(ROOT/f'models/chronos_lora_d22_s{steps}/finetuned-ckpt'),device_map=DEVICE)
                p,_=predict_chronos(model,d,[a.origin],a.days);del model;gc.collect();torch.cuda.empty_cache()
            else:
                if pipe is None:pipe=base_chronos()
                if name.startswith('chronos_joint'):
                    raw,physical,_=predict_joint(pipe,d,[a.origin],a.days);p=physical if name.endswith('physical') else raw
                else:p,_=predict_chronos(pipe,d,[a.origin],a.days,name=='chronos_cov')
        preds[name]=p
    if primary.startswith('ensemble'):
        p=np.clip(sum(preds[k]*w for k,w in weights.items() if w>1e-8),0,1)
        if primary=='ensemble_peak':p=peak_adjust(d,p,[a.origin],s['peak_adjustment']['beta'],s['peak_adjustment']['anchor'])
    else:p=np.clip(preds[primary],0,1)
    start=pd.Timestamp('2019-05-01',tz='America/Los_Angeles')+pd.Timedelta(days=a.origin)
    dates=pd.date_range(start,periods=a.days*96,freq='15min');rows=[]
    for i,name in enumerate(d['names']):
        for k,t in enumerate(dates):rows.append((str(name),str(d['cells'][i]),t.isoformat(),float(p[i,0,k]),primary))
    target=ROOT/f'results/deployment_forecast_origin{a.origin}_{a.days}day.csv'
    pd.DataFrame(rows,columns=['series','cell','timestamp_PT','predicted_util','model']).to_csv(target,index=False)
    print('Saved',target,'\nForecast starts at dataset time',start,'not the current real-world date.',flush=True)

if __name__=='__main__':main()
