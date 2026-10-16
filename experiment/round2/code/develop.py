import json,gc,time,traceback
import numpy as np
import pandas as pd
import torch
from round2_lib import *

def main():
    d=load();y,m=truth(d,DEV);pipe=base_chronos()
    path=R/'results/development.npz';preds={};rows=[];fail=[]
    if path.exists():
        z=np.load(path);preds={k:z[k] for k in z.files}
        rows=[dict(model=k,**metrics(y,v,m)) for k,v in preds.items()]
    protocol=dict(development_origins=DEV,test_origins=TEST,train_cutoffs=[14,22],configurations=CONFIGS,goal='At least 5 percent relative WAPE reduction versus reproduced old equal-weight Chronos2+TiRex2, plus paired day/domain checks',selection='development-only selection, two consecutive four-day blocks, regularization towards the strong reproduced baseline',limitation='Round two is adaptive research on an already examined month, not independent confirmation')
    (R/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def save(name,p):
        preds[name]=p;score=dict(model=name,**metrics(y,p,m));rows.append(score)
        np.savez_compressed(path,**preds);pd.DataFrame(rows).to_csv(R/'results/development.csv',index=False)
        print('RESULT',json.dumps(score),flush=True)
    for kind in ['uni','cov','calendar','cov_calendar','cell','cellprod','all']:
        name='zero_'+kind
        if name in preds:continue
        try:save(name,predict(pipe,d,DEV,kind))
        except Exception as e:
            traceback.print_exc();fail.append(dict(model=name,error=str(e)))
    for name,cfg in CONFIGS.items():
        if name in preds:continue
        ft=None
        try:
            print('TRAIN',name,flush=True);started=time.time();ft=train(pipe,d,14,cfg,name)
            save(name,predict(ft,d,DEV,cfg['kind']));print('SECONDS',time.time()-started,flush=True)
        except Exception as e:
            traceback.print_exc();fail.append(dict(model=name,error=str(e)))
        finally:
            del ft;gc.collect();torch.cuda.empty_cache();(R/'results/failures.json').write_text(json.dumps(fail,indent=2))
    print(pd.DataFrame(rows).sort_values('wape')[['model','wape','peak_mape']].to_string(index=False),flush=True)

if __name__=='__main__':main()
