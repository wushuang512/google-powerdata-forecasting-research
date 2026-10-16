import json,gc,time,traceback
from round2_lib import *
import pandas as pd

EXTRA={
 'lora_peer_lr3e5_s300':dict(mode='lora',kind='peer',lr=3e-5,steps=300,context=1344),
 'lora_peerprod_lr3e5_s300':dict(mode='lora',kind='peerprod',lr=3e-5,steps=300,context=1344),
 'lora_parts_lr3e5_s300':dict(mode='lora',kind='parts',lr=3e-5,steps=300,context=1344),
 'full_parts_lr1e6_s300':dict(mode='full',kind='parts',lr=1e-6,steps=300,context=1344)
}
def main():
    d=load();y,m=truth(d,DEV);pipe=base_chronos();path=R/'results/extras.npz';p=dict(np.load(path)) if path.exists() else {};rows=[]
    (R/'extra_configs.json').write_text(json.dumps(EXTRA,indent=2))
    def save(name,z):
        p[name]=z;np.savez_compressed(path,**p)
        row=dict(model=name,**metrics(y,z,m));print('RESULT',json.dumps(row),flush=True)
    for kind in ['cell','peer','peerprod','parts']:
        name='zero_'+kind
        if name not in p:
            try:save(name,predict(pipe,d,DEV,kind))
            except Exception:traceback.print_exc()
    for name,cfg in EXTRA.items():
        if name in p:continue
        ft=None
        try:
            print('TRAIN',name,flush=True);ft=train(pipe,d,14,cfg,name);save(name,predict(ft,d,DEV,cfg['kind']))
        except Exception:traceback.print_exc()
        finally:del ft;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame([dict(model=k,**metrics(y,z,m)) for k,z in p.items()]).to_csv(R/'results/extras.csv',index=False)

if __name__=='__main__':main()
