import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import json,time,pickle,gc,hashlib
import numpy as np
import pandas as pd
import torch
from common import *
from classical import fit_tree,predict_tree
from deep_models import base_chronos,fit_chronos,predict_chronos,predict_joint,PatchModel,train_patch,predict_patch,DEVICE
from selection import optimize_weights,peak_adjust

def history_predict(d,origins,cfg,h=1):
    p=baseline(d,origins,h)
    for j,o in enumerate(origins):
        for day in range(h):
            a=p[:,j,day*96:(day+1)*96]
            shift=d['y'][:,o-1,-12:].mean(1)-d['y'][:,o-2,-12:].mean(1)
            p[:,j,day*96:(day+1)*96]=smooth(a,cfg['w'])+cfg['alpha']*shift[:,None]*np.exp(-(np.arange(96)[None,:]+day*96)/48)
    return np.clip(p,0,1)

def main():
    started=time.time();d=load()
    selection=json.loads((ROOT/'results/selection24.json').read_text())
    cfgs=json.loads((ROOT/'results/classical_configs.json').read_text());fam=selection['families']
    dev7={k:baseline(d,DEV7,7,k) for k in ['last','week','mean3','mean7']}
    test24={k:baseline(d,TEST24,1,k) for k in ['last','week','mean3','mean7']}
    test7={k:baseline(d,TEST7,7,k) for k in ['last','week','mean3','mean7']}
    testq={};devq7={};testq7={};timing=[]
    # Generate candidate predictions but do not score any test prediction until
    # both the 24-hour and seven-day selections have been frozen.
    for family in ['history','ridge','lgb_load','lgb_prod']:
        name=fam[family];cfg=cfgs[name];t=time.time()
        if family=='history':
            dev7[name]=history_predict(d,DEV7,cfg,7);test24[name]=history_predict(d,TEST24,cfg);test7[name]=history_predict(d,TEST7,cfg,7)
        else:
            devmodel,_=fit_tree(d,14,cfg);dev7[name]=predict_tree(devmodel,d,DEV7,cfg,7)
            model,features=fit_tree(d,22,cfg)
            with (ROOT/f'models/{name}_final.pkl').open('wb') as f:pickle.dump(model,f)
            test24[name]=predict_tree(model,d,TEST24,cfg);test7[name]=predict_tree(model,d,TEST7,cfg,7)
            if family.startswith('lgb'):
                pd.DataFrame({'feature':features,'importance_gain':model.booster_.feature_importance(importance_type='gain')}).sort_values('importance_gain',ascending=False).to_csv(ROOT/f'results/{family}_feature_importance.csv',index=False)
        timing.append(dict(model=name,seconds=time.time()-t));print('final predictions',name,flush=True)
    for family in ['patch_plain','patch_physical']:
        name=fam[family];ep=int(name.split('_e')[-1]);physical=family=='patch_physical';t=time.time()
        model=PatchModel().to(DEVICE);model.load_state_dict(torch.load(ROOT/f'models/{name}_dev.pt',map_location=DEVICE,weights_only=True))
        dev7[name]=predict_patch(model,d,DEV7,7)
        model,_,hist=train_patch(d,22,ep,physical)
        torch.save(model.state_dict(),ROOT/f'models/{name}_final.pt')
        test24[name]=predict_patch(model,d,TEST24);test7[name]=predict_patch(model,d,TEST7,7)
        timing.append(dict(model=name,seconds=time.time()-t));del model;gc.collect();torch.cuda.empty_cache()
    pipe=base_chronos()
    for cov in [False,True]:
        name='chronos_cov' if cov else 'chronos_zero';t=time.time()
        dev7[name],devq7[name]=predict_chronos(pipe,d,DEV7,7,cov)
        test24[name],testq[name]=predict_chronos(pipe,d,TEST24,1,cov)
        test7[name],testq7[name]=predict_chronos(pipe,d,TEST7,7,cov)
        timing.append(dict(model=name,seconds=time.time()-t))
    dev7['chronos_joint_raw'],dev7['chronos_joint_physical'],_=predict_joint(pipe,d,DEV7,7)
    test24['chronos_joint_raw'],test24['chronos_joint_physical'],diag=predict_joint(pipe,d,TEST24)
    test7['chronos_joint_raw'],test7['chronos_joint_physical'],diag7=predict_joint(pipe,d,TEST7,7)
    (ROOT/'results/joint_test_audit.json').write_text(json.dumps({'day':diag,'week':diag7},indent=2))
    name=fam['chronos_lora'];steps=int(name.rsplit('_',1)[-1]);t=time.time()
    from chronos import Chronos2Pipeline
    devpipe=Chronos2Pipeline.from_pretrained(str(ROOT/f'models/chronos_lora_d14_s{steps}/finetuned-ckpt'),device_map=DEVICE)
    dev7[name],devq7[name]=predict_chronos(devpipe,d,DEV7,7)
    del devpipe;gc.collect();torch.cuda.empty_cache()
    finalpipe=fit_chronos(pipe,d,22,steps)
    test24[name],testq[name]=predict_chronos(finalpipe,d,TEST24)
    test7[name],testq7[name]=predict_chronos(finalpipe,d,TEST7,7)
    timing.append(dict(model=name,seconds=time.time()-t))
    # Freeze seven-day choice using development seven-day labels only.
    y7,m7=truth(d,DEV7,7)
    weights7,p7=optimize_weights({k:dev7[k] for k in selection['weights']},y7,m7)
    dev7['ensemble7']=np.clip(p7,0,1)
    score7={k:metrics(y7,p,m7) for k,p in dev7.items()}
    primary7=min(score7,key=lambda k:score7[k]['wape'])
    ss=dict(weights=weights7,primary7=primary7,development_scores=score7,test_used_for_selection=False)
    text=json.dumps(ss,indent=2);(ROOT/'results/selection7.json').write_text(text)
    (ROOT/'results/selection7.sha256').write_text(hashlib.sha256(text.encode()).hexdigest())
    test7['ensemble7']=np.clip(sum(test7[k]*w for k,w in weights7.items()),0,1)
    test24['ensemble']=np.clip(sum(test24[k]*w for k,w in selection['weights'].items()),0,1)
    pk=selection['peak_adjustment'];test24['ensemble_peak']=peak_adjust(d,test24['ensemble'],TEST24,pk['beta'],pk['anchor'])
    np.savez_compressed(ROOT/'results/dev7.npz',**dev7)
    np.savez_compressed(ROOT/'results/test24_predictions.npz',**test24)
    np.savez_compressed(ROOT/'results/test7_predictions.npz',**test7)
    np.savez_compressed(ROOT/'results/test24_quantiles.npz',**testq)
    np.savez_compressed(ROOT/'results/dev7_quantiles.npz',**devq7)
    np.savez_compressed(ROOT/'results/test7_quantiles.npz',**testq7)
    pd.DataFrame(timing).to_csv(ROOT/'results/runtime.csv',index=False)
    for origins,h,preds,filename in [(TEST24,1,test24,'test24_metrics'),(TEST7,7,test7,'test7_metrics')]:
        y,m=truth(d,origins,h)
        rows=[dict(model=k,**metrics(y,p,m)) for k,p in preds.items()]
        df=pd.DataFrame(rows).sort_values('wape');df.to_csv(ROOT/f'results/{filename}.csv',index=False)
        print(filename,df.to_string(index=False),flush=True)
    print('PRIMARY24',selection['primary24'],'PRIMARY7',primary7,'SECONDS',time.time()-started,flush=True)

if __name__=='__main__':main()
