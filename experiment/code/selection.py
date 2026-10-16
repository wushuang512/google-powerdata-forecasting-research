import json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from common import *

def optimize_weights(preds,y,m):
    keys=list(preds);a=np.stack([preds[k] for k in keys])
    norm=m/np.maximum((np.abs(y)*m).sum((1,2),keepdims=True),1e-10)/len(y)
    def fun(w):return float((np.abs(y-np.einsum('k,ksoh->soh',w,a))*norm).sum())
    def jac(w):return -np.einsum('soh,ksoh->k',np.sign(y-np.einsum('k,ksoh->soh',w,a))*norm,a)
    res=minimize(fun,np.ones(len(keys))/len(keys),jac=jac,bounds=[(0,1)]*len(keys),constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    w=np.maximum(res.x,0);w/=w.sum()
    return dict(zip(keys,w.tolist())),np.einsum('k,ksoh->soh',w,a)

def peak_adjust(d,p,origins,beta,anchor='last'):
    if beta==0:return np.clip(p,0,1)
    out=p.copy();h=p.shape[-1]//96
    for j,o in enumerate(origins):
        target=d['y'][:,o-1].max(1) if anchor=='last' else d['y'][:,o-3:o].max(2).mean(1)
        for day in range(h):
            a=p[:,j,day*96:(day+1)*96]
            lo=a.min(1,keepdims=True);hi=a.max(1,keepdims=True)
            weight=((a-lo)/np.maximum(hi-lo,1e-8))**8
            wm=weight.mean(1,keepdims=True)
            # Zero-sum correction preserves the daily mean before clipping.
            delta=(weight-wm)/np.maximum(1-wm,1e-8)*beta*(target[:,None]-hi)
            out[:,j,day*96:(day+1)*96]=np.clip(a+delta,0,1)
    return out

def select():
    d=load();y,m=truth(d,DEV24);allp={};rows=[]
    for group in ['classical','patch','chronos']:
        f=ROOT/f'results/dev_{group}.npz'
        if not f.exists():raise FileNotFoundError(f)
        z=np.load(f);allp.update({k:z[k] for k in z.files})
    scores={k:metrics(y,p,m) for k,p in allp.items()}
    best=lambda names:min(names,key=lambda k:scores[k]['wape'])
    names=list(allp)
    families={
        'history':best([k for k in names if k.startswith('history')]),
        'ridge':best([k for k in names if k.startswith('ridge')]),
        'lgb_load':best([k for k in names if k.startswith('lgb_p0')]),
        'lgb_prod':best([k for k in names if k.startswith('lgb_p1')]),
        'patch_plain':best([k for k in names if k.startswith('patch_plain')]),
        'patch_physical':best([k for k in names if k.startswith('patch_physical')]),
        'chronos_lora':best([k for k in names if k.startswith('chronos_lora')])}
    pool={k:allp[k] for k in [families['history'],families['lgb_prod'],families['patch_physical'],'chronos_zero','chronos_cov',families['chronos_lora'],'chronos_joint_physical']}
    weights,p=optimize_weights(pool,y,m);allp['ensemble']=np.clip(p,0,1);scores['ensemble']=metrics(y,allp['ensemble'],m)
    grids=[]
    for anchor in ['last','mean3']:
        for beta in [0,.25,.5,1]:
            pp=peak_adjust(d,p,DEV24,beta,anchor)
            grids.append(dict(anchor=anchor,beta=beta,**metrics(y,pp,m)))
    # Curve objective primary, allow <=0.01 percentage-point WAPE cost to reduce peaks.
    eligible=[g for g in grids if g['wape']<=scores['ensemble']['wape']+.01]
    picked=min(eligible,key=lambda g:g['peak_mape'])
    allp['ensemble_peak']=peak_adjust(d,p,DEV24,picked['beta'],picked['anchor'])
    scores['ensemble_peak']=metrics(y,allp['ensemble_peak'],m)
    primary=min(['ensemble','ensemble_peak']+[k for k in pool],key=lambda k:scores[k]['wape'])
    selection=dict(families=families,weights=weights,peak_adjustment=picked,primary24=primary,development_scores=scores,peak_grid=grids,test_used_for_selection=False)
    content=json.dumps(selection,indent=2)
    (ROOT/'results/selection24.json').write_text(content)
    (ROOT/'results/selection24.sha256').write_text(hashlib.sha256(content.encode()).hexdigest())
    np.savez_compressed(ROOT/'results/dev_selected.npz',**allp)
    pd.DataFrame([dict(model=k,**v) for k,v in scores.items()]).sort_values('wape').to_csv(ROOT/'results/development_all.csv',index=False)
    print('FROZEN_SELECTION',json.dumps({k:selection[k] for k in ['families','weights','peak_adjustment','primary24']},indent=2),flush=True)

if __name__=='__main__':select()
