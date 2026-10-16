import json, time, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from common import *

def features(d,o,prod=False):
    y=d['y'][:,:o]; n=57; t=96
    f=[]; names=[]
    def add(x,name):
        x=np.asarray(x)
        if x.ndim==1:x=np.repeat(x[:,None],96,axis=1)
        f.append(x);names.append(name)
    for lag in [1,2,3,7]:
        a=y[:,-lag]
        for w in [1,5,13]:add(smooth(a,w),f'lag{lag}_smooth{w}')
    for k in [3,7]:
        add(y[:,-k:].mean(1),f'mean{k}')
        add(np.median(y[:,-k:],axis=1),f'median{k}')
        add(y[:,-k:].std(1),f'slot_std{k}')
    flat=y.reshape(n,-1)
    for k in [1,4,12,24,48,96,288,672]:
        add(flat[:,-k:].mean(1),f'recent_mean{k}')
    for lag in [1,2,3,7]:
        a=y[:,-lag]
        for label,fn in [('min',np.min),('max',np.max),('std',np.std)]:add(fn(a,axis=1),f'day{lag}_{label}')
    for k in [4,12,24,48,96]:
        add(flat[:,-k:].mean(1)-flat[:,-96-k:-96].mean(1),f'level_change{k}')
    add(y[:,-1]-y[:,-1].mean(1)[:,None],'last_shape')
    cellmean=np.empty((57,96))
    for c in np.unique(d['cells']):
        idx=d['cells']==c
        # Normalise each member using only its own known context.
        norm=y[idx,-1]/np.maximum(y[idx,-1].mean(1)[:,None],1e-5)
        cellmean[idx]=norm.mean(0)
    add(cellmean,'cell_relative_shape')
    slot=np.arange(96)
    for k in [1,2,3]:
        add(np.tile(np.sin(2*np.pi*k*slot/96),(57,1)),f'sin{k}')
        add(np.tile(np.cos(2*np.pi*k*slot/96),(57,1)),f'cos{k}')
    dow=(2+o)%7 # 2019-05-01 was Wednesday
    add(np.full(57,dow),'day_of_week')
    add(np.full(57,int(dow>=5)),'weekend')
    if prod:
        p=d['prod'][:,:o];v=d['pvalid'][:,:o]
        for lag in [1,2,7]:
            add(p[:,-lag],f'production_lag{lag}')
            add(v[:,-lag].astype(float),f'production_valid{lag}')
        add(p[:,-1].mean(1),'production_mean')
        add(p[:,-1].mean(1)-p[:,-2].mean(1),'production_change')
        add(v[:,-7:].mean((1,2)),'production_valid_fraction')
    return np.stack(f,axis=-1).reshape(-1,len(f)),names

def fit_tree(d,cutoff,cfg,exclude_cell=None):
    xs=[];ys=[];weights=[]
    for o in range(7,cutoff):
        x,names=features(d,o,cfg.get('prod',False))
        target=(d['y'][:,o]-d['y'][:,o-1]).reshape(-1)
        mask=d['valid'][:,o].copy()
        if exclude_cell is not None:mask[d['cells']==exclude_cell]=False
        mask=mask.ravel()
        scale=np.maximum(d['y'][:,:cutoff].mean((1,2)),.05)
        weight=np.repeat(1/scale,96)
        xs.append(x[mask]);ys.append(target[mask]);weights.append(weight[mask])
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights)
    if cfg['kind']=='ridge':
        model=make_pipeline(StandardScaler(),Ridge(alpha=cfg['alpha']))
        model.fit(x,y,ridge__sample_weight=w)
    else:
        model=LGBMRegressor(n_estimators=cfg['trees'],num_leaves=cfg['leaves'],max_depth=-1,min_child_samples=cfg['min_child'],learning_rate=.035,reg_lambda=10,objective=cfg.get('objective','regression_l1'),n_jobs=4,verbosity=-1,random_state=20260929,deterministic=True,force_col_wise=True)
        model.fit(x,y,sample_weight=w,feature_name=names)
    return model,names

def predict_tree(model,d,origins,cfg,h=1):
    out=[]
    for o in origins:
        # For multi-day forecasts, only append predictions, never future values.
        dd={k:v for k,v in d.items()}
        dd['y']=d['y'][:,:o].copy();dd['prod']=d['prod'][:,:o].copy();dd['pvalid']=d['pvalid'][:,:o].copy()
        horizon=[]
        for j in range(h):
            x,_=features(dd,o+j,cfg.get('prod',False))
            p=model.predict(x).reshape(57,96)+dd['y'][:,-1]
            p=np.clip(p,0,1)
            horizon.append(p)
            dd['y']=np.concatenate([dd['y'],p[:,None]],axis=1)
            # production unknown after origin: persist last known day, mark unavailable
            dd['prod']=np.concatenate([dd['prod'],dd['prod'][:,-1:]],axis=1)
            dd['pvalid']=np.concatenate([dd['pvalid'],np.zeros((57,1,96),bool)],axis=1)
        out.append(np.concatenate(horizon,axis=-1))
    return np.stack(out,axis=1)

def develop():
    d=load(); y,m=truth(d,DEV24)
    predictions={};rows=[];configs={}
    for name in ['last','week','mean3','mean7','level']:
        p=baseline(d,DEV24,kind=name)
        predictions[name]=p;rows.append(dict(model=name,**metrics(y,p,m)))
    # Predefined compact grid; only the four development origins determine choices.
    for w in [5,13,25]:
        for alpha in [0,.5,1]:
            p=baseline(d,DEV24)
            for j,o in enumerate(DEV24):
                p[:,j]=smooth(p[:,j],w)
                shift=d['y'][:,o-1,-12:].mean(1)-d['y'][:,o-2,-12:].mean(1)
                p[:,j]+=alpha*shift[:,None]*np.exp(-np.arange(96)[None,:]/48)
            name=f'history_w{w}_trend{alpha}'
            predictions[name]=np.clip(p,0,1); configs[name]=dict(kind='history',w=w,alpha=alpha)
            rows.append(dict(model=name,**metrics(y,predictions[name],m)))
    grid=[]
    for a in [10,100,1000]:grid.append((f'ridge_{a}',dict(kind='ridge',alpha=a,prod=False)))
    for prod in [False,True]:
        for leaves,trees,minimum in [(15,200,100),(31,300,100),(31,600,100),(63,400,50)]:
            grid.append((f'lgb_p{int(prod)}_l{leaves}_n{trees}',dict(kind='lgb',prod=prod,leaves=leaves,trees=trees,min_child=minimum)))
    for name,cfg in grid:
        start=time.time();model,names=fit_tree(d,14,cfg)
        p=predict_tree(model,d,DEV24,cfg)
        predictions[name]=p; configs[name]=cfg
        row=dict(model=name,**metrics(y,p,m),seconds=time.time()-start);rows.append(row)
        print(json.dumps(row),flush=True)
    np.savez_compressed(ROOT/'results/dev_classical.npz',**predictions)
    pd.DataFrame(rows).to_csv(ROOT/'results/dev_classical.csv',index=False)
    (ROOT/'results/classical_configs.json').write_text(json.dumps(configs,indent=2))
    print(pd.DataFrame(rows).sort_values('wape')[['model','wape','peak_mape']].to_string(index=False),flush=True)
    # Baseline reproduction only; never used to select a candidate.
    yt,mt=truth(d,TEST24);p=baseline(d,TEST24)
    r=metrics(yt,p,mt)
    (ROOT/'results/baseline_reproduction.json').write_text(json.dumps(r,indent=2))
    print('BASELINE_REPRODUCTION',r,flush=True)

if __name__=='__main__':develop()
