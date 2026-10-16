"""Frozen source local24 and chronological NLR local24/Ridge fitting."""
from pathlib import Path
import gc,sys,warnings
import numpy as np
from core import *
def setup(root):
    sys.path.insert(0,str(root/'experiment/round44/reference/PowerData_Prediction'))
def fields(x,date):
    import pandas as pd
    from power_forecast.attainment_local import feature_cube
    # Build original feature vocabulary on observed days plus a NaN placeholder.
    history=x[:,:,None];dummy=np.full((1,96,1),np.nan)
    arr=np.concatenate([history,dummy]);m=np.isfinite(arr);prod=np.full_like(arr,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');cube,names=feature_cube(arr,prod,m,pd.DataFrame({'cell':['NLR']}))
    F=cube[-1].reshape(96,-1).astype(float)
    for n in ['series_id','cell_id','loo_lag1','loo_med3','series_over_loo_lag1','loo_trend']:
        F[:,names.index(n)]=np.nan
    # Original numeric time features; deterministic source-clock weekday.
    import datetime
    dow=datetime.date.fromisoformat(str(date)).weekday()
    for n,v in [('dow_sin',np.sin(2*np.pi*dow/7)),('dow_cos',np.cos(2*np.pi*dow/7)),('is_weekend',float(dow>=5))]:F[:,names.index(n)]=v
    return F,names
def prediction(model,x,date):
    F,names=fields(x,date);assert names==model['feature_names']
    if model['kind']=='ridge':return model['pipeline'].predict(F)+x[-1]
    gbm=np.mean([g.predict(F) for g in model['models']],axis=0)+x[-1]
    lag=sum(c*x[-lag] for c,lag in zip(model['coefficients'],model['lags']))
    return np.clip(.5*gbm+.5*lag,x.min(0),x.max(0))
def smooth(x,h):
    history=x[-24:];level=history.mean(-1);levels=[]
    for half in [3.,5.]:
        w=.5**(np.arange(len(history))/half);levels.append((w/w.sum())@level[::-1])
    norm=history/np.maximum(level[:,None],1e-8);grand=norm.mean(0);w=.5**(np.arange(len(history))[::-1]/10);w/=w.sum()
    shape=grand+(w[:,None]*(norm-grand)).sum(0)
    return np.tile(np.mean(levels)*shape,h)
def fit(root,folder,phase):
    import joblib,lightgbm as lgb
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    setup(root);z=np.load(folder/'data/nlr_15min.npz');x=z['x']/z['scales'][:,None,None];mask=z['valid'];dates=z['dates']
    cutoff=int(np.searchsorted(dates,'2023-01-01' if phase=='dev' else'2024-01-01'))
    start=int(np.searchsorted(dates,'2016-06-12'))+28
    for j in range(2):
        rows=[];ys=[];lags=[];last=[]
        for o in range(start,cutoff,7):
            if mask[j,o-28:o].mean()<.95:continue
            f,names=fields(x[j,o-28:o],dates[o]);valid=mask[j,o];rows.append(f[valid]);ys.append((x[j,o]-x[j,o-1])[valid])
            lags.append(np.stack([x[j,o-v] for v in [1,2,3,4,5,6,7,14]],1)[valid]);last.append(x[j,o][valid])
        X=np.concatenate(rows);Y=np.concatenate(ys);L=np.concatenate(lags);target=np.concatenate(last)
        coeff=np.linalg.lstsq(L*np.sqrt(abs(target))[:,None],target*np.sqrt(abs(target)),rcond=None)[0]
        models=[]
        for seed in [101,202,303,404]:
            g=lgb.LGBMRegressor(objective='l1',num_leaves=15,learning_rate=.04,n_estimators=350,min_child_samples=40,subsample=.8,
                subsample_freq=1,colsample_bytree=.8,reg_lambda=1,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True,random_state=seed)
            g.fit(X,Y);models.append(g)
        joblib.dump(dict(kind='local24',models=models,coefficients=coeff,lags=[1,2,3,4,5,6,7,14],feature_names=names),folder/'models'/f'nlr_local_{phase}_{j}.joblib')
        pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=10))
        pipe.fit(X,Y);joblib.dump(dict(kind='ridge',pipeline=pipe,feature_names=names),folder/'models'/f'nlr_ridge_{phase}_{j}.joblib')
        save(folder/'results'/f'local_training_{phase}_{j}.json',dict(utc=utc(),cutoff=str(dates[cutoff-1]),samples=len(Y),stride_days=7,seeds=4,
            source_power_only=True,source_fields_unavailable=['production','other_series','static_series_ID'],feature_names=names,all_labels_before_cutoff=True))
        print('NLR real local24/Ridge fitted',phase,j,len(Y),flush=True)
def collect(root,folder,phase):
    import joblib
    setup(root);z=np.load(folder/'data/nlr_15min.npz');x=z['x']/z['scales'][:,None,None];dates=z['dates'];origins=protocol_check(folder)['nlr_origins'][phase]
    source=joblib.load(root/'experiment/round44/results/local_cut22.joblib')
    frozen=dict(kind='local24',models=source.models,coefficients=source.coefficients,lags=source.lags,feature_names=source.feature_names)
    modelphase='dev' if phase=='dev' else'test';out={}
    for label in ['google_local24','nlr_local24','nlr_ridge']:
        p=[]
        for j in range(2):
            model=frozen if label=='google_local24' else joblib.load(folder/'models'/f'nlr_{"local" if label=="nlr_local24" else"ridge"}_{modelphase}_{j}.joblib')
            p.append(np.stack([prediction(model,x[j,o-28:o],dates[o]) for o in origins]))
        out[label+'_H1']=np.array(p)
    out['local7_H7']=np.stack([np.stack([smooth(x[j,o-24:o],7) for o in origins]) for j in range(2)])
    np_save(folder/'cache'/f'local_{phase}.npz',**out)
