from pathlib import Path
import argparse,hashlib,gc,joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('phase',choices=['validation','final']);a=ap.parse_args();F,cfg,d,q=load(a.root)
def fit(H,params,phases,out):
    xx=[];yy=[];windows=[]
    leads=np.unique(np.r_[np.arange(0,H*96,8 if H==1 else 16),H*96-1])
    for phase in phases:
        if phase=='train_validation':
            end=q['split']['validation'][1];oo=[o for o in range(14,end-H+1) if np.isfinite(d['y'][o-14:o+H]).all() and np.isfinite(d['covariates'][o-14:o]).all()]
        else:oo=origins(q,phase,H)
        for o in oo:
            xx.append(features(d,o,H)[leads]);yy.append(d['y'][o:o+H].reshape(-1)[leads]);windows.append(dict(origin=o,label_end_exclusive=o+H,phase=phase))
    X=np.concatenate(xx);Y=np.concatenate(yy);scale=StandardScaler();Z=scale.fit_transform(X);model=LGBMRegressor(**params,learning_rate=.04,min_child_samples=50,random_state=20261013,n_jobs=2,verbosity=-1);model.fit(Z,Y)
    joblib.dump(dict(scaler=scale,model=model,protocol=sha(F/'docs/protocol.json')),out)
    save(out.with_suffix('.json'),dict(real_target_training=True,rows=len(Y),windows=windows,parameters=params,model_sha256=sha(out),protocol_sha256=sha(F/'docs/protocol.json'),training_end_exclusive=max(w['label_end_exclusive'] for w in windows),phase=a.phase))
    return scale,model
def infer(sc,model,H,phase):return np.stack([np.maximum(0,model.predict(sc.transform(features(d,o,H)))) for o in origins(q,phase,H)])
for H in [1,7]:
    if a.phase=='validation':
        rows=[]
        for i,params in enumerate(cfg['lightgbm_candidates']):
            out=F/'models'/f'LightGBM_H{H}_train_c{i}.joblib';sc,model=fit(H,params,['train'],out);pred=infer(sc,model,H,'validation');name=f'LightGBM_c{i}'
            save_forecasts(F,cfg,d,q,'validation',H,name,pred,dict(training_end_exclusive=q['split']['train'][1],parameters=params));row=dict(candidate=i,**metrics(target(d,origins(q,'validation',H),H),pred,cfg['mape_threshold']));rows.append(row);print('LGBM DEV',H,i,row['curve_mape_percent'],flush=True);del sc,model;gc.collect()
        selected=min(rows,key=lambda r:(r['curve_mape_percent'],r['candidate']))['candidate'];save(F/'results'/f'LightGBM_H{H}_validation_selection.json',dict(selected=selected,rows=rows,selection_metric='validation origin mean MAPE',test_scored=False))
    else:
        selection=read(F/'results/selection.json');assert sha(F/'results/selection.json')==(F/'results/selection.sha256').read_text().strip();i=selection['tasks'][str(H)]['lightgbm_candidate'];out=F/'models'/f'LightGBM_H{H}_train_validation.joblib';sc,model=fit(H,cfg['lightgbm_candidates'][i],['train_validation'],out);pred=infer(sc,model,H,'test');save_forecasts(F,cfg,d,q,'test',H,'LightGBM',pred,dict(training_end_exclusive=q['split']['validation'][1],parameters=cfg['lightgbm_candidates'][i]));del sc,model;gc.collect();print('LGBM final actual refit/test complete',H,flush=True)
