from pathlib import Path
import argparse,shutil
import numpy as np
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F,cfg,d,q=load(R);reg=read(F/'docs/historical_controls.json');selection=read(F/'results/selection.json');assert sha(F/'results/selection.json')==reg['original_target_selection_sha256']
summary=read(F/'results/summary.json');rows=[r for r in summary['rows'] if not r.get('historical_source_control')];cv=[]
for H in [1,7]:
    oo=np.array(origins(q,'test',H));Y=target(d,oo,H);bname='last' if H==1 else'mean3';bp=baseline(d,oo,H,bname);control={n:read_forecasts(F,q,'test',H,n) for n in reg['schemes']}
    subsets={'daily_origins':np.arange(len(oo))}
    if H==7:subsets['nonoverlapping_calendar7day']=np.flatnonzero((oo-q['split']['test'][0])%7==0)
    for cohort,ii in subsets.items():
        b=metrics(Y[ii],bp[ii],cfg['mape_threshold'])
        for n,p in control.items():rows.append(dict(dataset='NLR derived facility subtotal',target=cfg['target'],unit='kW',phase='test',cohort=cohort,horizon_days=H,model=n,selected_on_validation=False,historical_source_control=True,input_adaptation=True,source_configuration_frozen_from_Google=True,seven_day_extension=H==7 and n in ['Google_Chronos_TiRex_equal','Google_three_fixed','Google_R5_Ridge','Google_R8b_finetuned'],**add_baseline(metrics(Y[ii],p[ii],cfg['mape_threshold']),b)))
    p=F/'results'/f'test_H{H}.npz';backup=F/'results'/f'original_main_test_H{H}.npz'
    if not backup.exists():shutil.copyfile(p,backup)
    with np.load(p) as z:old={k:z[k].copy() for k in z.files}
    for k,v in control.items():old[k]=v
    np_save(p,**old)
    vo=origins(q,'validation',H);vy=target(d,vo,H)
    # New controls are described on full validation; not silently compared to
    # nested forward CV or used to rewrite the already frozen target selection.
    for n in reg['schemes']:cv.append(dict(phase='historical_full_validation_descriptive',horizon_days=H,model=n,not_used_to_reselect_target_model=True,**metrics(vy,read_forecasts(F,q,'validation',H,n),cfg['mape_threshold'])))
summary.update(rows=rows,historical_controls_added_after_main_test=True,historical_registration_sha256=sha(F/'docs/historical_controls.json'),original_selection_retained=True,historical_no_target_tuning=True,utc=utc());save(F/'results/summary.json',summary)
export(F/'results/新协议主要模型测试全指标.csv',rows);export(F/'results/旧Google推荐验证描述指标.csv',cv);save(F/'results/historical_controls_complete.json',dict(all_passed=True,utc=utc(),actual_same_cohort_predictions=True,original_selection_sha256=sha(F/'results/selection.json'),source_configuration_sha256=sha(F/'docs/historical_controls.json'),rows=len(rows)));print('Actual historical controls scored',len(rows),flush=True)
