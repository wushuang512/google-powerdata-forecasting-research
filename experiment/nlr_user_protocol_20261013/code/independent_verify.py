"""Independent arrays, source, missingness, freeze and output checks."""
from pathlib import Path
import argparse,copy,hashlib,json
import numpy as np
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F,cfg,d,q=load(R);checks=[];bindings={}
def check(n,b):checks.append(dict(name=n,passed=bool(b)));assert b,n
def bind(p):bindings[str(p.relative_to(R))]=sha(p)
quality=read(F/'data/quality_report.json');check('Prepared data matches original preprocessing',sha(F/'data/processed.npz')==quality['processed_sha256']);check('UTC provisional and not verified',not quality['timezone_verified'] and cfg['timezone']=='UTC')
raw=d['raw_covariates'];check('Derived label sum of interval means, not meter',np.allclose(d['y'],raw[:,:,:5].sum(-1),equal_nan=True))
check('Any missing component leaves target missing',np.array_equal(np.isfinite(d['y']),np.isfinite(raw[:,:,:5]).all(-1)))
# Rebuild limited one-sided fill independently for each covariate.
for j in range(7):
    v=raw[:,:,j].reshape(-1);idx=np.arange(len(v));last=np.maximum.accumulate(np.where(np.isfinite(v),idx,-1));allowed=(last>=0)&(idx-last<=4);out=np.where(allowed,v[np.maximum(last,0)],np.nan)
    check('Exactly causal max4 fill '+str(j),np.allclose(out,d['covariates'][:,:,j].reshape(-1),equal_nan=True))
for k,v in quality['source_audit'].items():check('Raw source SHA unchanged '+k,sha(Path(v['file']))==v['sha256']);bind(Path(v['file']))
check('Weather1970 out-of-range discarded',quality['source_audit']['weather']['outside_date_range_removed']>=1)
check('No target label fill',quality['targets_never_filled'] and not cfg['target_fill'] and not cfg['past_target_fill'])
reg=read(F/'docs/historical_controls.json');check('Original target selection unchanged',sha(F/'results/selection.json')==reg['original_target_selection_sha256']);check('Historical registration intact',sha(F/'docs/historical_controls.json')==(F/'docs/historical_controls.sha256').read_text().strip())
for rel,digest in reg['bindings'].items():check('Frozen Google binding '+rel,sha(R/rel)==digest);bind(R/rel)
for p in (F/'models').glob('LightGBM*.json'):
    m=read(p);end=q['split']['train'][1] if m['phase']=='validation' else q['split']['validation'][1]
    check('Target LightGBM label bound '+p.name,m['training_end_exclusive']<=end and all(w['origin']+1<=w['label_end_exclusive']<=end for w in m['windows']));check('Model file SHA '+p.name,sha(p.with_suffix('.joblib'))==m['model_sha256']);bind(p);bind(p.with_suffix('.joblib'))
for p in (F/'models').glob('*_scale.json'):
    m=read(p);end=m['end_exclusive'];known=d['y'][:end];expected=float(np.quantile(known[np.isfinite(known)],.95)/.8);check('Train-only scale '+p.name,np.isclose(expected,m['scale'],rtol=0,atol=1e-12));check('Original foundation weight SHA '+p.name,sha(Path(m['weight_file']))==m['weight_sha256']);bind(p);bind(Path(m['weight_file']))
summary=read(F/'results/summary.json');check('All expected methods and cohorts scored',len(summary['rows'])==42)
for H in [1,7]:
    with np.load(F/'results'/f'test_H{H}.npz') as z,np.load(F/'results'/f'original_main_test_H{H}.npz') as old:
        for k in old.files:check('Previously scored main outputs unchanged '+str(H)+k,np.array_equal(z[k],old[k]))
        oo=z['origins'];Y=z['truth'];check('Correct truth/alignment '+str(H),np.array_equal(Y,target(d,list(oo),H)))
        for row in summary['rows']:
            if row['horizon_days']!=H:continue
            ii=np.arange(len(oo)) if row['cohort']=='daily_origins' else np.flatnonzero((oo-q['split']['test'][0])%7==0)
            if row['cohort']!='daily_origins':check('Seven-day cohorts truly nonoverlap '+row['model'],(np.diff(oo[ii])>=7).all())
            P=z[row['model']][ii];check('Finite nonnegative forecasts '+row['model']+str(H),np.isfinite(P).all() and (P>=0).all())
            # Separate implementation, not a call to the producer metrics.
            true=Y[ii];err=abs(true-P);mask=abs(true)>cfg['mape_threshold'];counts=mask.sum(1);ape=np.divide(err,abs(true),out=np.zeros_like(err),where=mask);mm=np.divide((ape*mask).sum(1),counts,out=np.full(len(ii),np.nan),where=counts>0)
            mape=float(np.nanmean(mm)*100);wape=float(err.sum()/abs(true).sum()*100);rmse=float(np.sqrt(np.mean((true-P)**2)))
            check('Independent MAPE '+row['model']+str(H)+row['cohort'],np.isclose(mape,row['curve_mape_percent'],atol=1e-10,rtol=0));check('Independent WAPE '+row['model']+str(H)+row['cohort'],np.isclose(wape,row['curve_wape_percent'],atol=1e-10,rtol=0));check('Independent RMSE '+row['model']+str(H)+row['cohort'],np.isclose(rmse,row['curve_rmse'],atol=1e-10,rtol=0))
            check('MAPE exclusion counts '+row['model'],int(mask.sum())==row['mape_n'] and int((~mask).sum())==row['mape_excluded'])
            ty=true.reshape(len(ii),H,96).mean(-1);py=P.reshape(len(ii),H,96).mean(-1);dw=float(100*abs(ty-py).sum()/abs(ty).sum());dr=float(np.sqrt(np.mean((ty-py)**2)))
            check('Daily means scored after aggregation '+row['model']+str(H),np.isclose(dw,row['daily_wape_percent'],atol=1e-10,rtol=0) and np.isclose(dr,row['daily_rmse'],atol=1e-10,rtol=0))
    bind(F/'results'/f'test_H{H}.npz');bind(F/'results'/f'original_main_test_H{H}.npz')
for p in [F/'docs/protocol.json',F/'docs/historical_controls.json',F/'results/selection.json',F/'data/processed.npz',F/'data/quality_report.json',F/'results/summary.json',F/'results/新协议主要模型测试全指标.csv',R/'当前推荐方案.json']:bind(p)
probe=read(F/'results/tirex_gpu_probe.json');check('TiRex runtime choice did not use labels',probe.get('no_labels_or_scores_used_for_device_choice',False))
if probe['passed']:check('Actual CPU GPU equivalence checked',probe['cpu_gpu_same_input_comparison_passed'] and probe['max_absolute_normalized_difference']<=1e-5)
save(F/'results/independent_verification.json',dict(all_passed=True,utc=utc(),count=len(checks),checks=checks,bindings=bindings,source_data_and_original_recommendation_preserved=True,scope='42 rows; derived subtotal; overlapping and calendar nonoverlap; frozen target model and separately appended historical controls'))
print(json.dumps(dict(all_passed=True,checks=len(checks),rows=42),ensure_ascii=False))
