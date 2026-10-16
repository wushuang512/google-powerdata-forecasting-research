"""Independent arithmetic, cleanup integrity and cached forecast verification."""
from pathlib import Path
import datetime,hashlib,json,sys,tomllib,os
import numpy as np,pandas as pd
TOP=Path(sys.argv[1]).resolve();B=TOP/'experiment';O=B/'model_comparison_20261005';S=json.loads((O/'results/summary.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
    return h.hexdigest()
checks={};z=np.load(B/'results/data.npz'); yy=np.stack([z['y'][:,i] for i in range(22,31)],1); mask=np.stack([z['valid'][:,i] for i in range(22,31)],1)
for rel,h in S['sources'].items():checks['source_'+rel]=sha(TOP/rel)==h
for key,a in S['models'].items():
    p=np.load(B/a['source'])[a['array']].astype('float64')
    err=abs(yy-p);wape=100*np.mean(np.sum(err*mask,(1,2))/np.sum(abs(yy)*mask,(1,2)))
    count=mask.sum(-1);ya=(yy*mask).sum(-1)/count;pa=(p*mask).sum(-1)/count
    daily=float(np.mean(100*abs(ya-pa).sum(-1)/abs(ya).sum(-1)))
    peak=float(100*np.mean(abs(np.where(mask,yy,-np.inf).max(-1)-np.where(mask,p,-np.inf).max(-1))/np.where(mask,yy,-np.inf).max(-1)))
    checks['curve_'+key]=abs(wape-a['wape'])<1e-12
    checks['daily_'+key]=abs(daily-a['daily_wape'])<1e-12
    checks['peak_'+key]=abs(peak-a['peak_mape'])<1e-12
plan=json.loads((O/'cleanup_plan.json').read_text());record=json.loads((O/'cleanup_record.json').read_text(encoding='utf-8-sig'))
checks['cleanup_state']=record['state']=='completed'
checks['deleted_paths_absent']=all(not (TOP/t['relative_path']).exists() for t in plan['targets'])
for rel,a in plan['protected_files'].items():checks['retained_'+rel]=(TOP/rel).stat().st_size==a['bytes'] and sha(TOP/rel)==a['sha256']
best=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv');old=pd.read_csv(B/'round5/results/selected_deployment_origin31_1day.csv')
prior=old.predicted_util.to_numpy().reshape(57,1,96);new=np.load(B/'round8b/results/component_deployment_origin31.npz')['tf3_finetuned'];original=np.load(B/'round4/results/deployment_origin31_1day.npz')['tf3_cellprod']
sel=json.loads((B/'round8b/results/selection.json').read_text());a=sel['chosen']['alpha'];candidate=np.clip(prior+.5*(new-original),0,1);selected=np.clip((1-a)*prior+a*candidate,0,1)
checks['frozen_selection_sha']=sha(B/'round8b/results/selection.json')==(B/'round8b/results/selection.sha256').read_text().strip()
checks['cached_deployment_exact']=np.array_equal(selected.ravel(),best.predicted_util.to_numpy()) or np.allclose(selected.ravel(),best.predicted_util.to_numpy(),rtol=0,atol=2e-16)
checks['deployment_rows']=len(best)==5472
checks['deployment_domains']=best.series.nunique()==57
timecol=next(k for k in best.columns if 'time' in k.lower());times=pd.to_datetime(best[timecol])
checks['deployment_date']=times.min()==pd.Timestamp('2019-06-01') and times.max()==pd.Timestamp('2019-06-01 23:45')
checks['deployment_finite_bounded']=bool(np.isfinite(best.predicted_util).all() and best.predicted_util.between(0,1).all())
checks['deployment_interval_ordered']=bool(((best.lower80_empirical<=best.predicted_util)&(best.predicted_util<=best.upper80_empirical)).all())
checks['entry09_intact']='predict_finetuned.py' in (TOP/'09_现有最佳微调_预测次日.cmd').read_text(encoding='utf-8-sig')
checks['entry16_delegates']='09_现有最佳微调_预测次日.cmd' in (TOP/'16_直接预测共享状态_预测次日.cmd').read_text(encoding='utf-8-sig')
checks['automation_paused']=tomllib.loads((Path(os.environ['USERPROFILE'])/'.codex/automations/google/automation.toml').read_text())['status']=='PAUSED'
checks={k:bool(v) for k,v in checks.items()}
out=dict(all_passed=all(checks.values()),check_count=len(checks),checks=checks,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    retained_forecast_sha256=sha(B/'round8b/results/selected_deployment_origin31_1day.csv'),
    cached_reproduction=True,new_heavy_inference=False,independent_confirmation=False,
    historical_round14_cpu_model_replay_no_longer_supported='Dedicated weights deleted by user request; original 123 checks describe pre-cleanup evidence; score arrays retained')
(O/'results/verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
assert out['all_passed'],[k for k,v in checks.items() if not v]
print(json.dumps(dict(all_passed=True,checks=len(checks),forecast_rows=len(best),deleted_files=record['deleted_files'])))
