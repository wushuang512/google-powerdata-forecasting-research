"""Register the expanded comparison before any new development scoring."""
from pathlib import Path
import csv, datetime, hashlib, json, sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(sys.argv[1]).resolve(); F=ROOT/'experiment/generalization_extended_20261013'
OLD=ROOT/'experiment/generalization_validation_20261012'
def sha(p):
    with p.open('rb') as fp:return hashlib.file_digest(fp,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
for d in ['code','data','models','cache','results','docs']:(F/d).mkdir(parents=True,exist_ok=True)
assert not (F/'docs/protocol.json').exists(),'Never replace a registered comparison'
with np.load(OLD/'data/nlr_15min.npz') as z: data={k:z[k].copy() for k in z.files}
names=['facility_total_kw','it_power_kw'];dates=data['dates'];cases={};coverage=[]
for case,cut,dev_start,evals in [('NLR_2021','2020-01-01','2020-01-01',[(2021,'test')]),
    ('NLR_2024_2025','2023-01-01','2023-01-01',[(2024,'test'),(2025,'confirm')])]:
    cutoff=int(np.searchsorted(dates,cut)); scales=np.array([np.quantile(data['y'][j,:cutoff][data['valid'][j,:cutoff]],.95)/.8 for j in range(2)])
    origins={}
    periods=[(int(dev_start[:4]),'dev')]+evals
    for year,phase in periods:
        start=int(np.searchsorted(dates,f'{year}-01-01'));end=int(np.searchsorted(dates,f'{year+1}-01-01'))
        candidates=range(start,min(end,len(dates)-6),7);oo=[]
        for o in candidates:
            mask=data['valid'][:,o-28:o]
            eligible=bool(o>=28 and mask.shape[1]==28 and (mask.mean((1,2))>=.95).all() and mask[:,-1,-1].all())
            coverage.append(dict(case=case,phase=phase,date=str(dates[o]),origin=o,eligible=eligible,history_only=True))
            if eligible:oo.append(o)
        assert len(oo)>=6,(case,phase,len(oo))
        origins[phase]=oo
    np.savez_compressed(F/'data'/f'{case}.npz',y=data['y'],x=data['x'],valid=data['valid'],dates=dates,scales=scales,names=np.array(names),ppd=96)
    cases[case]=dict(kind='power',unit='kW',cutoff=cutoff,cutoff_date=cut,origins=origins,
        target_sources=1,targets_dependent_same_facility=True,external=True,
        evaluation_previously_in_prior_training=case=='NLR_2021',evaluation_previously_scored=case!='NLR_2021')
sys.path.insert(0,str(ROOT/'experiment/code'));import common
d=common.load();y=d['y'];m=d['valid'];x=y.copy().reshape(57,-1)
for j in range(57):
    pos=np.maximum.accumulate(np.where(m[j].reshape(-1),np.arange(x.shape[1]),-1))
    x[j]=np.where(pos>=0,x[j,np.maximum(pos,0)],0)
x=x.reshape(y.shape);gd=np.arange('2019-05-01','2019-06-01',dtype='datetime64[D]').astype(str)
np.savez_compressed(F/'data/Google2019.npz',y=y,x=x,valid=m,dates=gd,scales=np.ones(57),names=d['names'],ppd=96)
cases['Google2019']=dict(kind='power',unit='normalized power',cutoff=14,final_cutoff=22,
    origins=dict(dev=list(range(14,22)),test1=list(range(22,31)),test7=[22,23,24]),
    target_sources=10,external=False,evaluation_previously_scored=True)
with np.load(OLD/'data/sea_daily.npz') as z:sea={k:z[k].copy() for k in z.files}
np.savez_compressed(F/'data/Sea2022.npz',**sea,names=np.array(['daily_total_energy']),ppd=1)
cases['Sea2022']=dict(kind='daily_energy',unit='kWh',cutoff=14,final_cutoff=22,
    origins=dict(dev1=list(range(14,21)),dev7=[14,15],test1=list(range(22,30)),test7=[22,23]),
    target_sources=1,external=True,evaluation_previously_scored=True,
    neural_weight_training='Not statistically supported by 14 daily observations; actual low-dimensional target residual calibration for each compatible model instead',
    intraday_curves_and_peaks_unavailable=True)
rows=list(csv.DictReader((ROOT/'模型各项指标完整结果.csv').open(encoding='utf-8-sig',newline='')))
save(F/'docs/historical_candidate_catalog.json',dict(rows=[{k:r.get(k) for k in ['horizon_days','model','model_name','status','development_wape_original_grid','source']} for r in rows],
    purpose='Coverage audit of all archived families; exposed test ranking is not used for new model or hyperparameter selection'))
protocol=dict(version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    user_request='Fit all existing datasets where the task and sample size permit; compare all historically promising families; concise results-first report',
    cases=cases,neural_families=['chronos2','tirex2','timesfm25','timesfm3','toto2','patchtst','moirai2','ttm'],
    historical_method_families=['Equal Chronos2 TiRex2','25/25/50 Chronos2 TiRex2 TimesFM3','Ridge residual calibration',
        'Direct Ridge','Direct LightGBM','JointFactorFourier','Multiscale sparse MoE','HDSF halfday simplex','HDSF smooth overlapping weights','R48 fixed original control'],
    neural_budget=dict(scope='last block plus output head; TTM decoder/head',checkpoints=[0,16,48],steps=48,
        learning_rate=3e-6,seed=20261013,train_labels_only=True,select_checkpoints_on_DEV_early_only=True),
    training_horizons='96 points for Chronos2 TiRex2 TimesFM2.5; alternating96/672 for TimesFM3 Toto2 PatchTST TTM; native64-point quantiles for Moirai2. Both forecast horizons are evaluated; training scopes are explicitly labelled.',
    within_target_fit='Raw neural parameters only for datasets with adequate training windows. Sea daily model-specific ridge residual calibration is separately labelled and does not claim neural weight updates.',
    origin_rule='Weekly fixed calendar candidates for NLR, eligibility strictly past28day>=95% and latest valid; no future-target-availability filtering',
    selection='DEV early half for neural checkpoints and meta-parameter fit, late half for model/combination choice; frozen SHA before evaluation. For Google and Sea use purged task-specific early/late DEV origins. No test reselection.',
    metrics='Same 9.28 aligned evaluator: macro per-series WAPE; complete96 daily means; MAE RMSE P95 APE hit2/hit5 bias daily errors and hits peak amplitude/time/underestimate; point masks shared. Daily energy uses ppd1 and no intraday peaks.',
    uncertainty='Descriptive paired-period/quarter gains plus dependency-preserving 7/14/28-day block sensitivity when origin counts permit; explicitly exploratory, not calibrated significance',
    reporting='No pooled cross-unit WAPE. Rank within dataset/task; assess across-period and across-target consistency; no universal winner guarantee.',
    cluster='Actual cell-a CPU/memory history only; conditional power-only/covariate control within original Google, 5 covered and57 total, never counted as new independent power dataset',
    licenses='Original data/model licenses retained; TimesFM3 noncommercial research restriction persists',
    pretraining_overlap_unknown=True,old_tests_exposed=True,existing_selections_and_deployments_unchanged=True,
    source_bindings={str(p.relative_to(ROOT)):sha(p) for p in [OLD/'data/nlr_15min.npz',OLD/'data/sea_daily.npz',
        ROOT/'当前推荐方案.json',ROOT/'experiment/round46/code/hdsf_foundation_adapters.py',ROOT/'experiment/round35/code/moirai_native_adapter.py',ROOT/'experiment/round29/code/aligned_metrics.py']})
save(F/'docs/protocol.json',protocol);(F/'docs/protocol.sha256').write_text(sha(F/'docs/protocol.json'),encoding='ascii')
with (F/'data/weekly_origin_audit.csv').open('w',encoding='utf-8-sig',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=coverage[0].keys());w.writeheader();w.writerows(coverage)
save(F/'results/preparation.json',dict(all_passed=True,cases={k:{'cutoff':v['cutoff'],'origins':{p:len(o) for p,o in v['origins'].items()}} for k,v in cases.items()},
    future_labels_not_used_for_origin_eligibility=True,source_preparations_unchanged=True))
print(json.dumps(dict(all_passed=True,cases=protocol['cases']),ensure_ascii=True))
