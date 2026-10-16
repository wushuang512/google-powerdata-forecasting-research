"""New user protocol; raw sources unchanged, label means distinct from past imputation."""
from pathlib import Path
import argparse,gc,sys
import numpy as np,pandas as pd,pyarrow.parquet as pq
from core import *
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F=R/'experiment/nlr_strict_protocol_20261013'
for s in ['code','data','models','cache','results','docs']:(F/s).mkdir(parents=True,exist_ok=True)
if (F/'data/processed.npz').exists():
    info=read(F/'data/quality_report.json');assert info['processed_sha256']==sha(F/'data/processed.npz');print('Existing audited new-protocol preprocessing reused');raise SystemExit()
fields=['it_power_kw','cooling_kw','hvac_kw','plug_and_light_kw','pump_kw'];weather=['outdoor_air_temp','outdoor_air_humidity']
cfg=dict(version=2,strict_user_protocol_rerun=True,created_utc=utc(),target='facility_power_kw',target_definition='sum of five separately aggregated 15min interval means; any unavailable component => missing; derived subtotal, not direct facility meter',
 timezone='UTC',timezone_verified=False,source_clock_is_naive=True,start='2015-01-01',end='2026-01-01',frequency_minutes=15,closed='left',label='left',history_days=14,history_points=1344,
 covariate_ffill_limit=4,target_fill=False,past_target_fill=False,power_range_kw=[0,20000],temperature_range_c=[-70,60],humidity_range_percent=[0,100],
 physical_bounds_are_fixed_broad_screen_not_rated_capacity=True,split_fractions=[.6,.2,.2],minimum_validation_days=8,minimum_test_days=9,
 split_interpretation='trim only leading/trailing incomplete target days; continuous whole calendar days split in order, retain internal missing dates',
 aggregation='finite sample mean per component after duplicate timestamp mean; no minimum15-minute-record coverage imposed',origin_time='00:00 each source-clock day, previous interval starts t-15min',
 label_eligibility='all H labels inside split and finite; same cohort for every model; historical label unfilled and all historical covariates finite after limited forward fill',
 future_inputs='known calendar only; no actual future component loads or weather',mape_threshold=1e-8,mape_aggregation='point MAPE per origin then equal origin mean; excluded counts and pooled supplement reported',
 model_selection='validation-origin mean MAPE; HDSF purged forward validation CV MAPE, in-sample score separately labelled',
 test_policy='all tasks frozen before first test inference; rolling history updates only, no online refits; daily and fixed-calendar every7day subset',
 all_candidates_registered_before_validation=True,recompute_all_forecasts_in_new_folder=True,source_raw_files_reused_by_verified_sha_only=True,
 models=['LightGBM historical power weather calendar','TimesFM3 original weights history14d','Toto2 original weights history14d','Moirai2 original weights history14d','HDSF four-expert halfday target simplex'],
 baseline_models=['previous day','previous week','previous3day mean'],foundation_target_gradient_training=False,historical_candidate_count=6,
 overall_selection='common purged forward-validation origin-mean MAPE for all fourteen predeclared methods, including HDSF; never TEST ranking',
 independent_confirmation=False,previous_test_exposure_explicitly_acknowledged=True,
 baseline_for_relative_gains={'1':'previous day','7':'previous3day mean'},hdsf_cv_folds=3,hdsf_alpha_grid=[0,.5,1],hdsf_segment_points=48,
 lightgbm_candidates=[dict(num_leaves=15,reg_lambda=10,n_estimators=200),dict(num_leaves=31,reg_lambda=20,n_estimators=200)],
 hdsf_fit_objective='WAPE/L1 halfday simplex as original; model selection remains origin mean MAPE',
 code_sha256={p.name:sha(p) for p in (F/'code').glob('*.py')},previous_test_exposure=True,independent_external_facilities=1,
 reference_sources=['https://data.nlr.gov/submissions/300','https://otexts.com/fpp3/tscv.html','https://otexts.com/fpp3/accuracy.html'])
save(F/'docs/protocol.json',cfg);(F/'docs/protocol.sha256').write_text(sha(F/'docs/protocol.json'))
quality=dict(utc=utc(),timezone_verified=False,configuration_timezone='UTC provisional only',source_timezone_not_supplied=True,source_audit={},raw_data_modified=False)
def aggregate(file,cols,kind):
    path=R/'nlr-hpc-pue/raw'/file;df=pq.read_table(path,columns=['ts']+cols).to_pandas();n=len(df);ts=pd.to_datetime(df.pop('ts'),utc=True,errors='coerce');df.index=ts
    range_mask=((ts>=pd.Timestamp(cfg['start'],tz='UTC'))&(ts<pd.Timestamp(cfg['end'],tz='UTC'))).to_numpy();removed=int((~range_mask).sum());df=df.loc[range_mask].sort_index();counts={}
    for col in cols:
        val=pd.to_numeric(df[col],errors='coerce')
        if col=='outdoor_air_temp':val=(val-32)*5/9;low,high=cfg['temperature_range_c']
        elif col=='outdoor_air_humidity':low,high=cfg['humidity_range_percent']
        else:low,high=cfg['power_range_kw']
        valid=np.isfinite(val)&(val>=low)&(val<=high);counts[col]=int((~valid).sum());df[col]=val.where(valid)
    duplicates=int(df.index.duplicated().sum());df=df.groupby(level=0,sort=True).mean();agg=df.resample('15min',closed='left',label='left').mean();count=df.resample('15min',closed='left',label='left').count()
    quality['source_audit'][kind]=dict(file=str(path),sha256=sha(path),original_rows=n,outside_date_range_removed=removed,duplicates_averaged=duplicates,masked_invalid_rows=counts,rows_after_dedup=len(df),interval_finite_record_counts='separate CSV per field; not required to be15')
    del df;gc.collect();return agg,count
power,pc=aggregate('esif.influx.buildingData.PUE.combined.parquet',fields,'power');outside,wc=aggregate('esif.influx.buildingData.outside.combined.parquet',weather,'weather')
df=power.join(outside,how='left');count=pc.join(wc,how='left').fillna(0).astype(int);df['facility_power_kw']=df[fields].sum(axis=1,min_count=5)
dates=df.index.floor('D');complete=df[cfg['target']].notna().groupby(dates).agg(['sum','count']);full=complete[(complete['sum']==96)&(complete['count']==96)].index
if len(full)<32:raise RuntimeError('Too few completely covered boundary days for requested train/validation/test')
first,last=full[0],full[-1];inside=(df.index>=first)&(df.index<last+pd.Timedelta(days=1));df=df.loc[inside];count=count.loc[inside]
assert len(df)%96==0;D=len(df)//96;ntrain=int(np.floor(.6*D));nval=max(8,int(np.floor(.2*D)));ntest=D-ntrain-nval
if ntest<9:raise RuntimeError('Need at least9 test days after temporal splits')
split=dict(train=[0,ntrain],validation=[ntrain,ntrain+nval],test=[ntrain+nval,D]);cov=df[fields+weather].ffill(limit=4)
dates=np.array([str(t)[:10] for t in pd.date_range(first,last,freq='D')]);y=df[cfg['target']].to_numpy().reshape(D,96);it=df['it_power_kw'].to_numpy().reshape(D,96);X=cov.to_numpy().reshape(D,96,7)
rawcov=df[fields+weather].to_numpy().reshape(D,96,7);grids=np.arange(D*96).reshape(D,96);origin_records=[];origins={}
for phase,(start,end) in split.items():
    for H in [1,7]:
        eligible=[]
        for o in range(start,end):
            enough=o>=14;past_target=bool(enough and np.isfinite(y[o-14:o]).all());past_cov=bool(enough and np.isfinite(X[o-14:o]).all());bounded=o+H<=end;labels=bool(bounded and np.isfinite(y[o:o+H]).all())
            ok=past_target and past_cov and bounded and labels;reason='eligible' if ok else'insufficient_history' if not enough else'missing_past_target' if not past_target else'missing_past_covariates' if not past_cov else'labels_cross_split' if not bounded else'missing_labels'
            origin_records.append(dict(phase=phase,horizon_days=H,origin_day=o,date=dates[o],eligible=ok,reason=reason));
            if ok:eligible.append(o)
        origins[phase+str(H)]=eligible
        if not eligible:raise RuntimeError(f'No valid {phase} {H}day windows under frozen strict missingness protocol; cannot fabricate a result')
np_save(F/'data/processed.npz',y=y,it_y=it,covariates=X,raw_covariates=rawcov,dates=dates,fields=np.array(fields+weather))
out=df.copy();out.index.name='interval_start_UTC_provisional';out.to_csv(F/'data/processed_15min.csv',encoding='utf-8-sig');count.columns=[c+'_raw_finite_records' for c in count.columns];count.to_csv(F/'data/raw_interval_coverage.csv',encoding='utf-8-sig');export(F/'data/window_eligibility.csv',origin_records)
quality.update(derived_target_not_meter=True,first_complete_day=str(first),last_complete_day=str(last),calendar_days=D,retained_intervals=len(df),internal_missing_target_intervals=int((~np.isfinite(y)).sum()),complete_target_days=int(np.isfinite(y).all(1).sum()),
 split=split,split_dates={k:[dates[v[0]],dates[v[1]-1]] for k,v in split.items()},eligible_origins={k:len(v) for k,v in origins.items()},skip_counts={k:{reason:sum(r['phase']==k and r['reason']==reason for r in origin_records) for reason in sorted(set(r['reason'] for r in origin_records))} for k in split},
 processed_sha256=sha(F/'data/processed.npz'),protocol_sha256=sha(F/'docs/protocol.json'),historical_ffill_actual_intervals=int((np.isfinite(X)&~np.isfinite(rawcov)).sum()),targets_never_filled=True)
save(F/'data/origins.json',dict(origins=origins,split=split,dates=quality['split_dates']));save(F/'data/quality_report.json',quality)
print(json.dumps(dict(complete=True,days=D,split_dates=quality['split_dates'],eligible=quality['eligible_origins'],missing_target_intervals=quality['internal_missing_target_intervals']),ensure_ascii=False))
