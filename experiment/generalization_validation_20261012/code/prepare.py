"""Read-only source processing; quality eligibility does not use forecast errors."""
from pathlib import Path
import argparse,sys
sys.dont_write_bytecode=True
import numpy as np
from core import *
def prepare_nlr(root,folder):
    import pyarrow.parquet as pq
    p=root/'nlr-hpc-pue/raw/esif.influx.buildingData.PUE.combined.parquet'
    first=np.datetime64('2015-11-10','us');end=np.datetime64('2025-08-30','us');nb=int((end-first)/np.timedelta64(15,'m'))
    sums=np.zeros((2,nb));counts=np.zeros((2,nb),np.int32);span0=np.full((2,nb),np.inf);span1=np.full((2,nb),-np.inf)
    gaps=np.zeros(nb);previous=None;rows=0;invalid=np.zeros(2,np.int64)
    fields=['it_power_kw','cooling_kw','hvac_kw','plug_and_light_kw','pump_kw']
    for batch in pq.ParquetFile(p).iter_batches(batch_size=65536,columns=['ts']+fields):
        t=batch.column(0).to_numpy(zero_copy_only=False).astype('datetime64[us]').astype(np.int64)
        assert (np.diff(t)>0).all() and (previous is None or t[0]>previous)
        cols=np.stack([batch.column(i+1).to_numpy(zero_copy_only=False) for i in range(5)])
        idx=((t-first.astype(np.int64))//900000000).astype(int);assert (idx>=0).all() and (idx<nb).all()
        full=np.isfinite(cols).all(0)&(cols>=0).all(0)&(cols[0]>0)
        valid_it=np.isfinite(cols[0])&(cols[0]>0);values=np.stack([cols.sum(0),cols[0]])
        delta=np.diff(t,prepend=t[0] if previous is None else previous)/1e6
        prevbin=np.r_[idx[0] if previous is None else (previous-first.astype(np.int64))//900000000,idx[:-1]]
        same=prevbin==idx;np.maximum.at(gaps,idx[same],delta[same]);previous=t[-1]
        for j,good in enumerate([full,valid_it]):
            sums[j]+=np.bincount(idx[good],weights=values[j,good],minlength=nb)
            counts[j]+=np.bincount(idx[good],minlength=nb).astype(np.int32)
            np.minimum.at(span0[j],idx[good],t[good]/1e6);np.maximum.at(span1[j],idx[good],t[good]/1e6)
            invalid[j]+=int((~good).sum())
        rows+=len(t)
        if rows%524288==0:print('NLR source records processed',rows,flush=True)
    valid=(counts>=12)&((span1-span0)>=600)&(gaps[None]<=180)
    actual=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0)
    # No backward fill or interpolation. Unobserved initial cells stay zero and
    # are never eligible as a model context; later missing cells use past only.
    x=actual.copy()
    for j in range(2):
        positions=np.where(valid[j],np.arange(nb),-1);positions=np.maximum.accumulate(positions)
        x[j]=np.where(positions>=0,actual[j,np.maximum(positions,0)],0)
    dates=np.arange(first.astype('datetime64[D]'),end.astype('datetime64[D]'))
    y,mask,x=[a.reshape(2,-1,96) for a in [actual,valid,x]]
    train_end=int(np.searchsorted(dates,np.datetime64('2023-01-01')))
    scales=np.array([np.quantile(y[j,:train_end][mask[j,:train_end]],.95)/.8 for j in range(2)])
    origins={};coverage=[]
    for phase,year,months in [('dev',2023,range(1,13)),('test',2024,range(1,13)),('confirm',2025,range(1,9))]:
        origins[phase]=[]
        for month in months:
            for day in [1,15]:
                date=np.datetime64(f'{year}-{month:02d}-{day:02d}');o=int(np.searchsorted(dates,date));hist=mask[:,o-28:o]
                eligible=bool((hist.mean((1,2))>=.95).all() and hist[:,-1,-1].all() and mask[:,o:o+7].any())
                coverage.append(dict(phase=phase,date=str(date),origin=o,eligible=eligible,history_coverage=hist.mean((1,2)).tolist(),target_valid_points=mask[:,o:o+7].sum((1,2)).tolist()))
                if eligible:origins[phase].append(o)
    np_save(folder/'data/nlr_15min.npz',y=y,valid=mask,x=x,dates=dates.astype(str),scales=scales)
    info=dict(utc=utc(),rows=rows,invalid_source_rows=invalid.tolist(),shape=list(y.shape),names=['facility_total_kw','it_power_kw'],unit='kW',
        facility_formula='it_power_kw+cooling_kw+hvac_kw+plug_and_light_kw+pump_kw; no extra 2.67kW',
        timezone='not supplied; use naive source clock, do not claim local civil time or UTC',
        bin_rule='left-closed 15min arithmetic mean of valid samples; >=12 samples, observed span>=600s, internal gap<=180s',
        missing_input='causal forward fill; unchanged missing target mask; no backward/future interpolation',
        origins=origins,coverage=coverage,scales=scales.tolist(),scale_fit_label_end='2022-12-31',
        source_sha256=sha(p),prepared_sha256=sha(folder/'data/nlr_15min.npz'))
    assert all(len(v)>=6 for v in origins.values()),origins
    save(folder/'data/nlr_preparation.json',info);export(folder/'data/nlr_origin_coverage.csv',coverage)
    print('NLR eligible fixed monthly origins', {k:len(v) for k,v in origins.items()},flush=True)
def prepare_sea(root,folder):
    import openpyxl,datetime
    p=root/'海光寺-能耗能效报表_9.1-9.30电能统计V2.xlsx';w=openpyxl.load_workbook(p,read_only=True,data_only=True)
    records=list(w['报表'].iter_rows(min_row=21,max_row=110,min_col=1,max_col=8,values_only=True));by={}
    for r in records:
        date=str(r[0])[:10];by.setdefault(date,[]).append(r)
    values=[];dates=[]
    for date,group in sorted(by.items()):
        assert len(group)==3
        for col in [1,2,4]:assert abs(float(group[0][col])-float(group[1][col])-float(group[2][col]))<1e-6
        dates.append(date);values.append(float(group[0][4]))
    w.close();y=np.array(values)[None,:,None];assert len(values)==30
    np_save(folder/'data/sea_daily.npz',y=y,valid=np.ones_like(y,bool),x=y.copy(),dates=np.array(dates),scales=np.array([np.quantile(y[:,:14],.95)/.8]))
    save(folder/'data/sea_preparation.json',dict(utc=utc(),days=30,unit='kWh',source_sha256=sha(p),sheet='报表',range='A20:H110',
        hierarchical_rows='daily total selected; two subordinate rows not counted as extra independent series',
        daily_only=True,old_month_previously_exposed=True,independent_confirmation=False,
        origins={'dev1':list(range(14,21)),'dev7':[14,15],'test1':list(range(22,30)),'test7':[22,23]},
        unavailable=['intraday curve','intraday peak magnitude/time','independent unseen month']))
def register(root,folder):
    p=folder/'docs/protocol.json'
    if p.exists():protocol_check(folder);return
    n=read(folder/'data/nlr_preparation.json')
    binding={str(p.relative_to(root)):sha(p) for p in [root/'experiment/round48/deployment/selected_configuration.json',root/'experiment/round48/results/predictions.npz',root/'experiment/round44/results/selection.json',root/'experiment/round44/results/local_cut22.joblib',root/'experiment/round47/models/patchtst_curve_cut22.pt',root/'experiment/round46/code/hdsf_foundation_adapters.py',root/'experiment/round44/results/shared_weights.json',root/'experiment/round44/results/official_models_downloads.json']}
    cfg=read(root/'experiment/round46/results/selection.json')['families']
    for f in ['timesfm3','toto2','patchtst']:
        q=root/'experiment/round46/models'/f'{f}_{cfg[f]["config"]}_cut22.pt';binding[str(q.relative_to(root))]=sha(q)
    protocol=dict(created_utc=utc(),version=1,user_request='科学泛化验证，保持统一评价指标',
        source_scope='NLR facility/IT separately; SeaGuang daily energy only; GoogleCluster same-source ablation only',
        fixed_google_current='R48_ROBUST_HDSF_USER_ADOPTED',source_bindings=binding,
        code_sha256={p.name:sha(p) for p in (folder/'code').glob('*.py')},
        prepared_sha256={p.name:sha(p) for p in (folder/'data').glob('*.npz')},
        nlr_origins=n['origins'],origin_rule='1st/15th of every month; each target past28day coverage>=95%, latest inputbin valid. Fixed candidate dates, quality-only exclusions logged. A representative calendar backtest, not all days.',
        nlr_train='2016-06-12 to2022-12-31',nlr_dev='2023 fixed origins',nlr_test='2024 fixed origins',nlr_confirmation='2025 Jan-Aug fixed origins; no 2024 retuning',
        horizons=[1,7],history='Chronos28days; TimesFM long/Patch21days; other14days; all known history only',
        adapter='Per-target training-only scale; source clock; no fictitious PDU or production channel. Single-target joint modes collapse to univariate. Nonnegative output, no artificial kW upper cap.',
        current_transfer='Google model weights and halfday simplex unchanged; missing production/static/other-series features follow missing tree paths. Group shrink inactive when fewer than3 comparable facilities.',
        refit='NLR local24 same source73 feature layout with unavailable columns NA, fourseed350tree15leaf plus train-only lag blend; Ridge strong control. Neural originalweights unchanged. Halfday simplex fit on early2023 origins; alpha against global-simplex selected on late2023 only; freeze before TEST.',
        simplex_candidates=[0,.25,.5,1],simplex_training_segment_points=48,
        metrics='Unchanged per-series macro WAPE, complete96 daily mean, MAE/RMSE/P95APE/hit2/hit5/absolute bias/peak amplitude/time/underestimate; 80%empirical intervals. Sea daily granularity peak fields unavailable.',
        zero_APE='NA with counts; no zero denominator concealment',
        intervals='Point transfer fixed; residual interval calibration uses target DEV and is explicitly target-calibrated, not zero-shot uncertainty',
        uncertainty='Paired origin-month bootstrap 2000replications, both origins and all horizon points within month kept together; at most12 testmonth clusters, no IIDpoint significance. Confirmation separate.',
        cluster='Fixed LightGBM known-history power vs power+CPU/memory inputs, train14/dev<=22/final22; same originalorigins; fulla5 andall57 conditional residual results; no independentconfirmation.',
        source_foundation_training_data_overlap_unknown=True,no_test_parameter_selection=True,
        resource_policy='One isolated model process at a time; use existingGPU except TiRex nativeCPU; emergency check between calls only, no 3GiB startup gate; no other applications closed.',
        license='Original licenses retained; TimesFM3 noncommercial research only')
    save(p,protocol);(p.with_suffix('.sha256')).write_text(sha(p),encoding='ascii')
    print('Protocol frozen before DEV and TEST',sha(p),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',choices=['data','nlr','sea','register','parity'],required=True);a=ap.parse_args();root=Path(a.root);folder=root/'experiment/generalization_validation_20261012'
    for d in ['data','docs','results','models','cache']:(folder/d).mkdir(parents=True,exist_ok=True)
    if a.stage=='data':prepare_nlr(root,folder);prepare_sea(root,folder)
    elif a.stage=='nlr':prepare_nlr(root,folder)
    elif a.stage=='sea':prepare_sea(root,folder)
    elif a.stage=='register':register(root,folder)
    else:parity(root,folder/'results/metric_parity.json');print('Canonical metric parity passed',flush=True)
