from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,sys,datetime,math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

T=Path(sys.argv[1]);R=T/'nlr-hpc-pue';Q=R/'results'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest_path=R/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
weather=R/'raw/esif.influx.buildingData.outside.combined.parquet'
if not weather.exists():
    part=weather.with_name(weather.name+'.part');assert part.stat().st_size==58865468
    with part.open('rb') as f:assert f.read(4)==b'PAR1';f.seek(-4,2);assert f.read(4)==b'PAR1'
    pf=pq.ParquetFile(part);assert pf.metadata.num_rows>0;pf.close()
    part.replace(weather)
    manifest['files'].append(dict(path=str(weather.relative_to(R)),source_url='https://data.nlr.gov/system/files/300/1757105566-esif.influx.buildingData.outside.combined_2.parquet',bytes=weather.stat().st_size,sha256=sha(weather),transport='Windows curl HTTPS with certificate verification and official redirects',http_status=200,http_size_download=58865468,complete=True,acquired_utc=utc(),signed_redirect_queries_not_saved=True))
    save(manifest_path,manifest)

files={};month_records=[];relation={k:0. for k in ['count','absolute_pue_sum_error','absolute_pue_sum_plus_2_67_error','offset_kw_sum','offset_kw_sq_sum','offset_abs_le_0_1','offset_abs_le_1']};offset_sample=[]
for kind,path in [('power',R/'raw/esif.influx.buildingData.PUE.combined.parquet'),('weather',weather)]:
    print('Auditing '+kind+' in bounded batches',flush=True)
    pf=pq.ParquetFile(path,pre_buffer=False);schema=pf.schema_arrow;fields=schema.names;rows=pf.metadata.num_rows
    declared={r['path']:r for r in manifest['files']};entry=declared[str(path.relative_to(R))]
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
    temporal_type=schema.field('ts').type;assert pa.types.is_timestamp(temporal_type)
    unit=temporal_type.unit;divisor={'s':1,'ms':1000,'us':1000000,'ns':1000000000}[unit]
    timestamp_index=Q/f'.timestamp_audit_{kind}.npy';assert timestamp_index.resolve().is_relative_to(Q.resolve())
    index=np.lib.format.open_memmap(timestamp_index,mode='w+',dtype='int64',shape=(rows,));cursor=0;observed=0;previous=None;out_of_order=0;adjacent_duplicates=0
    stats={name:dict(type=str(schema.field(name).type),nulls=0,nonfinite_nonnull=0,finite=0,negative=0,zero=0,min=None,max=None,first_finite_ts=None,last_finite_ts=None) for name in fields if name!='ts'}
    monthly=defaultdict(Counter);anom=Counter()
    for batch in pf.iter_batches(batch_size=65536,use_threads=False):
        values=batch.to_pydict() if False else None
        arrays={name:batch.column(i) for i,name in enumerate(fields)}
        ts=arrays['ts'].to_numpy(zero_copy_only=False).astype('datetime64['+unit+']');validts=~np.isnat(ts);ticks=ts[validts].view('int64');index[cursor:cursor+len(ticks)]=ticks;cursor+=len(ticks)
        if len(ticks):
            delta=np.diff(ticks)
            out_of_order+=int((delta<0).sum());adjacent_duplicates+=int((delta==0).sum())
            if previous is not None:out_of_order+=int(ticks[0]<previous);adjacent_duplicates+=int(ticks[0]==previous)
            previous=int(ticks[-1])
        months=ts.astype('datetime64[M]');uniq,counts=np.unique(months[validts],return_counts=True)
        for month,count in zip(uniq,counts):monthly[str(month)]['rows']+=int(count)
        numeric={}
        for name in stats:
            ar=arrays[name];st=stats[name];st['nulls']+=ar.null_count
            typ=schema.field(name).type
            if not (pa.types.is_integer(typ) or pa.types.is_floating(typ)):continue
            v=ar.to_numpy(zero_copy_only=False);good=np.isfinite(v);numeric[name]=v
            st['finite']+=int(good.sum());st['nonfinite_nonnull']+=int((~good).sum())-ar.null_count;st['negative']+=int((v[good]<0).sum());st['zero']+=int((v[good]==0).sum())
            if good.any():
                lo=float(v[good].min());hi=float(v[good].max());st['min']=lo if st['min'] is None else min(lo,st['min']);st['max']=hi if st['max'] is None else max(hi,st['max'])
                dated=ts[good&validts]
                if len(dated):
                    lo=str(dated.min());hi=str(dated.max());st['first_finite_ts']=lo if st['first_finite_ts'] is None else min(lo,st['first_finite_ts']);st['last_finite_ts']=hi if st['last_finite_ts'] is None else max(hi,st['last_finite_ts'])
            if name in ['pue','it_power_kw','outdoor_air_temp','outdoor_air_humidity']:
                uu,cc=np.unique(months[good&validts],return_counts=True)
                for month,count in zip(uu,cc):monthly[str(month)]['finite_'+name]+=int(count)
        if kind=='power':
            pue=numeric['pue'];it=numeric['it_power_kw'];anom['pue_below_one']+=int(((pue<1)&np.isfinite(pue)).sum());anom['pue_nonpositive']+=int(((pue<=0)&np.isfinite(pue)).sum());anom['pue_above_five_diagnostic']+=int(((pue>5)&np.isfinite(pue)).sum());anom['it_below_1kw_diagnostic']+=int(((it<1)&np.isfinite(it)).sum())
            component_names=['it_power_kw','cooling_kw','hvac_kw','plug_and_light_kw','pump_kw'];good=validts&np.isfinite(pue)&np.isfinite(it)&(it>0)
            for name in component_names:good &=np.isfinite(numeric[name])
            total=sum(numeric[n][good] for n in component_names);pred=total/it[good];offset=pue[good]*it[good]-total
            relation['count']+=len(offset);relation['absolute_pue_sum_error']+=float(abs(pred-pue[good]).sum());relation['absolute_pue_sum_plus_2_67_error']+=float(abs((total+2.67)/it[good]-pue[good]).sum());relation['offset_kw_sum']+=float(offset.sum());relation['offset_kw_sq_sum']+=float((offset**2).sum());relation['offset_abs_le_0_1']+=int((abs(offset)<=.1).sum());relation['offset_abs_le_1']+=int((abs(offset)<=1).sum());offset_sample.extend(offset[::32].tolist())
        else:
            humidity=numeric.get('outdoor_air_humidity');temp=numeric.get('outdoor_air_temp')
            if humidity is not None:anom['humidity_outside_0_100']+=int(((humidity<0)|(humidity>100)).sum())
            if temp is not None:anom['temperature_outside_minus100_150F_diagnostic']+=int(((temp<-100)|(temp>150)).sum())
        observed+=batch.num_rows
    assert observed==rows
    index.flush();ordered=index[:cursor];ordered.sort(kind='quicksort');cadence=Counter();dups=0;gaps5=0;gaps15=0;gapsday=0;max_gap=-1;max_pair=None
    prior=None
    for start in range(0,cursor,65536):
        ticks=np.asarray(ordered[start:start+65536]);part=np.concatenate(([prior],ticks)) if prior is not None else ticks;dt=np.diff(part)
        if len(dt):
            dups+=int((dt==0).sum());seconds=dt/divisor;gaps5+=int((seconds>300).sum());gaps15+=int((seconds>900).sum());gapsday+=int((seconds>86400).sum())
            vv,cc=np.unique(np.rint(seconds[seconds>0]).astype('int64'),return_counts=True)
            cadence.update(dict(zip(map(int,vv),map(int,cc))))
            j=int(dt.argmax())
            if dt[j]>max_gap:max_gap=int(dt[j]);max_pair=[int(part[j]),int(part[j+1])]
        if len(ticks):prior=int(ticks[-1])
    def iso(tick):return str(np.datetime64(tick,unit))
    def weighted_quantile(q):
        want=q*sum(cadence.values());acc=0
        for value,count in sorted(cadence.items()):
            acc+=count
            if acc>=want:return value
        return None
    timeline=dict(min=iso(int(ordered[0])),max=iso(int(ordered[-1])),timestamp_type=str(temporal_type),timezone=temporal_type.tz,
        timezone_supplied=temporal_type.tz is not None,nonnull_timestamps=cursor,unique_timestamps=cursor-dups,duplicate_timestamps=dups,
        original_out_of_order_steps=out_of_order,original_adjacent_duplicates=adjacent_duplicates,
        positive_interval_seconds_rounded_median=weighted_quantile(.5),positive_interval_seconds_rounded_p95=weighted_quantile(.95),
        common_intervals_seconds_rounded=cadence.most_common(10),gaps_over_5min=gaps5,gaps_over_15min=gaps15,gaps_over_one_day=gapsday,
        longest_gap_seconds=max_gap/divisor,longest_gap_times=[iso(x) for x in max_pair],cadence_is_not_assumed_regular=True)
    zero_left=int(np.searchsorted(ordered,0,side='left'));zero_right=int(np.searchsorted(ordered,0,side='right'))
    timeline['unix_epoch_zero_timestamp_rows']=zero_right-zero_left
    timeline['first_timestamp_after_unix_epoch']=iso(int(ordered[zero_right])) if zero_right<cursor else None
    for st in stats.values():st['null_percent']=100*st['nulls']/rows
    files[kind]=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),rows=rows,row_groups=pf.metadata.num_row_groups,
        schema=[dict(name=f.name,type=str(f.type)) for f in schema],timeline=timeline,fields=stats,anomaly_counts=dict(anom),all_batches_decoded=True)
    for month,v in sorted(monthly.items()):month_records.append(dict(dataset=kind,month=month,**dict(v)))
    del ordered
    index._mmap.close()
    del index
    timestamp_index.unlink()
    pf.close()
    print(json.dumps(dict(dataset=kind,rows=rows,time=timeline,anomalies=dict(anom)),ensure_ascii=True),flush=True)
count=relation.pop('count')
relation_out=dict(matched_rows=int(count),sum_components_pue_mae=relation['absolute_pue_sum_error']/count,
    sum_plus_2_67kw_pue_mae=relation['absolute_pue_sum_plus_2_67_error']/count,offset_kw_mean=relation['offset_kw_sum']/count,
    offset_kw_rmse=math.sqrt(relation['offset_kw_sq_sum']/count),offset_abs_le_0_1kw_percent=100*relation['offset_abs_le_0_1']/count,
    offset_abs_le_1kw_percent=100*relation['offset_abs_le_1']/count,offset_kw_systematic_stride32_sample_quantiles=np.quantile(offset_sample,[0,.01,.5,.99,1]).tolist(),
    note='Accounting diagnostic using simultaneous raw fields, not a forecast or permission to add the filter-pump constant twice; no rows cleaned or imputed.')
out=dict(all_files_readable=True,original_data_unchanged=True,no_rows_cleaned_or_imputed=True,files=files,component_accounting_diagnostic=relation_out,
    suitability='usable with timestamp alignment, target-specific masks and separate independent temporal evaluation; no training performed',created_utc=utc())
out['catalog_actual_schema_variations']={
    'weather_catalog_to_actual':{'outside_air_temp':'outdoor_air_temp','outside_air_humidity':'outdoor_air_humidity'},
    'weather_unit_field_is_not_physical_units':'Observed nonnull string is outside; Fahrenheit and relative humidity percent are from the official catalog.',
    'power_extra_fields':['tags','ere'],
    'energy_reuse_and_ere':'Both preserved; exact distinction is not resolved by the downloaded README and must not be silently merged.',
    'raw_schema_not_renamed':True}
save(Q/'quality_audit.json',out)
import csv
keys=list(dict.fromkeys(k for r in month_records for k in r))
with (Q/'monthly_coverage.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(month_records)
manifest['selected_data_files_complete']=True;manifest['raw_data_bytes']=sum(r['bytes'] for r in manifest['files'] if r['path'].startswith('raw'))
manifest['data_quality_audit']='results/quality_audit.json';manifest['all_batches_readable']=True;manifest['updated_utc']=utc();save(manifest_path,manifest)
save(Q/'download_progress.json',dict(state='complete',selected_data_files=2,bytes=manifest['raw_data_bytes'],remaining_download_bytes=0,all_batches_readable=True,utc=utc()))
if (Q/'download_error.json').exists():
    save(Q/'download_recovery.json',dict(state='resolved',prior_failure_evidence='results/download_error.json',weather_recovered_via='Windows curl HTTPS',all_files_complete=True,sha256_verified=True,utc=utc()))
print(json.dumps(dict(all_files_readable=True,raw_data_bytes=manifest['raw_data_bytes'],component_accounting=relation_out),ensure_ascii=True),flush=True)
