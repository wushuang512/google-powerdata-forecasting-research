"""Independently aggregate raw minute records with NumPy, without producer code."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np,pandas as pd,pyarrow.parquet as pq
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root);F=R/'experiment/nlr_user_protocol_20261013'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
cfg=read(F/'docs/protocol.json');quality=read(F/'data/quality_report.json');checks=[];details={}
def check(name,passed):checks.append(dict(name=name,passed=bool(passed)));assert passed,name
with np.load(F/'data/processed.npz') as z:raw=z['raw_covariates'];fields=z['fields'];dates=z['dates']
N=raw.shape[0]*96;step=15*60*10**9;begin=pd.Timestamp(str(dates[0]),tz='UTC').value
lo=pd.Timestamp(cfg['start'],tz='UTC').value;hi=pd.Timestamp(cfg['end'],tz='UTC').value
count_table=pd.read_csv(F/'data/raw_interval_coverage.csv',encoding='utf-8-sig')
for kind,cols in [('power',list(fields[:5])),('weather',list(fields[5:]))]:
    path=Path(quality['source_audit'][kind]['file']);check('Raw hash '+kind,sha(path)==quality['source_audit'][kind]['sha256'])
    sums=np.zeros((N,len(cols)));counts=np.zeros((N,len(cols)),dtype=np.int64);masked={c:0 for c in cols};outside=0;rows=0
    # Producer has no duplicates in either actual source. Require it explicitly.
    check('No duplicate timestamps in audited source '+kind,quality['source_audit'][kind]['duplicates_averaged']==0)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100000,columns=['ts']+cols):
        df=batch.to_pandas();ts=pd.to_datetime(df['ts'],utc=True,errors='coerce').dt.as_unit('ns').astype('int64').to_numpy();rows+=len(ts)
        validtime=(ts>=lo)&(ts<hi);outside+=int((~validtime).sum());binid=(ts-begin)//step;inside=validtime&(binid>=0)&(binid<N)
        for j,c in enumerate(cols):
            val=pd.to_numeric(df[c],errors='coerce').to_numpy(dtype=float)
            if c=='outdoor_air_temp':val=(val-32)*(5/9);bounds=cfg['temperature_range_c']
            elif c=='outdoor_air_humidity':bounds=cfg['humidity_range_percent']
            else:bounds=cfg['power_range_kw']
            valid=np.isfinite(val)&(val>=bounds[0])&(val<=bounds[1]);masked[c]+=int((validtime&~valid).sum());take=inside&valid
            sums[:,j]+=np.bincount(binid[take],weights=val[take],minlength=N)
            counts[:,j]+=np.bincount(binid[take],minlength=N)
    means=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0);maxdelta={}
    for j,c in enumerate(cols):
        expected=raw[:,:,list(fields).index(c)].reshape(-1);mask=np.isfinite(expected)
        check('Raw finite mask exactly preserved '+c,np.array_equal(np.isfinite(means[:,j]),mask))
        delta=float(np.max(abs(means[mask,j]-expected[mask])));maxdelta[c]=delta
        check('Independent minute mean including correct temperature units '+c,np.allclose(means[:,j],expected,equal_nan=True,atol=1e-8,rtol=1e-12))
        check('Independent actual minute coverage counts '+c,np.array_equal(counts[:,j],count_table[c+'_raw_finite_records'].to_numpy()))
    source=quality['source_audit'][kind]
    check('Out of range and epoch removal '+kind,outside==source['outside_date_range_removed'])
    check('Fixed abnormal range masks '+kind,masked==source['masked_invalid_rows'])
    check('All raw rows streamed '+kind,rows==source['original_rows'])
    details[kind]=dict(rows=rows,outside_removed=outside,invalid_masked=masked,max_absolute_mean_difference=maxdelta)
result=dict(all_passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),count=len(checks),checks=checks,details=details,processed_sha256=sha(F/'data/processed.npz'),coverage_csv_sha256=sha(F/'data/raw_interval_coverage.csv'),method='Separate streaming NumPy sums and counts from original Parquet, not pandas producer resample')
(F/'results/raw_stream_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(all_passed=True,checks=len(checks),details=details),ensure_ascii=False))
