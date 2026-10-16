"""Re-export verified May31 historical inference, with explicit PT and UTC dates."""
from pathlib import Path
import sys,argparse,json,hashlib
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--days',type=int,choices=[1,7],default=7);a=ap.parse_args();T=Path(a.root).resolve();B=T/'experiment';R=B/'round19';Q=R/'results';O=Q/'exports';O.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(Q/'independent_verification.json')['all_passed'];s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();manifest=read(Q/'deployment_bundle.json');assert all(sha(T/p)==h for p,h in manifest['source_hashes'].items())
stamps=pd.date_range('2019-06-01',periods=a.days*96,freq='15min',tz='America/Los_Angeles');rows=[];daily=[]
if a.days==1:
 df=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv');assert len(df)==5472
 for r in df.to_dict('records'):
  t=pd.Timestamp(r['local_trace_time_PT']).tz_localize('America/Los_Angeles');rows.append(dict(pdu=r['series'],cell=r['cell'],timestamp_PT=t.isoformat(),timestamp_UTC=t.tz_convert('UTC').isoformat(),forecast_normalized_power=r['predicted_util'],lower80_empirical=r['lower80_empirical'],upper80_empirical=r['upper80_empirical'],horizon_days=1,method='Chronos-2 plus TiRex-2 plus TimesFM3 fixed weights with Ridge residual calibration and TimesFM3 final-layer finetuning',historical_replay=True,independent_confirmation=False))
else:
 sys.path.insert(0,str(B/'code'));import common
 d=common.load()
 with np.load(Q/'deployment7_origin31.npz') as z:p=z['selected'].copy()
 with np.load(Q/'dev_predictions.npz') as z:raw=z[s['candidate']['id']].copy()
 with np.load(B/'round18/results/dev_predictions.npz') as z:base=z['prior'].copy()
 dev=(1-s['chosen_alpha'])*base+s['chosen_alpha']*raw;y,m=common.truth(d,[14,15],7);rad=np.array([np.quantile(abs(dev[i]-y[i])[m[i]],.8) for i in range(57)])
 for i in range(57):
  for j,t in enumerate(stamps):
   value=float(p[i,0,j]);rows.append(dict(pdu=str(d['names'][i]),cell=str(d['cells'][i]),timestamp_PT=t.isoformat(),timestamp_UTC=t.tz_convert('UTC').isoformat(),forecast_normalized_power=value,lower80_empirical=max(0.,value-rad[i]),upper80_empirical=min(1.,value+rad[i]),horizon_days=7,method='TimesFM3 native672 plus zero-intercept Ridge1000 Fourier3 residual plus global forecast-total projection alpha0.25',historical_replay=True,independent_confirmation=False))
df=pd.DataFrame(rows);assert len(df)==57*a.days*96 and df['pdu'].nunique()==57;assert np.isfinite(df[['forecast_normalized_power','lower80_empirical','upper80_empirical']]).all().all();assert (df['lower80_empirical']<=df['forecast_normalized_power']).all() and (df['upper80_empirical']>=df['forecast_normalized_power']).all();assert df['forecast_normalized_power'].between(0,1).all()
point=O/f'selected_{a.days}day_curve.csv';df.to_csv(point,index=False,encoding='utf-8-sig');df['date_PT']=df['timestamp_PT'].str[:10];dm=df.groupby(['pdu','cell','date_PT'],sort=False)['forecast_normalized_power'].mean().reset_index(name='daily_mean_normalized_power');assert len(dm)==57*a.days;day=O/f'selected_{a.days}day_daily_mean.csv';dm.to_csv(day,index=False,encoding='utf-8-sig')
meta=dict(days=a.days,rows=len(df),daily_rows=len(dm),pdus=57,trace_timezone='America/Los_Angeles',first_PT=stamps[0].isoformat(),last_PT=stamps[-1].isoformat(),first_UTC=stamps[0].tz_convert('UTC').isoformat(),last_UTC=stamps[-1].tz_convert('UTC').isoformat(),historical_replay=True,history_ends='2019-05-31 23:45 PT',no_june_truth=True,not_current_day_load_forecast=True,empirical_intervals='7day: absolute-error80th percentile pooled per PDU over two overlapping DEV windows also used for selection; exploratory, no guaranteed coverage. 1day: existing round8b empirical intervals.',selection_sha256=sha(Q/'selection.json'),point_file=str(point),point_sha256=sha(point),daily_file=str(day),daily_sha256=sha(day))
(O/f'export_{a.days}day_verification.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False))
