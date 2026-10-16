"""Current recommended balanced R26 historical deployment, with frozen source chain."""
from pathlib import Path
import sys,ast,json,argparse,hashlib
import numpy as np
import pandas as pd
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round26';Q=R/'results';O=Q/'exports'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='read');ap.add_argument('--days',type=int,choices=[1,7],default=7);a=ap.parse_args()
if a.stage=='read':
 v=read(O/'verification.json');assert v['all_passed'] and sha(v[f'file{a.days}'])==v[f'sha{a.days}'];print(json.dumps(dict(recommended_method=v['method7'] if a.days==7 else v['method1'],curve_file=v[f'file{a.days}'],daily_file=v[f'daily_file{a.days}'],historical_replay=True,not_current_day=True,independent_confirmation=False),ensure_ascii=False));sys.exit(0)
assert a.stage=='export';sys.path.insert(0,str(B/'code'));import common;d=common.load();s=read(Q/'selection.json');s25=read(B/'round25/results/selection.json');assert s['horizons']['7']['chosen_alpha']==.25
p=B/'round25/code/level_shape_experiment.py';env={'__file__':str(p),'__name__':'pure_level_shape'};tr=ast.parse(p.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tr.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets));exec(compile(ast.Module(body=tr.body[:cut],type_ignores=[]),str(p),'exec'),env)
with np.load(B/'round24/results/exports/deployment7_origin31.npz') as z:base24=z['selected'].copy();base19=z['prior'].copy()
cp25=env['candidate'](d,base24,base19,s25['horizons']['7']['candidate']);p25=(1-s25['horizons']['7']['chosen_alpha'])*base24+s25['horizons']['7']['chosen_alpha']*cp25;pred=.75*base19+.25*p25;assert pred.shape==(57,1,672) and np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
assert np.array_equal(pred[d['cells']=='a'],base19[d['cells']=='a']);assert all(np.max(abs(pred[d['cells']==c].sum((0,2))-base19[d['cells']==c].sum((0,2))))<1e-8 for c in np.unique(d['cells']))
O.mkdir(exist_ok=True);np.savez_compressed(O/'deployment_origin31.npz',prior7=base19,r24=base24,r25=p25,selected7=pred)
with np.load(R/'models/interval7_radius.npz') as z:rad=z['radius']
stamps=pd.date_range('2019-06-01',periods=672,freq='15min',tz='America/Los_Angeles');rows=[]
for i,name in enumerate(d['names']):
 for j,t in enumerate(stamps):rows.append(dict(pdu=str(name),cell=str(d['cells'][i]),timestamp_PT=t.isoformat(),timestamp_UTC=t.tz_convert('UTC').isoformat(),forecast_normalized_power=pred[i,0,j],lower80_empirical=max(0,pred[i,0,j]-rad[i]),upper80_empirical=min(1,pred[i,0,j]+rad[i]),horizon_days=7,historical_replay=True))
df7=pd.DataFrame(rows);df1=pd.read_csv(B/'round24/results/exports/selected_1day_curve.csv');info=dict(all_passed=True,rows1=5472,rows7=38304,daily_rows7=399,pdus=57,historical_replay=True,no_june_truth=True,not_current_day=True,selection_sha256=sha(Q/'selection.json'),a5_unchanged=True,normalized_cell_total_preserved=True,method7='TimesFM3 native672 plus Ridge1000 Fourier residual and total constraint, plus fixed25% joint HGB point and sparse Ridge peak correction with daily-level gain1.5; a5 unchanged and per-cell normalized totals preserved',method1='Chronos-2 TiRex-2 TimesFM3 fixed weights plus Ridge residual and TimesFM3 final-layer finetuning',small_tradeoffs_accepted=True,independent_confirmation=False)
for H,df in [(1,df1),(7,df7)]:
 assert len(df)==57*H*96 and df['pdu'].nunique()==57;point=O/f'selected_{H}day_curve.csv';df.to_csv(point,index=False,encoding='utf-8-sig');df['date_PT']=df['timestamp_PT'].str[:10];dm=df.groupby(['pdu','cell','date_PT'],sort=False)['forecast_normalized_power'].mean().reset_index(name='daily_mean_normalized_power');daily=O/f'selected_{H}day_daily_mean.csv';dm.to_csv(daily,index=False,encoding='utf-8-sig');info[f'file{H}']=str(point);info[f'sha{H}']=sha(point);info[f'daily_file{H}']=str(daily);info[f'daily_sha{H}']=sha(daily)
info['source_hashes']={f:sha(T/f) for f in ['experiment/round26/results/selection.json','experiment/round25/results/selection.json','experiment/round25/code/level_shape_experiment.py','experiment/round24/results/exports/deployment7_origin31.npz','experiment/round24/results/exports/selected_1day_curve.csv','experiment/results/data.npz']}
(O/'verification.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(info,ensure_ascii=False))

