"""Verified June1 historical inference for the frozen R24 day/peak alternative."""
from pathlib import Path
import sys,ast,json,argparse,pickle,hashlib
import numpy as np
import pandas as pd
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round24';Q=R/'results';O=Q/'exports'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utilities(n,name):
 p=B/f'round{n}/code'/name;env={'__file__':str(p),'__name__':'pure_deployment_utilities'};tr=ast.parse(p.read_text(encoding='utf-8'));cut=next(i for i,a in enumerate(tr.body) if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in a.targets));exec(compile(ast.Module(body=tr.body[:cut],type_ignores=[]),str(p),'exec'),env);return env
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='read');ap.add_argument('--days',type=int,default=7,choices=[1,7]);args=ap.parse_args()
if args.stage=='read':
 info=read(O/'verification.json');assert info['all_passed'];print(json.dumps(dict(file=info[f'file{args.days}'],historical_replay=True,warning='R24 day/peak research alternative; slightly worse curve and threshold fractions than R19; not current-day real load.'),ensure_ascii=False));sys.exit(0)
assert args.stage=='export' and read(Q/'independent_verification.json')['all_passed']
for n in [22,23,24]:assert read(B/f'round{n}/results/independent_verification.json')['all_passed']
u22=utilities(22,'point_residual_experiment.py');u23=utilities(23,'joint_point_peak_experiment.py');u24=utilities(24,'volatility_gate_experiment.py');d=u24['common'].load();s24=read(Q/'selection.json');s23=read(B/'round23/results/selection.json');s22=read(B/'round22/results/selection.json');assert s24['horizons']['1']['chosen_alpha']==0
base=u24['z']('round19/results/deployment7_origin31.npz','selected');native=u24['z']('round18/results/deployment7_origin31.npz','prior');models=pickle.load((B/'round22/models/final_heads.pkl').open('rb'));c22=s22['horizons']['7']['candidate'];point=u22['candidate'](base,u22['residual'](d,[31],native,models[c22['model']],7),c22,7);det=u23['e']['details'](d,[31],native,u23['e']['models'](True),7);proposal=u23['candidate'](base,point,det,s23['horizons']['7']['candidate'],7);raw=u24['candidate'](d,[31],base,proposal,s24['horizons']['7']['candidate']);alpha=s24['horizons']['7']['chosen_alpha'];p=(1-alpha)*base+alpha*raw
assert p.shape==(57,1,672) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1 and np.array_equal(p[d['cells']=='a'],base[d['cells']=='a'])
O.mkdir(exist_ok=True);np.savez_compressed(O/'deployment7_origin31.npz',prior=base,point=point,proposal=proposal,raw=raw,selected=p);rad=u24['z']('round24/models/interval7_radius.npz','radius')
old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv');old=old.rename(columns={'series':'pdu','predicted_util':'forecast_normalized_power','lower80_empirical':'lower80_empirical','upper80_empirical':'upper80_empirical'})
old['timestamp_PT']=[pd.Timestamp(x).tz_localize('America/Los_Angeles').isoformat() for x in old['local_trace_time_PT']];old['timestamp_UTC']=[pd.Timestamp(x).tz_convert('UTC').isoformat() for x in old['timestamp_PT']];old['horizon_days']=1;old['historical_replay']=True;old=old[['pdu','cell','timestamp_PT','timestamp_UTC','forecast_normalized_power','lower80_empirical','upper80_empirical','horizon_days','historical_replay']]
times=pd.date_range('2019-06-01',periods=672,freq='15min',tz='America/Los_Angeles');rows=[]
for i,name in enumerate(d['names']):
 for j,t in enumerate(times):rows.append(dict(pdu=str(name),cell=str(d['cells'][i]),timestamp_PT=t.isoformat(),timestamp_UTC=t.tz_convert('UTC').isoformat(),forecast_normalized_power=p[i,0,j],lower80_empirical=max(0,p[i,0,j]-rad[i]),upper80_empirical=min(1,p[i,0,j]+rad[i]),horizon_days=7,historical_replay=True))
df7=pd.DataFrame(rows);info=dict(all_passed=True,rows1=5472,rows7=38304,daily_rows7=399,pdus=57,historical_replay=True,no_june_truth=True,not_current_day=True,trace_timezone='America/Los_Angeles',selection_sha256=sha(Q/'selection.json'),a5_same_as_prior=True,per_cell_forecast_total_preserved=True,research_tradeoff=True)
for H,df in [(1,old),(7,df7)]:
 assert len(df)==57*H*96 and df['pdu'].nunique()==57;path=O/f'selected_{H}day_curve.csv';df.to_csv(path,index=False,encoding='utf-8-sig');df['date_PT']=df['timestamp_PT'].str[:10];dm=df.groupby(['pdu','cell','date_PT'],sort=False)['forecast_normalized_power'].mean().reset_index(name='daily_mean_normalized_power');dm.to_csv(O/f'selected_{H}day_daily_mean.csv',index=False,encoding='utf-8-sig');info[f'file{H}']=str(path);info[f'sha{H}']=sha(path)
info['source_hashes']={f:sha(T/f) for f in ['experiment/round24/results/selection.json','experiment/round23/results/selection.json','experiment/round22/results/selection.json','experiment/round22/models/final_heads.pkl','experiment/round18/models/final_heads.pkl','experiment/round19/results/deployment7_origin31.npz','experiment/round18/results/deployment7_origin31.npz','experiment/results/data.npz']}
(O/'verification.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(info,ensure_ascii=False))

