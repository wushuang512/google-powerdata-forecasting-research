from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
import pandas as pd
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round19';Q=R/'results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=read(T/'交付文件清单.json');qa=read(R/'docs/report_qa/verification.json');s=read(Q/'selection.json');result=read(Q/'summary.json');checks={}
checks['round18_and19_finished']=all(read(B/f'round{n}/results/resource_status.json')['stage']=='completed' and (B/f'round{n}/results/pipeline_complete.json').exists() for n in [18,19])
checks['model_implementation_verified']=all(read(B/f'round{n}/results/independent_verification.json')['all_passed'] for n in [18,19])
checks['correct_frozen_research7']=s['candidate']['id']=='global_peak0' and s['chosen_alpha']==.25 and m['selected_research_model7']=='round19_global_peak0_alpha025' and m['selected_model24']=='round8b'
checks['truthful_multiple_metric_claim']=m['multiple_metric_goal_met_descriptive'] and len(m['sevenday_leading_metrics'])==7 and m['sevenday_leading_metrics']==result['leading_metrics7'] and not m['all_metric_goal_met'] and not m['independent_confirmation']
checks['all19_pages_current_reviewed']=qa['all_passed'] and qa['all_pages_visually_reviewed'] and qa['pages']==19 and qa['page_numbers_reviewed']==list(range(1,20)) and qa['document_sha256']==sha(T/'Google PowerData 2019 三项预测任务与模型比较汇报.docx')==m['report_sha256']
checks['all_page_and_pdf_hashes']=all(sha(x['file'])==x['sha256'] for x in qa['images']) and sha(qa['pdf'])==qa['pdf_sha256']
bundle=read(Q/'deployment_bundle.json');checks['deployment_bound_sources']=all(sha(T/p)==h for p,h in bundle['source_hashes'].items())
checks['actual_point_counts_bounds_intervals_times_and_values']=True;checks['actual_daily_means_from_same_curve']=True
with np.load(Q/'deployment7_origin31.npz') as z:pred7=z['selected'].copy()
sys.path.insert(0,str(B/'code'));import common
d=common.load()
for h in [1,7]:
 ex=read(Q/f'exports/export_{h}day_verification.json');df=pd.read_csv(ex['point_file']);dm=pd.read_csv(ex['daily_file']);ts=pd.to_datetime(df['timestamp_UTC'],utc=True)
 ok=len(df)==57*h*96 and df['pdu'].nunique()==57 and not df.duplicated(['pdu','timestamp_PT']).any() and df.groupby('pdu').size().eq(h*96).all() and df['forecast_normalized_power'].between(0,1).all() and np.isfinite(df[['forecast_normalized_power','lower80_empirical','upper80_empirical']]).all().all() and (df['lower80_empirical']<=df['forecast_normalized_power']).all() and (df['upper80_empirical']>=df['forecast_normalized_power']).all() and df['lower80_empirical'].between(0,1).all() and df['upper80_empirical'].between(0,1).all()
 expected=pd.date_range('2019-06-01T07:00:00Z',periods=h*96,freq='15min');ok &= pd.DatetimeIndex(sorted(ts.unique())).equals(expected) and all(pd.Timestamp(pt).tz_convert('UTC')==pd.Timestamp(ut) for pt,ut in zip(df['timestamp_PT'],df['timestamp_UTC'])) and ex['historical_replay'] and ex['not_current_day_load_forecast'] and sha(ex['point_file'])==ex['point_sha256'] and sha(ex['daily_file'])==ex['daily_sha256']
 if h==7:
  ok &= all(np.allclose(df.loc[df['pdu']==str(name),'forecast_normalized_power'].to_numpy(),pred7[i,0],rtol=0,atol=1e-15) for i,name in enumerate(d['names']))
 else:
  old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv');ok &= np.allclose(df['forecast_normalized_power'],old['predicted_util'],rtol=0,atol=1e-15)
 checks['actual_point_counts_bounds_intervals_times_and_values'] &= bool(ok)
 df['date_PT']=df['timestamp_PT'].str[:10];got=df.groupby(['pdu','cell','date_PT'],sort=False)['forecast_normalized_power'].mean().reset_index(name='daily_mean_normalized_power')
 checks['actual_daily_means_from_same_curve'] &= len(dm)==57*h and got[['pdu','cell','date_PT']].equals(dm[['pdu','cell','date_PT']]) and np.allclose(got['daily_mean_normalized_power'],dm['daily_mean_normalized_power'],rtol=0,atol=1e-15)
csv1=list(csv.DictReader((T/'原生七天调优多维度对照.csv').open(encoding='utf-8-sig')));csv2=list(csv.DictReader((T/'原生七天调优每日平均负荷比较.csv').open(encoding='utf-8-sig')))
checks['complete22_named_methods_tables']=len(csv1)==len(csv2)==22 and all(x['model_name'].strip() for x in csv1+csv2) and all(abs(float(csv1[-1][k])-value)<1e-12 for k,value in result['scores7']['selected'].items())
daily=read(Q/'daily_mean_evaluation.json');checks['daily_three_lowest_but_not_selection']=daily['supplementary_not_used_for_selection'] and len(daily['new_leading_daily_metrics'])==3 and all(abs(float(csv2[-1][k])-daily['rows'][-1][k])<1e-12 for k in ['daily_wape','daily_mae','daily_rmse'])
checks['new_entries_and_readable_record_present']=all((T/f'17_{h}天按跨度选择模型历史复现.cmd').exists() for h in [1,7]) and (T/'第19轮多项最优研究记录.md').exists() and (B/'round18/results/deployment_time_erratum.json').exists()
checks['inventory_entries_unique']=len(m['all_files'])==len({x['path'] for x in m['all_files']})==m['files']
checks['inventory_all_files_exist_and_sizes']=all((T/x['path']).is_file() and (T/x['path']).stat().st_size==x['bytes'] for x in m['all_files'])
checks['inventory_total_bytes']=sum(x['bytes'] for x in m['all_files'])==m['bytes']
actual={str(p.relative_to(T)) for p in T.rglob('*') if p.is_file() and p!=T/'交付文件清单.json'}
checks['inventory_no_missing_new_files']=actual=={x['path'] for x in m['all_files']}
checks['physical_root_d']=str(T).startswith('D:')
assert all(checks.values()),checks
out=dict(all_passed=True,checks={k:bool(v) for k,v in checks.items()},files=m['files'],bytes=m['bytes'],word_pages=19,research7_alpha=.25,seven_best=True,all_metric_goal_met=False,independent_confirmation=False)
if '--record-in-manifest' in sys.argv:
 m['inventory_verified']=True;m['final_delivery_verification']=out;(T/'交付文件清单.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))

