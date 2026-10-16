from pathlib import Path
import sys,json,hashlib,csv,datetime,tomllib,subprocess,os
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pypdf import PdfReader
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round26';Q=R/'results';O=Q/'exports'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=read(T/'交付文件清单.json');qa=read(R/'docs/report_qa/verification.json');s=read(Q/'selection.json');pr=read(R/'docs/protocol.json');result=read(Q/'summary.json');cmp=read(Q/'recommendation_comparison.json');ex=read(O/'verification.json');checks={}
checks['all_seven_real_rounds_finished']=all(read(B/f'round{n}/results/resource_status.json')['stage']=='completed' and (B/f'round{n}/results/pipeline_complete.json').exists() for n in range(20,27))
checks['all_scientific_implementation_audits_passed']=all(read(B/f'round{n}/results/independent_verification.json')['all_passed'] for n in range(20,27))
checks['frozen_choices_preserved']=all(sha(B/f'round{n}/results/selection.json')==(B/f'round{n}/results/selection.sha256').read_text().strip() for n in range(20,27))
checks['current_bound_protocol_and_inputs']=sha(R/'docs/protocol.json')==s['protocol_sha256'] and sha(R/'code/quarter_joint_experiment.py')==pr['script_sha256'] and sha(R/'code/source_metrics.py')==pr['metrics_script_sha256'] and all(sha(B/f)==h for f,h in pr['source_hashes'].items())
checks['current_frozen_development_and_selected_weights']=s['horizons']['7']['chosen_alpha']==pr['fixed_strength']==.25 and s['horizons']['7']['development_supported'] and s['horizons']['1']['chosen_alpha']==0 and sha(Q/'dev_summary.json')==s['dev_summary_sha256'] and sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256'] and sha(R/'models/interval7_radius.npz')==s['interval_radius_sha256'] and result['selection']==s
checks['user_preference_prospectively_registered']=pr['created_utc']<s['frozen_utc'] and 'human explicitly accepts' in pr['user_preference'] and cmp['user_accepts_minor_tradeoffs'] and m['small_tradeoffs_accepted']
checks['correct_recommended_names_and_scope']=m['selected_model24']=='round8b' and m['selected_research_model7']=='round26_fixed_quarter_joint_alpha025' and not m['all_metric_goal_met'] and not m['independent_confirmation'] and not cmp['independent_confirmation']
fields=cmp['fields'];better=[];same=[];worse=[];before=result['scores']['7']['prior'];after=result['scores']['7']['selected']
for k in fields:
 delta=after[k]-before[k] if k.startswith('hit') else before[k]-after[k]
 (same if abs(delta)<1e-12 else better if delta>0 else worse).append(k)
checks['exact_eight_one_five_tradeoffs']=len(fields)==14 and (better,same,worse)==(cmp['improved'],cmp['equal'],cmp['minor_regressions']) and (len(better),len(same),len(worse))==(8,1,5) and m['recommendation_comparison']==cmp
t=pr['tolerances'];c=after;b=before
checks['test_tradeoffs_within_registered_limits']=all(c[k]<=b[k]+t[key]+1e-12 for k,key in [('wape','wape_pp'),('daily_wape','daily_wape_pp'),('peak_mape','peak_mape_pp'),('p95_ape','p95_ape_pp'),('cella5_wape','group_wape_pp'),('other52_wape','group_wape_pp'),('absolute_bias_percent','absolute_bias_pp'),('peak_time_mae_minutes','peak_time_minutes')]) and all(c[k]<=b[k]*(1+t[key])+1e-12 for k,key in [('mae','mae_relative'),('rmse','rmse_relative'),('daily_mae','daily_mae_relative'),('daily_rmse','daily_rmse_relative')]) and all(c[k]>=b[k]-t[key]-1e-12 for k,key in [('hit2_percent','hit2_percent_pp'),('hit5_percent','hit5_percent_pp')])
checks['all26_pages_current_actually_reviewed']=qa['all_passed'] and qa['all_pages_visually_reviewed'] and qa['layout_passed'] and qa['pages']==26==len(PdfReader(qa['pdf']).pages)==m['word_pages'] and qa['page_numbers_reviewed']==list(range(1,27)) and qa['document_sha256']==sha(T/'Google PowerData 2019 三项预测任务与模型比较汇报.docx')==m['report_sha256']
checks['all_render_image_pdf_hashes']=len(qa['images'])==26 and all(sha(x['file'])==x['sha256'] for x in qa['images']) and sha(qa['pdf'])==qa['pdf_sha256']
checks['all_original_metric_rows_source_and_csvs_bound']=False
orig=read(B/'round20/results/original_metrics_complete.json');ov=read(B/'round20/results/original_metrics_final_verification.json')
rows=list(csv.DictReader((T/'9月28日指标完整模型对照.csv').open(encoding='utf-8-sig')));intervals=list(csv.DictReader((T/'9月28日指标预测区间对照.csv').open(encoding='utf-8-sig')))
checks['all_original_metric_rows_source_and_csvs_bound']=ov['all_passed'] and sha(B/'round20/results/original_metrics_complete.json')==ov['summary_sha256'] and sha(T/'9月28日指标完整模型对照.csv')==ov['csv_sha256'] and sha(T/'9月28日指标预测区间对照.csv')==ov['interval_csv_sha256'] and len(rows)==len(orig['rows'])==58 and len(intervals)==len(orig['intervals'])==6 and orig['all_original_metric_categories_covered'] and len(orig['reference_inventory']['metrics'])==11 and sha(T/'参考材料/9.28算力中心需求预测实验.docx')==orig['reference_inventory']['reference_sha256'] and all(x['model_name'].strip() for x in rows)
checks['correct_original_baselines_and_current_rows']=all(next(x for x in orig['rows'] if x['model']=='current_selected' and x['horizon_days']==h)['source_baseline']==('前一天同刻' if h==1 else '前三天同刻均值') for h in [1,7])
for h in [1,7]:
 x=next(x for x in orig['rows'] if x['model']=='current_selected' and x['horizon_days']==h)
 checks['correct_original_baselines_and_current_rows'] &= all(abs(x[k]-result['scores'][str(h)]['selected'][k])<1e-12 for k in fields)
checks['tirex_historical_coverage_not_fabricated']=any(x['kind'].startswith('历史引用') and x['coverage_percent']==56.6 and x['mean_width_normalized'] is None and x['winkler80'] is None for x in orig['intervals'])
sys.path.insert(0,str(B/'code'));import common;d=common.load()
with np.load(O/'deployment_origin31.npz') as z:pred=z['selected7'].copy();base=z['prior7'].copy();r25=z['r25'].copy()
checks['deployment_exact_frozen_quarter_a5_and_totals']=np.array_equal(pred,.75*base+.25*r25) and np.array_equal(pred[d['cells']=='a'],base[d['cells']=='a']) and all(np.max(abs(pred[d['cells']==cell].sum((0,2))-base[d['cells']==cell].sum((0,2))))<1e-8 for cell in np.unique(d['cells']))
with np.load(B/'round24/results/exports/deployment7_origin31.npz') as z:r24=z['selected'].copy();base19=z['prior'].copy()
cfg25=read(B/'round25/results/selection.json')['horizons']['7']['candidate'];assert cfg25['mean_gain']==1.5 and cfg25['shape_gain']==1 and cfg25['smooth']==1
change=(r24-base19).reshape(57,1,7,96).mean(-1,keepdims=True)*.5;raw=np.clip(r24+np.broadcast_to(change,(57,1,7,96)).reshape(57,1,672),0,1);got25=raw.copy()
for cell in np.unique(d['cells']):
 ix=d['cells']==cell;bb=r24[ix,0];qq=raw[ix,0]
 if np.array_equal(bb,qq):continue
 lo=-2.;hi=2.
 for _ in range(55):
  mid=(lo+hi)/2
  if np.clip(qq+mid,0,1).sum()>bb.sum():hi=mid
  else:lo=mid
 got25[ix,0]=np.clip(qq+(lo+hi)/2,0,1)
checks['deployment_joint_level_candidate_independently_reproduced']=np.allclose(got25,r25,rtol=0,atol=1e-15) and np.array_equal(base,base19)
checks['deployment_inputs_and_exports_bound']=ex['all_passed'] and all(sha(T/p)==h for p,h in ex['source_hashes'].items()) and ex['selection_sha256']==sha(Q/'selection.json')
checks['actual_rows_bounds_intervals_time_and_source_values']=True;checks['actual_daily_means_same_curve']=True
for h in [1,7]:
 df=pd.read_csv(ex[f'file{h}']);dm=pd.read_csv(ex[f'daily_file{h}']);ts=pd.to_datetime(df['timestamp_UTC'],utc=True)
 expected=pd.date_range('2019-06-01T07:00:00Z',periods=h*96,freq='15min');val=df[['forecast_normalized_power','lower80_empirical','upper80_empirical']]
 ok=len(df)==57*h*96 and df['pdu'].nunique()==57 and not df.duplicated(['pdu','timestamp_PT']).any() and df.groupby('pdu').size().eq(h*96).all() and np.isfinite(val).all().all() and val.ge(0).all().all() and val.le(1).all().all() and (df['lower80_empirical']<=df['forecast_normalized_power']).all() and (df['forecast_normalized_power']<=df['upper80_empirical']).all()
 ok &= pd.DatetimeIndex(sorted(ts.unique())).equals(expected) and all(pd.Timestamp(pt).tz_convert('UTC')==pd.Timestamp(ut) for pt,ut in zip(df['timestamp_PT'],df['timestamp_UTC'])) and ex['historical_replay'] and ex['no_june_truth'] and ex['not_current_day'] and sha(ex[f'file{h}'])==ex[f'sha{h}'] and sha(ex[f'daily_file{h}'])==ex[f'daily_sha{h}']
 if h==7:ok &= all(np.allclose(df.loc[df['pdu']==str(name),'forecast_normalized_power'].to_numpy(),pred[i,0],rtol=0,atol=1e-15) for i,name in enumerate(d['names']))
 else:
  old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv');ok &= np.allclose(df['forecast_normalized_power'],old['predicted_util'],rtol=0,atol=1e-15)
 checks['actual_rows_bounds_intervals_time_and_source_values'] &= bool(ok)
 df['date_PT']=df['timestamp_PT'].str[:10];got=df.groupby(['pdu','cell','date_PT'],sort=False)['forecast_normalized_power'].mean().reset_index(name='daily_mean_normalized_power')
 checks['actual_daily_means_same_curve'] &= len(dm)==57*h and got[['pdu','cell','date_PT']].equals(dm[['pdu','cell','date_PT']]) and np.allclose(got['daily_mean_normalized_power'],dm['daily_mean_normalized_power'],rtol=0,atol=1e-15)
checks['new19_entries_and_old_meanings_preserved']=all((T/f'19_{h}天当前推荐模型历史复现.cmd').exists() for h in [1,7]) and all((T/f'17_{h}天按跨度选择模型历史复现.cmd').exists() for h in [1,7]) and (T/'18_7天日均峰值研究替代历史复现.cmd').exists() and (T/'09_现有最佳微调_预测次日.cmd').exists() and (T/'02_TiRex_预测七天.cmd').exists() and (T/'第20至26轮全指标优化与采用记录.md').exists()
entry_env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8');entry_ok=True
for h in [1,7]:
 path=T/f'19_{h}天当前推荐模型历史复现.cmd';got=subprocess.run(['cmd.exe','/d','/c',str(path)],capture_output=True,text=True,encoding='utf-8',env=entry_env,check=True);info=json.loads(got.stdout)
 entry_ok &= info['curve_file']==ex[f'file{h}'] and info['daily_file']==ex[f'daily_file{h}'] and info['historical_replay'] and info['not_current_day'] and not info['independent_confirmation']
path=T/'18_7天日均峰值研究替代历史复现.cmd';got=subprocess.run(['cmd.exe','/d','/c',str(path)],capture_output=True,text=True,encoding='utf-8',env=entry_env,check=True);info=json.loads(got.stdout);entry_ok &= info['file']==str(B/'round24/results/exports/selected_7day_curve.csv') and info['historical_replay']
checks['three_actual_entry_readback_paths_passed']=bool(entry_ok)
audit=read(T/'google-clusterdata-2019/results/complete_projection_audit.json');checks['official_cella_complete_no_download_remaining']=audit['all_passed'] and audit['source_files']==12085 and m['actual_download_remaining']==0
auto=tomllib.loads((Path('${USER_HOME}/.codex/automations/google/automation.toml')).read_text(encoding='utf-8'));checks['continuation_still_paused']=auto['status']=='PAUSED'
owned_filter="$p=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'GooglePowerData2019' -and $_.CommandLine -match '(run_.*guarded|extract_cpu|covariate_experiment|complete_usage_experiment|point_residual_experiment|learned_peak_experiment|quarter_joint_experiment|volatility_gate_experiment|level_shape_experiment|joint_point_peak_experiment)' }; if($p){$p | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress}else{'[]'}"
proc=subprocess.run(['powershell.exe','-NoProfile','-Command',owned_filter],capture_output=True,text=True,check=True);active=json.loads(proc.stdout);checks['no_active_owned_model_or_download_processes']=active==[]
checks['inventory_unique']=len(m['all_files'])==len({x['path'] for x in m['all_files']})==m['files']
checks['inventory_exists_sizes_and_total']=all((T/x['path']).is_file() and (T/x['path']).stat().st_size==x['bytes'] for x in m['all_files']) and sum(x['bytes'] for x in m['all_files'])==m['bytes']
actual={str(p.relative_to(T)) for p in T.rglob('*') if p.is_file() and p!=T/'交付文件清单.json'};checks['inventory_no_omissions']=actual=={x['path'] for x in m['all_files']}
checks['physical_root_d']=str(T).startswith('D:')
out=dict(all_passed=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=m['files'],bytes=m['bytes'],word_pages=26,current_research7='round26',selected_strength=.25,improved=8,equal=1,minor_regressions=5,all_metric_goal_met=False,independent_confirmation=False,download_remaining=0,automation_status=auto['status'])
assert out['all_passed'],json.dumps(out,ensure_ascii=False)
if '--record-in-manifest' in sys.argv:
 m['inventory_verified']=True;m['final_delivery_verification']=out;m['final_delivery_all_passed']=True;m['final_delivery_check_count']=len(checks);m['independent_final_check_utc']=out['checked_utc'];(T/'交付文件清单.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
