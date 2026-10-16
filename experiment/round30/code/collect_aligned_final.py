from pathlib import Path
import sys,json,hashlib,datetime,csv,numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round30';Q=R/'results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=read(B/'round29/results/new_methods_verification.json');assert v['all_passed'];old=read(B/'round29/results/aligned_old_results.json');rows=old+v['new_rows'];labels={}
for rec in rows:
 if rec['model'].startswith('r29:TTM'):
  c=512 if '512' in rec['model'] else 1024;rec['model_name']=f'IBM TinyTimeMixer R2 零样本 历史{c}点 原生720点裁剪672点';rec['report_name']=f'IBM TinyTimeMixer R2\n{c}点历史 零样本直接672点'
 elif rec['model'].startswith('r29:Joint'):
  rec['model_name']='57PDU联合因子与傅里叶直接预测 '+rec['model'].split(':')[1];rec['report_name']=rec['model_name'].replace('JointFactorFourier ','\n')
 elif rec['model'].startswith('r30:TTM'):
  full=rec['model'].endswith('all');rec['model_name']='IBM TinyTimeMixer R2 512点历史 多任务曲线日均峰值 '+('全参数训练4轮 学习率0.00001' if full else'解码器与输出头训练4轮 学习率0.0001');rec['report_name']='IBM TinyTimeMixer R2 多任务训练\n'+('全参数 学习率10⁻⁵' if full else'解码器和输出头 学习率10⁻⁴')
def csvsave(path,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r if k not in ['source','aliases','prediction_digest']));f=Path(path).open('w',encoding='utf-8-sig',newline='');w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rr);f.close()
unique=[]
for H in [1,7]:
 ds={}
 for rec in sorted([r for r in rows if r['horizon_days']==H],key=lambda r:0 if r['status']=='当前综合推荐' else 1):
  digest=rec['prediction_digest']
  if digest not in ds:ds[digest]=dict(rec,aliases=[rec['model']])
  else:
   ds[digest]['aliases'].append(rec['model'])
   if ds[digest].get('development_wape_original_grid') is None and rec.get('development_wape_original_grid') is not None:ds[digest]['development_wape_original_grid']=rec['development_wape_original_grid']
 unique+=sorted(ds.values(),key=lambda r:r['wape'])
winners=[]
for H,task,fields in [(1,'24小时曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天每日均值',['daily_wape','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent'])]:
 rr=[r for r in unique if r['horizon_days']==H]
 for k in fields:
  higher=k.startswith('hit') or k.startswith('daily_hit');val=(max if higher else min)(r[k] for r in rr);ww=[r for r in rr if abs(r[k]-val)<1e-10];winners.append(dict(task=task,metric=k,value=val,models='；'.join(r['model_name'] for r in ww),keys=[r['model'] for r in ww],direction='越高越好' if higher else'越低越好'))
csvsave(T/'模型各项指标完整结果.csv',rows);csvsave(T/'9月28日统一口径模型复算.csv',rows);csvsave(T/'各任务单项最优结果.csv',winners)
old_iv=read(B/'round28/results/clear_comparison.json')['intervals'];ivs=old_iv+v['new_intervals'];csvsave(T/'模型预测区间结果.csv',ivs)
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metric_version='9.28 aligned v2 complete-day daily mask',reference_audit=read(B/'round29/results/metric_audit/audit.json'),rows=rows,unique_rows=unique,winners=winners,intervals=ivs,recommendation=dict(recommended24='round8b',recommended7='round26',new_universal_winner=False,old_experiment_freezes_unchanged=True),new_rounds={str(n):read(B/f'round{n}/results/summary.json') for n in [29,30]},all_checks_passed=True,independent_confirmation=False)
(Q/'aligned_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(rows=len(rows),unique=len(unique),recommendations=[{k:r[k] for k in ['horizon_days','model','wape','daily_wape','peak_mape']} for r in unique if r['status']=='当前综合推荐']),ensure_ascii=False))
