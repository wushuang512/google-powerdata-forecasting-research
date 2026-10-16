from pathlib import Path
import json,sys,csv,hashlib,datetime,numpy as np
T=Path(sys.argv[1]);B=T/'experiment';Q=B/'round40/results';p=Q/'continued_comparison.json';a=json.loads(p.read_text(encoding='utf-8'));assert a['all_checks_passed'] and len(a['rows'])==114
labels={
 (37,1):'Moirai2 小时同场站 Rank8输出头10%\n当前24h方案90% 全域日均协调',
 (37,7):'Moirai2 小时同场站中位数10%\n当前7天方案90% 全域日均协调',
 (38,1):'Moirai2 15分钟单变量10%\n当前24h方案90%',
 (38,7):'Moirai2 15分钟季节残差10%\n当前7天方案90%',
 (39,1):'Moirai2 15分钟相关域联合10%\n当前24h方案90%',
 (39,7):'Moirai2 15分钟相关域联合10%\n当前7天方案90%',
 (40,1):'Moirai2 三小时57域 CPU内存10%\n当前24h方案90% 全域日均协调',
 (40,7):'Moirai2 三小时57域 CPU内存10%\n当前7天方案90% 全域日均协调'}
for r in a['rows']+a['unique_rows']:
 for n in range(37,41):
  if r['model']==f'r{n}_diagnostic_H{r["horizon_days"]}' or r['model']==f'r{n}_selected_H{r["horizon_days"]}' and n==38 and r['horizon_days']==1:
   r['report_name']=labels[n,r['horizon_days']];r['model_name']='Salesforce Moirai 2.0 R small '+labels[n,r['horizon_days']].replace('Moirai2 ','').replace('\n','；')
   r['status']='开发期选定但测试验收未通过' if n==38 and r['horizon_days']==1 else'开发升级未通过的冻结研究候选'
def write(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r if k not in ['source','aliases','prediction_digest']))
 with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
for w in a['winners']:
 H=1 if w['task']=='24小时曲线' else 7
 w['models']='；'.join(r['model_name'] for r in a['unique_rows'] if r['horizon_days']==H and abs(r[w['metric']]-w['value'])<1e-10)
sys.path[:0]=[str(B/'code'),str(B/'round29/code')];import common;from aligned_metrics import evaluate
d=common.load();boot=[];rng=np.random.default_rng(20261007)
for H,n in [(1,37),(7,40)]:
 oo=list(range(22,31)) if H==1 else[22,23,24]
 with np.load(B/f'round{n}/results/test_predictions.npz') as z:
  new=z[f'diagnostic_H{H}'];old=z[f'selected_H{H}'];g=np.array([evaluate(common,d,[o],old[:,i:i+1],H)['wape']-evaluate(common,d,[o],new[:,i:i+1],H)['wape'] for i,o in enumerate(oo)])
 truth,mask=common.truth(d,oo,H);mass=(np.abs(truth)*mask).sum(-1);ea=(np.abs(truth-old)*mask).sum(-1);eb=(np.abs(truth-new)*mask).sum(-1);ids=rng.integers(0,len(g),(5000,len(g)));samples=100*np.mean((ea[:,ids].sum(-1)-eb[:,ids].sum(-1))/mass[:,ids].sum(-1),axis=0);score_new=evaluate(common,d,oo,new,H);score_old=evaluate(common,d,oo,old,H)
 boot.append(dict(horizon_days=H,round=n,gain_pp=score_old['wape']-score_new['wape'],relative_gain_percent=100*(1-score_new['wape']/score_old['wape']),paired_origin_gain_pp=g.tolist(),positive_origins=int((g>0).sum()),paired_origin_bootstrap95_gain_pp=np.quantile(samples,[.025,.975]).tolist(),bootstrap_recalculates_whole_PDU_WAPE=True,exploratory=True,independent_confirmation=False,warning='Only May2019 repeatedly inspected TEST; overlapping 7day origins; intervals descriptive, not independent significance'))
a['continued_bootstrap']=boot;a['all_curves_leading_one_frozen_method']=False;a['substantial_improvement']=False;a['recommended_deployment_unchanged']=True;a['cpu_dev_ablation']=[]
dv=json.loads((Q/'dev_summary.json').read_text(encoding='utf-8'))
for H in [1,7]:
 for family in ['power57_control_3hour','power57_cpu_mem_a_3hour']:
  s=dv[str(H)]['families'][family];a['cpu_dev_ablation'].append(dict(horizon_days=H,family=family,alpha=.25,total='none',development_wape=s['wape'],development_daily_wape=s['daily_wape'],cella5_wape=s['cella5_wape'],other52_wape=s['other52_wape']))
p.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
write(T/'模型各项指标完整结果.csv',a['rows']);write(T/'各任务单项最优结果.csv',a['winners']);write(T/'模型预测区间结果.csv',a['intervals']);write(T/'9月28日统一口径模型复算.csv',a['rows']);write(T/'9月28日指标完整模型对照.csv',a['rows']);write(T/'9月28日指标预测区间对照.csv',a['intervals']);write(Q/'真实CPU内存开发期对照.csv',a['cpu_dev_ablation'])
for n in range(37,41):
 assert not any(v['adopt'] for v in a['release_decisions'][str(n)].values())
 (B/f'round{n}/results/release_decision.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decisions=a['release_decisions'][str(n)],recommendation24='round32',recommendation7='round36',no_test_winner_reselection=True),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(rows=114,unique=92,bootstrap=boot,cpu_dev=a['cpu_dev_ablation']),ensure_ascii=False))
