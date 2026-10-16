"""One unified results package; descriptive rankings never change frozen experiments."""
from pathlib import Path
import sys,json,hashlib,csv,datetime
sys.dont_write_bytecode=True
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round28';Q=R/'results'
sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(B/'code'));from source_metrics import evaluate,interval_metrics;import common
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def z(p,k):
 with np.load(B/p) as a:return a[k].astype('float64')
def csvout(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
old=read(B/'round20/results/original_metrics_complete.json');ov=read(B/'round20/results/original_metrics_final_verification.json');assert ov['all_passed'] and sha(B/'round20/results/original_metrics_complete.json')==ov['summary_sha256']
for n in [27,28]:assert read(B/f'round{n}/results/independent_verification.json')['all_passed']
d=common.load();records=[];unique=[];sources={};priors={};selected={}
short={'best':'Chronos-2＋TiRex-2＋TimesFM3\nRidge校准及TimesFM3末层微调','r5':'Chronos-2＋TiRex-2＋TimesFM3\nRidge残差校准','r4':'Chronos-2 25%＋TiRex-2 25%\nTimesFM3 50%固定权重','r3':'Chronos-2＋TiRex-2＋TimesFM3\nRidge与前日偏差加权校准','r2curve':'Chronos-2三分支＋TiRex-2日历\n开发权重及3点平滑','r2peak':'Chronos-2三分支＋TiRex-2日历\n开发权重及前日峰值校正','equal':'Chronos-2 50%＋TiRex-2 50%','tirex':'TiRex-2 零样本单变量','chronos':'Chronos-2 零样本单变量','chronoscov':'Chronos-2 历史production\n功率代理协变量','tf25':'TimesFM 2.5 14天历史\nflip不变性','tf25_7':'TimesFM 2.5 14天历史\n直接672点及flip不变性','tf3':'TimesFM3 单变量21天历史','tf3_uni21_7':'TimesFM3 单变量21天历史\n直接672点','tf3cov':'TimesFM3 14天历史同cell目标\n历史production功率代理','ridge':'Ridge回归 正则1000','lgb':'LightGBM 15叶200树','last':'前一天同刻','last_source':'前一天同刻','mean3_source':'前三天同刻均值','week':'上周同刻','r13candidate':'Chronos-2＋TiRex-2＋TimesFM3\n三日加权SmoothL1残差校准','r9fusion':'Chronos-2＋TiRex-2＋TimesFM3\nRevIN与算电Cross-Attention MoE','r14candidate':'TiDE思路64维直接预测网络','ensemble7':'Chronos-2 LoRA及分量约束\nPatch Transformer及历史曲线','r5_transfer7':'Chronos-2＋TiRex-2＋TimesFM3\nRidge校准7天固定起报迁移','best_transfer7':'Chronos-2＋TiRex-2＋TimesFM3\nRidge及末层微调7天迁移','r3_transfer7':'Chronos-2＋TiRex-2＋TimesFM3\nRidge与前日偏差7天迁移','r15_diagnostic':'TimesFM3＋历史峰值100%\n全日均值保持仿射投影','r16_diagnostic':'TimesFM3＋前日形状25%\n历史峰值50%局部投影','r18_diagnostic':'TimesFM3＋零截距Ridge1000\n前三阶傅里叶残差25%','r19_selected':'TimesFM3＋Ridge1000傅里叶\n全场预测总量约束25%','r20_diagnostic':'原跨度方案＋Ridge学习峰值\n全日均值保持修正','r21_diagnostic':'原跨度方案＋Ridge学习峰值\n最高负荷点稀疏修正','r22_diagnostic':'原跨度方案＋HGB绝对损失7叶\n时点残差及总量约束25%','r23_diagnostic':'原跨度方案＋HGB时点\nRidge稀疏峰值双头25%','r24_diagnostic':'原跨度方案＋HGB时点与稀疏峰值\na5保持及cell总量约束','r25_diagnostic':'TimesFM3＋Ridge＋HGB稀疏峰值\n日均增益1.5及cell总量约束','r26_diagnostic':'TimesFM3＋Ridge时频总量校准\nHGB时点及稀疏峰值25%','current_selected':'Chronos-2＋TiRex-2＋TimesFM3\nRidge校准及TimesFM3末层微调'}
def array_for(H,key):
 if f'{H}:{key}' in old['source_arrays']:
  s=old['source_arrays'][f'{H}:{key}'];return z(s['path'],s['array']),dict(path=s['path'],array=s['array'],sha256=sha(B/s['path']))
 if key in ['last_source','mean3_source']:return common.baseline(d,list(range(22,31)) if H==1 else[22,23,24],H,kind='last' if key=='last_source' else'mean3'),None
 if key=='current_selected':n=26;arr=f'selected{H}'
 elif key=='r19_selected':n=19;arr='selected7'
 else:
  n=int(key[1:key.index('_')]);arr=f'candidate{H}' if n in [15,16] else f'diagnostic{H}'
 path=f'round{n}/results/test_predictions.npz';return z(path,arr),dict(path=path,array=arr,sha256=sha(B/path))
for original in old['rows']:
 H=original['horizon_days'];key=original['model'];p,src=array_for(H,key);sc=evaluate(common,d,list(range(22,31)) if H==1 else[22,23,24],p,H);assert all(abs(sc[k]-original[k])<1e-10 for k in ['wape','mae','daily_wape','peak_mape','hit2_percent','hit5_percent'])
 status='已评估对照'
 if key in ['r13candidate','r9fusion','r14candidate'] or key.endswith('_diagnostic') and key not in ['r24_diagnostic','r25_diagnostic','r26_diagnostic']:status='未采用诊断'
 if key in ['r24_diagnostic','r25_diagnostic']:status='日均峰值研究替代'
 if key in ['current_selected','r26_diagnostic'] or H==1 and key=='best':status='当前综合推荐'
 if H==7 and key=='r19_selected':status='此前综合推荐'
 label=short.get(key,original['model_name'])
 if '原跨度方案' in label:label=label.replace('原跨度方案','Chronos-2＋TiRex-2＋TimesFM3' if H==1 else'TimesFM3＋Ridge傅里叶总量校准')
 if H==7 and key=='current_selected':label=short['r26_diagnostic']
 if H==1 and key=='tirex':label+=' 直接96点'
 if H==7 and key=='tirex':label+=' 分段递推7天'
 rec=dict(original);rec.update(report_name=label,status=status,prediction_digest=hashlib.sha256(p.tobytes()).hexdigest(),source=src);records.append(rec)
 if key.endswith('_diagnostic') and int(key[1:key.index('_')]) in range(20,27):
  n=int(key[1:key.index('_')]);ds=read(B/f'round{n}/results/selection.json')['horizons'][str(H)]
  if n==26:devp=z('round26/results/dev_predictions.npz',f'candidate{H}')
  else:devp=(1-ds['diagnostic_alpha'])*z(f'round{n}/results/dev_predictions.npz',f'prior{H}')+ds['diagnostic_alpha']*z(f'round{n}/results/dev_predictions.npz',f'candidate{H}')
  rec['development_wape_original_grid']=evaluate(common,d,[14,16,18,20] if H==1 else[14,15],devp[:,[0,2,4,6] if H==1 else [0,1]],H)['wape']
for n in [27,28]:
 summ=read(B/f'round{n}/results/summary.json')
 for H in [1,7]:
  origins=list(range(22,31)) if H==1 else[22,23,24];s=summ['selection']['horizons'][str(H)];p=z(f'round{n}/results/test_predictions.npz',f'diagnostic{H}');sc=evaluate(common,d,origins,p,H);base=common.baseline(d,origins,H,kind='last' if H==1 else'mean3');y,m=common.truth(d,origins,H);per=(abs(p-y)*m).sum((1,2))/(abs(y)*m).sum((1,2));pb=(abs(base-y)*m).sum((1,2))/(abs(y)*m).sum((1,2));a=s['diagnostic_alpha'];stem='LightGBM L1稳健时点' if n==27 else'LightGBM限幅日均与25%形状';name=('Chronos-2＋TiRex-2＋TimesFM3\nRidge及末层微调＋' if H==1 else'TimesFM3＋Ridge＋HGB稀疏峰值\n再加')+stem+f'{a*100:g}%';rec=dict(horizon_days=H,model=f'r{n}_frozen',model_name=name.replace('\n',' '),report_name=name,status='开发未支持 未采用诊断' if s['chosen_alpha']==0 else'开发冻结研究替代',selected_alpha=s['chosen_alpha'],diagnostic_alpha=a,**sc,improved_pdus=int((per<pb-1e-12).sum()),total_pdus=57,improved_cells=int(sum(per[d['cells']==c].mean()<pb[d['cells']==c].mean()-1e-12 for c in np.unique(d['cells']))),total_cells=10,improved_origins=int(sum(common.metrics(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1])['wape']<common.metrics(y[:,j:j+1],base[:,j:j+1],m[:,j:j+1])['wape']-1e-10 for j in range(len(origins)))),total_origins=len(origins),prediction_digest=hashlib.sha256(p.tobytes()).hexdigest(),source=dict(path=f'round{n}/results/test_predictions.npz',array=f'diagnostic{H}',sha256=sha(B/f'round{n}/results/test_predictions.npz')))
  cp=z(f'round{n}/results/dev_predictions.npz',f'raw_candidate{H}');bp=z(f'round{n}/results/dev_predictions.npz',f'prior{H}');dp=(1-a)*bp+a*cp;sel=[0,2,4,6] if H==1 else [0,1];rec['development_wape_original_grid']=evaluate(common,d,[14,16,18,20] if H==1 else[14,15],dp[:,sel],H)['wape'];records.append(rec)
for H in [1,7]:
 group=[r for r in records if r['horizon_days']==H];digest={}
 for r in sorted(group,key=lambda r:0 if r['status']=='当前综合推荐' else 1):
  k=r['prediction_digest']
  if k not in digest:rr=dict(r);rr['aliases']=[r['model']];digest[k]=rr
  else:
   digest[k]['aliases'].append(r['model'])
   if digest[k].get('development_wape_original_grid') is None and r.get('development_wape_original_grid') is not None:digest[k]['development_wape_original_grid']=r['development_wape_original_grid']
 unique.extend(sorted(digest.values(),key=lambda r:r['wape']))
winners=[]
taskmetrics={'24小时曲线':['wape','mae','rmse','daily_wape','peak_mape','p95_ape','hit2_percent','hit5_percent','peak_time_mae_minutes','absolute_bias_percent','cella5_wape','other52_wape'],'7天曲线':['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent','peak_time_mae_minutes','absolute_bias_percent','cella5_wape','other52_wape'],'7天每日平均负荷':['daily_wape','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent']}
for task,metrics in taskmetrics.items():
 H=1 if task.startswith('24') else 7;group=[r for r in unique if r['horizon_days']==H]
 for metric in metrics:
  higher=metric.startswith('hit') or metric.startswith('daily_hit');best=(max if higher else min)(r[metric] for r in group);wins=[r for r in group if abs(r[metric]-best)<1e-10];winners.append(dict(task=task,metric=metric,direction='越高越好' if higher else'越低越好',best_value=best,models='；'.join(r['model_name'] for r in wins),keys=[r['model'] for r in wins],current_recommended_is_best=any(r['status']=='当前综合推荐' for r in wins)))
iv=[]
for x in old['intervals']:
 r=dict(x)
 if x['kind'].startswith('本轮实测') and '第26轮' in x['method']:r['method']='24小时当前综合推荐经验80%区间' if x['horizon_days']==1 else'7天当前综合推荐经验80%区间'
 iv.append(r)
for n in [27,28]:
 r=read(B/f'round{n}/results/summary.json');iv.append(dict(horizon_days=7,method=f'第{n}轮LightGBM开发冻结研究替代经验80%区间',kind='本轮实测 开发冻结研究替代',**r['scores']['7']['interval_selected']))
# Only a reporting recommendation is retained; R27/R28 experimental frozen weights remain untouched.
decision=dict(recommended24='round8b',recommended7='round26',reason='R27 improves daily/peak but worsens curve/RMSE and drops hit5 beyond0.05pp. R28 improves daily/peak-time/bias but worsens curve/MAE/RMSE/P95/peak amplitude. It qualifies as a small-tradeoff research alternative, not a broader improvement over the balanced R26 recommendation.',new_universal_winner=False,previous_freezes_unchanged=True,independent_confirmation=False)
csvout(T/'模型各项指标完整结果.csv',[{k:v for k,v in r.items() if k not in ['source','prediction_digest']} for r in records]);csvout(T/'各任务单项最优结果.csv',winners);csvout(T/'模型预测区间结果.csv',iv)
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=records,unique_rows=unique,winners=winners,intervals=iv,recommendation=decision,source_reference=old['reference_inventory'],original_metrics_json_sha256=sha(B/'round20/results/original_metrics_complete.json'),csv_sha256=sha(T/'模型各项指标完整结果.csv'),winners_csv_sha256=sha(T/'各任务单项最优结果.csv'),interval_csv_sha256=sha(T/'模型预测区间结果.csv'),duplicate_policy='Exactly identical float64 prediction arrays merge in Word, aliases retained; complete CSV retains every62 individual record.',independent_confirmation=False)
save=Q/'clear_comparison.json';save.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(rows=len(records),unique_counts={H:sum(x['horizon_days']==H for x in unique) for H in [1,7]},recommendation=decision),ensure_ascii=False))
