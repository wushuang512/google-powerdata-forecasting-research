"""Original-reference inventory and unified retrospective metrics. No fitting or selection."""
from pathlib import Path
import sys,json,hashlib,csv,datetime
import numpy as np
from docx import Document
from docx.oxml.ns import qn
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round20';Q=R/'results';sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(B/'code'));import common
from source_metrics import evaluate,interval_metrics
d=common.load()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def z(p,k):
 with np.load(B/p) as a:return a[k].astype('float64')
def csvout(p,rows):
 keys=[]
 for row in rows:
  for k in row:
   if k not in keys:keys.append(k)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
ref=T/'参考材料/9.28算力中心需求预测实验.docx';doc=Document(ref)
blocks=[dict(body_index=i,text=''.join(n.text or '' for n in e.iter(qn('w:t')))) for i,e in enumerate(doc.element.body) if e.tag in [qn('w:p'),qn('w:tbl')]]
inventory=[
 dict(metric='开发或验证WAPE',source_blocks=[24,33,51],status='按9.28原DEV24起报14/16/18/20复算匹配预测；新调优另列前后4日，7天起报14/15。未保存对应开发数组的对照留空不编造。'),
 dict(metric='测试曲线WAPE',source_blocks=[24,33,37],status='24小时和7天分别复算；逐PDU比值后57域等权平均。'),
 dict(metric='相对基线误差减少比例',source_blocks=[25,33,37,39],status='24小时前一天同刻，7天前三天同刻均值；负值表示退步。'),
 dict(metric='MAE',source_blocks=[25,27],status='Google以归一化利用率，海光寺原文kWh；本轮不重新评估海光寺。'),
 dict(metric='时点误差不超过2%的占比',source_blocks=[46],status='有效点APE<=2%，百分比越高越好。'),
 dict(metric='时点误差不超过5%的占比',source_blocks=[46],status='有效点APE<=5%，百分比越高越好。'),
 dict(metric='时点APE95分位',source_blocks=[46],status='有效点APE的95分位，越低越好。'),
 dict(metric='每日最大值平均百分比误差',source_blocks=[46,48,51],status='每日有效点最大值幅值MAPE；时刻误差另外报告。'),
 dict(metric='7天每日均值WAPE',source_blocks=[37],status='同日真值与预测均只取有效时刻；输出部署使用完整96点均值。'),
 dict(metric='80%预测区间实际覆盖率',source_blocks=[48],status='保存区间逐点复算，并新增平均宽度和Winkler80防止仅靠加宽区间。TiRex旧56.6%原生区间未保存数组，仅历史引用。'),
 dict(metric='改善的序列cell与日期数',source_blocks=[35,43],status='对照同任务基线，复算域数、cell数与起报数；重叠7天窗口不是独立日。')]
save=dict(reference=str(ref),reference_sha256=sha(ref),blocks=blocks,metrics=inventory,all_original_google_metric_categories_covered=True,scope='Google only; original Haiguangsi and history/context/leave-cell-out ablations are historical reference, not this new retraining.')
(Q/'reference_metric_inventory.json').write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8')
old=read(B/'model_comparison_20261005/results/task_revision/summary.json');names=old['short_names'];rows=[];arrays={1:{},7:{}};srcs={}
for H,records in [(1,old['models']),(7,old['seven_day'])]:
 origins=list(range(22,31)) if H==1 else [22,23,24]
 for rec in records:
  p=z(rec['source'],rec['array']);key=rec['model'];arrays[H][key]=p;srcs[f'{H}:{key}']=dict(path=rec['source'],array=rec['array'],sha256=sha(B/rec['source']))
 arrays[H]['mean3_source']=common.baseline(d,origins,H,kind='mean3');arrays[H]['last_source']=common.baseline(d,origins,H,kind='last')
 if H==7:
  for n in [15,16]:arrays[H][f'r{n}_diagnostic']=z(f'round{n}/results/test_predictions.npz','candidate7')
  arrays[H]['r18_diagnostic']=z('round18/results/test_predictions.npz','diagnostic7');arrays[H]['r19_selected']=z('round19/results/test_predictions.npz','selected7')
 for n in [20,21,22,23,24,25,26]:
  if n>=25 and H==1:continue
  if not (B/f'round{n}/results/test_complete.json').exists():
   assert '--available-only' in sys.argv,'Complete all rounds before final evaluation';continue
  arrays[H][f'r{n}_diagnostic']=z(f'round{n}/results/test_predictions.npz',f'diagnostic{H}')
 # Explicit current base is unchanged if all latest strengths are0.
 n=max(n for n in [20,21,22,23,24,25,26] if (B/f'round{n}/results/test_complete.json').exists());arrays[H]['current_selected']=z(f'round{n}/results/test_predictions.npz',f'selected{H}')
 for ident,p in arrays[H].items():
  label=names.get(ident,dict(mean3_source='前三天同刻均值 9.28七天基线',last_source='前一天同刻 9.28次日基线',r15_diagnostic='历史峰值100%保持日均的整曲线投影 第15轮未采用',r16_diagnostic='前日形状25%历史峰值50%局部投影 第16轮未采用',r18_diagnostic='TimesFM3原生672点 零截距Ridge1000前三阶傅里叶残差25% 第18轮未采用',r19_selected='TimesFM3原生672点 零截距Ridge1000傅里叶残差 全场预测总量约束25%',r20_diagnostic='原保留模型加TimesFM3学习峰值头 全日均值约束投影 第20轮未采用',r21_diagnostic='原保留模型加TimesFM3学习峰值头 最高负荷点稀疏投影 第21轮冻结诊断',current_selected='Chronos-2 TiRex-2 TimesFM3固定权重 Ridge校准 TimesFM3末层微调' if H==1 else 'TimesFM3原生672点 零截距Ridge1000傅里叶残差 全场预测总量约束25%').get(ident,ident))
  if ident=='current_selected' and (B/'round21/results/selection.json').exists():
   sel=read(B/f'round{n}/results/selection.json')['horizons'][str(H)]
   if sel['chosen_alpha']>0:label+=' 加固定25%HGB时点残差和Ridge稀疏峰值双头 日均修正增益1.5 保留a集群 分cell总量约束 第26轮当前均衡推荐'
  if ident.startswith('r') and ident.endswith('_diagnostic') and int(ident[1:ident.index('_')]) in [22,23,24,25,26]:
   label={22:'原曲线方案 加HGB绝对损失7叶时点残差与全场总量约束 第22轮未采用诊断',23:'原曲线方案 加HGB时点残差与Ridge稀疏峰值双头 第23轮未采用诊断',24:'TimesFM3 Ridge傅里叶与总量约束 加HGB时点残差及稀疏峰值头 保留a集群 分cell总量约束 第24轮冻结研究替代',25:'第24轮HGB时点与稀疏峰值方案 日均修正增益1.5日内形状1 第25轮冻结研究替代',26:'第19轮TimesFM3 Ridge时频与总量约束 加固定25%第25轮双头修正 第26轮当前均衡推荐'}[int(ident[1:ident.index('_')])]
  a=evaluate(common,d,origins,p,H);base=common.baseline(d,origins,H,kind='last' if H==1 else 'mean3');y,m=common.truth(d,origins,H);pe=(abs(p-y)*m).sum((1,2))/(abs(y)*m).sum((1,2));be=(abs(base-y)*m).sum((1,2))/(abs(y)*m).sum((1,2));nc=0
  for cell in np.unique(d['cells']):nc+=bool(pe[d['cells']==cell].mean()<be[d['cells']==cell].mean()-1e-12)
  nd=sum(common.metrics(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1])['wape']<common.metrics(y[:,j:j+1],base[:,j:j+1],m[:,j:j+1])['wape']-1e-10 for j in range(len(origins)))
  rows.append(dict(horizon_days=H,model=ident,model_name=label,improved_pdus=int((pe<be-1e-12).sum()),total_pdus=57,improved_cells=nc,total_cells=10,improved_origins=nd,total_origins=len(origins),**a))
 # Correct original DEV grid, without pretending all old TEST records have matched DEV caches.
 dev_orig=[14,16,18,20] if H==1 else [14,15];dev={}
 if H==1:
  dev['current_selected']=z(f'round{n}/results/dev_predictions.npz','selected1')[:,[0,2,4,6]]
  dev['tf3']=z('round3/results/tf3_dev.npz','tf3_uni_context2016')[:,[0,2,4,6]]
  for ident,key in [('tirex','tirex_uni'),('chronos','zero_uni'),('equal','old_equal_mix')]:dev[ident]=z('round2/results/development_selected.npz',key)[:,[0,2,4,6]]
  for ident,key in [('lgb','lgb_p0_l15_n200'),('ridge','ridge_1000')]:dev[ident]=z('results/dev_classical.npz',key)
 else:
  dev['current_selected']=z(f'round{n}/results/dev_predictions.npz','selected7');dev['r19_selected']=.75*z('round18/results/dev_predictions.npz','prior')+.25*z('round19/results/dev_predictions.npz','global_peak0');dev['tf3_uni21_7']=z('round15/results/tf3_dev7.npz','prediction')
 for ident,kind in [('last_source','last'),('mean3_source','mean3')]:dev[ident]=common.baseline(d,dev_orig,H,kind=kind)
 for row in [r for r in rows if r['horizon_days']==H]:
  if row['model'] in dev:row['development_wape_original_grid']=evaluate(common,d,dev_orig,dev[row['model']],H)['wape']
  else:row['development_wape_original_grid']=None
csvout(T/'9月28日指标完整模型对照.csv',rows)
iv=[]
for H in [1,7]:
 n=max(n for n in [20,21,22,23,24,25,26] if (B/f'round{n}/results/test_complete.json').exists());center=arrays[H]['current_selected'];lo=z(f'round{n}/results/test_predictions.npz',f'lower{H}');hi=z(f'round{n}/results/test_predictions.npz',f'upper{H}');y,m=common.truth(d,list(range(22,31)) if H==1 else [22,23,24],H);iv.append(dict(horizon_days=H,method='次日保留方案或七天第26轮均衡推荐 开发残差经验80%区间',kind='本轮实测 非原生基础模型分位数',**interval_metrics(y,center,m,lo,hi)))
 if H==7:
  center=z('round21/results/test_predictions.npz','selected7');lo=z('round21/results/test_predictions.npz','lower7');hi=z('round21/results/test_predictions.npz','upper7');iv.append(dict(horizon_days=7,method='第19轮七天曲线推荐方案 开发残差经验80%区间',kind='本轮实测 既有曲线推荐方案',**interval_metrics(y,center,m,lo,hi)))
 # Saved raw Chronos intervals are available and audited; no raw TiRex arrays were saved.
 files='results/test24_quantiles.npz' if H==1 else 'results/test7_quantiles.npz';qq=z(files,'chronos_zero');iv.append(dict(horizon_days=H,method='Chronos-2零样本原生80%区间',kind='保存原生分位数',**interval_metrics(y,qq[...,1],m,qq[...,0],qq[...,2])))
iv.append(dict(horizon_days=7,method='TiRex-2 9.28原文原生80%区间',kind='历史引用 原区间数组未保存 不作为本轮实測排序',nominal_percent=80,coverage_percent=56.6,mean_width_normalized=None,winkler80=None,below_percent=None,above_percent=None,valid_points=None))
csvout(T/'9月28日指标预测区间对照.csv',iv)
# Preserve the source's old reported figures separately, never mix them into current recomputed rankings.
historical=dict(curve24_equal_wape=2.416,curve24_last_wape=3.199,equal_hit2=55.8,equal_hit5=88.1,equal_p95=7.02,equal_peak=2.11,last_hit2=45.3,last_hit5=78.5,last_p95=9.63,last_peak=1.07,curve7_mean3_wape=3.212,curve7_tirex_wape=2.880,daily7_mean3_wape=2.262,daily7_tirex_wape=2.127,tirex7_raw80_coverage=56.6)
summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,intervals=iv,reference_inventory=save,original_historical_reference_only=historical,source_arrays=srcs,all_original_metric_categories_covered=True,current_inference_source=f'round{n}',recommended='round8b24 and round26 balanced seven-day under user minor-tradeoff preference',previous_recommendation='round19 seven-day',alternative_research='round24 and25 DEV-selected day/peak tradeoffs',original_daily_figures_not_exactly_equal_current=True,discrepancy_note='9.28 historical figures and current uniform valid-slot recomputation are separate. Some model configurations changed and original daily arrays/implementation not attached; no claim all differences are tuning gains.',independent_confirmation=False)
(Q/'original_metrics_complete.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(rows=len(rows),intervals=len(iv),coverage_current=[r['coverage_percent'] for r in iv if r['kind'].startswith('本轮')],all_original_metrics=True,pending21=not (B/'round21/results/test_complete.json').exists()),ensure_ascii=False))

