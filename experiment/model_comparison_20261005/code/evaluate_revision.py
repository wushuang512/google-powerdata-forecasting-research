"""Three-task report revision from retained frozen forecasts; no model fitting."""
from pathlib import Path
import json,sys,hashlib,datetime
import numpy as np,pandas as pd
TOP=Path(sys.argv[1]).resolve(); B=TOP/'experiment'; O=B/'model_comparison_20261005'
OUT=O/'results/task_revision'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(B/'code')); import common
D=common.load(); old=json.loads((O/'results/summary.json').read_text())
NAMES={
 'best':'Chronos-2＋TiRex-2＋TimesFM3 三模型固定权重预测 加 Ridge残差校准与TimesFM3末层微调',
 'r5':'Chronos-2＋TiRex-2＋TimesFM3 三模型固定权重预测 加 Ridge残差校准',
 'r4':'Chronos-2 25%＋TiRex-2 25%＋TimesFM3 50% 固定权重预测',
 'r3':'Chronos-2＋TiRex-2＋TimesFM3 加 Ridge残差与前日偏差校准的开发期加权预测',
 'r2curve':'Chronos-2三分支微调与分量约束＋TiRex-2日历分支 开发期加权预测 加3点平滑',
 'r2peak':'Chronos-2三分支微调与分量约束＋TiRex-2日历分支 开发期加权预测 加3点平滑及前日峰值校正',
 'equal':'Chronos-2 50%＋TiRex-2 50% 等权预测',
 'tirex':'TiRex-2 零样本单变量预测',
 'chronos':'Chronos-2 零样本单变量预测',
 'chronoscov':'Chronos-2 历史production功率代理协变量预测',
 'tf25':'TimesFM 2.5 14天历史单变量预测 启用force_flip_invariance',
 'tf3':'TimesFM3 21天历史单变量预测',
 'tf3cov':'TimesFM3 14天历史同集群联合目标与production功率代理协变量预测',
 'ridge':'Ridge回归 正则参数alpha 1000',
 'lgb':'LightGBM回归 15叶节点 200棵树 无production协变量',
 'last':'前一天同刻曲线重复基线',
 'week':'上周同刻基线',
 'ensemble7':'Chronos-2 LoRA与协变量及分量约束预测＋Patch Transformer＋历史曲线 开发期加权预测',
 'r13candidate':'三日近期加权SmoothL1残差校准器 h3 p2 未采用候选',
 'r9fusion':'RevIN多尺度分解＋算电Cross-Attention＋稀疏MoE真实CPU残差候选 10轮训练 未采用',
 'r14candidate':'TiDE思路的64维局部直接预测网络 20轮训练 两固定种子平均 未采用',
}
SHORT={
 'best':'Chronos-2＋TiRex-2＋TimesFM3\nRidge残差校准＋TimesFM3末层微调',
 'r5':'Chronos-2＋TiRex-2＋TimesFM3\nRidge残差校准',
 'r4':'Chronos-2 25%＋TiRex-2 25%\n＋TimesFM3 50%',
 'r3':'Chronos-2＋TiRex-2＋TimesFM3\nRidge与前日偏差加权校准',
 'r2curve':'Chronos-2三分支＋TiRex-2日历\n开发期加权与3点平滑',
 'r2peak':'Chronos-2三分支＋TiRex-2日历\n加权平滑与前日峰值校正',
 'equal':'Chronos-2 50%＋TiRex-2 50%',
 'tirex':'TiRex-2 零样本单变量',
 'chronos':'Chronos-2 零样本单变量',
 'chronoscov':'Chronos-2 历史功率代理协变量',
 'tf25':'TimesFM 2.5 单变量\n14天历史与flip不变性',
 'tf3':'TimesFM3 单变量 21天历史',
 'tf3cov':'TimesFM3 同集群目标与功率代理\n14天历史',
 'ridge':'Ridge回归 alpha 1000',
 'lgb':'LightGBM 15叶 200树',
 'last':'前一天同刻曲线重复',
 'week':'上周同刻',
 'ensemble7':'Chronos-2 LoRA与协变量及分量约束\n＋Patch Transformer＋历史曲线',
}
NAMES.update({
 'r5_transfer7':NAMES['r5']+' 七天固定起报逐日校准迁移版',
 'best_transfer7':NAMES['best']+' 七天固定起报逐日校准迁移版',
 'r3_transfer7':NAMES['r3']+' 七天固定起报逐日残差校准迁移版',
 'tf3_uni21_7':NAMES['tf3']+' 直接672点预测',
 'tf25_7':NAMES['tf25']+' 直接672点预测',
})
SHORT.update({
 'r5_transfer7':'Chronos-2＋TiRex-2＋TimesFM3\nRidge校准 7天固定起报迁移',
 'best_transfer7':'Chronos-2＋TiRex-2＋TimesFM3\nRidge校准＋TimesFM3末层微调\n7天固定起报迁移',
 'r3_transfer7':'Chronos-2＋TiRex-2＋TimesFM3\nRidge与偏差校准 7天迁移',
 'tf3_uni21_7':SHORT['tf3'], 'tf25_7':SHORT['tf25'],
})
SOURCES={};CHECKS={}; C={}
def load(rel,key):
 p=B/rel
 if rel not in C:
  with np.load(p,allow_pickle=False) as z:C[rel]={k:z[k] for k in z.files}
  SOURCES[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
 return C[rel][key].astype('float64')
def score(y,p,m):return common.metrics(y,p,m)
y,m=common.truth(D,list(range(22,31)))
daily24=[]
for i,a in old['models'].items():
 a['name']=NAMES[i];a['short_name']=SHORT.get(i,NAMES[i])
 q=load(a['source'],a['array']); res=score(y,q,m)
 CHECKS['24h_'+i]=all(abs(a[k]-res[k])<1e-10 for k in res if isinstance(res[k],(float,int)))
 # all prior saved multidimensional metrics remain unchanged
daily24=list(old['models'].values())
regs=[
 ('tirex','round2/results/test7_predictions.npz','tirex_uni'),
 ('r4','round4/results/test7.npz','best_mix'),
 ('r2curve','round2/results/test7_predictions.npz','new_curve'),
 ('r2peak','round2/results/test7_predictions.npz','new_peak'),
 ('tf3cov','round4/results/test7.npz','tf3_cellprod'),
 ('ensemble7','results/test7_predictions.npz','ensemble7'),
 ('equal','round2/results/test7_predictions.npz','old_equal_mix'),
 ('lgb','results/test7_predictions.npz','lgb_p0_l15_n200'),
 ('ridge','results/test7_predictions.npz','ridge_1000'),
 ('chronos','round2/results/test7_predictions.npz','zero_uni'),
 ('chronoscov','round2/results/test7_predictions.npz','zero_cov'),
 ('last','results/test7_predictions.npz','last'),
 ('week','results/test7_predictions.npz','week'),
]
completed=B/'seven_day_completion_20261005/results/pipeline_complete.json'
missing=['best','r5','r3','tf25','tf3']
completion=None
if completed.exists():
 completion=json.loads((completed.parent/'summary.json').read_text())
 assert completion['fixed_weights'] and not completion['new_fit'] and completion['feature_future_perturbation_passed']
 for ident in ['r5_transfer7','best_transfer7','r3_transfer7','tf3_uni21_7','tf25_7']:
  regs.append((ident,'seven_day_completion_20261005/results/test7_predictions.npz',ident))
 missing=[]
 for stage in ['tf3_uni21_7','tf3_finetuned7','chronos_mirror7','tf25_7']:
  meta=json.loads((B/f'seven_day_completion_20261005/cache/{stage}_control.json').read_text())
  CHECKS['frozen_control_'+stage]=meta['checks']['max_abs']<=1e-5
 CHECKS['r3_residual_frozen_reproduction']=completion['residual_control_max_abs']<=1e-10
 CHECKS['feature_future_perturbation']=completion['feature_future_perturbation_passed']
y7,m7=common.truth(D,[22,23,24],7);Y=y7.reshape(57,3,7,96);M=m7.reshape(57,3,7,96)
count=M.sum(-1); ym=(Y*M).sum(-1)/count.clip(1);dv=count>0
seven=[];means=[];horizons=[];rows=[]; preds={};independent={}
for i,rel,key in regs:
 p=load(rel,key);assert p.shape==(57,3,672) and np.isfinite(p).all();preds[i]=p
 ps=p.reshape(57,3,7,96); pm=(ps*M).sum(-1)/count.clip(1); deploy_mean=ps.mean(-1)
 a=dict(model=i,name=NAMES[i],short_name=SHORT[i],source=rel,array=key,**score(y7,p,m7));seven.append(a)
 if completion is not None and i in completion['scores']:
  CHECKS['new_inference_score_'+i]=abs(a['wape']-completion['scores'][i]['wape'])<1e-12
 dm=dict(model=i,name=NAMES[i],daily_wape=a['daily_wape'],daily_mean_mae=float(abs(pm-ym)[dv].mean()),daily_mean_rmse=float(np.sqrt(np.square(pm-ym)[dv].mean())),daily_mean_bias=float((pm-ym)[dv].mean()),domain_origin_days=int(dv.sum()),source=rel,array=key)
 # independent explicit PDU/date/day loop, never calling common.metrics
 ratios=[];absdaily=[]
 for s in range(57):
  num=den=0
  for o in range(3):
   for h in range(7):
    v=M[s,o,h];actual=float(Y[s,o,h,v].mean());pred=float(ps[s,o,h,v].mean());num+=abs(pred-actual);den+=abs(actual);absdaily.append(abs(pred-actual))
  ratios.append(100*num/den)
 CHECKS['independent_daily_'+i]=abs(float(np.mean(ratios))-a['daily_wape'])<1e-10 and abs(np.mean(absdaily)-dm['daily_mean_mae'])<1e-12
 means.append(dm)
 for h in range(7):
  z=score(Y[:,:,h],ps[:,:,h],M[:,:,h]);horizons.append(dict(model=i,name=NAMES[i],forecast_day=h+1,curve_wape=z['wape'],daily_mean_wape=z['daily_wape'],valid_points=z['valid_points']))
 for s,n in enumerate(D['names']):
  for o,origin in enumerate([22,23,24]):
   for h in range(7):rows.append(dict(model=i,name=NAMES[i],pdu=str(n),cell=str(D['cells'][s]),origin_end_day=origin,forecast_day=h+1,target_date=f'2019-05-{origin+h+1:02}',actual_valid_slot_mean=float(ym[s,o,h]),predicted_valid_slot_mean=float(pm[s,o,h]),predicted_full96_mean=float(deploy_mean[s,o,h]),valid_slots=int(count[s,o,h]),absolute_error=float(abs(pm[s,o,h]-ym[s,o,h]))))
CHECKS['truth_7_valid_114902']=int(m7.sum())==114902
CHECKS['truth_days_1197']=int(dv.sum())==57*3*7
CHECKS['baseline_last7']=np.array_equal(preds['last'],common.baseline(D,[22,23,24],7,'last'))
CHECKS['baseline_week7']=np.array_equal(preds['week'],common.baseline(D,[22,23,24],7,'week'))
CHECKS['equal_mix7']=np.allclose(preds['equal'],.5*preds['chronos']+.5*preds['tirex'],rtol=0,atol=1e-7)
CHECKS['r4_formula7']=np.allclose(preds['r4'],np.clip(.5*preds['equal']+.5*preds['tf3cov'],0,1),rtol=0,atol=1e-12)
z5=C['round5/results/test_predictions.npz'];z8=C['round8b/results/test_predictions.npz']
s5=json.loads((B/'round5/results/selection.json').read_text());s8=json.loads((B/'round8b/results/selection.json').read_text())
CHECKS['r5_exact_calibration_formula']=np.array_equal(z5['selected'],np.clip(.5*z5['anchor']+.5*z5['calibration'],0,1))
CHECKS['r5_family_calibration']=s5['selected']['family']=='calibration' and s5['selected']['alpha']==.5 and s5['chosen']['calibration']=='residual_ridge100000'
CHECKS['r8_formula']=np.array_equal(z8['selected'],np.clip(.5*z8['prior']+.5*z8['candidate'],0,1))
CHECKS['r8_frozen_config']=s8['config']=='best_tail_lr1e6_teacher01_s100' and s8['chosen']['alpha']==.5
f=load('round3/results/tf3_test.npz','tf3_cellprod');eq=load('round2/results/test24_predictions.npz','old_equal_mix');r4=load('round4/results/test_predictions.npz','round4_selected')
CHECKS['r4_formula24']=np.allclose(r4,np.clip(.5*eq+.5*f,0,1),rtol=0,atol=1e-12)
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
for fn,arr in [('comparison_24h.csv',daily24),('comparison_7day_curve.csv',seven),('comparison_7day_daily_mean.csv',means),('seven_day_by_horizon.csv',horizons),('seven_day_daily_means_detail.csv',rows)]:pd.DataFrame(arr).to_csv(OUT/fn,index=False,encoding='utf-8-sig')
single=np.load(B/'round3/results/tf3_test.npz')['tf3_cellprod'].astype('float64')
tf3cov24=score(y,single,m)
CHECKS={k:bool(v) for k,v in CHECKS.items()}
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=daily24,seven_day=seven,daily_mean=means,horizons=horizons,names=NAMES,short_names=SHORT,sources=SOURCES,checks=CHECKS,all_passed=all(CHECKS.values()),
 task_protocol=dict(h24_points=96,h24_origins=list(range(22,31)),h24_valid_points=int(m.sum()),h7_points=672,h7_origins=[22,23,24],h7_valid_points=int(m7.sum()),h7_valid_pdu_days=int(dv.sum()),seven_windows_overlap=True,daily_means_derived_from_curves=True,no_new_training=True,no_test_reselection=True,independent_confirmation=False),
 tf3cov24=tf3cov24,
 missing_seven_day=missing,
 actual_missing_tasks_completed=bool(completion),
 transferred_seven_day_methods=['r5_transfer7','best_transfer7','r3_transfer7'] if completion else [],
 seven_task_key={'best':'best_transfer7','r5':'r5_transfer7','r3':'r3_transfer7','tf25':'tf25_7','tf3':'tf3_uni21_7'} if completion else {},
 daily_mean_mask='Scoring true and predicted means on same valid target slots; prediction-only exported full96 mean uses all96 forecast points',
 best24='best',best7curve=min(seven,key=lambda a:a['wape'])['model'],best7daily=min(means,key=lambda a:a['daily_wape'])['model'],
 labels_changed_only=not bool(completion))
dump('summary.json',out);assert out['all_passed'],[k for k,v in CHECKS.items() if not v]
print(json.dumps(dict(checks=len(CHECKS),all_passed=True,best7curve=out['best7curve'],best7daily=out['best7daily'],seven_day=[{k:a[k] for k in ['model','wape','daily_wape','peak_mape']} for a in seven]),ensure_ascii=False,indent=2))
