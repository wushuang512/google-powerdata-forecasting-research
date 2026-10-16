from pathlib import Path
import sys,json,hashlib,datetime,csv
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round44';Q=R/'results';sys.path[:0]=[str(B/'code'),str(B/'round29/code'),str(B/'round28/code')]
import common
from aligned_metrics import evaluate
from source_metrics import interval_metrics
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def export(p,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
d=common.load();s=read(Q/'summary.json');assert read(Q/'science_verification.json')['all_passed'];old=read(B/'round43/results/factor_comparison.json');z=np.load(Q/'predictions.npz');rows=[];intervals=[];checks=[]
names={'hdsf':'HDSF半日单纯形融合（原项目D14/D22固定训练）','chronos2_long':'Amazon Chronos-2（最多28天历史，CPU）','tirex2':'NX-AI TiRex-2（14天历史，CPU）','timesfm25':'Google TimesFM2.5 200M（14天历史，CPU）','toto2_joint':'Datadog Toto2.0 313M（同cell及设备类型联合，14天）','toto2_single':'Datadog Toto2.0 313M（单域，14天）','timesfm30':'Google TimesFM3.0（57域联合，14天）','timesfm30_long':'Google TimesFM3.0（57域联合，最多21天）','timesfm30_uni':'Google TimesFM3.0（单域，最多21天）','patchtstfm_long':'IBM Granite PatchTST-FM-r2 385M（最多21天，CPU）','local24':'四种子LightGBM残差与加权滞后回归各50%（开发D14／测试D22）','local7':'历史水平与日内形状平滑（水平半衰期3／5天，形状10天）'}
for H in [1,7]:
 oo=common.TEST24 if H==1 else common.TEST7;do=common.DEV24 if H==1 else common.DEV7;y,m=common.truth(d,oo,H);yd,md=common.truth(d,do,H)
 for method,score in s['scores'][str(H)].items():
  p=z[f'{method}_H{H}'];dp=z[f'dev_{method}_H{H}'];re=evaluate(common,d,oo,p,H);assert re==score;dv=evaluate(common,d,do,dp,H);assert dv==s['development'][str(H)][method]
  rows.append(dict(horizon_days=H,model=f'r44_{method}_H{H}',model_name=names[method],report_name=names[method],status='原项目开发冻结复现候选' if method=='hdsf' else'原项目成员独立复现对照',source=dict(path='round44/results/predictions.npz',array=f'{method}_H{H}',sha256=sha(Q/'predictions.npz')),prediction_digest=hashlib.sha256(p.astype('float64').tobytes()).hexdigest(),development_wape_original_grid=dv['wape'],**score))
  loq=np.array([np.quantile((yd-dp)[i][md[i]],.1) for i in range(57)]);hiq=np.array([np.quantile((yd-dp)[i][md[i]],.9) for i in range(57)]);lower=np.clip(p+loq[:,None,None],0,1);upper=np.clip(p+hiq[:,None,None],0,1);iv=interval_metrics(y,p,m,lower,upper);assert np.isfinite(lower).all() and np.isfinite(upper).all() and (lower<=upper).all();intervals.append(dict(horizon_days=H,method=names[method]+' 开发残差10/90经验80%区间',kind='附加经验校准，非HDSF原生分位数',**iv));np.savez_compressed(Q/f'interval_{method}_H{H}.npz',lower=lower,upper=upper,loq=loq,hiq=hiq)
allrows=old['rows']+rows;unique=[]
for H in [1,7]:
 bank={}
 for r in sorted([r for r in allrows if r['horizon_days']==H],key=lambda r:0 if r['status']=='当前综合推荐' else 1):
  key=r['prediction_digest']
  if key not in bank:bank[key]=dict(r,aliases=[r['model']])
  else:bank[key]['aliases'].append(r['model'])
 unique.extend(sorted(bank.values(),key=lambda r:r['wape']))
winners=[]
for H,task,fields in [(1,'24小时完整曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天完整曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天每日平均负荷',['daily_wape','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent'])]:
 for k in fields:
  rr=[r for r in unique if r['horizon_days']==H];value=(max if 'hit' in k else min)(r[k] for r in rr);winners.append(dict(task=task,metric=k,value=value,models='；'.join(r['model_name'] for r in rr if abs(r[k]-value)<1e-10),direction='越高越好' if 'hit' in k else'越低越好'))
boot={};delta={}
for H in [1,7]:
 cur=next(r for r in old['unique_rows'] if r['horizon_days']==H and r['status']=='当前综合推荐');src=cur['source'];prior=np.load(B/src['path'])[src['array']];p=z[f'hdsf_H{H}'];oo=common.TEST24 if H==1 else common.TEST7;y,m=common.truth(d,oo,H);a=(abs(y-prior)*m).sum(-1);b=(abs(y-p)*m).sum(-1);den=(abs(y)*m).sum(-1);rng=np.random.default_rng(20261008);indices=rng.integers(0,len(oo),(2000,len(oo)));gain=100*(a[:,indices].sum(-1)/den[:,indices].sum(-1)-b[:,indices].sum(-1)/den[:,indices].sum(-1)).mean(0);boot[str(H)]=dict(gain_pp=cur['wape']-s['scores'][str(H)]['hdsf']['wape'],paired_origin_exploratory_95_gain_pp=np.quantile(gain,[.025,.975]).tolist(),origins=len(oo),overlapping_windows=H==7,independent_confirmation=False,bootstrap_seed=20261008,draws=2000)
 delta[str(H)]=dict(current_model_name=cur['model_name'],current_model=cur['model'],metrics={k:dict(current=cur[k],hdsf=s['scores'][str(H)]['hdsf'][k],difference_hdsf_minus_current=s['scores'][str(H)]['hdsf'][k]-cur[k]) for k in ['wape','daily_wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent','absolute_bias_percent','peak_time_mae_minutes','cella5_wape','other52_wape']})
 # Export all historical rolling forecasts, with labels appended only after freezing.
 import pandas as pd
 rr=[];dr=[];lp=np.load(Q/f'interval_hdsf_H{H}.npz');lower=lp['lower'];upper=lp['upper']
 for j,o in enumerate(oo):
  timestamps=pd.date_range('2019-05-01',periods=31*96,freq='15min',tz='America/Los_Angeles')[o*96:(o+H)*96]
  for i,name in enumerate(d['names']):
   for k,t in enumerate(timestamps):rr.append(dict(origin_day=o,series_key=str(name),timestamp=t.isoformat(),power_normalized=float(p[i,j,k]),actual_normalized=float(y[i,j,k]),target_valid=bool(m[i,j,k]),lower80=float(lower[i,j,k]),upper80=float(upper[i,j,k]),method=names['hdsf'],historical_replay=True))
   for day in range(H):
    sl=slice(day*96,(day+1)*96);dr.append(dict(origin_day=o,target_day=o+day+1,series_key=str(name),power_daily_mean=float(p[i,j,sl].mean()),actual_daily_mean=float(y[i,j,sl].mean()),all96valid=bool(m[i,j,sl].all()),method=names['hdsf']))
 export(Q/f'HDSF_H{H}_历史滚动预测.csv',rr);export(Q/f'HDSF_H{H}_每日平均负荷.csv',dr)
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metric_version='9.28 v2 perPDU macro WAPE, full valid day daily metrics',reference_audit=read(Q/'metric_reference_audit.json'),source_commit=read(R/'docs/protocol.json')['commit'],rows=allrows,unique_rows=unique,new_rows=rows,winners=winners,intervals=old['intervals']+intervals,new_intervals=intervals,hdsf_summary=s,hdsf_science=read(Q/'science_verification.json'),bootstrap=boot,delta_vs_current=delta,old_deployment_retained=True,reproduction_candidate_not_test_selected=True,independent_confirmation=False,all_checks_passed=True)
save(Q/'hdsf_comparison.json',out);export(Q/'HDSF统一指标结果.csv',rows);export(Q/'HDSF与此前模型全部指标比较.csv',allrows);export(Q/'各任务单项最优结果_含HDSF.csv',winners);export(Q/'预测区间结果_含HDSF.csv',out['intervals']);print(json.dumps(dict(rows=len(allrows),new_rows=len(rows),unique=len(unique),hdsf_scores={str(H):s['scores'][str(H)]['hdsf'] for H in [1,7]}),ensure_ascii=False))
