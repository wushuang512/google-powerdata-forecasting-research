from pathlib import Path
import sys,json,hashlib,datetime,ast,pickle,csv,numpy as np
from types import SimpleNamespace
from contextlib import nullcontext
T=Path(sys.argv[1]);B=T/'experiment';Q=B/'round35/results';sys.path[:0]=[str(B/'round29/code'),str(B/'code')]
import common
from aligned_metrics import evaluate
d=common.load();checks=[];rows=[];intervals=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,value):
 checks.append(dict(check=name,passed=bool(value)));assert value,name
def prior(phase,H):
 with np.load(B/f'round{32 if H==1 else 31}/results/{phase}_predictions.npz') as z:return z[('TTM_head_sparsepeak50_strength0.1_H1' if H==1 else 'TTM_head_shape10_H7') if phase=='dev' else f'selected_H{H}'].copy()
def funcs(p,names,ns):
 nodes=[n for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names];check('actual function extraction '+p.name,len(nodes)==len(names));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns
def csvsave(p,data):
 keys=list(dict.fromkeys(k for r in data for k in r if k not in ['source','aliases','prediction_digest']));f=p.open('w',encoding='utf-8-sig',newline='');w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(data);f.close()
for nr in [34,35,36]:
 R=B/f'round{nr}';res=R/'results';s=read(res/'summary.json');sel=read(res/'selection.json');check(f'R{nr} freeze digest',sha(res/'selection.json')==(res/'selection.sha256').read_text());check(f'R{nr} DEV-before-TEST freeze',sel['no_test_used'] and sel['protocol_sha256']==sha(R/'docs/protocol.json') and sel['dev_predictions_sha256']==sha(res/'dev_predictions.npz') and (res/'selection.json').stat().st_mtime<(res/'test_predictions.npz').stat().st_mtime);check(f'R{nr} pipeline complete',read(res/'resource_status.json')['stage']=='completed')
 if nr==34:
  ns=funcs(R/'code/risk_curve_experiment.py',['features','uncertainty','reconstruct'],dict(np=np,d=d));models=pickle.load((R/'models/heads_cut22.pkl').open('rb'))
 elif nr==35:
  ns=funcs(R/'code/moirai_curve_experiment.py',['compose'],dict(np=np,d=d));check('official adapter smoke matched',read(res/'adapter_verification.json')['all_passed'] and read(res/'adapter_verification.json')['full_official_forecast_vs_adapter_max_difference']==0)
 else:
  ns=funcs(B/'round34/code/risk_curve_experiment.py',['uncertainty'],dict(np=np,d=d));ns=funcs(R/'code/moirai_peak_coordination.py',['reconstruct','eligible'],ns);check('R36 nativeDEV binding',sel['native_dev_sha256']==sha(B/'round35/results/native_dev.npz'))
 with np.load(res/'test_predictions.npz') as zz,np.load(res/'dev_predictions.npz') as dd:
  for H in [1,7]:
   oo=list(range(22,31)) if H==1 else[22,23,24];dev=[14,16,18,20] if H==1 else[14,15];sh=sel['horizons'][str(H)];cfg=(sh['selected'] or sh['diagnostic'])['config'];base=prior('test',H)
   if nr==34:
    fresh=ns['reconstruct'](base,models,cfg,oo);altered=dict(d);altered['y']=d['y'].copy();altered['valid']=d['valid'].copy();altered['y'][:,22:]=999;altered['valid'][:,22:]=False;ns['d']=altered;changed=ns['reconstruct'](base[:,0:1],models,cfg,[22]);ns['d']=d;check(f'R34 H{H} actual future perturbation invariant',np.array_equal(changed,fresh[:,0:1]))
   elif nr==35:
    with np.load(res/'native_test.npz') as nz:raw=np.stack([nz[f'{cfg["family"]}_H{H}_o{o}'][:,4] for o in oo],1)
    fresh=ns['compose'](base,raw,cfg,oo)
   else:
    family=read(R/'docs/protocol.json')['families'][str(H)]
    with np.load(B/'round35/results/native_test.npz') as nz:raw=np.stack([nz[f'{family}_H{H}_o{o}'][:,4] for o in oo],1)
    fresh=ns['reconstruct'](base,raw,cfg,oo);altered=dict(d);altered['y']=d['y'].copy();altered['y'][:,22:]=999;ns['d']=altered;changed=ns['reconstruct'](base[:,0:1],raw[:,0:1],cfg,[22]);ns['d']=d;check(f'R36 H{H} future perturbation invariant',np.array_equal(changed,fresh[:,0:1]))
   check(f'R{nr} H{H} exact recipe replay',np.array_equal(fresh,zz[f'diagnostic_H{H}']));check(f'R{nr} H{H} freeze retain/adopt',np.array_equal(zz[f'selected_H{H}'],fresh if sh['selected'] else base))
   for typ in ['diagnostic','selected']:
    p=zz[f'{typ}_H{H}'];sc=evaluate(common,d,oo,p,H);check(f'R{nr} H{H} {typ} all metrics replay',all(abs(sc[k]-s['scores'][str(H)][typ][k])<1e-10 if isinstance(sc[k],(int,float)) else sc[k]==s['scores'][str(H)][typ][k] for k in sc));dp=dd[f'{cfg["id"]}_H{H}'] if typ=='diagnostic' or sh['selected'] else prior('dev',H);dm=evaluate(common,d,dev,dp,H)
    name=('历史日最大值和历史不确定性 加完整日均保持峰值曲线重建' if nr==34 and cfg['head']=='history' else f'HGB日均峰值头和历史不确定性 {cfg["id"]}') if nr==34 else(f'Salesforce Moirai 2.0 R small {cfg["family"]} {cfg["mode"]}' if nr==35 else f'Salesforce Moirai 2.0 R small {family} {cfg["alpha"]*100:g}%＋'+('Chronos-2 TiRex-2 TimesFM3 Ridge TinyTimeMixer' if H==1 else'TimesFM3 Ridge1000 HGB TinyTimeMixer')+f' {(1-cfg["alpha"])*100:g}% {cfg["total"]} 历史峰值{cfg["peak_beta"]*100:g}%')
    status='开发冻结研究替代' if sh['selected'] else'未采用诊断'
    if typ=='selected' and sh['selected'] is None:name=('Chronos-2＋TiRex-2＋TimesFM3 Ridge及末层微调＋TinyTimeMixer峰值10%' if H==1 else'TimesFM3＋Ridge1000＋HGB时点稀疏峰值＋TinyTimeMixer形状10%')+f' 第{nr}轮冻结保留';status='冻结保留当前推荐'
    if nr==36:status='当前综合推荐' if H==7 and typ=='selected' else'开发冻结测试未通过部署' if H==1 else'开发冻结研究替代'
    label=name if nr!=36 else('Salesforce Moirai2 单变量50%\nChronos-2＋TiRex-2＋TimesFM3＋TinyTimeMixer50%' if H==1 else'Salesforce Moirai2 同cell14天10%\nTimesFM3＋Ridge＋HGB＋TinyTimeMixer90%\n全57预测日均协调')
    rows.append(dict(horizon_days=H,model=f'r{nr}_{typ}_H{H}',model_name=name,report_name=label,status=status,source=dict(path=f'round{nr}/results/test_predictions.npz',array=f'{typ}_H{H}',sha256=sha(res/'test_predictions.npz')),prediction_digest=hashlib.sha256(p.astype('float64').tobytes()).hexdigest(),development_wape_original_grid=dm['wape'],**sc))
    if typ=='diagnostic':
     yd,md=common.truth(d,dev,H);yt,mt=common.truth(d,oo,H);qlo=np.array([np.quantile((yd-dp)[i][md[i]],.1) for i in range(57)]);qhi=np.array([np.quantile((yd-dp)[i][md[i]],.9) for i in range(57)]);lo=np.clip(p+qlo[:,None,None],0,1);hi=np.clip(p+qhi[:,None,None],0,1);wink=hi-lo+10*np.maximum(lo-yt,0)+10*np.maximum(yt-hi,0)
     intervals.append(dict(horizon_days=H,method=f'R{nr} '+name+' 开发残差10/90经验区间',nominal_percent=80,coverage_percent=float(100*((yt>=lo)&(yt<=hi))[mt].mean()),mean_width_normalized=float((hi-lo)[mt].mean()),winkler80=float(wink[mt].mean()),kind='Same fixed DEV-only residual10/90 protocol as earlier new architectures; no interval parameter search or TEST selection. Exploratory, not independent coverage.'))
     np.savez_compressed(res/f'interval_diagnostic_H{H}.npz',lower=lo,upper=hi,loq=qlo,hiq=qhi)
 with np.load(res/'deployment.npz') as zz:
  ex=res/'exports';ex.mkdir(exist_ok=True)
  for H in [1,7]:
   for typ in ['diagnostic','selected']:
    p=zz[f'{typ}_H{H}'];check(f'R{nr} H{H} {typ} June replay range',p.shape==(57,1,H*96) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1);pts=[];days=[];ts=np.datetime64('2019-06-01T00:00')+np.arange(H*96)*np.timedelta64(15,'m')
    iv=np.load(res/f'interval_diagnostic_H{H}.npz') if typ=='diagnostic' or nr==36 else None
    for i in range(57):
     for j in range(H*96):
      rec=dict(pdu=str(d['names'][i]),cell=str(d['cells'][i]),timestamp_local_PT=str(ts[j]),forecast_normalized_power=float(p[i,0,j]),historical_replay=True,current_live_forecast=False)
      if iv is not None:rec.update(lower80_empirical=float(np.clip(p[i,0,j]+iv['loq'][i],0,1)),upper80_empirical=float(np.clip(p[i,0,j]+iv['hiq'][i],0,1)),interval_source='Fixed DEV residual10/90; no TEST calibration')
      pts.append(rec)
     for j in range(H):days.append(dict(pdu=str(d['names'][i]),cell=str(d['cells'][i]),date_PT=f'2019-06-{j+1:02d}',daily_mean_normalized_power=float(p[i,0,j*96:(j+1)*96].mean()),historical_replay=True))
    csvsave(ex/f'{typ}_{H}day_curve.csv',pts);csvsave(ex/f'{typ}_{H}day_daily_mean.csv',days)
    if iv is not None:iv.close()
release=read(B/'round36/results/summary.json');v=release['scores']['7'];gain5=v['selected']['cella5_wape']-v['prior']['cella5_wape'];check('R36a5 exception is documentedtinybutexceedsoriginal',.005<gain5<.01);tolerance={'mae':v['prior']['mae']*.002,'rmse':v['prior']['rmse']*.002,'p95_ape':.01,'hit2_percent':.05,'hit5_percent':.05,'other52_wape':.005,'absolute_bias_percent':.005,'peak_time_mae_minutes':5,'daily_mae':v['prior']['daily_mae']*.005,'daily_rmse':v['prior']['daily_rmse']*.005}
check('R36H7 other registered secondary bounds passed',all(v['selected'][k]>=v['prior'][k]-x if k.startswith('hit') else v['selected'][k]<=v['prior'][k]+x for k,x in tolerance.items()));check('R36H7 primary curve/daily improvesandpeakclose',v['selected']['wape']<v['prior']['wape']-.001 and v['selected']['daily_wape']<v['prior']['daily_wape'] and v['selected']['peak_mape']<=v['prior']['peak_mape']+.1);check('R36H1 rejectedcurve deterioration',release['scores']['1']['selected']['wape']>release['scores']['1']['prior']['wape'])
decision=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),adopted24='round32',adopted7='round36',dev_selection_unchanged=True,test_grid_search=False,original_all_tolerances_passed=False,a5_wape_increase_pp=gain5,original_a5_tolerance_pp=.005,excess_pp=gain5-.005,authorization='User explicitly accepts a method with leadingimportantmetrics andverytiny secondary gaps. This is a transparent deployment judgement after evaluation, not a rewritten frozen selection orindependent scientific proof.',r35_diagnostic_not_deployed_despite_lower_test_wape=True);(B/'round36/results/deployment_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
# Actual native wrapper with a deterministic recording stand-in isolates temporal access.
class Output:
 def __init__(self,a):self.a=a
 def numpy(self):return self.a
 def cpu(self):return self
class RecordingModel:
 def __init__(self,H,dim):self.H=H;self.hparams=SimpleNamespace(target_dim=dim);self.inputs=[]
 def __call__(self,**kw):
  self.inputs.append(tuple(np.array(kw[k],copy=True) for k in ['past_target','past_observed_target','past_is_pad']));x=kw['past_target'];v=x.mean(1);out=np.tile(v[:,None,None,:],(1,9,self.H*96,1));return Output(out[...,0] if self.hparams.target_dim==1 else out)
flat=d['y'].reshape(57,-1).copy();valid=d['valid'].reshape(57,-1).copy();ft=SimpleNamespace(tensor=lambda a,**kw:np.array(a,dtype=kw.get('dtype')),float32=np.float32,bool=bool,zeros=lambda a,**kw:np.zeros(a,dtype=kw.get('dtype')),inference_mode=nullcontext);ns=funcs(B/'round35/code/moirai_curve_experiment.py',['native_predict'],dict(np=np,torch=ft,flat=flat,valid=valid,DEVICE='cpu'))
for dim in [1,5]:
 model=RecordingModel(1,dim);group=None if dim==1 else[np.arange(5)];a=ns['native_predict'](model,14,672,group,1);before=model.inputs.copy();ns['flat']=flat.copy();ns['valid']=valid.copy();ns['flat'][:,14*96:]=999;ns['valid'][:,14*96:]=False;model.inputs=[];b=ns['native_predict'](model,14,672,group,1);check(f'Native dim{dim} future perturbation leaves actual wrapper inputs/predictions identical',np.array_equal(a,b) and all(np.array_equal(x,y) for aa,bb in zip(before,model.inputs) for x,y in zip(aa,bb)));ns['flat']=flat;ns['valid']=valid
# Check each native cache is explicitly bound to history, code, protocol and weights.
R=B/'round35';ns=funcs(R/'code/moirai_curve_experiment.py',['cache_signature'],dict(np=np,hashlib=hashlib,json=json,torch=SimpleNamespace(__version__=read(Q/'adapter_verification.json')['torch_version']),R=R,C=R/'code',B=B,d=d,sha=sha,flat=flat,valid=valid))
for file in sorted((Q/'native_cache_cuda').glob('*.npz')):
 phase,family,hh,oo=file.stem.split('_');H=int(hh[1:]);o=int(oo[1:])
 with np.load(file) as z:check('Native cache binding '+file.name,str(z['signature'])==ns['cache_signature'](family,H,o) and np.isfinite(z['quantiles']).all())
old=read(B/'round32/results/continued_comparison.json');allrows=old['rows']+rows;uni=[]
for r in allrows:
 if r['horizon_days']==7 and r['status']=='当前综合推荐' and r['model']!='r36_selected_H7':r['status']='此前综合推荐'
for H in [1,7]:
 lookup={}
 for r in sorted([x for x in allrows if x['horizon_days']==H],key=lambda r:0 if r['status']=='当前综合推荐' else 1):
  key=r['prediction_digest']
  if key not in lookup:lookup[key]=dict(r,aliases=[r['model']])
  else:lookup[key]['aliases'].append(r['model'])
 uni+=sorted(lookup.values(),key=lambda r:r['wape'])
winners=[]
for H,task,fields in [(1,'24小时曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天每日均值',['daily_wape','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent'])]:
 for k in fields:
  rr=[r for r in uni if r['horizon_days']==H];v=(max if 'hit' in k else min)(r[k] for r in rr);winners.append(dict(task=task,metric=k,value=v,models='；'.join(r['model_name'] for r in rr if abs(r[k]-v)<1e-10),direction='越高越好' if 'hit' in k else'越低越好'))
with np.load(B/'round36/results/test_predictions.npz') as z:p=z['selected_H7'].copy()
b=prior('test',7);y,m=common.truth(d,[22,23,24],7);rng=np.random.default_rng(20261007);boot=[]
for rep in range(3000):
 ix=rng.integers(0,3,3);den=(abs(y[:,ix])*m[:,ix]).sum((1,2));boot.append(float(np.mean(100*((abs(b[:,ix]-y[:,ix])-abs(p[:,ix]-y[:,ix]))*m[:,ix]).sum((1,2))/den)))
gain=evaluate(common,d,[22,23,24],b,7)['wape']-evaluate(common,d,[22,23,24],p,7)['wape'];explore=dict(paired_origin_bootstrap_95_gain_pp=np.quantile(boot,[.025,.975]).tolist(),replicates=3000,independent=False,note='Only3overlappingoriginwindows, sameMayTEST repeatedlyviewed; intervalnotindependent significance proof.',gain_pp=gain)
out=dict(old,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=allrows,unique_rows=uni,winners=winners,intervals=old['intervals']+intervals,curves_priority_rounds={str(n):read(B/f'round{n}/results/summary.json') for n in [34,35,36]},round36_bootstrap=explore,all_checks_passed=all(c['passed'] for c in checks),evaluation_priority='24h and7d curveWAPE priority,7d dailyWAPE,peak close allowable. This is a priority,not a proven allwinner.',recommendation=dict(recommended24='round32',recommended7='round36',previous7='round31',new_universal_winner=False,exception='R36a5TESTWAPE+0.008071987pp exceeds originallyregistered0.005pp by0.003071987pp. Accepted underuser preference allowingtinysecondaryregrets; preserveoriginalthreshold/freeze/releasepolicy anddonotclaimallgatespassed. NoalternateTESTcandidatechosen.'))
(Q/'curves_priority_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');(Q/'independent_verification.json').write_text(json.dumps(dict(all_passed=True,check_count=len(checks),checks=checks,created_utc=out['created_utc'],raw_rows=len(allrows),unique_predictions=len(uni),future_check_scope='Actual wrapper input slicing with recording model; does not establish external independence or absence of adaptive overfit.'),ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['模型各项指标完整结果.csv','9月28日统一口径模型复算.csv','9月28日指标完整模型对照.csv']:csvsave(T/name,allrows)
csvsave(T/'各任务单项最优结果.csv',winners)
for name in ['模型预测区间结果.csv','9月28日指标预测区间对照.csv']:csvsave(T/name,out['intervals'])
print(json.dumps(dict(all_passed=True,checks=len(checks),rows=len(allrows),unique=len(uni)),ensure_ascii=False))
