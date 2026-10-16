from pathlib import Path
import sys,json,hashlib,datetime,ast,csv,numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round32';Q=R/'results';sys.path.insert(0,str(B/'round29/code'));sys.path.insert(0,str(B/'code'));import common;from aligned_metrics import evaluate
d=common.load();checks=[];rows=[];ivs=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(n,v):checks.append(dict(check=n,passed=bool(v)));assert v,n
def csvsave(path,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r if k not in ['source','aliases','prediction_digest']));f=path.open('w',encoding='utf-8-sig',newline='');w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rr);f.close()
for nr in [31,32]:
 rr=B/f'round{nr}';s=read(rr/'results/summary.json');sel=s['selection'];check(f'R{nr} frozen andbeforeTEST',sha(rr/'results/selection.json')==(rr/'results/selection.sha256').read_text() and (rr/'results/selection.json').stat().st_mtime<(rr/'results/test_predictions.npz').stat().st_mtime);check(f'R{nr} protocol andmetricSHA',sel['protocol_sha256']==sha(rr/'docs/protocol.json') and read(rr/'docs/protocol.json')['metric_version_sha256']==sha(B/'round29/code/aligned_metrics.py'));check(f'R{nr} guard completed',read(rr/'results/resource_status.json')['stage']=='completed')
 src=ast.parse((rr/'code/ttm_curve_reconstruction.py').read_text(encoding='utf-8'));nodes=[v for v in src.body if isinstance(v,ast.FunctionDef) and v.name in ['reconstruct','project']];ns=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual reconstruction>','exec'),ns)
 with np.load(rr/'results/test_predictions.npz') as pp,np.load(rr/'results/dev_predictions.npz') as dev:
  for H in [1,7]:
   orig=list(range(22,31)) if H==1 else[22,23,24];oo=[14,16,18,20] if H==1 else[14,15];sh=sel['horizons'][str(H)];it=sh['selected'] or sh['diagnostic'];source=f'TTM512 multitask {it["expert"]}_H{H}'
   with np.load(B/'round30/results/test_predictions.npz') as tt:ttm=tt[source]
   with np.load(B/'round26/results/test_predictions.npz') as base:z=base[f'selected{H}']
   kwargs={} if nr==31 else dict(strength=it['strength']);fresh=ns['reconstruct'](z,ttm,it['kind'],**kwargs);check(f'R{nr} H{H} actual replay',np.array_equal(fresh,pp[f'diagnostic_H{H}']));new=pp[f'selected_H{H}'];diag=pp[f'diagnostic_H{H}'];check(f'R{nr} H{H} selectionchoice',np.array_equal(new,diag) if sh['selected'] else np.array_equal(new,z) if nr==31 or H==1 else np.array_equal(new,np.load(B/'round31/results/test_predictions.npz')['selected_H7']))
   for typ,p in [('diagnostic',diag),('selected',new)]:
    metrics=evaluate(common,d,orig,p,H);check(f'R{nr} H{H} {typ} metricrecompute',all(abs(metrics[k]-s['scores'][str(H)][typ][k])<1e-11 for k in ['wape','daily_wape','peak_mape','p95_ape','hit2_percent','hit5_percent','rmse']));dv=dev[f'{it["model"]}_H{H}'] if typ=='diagnostic' or sh['selected'] else dev[f'prior_H{H}'];dm=evaluate(common,d,oo,dv,H);key=f'r{nr}_{typ}_H{H}';strength=it.get('strength',.1 if it['kind']=='shape10' else .5);name=('Chronos-2＋TiRex-2＋TimesFM3 Ridge与末层微调' if H==1 else'TimesFM3＋Ridge1000＋HGB时点稀疏峰值')+f' TinyTimeMixer多任务{it["expert"]} '+('零均值形状' if it['kind']=='shape10' else'稀疏峰值')+f'{100*strength:g}%';label=('Chronos-2＋TiRex-2＋TimesFM3\nRidge及末层微调＋TinyTimeMixer峰值' if H==1 else'TimesFM3＋Ridge＋HGB时点稀疏峰值\nTinyTimeMixer日均保持形状')+f'{100*strength:g}%';status='当前综合推荐' if (nr,H,typ) in [(32,1,'selected'),(31,7,'selected')] else'开发冻结研究替代' if sh['selected'] else'未采用诊断'
    if sh['selected'] is None and typ=='selected':
     name='Chronos-2＋TiRex-2＋TimesFM3 Ridge与末层微调 第31轮冻结保留' if H==1 else'TimesFM3＋Ridge＋HGB时点稀疏峰值 TinyTimeMixer零均值形状10% 第32轮冻结保留'
     label='Chronos-2＋TiRex-2＋TimesFM3\nRidge与末层微调 冻结保留' if H==1 else'TimesFM3＋Ridge＋HGB时点稀疏峰值\nTinyTimeMixer日均保持形状10% 冻结保留';status='已评估对照'
    rows.append(dict(horizon_days=H,model=key,model_name=name,report_name=label,status=status,source=dict(path=f'round{nr}/results/test_predictions.npz',array=f'{typ}_H{H}',sha256=sha(rr/'results/test_predictions.npz')),prediction_digest=hashlib.sha256(p.astype('float64').tobytes()).hexdigest(),development_wape_original_grid=dm['wape'],**metrics))
   if H==7 and it['kind']=='shape10':check(f'R{nr} unchanged daily total',np.max(abs(diag.reshape(57,-1,96).sum(-1)-z.reshape(57,-1,96).sum(-1)))<1e-9)
 with np.load(rr/'results/deployment.npz') as dep:
  for H in [1,7]:
   for typ in ['diagnostic','selected']:
    p=dep[f'{typ}_H{H}'];check(f'R{nr} {typ}H{H} deploy',p.shape==(57,1,H*96) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1);ex=rr/'results/exports';ex.mkdir(exist_ok=True);points=[];days=[];ts=np.datetime64('2019-06-01T00:00')+np.arange(H*96)*np.timedelta64(15,'m')
    for i in range(57):
     for j in range(H*96):points.append(dict(pdu=str(d['names'][i]),cell=str(d['cells'][i]),timestamp_local_PT=str(ts[j]),forecast_normalized_power=float(p[i,0,j]),historical_replay=True,current_live_forecast=False))
     for j in range(H):days.append(dict(pdu=str(d['names'][i]),cell=str(d['cells'][i]),date_PT=f'2019-06-{j+1:02d}',daily_mean_normalized_power=float(p[i,0,j*96:(j+1)*96].mean()),historical_replay=True))
    csvsave(ex/f'{typ}_{H}day_curve.csv',points);csvsave(ex/f'{typ}_{H}day_daily_mean.csv',days)
   # Empirical residual intervals center the actually frozen selected curve.
   sh=sel['horizons'][str(H)];it=sh['selected'] or sh['diagnostic'];yp,mm=common.truth(d,[14,16,18,20] if H==1 else[14,15],H)
   with np.load(rr/'results/dev_predictions.npz') as zz:cp=zz[f'{it["model"]}_H{H}'] if sh['selected'] else zz[f'prior_H{H}']
   loq=np.array([np.quantile((yp-cp)[i][mm[i]],.1) for i in range(57)]);hiq=np.array([np.quantile((yp-cp)[i][mm[i]],.9) for i in range(57)])
   with np.load(rr/'results/test_predictions.npz') as zz:p=zz[f'selected_H{H}']
   y,m=common.truth(d,list(range(22,31)) if H==1 else[22,23,24],H);lo=np.clip(p+loq[:,None,None],0,1);hi=np.clip(p+hiq[:,None,None],0,1);width=hi-lo;wink=width+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0);ivs.append(dict(horizon_days=H,method=f'R{nr} 开发冻结TinyTimeMixer指导重建经验80%区间',nominal_percent=80,coverage_percent=float(100*((y>=lo)&(y<=hi))[m].mean()),mean_width_normalized=float(width[m].mean()),winkler80=float(wink[m].mean()),kind='DEV-only residual10/90 quantiles, no independent coverage guarantee'));np.savez_compressed(rr/f'results/interval{H}.npz',lower=lo,upper=hi,loq=loq,hiq=hiq)
   ex=rr/'results/exports';path=ex/f'selected_{H}day_curve.csv'
   with path.open(encoding='utf-8-sig') as ff:rrr=list(csv.DictReader(ff))
   for index,rec in enumerate(rrr):i=index//(H*96);v=float(rec['forecast_normalized_power']);rec.update(lower80_empirical=float(np.clip(v+loq[i],0,1)),upper80_empirical=float(np.clip(v+hiq[i],0,1)),interval_source='DEV residual10/90 only')
   csvsave(path,rrr)
old=read(B/'round30/results/aligned_comparison.json');allrows=old['rows']+rows
for r in allrows:
 if r in old['rows'] and r['status']=='当前综合推荐':r['status']='此前综合推荐'
 # R32 7day selected is an alias of current R31; identify it as retained current.
 if r['model']=='r32_selected_H7':r['status']='当前综合推荐'
uni=[]
for H in [1,7]:
 dd={}
 for r in sorted([x for x in allrows if x['horizon_days']==H],key=lambda r:0 if r['status']=='当前综合推荐' else 1):
  key=r['prediction_digest']
  if key not in dd:dd[key]=dict(r,aliases=[r['model']])
  else:
   dd[key]['aliases'].append(r['model'])
   if dd[key].get('development_wape_original_grid') is None and r.get('development_wape_original_grid') is not None:dd[key]['development_wape_original_grid']=r['development_wape_original_grid']
 uni+=sorted(dd.values(),key=lambda r:r['wape'])
win=[]
for H,task,fields in [(1,'24小时曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天每日均值',['daily_wape','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent'])]:
 for k in fields:
  rr=[r for r in uni if r['horizon_days']==H];higher=k.startswith('hit') or k.startswith('daily_hit');val=(max if higher else min)(r[k] for r in rr);win.append(dict(task=task,metric=k,value=val,models='；'.join(r['model_name'] for r in rr if abs(r[k]-val)<1e-10),direction='越高越好' if higher else'越低越好'))
boot={};rng=np.random.default_rng(20261007)
for H,nr in [(1,32),(7,31)]:
 with np.load(B/f'round{nr}/results/test_predictions.npz') as zz:p=zz[f'selected_H{H}']
 with np.load(B/'round26/results/test_predictions.npz') as zz:b=zz[f'selected{H}']
 y,m=common.truth(d,list(range(22,31)) if H==1 else[22,23,24],H);v=[]
 for rep in range(3000):
  idx=rng.integers(0,len(y[0]),len(y[0]));den=(abs(y[:,idx])*m[:,idx]).sum((1,2));v.append(float(np.mean(100*((abs(b[:,idx]-y[:,idx])-abs(p[:,idx]-y[:,idx]))*m[:,idx]).sum((1,2))/den)))
 boot[str(H)]=dict(paired_origin_bootstrap_95_gain_pp=np.quantile(v,[.025,.975]).tolist(),replicates=3000,independent=False,note='Exploratory only; same month repeatedly inspected; 7day origins overlap and3 are not independent samples.')
out=dict(old,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=allrows,unique_rows=uni,winners=win,intervals=old['intervals']+ivs,recommendation=dict(recommended24='round32',recommended7='round31',previous24='round8b',previous7='round26',new_universal_winner=False,criterion='DEV frozen >=2 categories,14 minor tradeoff bounds, matched TEST tradeoffs verified; user previously accepts tiny regrets'),continuation_rounds={str(n):read(B/f'round{n}/results/summary.json') for n in [31,32]},bootstrap=boot)
csvsave(T/'模型各项指标完整结果.csv',allrows);csvsave(T/'9月28日统一口径模型复算.csv',allrows);csvsave(T/'各任务单项最优结果.csv',win);csvsave(T/'模型预测区间结果.csv',out['intervals']);(Q/'continued_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');(Q/'independent_verification.json').write_text(json.dumps(dict(all_passed=all(c['passed'] for c in checks),checks=checks,check_count=len(checks),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bootstrap=boot),ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(all_passed=True,checks=len(checks),rows=len(allrows),unique=len(uni),bootstrap=boot),ensure_ascii=False))
