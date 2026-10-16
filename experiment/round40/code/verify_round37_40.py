from pathlib import Path
import sys,json,hashlib,ast,csv,datetime,contextlib,types,gc,numpy as np
T=Path(sys.argv[1]);B=T/'experiment';OUT=B/'round40/results';sys.path[:0]=[str(B/'code'),str(B/'round29/code')];import common;from aligned_metrics import evaluate
d=common.load();checks=[];rows=[];intervals=[];releases={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(n,v):checks.append(dict(check=n,passed=bool(v)));assert v,n
def functions(p,names,ns):
 nodes=[n for n in ast.parse(Path(p).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names];check('actual source functions '+Path(p).name,len(nodes)==len(names));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns
def prior(phase,H):
 with np.load(B/f'round{32 if H==1 else 36}'/'results'/('deployment.npz' if phase=='deploy' else f'{phase}_predictions.npz')) as z:return z[('TTM_head_sparsepeak50_strength0.1_H1' if H==1 else'a0.1_global_daily_peak0_H7') if phase=='dev' else f'selected_H{H}'].copy()
def export(p,data):
 keys=list(dict.fromkeys(k for r in data for k in r if k not in ['source','aliases','prediction_digest']));f=Path(p).open('w',encoding='utf-8-sig',newline='');w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(data);f.close()
class Tensor:
 def __init__(self,v):self.v=np.asarray(v)
 @property
 def shape(self):return self.v.shape
 def __getitem__(self,k):return Tensor(self.v[k])
 def cpu(self):return self
 def numpy(self):return self.v
class ArrayContext:
 def __init__(self,a):self.a=a
 def __enter__(self):return self.a
 def __exit__(self,*args):pass
class NPProxy:
 def __init__(self,o):self.o=o
 def __getattr__(self,k):return getattr(np,k)
 def load(self,p,*args,**kwargs):
  if Path(p)==T/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz':
   with np.load(p) as z:a={k:z[k].copy() for k in z.files}
   a['aggregate'].reshape(57,-1,6)[:,self.o*96:,0:2]+=123
   return ArrayContext(a)
  return np.load(p,*args,**kwargs)
def boundary_check(nr,filename,proto):
 flat=d['y'].reshape(57,-1).copy();valid=d['valid'].reshape(57,-1).copy();o=22;t=o*96;captures=[]
 fake=types.SimpleNamespace(float32=np.float32,bool=bool,tensor=lambda v,**kw:Tensor(v),zeros=lambda s,**kw:Tensor(np.zeros(s)),inference_mode=contextlib.nullcontext,cuda=types.SimpleNamespace(empty_cache=lambda:None))
 def model(ctx,L,dim,*args):
  def run(x,obs,pad):
   captures.append((x.v.copy(),obs.v.copy()))
   v=np.mean(x.v,axis=1);v=.5+.00001*v
   p=np.broadcast_to(v[:,None,None,:],(len(v),9,L,dim)).copy()
   return Tensor(p[...,0] if dim==1 else p)
  return run
 names=['hist','infer_one']+(['profile'] if nr==38 else['correlation_groups'] if nr==39 else['usage_channels'] if nr==40 else[])
 ns=functions(B/f'round{nr}/code'/filename,names,dict(np=np,flat=flat,valid=valid,d=d,model=model,torch=fake,DEV='audit recording model',gc=gc,hashlib=hashlib,read=read,sha=sha,save=lambda *a:None,T=T))
 for family in proto['families']:
  captures.clear();p0=ns['infer_one'](family,7,o,14);before=[(a.copy(),b.copy()) for a,b in captures]
  flat[:,t:]+=777;valid[:,t:]=~valid[:,t:];ns['np']=NPProxy(o) if nr==40 else np
  captures.clear();p1=ns['infer_one'](family,7,o,14)
  check(f'R{nr} {family["id"]} actual inputwrapper future perturbation',np.array_equal(p0,p1) and len(before)==len(captures) and all(np.array_equal(a,c) and np.array_equal(b,e) for (a,b),(c,e) in zip(before,captures)))
  flat[:,t:]-=777;valid[:,t:]=~valid[:,t:];ns['np']=np
 check(f'R{nr} boundaryscope declared',True)
for nr,filename in [(37,'multires_moirai.py'),(38,'native_residual_moirai.py'),(39,'correlation_moirai.py'),(40,'usage_moirai.py')]:
 R=B/f'round{nr}';Q=R/'results';proto=read(R/'docs/protocol.json');sel=read(Q/'selection.json');summary=read(Q/'summary.json');check(f'R{nr} freeze hashandDEVonly',sha(Q/'selection.json')==(Q/'selection.sha256').read_text() and sel['no_test_used']);check(f'R{nr} protocolDEVbinding',sel['protocol_sha256']==sha(R/'docs/protocol.json') and sel['native_dev_sha256']==sha(Q/'native_dev.npz') and sel['dev_predictions_sha256']==sha(Q/'dev_predictions.npz'));check(f'R{nr} freeze precedesTEST',(Q/'selection.json').stat().st_mtime<(Q/'test_predictions.npz').stat().st_mtime);check(f'R{nr} completed',read(Q/'resource_status.json')['stage']=='completed');check(f'R{nr} metricandweightsimmutable',proto['metric_sha256']==sha(B/'round29/code/aligned_metrics.py') and proto['source_weights_sha256']==sha(B/'round35/models/moirai2/model.safetensors'))
 boundary_check(nr,filename,proto)
 flat=d['y'].reshape(57,-1);valid=d['valid'].reshape(57,-1)
 for phase in ['dev','test','deploy']:
  requests=[(f,H,o) for f in proto['families'] for H,oo in [(1,[14,16,18,20]),(7,[14,15])] for o in oo] if phase=='dev' else [(sel['horizons'][str(H)]['family'],H,o) for H,oo in ([(1,list(range(22,31))),(7,[22,23,24])] if phase=='test' else [(1,[31]),(7,[31])]) for o in oo]
  with np.load(Q/f'native_{phase}.npz') as bank:
   good=True
   for f,H,o in requests:
    cut=14 if phase=='dev' else 22;key=f'{f["id"]}_H{H}_o{o}';weight=sha(R/f'models/{f["variant"]}_rate{f["rate"]}_cut{cut}.pt') if f['variant']!='base' else'base';extra=sha(T/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz') if nr==40 else''
    fingerprint=hashlib.sha256((extra+sha(R/'docs/protocol.json')+sha(R/'code'/filename)+sha(B/'round35/code/moirai_native_adapter.py')+sha(B/'round35/models/moirai2/model.safetensors')+weight).encode()+np.ascontiguousarray(flat[:,(o-f['context_days'])*96:o*96]).tobytes()+np.ascontiguousarray(valid[:,(o-f['context_days'])*96:o*96]).tobytes()).hexdigest()
    with np.load(Q/'native_cache'/f'{phase}_{key}.npz') as z:good &= str(z['signature'])==fingerprint and np.array_equal(z['quantiles'],bank[key]) and z['quantiles'].shape==(57,9,H*96//f['rate'])
   check(f'R{nr} {phase} allcache source script weight input signatures',good)
 if nr in [37,38]:
  check(f'R{nr} zeroadapter equivalence',read(Q/'adapter_verification.json')['zero_adapter_equals_original'])
  for training in Q.glob('train_*_cut*.json'):
   meta=read(training);cut=int(training.stem.split('_cut')[-1]);name=training.stem.removeprefix('train_')+'.pt';check(f'R{nr} '+training.name+' weights/train cutoff',meta['training_labels_end_exclusive']==cut*96 and meta['backbone_frozen'] and meta['epochs']==4 and meta['weights_sha256']==sha(R/'models'/name))
  for fp in Q.glob('features_*_cut*.npz'):
   cut=int(fp.stem.split('_cut')[-1]);rate=int(fp.stem.split('_rate')[-1].split('_')[0])
   with np.load(fp) as z:
    ids=z['pdu'];times=z['time'];mask=z['mask'];check(f'R{nr} {fp.name} labels truly beforecut',mask.any() and times[mask].min()*rate>=7*96 and (times[mask].max()+1)*rate<=cut*96);count=np.zeros((57,cut*96//rate),int)
    for i in range(len(ids)):np.add.at(count[ids[i]],times[i,mask[i]],1)
    check(f'R{nr} {fp.name} inverseoccurrenceweights',all(np.allclose(z['weight'][i,mask[i]],1/count[ids[i],times[i,mask[i]]],rtol=1e-7,atol=1e-9) for i in range(len(ids))))
    if nr==38:check(f'R38 {fp.name} positivePowernorm',np.isfinite(z['norm']).all() and(z['norm']>0).all())
 if nr==38:check('R38 unmodified nativecontrols match',read(Q/'native_control_regression.json')['all_passed'])
 if nr==39:
  for cut in [14,22]:
   g=read(Q/f'correlation_groups_cut{cut}.json');check(f'correlation groups{cut} train-onlyandbalanced',g['data_end_exclusive']==cut*96 and sorted(sum(g['groups'],[]))==list(range(57)) and max(map(len,g['groups']))<=8 and g['source_prefix_sha256']==hashlib.sha256(np.ascontiguousarray(d['y'].reshape(57,-1)[:,:cut*96]).tobytes()).hexdigest())
 if nr==40:check('R40 matchednocovariatecontrols match',read(Q/'matched_control_regression.json')['all_passed'])
 ns=functions(R/'code'/filename,['reconstruct','eligible'],dict(np=np));release={}
 with np.load(Q/'test_predictions.npz') as zz,np.load(Q/'dev_predictions.npz') as dv,np.load(Q/'native_test.npz') as nz:
  for H in [1,7]:
   oo=list(range(22,31)) if H==1 else[22,23,24];do=[14,16,18,20] if H==1 else[14,15];sh=sel['horizons'][str(H)];f=sh['family'];cfg=sh['selected'] or sh['diagnostic'];base=prior('test',H);raw=np.stack([nz[f'{f["id"]}_H{H}_o{o}'][:,4] for o in oo],1);fresh=ns['reconstruct'](base,raw,f['rate'],cfg['alpha'],cfg['total']);check(f'R{nr} H{H} actual reconstructionreplay',np.array_equal(fresh,zz[f'diagnostic_H{H}']));check(f'R{nr} H{H} selectedretain correct',np.array_equal(zz[f'selected_H{H}'],fresh if sh['selected'] else base));decision=ns['eligible'](summary['scores'][str(H)]['diagnostic'],summary['scores'][str(H)]['prior'],H);release[str(H)]=dict(dev_selected=sh['selected'] is not None,test_registered_gates=decision,adopt=bool(sh['selected'] and decision['eligible']),family=f,cfg=cfg)
   for typ in ['diagnostic','selected']:
    p=zz[f'{typ}_H{H}'];sc=evaluate(common,d,oo,p,H);check(f'R{nr} H{H} {typ} allmetricreplay',all(abs(v-summary['scores'][str(H)][typ][k])<1e-10 if isinstance(v,(int,float)) else v==summary['scores'][str(H)][typ][k] for k,v in sc.items()));vp=dv[f'{cfg["key"]}_H{H}'] if typ=='diagnostic' or sh['selected'] else prior('dev',H);vm=evaluate(common,d,do,vp,H)
    desc=('多分辨率' if nr==37 else'原生季节变化量' if nr==38 else'训练期相关域联合' if nr==39 else'真实CPU内存辅助')+'Moirai 2.0 R small';name=f'{desc} {f["id"]} 权重{cfg["alpha"]*100:g}% {cfg["total"]}＋'+('Chronos-2/TiRex-2/TimesFM3/Ridge/TinyTimeMixer' if H==1 else'Moirai2/TimesFM3/Ridge/HGB/TinyTimeMixer')+f'既有方案{(1-cfg["alpha"])*100:g}%';status='开发冻结测试未满足全部门槛' if sh['selected'] else'未通过开发升级的冻结研究诊断'
    if typ=='selected' and not sh['selected']:name=f'第{nr}轮按冻结保留既有'+('24小时第32轮' if H==1 else'7天第36轮');status='冻结保留当前推荐'
    rows.append(dict(horizon_days=H,model=f'r{nr}_{typ}_H{H}',model_name=name,report_name=desc+f' {f["rate"]*15:g}分钟\n{f["group"]}组 {f["variant"]}\n权重{cfg["alpha"]*100:g}% '+cfg['total'],status=status,source=dict(path=f'round{nr}/results/test_predictions.npz',array=f'{typ}_H{H}',sha256=sha(Q/'test_predictions.npz')),prediction_digest=hashlib.sha256(p.astype('float64').tobytes()).hexdigest(),development_wape_original_grid=vm['wape'],**sc))
    if typ=='diagnostic':
     y,m=common.truth(d,do,H);yt,mt=common.truth(d,oo,H);loq=np.array([np.quantile((y-vp)[i][m[i]],.1) for i in range(57)]);hiq=np.array([np.quantile((y-vp)[i][m[i]],.9) for i in range(57)]);lo=np.clip(p+loq[:,None,None],0,1);hi=np.clip(p+hiq[:,None,None],0,1);wink=hi-lo+10*np.maximum(lo-yt,0)+10*np.maximum(yt-hi,0);iv=dict(horizon_days=H,method=f'R{nr} {desc} 开发残差10/90经验80%区间',nominal_percent=80,coverage_percent=float(100*((yt>=lo)&(yt<=hi))[mt].mean()),mean_width_normalized=float((hi-lo)[mt].mean()),winkler80=float(wink[mt].mean()),kind='Fixed earlierDEV residual quantilepolicy; noTEST intervalsearch, explorationonly');intervals.append(iv);np.savez_compressed(Q/f'interval_diagnostic_H{H}.npz',loq=loq,hiq=hiq,lower=lo,upper=hi)
 releases[str(nr)]=release
 with np.load(Q/'deployment.npz') as dep:
  ex=Q/'exports';ex.mkdir(exist_ok=True)
  for H in [1,7]:
   iv=np.load(Q/f'interval_diagnostic_H{H}.npz')
   for typ in ['diagnostic']:
    p=dep[f'{typ}_H{H}'];check(f'R{nr} H{H} {typ} deploymentshape',p.shape==(57,1,H*96) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1);pts=[];day=[];ts=np.datetime64('2019-06-01T00:00')+np.arange(H*96)*np.timedelta64(15,'m')
    for i in range(57):
     for k in range(H*96):pts.append(dict(pdu=str(d['names'][i]),cell=str(d['cells'][i]),timestamp_local_PT=str(ts[k]),forecast_normalized_power=float(p[i,0,k]),lower80_empirical=float(np.clip(p[i,0,k]+iv['loq'][i],0,1)),upper80_empirical=float(np.clip(p[i,0,k]+iv['hiq'][i],0,1)),historical_replay=True,current_live_forecast=False,interval_scope='Frozen research diagnostic with earlier DEV residual calibration; not current recommendation'))
     for k in range(H):day.append(dict(pdu=str(d['names'][i]),date_PT=f'2019-06-{k+1:02d}',daily_mean_normalized_power=float(p[i,0,k*96:(k+1)*96].mean()),historical_replay=True))
    export(ex/f'{typ}_{H}day_curve.csv',pts);export(ex/f'{typ}_{H}day_daily_mean.csv',day)
   iv.close()
old=read(B/'round35/results/curves_priority_comparison.json');allrows=old['rows']+rows;uni=[]
for H in [1,7]:
 bank={}
 for r in sorted([x for x in allrows if x['horizon_days']==H],key=lambda r:0 if r['status']=='当前综合推荐' else 1):
  key=r['prediction_digest']
  if key not in bank:bank[key]=dict(r,aliases=[r['model']])
  else:bank[key]['aliases'].append(r['model'])
 uni+=sorted(bank.values(),key=lambda r:r['wape'])
winners=[]
for H,task,keys in [(1,'24小时曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天每日均值',['daily_wape','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent'])]:
 for key in keys:
  rr=[r for r in uni if r['horizon_days']==H];v=(max if'hit'in key else min)(r[key] for r in rr);winners.append(dict(task=task,metric=key,value=v,models='；'.join(r['model_name'] for r in rr if abs(r[key]-v)<1e-10),direction='越高越好' if'hit'in key else'越低越好'))
out=dict(old,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=allrows,unique_rows=uni,winners=winners,intervals=old['intervals']+intervals,continuation_37_40={str(n):read(B/f'round{n}/results/summary.json') for n in range(37,41)},release_decisions=releases,all_checks_passed=all(c['passed'] for c in checks));(OUT/'continued_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'independent_verification.json').write_text(json.dumps(dict(all_passed=True,checks=checks,check_count=len(checks),rows=len(allrows),unique=len(uni),release_decisions=releases),ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(all_passed=True,checks=len(checks),rows=len(allrows),unique=len(uni),releases=releases),ensure_ascii=False))
