from pathlib import Path
import sys,json,hashlib,ast,csv,contextlib,types,gc,numpy as np
T=Path(sys.argv[1]);B=T/'experiment';OUT=B/'round43/results';sys.path[:0]=[str(B/'code'),str(B/'round29/code')];import common;from aligned_metrics import evaluate
d=common.load();flat=d['y'].reshape(57,-1);valid=d['valid'].reshape(57,-1);checks=[];rows=[];intervals=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,v):checks.append(dict(check=name,passed=bool(v)));assert v,name
def funcs(path,names,ns):
 nodes=[n for n in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def prior(phase,H):
 with np.load(B/f'round{32 if H==1 else 36}'/'results'/('deployment.npz' if phase=='deploy' else f'{phase}_predictions.npz')) as z:return z[('TTM_head_sparsepeak50_strength0.1_H1' if H==1 else'a0.1_global_daily_peak0_H7') if phase=='dev' else f'selected_H{H}'].astype('float64')
def export(p,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r if k not in ['source','aliases','prediction_digest']))
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rr)
class Tensor:
 def __init__(self,a):self.a=np.asarray(a)
 @property
 def shape(self):return self.a.shape
 def cpu(self):return self
 def numpy(self):return self.a
def boundary(nr,src,proto):
 ff=flat.copy();vv=valid.copy();caps=[]
 fake=types.SimpleNamespace(float32=np.float32,bool=bool,tensor=lambda a,**kw:Tensor(a),zeros=lambda s,**kw:Tensor(np.zeros(s)),inference_mode=contextlib.nullcontext)
 def model(f,ctx,L,cut):
  def call(x,m,pad):
   caps.append((x.a.copy(),m.a.copy()));p=np.broadcast_to(x.a.mean(1)[:,None,None,:],(1,9,L,f['rank'])).copy();return Tensor(p[...,0] if f['rank']==1 else p)
  return call
 ns=funcs(src,['factor_basis','history','infer_one','tensors'],dict(np=np,R=B/f'round{nr}',Q=B/f'round{nr}/results',flat=ff,valid=vv,sha=sha,save=lambda *a:None,model=model,torch=fake,DEV='recorded boundary only',CURRENT_MODE='level',hashlib=hashlib))
 for f in proto['families']:
  caps.clear();p=ns['infer_one'](f,7,22,22);old=[(a.copy(),b.copy()) for a,b in caps];ff[:,22*96:]+=123;vv[:,22*96:]=~vv[:,22*96:];caps.clear();q=ns['infer_one'](f,7,22,22);check(f'R{nr} {f["id"]} actual wrapper future boundary',np.array_equal(p,q) and len(old)==len(caps) and all(np.array_equal(a,c) and np.array_equal(b,e) for (a,b),(c,e) in zip(old,caps)));ff[:,22*96:]-=123;vv[:,22*96:]=~vv[:,22*96:]
for nr in [41,42,43]:
 R=B/f'round{nr}';Q=R/'results';src=R/'code/joint_factor_moirai.py';proto=read(R/'docs/protocol.json');sel=read(Q/'selection.json');s=read(Q/'summary.json');check(f'R{nr} completed',read(Q/'resource_status.json')['stage']=='completed');check(f'R{nr} metrics and original weights intact',proto['metric_sha256']==sha(B/'round29/code/aligned_metrics.py') and proto['original_weights_sha256']==sha(B/'round35/models/moirai2/model.safetensors'))
 if nr==43:
  repair=read(Q/'cache_signature_repair.json');check('R43 only signature implementation repaired',repair['all_passed'] and repair['current_script_sha256']==sha(src) and repair['original_script_sha256']==sha(R/'docs/cache_signature_before.py') and sha(Q/'selection.json')==repair['selection_sha256'] and all(sha(Q/f'native_{p}.npz')==v for p,v in repair['native_arrays_before_sha256'].items()));source_text=(R/'docs/cache_signature_before.py').read_text(encoding='utf-8')
 else:source_text=src.read_text(encoding='utf-8')
 check(f'R{nr} registered canonical source text binding',hashlib.sha256(source_text.encode()).hexdigest()==proto['script_sha256']);check(f'R{nr} frozen DEV only and identical scheme',sel['no_test_used'] and sel['one_shared_configuration_both_horizons'] and sha(Q/'selection.json')==(Q/'selection.sha256').read_text() and sel['protocol_sha256']==sha(R/'docs/protocol.json') and sel['native_dev_sha256']==sha(Q/'native_dev.npz') and sel['dev_predictions_sha256']==sha(Q/'dev_predictions.npz'));check(f'R{nr} freeze precedes TEST', (Q/'selection.json').stat().st_mtime<(Q/'test_predictions.npz').stat().st_mtime)
 bases={}
 for cut in [14,22]:
  p=R/f'models/factor_basis_cut{cut}.npz'
  with np.load(p) as z:b={k:z[k].copy() for k in ['mean','scale','vectors','variance']};finger=str(z['signature'])
  y=flat[:,:cut*96];m=valid[:,:cut*96];mu=(y*m).sum(-1)/m.sum(-1);std=np.sqrt(((y-mu[:,None])**2*m).sum(-1)/m.sum(-1)).clip(.01);check(f'R{nr} cut{cut} PCA means/std only valid trainingprefix',np.array_equal(mu,b['mean']) and np.array_equal(std,b['scale']) and finger==hashlib.sha256(sha(R/'docs/protocol.json').encode()+np.ascontiguousarray(y).tobytes()+np.ascontiguousarray(m).tobytes()).hexdigest());check(f'R{nr} cut{cut} orthonormal deterministic sign basis',np.max(abs(b['vectors']@b['vectors'].T-np.eye(6)))<1e-10 and (b['vectors'][np.arange(6),np.abs(b['vectors']).argmax(1)]>=0).all());bases[cut]=b
 boundary(nr,src,proto)
 for phase in ['dev','test','deploy']:
  cut=14 if phase=='dev' else 22;families=proto['families'] if phase=='dev' else[sel['family']]
  with np.load(Q/f'native_{phase}.npz') as bank:
   good=True
   for f in families:
    for H in [1,7]:
     oo=([14,16,18,20] if H==1 else[14,15]) if phase=='dev' else([31] if phase=='deploy' else(list(range(22,31)) if H==1 else[22,23,24]))
     for o in oo:
      key=f'{f["id"]}_H{H}_o{o}';extra=1 if nr==43 and f['mode']=='daily_increment' else 0;start=o-f['context_days']-extra;weight=sha(R/f'models/{f["id"]}_cut{cut}.pt') if f['variant']!='base' else'base';finger=hashlib.sha256((sha(R/'docs/protocol.json')+sha(src)+sha(B/'round35/code/moirai_native_adapter.py')+sha(B/'round35/models/moirai2/model.safetensors')+sha(R/f'models/factor_basis_cut{cut}.npz')+weight).encode()+np.ascontiguousarray(flat[:,start*96:o*96]).tobytes()+np.ascontiguousarray(valid[:,start*96:o*96]).tobytes()).hexdigest()
      with np.load(Q/'native_cache'/f'{phase}_{key}.npz') as z:good &= str(z['signature'])==finger and np.array_equal(z['quantiles'],bank[key]) and z['quantiles'].shape==(f['rank'],9,H*96//f['rate']) and np.isfinite(z['quantiles']).all()
   check(f'R{nr} {phase} all native caches fully bound',good)
 ns=funcs(src,['reconstruct','eligible'],dict(np=np));f=sel['family'];cfg=sel['selected'] or sel['diagnostic'];gates={}
 with np.load(Q/'native_dev.npz') as native,np.load(Q/'dev_predictions.npz') as dv,np.load(Q/'native_test.npz') as test,np.load(Q/'test_predictions.npz') as zz:
  for H in [1,7]:
   oo=list(range(22,31)) if H==1 else[22,23,24];do=[14,16,18,20] if H==1 else[14,15];base=prior('test',H);raw=np.stack([test[f'{f["id"]}_H{H}_o{o}'][:,4] for o in oo],1);p=ns['reconstruct'](base,raw,f['rate'],cfg['alpha'],cfg['total'],bases[22],f['rank']);check(f'R{nr} H{H} exact reconstruction',np.array_equal(p,zz[f'diagnostic_H{H}']));check(f'R{nr} H{H} retain according to freeze',np.array_equal(zz[f'selected_H{H}'],p if sel['selected'] else base));gates[str(H)]=ns['eligible'](s['scores'][str(H)]['diagnostic'],s['scores'][str(H)]['prior'],H);check(f'R{nr} H{H} registered TEST gates replay',gates[str(H)]==s['test_gates'][str(H)])
   for typ in ['diagnostic','selected']:
    p=zz[f'{typ}_H{H}'];score=evaluate(common,d,oo,p,H);check(f'R{nr} H{H} {typ} full metric replay',all(abs(v-s['scores'][str(H)][typ][k])<1e-10 if isinstance(v,(int,float)) else v==s['scores'][str(H)][typ][k] for k,v in score.items()));vp=dv[f'{cfg["key"]}_H{H}'] if typ=='diagnostic' or sel['selected'] else prior('dev',H);dev=evaluate(common,d,do,vp,H)
    title='Moirai2 '+('共同因子' if nr==41 else'共同因子注意力训练' if nr==42 else'每日变化量匹配对照');label=title+f' PCA{f["rank"]} {f["rate"]*15}分钟\n'+(f['variant']+' ' if nr==42 else f.get('mode',''))+f'权重{cfg["alpha"]*100:g}% '+cfg['total'];name='Salesforce Moirai 2.0 R small '+label.replace('\n','；')+'；当前相应跨度方案'+f'{100*(1-cfg["alpha"]):g}%';status='开发共同升级未通过的冻结研究候选'
    if typ=='selected' and not sel['selected']:name=f'第{nr}轮冻结保留当前推荐';label=name;status='冻结保留当前推荐'
    rows.append(dict(horizon_days=H,model=f'r{nr}_{typ}_H{H}',model_name=name,report_name=label,status=status,development_wape_original_grid=dev['wape'],source=dict(path=f'round{nr}/results/test_predictions.npz',array=f'{typ}_H{H}',sha256=sha(Q/'test_predictions.npz')),prediction_digest=hashlib.sha256(p.astype('float64').tobytes()).hexdigest(),**score))
    if typ=='diagnostic':
     y,m=common.truth(d,do,H);yt,mt=common.truth(d,oo,H);lo=np.array([np.quantile((y-vp)[i][m[i]],.1) for i in range(57)]);hi=np.array([np.quantile((y-vp)[i][m[i]],.9) for i in range(57)]);lower=np.clip(p+lo[:,None,None],0,1);upper=np.clip(p+hi[:,None,None],0,1);w=upper-lower+10*np.maximum(lower-yt,0)+10*np.maximum(yt-upper,0);intervals.append(dict(horizon_days=H,method=f'R{nr} '+title+' 开发残差10/90经验80%区间',nominal_percent=80,coverage_percent=float(100*((yt>=lower)&(yt<=upper))[mt].mean()),mean_width_normalized=float((upper-lower)[mt].mean()),winkler80=float(w[mt].mean())));np.savez_compressed(Q/f'interval_diagnostic_H{H}.npz',loq=lo,hiq=hi,lower=lower,upper=upper)
 check(f'R{nr} whole shared release replay',s['adopt_shared_scheme']==bool(sel['selected'] and all(g['eligible'] for g in gates.values())))
 with np.load(Q/'deployment.npz') as dep:
  ex=Q/'exports';ex.mkdir(exist_ok=True)
  for H in [1,7]:
   p=dep[f'diagnostic_H{H}'];check(f'R{nr} H{H} deploy shape values',p.shape==(57,1,H*96) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1)
   with np.load(Q/f'interval_diagnostic_H{H}.npz') as z:lo=z['loq'];hi=z['hiq']
   ts=np.datetime64('2019-06-01T00:00')+np.arange(H*96)*np.timedelta64(15,'m');pts=[dict(pdu=str(d['names'][i]),cell=str(d['cells'][i]),timestamp_local_PT=str(ts[k]),forecast_normalized_power=float(p[i,0,k]),lower80_empirical=float(np.clip(p[i,0,k]+lo[i],0,1)),upper80_empirical=float(np.clip(p[i,0,k]+hi[i],0,1)),historical_replay=True,current_live_forecast=False) for i in range(57) for k in range(H*96)];daily=[dict(pdu=str(d['names'][i]),date_PT=f'2019-06-{k+1:02d}',daily_mean_normalized_power=float(p[i,0,k*96:(k+1)*96].mean()),historical_replay=True) for i in range(57) for k in range(H)];export(ex/f'diagnostic_{H}day_curve.csv',pts);export(ex/f'diagnostic_{H}day_daily_mean.csv',daily)
 if nr==42:
  with np.load(Q/'native_dev.npz') as new,np.load(B/'round41/results/native_dev.npz') as old:check('R42 untrained matched control regression',all(np.max(abs(new[k]-old[k]))<1e-7 for k in new.files if '_base_' in k))
import torch
from safetensors.torch import load_file
original=load_file(B/'round35/models/moirai2/model.safetensors')
for p in (B/'round42/results').glob('train_*_cut*.json'):
 meta=read(p);f=meta['config'];cut=int(p.stem.split('_cut')[-1]);weight=B/f'round42/models/{f["id"]}_cut{cut}.pt';state=torch.load(weight,map_location='cpu',weights_only=True);allowed=lambda name:f['scope']=='full' or name.startswith('out_proj.') or any(name.startswith(f'encoder.layers.{i}.') for i in range(6-f['last_layers'],6));check(p.name+' actual trained parameters and frozen others',sum(v.numel() for k,v in state.items() if allowed(k))==meta['parameters'] and all(torch.equal(v,original[k]) for k,v in state.items() if not allowed(k)) and any(not torch.equal(v,original[k]) for k,v in state.items() if 'self_attn' in k and allowed(k)));check(p.name+' onlytraininglabels originalteacher and weights bound',meta['actual_label_end_exclusive']<=meta['training_label_cut']==cut*96 and meta['steps']==80 and meta['protocol_sha256']==sha(B/'round42/docs/protocol.json') and meta['weights_sha256']==sha(weight) and meta['original_weights_sha256']==sha(B/'round35/models/moirai2/model.safetensors'));del state;gc.collect()
old=read(B/'round40/results/continued_comparison.json');allrows=old['rows']+rows;unique=[]
for H in [1,7]:
 bank={}
 for r in sorted([r for r in allrows if r['horizon_days']==H],key=lambda r:0 if r['status']=='当前综合推荐' else 1):
  key=r['prediction_digest']
  if key not in bank:bank[key]=dict(r,aliases=[r['model']])
  else:bank[key]['aliases'].append(r['model'])
 unique+=sorted(bank.values(),key=lambda r:r['wape'])
winners=[]
for H,task,kk in [(1,'24小时曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天曲线',['wape','mae','rmse','peak_mape','p95_ape','hit2_percent','hit5_percent']),(7,'7天每日均值',['daily_wape','daily_mae','daily_rmse','daily_hit2_percent','daily_hit5_percent'])]:
 for k in kk:
  rr=[r for r in unique if r['horizon_days']==H];value=(max if'hit'in k else min)(r[k] for r in rr);winners.append(dict(task=task,metric=k,value=value,models='；'.join(r['model_name'] for r in rr if abs(r[k]-value)<1e-10),direction='越高越好' if'hit'in k else'越低越好'))
out=dict(old,rows=allrows,unique_rows=unique,winners=winners,intervals=old['intervals']+intervals,factor_rounds={str(n):read(B/f'round{n}/results/summary.json') for n in [41,42,43]},all_checks_passed=True,factor_checks=len(checks));(OUT/'factor_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'independent_verification.json').write_text(json.dumps(dict(all_passed=True,check_count=len(checks),checks=checks,rows=len(allrows),unique_predictions=len(unique)),ensure_ascii=False,indent=2),encoding='utf-8');export(T/'模型各项指标完整结果.csv',allrows);export(T/'各任务单项最优结果.csv',winners);export(T/'模型预测区间结果.csv',out['intervals'])
for f in ['9月28日统一口径模型复算.csv','9月28日指标完整模型对照.csv']:(T/f).write_bytes((T/'模型各项指标完整结果.csv').read_bytes())
(T/'9月28日指标预测区间对照.csv').write_bytes((T/'模型预测区间结果.csv').read_bytes());print(json.dumps(dict(all_passed=True,checks=len(checks),rows=len(allrows),unique=len(unique)),ensure_ascii=False))
