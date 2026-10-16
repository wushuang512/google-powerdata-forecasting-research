from pathlib import Path
import sys,json,hashlib,datetime,argparse,gc,ast,numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=B/'round35';Q=R/'results';C=R/'code';sys.path[:0]=[str(R/'runtime'),str(C),str(C/'official_uni2ts/src'),str(B/'code'),str(B/'round29/code')];import torch;import common;from aligned_metrics import evaluate;from moirai_native_adapter import build,load_forecast_class,load_module
assert torch.cuda.is_available(),'CUDA experiment must not silently fall back';DEVICE=torch.device('cuda');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def build_cuda(context,horizon,dim=1):return build(R,context,horizon,dim).to(DEVICE)
torch.set_num_threads(2);torch.manual_seed(20261007);d=common.load();proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'));flat=d['y'].reshape(57,-1);valid=d['valid'].reshape(57,-1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def cache_signature(family,H,o):
 context=672 if family=='uni7' else 1344
 files=[R/'docs/protocol.json',C/'moirai_curve_experiment.py',C/'moirai_native_adapter.py',C/'official_uni2ts/src/uni2ts/model/moirai2/forecast.py',C/'official_uni2ts/src/uni2ts/model/moirai2/module.py',R/'models/moirai2/config.json',R/'models/moirai2/model.safetensors',B/'round29/code/aligned_metrics.py']
 h=hashlib.sha256();h.update(json.dumps({'family':family,'H':H,'origin':o,'torch':torch.__version__,'numpy':np.__version__,'device':'cuda','tf32':False},sort_keys=True).encode())
 for p in files:h.update(sha(p).encode())
 for a in [flat[:,o*96-context:o*96],valid[:,o*96-context:o*96],d['cells']]:h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
def z(path,key):
 with np.load(B/path) as a:return a[key].astype('float64')
def origins(phase,H):return ([14,16,18,20] if H==1 else [14,15]) if phase=='dev' else(list(range(22,31)) if H==1 else [22,23,24])
def prior(phase,H):return z(f'round32/results/{phase}_predictions.npz','TTM_head_sparsepeak50_strength0.1_H1' if phase=='dev' else 'selected_H1') if H==1 else z(f'round31/results/{phase}_predictions.npz','TTM_head_shape10_H7' if phase=='dev' else 'selected_H7')
def native_predict(model,o,context,groups,H):
 result=np.zeros((57,9,H*96))
 with torch.inference_mode():
  if model.hparams.target_dim==1:
   for start in range(0,57,4):
    ix=np.arange(start,min(start+4,57));x=torch.tensor(flat[ix,o*96-context:o*96],dtype=torch.float32,device=DEVICE)[:,:,None];m=torch.tensor(valid[ix,o*96-context:o*96],device=DEVICE)[:,:,None];p=model(past_target=x,past_observed_target=m,past_is_pad=torch.zeros((len(ix),context),dtype=torch.bool,device=DEVICE)).cpu().numpy();assert p.shape==(len(ix),9,H*96);result[ix]=p
  else:
   for ix in groups:
    x=torch.tensor(flat[ix,o*96-context:o*96].T[None],dtype=torch.float32,device=DEVICE);m=torch.tensor(valid[ix,o*96-context:o*96].T[None],device=DEVICE);p=model(past_target=x,past_observed_target=m,past_is_pad=torch.zeros((1,context),dtype=torch.bool,device=DEVICE)).cpu().numpy();assert p.shape==(1,9,H*96,len(ix));result[ix]=p[0].transpose(2,0,1)
 assert np.isfinite(result).all();return np.clip(result,0,1)
def smoke():
 m=build_cuda(672,96,1);x=torch.tensor(flat[0,14*96-672:14*96],dtype=torch.float32,device=DEVICE)[None,:,None];mask=torch.tensor(valid[0,14*96-672:14*96],device=DEVICE)[None,:,None];pad=torch.zeros((1,672),dtype=torch.bool,device=DEVICE)
 with torch.inference_mode():a=m(x,mask,pad);b=m(x,mask,pad)
 # Full official class compiled with postponed unused service annotations, same module/hparams.
 from types import SimpleNamespace
 import moirai_native_adapter as adapter
 src=C/'official_uni2ts/src/uni2ts/model/moirai2/forecast.py';tree=ast.parse(src.read_text(encoding='utf-8'));cl=next(n for n in tree.body if isinstance(n,ast.ClassDef));ns=dict(vars(adapter),L=SimpleNamespace(LightningModule=type('MinimalLightning',(torch.nn.Module,),dict(save_hyperparameters=m.save_hyperparameters.__func__,device=type(m).device))),Moirai2Module=type(m.module));import __future__;exec(compile(ast.Module(body=[cl],type_ignores=[]),str(src),'exec',flags=__future__.annotations.compiler_flag),ns);ref=ns['Moirai2Forecast'](96,1,0,0,672,module=m.module).eval()
 with torch.inference_mode():r=ref(x,mask,pad)
 assert torch.equal(a,b) and torch.equal(a,r);assert a.shape==(1,9,96)
 save('adapter_verification.json',dict(all_passed=True,strict_weights=True,deterministic=True,output_shape=list(a.shape),full_official_forecast_vs_adapter_max_difference=float((a-r).abs().max()),parameters=sum(v.numel() for v in m.parameters()),torch_version=torch.__version__,numpy_version=np.__version__,note='Official neural module and mathematical inference methods preserved. Unused Lightning/GluonTS service frontend omitted. Runtime differs from repository pinned training dependencies; smoke comparison is an implementation check, not external accuracy proof. Official formatter squeezes the singleton univariate target dimension.',official_module_sha256=sha(C/'official_uni2ts/src/uni2ts/model/moirai2/module.py'),official_forecast_sha256=sha(src),adapter_sha256=sha(C/'moirai_native_adapter.py')));print('official adapter equivalence passed',flush=True)
def infer(phase):
 if phase=='dev':requests=[(family,H,o) for family in proto['families'] for H in [1,7] for o in origins('dev',H)]
 elif phase=='test':
  sel=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();requests=[]
  for H in [1,7]:
   cfg=(sel['horizons'][str(H)]['selected'] or sel['horizons'][str(H)]['diagnostic'])['config'];requests += [(cfg['family'],H,o) for o in origins('test',H)]
 else:
  sel=read(Q/'selection.json');requests=[((sel['horizons'][str(H)]['selected'] or sel['horizons'][str(H)]['diagnostic'])['config']['family'],H,31) for H in [1,7]]
 requests=list(dict.fromkeys(requests));bank={};current_key=None;model=None
 for family,H,o in requests:
  key=f'{family}_H{H}_o{o}';file=Q/'native_cache_cuda'/f'{phase}_{key}.npz';file.parent.mkdir(exist_ok=True)
  signature=cache_signature(family,H,o)
  if file.exists():
   with np.load(file) as cached:
    assert str(cached['signature'])==signature,'Incompatible cache; preserve and audit before replacement'
    bank[key]=cached['quantiles']
   continue
  context=672 if family=='uni7' else 1344
  if family.startswith('uni'):
   groupkey=(family,H,1)
   if current_key!=groupkey:
    if model is not None:del model;gc.collect()
    torch.cuda.empty_cache();model=build_cuda(context,H*96,1);current_key=groupkey
   p=native_predict(model,o,context,None,H)
  else:
   p=np.zeros((57,9,H*96))
   for cell in np.unique(d['cells']):
    ix=np.where(d['cells']==cell)[0];groupkey=(family,H,len(ix))
    if current_key!=groupkey:
     if model is not None:del model;gc.collect()
     torch.cuda.empty_cache();model=build_cuda(context,H*96,len(ix));current_key=groupkey
    cp=native_predict(model,o,context,[ix],H);p[ix]=cp[ix]
  tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,quantiles=p,signature=signature);tmp.replace(file);bank[key]=p;print(phase,key,flush=True)
 np.savez_compressed(Q/f'native_{phase}.npz',**bank)
def compose(base,raw,cfg,oo):
 mode=cfg['mode']
 if mode=='raw':return raw
 if mode=='mix25':return np.clip(.75*base+.25*raw,0,1)
 if mode=='shape25':
  b=base.reshape(57,len(oo),-1,96);r=raw.reshape(b.shape);out=.75*b+.25*(r-r.mean(-1)[...,None]+b.mean(-1)[...,None]);return np.clip(out,0,1).reshape(base.shape)
 if mode=='historic_peak':
  p=raw.copy().reshape(57,len(oo),-1,96)
  for j,o in enumerate(oo):
   pk=d['y'][:,o-1].max(-1)
   for day in range(p.shape[2]):
    v=p[:,j,day];target=np.clip(pk,v.mean(-1),1);ix=v.argmax(-1);v[:]=np.minimum(v,target[:,None]);v[np.arange(57),ix]=target
  return p.reshape(base.shape)
 raise ValueError(mode)
def eligible(s,b,bestpeak,H):
 primary=['wape'] if H==1 else ['wape','daily_wape'];ok=all(s[k]<=b[k]+1e-10 for k in primary) and any(s[k]<=b[k]-.001 for k in primary);tol={'mae':b['mae']*.002,'rmse':b['rmse']*.002,'p95_ape':.01,'hit2_percent':.05,'hit5_percent':.05,'cella5_wape':.005,'other52_wape':.005,'absolute_bias_percent':.005,'peak_time_mae_minutes':5};small=all((s[k]>=b[k]-v if k.startswith('hit') else s[k]<=b[k]+v) for k,v in tol.items());peak_ok=s['peak_mape']<=bestpeak+.15;cost=np.mean([s[k]/b[k] for k in primary]);return dict(eligible=bool(ok and small and peak_ok),primary_cost=float(cost),peak_close_to_best=peak_ok,secondary_small=small)
def develop():
 assert read(Q/'adapter_verification.json')['all_passed'];sel={};scores={};bank={}
 with np.load(Q/'native_dev.npz') as z:
  for H in [1,7]:
   oo=origins('dev',H);base=prior('dev',H);bs=evaluate(common,d,oo,base,H);bestpk=min(evaluate(common,d,oo,common.baseline(d,oo,H,k),H)['peak_mape'] for k in ['last','mean3','week']);rr=[];scores[str(H)]={'prior':bs,'candidates':{}}
   for cfg in proto['configs']:
    raw=np.stack([z[f'{cfg["family"]}_H{H}_o{o}'][:,4] for o in oo],1);p=compose(base,raw,cfg,oo);sc=evaluate(common,d,oo,p,H);scores[str(H)]['candidates'][cfg['id']]=sc;bank[f'{cfg["id"]}_H{H}']=p;rr.append(dict(config=cfg,**eligible(sc,bs,bestpk,H)))
   rr.sort(key=lambda r:(not r['eligible'],r['primary_cost'],r['config']['id']));pick=next((x for x in rr if x['eligible']),None);sel[str(H)]=dict(selected=pick,diagnostic=rr[0],retain_prior=pick is None,ranking=rr)
 np.savez_compressed(Q/'dev_predictions.npz',**bank);save('dev_summary.json',scores);assert not (Q/'selection.json').exists();save('selection.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),horizons=sel,protocol_sha256=sha(R/'docs/protocol.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),no_test_used=True));(Q/'selection.sha256').write_text(sha(Q/'selection.json'));print(json.dumps(sel,ensure_ascii=False),flush=True)
def test():
 sel=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();bank={};scores={}
 with np.load(Q/'native_test.npz') as z:
  for H in [1,7]:
   oo=origins('test',H);base=prior('test',H);sh=sel['horizons'][str(H)];cfg=(sh['selected'] or sh['diagnostic'])['config'];raw=np.stack([z[f'{cfg["family"]}_H{H}_o{o}'][:,4] for o in oo],1);p=compose(base,raw,cfg,oo);bank[f'diagnostic_H{H}']=p;bank[f'selected_H{H}']=p if sh['selected'] else base;scores[str(H)]={k:evaluate(common,d,oo,v,H) for k,v in [('prior',base),('diagnostic',p),('selected',bank[f'selected_H{H}'])]}
 np.savez_compressed(Q/'test_predictions.npz',**bank);save('summary.json',dict(selection=sel,scores=scores,selection_sha256=sha(Q/'selection.json'),independent_confirmation=False,no_test_grid=True));print(json.dumps(scores,ensure_ascii=False),flush=True)
def deploy():
 sel=read(Q/'selection.json');bank={}
 with np.load(Q/'native_deploy.npz') as z:
  for H,nr in [(1,32),(7,31)]:
   base=zload=zread=None;base=np.load(B/f'round{nr}/results/deployment.npz')[f'selected_H{H}'];sh=sel['horizons'][str(H)];cfg=(sh['selected'] or sh['diagnostic'])['config'];raw=z[f'{cfg["family"]}_H{H}_o31'][:,4,None,:];p=compose(base,raw,cfg,[31]);bank[f'diagnostic_H{H}']=p;bank[f'selected_H{H}']=p if sh['selected'] else base
 np.savez_compressed(Q/'deployment.npz',**bank)
stage=args.stage
if stage=='smoke':smoke()
elif stage.startswith('native_'):infer(stage.split('_')[1])
else:{'develop':develop,'test':test,'deploy':deploy}[stage]()
