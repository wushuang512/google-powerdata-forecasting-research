from pathlib import Path
import sys,json,hashlib,datetime,argparse,ast,numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=B/'round36';Q=R/'results';sys.path[:0]=[str(B/'code'),str(B/'round29/code')];import common;from aligned_metrics import evaluate
d=common.load();proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'));ns=dict(np=np,d=d);tree=ast.parse((B/'round34/code/risk_curve_experiment.py').read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='uncertainty'];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual R34 uncertainty>','exec'),ns);uncertainty=ns['uncertainty']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def origins(phase,H):return ([14,16,18,20] if H==1 else[14,15]) if phase=='dev' else([31] if phase=='deploy' else(list(range(22,31)) if H==1 else[22,23,24]))
def prior(phase,H):
 p=B/f'round{32 if H==1 else 31}'/'results'/('deployment.npz' if phase=='deploy' else f'{phase}_predictions.npz')
 with np.load(p) as z:return z[('TTM_head_sparsepeak50_strength0.1_H1' if H==1 else'TTM_head_shape10_H7') if phase=='dev' else f'selected_H{H}'].copy()
def native(phase,H,oo):
 family=proto['families'][str(H)]
 with np.load(B/f'round35/results/native_{phase}.npz') as z:return np.stack([z[f'{family}_H{H}_o{o}'][:,4] for o in oo],1)
def reconstruct(base,raw,cfg,oo):
 p=(1-cfg['alpha'])*base+cfg['alpha']*raw;p=p.reshape(57,len(oo),-1,96);b=base.reshape(p.shape)
 groups=[np.arange(57)] if cfg['total']=='global_daily' else[np.where(d['cells']==c)[0] for c in np.unique(d['cells'])] if cfg['total']=='cell_daily' else[]
 for ix in groups:
  target=b[ix].sum((0,3));lo=np.full(target.shape,-1.);hi=np.full(target.shape,1.)
  for _ in range(48):
   delta=(lo+hi)/2;value=np.clip(p[ix]+delta[None,:,:,None],0,1).sum((0,3));over=value>target;hi=np.where(over,delta,hi);lo=np.where(over,lo,delta)
  p[ix]=np.clip(p[ix]+((lo+hi)/2)[None,:,:,None],0,1)
 if cfg['peak_beta']:
  for j,o in enumerate(oo):
   _,sigma=uncertainty(o);hist_peak=d['y'][:,o-1].max(-1)
   for day in range(p.shape[2]):
    v=p[:,j,day];mu=v.mean(-1);target=np.clip((1-cfg['peak_beta'])*v.max(-1)+cfg['peak_beta']*hist_peak,mu,1);anchor=v.argmax(-1);lo=np.full(57,-100.);hi=np.full(57,100.)
    for _ in range(48):
     lam=(lo+hi)/2;q=np.clip(v+lam[:,None]*sigma,0,target[:,None]);q[np.arange(57),anchor]=target;over=q.mean(-1)>mu;hi=np.where(over,lam,hi);lo=np.where(over,lo,lam)
    q=np.clip(v+((lo+hi)/2)[:,None]*sigma,0,target[:,None]);q[np.arange(57),anchor]=target;assert np.max(abs(q.mean(-1)-mu))<1e-8;p[:,j,day]=q
 return p.reshape(base.shape)
def eligible(s,b,H):
 keys=['wape'] if H==1 else['wape','daily_wape'];primary=all(s[k]<=b[k]+1e-10 for k in keys) and s['wape']<b['wape']-.001;peak=s['peak_mape']<=b['peak_mape']+.1;tol={'mae':b['mae']*.002,'rmse':b['rmse']*.002,'p95_ape':.01,'hit2_percent':.05,'hit5_percent':.05,'cella5_wape':.005,'other52_wape':.005,'absolute_bias_percent':.005,'peak_time_mae_minutes':5,'daily_mae':b['daily_mae']*.005,'daily_rmse':b['daily_rmse']*.005};secondary=all(s[k]>=b[k]-v if k.startswith('hit') else s[k]<=b[k]+v for k,v in tol.items());return dict(eligible=bool(primary and peak and secondary),primary_cost=float(np.mean([s[k]/b[k] for k in keys])),curve_nonworse=primary,peak_within_prior_0_1pp=peak,secondary_small=secondary)
if args.stage=='develop':
 bank={};scores={};sel={}
 for H in [1,7]:
  oo=origins('dev',H);b=prior('dev',H);raw=native('dev',H,oo);bs=evaluate(common,d,oo,b,H);rr=[];scores[str(H)]=dict(prior=bs,candidates={})
  for cfg in proto['configs']:
   p=reconstruct(b,raw,cfg,oo);sc=evaluate(common,d,oo,p,H);bank[f'{cfg["id"]}_H{H}']=p;scores[str(H)]['candidates'][cfg['id']]=sc;rr.append(dict(config=cfg,**eligible(sc,bs,H)))
  rr.sort(key=lambda x:(not x['eligible'],x['primary_cost'],x['config']['id']));pick=next((x for x in rr if x['eligible']),None);sel[str(H)]=dict(selected=pick,diagnostic=rr[0],retain_prior=pick is None,ranking=rr)
 np.savez_compressed(Q/'dev_predictions.npz',**bank);save('dev_summary.json',scores);assert not (Q/'selection.json').exists();save('selection.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_test_used=True,horizons=sel,protocol_sha256=sha(R/'docs/protocol.json'),native_dev_sha256=sha(B/'round35/results/native_dev.npz'),dev_predictions_sha256=sha(Q/'dev_predictions.npz')));(Q/'selection.sha256').write_text(sha(Q/'selection.json'));print(json.dumps({h:{'selected':v['selected'],'diagnostic':v['diagnostic']} for h,v in sel.items()},ensure_ascii=False),flush=True)
else:
 sel=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();phase='test' if args.stage=='test' else'deploy';bank={};scores={}
 for H in [1,7]:
  oo=origins(phase,H);b=prior(phase,H);raw=native(phase,H,oo);sh=sel['horizons'][str(H)];cfg=(sh['selected'] or sh['diagnostic'])['config'];p=reconstruct(b,raw,cfg,oo);bank[f'diagnostic_H{H}']=p;bank[f'selected_H{H}']=p if sh['selected'] else b
  if phase=='test':scores[str(H)]={k:evaluate(common,d,oo,v,H) for k,v in [('prior',b),('diagnostic',p),('selected',bank[f'selected_H{H}'])]}
 np.savez_compressed(Q/('test_predictions.npz' if phase=='test' else'deployment.npz'),**bank)
 if phase=='test':save('summary.json',dict(selection=sel,scores=scores,independent_confirmation=False,no_test_grid=True));print(json.dumps(scores,ensure_ascii=False),flush=True)
