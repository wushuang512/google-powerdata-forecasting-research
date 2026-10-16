from pathlib import Path
import sys,json,hashlib,datetime,argparse,pickle,numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',required=True);args=ap.parse_args();T=Path(args.root);B=T/'experiment';R=B/'round34';Q=R/'results';sys.path.insert(0,str(B/'code'));sys.path.insert(0,str(B/'round29/code'));import common;from aligned_metrics import evaluate
d=common.load();proto=json.loads((R/'docs/protocol.json').read_text(encoding='utf-8'));CORE=['wape','daily_wape','peak_mape'];EXTRA=['mae','rmse','p95_ape','hit2_percent','hit5_percent','cella5_wape','other52_wape','absolute_bias_percent','peak_time_mae_minutes','daily_mae','daily_rmse']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(Q/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def z(path,key):
 with np.load(B/path) as a:return a[key].astype('float64')
def orig(phase,H):return ([14,16,18,20] if H==1 else [14,15]) if phase=='dev' else(list(range(22,31)) if H==1 else [22,23,24])
def current(phase,H):
 if H==1:return z(f'round32/results/{phase}_predictions.npz','TTM_head_sparsepeak50_strength0.1_H1' if phase=='dev' else 'selected_H1')
 return z(f'round31/results/{phase}_predictions.npz','TTM_head_shape10_H7' if phase=='dev' else 'selected_H7')
def controls(H):
 oo=orig('dev',H);refs={k:common.baseline(d,oo,H,k) for k in ['last','mean3','week']};refs['current']=current('dev',H)
 if H==1:
  refs['TimesFM3 uni21']=z('round3/results/tf3_dev.npz','tf3_uni_context2016')[:,[0,2,4,6]]
  for name,key in [('Chronos2','zero_uni'),('TiRex2','tirex_uni'),('C2T2 equal','old_equal_mix')]:refs[name]=z('round2/results/development_selected.npz',key)[:,[0,2,4,6]]
 else:
  refs['TimesFM3 uni21']=z('round15/results/tf3_dev7.npz','prediction')
  for key in ['chronos_zero','chronos_cov','ensemble7']:refs[key]=z('results/dev7.npz',key)
 prior=z('round22/results/dev_predictions.npz',f'prior{H}');cp=z('round22/results/dev_predictions.npz',f'candidate{H}');alpha=read(B/'round22/results/selection.json')['horizons'][str(H)]['diagnostic_alpha'];p=(1-alpha)*prior+alpha*cp;refs['R22 frozen diagnostic']=p[:,[0,2,4,6]] if H==1 else p
 return refs
def features(o,h):
 hist=d['y'][:,o-7:o];mu=hist.mean(-1);mx=hist.max(-1);std=hist.std(-1);norm=mu[:,-3:].mean(-1).clip(.05);cols=[]
 for k in [1,3,7]:
  for stat in [mu,mx,std]:cols.append(stat[:,-k:].mean(-1)/norm)
 cols += [(mu[:,-1]-mu[:,-3:].mean(-1))/norm,(mx[:,-1]-mx[:,-3:].mean(-1))/norm,(mu[:,-1]-mu[:,0])/norm,(mx[:,-1]-mx[:,0])/norm,norm,np.full(57,h/7),np.full(57,np.sin(2*np.pi*(o+h-1)/7)),np.full(57,np.cos(2*np.pi*(o+h-1)/7))]
 peer_mu=np.zeros((57,3));peer_mx=np.zeros((57,3))
 for cell in np.unique(d['cells']):
  ix=d['cells']==cell;gmu=mu[ix].mean(0);gmx=mx[ix].mean(0)
  for j,k in enumerate([1,3,7]):peer_mu[ix,j]=gmu[-k:].mean();peer_mx[ix,j]=gmx[-k:].mean()
 cols += [peer_mu[:,j]/norm for j in range(3)]+[peer_mx[:,j]/norm for j in range(3)]+[(d['cells']==c).astype(float) for c in np.unique(d['cells'])]
 return np.stack(cols,1),norm,np.stack([mu[:,-1],mx[:,-1]],1)
def fit_heads(cut):
 from sklearn.ensemble import HistGradientBoostingRegressor
 xs=[];ys=[];dates=[]
 for o in range(7,cut):
  for h in range(1,min(7,cut-o)+1):
   x,n,a=features(o,h);j=o+h-1;ok=d['valid'][:,j].all(-1);target=np.stack([d['y'][:,j].mean(-1),d['y'][:,j].max(-1)],1);xs.append(x[ok]);ys.append(((target-a)/n[:,None])[ok]);dates.append(np.full(ok.sum(),j))
 X=np.concatenate(xs);Y=np.concatenate(ys);dt=np.concatenate(dates);w=1/np.bincount(dt)[dt];w*=len(w)/w.sum();models={}
 for kind,loss in [('hgb_absolute','absolute_error'),('hgb_squared','squared_error')]:
  heads=[]
  for k in range(2):
   m=HistGradientBoostingRegressor(loss=loss,max_leaf_nodes=7,max_iter=60,learning_rate=.03,min_samples_leaf=128,l2_regularization=10,random_state=20261007);m.fit(X,Y[:,k],sample_weight=w);heads.append(m)
  models[kind]=heads
 with (R/f'models/heads_cut{cut}.pkl').open('wb') as f:pickle.dump(models,f)
 save(f'training_cut{cut}.json',dict(label_end_exclusive=cut,rows=len(X),features=X.shape[1],weights='inverse target date frequency',losses=['absolute_error','squared_error'],head_targets='mean andmax normalized residual to observed lastday, train-only complete96 labels',seed=20261007));return models
def uncertainty(o):
 hist=d['y'][:,o-7:o];dev=hist-hist.mean(-1)[:,:,None];dev=dev-dev.mean(1)[:,None,:];samples=np.concatenate([np.roll(dev,k,axis=-1) for k in [-2,-1,0,1,2]],axis=1);sigma=samples.std(1).clip(.001);return samples,sigma
def reconstruct(base,models,cfg,oo):
 out=[]
 for j,o in enumerate(oo):
  samples,sigma=uncertainty(o);days=[]
  for h in range(1,base.shape[-1]//96+1):
   b=base[:,j,(h-1)*96:h*96];x,n,a=features(o,h)
   heads=a if cfg['head']=='history' else np.clip(a+n[:,None]*np.stack([m.predict(x) for m in models[cfg['head']]],1),0,1)
   target_mean=(1-cfg['mean_strength'])*b.mean(-1)+cfg['mean_strength']*heads[:,0];target_peak=np.clip(heads[:,1],target_mean,1)
   if cfg['anchor']=='max':anchor=b.argmax(-1)
   else:
    scenarios=b[:,None,:]+samples;delta=np.mean(abs(scenarios-target_peak[:,None,None])-abs(scenarios-b[:,None,:]),axis=1);delta+=.25*(np.mean((scenarios-target_peak[:,None,None])**2-(scenarios-b[:,None,:])**2,axis=1))/sigma;anchor=delta.argmin(-1)
   # Gaussian-CDF water-filling: common standardized shift, differing historical uncertainty.
   lo=np.full(57,-100.);hi=np.full(57,100.)
   for _ in range(55):
    lam=(lo+hi)/2;q=np.clip(b+lam[:,None]*sigma,0,target_peak[:,None]);q[np.arange(57),anchor]=target_peak;over=q.mean(-1)>target_mean;hi=np.where(over,lam,hi);lo=np.where(over,lo,lam)
   q=np.clip(b+((lo+hi)/2)[:,None]*sigma,0,target_peak[:,None]);q[np.arange(57),anchor]=target_peak;assert np.max(abs(q.mean(-1)-target_mean))<1e-8;days.append(q)
  out.append(np.concatenate(days,-1))
 return np.stack(out,1)
def rank_score(s,b,envelope):
 convert=lambda k,v:100-v if k.startswith('hit') else v
 keys=CORE+EXTRA;ratios={k:convert(k,s[k])/max(convert(k,envelope[k]),1e-8) for k in keys};leads=[k for k in keys if ratios[k]<=1+1e-9];primary_max=max(ratios[k] for k in CORE);primary_mean=float(np.mean([ratios[k] for k in CORE]));tol={'mae':b['mae']*.002,'rmse':b['rmse']*.002,'p95_ape':.01,'hit2_percent':.05,'hit5_percent':.05,'cella5_wape':.005,'other52_wape':.005,'absolute_bias_percent':.005,'peak_time_mae_minutes':5,'daily_mae':b['daily_mae']*.005,'daily_rmse':b['daily_rmse']*.005};secondary_ok=all((s[k]>=b[k]-v if k.startswith('hit') else s[k]<=b[k]+v) for k,v in tol.items());primary_ok=all(s[k]<=b[k]+1e-10 for k in CORE);improvement=any(s[k]<b[k]-1e-8 for k in CORE);eligible=bool(primary_ok and improvement and secondary_ok)
 return dict(primary_worst_regret=primary_max,primary_mean_ratio=primary_mean,leading_metrics=leads,all_metrics_leading=len(leads)==len(keys),primary_all_leading=all(ratios[k]<=1+1e-9 for k in CORE),secondary_small=secondary_ok,eligible=eligible,ratios=ratios)
stage=args.stage
if stage=='develop':
 models=fit_heads(14);bank={};selection={};devscores={}
 for H in [1,7]:
  oo=orig('dev',H);base=current('dev',H);bs=evaluate(common,d,oo,base,H);refs={k:evaluate(common,d,oo,p,H) for k,p in controls(H).items()};env={k:(max if k.startswith('hit') else min)(s[k] for s in refs.values()) for k in CORE+EXTRA};rr=[];devscores[str(H)]=dict(prior=bs,controls=refs,envelope=env,candidates={})
  for cfg in proto['configs']:
   p=reconstruct(base,models,cfg,oo);sc=evaluate(common,d,oo,p,H);v=rank_score(sc,bs,env);rr.append(dict(config=cfg,**v));bank[f'{cfg["id"]}_H{H}']=p;devscores[str(H)]['candidates'][cfg['id']]=sc
  rr.sort(key=lambda r:(not r['eligible'],-sum(k in CORE for k in r['leading_metrics']),r['primary_worst_regret'],r['primary_mean_ratio'],r['config']['id']));choice=next((r for r in rr if r['eligible']),None);selection[str(H)]=dict(selected=choice,diagnostic=rr[0],retain_prior=choice is None,ranking=rr);bank[f'prior_H{H}']=base
 np.savez_compressed(Q/'dev_predictions.npz',**bank);save('dev_summary.json',devscores);assert not (Q/'selection.json').exists();save('selection.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),horizons=selection,protocol_sha256=sha(R/'docs/protocol.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),no_test_used=True));(Q/'selection.sha256').write_text(sha(Q/'selection.json'));print(json.dumps(selection,ensure_ascii=False),flush=True)
elif stage=='test':
 sel=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text() and sha(R/'docs/protocol.json')==sel['protocol_sha256'];models=fit_heads(22);bank={};scores={}
 for H in [1,7]:
  oo=orig('test',H);base=current('test',H);sh=sel['horizons'][str(H)];it=sh['selected'] or sh['diagnostic'];p=reconstruct(base,models,it['config'],oo);new=p if sh['selected'] else base;bank[f'diagnostic_H{H}']=p;bank[f'selected_H{H}']=new;scores[str(H)]={k:evaluate(common,d,oo,v,H) for k,v in [('prior',base),('diagnostic',p),('selected',new)]}
 np.savez_compressed(Q/'test_predictions.npz',**bank);save('summary.json',dict(selection=sel,selection_sha256=sha(Q/'selection.json'),scores=scores,independent_confirmation=False,all_candidate_test_sweep=False));print(json.dumps(scores,ensure_ascii=False),flush=True)
elif stage=='deploy':
 sel=read(Q/'selection.json');models=pickle.load((R/'models/heads_cut22.pkl').open('rb'));bank={}
 for H,nr in [(1,32),(7,31)]:
  base=z(f'round{nr}/results/deployment.npz',f'selected_H{H}');it=sel['horizons'][str(H)]['selected'] or sel['horizons'][str(H)]['diagnostic'];p=reconstruct(base,models,it['config'],[31]);bank[f'diagnostic_H{H}']=p;bank[f'selected_H{H}']=p if sel['horizons'][str(H)]['selected'] else base
 np.savez_compressed(Q/'deployment.npz',**bank);print('deployment saved')
else:raise ValueError(stage)
