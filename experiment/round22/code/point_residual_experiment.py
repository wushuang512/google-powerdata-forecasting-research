"""Compact pointwise native-forecast residual learning; only pre-origin history and forecast inputs."""
from pathlib import Path
import sys,ast,json,argparse,pickle,hashlib
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round22';Q=R/'results'
sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(B/'code'));from source_metrics import evaluate,interval_metrics;import common
src=B/'round20/code/learned_peak_experiment.py';e={'__file__':str(src),'__name__':'prior_pure_utilities'}
tree=ast.parse(src.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets));exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(src),'exec'),e)
read=e['read'];save=e['save'];sha=e['sha'];utc=e['utc'];z=e['z'];guard=e['guard'];cost=e['cost'];score=e['score'];CORE=e['CORE']
MODEL_CONFIGS=[dict(id=f'hgb_{loss}_leaves{leaf}',loss=loss,leaves=leaf,iterations=60 if leaf==7 else 80,l2=10 if leaf==7 else 100) for loss in ['absolute_error','squared_error'] for leaf in [7,15,31]]
CONFIGS=[dict(id='prior',model=None,neutrality='global')]+[dict(id=c['id']+'_'+n,model=c['id'],neutrality=n) for c in MODEL_CONFIGS for n in ['global','daily_global']]
ALPHAS=[0.,.05,.1,.25,.5]
def features(d,o,h,b):
 hist=d['y'][:,o-7:o];m=d['valid'][:,o-7:o];mu=(hist*m).sum(-1)/m.sum(-1).clip(1);scale=mu.mean(-1).clip(.05);last=hist[:,-1];histmean=hist.mean(1);phase=np.arange(96)*2*np.pi/96
 vals=[b/scale[:,None],last/scale[:,None],histmean/scale[:,None],(b-last)/scale[:,None],(b-histmean)/scale[:,None],np.roll(b,1,-1)/scale[:,None],np.roll(b,-1,-1)/scale[:,None]]
 for a in [b.mean(-1),b.max(-1),b.std(-1),mu[:,-1],mu[:,-3:].mean(-1),mu.mean(-1),mu[:,-1]-mu[:,0]]:vals.append(np.broadcast_to((a/scale)[:,None],b.shape))
 for f in [np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)]:vals.append(np.broadcast_to(f[None,:],b.shape))
 day=(2+o+h)%7
 for v in [np.sin(day*2*np.pi/7),np.cos(day*2*np.pi/7),h/6]:vals.append(np.full(b.shape,v))
 for cell in sorted(set(d['cells'])):vals.append(np.broadcast_to((d['cells']==cell)[:,None],b.shape))
 return np.stack(vals,-1).astype('float32'),scale
def fit(d,cutoff):
 from sklearn.ensemble import HistGradientBoostingRegressor
 bank=z('round17/results/native_train7.npz','prediction');xx=[];yy=[];ww=[];ends=[o+h for o in range(7,cutoff) for h in range(7) if o+h<cutoff];freq=np.bincount(ends)
 for o in range(7,cutoff):
  for h in range(7):
   end=o+h
   if end>=cutoff:continue
   b=bank[:,o-7,h*96:(h+1)*96];x,scale=features(d,o,h,b);slot=np.arange(96)%6==(o+h)%6;keep=d['valid'][:,end]&slot[None,:];res=(d['y'][:,end]-b)/scale[:,None];xx.append(x[keep]);yy.append(res[keep]);ww.append(np.full(keep.sum(),1/freq[end]))
 x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w=w*len(w)/w.sum();models={}
 save(R/f'models/train_inputs_cutoff{cutoff}.json',dict(shape=list(x.shape),targets=len(y),last_target_index=cutoff-1,X_sha256=hashlib.sha256(x.tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),weight_sha256=hashlib.sha256(w.tobytes()).hexdigest()))
 for c in MODEL_CONFIGS:
  model=HistGradientBoostingRegressor(loss=c['loss'],learning_rate=.03,max_iter=c['iterations'],max_leaf_nodes=c['leaves'],min_samples_leaf=128,l2_regularization=c['l2'],early_stopping=False,random_state=20261006);model.fit(x,y,sample_weight=w);models[c['id']]=model;print('FIT',cutoff,c['id'],len(y),flush=True)
 return models
def residual(d,origins,ref,model,H):
 out=np.zeros_like(ref)
 for j,o in enumerate(origins):
  for h in range(H):
   x,scale=features(d,o,h,ref[:,j,h*96:(h+1)*96]);q=np.clip(model.predict(x.reshape(-1,x.shape[-1])).reshape(57,96),-.05,.05)*scale[:,None];out[:,j,h*96:(h+1)*96]=q
 return out
def candidate(base,delta,cfg,H):
 if cfg['id']=='prior':return base.copy()
 raw=np.clip(base+delta,0,1);out=raw.copy()
 for j in range(base.shape[1]):
  slices=[slice(None)] if cfg['neutrality']=='global' else [slice(h*96,(h+1)*96) for h in range(H)]
  for sl in slices:
   b=base[:,j,sl];q=raw[:,j,sl];lo=-2.;hi=2.;target=b.sum()
   for _ in range(55):
    mid=(lo+hi)/2;v=np.clip(q+mid,0,1)
    if v.sum()>target:hi=mid
    else:lo=mid
   out[:,j,sl]=np.clip(q+(lo+hi)/2,0,1);assert abs(out[:,j,sl].sum()-target)<1e-8
 return out
def prepare():
 for k in ['code','docs','models','results']:(R/k).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];return
 files=['results/data.npz','round17/results/native_train7.npz','round17/results/native_train7_verification.json','round20/code/learned_peak_experiment.py','round19/results/dev_predictions.npz','round19/results/test_predictions.npz','round18/results/dev_predictions.npz','round18/results/test_predictions.npz','round11/results/dev_predictions.npz','round8b/results/test_predictions.npz','round3/results/tf3_dev.npz','round3/results/tf3_test.npz','round9/results/selected_intervals_test.npz','round21/results/dev_summary.json','round21/results/selection.json']
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),metrics_script_sha256=sha(R/'code/source_metrics.py'),models=MODEL_CONFIGS,configs=CONFIGS,alphas=ALPHAS,reference_document_sha256=sha(T/'参考材料/9.28算力中心需求预测实验.docx'),training='New point residual heads train from audited native TimesFM3 direct672 origins7..21. DEV only label dates<14; final<22. Fixed stride6 rotated by known origin/horizon, inverse repeated target-date weighting, no PDU identifier feature, no future workload. Small tree models, no backbone reload.',features='Native prediction and its adjacent predicted slots, past last-day and7day same-slot power, past daily statistics, known Fourier clock/weekday/horizon/cell. Targets residual/observed past7day mean, corrected residual clipped +/-5%historical mean.',application='Current R8b24 or R19 seven-day prior plus learned native residual; this delta-transfer is a modeling hypothesis, not an in-sample refined-prior error fit. Global or per-future-day forecast-total projection on normalized outputs, not energy conservation.',selection='Per known horizon separately: early DEV24 origins14..17 or DEV7 origin14 choose one family using unchanged R20 eight-metric cost and strict no-core-regression guard. Then later DEV24 or DEV7 origin15 plus aggregates choose strengths0/.05/.1/.25/.5 with same guards; at least.0001pp curve or peak improvement. One-SE shrink. Otherwise0. One diagnostic25% or positive selected strength frozen before TEST; never select based on TEST.',test_seen_before=True,independent_confirmation=False,resource_policy=dict(cache_only_admission_physical_gib=2.125,admission_commit_gib=6,owned_job_commit_hard_cap_bytes=512*2**20,runtime_physical_floor_gib=1.5,runtime_commit_floor_gib=2),source_hashes={f:sha(B/f) for f in files}))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii')
def develop(d):
 assert not (Q/'selection.json').exists();hs=fit(d,14);pickle.dump(hs,(R/'models/dev_heads.pkl').open('wb'));summ={};freeze={};saved={}
 for H in [1,7]:
  origins=list(range(14,22)) if H==1 else [14,15];base=e['load_base']('dev',H);ref=e['reference']('dev',H);count=4 if H==1 else 1;early=score(d,origins[:count],base[:,:count],H);late=score(d,origins[count:],base[:,count:],H);aggregate=score(d,origins,base,H);rows=[];preds={};deltas={k:residual(d,origins,ref,m,H) for k,m in hs.items()}
  for c in CONFIGS:
   p=candidate(base,deltas.get(c['model']),c,H);preds[c['id']]=p;s=score(d,origins[:count],p[:,:count],H);rows.append(dict(config=c,scores=s,cost=cost(s,early),guard_passed=guard(s,early)));print('DEV POINT',H,c['id'],rows[-1]['cost'],rows[-1]['guard_passed'],flush=True)
  pool=[r for r in rows if r['config']['id']!='prior'];el=[r for r in pool if r['guard_passed']];best=min(el or pool,key=lambda x:x['cost']);cfg=best['config'];cp=preds[cfg['id']];weights=[]
  for a in ALPHAS:
   p=(1-a)*base+a*cp;se=score(d,origins[:count],p[:,:count],H);sl=score(d,origins[count:],p[:,count:],H);sa=score(d,origins,p,H);ok=a>0 and guard(se,early) and guard(sl,late) and guard(sa,aggregate) and any(sa[k]<aggregate[k]-.0001 for k in ['wape','peak_mape']);vals=[]
   for j,o in enumerate(origins[count:],count):
    for h in range(H):vals.append(cost(score(d,[o+h],p[:,j:j+1,h*96:(h+1)*96],1),score(d,[o+h],base[:,j:j+1,h*96:(h+1)*96],1))-1)
   weights.append(dict(alpha=a,early=se,late=sl,aggregate=sa,cost=cost(sl,late),paired_date_se=float(np.std(vals,ddof=1)/np.sqrt(len(vals))),eligible=ok))
  el=[w for w in weights if w['eligible']]
  if el:
   win=min(el,key=lambda x:x['cost']);pick=min([w for w in el if w['cost']<=win['cost']+win['paired_date_se']],key=lambda x:x['alpha'])
  else:pick=weights[0]
  freeze[str(H)]=dict(candidate=cfg,chosen_alpha=pick['alpha'],diagnostic_alpha=pick['alpha'] if pick['alpha']>0 else.25,development_supported=bool(el));summ[str(H)]=dict(configs=rows,weights=weights,baseline_aggregate=aggregate,family_early=best);saved[f'prior{H}']=base;saved[f'candidate{H}']=cp;saved[f'selected{H}']=(1-pick['alpha'])*base+pick['alpha']*cp
 np.savez_compressed(Q/'dev_predictions.npz',**saved);save(Q/'dev_summary.json',dict(horizons=summ,testing_not_loaded=True));y,m=common.truth(d,[14,15],7);rad=np.array([np.quantile(abs(saved['selected7'][i]-y[i])[m[i]],.8) for i in range(57)]);np.savez_compressed(R/'models/interval7_radius.npz',radius=rad)
 s=dict(frozen_utc=utc(),horizons=freeze,protocol_sha256=sha(R/'docs/protocol.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_heads_sha256=sha(R/'models/dev_heads.pkl'),interval_radius_sha256=sha(R/'models/interval7_radius.npz'),independent_confirmation=False);save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();hs=fit(d,22);pickle.dump(hs,(R/'models/final_heads.pkl').open('wb'));scores={};arrays={}
 for H in [1,7]:
  origins=list(range(22,31)) if H==1 else [22,23,24];base=e['load_base']('test',H);ref=e['reference']('test',H);c=s['horizons'][str(H)];cp=candidate(base,residual(d,origins,ref,hs[c['candidate']['model']],H),c['candidate'],H);p=(1-c['chosen_alpha'])*base+c['chosen_alpha']*cp;diag=(1-c['diagnostic_alpha'])*base+c['diagnostic_alpha']*cp;arrays.update({f'prior{H}':base,f'candidate{H}':cp,f'diagnostic{H}':diag,f'selected{H}':p});scores[str(H)]={k:score(d,origins,v,H) for k,v in dict(prior=base,candidate=diag,selected=p).items()}
  if H==7:
   rad=z('round22/models/interval7_radius.npz','radius')[:,None,None];lo=np.clip(p-rad,0,1);hi=np.clip(p+rad,0,1)
  else:lo=np.clip(z('round9/results/selected_intervals_test.npz','lower')+p-base,0,1);hi=np.clip(z('round9/results/selected_intervals_test.npz','upper')+p-base,0,1)
  arrays[f'lower{H}']=lo;arrays[f'upper{H}']=hi;y,m=common.truth(d,origins,H);scores[str(H)]['interval_selected']=interval_metrics(y,p,m,lo,hi)
 np.savez_compressed(Q/'test_predictions.npz',**arrays);save(Q/'summary.json',dict(selection=s,scores=scores,no_post_test_selection=True,independent_confirmation=False));print('ACTUAL POINT TEST',json.dumps(scores),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];d=common.load();{'develop':develop,'test':test}[a.stage](d)

