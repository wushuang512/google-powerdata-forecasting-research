"""Prospective robust point residual calibration on the R26 recommended forecasts."""
from pathlib import Path
import sys,ast,json,argparse,pickle,hashlib,datetime
sys.dont_write_bytecode=True
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round27';Q=R/'results'
sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(B/'code'));from source_metrics import evaluate,interval_metrics;import common
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def z(p,k):
 with np.load(B/p) as a:return a[k].copy()
src=B/'round22/code/point_residual_experiment.py';tree=ast.parse(src.read_text(encoding='utf-8'));functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['features','residual']];exec(compile(ast.Module(body=functions,type_ignores=[]),str(src),'exec'))
MODELS=[dict(id=f'lgb_{loss}_leaf{leaf}',loss=loss,leaves=leaf,iterations=60 if leaf==7 else 80) for loss in ['huber','regression_l1'] for leaf in [7,15]]
CONFIGS=[dict(id=c['id']+f'_debit{int(beta*100)}',model=c['id'],debit=beta) for c in MODELS for beta in [0.,.5]]
ALPHAS=[0.,.025,.05,.1,.25]
TOL=dict(wape_pp=.005,daily_wape_pp=.005,peak_mape_pp=.005,p95_ape_pp=.01,mae_relative=.002,rmse_relative=.002,hit_pp=.05,group_wape_pp=.005,bias_pp=.005,peak_time_minutes=5,daily_error_relative=.005)
def guard(s,b):
 return all(s[k]<=b[k]+v for k,v in [('wape',.005),('daily_wape',.005),('peak_mape',.005),('p95_ape',.01),('cella5_wape',.005),('other52_wape',.005),('absolute_bias_percent',.005),('peak_time_mae_minutes',5)]) and all(s[k]<=b[k]*1.002 for k in ['mae','rmse']) and all(s[k]<=b[k]*1.005 for k in ['daily_mae','daily_rmse']) and all(s[k]>=b[k]-.05 for k in ['hit2_percent','hit5_percent'])
def categories(s,b):return [k for k,ok in [('curve',s['wape']<b['wape']-.0001),('daily',s['daily_wape']<b['daily_wape']-.0001),('peak',s['peak_mape']<b['peak_mape']-.0001),('distribution',s['p95_ape']<b['p95_ape']-.0001 or any(s[k]>b[k]+.005 for k in ['hit2_percent','hit5_percent']))] if ok]
def cost(s,b):return float(sum(w*s[k]/max(b[k],1e-8) for k,w in [('wape',2),('mae',1),('rmse',1),('daily_wape',2),('peak_mape',2),('p95_ape',1)])/9)
def score(d,origins,p,H):return evaluate(common,d,origins,p,H)
def training(d,cutoff):
 bank=z('round17/results/native_train7.npz','prediction');xx=[];yy=[];ww=[];ends=[o+h for o in range(7,cutoff) for h in range(7) if o+h<cutoff];freq=np.bincount(ends)
 for o in range(7,cutoff):
  for h in range(7):
   end=o+h
   if end>=cutoff:continue
   b=bank[:,o-7,h*96:(h+1)*96];x,scale=features(d,o,h,b);keep=d['valid'][:,end]&(np.arange(96)%6==(o+h)%6)[None,:];res=(d['y'][:,end]-b)/scale[:,None];xx.append(x[keep]);yy.append(res[keep]);ww.append(np.full(keep.sum(),1/freq[end]))
 x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w=w*len(w)/w.sum();return x,y,w
def fit(d,cutoff,which=None):
 from lightgbm import LGBMRegressor
 x,y,w=training(d,cutoff);save(R/f'models/train_inputs_cutoff{cutoff}.json',dict(shape=list(x.shape),last_target_index=cutoff-1,X_sha256=hashlib.sha256(x.tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),weight_sha256=hashlib.sha256(w.tobytes()).hexdigest()));out={}
 for c in MODELS:
  if which is not None and c['id'] not in which:continue
  model=LGBMRegressor(objective=c['loss'],alpha=.9,n_estimators=c['iterations'],num_leaves=c['leaves'],learning_rate=.03,min_child_samples=128,reg_lambda=20,max_bin=63,n_jobs=2,random_state=20261006,verbosity=-1,force_col_wise=True,deterministic=True);model.fit(x,y,sample_weight=w);out[c['id']]=model;print('FIT',cutoff,c['id'],len(y),flush=True)
 return out
def inputs(split,H):
 if split=='deploy':
  import pandas as pd
  if H==7:base=z('round26/results/exports/deployment_origin31.npz','selected7')
  else:
   df=pd.read_csv(B/'round26/results/exports/selected_1day_curve.csv');base=df['forecast_normalized_power'].to_numpy().reshape(57,1,96)
  ref=z('round18/results/deployment7_origin31.npz','prior')[:,:,:H*96];return base,ref
 base=z(f'round26/results/{split}_predictions.npz',f'selected{H}')
 ref=z(f'round3/results/tf3_{split}.npz','tf3_uni_context2016') if H==1 else z(f'round18/results/{split}_predictions.npz','prior' if split=='dev' else'prior7');return base,ref
def candidate(d,base,ref,delta,cfg):
 raw=np.clip(base+delta-cfg['debit']*(base-ref),0,1);out=raw.copy()
 for j in range(base.shape[1]):
  for cell in np.unique(d['cells']):
   ix=d['cells']==cell
   if cell=='a':out[ix,j]=base[ix,j];continue
   target=base[ix,j].sum();v=raw[ix,j];lo=-2.;hi=2.
   for _ in range(55):
    mid=(lo+hi)/2
    if np.clip(v+mid,0,1).sum()>target:hi=mid
    else:lo=mid
   out[ix,j]=np.clip(v+(lo+hi)/2,0,1)
   assert abs(out[ix,j].sum()-target)<1e-8
 return out
def prefix_check():
 dev=z('round17/results/native_train7.npz','prediction')[:,7:15,:96];orig=z('round3/results/tf3_dev.npz','tf3_uni_context2016');test=z('round18/results/test_predictions.npz','prior7')[:,:,:96];control=z('round3/results/tf3_test.npz','tf3_uni_context2016')[:,:3];a=float(abs(dev-orig).max());b=float(abs(test-control).max());assert max(a,b)<1e-5;return dict(dev_max_abs_difference=a,test_first_three_max_abs_difference=b,sampled_controls_only=True,deployment_ref_first96_of_saved_native672=True)
def prepare():
 for k in ['code','docs','models','results']:(R/k).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];return
 prefix=prefix_check();files=['results/data.npz','round17/results/native_train7.npz','round17/results/native_train7_verification.json','round22/code/point_residual_experiment.py','round26/results/dev_predictions.npz','round26/results/test_predictions.npz','round26/results/selection.json','round26/results/exports/deployment_origin31.npz','round26/results/exports/selected_1day_curve.csv','round18/results/dev_predictions.npz','round18/results/test_predictions.npz','round18/results/deployment7_origin31.npz','round3/results/tf3_dev.npz','round3/results/tf3_test.npz']
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),metrics_script_sha256=sha(R/'code/source_metrics.py'),models=MODELS,configs=CONFIGS,alphas=ALPHAS,probe_alpha=.1,tolerances=TOL,base='R26 recommended 1day R8b and7day balanced prior',training='DEV labels index<14; final<22. Audited native672 training, fixed stride6 rotated by origin/horizon, inverse repeated-target date weights. Four LightGBM Huber/L1 heads, no neural reload. Residual learning from native forecasts; application to stronger prior remains a transfer hypothesis.',candidate='prior plus robust normalized point residual, optionally subtract half of already-existing prior minus native calibration to avoid double correction. Preserve a5 and each cell forecast total, clip0..1. These normalized forecasts are not physical energy conservation.',selection='For H1 early origins14..17; H7 early14: rank8 configs by fixed10% probe cost subject to minor-tradeoff guard, or best rejected as diagnostic. Confirm weights0/.025/.05/.1/.25 on late H1 origins18..21 or H7 origin15, plus early and aggregate guards, aggregate>=2 distinct improved categories, require late cost<1. Apply paired-date oneSE shrink among eligible. Freeze before scoring exactly one diagnostic or selected candidate. Never TEST rerank; if none alpha0 keepsR26.',prefix_controls=prefix,independent_confirmation=False,test_seen_before=True,source_hashes={f:sha(B/f) for f in files},resource_policy=dict(admission_physical_gib=2.125,admission_commit_gib=6,job_commit_hard_cap_bytes=512*2**20,runtime_physical_floor_gib=1.5,runtime_commit_floor_gib=2)))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii')
def develop(d):
 assert not (Q/'selection.json').exists();models=fit(d,14);pickle.dump(models,(R/'models/dev_heads.pkl').open('wb'));summ={};selected={};arrays={}
 for H in [1,7]:
  origins=list(range(14,22)) if H==1 else[14,15];early_n=4 if H==1 else 1;base,ref=inputs('dev',H);be=score(d,origins[:early_n],base[:,:early_n],H);bl=score(d,origins[early_n:],base[:,early_n:],H);ba=score(d,origins,base,H);deltas={k:residual(d,origins,ref,m,H) for k,m in models.items()};rows=[];preds={}
  for cfg in CONFIGS:
   cp=candidate(d,base,ref,deltas[cfg['model']],cfg);preds[cfg['id']]=cp;probe=.9*base+.1*cp;sc=score(d,origins[:early_n],probe[:,:early_n],H);rows.append(dict(config=cfg,cost=cost(sc,be),scores=sc,guard_passed=guard(sc,be)));print('DEV ROBUST',H,cfg['id'],rows[-1]['cost'],flush=True)
  eligible=[v for v in rows if v['guard_passed']];best=min(eligible or rows,key=lambda v:v['cost']);cfg=best['config'];cp=preds[cfg['id']];weights=[]
  for a in ALPHAS:
   p=(1-a)*base+a*cp;se=score(d,origins[:early_n],p[:,:early_n],H);sl=score(d,origins[early_n:],p[:,early_n:],H);sa=score(d,origins,p,H);ok=a>0 and guard(se,be) and guard(sl,bl) and guard(sa,ba) and len(categories(sa,ba))>=2 and cost(sl,bl)<1;vals=[]
   for j,o in enumerate(origins[early_n:],early_n):
    for h in range(H):vals.append(cost(score(d,[o+h],p[:,j:j+1,h*96:(h+1)*96],1),score(d,[o+h],base[:,j:j+1,h*96:(h+1)*96],1))-1)
   weights.append(dict(alpha=a,cost=cost(sl,bl),paired_date_se=float(np.std(vals,ddof=1)/np.sqrt(len(vals))),eligible=ok,early=se,late=sl,aggregate=sa,improved_categories=categories(sa,ba)))
  good=[v for v in weights if v['eligible']]
  if good:
   win=min(good,key=lambda v:v['cost']);pick=min([v for v in good if v['cost']<=win['cost']+win['paired_date_se']],key=lambda v:v['alpha'])
  else:pick=weights[0]
  selected[str(H)]=dict(candidate=cfg,chosen_alpha=pick['alpha'],diagnostic_alpha=pick['alpha'] if pick['alpha']>0 else.1,development_supported=bool(good));summ[str(H)]=dict(configs=rows,weights=weights,family_early=best,baseline_aggregate=ba);arrays.update({f'prior{H}':base,f'raw_candidate{H}':cp,f'selected{H}':(1-pick['alpha'])*base+pick['alpha']*cp})
  origins_full=origins;y,m=common.truth(d,origins_full,H);pred=arrays[f'selected{H}'];rad=np.array([np.quantile(abs(pred[i]-y[i])[m[i]],.8) for i in range(57)]);np.savez_compressed(R/f'models/interval{H}_radius.npz',radius=rad)
 np.savez_compressed(Q/'dev_predictions.npz',**arrays);save(Q/'dev_summary.json',dict(horizons=summ,testing_not_loaded=True));s=dict(frozen_utc=utc(),horizons=selected,protocol_sha256=sha(R/'docs/protocol.json'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),dev_heads_sha256=sha(R/'models/dev_heads.pkl'),independent_confirmation=False);save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();models=fit(d,22,{v['candidate']['model'] for v in s['horizons'].values()});pickle.dump(models,(R/'models/final_heads.pkl').open('wb'));arrays={};scores={}
 for H in [1,7]:
  origins=list(range(22,31)) if H==1 else[22,23,24];base,ref=inputs('test',H);cfg=s['horizons'][str(H)];cp=candidate(d,base,ref,residual(d,origins,ref,models[cfg['candidate']['model']],H),cfg['candidate']);p=(1-cfg['chosen_alpha'])*base+cfg['chosen_alpha']*cp;diag=(1-cfg['diagnostic_alpha'])*base+cfg['diagnostic_alpha']*cp;arrays.update({f'prior{H}':base,f'raw_candidate{H}':cp,f'diagnostic{H}':diag,f'selected{H}':p});scores[str(H)]={k:score(d,origins,v,H) for k,v in [('prior',base),('candidate',diag),('selected',p)]}
  if cfg['chosen_alpha']==0:lo=z('round26/results/test_predictions.npz',f'lower{H}');hi=z('round26/results/test_predictions.npz',f'upper{H}')
  else:
   rad=z(f'round27/models/interval{H}_radius.npz','radius')[:,None,None];lo=np.clip(p-rad,0,1);hi=np.clip(p+rad,0,1)
  arrays[f'lower{H}']=lo;arrays[f'upper{H}']=hi;y,m=common.truth(d,origins,H);scores[str(H)]['interval_selected']=interval_metrics(y,p,m,lo,hi)
 np.savez_compressed(Q/'test_predictions.npz',**arrays);save(Q/'summary.json',dict(selection=s,scores=scores,no_post_test_selection=True,independent_confirmation=False,final_heads_sha256=sha(R/'models/final_heads.pkl')));print('ACTUAL ROBUST TEST',json.dumps(scores),flush=True)
def deploy(d):
 import pandas as pd
 s=read(Q/'selection.json');models=pickle.load((R/'models/final_heads.pkl').open('rb'));O=Q/'exports';O.mkdir(exist_ok=True);arrays={};meta=dict(all_passed=True,historical_replay=True,no_june_truth=True,selection_sha256=sha(Q/'selection.json'))
 for H in [1,7]:
  cfg=s['horizons'][str(H)];base,ref=inputs('deploy',H);cp=candidate(d,base,ref,residual(d,[31],ref,models[cfg['candidate']['model']],H),cfg['candidate']);p=(1-cfg['chosen_alpha'])*base+cfg['chosen_alpha']*cp;arrays[f'selected{H}']=p;arrays[f'prior{H}']=base
  if cfg['chosen_alpha']==0:df=pd.read_csv(B/f'round26/results/exports/selected_{H}day_curve.csv')
  else:
   rad=z(f'round27/models/interval{H}_radius.npz','radius');stamps=pd.date_range('2019-06-01',periods=H*96,freq='15min',tz='America/Los_Angeles');rows=[]
   for i,name in enumerate(d['names']):
    for j,t in enumerate(stamps):rows.append(dict(pdu=str(name),cell=str(d['cells'][i]),timestamp_PT=t.isoformat(),timestamp_UTC=t.tz_convert('UTC').isoformat(),forecast_normalized_power=p[i,0,j],lower80_empirical=max(0,p[i,0,j]-rad[i]),upper80_empirical=min(1,p[i,0,j]+rad[i]),horizon_days=H,historical_replay=True))
   df=pd.DataFrame(rows)
  point=O/f'selected_{H}day_curve.csv';df.to_csv(point,index=False,encoding='utf-8-sig');df['date_PT']=df['timestamp_PT'].str[:10];daily=O/f'selected_{H}day_daily_mean.csv';df.groupby(['pdu','cell','date_PT'],sort=False)['forecast_normalized_power'].mean().reset_index(name='daily_mean_normalized_power').to_csv(daily,index=False,encoding='utf-8-sig');meta[f'file{H}']=str(point);meta[f'sha{H}']=sha(point);meta[f'daily_file{H}']=str(daily);meta[f'daily_sha{H}']=sha(daily);meta[f'rows{H}']=len(df)
 np.savez_compressed(O/'deployment_origin31.npz',**arrays);meta['source_hashes']={str(p.relative_to(T)):sha(p) for p in [Q/'selection.json',R/'models/final_heads.pkl',B/'round26/results/exports/deployment_origin31.npz',B/'round26/results/exports/selected_1day_curve.csv',B/'round18/results/deployment7_origin31.npz']};save(O/'verification.json',meta);print('DEPLOYMENT VERIFIED',json.dumps(meta),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];d=common.load();{'develop':develop,'test':test,'deploy':deploy}[a.stage](d)
