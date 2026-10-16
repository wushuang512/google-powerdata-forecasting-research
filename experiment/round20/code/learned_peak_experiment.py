"""Conservative learned peak amplitude and near-history peak location projection, DEV-frozen."""
from pathlib import Path
import sys,ast,json,argparse,pickle,datetime
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round20';Q=R/'results'
sys.path.insert(0,str(R/'code'));from source_metrics import evaluate,interval_metrics
src=B/'round19/code/structured_projection_experiment.py';e={'__file__':str(src),'__name__':'immutable_round19_utilities'}
tree=ast.parse(src.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets));exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(src),'exec'),e)
common=e['common'];read=e['read'];save=e['save'];sha=e['sha'];utc=e['utc'];z=e['z'];features=e['e']['features'];OLDMETRICS=e['METRICS']
CORE=['wape','mae','rmse','daily_wape','p95_ape','peak_mape','hit2_percent','hit5_percent']
CONFIGS=[dict(id='prior',strength=0.,gate=False,location='prior')]+[dict(id=f'learnedpeak_s{int(s*100)}_gate{int(g)}_{loc}',strength=s,gate=g,location=loc) for s in [.1,.25,.5] for g in [False,True] for loc in ['prior','near_history']]
ALPHAS=[0.,.25,.5,1.]
def score(d,origins,p,H):return evaluate(common,d,origins,p,H)
def cost(s,b):
 ratios=[(100-s[k])/max(100-b[k],1e-8) if k.startswith('hit') else s[k]/max(b[k],1e-8) for k in CORE]
 return .7*float(np.mean(ratios))+.3*s['peak_mape']/max(b['peak_mape'],1e-8)
def guard(s,b):
 return all(s[k]<=b[k]+1e-9 for k in ['wape','mae','rmse','daily_wape','p95_ape']) and all(s[k]>=b[k]-1e-9 for k in ['hit2_percent','hit5_percent']) and s['peak_mape']<=b['peak_mape']+.001 and all(s[k]<=b[k]+.001 for k in ['cella5_wape','other52_wape']) and s['absolute_bias_percent']<=b['absolute_bias_percent']+.005 and s['peak_time_mae_minutes']<=b['peak_time_mae_minutes']+15
def load_base(split,H):
 if H==1:return z('round11/results/dev_predictions.npz','prior') if split=='dev' else z('round8b/results/test_predictions.npz','selected')
 if split=='dev':
  b=z('round18/results/dev_predictions.npz','prior');return .75*b+.25*z('round19/results/dev_predictions.npz','global_peak0')
 return z('round19/results/test_predictions.npz','selected7')
def reference(split,H):
 if H==1:return z(f'round3/results/tf3_{split}.npz','tf3_uni_context2016')
 return z('round18/results/dev_predictions.npz','prior') if split=='dev' else z('round18/results/test_predictions.npz','prior7')
def models(final=False):return pickle.load((B/f'round18/models/{"final" if final else "dev"}_heads.pkl').open('rb'))['zero'][1000]
def details(d,origins,ref,model,H):
 out=[]
 for j,o in enumerate(origins):
  day=[]
  for h in range(H):
   x,scale,hist=features(d,o,h,ref[:,j,h*96:(h+1)*96]);q=np.clip(model.predict(x)[:,7],-.05,.05)*scale;target=ref[:,j,h*96:(h+1)*96].max(-1)+q
   day.append((target,scale,hist))
  out.append(day)
 return out
def candidate(base,det,cfg,H):
 if cfg['id']=='prior':return base.copy()
 out=base.copy()
 for j,days in enumerate(det):
  for h,(target,scale,hist) in enumerate(days):
   b=base[:,j,h*96:(h+1)*96];mx=b.max(-1);delta=np.clip(target-mx,-.01*scale,.01*scale)*cfg['strength']
   if cfg['gate']:delta=np.where((target-mx)*(hist.max(-1)-mx)>0,delta,0)
   top=np.clip(mx+delta,b.mean(-1),1);ix=b.argmax(-1)
   if cfg['location']=='near_history':
    anchor=hist.argmax(-1);slots=np.arange(96)[None,:];eligible=(b>=mx[:,None]-.003*scale[:,None])&(abs(slots-ix[:,None])<=8)
    ix=np.argmin(np.where(eligible,abs(slots-anchor[:,None]),999),axis=-1)
   lo=np.full(57,-2.);hi=np.full(57,2.);total=b.sum(-1)
   for _ in range(55):
    mid=(lo+hi)/2;v=np.clip(b+mid[:,None],0,top[:,None]);v[np.arange(57),ix]=top;large=v.sum(-1)>total;hi=np.where(large,mid,hi);lo=np.where(large,lo,mid)
   v=np.clip(b+((lo+hi)/2)[:,None],0,top[:,None]);v[np.arange(57),ix]=top
   zero=abs(delta)<1e-15;v[zero]=b[zero]
   assert np.max(abs(v.sum(-1)-total))<1e-8 and np.max(abs(v.max(-1)-top))<1e-10
   out[:,j,h*96:(h+1)*96]=v
 assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
 return out
def prepare():
 for k in ['code','docs','models','results']:(R/k).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];return
 files=['results/data.npz','code/common.py','round18/code/horizon_routed_experiment.py','round19/code/structured_projection_experiment.py','round18/models/dev_heads.pkl','round18/models/final_heads.pkl','round18/results/dev_predictions.npz','round18/results/test_predictions.npz','round19/results/dev_predictions.npz','round19/results/test_predictions.npz','round19/results/deployment7_origin31.npz','round18/results/deployment7_origin31.npz','round11/results/dev_predictions.npz','round8b/results/test_predictions.npz','round8b/results/selected_deployment_origin31_1day.csv','round9/results/selected_intervals_test.npz','round3/results/tf3_dev.npz','round3/results/tf3_test.npz','model_comparison_20261005/results/task_revision/summary.json']
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),metrics_script_sha256=sha(R/'code/source_metrics.py'),reference_document=str(T/'参考材料/9.28算力中心需求预测实验.docx'),reference_document_sha256=sha(T/'参考材料/9.28算力中心需求预测实验.docx'),configs=CONFIGS,alphas=ALPHAS,core=CORE,base24='R8b unchanged unless DEV supports this candidate',base7='R19 global-total Ridge1000 alpha25 current research scheme',method='Native TimesFM3 learned daily-peak head transferred conservatively to current base, with sum-preserving bounded daily projection and optional near-history peak location.',training='Reuse R18 zero-intercept Ridge1000 head trained only target dates<14 DEV and<22 final; no new network training. Peak head was trained on native7 residuals. For H1 use its h0 prediction around native21day96 reference, not an unreported trained96 head.',projection='Target peak correction capped at1% observed7day mean then strengths.1/.25/.5. Preserve full96 mean and[0,1], optional peak index nearest historical last-day max among forecast points within0.3%historical mean of current forecast max and8 slots of original argmax. No future truth or validity masks.',family_selection='For each known H separately use early DEV24 origins14..17 or DEV7 origin14; prefer guard-eligible nonprior minimum normalized8metric cost with peak30%extra weight. If none, freeze minimum-cost rejected diagnostic. Families do not read TEST.',strength_selection='DEV24 later18..21; DEV7 origin15 and aggregate14/15. Strengths0/.25/.5/1. Require no WAPE/MAE/RMSE/daily/P95 or hit2/hit5 regression beyond floating tolerance; peak+.001pp, groupWAPE+.001pp, absolutebias+.005pp and timing+15min guards. At least.0001pp aggregate curve or peak improvement. Prefer lowestcost then shrink within paired target-date1SE. If none0. Each horizon branch frozen once before TEST.',diagnostic='Only one family per horizon; diagnostic alpha equals selected if positive else.25, fixed before TEST. All configurations retained on DEV, no TEST family rank search.',intervals='Existing R8b80% intervals recentered at frozen H1 candidate without changing width. H7 empirical absolute-error80th percentile per PDU fitted on both completed DEV windows, then used for final TEST and June replay. Same DEV used for selection; overlap and small sample mean exploratory coverage, no guarantee. Interval width/Winkler also reported. No TEST width adjustment.',development24=list(range(14,22)),development7=[14,15],test24=list(range(22,31)),test7=[22,23,24],independent_confirmation=False,test_seen_before=True,source_hashes={f:sha(B/f) for f in files},resource_policy=dict(admission_physical_gib=3,admission_commit_gib=6,runtime_physical_gib=1.5,runtime_commit_gib=2)))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii');print('PROTOCOL FROZEN',flush=True)
def develop(d):
 assert not (Q/'selection.json').exists()
 summary={};frozen={};saved={}
 for H in [1,7]:
  origins=list(range(14,22)) if H==1 else [14,15];base=load_base('dev',H);ref=reference('dev',H);det=details(d,origins,ref,models(),H);count=4 if H==1 else 1;be=score(d,origins[:count],base[:,:count],H);bl=score(d,origins[count:],base[:,count:],H);ba=score(d,origins,base,H);rows=[];preds={}
  for cfg in CONFIGS:
   p=candidate(base,det,cfg,H);preds[cfg['id']]=p;sc=score(d,origins[:count],p[:,:count],H);rows.append(dict(config=cfg,scores=sc,guard_passed=guard(sc,be),cost=cost(sc,be)));print('DEV',H,cfg['id'],rows[-1]['cost'],rows[-1]['guard_passed'],flush=True)
  pool=[x for x in rows if x['config']['id']!='prior'];eligible=[x for x in pool if x['guard_passed']];best=min(eligible or pool,key=lambda x:x['cost']);cfg=best['config'];cp=preds[cfg['id']];weights=[]
  for a in ALPHAS:
   p=(1-a)*base+a*cp;se=score(d,origins[:count],p[:,:count],H);sl=score(d,origins[count:],p[:,count:],H);sa=score(d,origins,p,H);ok=a>0 and guard(se,be) and guard(sl,bl) and guard(sa,ba) and any(sa[k]<ba[k]-.0001 for k in ['wape','peak_mape'])
   vals=[]
   for j,o in enumerate(origins[count:],count):
    for h in range(H):
     ss=score(d,[o+h],p[:,j:j+1,h*96:(h+1)*96],1);bb=score(d,[o+h],base[:,j:j+1,h*96:(h+1)*96],1);vals.append(cost(ss,bb)-1)
   weights.append(dict(alpha=a,early=se,late=sl,aggregate=sa,cost=cost(sl,bl),paired_date_se=float(np.std(vals,ddof=1)/np.sqrt(len(vals))),eligible=ok))
  el=[x for x in weights if x['eligible']]
  if el:
   win=min(el,key=lambda x:x['cost']);pick=min([x for x in el if x['cost']<=win['cost']+win['paired_date_se']],key=lambda x:x['alpha'])
  else:pick=weights[0]
  frozen[str(H)]=dict(candidate=cfg,chosen_alpha=pick['alpha'],diagnostic_alpha=pick['alpha'] if pick['alpha']>0 else.25,development_supported=bool(el));summary[str(H)]=dict(configs=rows,weights=weights,baseline_aggregate=ba,family_early=best)
  saved[f'prior{H}']=base;saved[f'candidate{H}']=cp;saved[f'selected{H}']=(1-pick['alpha'])*base+pick['alpha']*cp
 np.savez_compressed(Q/'dev_predictions.npz',**saved);save(Q/'dev_summary.json',dict(horizons=summary,testing_not_loaded=True,overlapping_dev7=True))
 y,m=common.truth(d,[14,15],7);rad=np.array([np.quantile(abs(saved['selected7'][i]-y[i])[m[i]],.8) for i in range(57)]);np.savez_compressed(R/'models/interval7_radius.npz',radius=rad)
 sel=dict(frozen_utc=utc(),horizons=frozen,protocol_sha256=sha(R/'docs/protocol.json'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),interval_radius_sha256=sha(R/'models/interval7_radius.npz'),independent_confirmation=False);save(Q/'selection.json',sel);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(sel),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();assert sha(Q/'dev_summary.json')==s['dev_summary_sha256'] and sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256'];scores={};arrays={}
 for H in [1,7]:
  origins=list(range(22,31)) if H==1 else [22,23,24];base=load_base('test',H);ref=reference('test',H);cfg=s['horizons'][str(H)];cp=candidate(base,details(d,origins,ref,models(True),H),cfg['candidate'],H);p=(1-cfg['chosen_alpha'])*base+cfg['chosen_alpha']*cp;diag=(1-cfg['diagnostic_alpha'])*base+cfg['diagnostic_alpha']*cp;arrays.update({f'prior{H}':base,f'candidate{H}':cp,f'selected{H}':p,f'diagnostic{H}':diag});scores[str(H)]={k:score(d,origins,v,H) for k,v in dict(prior=base,candidate=diag,selected=p).items()}
  if H==7:
   rad=z('round20/models/interval7_radius.npz','radius')[:,None,None];lower=np.clip(p-rad,0,1);upper=np.clip(p+rad,0,1)
  else:
   lower=z('round9/results/selected_intervals_test.npz','lower')+(p-base);upper=z('round9/results/selected_intervals_test.npz','upper')+(p-base);lower=np.clip(lower,0,1);upper=np.clip(upper,0,1)
  arrays[f'lower{H}']=lower;arrays[f'upper{H}']=upper;y,m=common.truth(d,origins,H);scores[str(H)]['interval_selected']=interval_metrics(y,p,m,lower,upper)
 np.savez_compressed(Q/'test_predictions.npz',**arrays);save(Q/'summary.json',dict(selection=s,scores=scores,no_post_test_selection=True,independent_confirmation=False,actual_test_completed=True));print('ACTUAL TEST',json.dumps(scores),flush=True)
def deploy(d):
 s=read(Q/'selection.json');saved={}
 import pandas as pd
 old=pd.read_csv(B/'round8b/results/selected_deployment_origin31_1day.csv');base1=np.stack([old.loc[old['series']==name,'predicted_util'].to_numpy() for name in d['names']])[:,None,:];base7=z('round19/results/deployment7_origin31.npz','selected');ref7=z('round18/results/deployment7_origin31.npz','prior')
 # First96 native7 is the native21day96 reference; original canary was reproduced to1e-5.
 for H,base,ref in [(1,base1,ref7[:,:,:96]),(7,base7,ref7)]:
  c=s['horizons'][str(H)];cp=candidate(base,details(d,[31],ref,models(True),H),c['candidate'],H);p=(1-c['chosen_alpha'])*base+c['chosen_alpha']*cp;saved[f'selected{H}']=p
  if H==7:
   rad=z('round20/models/interval7_radius.npz','radius')[:,None,None];lo=np.clip(p-rad,0,1);hi=np.clip(p+rad,0,1)
  else:
   lo=np.stack([old.loc[old['series']==name,'lower80_empirical'].to_numpy() for name in d['names']])[:,None,:]+p-base;hi=np.stack([old.loc[old['series']==name,'upper80_empirical'].to_numpy() for name in d['names']])[:,None,:]+p-base;lo=np.clip(lo,0,1);hi=np.clip(hi,0,1)
  saved[f'lower{H}']=lo;saved[f'upper{H}']=hi
  times=pd.date_range('2019-06-01',periods=96*H,freq='15min',tz='America/Los_Angeles');rows=[]
  for i,name in enumerate(d['names']):
   for j,t in enumerate(times):rows.append(dict(pdu=str(name),cell=str(d['cells'][i]),timestamp_PT=t.isoformat(),timestamp_UTC=t.tz_convert('UTC').isoformat(),forecast_normalized_power=p[i,0,j],lower80_empirical=lo[i,0,j],upper80_empirical=hi[i,0,j],horizon_days=H,peak_correction_alpha=c['chosen_alpha'],historical_replay=True))
  df=pd.DataFrame(rows);assert len(df)==57*H*96;df.to_csv(Q/f'selected_{H}day_curve.csv',index=False,encoding='utf-8-sig');df['date_PT']=df['timestamp_PT'].str[:10];dm=df.groupby(['pdu','cell','date_PT'],sort=False)['forecast_normalized_power'].mean().reset_index(name='daily_mean_normalized_power');dm.to_csv(Q/f'selected_{H}day_daily_mean.csv',index=False,encoding='utf-8-sig')
 np.savez_compressed(Q/'deployment_origin31.npz',**saved);save(Q/'deployment_verification.json',dict(rows1=5472,rows7=38304,daily_rows7=399,finite_bounded=all(np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in saved.values()),selection_sha256=sha(Q/'selection.json'),historical_replay=True,no_june_truth=True,trace_timezone='America/Los_Angeles'));print('ACTUAL DEPLOYMENT COMPLETE',flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(R/'docs/protocol.json')==(R/'docs/protocol.sha256').read_text() and sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];d=common.load();{'develop':develop,'test':test,'deploy':deploy}[a.stage](d)

