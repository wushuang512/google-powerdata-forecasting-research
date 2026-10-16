"""Separate daily-level and intraday shape gains on the frozen R24 forecast correction."""
from pathlib import Path
import sys,ast,argparse,json
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round25';Q=R/'results'
src=B/'round24/code/volatility_gate_experiment.py';e={'__file__':str(src),'__name__':'prior_pure_utilities'};tree=ast.parse(src.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets));exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(src),'exec'),e)
common=e['common'];read=e['read'];save=e['save'];sha=e['sha'];utc=e['utc'];z=e['z'];score=e['score'];guard=e['guard'];cost=e['cost'];interval_metrics=e['interval_metrics']
CONFIGS=[dict(id='prior',mean_gain=1.,shape_gain=1.,smooth=1)]+[dict(id=f'mean{int(m*100)}_shape{int(s*100)}_smooth{w}',mean_gain=m,shape_gain=s,smooth=w) for m in [.5,1.,1.5] for s in [0.,.5,1.] for w in ([1] if s==0 else[1,3]) if (m,s,w)!=(1.,1.,1)]
ALPHAS=[0.,.25,.5,1.]
def candidate(d,base,original,cfg):
 if cfg['id']=='prior':return base.copy()
 from scipy.ndimage import uniform_filter1d
 correction=(base-original).reshape(57,base.shape[1],7,96);mean=correction.mean(-1,keepdims=True);shape=correction-mean
 sm=shape if cfg['smooth']==1 else uniform_filter1d(shape,size=cfg['smooth'],axis=-1,mode='nearest');sm=sm-sm.mean(-1,keepdims=True);change=(cfg['mean_gain']-1)*mean+(cfg['shape_gain']*sm-shape);raw=np.clip(base+change.reshape(base.shape),0,1);out=raw.copy()
 for j in range(base.shape[1]):
  for cell in np.unique(d['cells']):
   ix=d['cells']==cell;b=base[ix,j];q=raw[ix,j]
   if np.array_equal(q,b):continue
   lo=-2.;hi=2.;target=b.sum()
   for _ in range(55):
    mid=(lo+hi)/2;v=np.clip(q+mid,0,1)
    if v.sum()>target:hi=mid
    else:lo=mid
   out[ix,j]=np.clip(q+(lo+hi)/2,0,1);assert abs(out[ix,j].sum()-target)<1e-8
 assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1 and np.array_equal(out[d['cells']=='a'],base[d['cells']=='a'])
 return out
def prepare():
 for k in ['code','docs','models','results']:(R/k).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];return
 files=['results/data.npz','round24/code/volatility_gate_experiment.py','round24/results/selection.json','round24/results/dev_predictions.npz','round24/results/test_predictions.npz','round24/results/independent_verification.json','round24/results/exports/verification.json']
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),metrics_script_sha256=sha(R/'code/source_metrics.py'),configs=CONFIGS,alphas=ALPHAS,base='R24 DEV frozen selected7. H1 unchanged R8b, no duplicate H1 search.',method='Decompose R24 minus R19 prediction into full96 daily-level and zero-mean intraday shape; mean gains.5/1/1.5, shape0/.5/1, shape smoothing1/3 with perday recentering. Clip0..1 and percell forecast-total preservation, a5 unchanged.',selection='EarlyDEV7 origin14 select strict R20 guard-eligible minimum8metriccost nonprior or minimumcost rejected diagnostic. Later origin15 plus aggregate14/15 choose0/.25/.5/1 with same no-core-regression guards and .0001pp curve/peak gain; oneSE shrink. Exactly one fixed diagnostic.25 or chosen positive before TEST. No TEST reranking.',independent_confirmation=False,test_seen_before=True,source_hashes={f:sha(B/f) for f in files},resource_policy=dict(admission_physical_gib=2.125,admission_commit_gib=6,job_commit_hard_cap_bytes=512*2**20,runtime_physical_floor_gib=1.5,runtime_commit_floor_gib=2)))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii')
def develop(d):
 assert not (Q/'selection.json').exists();base=z('round24/results/dev_predictions.npz','selected7');orig=z('round24/results/dev_predictions.npz','prior7');be=score(d,[14],base[:,:1],7);bl=score(d,[15],base[:,1:],7);ba=score(d,[14,15],base,7);rows=[];preds={}
 for c in CONFIGS:
  p=candidate(d,base,orig,c);preds[c['id']]=p;s=score(d,[14],p[:,:1],7);rows.append(dict(config=c,scores=s,cost=cost(s,be),guard_passed=guard(s,be)));print('DEV LEVEL SHAPE',c['id'],rows[-1]['cost'],rows[-1]['guard_passed'],flush=True)
 pool=[r for r in rows if r['config']['id']!='prior'];elig=[r for r in pool if r['guard_passed']];best=min(elig or pool,key=lambda x:x['cost']);cfg=best['config'];cp=preds[cfg['id']];weights=[]
 for a in ALPHAS:
  p=(1-a)*base+a*cp;se=score(d,[14],p[:,:1],7);sl=score(d,[15],p[:,1:],7);sa=score(d,[14,15],p,7);ok=a>0 and guard(se,be) and guard(sl,bl) and guard(sa,ba) and any(sa[k]<ba[k]-.0001 for k in ['wape','peak_mape']);vals=[cost(score(d,[15+h],p[:,1:2,h*96:(h+1)*96],1),score(d,[15+h],base[:,1:2,h*96:(h+1)*96],1))-1 for h in range(7)];weights.append(dict(alpha=a,early=se,late=sl,aggregate=sa,cost=cost(sl,bl),paired_date_se=float(np.std(vals,ddof=1)/np.sqrt(7)),eligible=ok))
 el=[w for w in weights if w['eligible']]
 if el:
  win=min(el,key=lambda x:x['cost']);pick=min([w for w in el if w['cost']<=win['cost']+win['paired_date_se']],key=lambda x:x['alpha'])
 else:pick=weights[0]
 prior1=z('round24/results/dev_predictions.npz','selected1');selected=(1-pick['alpha'])*base+pick['alpha']*cp;np.savez_compressed(Q/'dev_predictions.npz',prior7=base,candidate7=cp,selected7=selected,selected1=prior1);save(Q/'dev_summary.json',dict(horizons={'7':dict(configs=rows,weights=weights,baseline_aggregate=ba,family_early=best)},testing_not_loaded=True))
 y,m=common.truth(d,[14,15],7);rad=np.array([np.quantile(abs(selected[i]-y[i])[m[i]],.8) for i in range(57)]);np.savez_compressed(R/'models/interval7_radius.npz',radius=rad)
 s=dict(frozen_utc=utc(),horizons={'1':dict(candidate={'id':'prior'},chosen_alpha=0.,diagnostic_alpha=0.,development_supported=False),'7':dict(candidate=cfg,chosen_alpha=pick['alpha'],diagnostic_alpha=pick['alpha'] if pick['alpha']>0 else.25,development_supported=bool(el))},protocol_sha256=sha(R/'docs/protocol.json'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),interval_radius_sha256=sha(R/'models/interval7_radius.npz'),independent_confirmation=False);save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();base=z('round24/results/test_predictions.npz','selected7');orig=z('round24/results/test_predictions.npz','prior7');c=s['horizons']['7'];cp=candidate(d,base,orig,c['candidate']);p=(1-c['chosen_alpha'])*base+c['chosen_alpha']*cp;diag=(1-c['diagnostic_alpha'])*base+c['diagnostic_alpha']*cp;arrays=dict(prior7=base,candidate7=cp,selected7=p,diagnostic7=diag)
 for k in ['prior1','candidate1','selected1','diagnostic1','lower1','upper1']:arrays[k]=z('round24/results/test_predictions.npz',k)
 rad=z('round25/models/interval7_radius.npz','radius')[:,None,None];lo=np.clip(p-rad,0,1);hi=np.clip(p+rad,0,1);arrays['lower7']=lo;arrays['upper7']=hi;scores={'1':read(B/'round24/results/summary.json')['scores']['1'],'7':{k:score(d,[22,23,24],v,7) for k,v in dict(prior=base,candidate=diag,selected=p).items()}};y,m=common.truth(d,[22,23,24],7);scores['7']['interval_selected']=interval_metrics(y,p,m,lo,hi);np.savez_compressed(Q/'test_predictions.npz',**arrays);save(Q/'summary.json',dict(selection=s,scores=scores,no_post_test_selection=True,independent_confirmation=False));print('ACTUAL LEVEL SHAPE TEST',json.dumps(scores['7']),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];d=common.load();{'develop':develop,'test':test}[a.stage](d)

