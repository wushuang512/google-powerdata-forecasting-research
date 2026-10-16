"""History-volatility gating of the DEV-frozen joint point/peak proposal, with conditional forecast-total projection."""
from pathlib import Path
import sys,ast,json,argparse
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round24';Q=R/'results'
src=B/'round23/code/joint_point_peak_experiment.py';e={'__file__':str(src),'__name__':'prior_joint_utilities'}
tree=ast.parse(src.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets));exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(src),'exec'),e)
common=e['common'];read=e['read'];save=e['save'];sha=e['sha'];utc=e['utc'];z=e['z'];score=e['score'];guard=e['guard'];cost=e['cost'];interval_metrics=e['interval_metrics']
CONFIGS=[dict(id='prior',gate='all',centering='global')]+[dict(id=g+'_'+c,gate=g,centering=c) for g in ['all','cv002_010','cv005_020','hold_a5'] for c in ['global','cell']]
ALPHAS=[0.,.1,.25,.5,1.]
def gate(d,o,cfg):
 if cfg['gate']=='all':return np.ones(57)
 if cfg['gate']=='hold_a5':return (d['cells']!='a').astype(float)
 hist=d['y'][:,o-7:o];m=d['valid'][:,o-7:o];count=m.sum((1,2)).clip(1);mu=(hist*m).sum((1,2))/count;std=np.sqrt((((hist-mu[:,None,None])**2)*m).sum((1,2))/count);cv=std/mu.clip(.05)
 low,high=(.002,.010) if cfg['gate']=='cv002_010' else(.005,.020)
 return np.clip((cv-low)/(high-low),0,1)
def candidate(d,orig,base,proposal,cfg):
 if cfg['id']=='prior':return base.copy()
 out=base.copy();groups=[np.arange(57)] if cfg['centering']=='global' else[np.where(d['cells']==c)[0] for c in np.unique(d['cells'])]
 for j,o in enumerate(orig):
  g=gate(d,o,cfg)[:,None];delta=proposal[:,j]-base[:,j];raw=base[:,j]+g*delta
  for ix in groups:
   if not g[ix].any():continue
   lo=-2.;hi=2.;target=base[ix,j].sum()
   for _ in range(55):
    mid=(lo+hi)/2;v=np.clip(raw[ix]+mid*g[ix],0,1)
    if v.sum()>target:hi=mid
    else:lo=mid
   out[ix,j]=np.clip(raw[ix]+((lo+hi)/2)*g[ix],0,1);assert abs(out[ix,j].sum()-target)<1e-8
  assert np.array_equal(out[g[:,0]==0,j],base[g[:,0]==0,j])
 return out
def prepare():
 for k in ['code','docs','models','results']:(R/k).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];return
 files=['results/data.npz','round23/code/joint_point_peak_experiment.py','round23/results/dev_predictions.npz','round23/results/test_predictions.npz','round23/results/dev_summary.json','round23/results/selection.json','round9/results/selected_intervals_test.npz']
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),metrics_script_sha256=sha(R/'code/source_metrics.py'),configs=CONFIGS,alphas=ALPHAS,motivation='R23 DEV7 curve and hit thresholds improve but late P95 and a5 guard fail. Prospectively evaluate whether history-volatility gate and cell consistency reduce unstable corrections; no test-selected domain/date routing.',gate='Observed past7day coefficient of variation only: soft ramps.002..010 or.005..020, no PDU-specific thresholds. hold_a5 is a predeclared known-source branch because DEV a5 regression and distinct complete CPU/resource mapping; never chosen by TEST group scores.',projection='Gate multiplies joint residual, then scalar total correction weighted by same gate within global/cell group. Inactive domains exactly retain prior. Conserves normalized forecast total, not physical energy.',selection='Unchanged R20 eight-core guard, earlyDEV24 orDEV7 firstorigin choose one family, laterDEV24 orDEV7 second plus aggregates choose0/.1/.25/.5/1, oneSE shrink, else0. One frozen diagnostic fixed.25 or selected positive. No TEST rank reselection.',independent_confirmation=False,test_seen_before=True,source_hashes={f:sha(B/f) for f in files},resource_policy=dict(admission_physical_gib=2.125,admission_commit_gib=6,job_commit_hard_cap_bytes=512*2**20,runtime_physical_floor_gib=1.5,runtime_commit_floor_gib=2)))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii')
def develop(d):
 assert not (Q/'selection.json').exists();summ={};frozen={};saved={}
 for H in [1,7]:
  orig=list(range(14,22)) if H==1 else[14,15];base=z('round23/results/dev_predictions.npz',f'prior{H}');proposal=z('round23/results/dev_predictions.npz',f'candidate{H}');count=4 if H==1 else 1;be=score(d,orig[:count],base[:,:count],H);bl=score(d,orig[count:],base[:,count:],H);ba=score(d,orig,base,H);rows=[];preds={}
  for cfg in CONFIGS:
   p=candidate(d,orig,base,proposal,cfg);preds[cfg['id']]=p;sc=score(d,orig[:count],p[:,:count],H);rows.append(dict(config=cfg,scores=sc,cost=cost(sc,be),guard_passed=guard(sc,be)));print('DEV GATE',H,cfg['id'],rows[-1]['cost'],rows[-1]['guard_passed'],flush=True)
  pool=[r for r in rows if r['config']['id']!='prior'];el=[r for r in pool if r['guard_passed']];best=min(el or pool,key=lambda x:x['cost']);c=best['config'];cp=preds[c['id']];weights=[]
  for a in ALPHAS:
   p=(1-a)*base+a*cp;se=score(d,orig[:count],p[:,:count],H);sl=score(d,orig[count:],p[:,count:],H);sa=score(d,orig,p,H);ok=a>0 and guard(se,be) and guard(sl,bl) and guard(sa,ba) and any(sa[k]<ba[k]-.0001 for k in ['wape','peak_mape']);vals=[]
   for j,o in enumerate(orig[count:],count):
    for h in range(H):vals.append(cost(score(d,[o+h],p[:,j:j+1,h*96:(h+1)*96],1),score(d,[o+h],base[:,j:j+1,h*96:(h+1)*96],1))-1)
   weights.append(dict(alpha=a,early=se,late=sl,aggregate=sa,cost=cost(sl,bl),paired_date_se=float(np.std(vals,ddof=1)/np.sqrt(len(vals))),eligible=ok))
  el=[w for w in weights if w['eligible']]
  if el:
   win=min(el,key=lambda x:x['cost']);pick=min([w for w in el if w['cost']<=win['cost']+win['paired_date_se']],key=lambda x:x['alpha'])
  else:pick=weights[0]
  frozen[str(H)]=dict(candidate=c,chosen_alpha=pick['alpha'],diagnostic_alpha=pick['alpha'] if pick['alpha']>0 else.25,development_supported=bool(el));summ[str(H)]=dict(configs=rows,weights=weights,baseline_aggregate=ba,family_early=best);saved[f'prior{H}']=base;saved[f'candidate{H}']=cp;saved[f'selected{H}']=(1-pick['alpha'])*base+pick['alpha']*cp
 np.savez_compressed(Q/'dev_predictions.npz',**saved);save(Q/'dev_summary.json',dict(horizons=summ,testing_not_loaded=True));y,m=common.truth(d,[14,15],7);rad=np.array([np.quantile(abs(saved['selected7'][i]-y[i])[m[i]],.8) for i in range(57)]);np.savez_compressed(R/'models/interval7_radius.npz',radius=rad)
 s=dict(frozen_utc=utc(),horizons=frozen,protocol_sha256=sha(R/'docs/protocol.json'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),interval_radius_sha256=sha(R/'models/interval7_radius.npz'),independent_confirmation=False);save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();arrays={};scores={}
 for H in [1,7]:
  orig=list(range(22,31)) if H==1 else[22,23,24];base=z('round23/results/test_predictions.npz',f'prior{H}');proposal=z('round23/results/test_predictions.npz',f'candidate{H}');c=s['horizons'][str(H)];cp=candidate(d,orig,base,proposal,c['candidate']);p=(1-c['chosen_alpha'])*base+c['chosen_alpha']*cp;diag=(1-c['diagnostic_alpha'])*base+c['diagnostic_alpha']*cp;arrays.update({f'prior{H}':base,f'candidate{H}':cp,f'selected{H}':p,f'diagnostic{H}':diag});scores[str(H)]={k:score(d,orig,v,H) for k,v in dict(prior=base,candidate=diag,selected=p).items()}
  if H==7:
   rad=z('round24/models/interval7_radius.npz','radius')[:,None,None];lo=np.clip(p-rad,0,1);hi=np.clip(p+rad,0,1)
  else:lo=np.clip(z('round9/results/selected_intervals_test.npz','lower')+p-base,0,1);hi=np.clip(z('round9/results/selected_intervals_test.npz','upper')+p-base,0,1)
  arrays[f'lower{H}']=lo;arrays[f'upper{H}']=hi;y,m=common.truth(d,orig,H);scores[str(H)]['interval_selected']=interval_metrics(y,p,m,lo,hi)
 np.savez_compressed(Q/'test_predictions.npz',**arrays);save(Q/'summary.json',dict(selection=s,scores=scores,no_post_test_selection=True,independent_confirmation=False));print('ACTUAL GATE TEST',json.dumps(scores),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];d=common.load();{'develop':develop,'test':test}[a.stage](d)

