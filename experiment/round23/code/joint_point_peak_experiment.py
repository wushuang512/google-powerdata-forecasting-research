"""Joint pointwise and sparse peak residual corrections from independently trained native heads."""
from pathlib import Path
import sys,ast,json,argparse
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';R=B/'round23';Q=R/'results'
src=B/'round21/code/localized_peak_experiment.py';e={'__file__':str(src),'__name__':'prior_sparse_peak_utilities'}
tree=ast.parse(src.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets));exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(src),'exec'),e)
common=e['common'];read=e['read'];save=e['save'];sha=e['sha'];utc=e['utc'];z=e['z'];score=e['score'];guard=e['guard'];cost=e['cost'];interval_metrics=e['interval_metrics']
CONFIGS=[dict(id='prior',point=0.,peak=0.)]+[dict(id=f'joint_point{int(p*100)}_peak{int(k*100)}',point=p,peak=k) for p in [0.,.05,.1,.25] for k in [0.,.1,.25,.5] if p or k]
ALPHAS=[0.,.25,.5,1.]
def peak_cfg(H):return read(B/'round21/results/selection.json')['horizons'][str(H)]['candidate']
def get(split,H):
 b=z(f'round22/results/{split}_predictions.npz',f'prior{H}');p=z(f'round22/results/{split}_predictions.npz',f'candidate{H}')
 return b,p
def candidate(base,point,det,cfg,H):
 if cfg['id']=='prior':return base.copy()
 b=(1-cfg['point'])*base+cfg['point']*point
 peak=e['candidate'](b,det,peak_cfg(H),H)
 return (1-cfg['peak'])*b+cfg['peak']*peak
def prepare():
 for k in ['code','docs','models','results']:(R/k).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];return
 files=['results/data.npz','round21/code/localized_peak_experiment.py','round21/results/dev_summary.json','round21/results/selection.json','round22/results/dev_summary.json','round22/results/selection.json','round22/results/dev_predictions.npz','round22/results/test_predictions.npz','round18/models/dev_heads.pkl','round18/models/final_heads.pkl','round3/results/tf3_dev.npz','round3/results/tf3_test.npz','round9/results/selected_intervals_test.npz']
 s=read(B/'round22/results/selection.json');assert all(s['horizons'][str(H)]['candidate']['id']=='hgb_absolute_error_leaves7_global' for H in [1,7])
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),metrics_script_sha256=sha(R/'code/source_metrics.py'),configs=CONFIGS,alphas=ALPHAS,training='Reuse R22 native pointwise HGB absolute-loss7leaves60iterations trained target dates<14 DEV and<22 final, plus R18 zero-intercept Ridge1000 peak head under same cutoffs. No new backbone loading.',method='Pointwise correction applied first, then R21 DEV-selected sparse daily-peak correction preserving corrected daily mean. Components selected from their own DEV records, not TEST scores.',motivation='R22 DEV7 strengths.05/.1 improved curve/hit2/hit5 but worsened peaks; R20/21 DEV peak correction had complementary peak gains and point-tail tradeoffs. Joint candidate hypothesis from development evidence.',selection='Per known H separately: earlyDEV24 origins14..17 orDEV7 origin14 select minimum8metriccost guard-eligible nonprior, else minimum-cost diagnostic. LaterDEV24 origins18..21 orDEV7 origin15 plus aggregates confirm common strength0/.25/.5/1 with unchanged strict R20 guard and .0001pp curve/peak improvement. One-SE shrink, otherwise0. Exactly one diagnostic at fixed.25 or positive strength frozen before TEST, never TEST rank selection.',component_peak_configs={str(H):peak_cfg(H) for H in [1,7]},independent_confirmation=False,test_seen_before=True,resource_policy=dict(admission_physical_gib=2.125,admission_commit_gib=6,job_commit_hard_cap_bytes=512*2**20,runtime_physical_floor_gib=1.5,runtime_commit_floor_gib=2),source_hashes={f:sha(B/f) for f in files}))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii')
def develop(d):
 assert not (Q/'selection.json').exists();summary={};frozen={};saved={}
 for H in [1,7]:
  orig=list(range(14,22)) if H==1 else [14,15];base,point=get('dev',H);det=e['details'](d,orig,e['reference']('dev',H),e['models'](),H);count=4 if H==1 else 1;be=score(d,orig[:count],base[:,:count],H);bl=score(d,orig[count:],base[:,count:],H);ba=score(d,orig,base,H);rows=[];preds={}
  for c in CONFIGS:
   p=candidate(base,point,det,c,H);preds[c['id']]=p;s=score(d,orig[:count],p[:,:count],H);rows.append(dict(config=c,scores=s,cost=cost(s,be),guard_passed=guard(s,be)));print('DEV JOINT',H,c['id'],rows[-1]['cost'],rows[-1]['guard_passed'],flush=True)
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
  frozen[str(H)]=dict(candidate=c,chosen_alpha=pick['alpha'],diagnostic_alpha=pick['alpha'] if pick['alpha']>0 else.25,development_supported=bool(el));summary[str(H)]=dict(configs=rows,weights=weights,baseline_aggregate=ba,family_early=best);saved[f'prior{H}']=base;saved[f'candidate{H}']=cp;saved[f'selected{H}']=(1-pick['alpha'])*base+pick['alpha']*cp
 np.savez_compressed(Q/'dev_predictions.npz',**saved);save(Q/'dev_summary.json',dict(horizons=summary,testing_not_loaded=True));y,m=common.truth(d,[14,15],7);rad=np.array([np.quantile(abs(saved['selected7'][i]-y[i])[m[i]],.8) for i in range(57)]);np.savez_compressed(R/'models/interval7_radius.npz',radius=rad)
 s=dict(frozen_utc=utc(),horizons=frozen,protocol_sha256=sha(R/'docs/protocol.json'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),interval_radius_sha256=sha(R/'models/interval7_radius.npz'),independent_confirmation=False);save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();arrays={};scores={}
 for H in [1,7]:
  orig=list(range(22,31)) if H==1 else [22,23,24];base,point=get('test',H);c=s['horizons'][str(H)];det=e['details'](d,orig,e['reference']('test',H),e['models'](True),H);cp=candidate(base,point,det,c['candidate'],H);p=(1-c['chosen_alpha'])*base+c['chosen_alpha']*cp;diag=(1-c['diagnostic_alpha'])*base+c['diagnostic_alpha']*cp;arrays.update({f'prior{H}':base,f'candidate{H}':cp,f'selected{H}':p,f'diagnostic{H}':diag});scores[str(H)]={k:score(d,orig,v,H) for k,v in dict(prior=base,candidate=diag,selected=p).items()}
  if H==7:
   rad=z('round23/models/interval7_radius.npz','radius')[:,None,None];lo=np.clip(p-rad,0,1);hi=np.clip(p+rad,0,1)
  else:lo=np.clip(z('round9/results/selected_intervals_test.npz','lower')+p-base,0,1);hi=np.clip(z('round9/results/selected_intervals_test.npz','upper')+p-base,0,1)
  arrays[f'lower{H}']=lo;arrays[f'upper{H}']=hi;y,m=common.truth(d,orig,H);scores[str(H)]['interval_selected']=interval_metrics(y,p,m,lo,hi)
 np.savez_compressed(Q/'test_predictions.npz',**arrays);save(Q/'summary.json',dict(selection=s,scores=scores,no_post_test_selection=True,independent_confirmation=False));print('ACTUAL JOINT TEST',json.dumps(scores),flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(__file__)==read(R/'docs/protocol.json')['script_sha256'];d=common.load();{'develop':develop,'test':test}[a.stage](d)

