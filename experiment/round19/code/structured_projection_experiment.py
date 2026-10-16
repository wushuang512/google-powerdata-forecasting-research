"""Frozen DEV-only search over output consistency projections of round18 residuals."""
from pathlib import Path
import sys,json,hashlib,datetime,argparse,ast
import numpy as np
T=Path(sys.argv[1]).resolve(); B=T/'experiment'; R=B/'round19'; Q=R/'results'
src=B/'round18/code/horizon_routed_experiment.py'; e={'__file__':str(src),'__name__':'immutable_round18_utilities'}
tree=ast.parse(src.read_text(encoding='utf-8'));cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in n.targets))
exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(src),'exec'),e)
common=e['common'];score=e['score'];sha=e['sha'];read=e['read'];save=e['save'];utc=e['utc'];z=e['z'];METRICS=e['METRICS'];leads=e['leads'];primary=e['primary'];envelope=e['envelope'];paired_se=e['paired_se']
CONFIGS=[dict(id='prior',neutrality='none',peak=False)]+[dict(id=f'{n}_peak{int(p)}',neutrality=n,peak=p) for n in ['none','global','cell','pdu'] for p in [False,True]]
ALPHAS=[0.,.1,.25,.5,1.]
def project(d,base,raw,cfg):
 if cfg['id']=='prior':return base.copy()
 out=raw.copy();groups=[np.arange(57)] if cfg['neutrality']=='global' else [np.where(d['cells']==c)[0] for c in np.unique(d['cells'])] if cfg['neutrality']=='cell' else [np.array([i]) for i in range(57)]
 for j in range(base.shape[1]):
  b=base[:,j];q=raw[:,j];upper=np.ones_like(q);pin=np.zeros(q.shape,bool)
  if cfg['peak']:
   bd=b.reshape(57,7,96);mx=bd.max(-1);upper=np.repeat(mx,96,axis=-1)
   pi=pin.reshape(57,7,96);np.put_along_axis(pi,bd.argmax(-1)[...,None],True,axis=-1)
  a=np.clip(q,0,upper);a[pin]=b[pin]
  if cfg['neutrality']!='none':
   for g in groups:
    bg=b[g];qg=q[g];ug=upper[g];pg=pin[g];lo=-2.;hi=2.;target=bg.sum()
    for _ in range(55):
     mid=(lo+hi)/2;v=np.clip(qg+mid,0,ug);v[pg]=bg[pg]
     if v.sum()>target:hi=mid
     else:lo=mid
    v=np.clip(qg+(lo+hi)/2,0,ug);v[pg]=bg[pg];a[g]=v
    assert abs(v.sum()-target)<1e-8
  out[:,j]=a
 assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
 if cfg['peak']:assert np.max(abs(out.reshape(57,-1,7,96).max(-1)-base.reshape(57,-1,7,96).max(-1)))<1e-12
 return out
def guard(s,b):
 return all(s[k]<=b[k]+.001 for k in ['wape','daily_wape']) and all(s[k]<=b[k]*1.005 for k in ['rmse','p95_ape']) and s['peak_mape']<=b['peak_mape']*1.02 and s['absolute_bias_percent']<=b['absolute_bias_percent']+.01 and s['cella5_wape']<=b['cella5_wape']+.01
def prepare():
 for k in ['code','docs','models','results']:(R/k).mkdir(parents=True,exist_ok=True)
 if (R/'docs/protocol.json').exists():assert read(R/'docs/protocol.json')['script_sha256']==sha(__file__);return
 files=['results/data.npz','round18/code/horizon_routed_experiment.py','round18/docs/protocol.json','round18/results/selection.json','round18/results/dev_predictions.npz','round18/results/test_predictions.npz','round18/results/deployment7_origin31.npz','round18/results/independent_verification.json','round17/code/multihorizon_residual_experiment.py','round16/code/joint_local_experiment.py','model_comparison_20261005/results/task_revision/summary.json','round15/results/test_predictions.npz','round16/results/test_predictions.npz']
 # refs7 are audited immutable historical development forecasts.
 files+=['round15/results/tf3_dev7.npz','round15/results/dev_candidate7.npz','round16/results/dev_candidate7.npz'] if (B/'round15/results/dev_candidate7.npz').exists() else []
 save(R/'docs/protocol.json',dict(created_utc=utc(),script_sha256=sha(__file__),configs=CONFIGS,alphas=ALPHAS,base_family='R18 DEV-frozen zero-intercept Ridge alpha1000 mean plus first3 Fourier harmonics; no refit and no new model downloads.',motivation='R18 native7 residual improved curve/daily but had bias and peak tradeoffs. This new registered exploration tests sum-neutral forecasts and prior daily-max preservation. Same month has been repeatedly examined; no independent confirmation.',projection='Closest additive-shift box projection preserves prior sum of normalized predictions per global/cell/PDU group over each7day window; optional caps and pins preserve each unmasked prior daily max. These are forecast consistency constraints, not physical energy conservation. No future truth or future validity mask used.',family_selection='First DEV origin14 fixed.25. Prefer nonprior with primary WAPE/daily within+.001pp, RMSE/P95<=1.005prior, peak<=1.02prior, abs_bias<=prior+.01pp and a5<=prior+.01pp. Choose most metric-best-control leads then minimum equally normalized primary loss. If none use minimum-loss diagnostic.',weight_selection='Same single family on DEV origin15 and aggregate14/15, strengths0/.1/.25/.5/1. Apply all guards at both origins; aggregate at least3 descriptive bests and at least one primary improvement .001pp. Among eligible maximize leads and minimize primary; smaller weight within paired-date1SE among same-leading-count choices. Otherwise0. Overlapping DEV dates are stability evidence only.',criterion_change='Focus changed prospectively from R18 requirement to retain baseline lead count to the user objective multiple metric bests with explicit bias/a5 guards. R18 frozen selection remains unchanged.',diagnostic='One nonprior family and strength frozen before TEST; diagnostic strength is chosen positive strength else.25. No TEST rank reselection.',source_hashes={f:sha(B/f) for f in files},independent_confirmation=False,test_seen_before=True,new_downloads=False,resource_policy=dict(admission_physical_gib=3,admission_commit_gib=6,runtime_physical_gib=1.5,runtime_commit_gib=2)))
 (R/'docs/protocol.sha256').write_text(sha(R/'docs/protocol.json'),encoding='ascii');print('PROTOCOL FROZEN',flush=True)
def develop(d):
 assert not (Q/'selection.json').exists()
 base=z('round18/results/dev_predictions.npz','prior');raw=z('round18/results/dev_predictions.npz','zero_centered_a1000_shape1_agree0');refs=e['e']['refs7'](d);refs['round18_frozen_diagnostic']=.75*base+.25*raw
 ev0=envelope(d,[14],{k:v[:,:1] for k,v in refs.items()},7);eva=envelope(d,[14,15],refs,7);s0=score(d,[14],base[:,:1],7);s1=score(d,[15],base[:,1:],7);sa=score(d,[14,15],base,7);preds={};rows=[]
 for c in CONFIGS:
  p=project(d,base,raw,c);preds[c['id']]=p;sc=score(d,[14],.75*base[:,:1]+.25*p[:,:1],7);lead=leads(sc,ev0);rows.append(dict(config=c,scores=sc,primary_loss=primary(sc,s0),guard_passed=guard(sc,s0),leading_metrics=lead,leading_count=len(lead)));print('DEV FAMILY',c['id'],rows[-1]['primary_loss'],rows[-1]['guard_passed'],flush=True)
 pool=[r for r in rows if r['config']['id']!='prior'];el=[r for r in pool if r['guard_passed']];best=min(el,key=lambda r:(-r['leading_count'],r['primary_loss'])) if el else min(pool,key=lambda r:r['primary_loss']);cfg=best['config'];p=preds[cfg['id']];weights=[]
 for a in ALPHAS:
  cp=(1-a)*base+a*p;s14=score(d,[14],cp[:,:1],7);s15=score(d,[15],cp[:,1:],7);sc=score(d,[14,15],cp,7);lead=leads(sc,eva);ok=a>0 and guard(s14,s0) and guard(s15,s1) and len(lead)>=3 and any(sc[k]<=sa[k]-.001 for k in ['wape','daily_wape']);weights.append(dict(alpha=a,scores14=s14,scores15=s15,scores=sc,leading_metrics=lead,leading_count=len(lead),primary_loss=primary(sc,sa),paired_date_se=paired_se(d,cp,base),eligible=ok))
 el=[w for w in weights if w['eligible']]
 if el:
  n=max(w['leading_count'] for w in el);pool=[w for w in el if w['leading_count']==n];win=min(pool,key=lambda w:w['primary_loss']);pick=min([w for w in pool if w['primary_loss']<=win['primary_loss']+win['paired_date_se']],key=lambda w:w['alpha'])
 else:pick=weights[0]
 np.savez_compressed(Q/'dev_predictions.npz',**preds);save(Q/'dev_summary.json',dict(configs=rows,weights=weights,baseline_scores=sa,baseline_leading_metrics=leads(sa,eva),metric_best_controls=eva,selected_family_early=best,testing_not_loaded=True,overlapping_dev_windows=True));s=dict(frozen_utc=utc(),candidate=cfg,chosen_alpha=pick['alpha'],diagnostic_alpha=pick['alpha'] if pick['alpha']>0 else.25,development_supported=bool(el),protocol_sha256=sha(R/'docs/protocol.json'),dev_summary_sha256=sha(Q/'dev_summary.json'),dev_predictions_sha256=sha(Q/'dev_predictions.npz'),independent_confirmation=False);save(Q/'selection.json',s);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');print('SELECTION FROZEN',json.dumps(s),flush=True)
def test(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();assert sha(Q/'dev_summary.json')==s['dev_summary_sha256'] and sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256'];base=z('round18/results/test_predictions.npz','prior7');raw=z('round18/results/test_predictions.npz','raw7');p=project(d,base,raw,s['candidate']);diag=(1-s['diagnostic_alpha'])*base+s['diagnostic_alpha']*p;selected=(1-s['chosen_alpha'])*base+s['chosen_alpha']*p;p24=z('round18/results/test_predictions.npz','selected24');np.savez_compressed(Q/'test_predictions.npz',prior7=base,projected7=p,diagnostic7=diag,selected7=selected,selected24=p24)
 old=read(B/'model_comparison_20261005/results/task_revision/summary.json');refs24={v['model']:z(v['source'],v['array']) for v in old['models']};refs7={v['model']:z(v['source'],v['array']) for v in old['seven_day']}
 for n in [15,16]:refs24[f'round{n}_candidate']=z(f'round{n}/results/test_predictions.npz','candidate24');refs7[f'round{n}_candidate']=z(f'round{n}/results/test_predictions.npz','candidate7')
 refs7['round18_frozen_diagnostic']=z('round18/results/test_predictions.npz','diagnostic7');ev7=envelope(d,[22,23,24],refs7,7);ev24=envelope(d,list(range(22,31)),refs24,1);sc7={k:score(d,[22,23,24],v,7) for k,v in dict(prior=base,diagnostic=diag,selected=selected).items()};sc24=score(d,list(range(22,31)),p24,1)
 save(Q/'summary.json',dict(selection=s,scores7=sc7,scores24=sc24,control_best7=ev7,control_best24=ev24,control_count7=21,control_count24=20,leading_metrics7=leads(sc7['selected'],ev7),diagnostic_leading_metrics7=leads(sc7['diagnostic'],ev7),no_post_test_selection=True,independent_confirmation=False));print('ACTUAL TEST',json.dumps(sc7),flush=True)
def deploy7(d):
 s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text();base=z('round18/results/deployment7_origin31.npz','prior');raw=z('round18/results/deployment7_origin31.npz','raw');p=project(d,base,raw,s['candidate']);selected=(1-s['chosen_alpha'])*base+s['chosen_alpha']*p;diag=(1-s['diagnostic_alpha'])*base+s['diagnostic_alpha']*p;np.savez_compressed(Q/'deployment7_origin31.npz',prior=base,projected=p,selected=selected,diagnostic=diag);save(Q/'deployment_verification.json',dict(shape=list(selected.shape),finite_bounded=bool(np.isfinite(selected).all() and selected.min()>=0 and selected.max()<=1),point_count=int(selected.size),selection_sha256=sha(Q/'selection.json'),historical_replay=True,trace_timezone='America/Los_Angeles',no_june_truth=True));print('DEPLOYED CACHE',selected.size,flush=True)
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',default='prepare');a=ap.parse_args()
if a.stage=='prepare':prepare()
else:
 assert sha(R/'docs/protocol.json')==(R/'docs/protocol.sha256').read_text() and read(R/'docs/protocol.json')['script_sha256']==sha(__file__);d=common.load()
 {'develop':develop,'test':test,'deploy7':deploy7}[a.stage](d)

