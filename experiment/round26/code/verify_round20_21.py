"""Independent audit of new rounds and source-threshold definitions; no parameter changes."""
from pathlib import Path
import sys,runpy,json,datetime,hashlib
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment';allchecks={};snapshots={}
for n in [20,21]:
 R=B/f'round{n}';Q=R/'results';p=R/'code'/('learned_peak_experiment.py' if n==20 else 'localized_peak_experiment.py');sys.argv=[str(p),str(T),'--stage','prepare'];e=runpy.run_path(str(p));d=e['common'].load();s=e['read'](Q/'selection.json');pr=e['read'](R/'docs/protocol.json');summary=e['read'](Q/'summary.json');dev=e['read'](Q/'dev_summary.json');v={}
 v['finished_and_sources_bound']=(Q/'pipeline_complete.json').exists() and all(e['sha'](B/f)==h for f,h in pr['source_hashes'].items())
 v['frozen_science']=e['sha'](p)==pr['script_sha256'] and e['sha'](R/'code/source_metrics.py')==pr['metrics_script_sha256'] and e['sha'](R/'docs/protocol.json')==s['protocol_sha256']==(R/'docs/protocol.sha256').read_text() and e['sha'](Q/'selection.json')==(Q/'selection.sha256').read_text()
 v['dev_and_interval_binding']=e['sha'](Q/'dev_summary.json')==s['dev_summary_sha256'] and e['sha'](Q/'dev_predictions.npz')==s['dev_predictions_sha256'] and e['sha'](R/'models/interval7_radius.npz')==s['interval_radius_sha256']
 with np.load(Q/'test_predictions.npz') as a:ps={k:a[k].copy() for k in a.files}
 with np.load(Q/'deployment_origin31.npz') as a:dp={k:a[k].copy() for k in a.files}
 v['all_prediction_dimensions_bounds']=all(a.shape==(57,9,96) if k.endswith('1') else a.shape==(57,3,672) for k,a in ps.items()) and all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in ps.values())
 v['dev_selected_exactly_reproduced']=True;v['test_metrics_and_threshold_counts_recomputed']=True;v['causal_candidate_all_origins']=True;v['daily_mean_constraint']=True;v['deployment_same_frozen_configuration']=True
 for H in [1,7]:
  rows=dev['horizons'][str(H)]['configs'];pool=[r for r in rows if r['config']['id']!='prior'];elig=[r for r in pool if r['guard_passed']];best=min(elig or pool,key=lambda x:x['cost']);fs=s['horizons'][str(H)];weights=dev['horizons'][str(H)]['weights'];elig=[w for w in weights if w['eligible']]
  if elig:
   win=min(elig,key=lambda x:x['cost']);pick=min([w for w in elig if w['cost']<=win['cost']+win['paired_date_se']],key=lambda x:x['alpha'])
  else:pick=weights[0]
  v['dev_selected_exactly_reproduced'] &= best['config']==fs['candidate'] and pick['alpha']==fs['chosen_alpha'] and bool(elig)==fs['development_supported']
  origins=list(range(22,31)) if H==1 else [22,23,24];y,m=e['common'].truth(d,origins,H);base=ps[f'prior{H}'];ref=e['reference']('test',H);model=e['models'](True);cfg=fs['candidate']
  for kind,key in [('prior',f'prior{H}'),('candidate',f'diagnostic{H}'),('selected',f'selected{H}')]:
   sc=e['score'](d,origins,ps[key],H);v['test_metrics_and_threshold_counts_recomputed'] &= all(sc[k]==summary['scores'][str(H)][kind][k] if isinstance(sc[k],str) else abs(sc[k]-summary['scores'][str(H)][kind][k])<1e-10 for k in sc)
   ape=100*abs(ps[key]-y)[m]/np.maximum(abs(y[m]),1e-8)
   v['test_metrics_and_threshold_counts_recomputed'] &= abs(100*np.count_nonzero(ape<=2)/m.sum()-sc['hit2_percent'])<1e-12 and abs(100*np.count_nonzero(ape<=5)/m.sum()-sc['hit5_percent'])<1e-12
  for j,o in enumerate(origins):
   alt={k:a.copy() for k,a in d.items()};alt['y'][:,o:]=.999;alt['valid'][:,o:]=False;alt['prod'][:,o:]=.999;alt['pvalid'][:,o:]=False;cp=e['candidate'](base[:,j:j+1],e['details'](alt,[o],ref[:,j:j+1],model,H),cfg,H);v['causal_candidate_all_origins'] &= np.array_equal(cp,ps[f'candidate{H}'][:,j:j+1])
  v['daily_mean_constraint'] &= np.max(abs(ps[f'candidate{H}'].reshape(57,len(origins),H,96).sum(-1)-base.reshape(57,len(origins),H,96).sum(-1)))<1e-8
  v['deployment_same_frozen_configuration'] &= dp[f'selected{H}'].shape==(57,1,96*H) and np.isfinite(dp[f'selected{H}']).all() and np.array_equal(ps[f'selected{H}'],(1-fs['chosen_alpha'])*base+fs['chosen_alpha']*ps[f'candidate{H}'])
  v[f'selected{H}_matches_prior_when_zero']=fs['chosen_alpha']!=0 or np.array_equal(ps[f'selected{H}'],ps[f'prior{H}'])
  im=e['interval_metrics'](y,ps[f'selected{H}'],m,ps[f'lower{H}'],ps[f'upper{H}']);v[f'interval{H}_recomputed']=all(abs(im[k]-summary['scores'][str(H)]['interval_selected'][k])<1e-10 for k in im)
  if H==7:
   with np.load(Q/'dev_predictions.npz') as a:de=a['selected7'].copy()
   yd,md=e['common'].truth(d,[14,15],7);rad=np.array([np.quantile(abs(de[i]-yd[i])[md[i]],.8) for i in range(57)]);v['interval_radius_uses_only_dev']=np.array_equal(rad,e['z'](f'round{n}/models/interval7_radius.npz','radius'))
 v['freeze_precedes_test_start']=s['frozen_utc']<(Q/'pipeline.log').read_text(encoding='utf-8').split('STAGE test ')[1].split(' bounded')[0].splitlines()[0]
 if n==21:
  amend=e['read'](Q/'resource_policy_amendment.json');v['resource_amendment_before_development']=amend['registered_before_development'] and amend['registered_utc']<s['frozen_utc'];v['all_stages_within_hard_job_cap']=all(e['read'](Q/(stage+'_complete.json'))['peak_job_committed_bytes']<=512*2**20 and e['read'](Q/(stage+'_complete.json'))['exit_code']==0 for stage in ['develop','test','deploy']);v['outside_top_support_unchanged']=True
  origins=list(range(14,22));base=e['load_base']('dev',1);details=e['details'](d,origins,e['reference']('dev',1),e['models'](),1)
  for cfg in e['CONFIGS'][1:]:
   cp=e['candidate'](base,details,cfg,1);changed=abs(cp-base)>1e-14;v['outside_top_support_unchanged'] &= changed.sum(-1).max()<=cfg['support']+1
 assert all(v.values()),(n,v)
 e['save'](Q/'independent_verification.json',dict(all_passed=True,checks={k:bool(x) for k,x in v.items()},checked_utc=e['utc'](),independent_confirmation=False));allchecks[str(n)]=v
# Independently reproduce macro WAPE and point-threshold figures on the current retained outputs.
a=json.loads((B/'round20/results/original_metrics_complete.json').read_text(encoding='utf-8'));v={}
for H in [1,7]:
 row=next(r for r in a['rows'] if r['model']=='current_selected' and r['horizon_days']==H)
 with np.load(B/'round21/results/test_predictions.npz') as z:p=z[f'selected{H}']
 origins=list(range(22,31)) if H==1 else [22,23,24];y,m=e['common'].truth(d,origins,H);macro=[];ape=[]
 for i in range(57):
  yy=y[i][m[i]];pp=p[i][m[i]];macro.append(100*sum(abs(pp-yy))/sum(abs(yy)));ape.extend((100*abs(pp-yy)/abs(yy)).tolist())
 v[f'macro_wape{H}']=abs(np.mean(macro)-row['wape'])<1e-10
 v[f'thresholds{H}']=abs(100*sum(x<=2 for x in ape)/len(ape)-row['hit2_percent'])<1e-10 and abs(100*sum(x<=5 for x in ape)/len(ape)-row['hit5_percent'])<1e-10
 base=e['common'].baseline(d,origins,H,kind='last' if H==1 else 'mean3');bs=e['common'].metrics(y,base,m)['wape'];v[f'correct_source_baseline{H}']=abs(bs-row['baseline_wape'])<1e-12 and abs(100*(1-row['wape']/bs)-row['relative_curve_error_reduction_percent'])<1e-12
v['all_original_categories']=a['all_original_metric_categories_covered'] and len(a['reference_inventory']['metrics'])==11 and a['reference_inventory']['reference_sha256']==e['sha'](T/'参考材料/9.28算力中心需求预测实验.docx')
v['fifty_actual_named_rows']=len(a['rows'])==50 and all(r['model_name'] for r in a['rows'])
v['raw_tirex_not_fabricated']=any(r['kind'].startswith('历史引用') and r['mean_width_normalized'] is None for r in a['intervals'])
assert all(v.values()),v
e['save'](B/'round20/results/original_metrics_verification.json',dict(all_passed=True,checks={k:bool(x) for k,x in v.items()},round_checks={k:len(x) for k,x in allchecks.items()},independent_confirmation=False,checked_utc=e['utc']()))
print(json.dumps(dict(all_passed=True,round_checks={k:len(x) for k,x in allchecks.items()},source_checks=len(v)),ensure_ascii=False))

