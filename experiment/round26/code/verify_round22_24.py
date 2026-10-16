from pathlib import Path
import sys,json,runpy,pickle
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment';out={}
for n,filename in [(22,'point_residual_experiment.py'),(23,'joint_point_peak_experiment.py'),(24,'volatility_gate_experiment.py')]:
 R=B/f'round{n}';Q=R/'results';p=R/'code'/filename;sys.argv=[str(p),str(T),'--stage','prepare'];e=runpy.run_path(str(p));d=e['common'].load();s=e['read'](Q/'selection.json');pr=e['read'](R/'docs/protocol.json');dev=e['read'](Q/'dev_summary.json');summary=e['read'](Q/'summary.json');v={}
 v['finished_and_bound_sources']=(Q/'pipeline_complete.json').exists() and all(e['sha'](B/f)==h for f,h in pr['source_hashes'].items())
 v['science_and_selection_hashes']=e['sha'](p)==pr['script_sha256'] and e['sha'](R/'code/source_metrics.py')==pr['metrics_script_sha256'] and e['sha'](Q/'selection.json')==(Q/'selection.sha256').read_text() and e['sha'](R/'docs/protocol.json')==s['protocol_sha256']==(R/'docs/protocol.sha256').read_text()
 v['development_binding']=e['sha'](Q/'dev_summary.json')==s['dev_summary_sha256'] and e['sha'](Q/'dev_predictions.npz')==s['dev_predictions_sha256']
 with np.load(Q/'test_predictions.npz') as a:ps={k:a[k].copy() for k in a.files}
 v['all_arrays_finite_bounded']=all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in ps.values())
 v['frozen_dev_choice_reproduced']=True;v['frozen_strength_predictions']=True;v['all_metrics_and_intervals_recomputed']=True;v['actual_candidate_future_labels_ignored']=True;v['conditional_total_preserved']=True
 for H in [1,7]:
  cfg=s['horizons'][str(H)];row=dev['horizons'][str(H)];pool=[x for x in row['configs'] if x['config']['id']!='prior'];elig=[x for x in pool if x['guard_passed']];best=min(elig or pool,key=lambda x:x['cost']);elig=[x for x in row['weights'] if x['eligible']]
  if elig:
   win=min(elig,key=lambda x:x['cost']);pick=min([x for x in elig if x['cost']<=win['cost']+win['paired_date_se']],key=lambda x:x['alpha'])
  else:pick=row['weights'][0]
  v['frozen_dev_choice_reproduced'] &= best['config']==cfg['candidate'] and pick['alpha']==cfg['chosen_alpha'] and bool(elig)==cfg['development_supported']
  b=ps[f'prior{H}'];cp=ps[f'candidate{H}'];v['frozen_strength_predictions'] &= np.array_equal(ps[f'selected{H}'],(1-cfg['chosen_alpha'])*b+cfg['chosen_alpha']*cp) and np.array_equal(ps[f'diagnostic{H}'],(1-cfg['diagnostic_alpha'])*b+cfg['diagnostic_alpha']*cp)
  orig=list(range(22,31)) if H==1 else[22,23,24];y,m=e['common'].truth(d,orig,H)
  for kind,key in [('prior',f'prior{H}'),('candidate',f'diagnostic{H}'),('selected',f'selected{H}')]:
   got=e['score'](d,orig,ps[key],H);v['all_metrics_and_intervals_recomputed'] &= all(got[k]==summary['scores'][str(H)][kind][k] if isinstance(got[k],str) else abs(got[k]-summary['scores'][str(H)][kind][k])<1e-10 for k in got)
  im=e['interval_metrics'](y,ps[f'selected{H}'],m,ps[f'lower{H}'],ps[f'upper{H}']);v['all_metrics_and_intervals_recomputed'] &= all(abs(im[k]-summary['scores'][str(H)]['interval_selected'][k])<1e-10 for k in im)
  for j,o in enumerate(orig):
   alt={k:a.copy() for k,a in d.items()};alt['y'][:,o:]=.987;alt['valid'][:,o:]=False;alt['prod'][:,o:]=.234;alt['pvalid'][:,o:]=False
   if n==22:
    hs=pickle.load((R/'models/final_heads.pkl').open('rb'));ref=e['e']['reference']('test',H)[:,j:j+1];delta=e['residual'](alt,[o],ref,hs[cfg['candidate']['model']],H);got=e['candidate'](b[:,j:j+1],delta,cfg['candidate'],H)
   elif n==23:
    ref=e['e']['reference']('test',H)[:,j:j+1];det=e['e']['details'](alt,[o],ref,e['e']['models'](True),H);point=e['z']('round22/results/test_predictions.npz',f'candidate{H}')[:,j:j+1];got=e['candidate'](b[:,j:j+1],point,det,cfg['candidate'],H)
   else:
    proposal=e['z']('round23/results/test_predictions.npz',f'candidate{H}')[:,j:j+1];got=e['candidate'](alt,[o],b[:,j:j+1],proposal,cfg['candidate'])
   v['actual_candidate_future_labels_ignored'] &= np.array_equal(got,cp[:,j:j+1])
  v['conditional_total_preserved'] &= np.max(abs(cp.sum((0,2))-b.sum((0,2))))<1e-8
  if n==24 and H==7:
   ix=d['cells']=='a';v['actual_a5_identical']=np.array_equal(ps['selected7'][ix],ps['prior7'][ix]);v['per_cell_forecast_totals']=all(np.max(abs(cp[d['cells']==c].sum((0,2))-b[d['cells']==c].sum((0,2))))<1e-8 for c in np.unique(d['cells']))
 v['selection_precedes_test_start']=s['frozen_utc']<(Q/'pipeline.log').read_text(encoding='utf-8').split('STAGE test ')[1].split(' bounded')[0].splitlines()[0]
 v['hard_job_limits_actual']=all(e['read'](Q/(st+'_complete.json'))['peak_job_committed_bytes']<=512*2**20 and e['read'](Q/(st+'_complete.json'))['exit_code']==0 for st in ['develop','test'])
 if n==22:
  from sklearn import ensemble
  original=ensemble.HistGradientBoostingRegressor;capture={};realSave=e['fit'].__globals__['save']
  class OnlyInputs:
   def __init__(self,**kwargs):pass
   def fit(self,*args,**kwargs):return self
  try:
   ensemble.HistGradientBoostingRegressor=OnlyInputs;e['fit'].__globals__['save']=lambda p,x:capture.update(x)
   for cutoff in [14,22]:
    alt={k:a.copy() for k,a in d.items()};alt['y'][:,cutoff:]=.111;alt['valid'][:,cutoff:]=False;alt['prod'][:,cutoff:]=.999;e['fit'](alt,cutoff);expected=e['read'](R/f'models/train_inputs_cutoff{cutoff}.json');v[f'actual_training_inputs_exclude_future_{cutoff}']=capture==expected
  finally:ensemble.HistGradientBoostingRegressor=original;e['fit'].__globals__['save']=realSave
  hs=pickle.load((R/'models/final_heads.pkl').open('rb'));v['six_declared_heads_and_parameters']=len(hs)==6 and all(hs[c['id']].loss==c['loss'] and hs[c['id']].max_iter==c['iterations'] and hs[c['id']].max_leaf_nodes==c['leaves'] and hs[c['id']].early_stopping is False for c in pr['models'])
 assert all(v.values()),(n,v)
 e['save'](Q/'independent_verification.json',dict(all_passed=True,checks={k:bool(x) for k,x in v.items()},checked_utc=e['utc'](),independent_confirmation=False));out[str(n)]=len(v)
print(json.dumps(dict(all_passed=True,checks=out)))

