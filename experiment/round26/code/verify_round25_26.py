from pathlib import Path
import sys,runpy,json
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment';audit={}
for n,name in [(25,'level_shape_experiment.py'),(26,'quarter_joint_experiment.py')]:
 R=B/f'round{n}';Q=R/'results';p=R/'code'/name;sys.argv=[str(p),str(T),'--stage','prepare'];e=runpy.run_path(str(p));d=e['common'].load();s=e['read'](Q/'selection.json');pr=e['read'](R/'docs/protocol.json');dev=e['read'](Q/'dev_summary.json');summary=e['read'](Q/'summary.json');v={}
 v['bound_science_and_sources']=e['sha'](p)==pr['script_sha256'] and all(e['sha'](B/f)==h for f,h in pr['source_hashes'].items()) and e['sha'](R/'code/source_metrics.py')==pr['metrics_script_sha256']
 v['frozen_choice_and_development']=e['sha'](Q/'selection.json')==(Q/'selection.sha256').read_text() and e['sha'](Q/'dev_summary.json')==s['dev_summary_sha256'] and e['sha'](Q/'dev_predictions.npz')==s['dev_predictions_sha256'] and e['sha'](R/'docs/protocol.json')==s['protocol_sha256']
 with np.load(Q/'test_predictions.npz') as a:ps={k:a[k].copy() for k in a.files}
 v['finite_bound_shapes']=ps['selected7'].shape==(57,3,672) and ps['selected1'].shape==(57,9,96) and all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in ps.values())
 v['metrics_recomputed']=True
 for kind,key in [('prior','prior7'),('candidate','diagnostic7'),('selected','selected7')]:
  sc=e['score'](d,[22,23,24],ps[key],7);v['metrics_recomputed'] &= all(sc[k]==summary['scores']['7'][kind][k] if isinstance(sc[k],str) else abs(sc[k]-summary['scores']['7'][kind][k])<1e-10 for k in sc)
 c=s['horizons']['7'];alpha=c['chosen_alpha']
 if n==25:
  row=dev['horizons']['7'];pool=[r for r in row['configs'] if r['config']['id']!='prior'];el=[r for r in pool if r['guard_passed']];best=min(el or pool,key=lambda r:r['cost']);el=[w for w in row['weights'] if w['eligible']]
  if el:
   win=min(el,key=lambda w:w['cost']);pick=min([w for w in el if w['cost']<=win['cost']+win['paired_date_se']],key=lambda w:w['alpha'])
  else:pick=row['weights'][0]
  v['actual_dev_selection']=best['config']==c['candidate'] and pick['alpha']==alpha
  original=e['z']('round24/results/test_predictions.npz','prior7');got=e['candidate'](d,ps['prior7'],original,c['candidate']);v['candidate_reproduced']=np.array_equal(got,ps['candidate7']);alt={k:a.copy() for k,a in d.items()};alt['y'][:]=.123;alt['valid'][:]=False;v['truth_not_used']=np.array_equal(got,e['candidate'](alt,ps['prior7'],original,c['candidate']));v['frozen_strength']=np.array_equal(ps['selected7'],(1-alpha)*ps['prior7']+alpha*ps['candidate7'])
 else:
  supported=all(e['admission_guard'](r['candidate'],r['prior']) for r in dev['splits']) and len(e['improvements'](dev['splits'][-1]['candidate'],dev['splits'][-1]['prior']))>=2
  v['actual_binary_dev_admission']=c['development_supported']==supported and alpha==(.25 if supported else 0)
  v['frozen_quarter_no_test_sweep']=np.array_equal(ps['diagnostic7'],.75*ps['prior7']+.25*ps['proposal7']) and np.array_equal(ps['selected7'],(1-alpha)*ps['prior7']+alpha*ps['proposal7'])
  v['current_test_tradeoffs_within_registered_limits']=e['admission_guard'](summary['scores']['7']['selected'],summary['scores']['7']['prior'])
  fields=['wape','mae','rmse','daily_wape','peak_mape','p95_ape','peak_time_mae_minutes','absolute_bias_percent','cella5_wape','other52_wape','hit2_percent','hit5_percent','daily_mae','daily_rmse'];better=[];same=[];worse=[]
  for k in fields:
   a=summary['scores']['7']['selected'][k];b=summary['scores']['7']['prior'][k];delta=(b-a) if not k.startswith('hit') else(a-b)
   (same if abs(delta)<1e-12 else better if delta>0 else worse).append(k)
  v['eight_improved_one_equal_five_tiny_regressions']=len(better)==8 and len(same)==1 and len(worse)==5
  e['save'](Q/'recommendation_comparison.json',dict(fields=fields,improved=better,equal=same,minor_regressions=worse,reference='round19 previously recommended',tolerances=pr['tolerances'],user_accepts_minor_tradeoffs=True,independent_confirmation=False))
 v['a5_exactly_preserved']=np.array_equal(ps['selected7'][d['cells']=='a'],e['z']('round19/results/test_predictions.npz','selected7')[d['cells']=='a'])
 v['cell_normalized_totals_preserved']=all(np.max(abs(ps['selected7'][d['cells']==cell].sum((0,2))-ps['prior7'][d['cells']==cell].sum((0,2))))<1e-8 for cell in np.unique(d['cells']))
 v['selected24_unchanged']=np.array_equal(ps['selected1'],e['z']('round8b/results/test_predictions.npz','selected'))
 v['selection_precedes_test_start']=s['frozen_utc']<(Q/'pipeline.log').read_text(encoding='utf-8').split('STAGE test ')[1].split(' bounded')[0].splitlines()[0]
 v['bounded_runners_completed']=all(e['read'](Q/(st+'_complete.json'))['exit_code']==0 and e['read'](Q/(st+'_complete.json'))['peak_job_committed_bytes']<=512*2**20 for st in ['develop','test'])
 y,m=e['common'].truth(d,[22,23,24],7);im=e['interval_metrics'](y,ps['selected7'],m,ps['lower7'],ps['upper7']);v['interval_metrics_recomputed']=all(abs(im[k]-summary['scores']['7']['interval_selected'][k])<1e-10 for k in im)
 assert all(v.values()),(n,v);e['save'](Q/'independent_verification.json',dict(all_passed=True,checks={k:bool(x) for k,x in v.items()},checked_utc=e['utc'](),independent_confirmation=False));audit[str(n)]=len(v)
print(json.dumps(dict(all_passed=True,checks=audit)))

