from pathlib import Path
import sys,ast,json,hashlib,pickle
sys.dont_write_bytecode=True
import numpy as np
T=Path(sys.argv[1]).resolve();B=T/'experiment';totals={}
def load(n,file):
 p=B/f'round{n}/code'/file;env={'__file__':str(p),'__name__':'read_only_audit_utilities'};tree=ast.parse(p.read_text(encoding='utf-8'));cut=next(i for i,a in enumerate(tree.body) if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ap' for t in a.targets));exec(compile(ast.Module(body=tree.body[:cut],type_ignores=[]),str(p),'exec'),env);return env
for n,file in [(27,'robust_point_experiment.py'),(28,'bounded_level_experiment.py')]:
 e=load(n,file);R=B/f'round{n}';Q=R/'results';pr=e['read'](R/'docs/protocol.json');s=e['read'](Q/'selection.json');result=e['read'](Q/'summary.json');dev=e['read'](Q/'dev_summary.json');d=e['common'].load();checks={}
 checks['scientific_code_and_sources_bound']=e['sha'](R/'code'/file)==pr['script_sha256'] and e['sha'](R/'code/source_metrics.py')==pr['metrics_script_sha256'] and all(e['sha'](B/p)==h for p,h in pr['source_hashes'].items())
 checks['choice_frozen_before_test']=e['sha'](Q/'selection.json')==(Q/'selection.sha256').read_text().strip() and s['frozen_utc']<(Q/'pipeline.log').read_text(encoding='utf-8').split('STAGE test ')[1].split(' bounded')[0] and all(e['sha'](Q/p)==s[k] for p,k in [('dev_summary.json','dev_summary_sha256'),('dev_predictions.npz','dev_predictions_sha256')])
 checks['real_bounded_runs_all_completed']=all(e['read'](Q/(stage+'_complete.json'))['exit_code']==0 and not e['read'](Q/(stage+'_complete.json'))['resource_pressure_stop'] and e['read'](Q/(stage+'_complete.json'))['peak_job_committed_bytes']<=512*2**20 for stage in ['develop','test','deploy'])
 checks['correct_development_selection']=True;checks['test_metrics_recomputed']=True;checks['manual_macro_and_daily_wape']=True;checks['finite_bounds_a5_and_cell_totals']=True;checks['saved_prediction_reproduced']=True;checks['future_labels_not_used_for_features_or_correction']=True
 models=pickle.load((R/'models/final_heads.pkl').open('rb'))
 with np.load(Q/'test_predictions.npz') as a:ps={k:a[k].copy() for k in a.files}
 for H in [1,7]:
  c=s['horizons'][str(H)];dr=dev['horizons'][str(H)];eligible=[r for r in dr['configs'] if r['guard_passed']];best=min(eligible or dr['configs'],key=lambda v:v['cost']);eligible=[r for r in dr['weights'] if r['eligible']]
  if eligible:
   win=min(eligible,key=lambda v:v['cost']);pick=min([v for v in eligible if v['cost']<=win['cost']+win['paired_date_se']],key=lambda v:v['alpha'])
  else:pick=dr['weights'][0]
  checks['correct_development_selection'] &= best['config']==c['candidate'] and pick['alpha']==c['chosen_alpha']
  origins=list(range(22,31)) if H==1 else[22,23,24];base,ref=e['inputs']('test',H);delta=e['residual'](d,origins,ref,models[c['candidate']['model']],H);cp=e['candidate'](d,base,ref,delta,c['candidate'],origins) if n==28 else e['candidate'](d,base,ref,delta,c['candidate']);p=(1-c['chosen_alpha'])*base+c['chosen_alpha']*cp
  checks['saved_prediction_reproduced'] &= np.array_equal(p,ps[f'selected{H}']) and np.array_equal(cp,ps[f'raw_candidate{H}'])
  for kind,key in [('prior',f'prior{H}'),('candidate',f'diagnostic{H}'),('selected',f'selected{H}')]:
   score=e['score'](d,origins,ps[key],H);checks['test_metrics_recomputed'] &= all(score[k]==result['scores'][str(H)][kind][k] if isinstance(score[k],str) else abs(score[k]-result['scores'][str(H)][kind][k])<1e-10 for k in score)
  y,m=e['common'].truth(d,origins,H);wape=np.mean([100*np.abs(y[i][m[i]]-p[i][m[i]]).sum()/np.abs(y[i][m[i]]).sum() for i in range(57)]);yd=y.reshape(57,-1,96);pd=p.reshape(57,-1,96);md=m.reshape(57,-1,96);count=md.sum(-1);yy=(yd*md).sum(-1)/count.clip(1);pp=(pd*md).sum(-1)/count.clip(1);valid=count>0;dw=np.mean([100*np.abs(yy[i][valid[i]]-pp[i][valid[i]]).sum()/np.abs(yy[i][valid[i]]).sum() for i in range(57)])
  checks['manual_macro_and_daily_wape'] &= abs(wape-result['scores'][str(H)]['selected']['wape'])<1e-10 and abs(dw-result['scores'][str(H)]['selected']['daily_wape'])<1e-10
  checks['finite_bounds_a5_and_cell_totals'] &= p.shape==(57,len(origins),H*96) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1 and np.allclose(p[d['cells']=='a'],base[d['cells']=='a'],rtol=0,atol=1e-15) and all(np.max(abs(p[d['cells']==cell].sum((0,2))-base[d['cells']==cell].sum((0,2))))<1e-8 for cell in np.unique(d['cells']))
  alt={k:v.copy() for k,v in d.items()};o=22;alt['y'][:,o:]=.123;alt['valid'][:,o:]=False;x,scale=e['features'](d,o,0,ref[:,0,:96]);xx,ss=e['features'](alt,o,0,ref[:,0,:96]);a=e['residual'](d,[o],ref[:,:1],models[c['candidate']['model']],H);aa=e['residual'](alt,[o],ref[:,:1],models[c['candidate']['model']],H);checks['future_labels_not_used_for_features_or_correction'] &= np.array_equal(x,xx) and np.array_equal(scale,ss) and np.array_equal(a,aa)
  if n==28:checks['future_labels_not_used_for_features_or_correction'] &= np.array_equal(e['candidate'](d,base[:,:1],ref[:,:1],a,c['candidate'],[o]),e['candidate'](alt,base[:,:1],ref[:,:1],aa,c['candidate'],[o]))
  iv=e['interval_metrics'](y,p,m,ps[f'lower{H}'],ps[f'upper{H}']);checks['test_metrics_recomputed'] &= all(abs(iv[k]-result['scores'][str(H)]['interval_selected'][k])<1e-10 for k in iv)
 if n==27:
  checks['training_cutoffs_and_digests']=True
  for cutoff in [14,22]:
   x,y,w=e['training'](d,cutoff);rec=e['read'](R/f'models/train_inputs_cutoff{cutoff}.json');alt={k:v.copy() for k,v in d.items()};alt['y'][:,cutoff:]=.123;alt['valid'][:,cutoff:]=False;xx,yy,ww=e['training'](alt,cutoff)
   checks['training_cutoffs_and_digests'] &= rec['last_target_index']==cutoff-1 and all(hashlib.sha256(a.tobytes()).hexdigest()==rec[k] for a,k in [(x,'X_sha256'),(y,'y_sha256'),(w,'weight_sha256')]) and np.array_equal(x,xx) and np.array_equal(y,yy) and np.array_equal(w,ww)
  heads=pickle.load((R/'models/dev_heads.pkl').open('rb'));checks['four_development_models_real_and_regularized']=len(heads)==4 and all(m.get_params()['min_child_samples']==128 and m.get_params()['reg_lambda']==20 and m.get_params()['learning_rate']==.03 for m in heads.values())
 else:checks['head_reuse_no_fake_new_training']=e['sha'](R/'models/final_heads.pkl')==e['sha'](B/'round27/models/final_heads.pkl') and all(v['candidate']['model']=='lgb_regression_l1_leaf15' for v in s['horizons'].values())
 ex=e['read'](Q/'exports/verification.json');checks['actual_historical_deployment_bound']=ex['all_passed'] and ex['rows1']==5472 and ex['rows7']==38304 and all(e['sha'](T/p)==h for p,h in ex['source_hashes'].items()) and all(e['sha'](ex[f'file{H}'])==ex[f'sha{H}'] for H in [1,7])
 assert all(checks.values()),(n,checks)
 e['save'](Q/'independent_verification.json',dict(all_passed=True,checks={k:bool(v) for k,v in checks.items()},checked_utc=e['utc'](),independent_confirmation=False));totals[n]=len(checks)
print(json.dumps(dict(all_passed=True,checks=totals)))
