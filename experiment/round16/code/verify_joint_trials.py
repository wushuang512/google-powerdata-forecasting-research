from pathlib import Path
import sys,runpy,json,pickle,hashlib,datetime
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment'
for round_no,script in [(15,'joint_objective_experiment.py'),(16,'joint_local_experiment.py')]:
 R=B/f'round{round_no}';Q=R/'results';p=R/'code'/script
 assert (Q/'pipeline_complete.json').exists()
 sys.argv=[str(p),str(T),'--stage','prepare'];e=runpy.run_path(str(p));read=e['read'];sha=e['sha'];d=e['common'].load();s=read(Q/'selection.json');q=read(Q/'summary.json');v={}
 protocol=read(R/'docs/protocol.json');v['frozen_protocol']=sha(R/'docs/protocol.json')==s['protocol_sha256'];v['frozen_selection']=sha(Q/'selection.json')==(Q/'selection.sha256').read_text();v['script_hash']=sha(p)==protocol['script_sha256'];v['sources_unchanged']=all(sha(B/f)==h for f,h in protocol['source_hashes'].items())
 v['dev_predictions_unchanged']=sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256'];v['dev_summary_unchanged']=sha(Q/'dev_summary.json')==s['dev_summary_sha256']
 with np.load(Q/'test_predictions.npz') as z:
  arrays={k:z[k].copy() for k in z.files}
 v['finite_bounded_arrays']=all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in arrays.values());v['actual_horizons']=all(arrays[k].shape==(57,9,96) for k in ['candidate24','selected24','prior24']) and all(arrays[k].shape==(57,3,672) for k in ['candidate7','selected7','prior7'])
 v['all_scores_recomputed']=True
 for H,orig in [(1,list(range(22,31))),(7,[22,23,24])]:
  for kind in ['prior','candidate','selected']:
   got=e['score'](d,orig,arrays[kind+str(24 if H==1 else 7)],H);ref=q['scores24' if H==1 else 'scores7'][kind]
   v['all_scores_recomputed'] &= all(abs(got[k]-ref[k])<1e-10 for k in got)
 a=s['chosen_alpha'];v['shared_frozen_weight']=all(np.array_equal(arrays[f'selected{h}'],(1-a)*arrays[f'prior{h}']+a*arrays[f'candidate{h}']) for h in [24,7]);v['alpha0_preserves_both']=a!=0 or all(np.array_equal(arrays[f'selected{h}'],arrays[f'prior{h}']) for h in [24,7])
 # Retrain with all future labels and masks changed: exact same fitted coefficients.
 alt={k:x.copy() for k,x in d.items()};alt['y'][:,22:]=.123;alt['valid'][:,22:]=False
 fit=e['fit_heads'](alt,22,s['candidate']['alpha']);saved=pickle.load((R/'models/joint_heads_final.pkl').open('rb'))
 v['final_training_ignores_future_labels']=np.allclose(fit[-1].coef_,saved[-1].coef_,rtol=0,atol=1e-12) and np.allclose(fit[-1].intercept_,saved[-1].intercept_,rtol=0,atol=1e-12)
 v['causal_all_test_origins']=True
 for o in range(22,31):
  aa={k:x.copy() for k,x in d.items()};aa['y'][:,o:]=.99;aa['valid'][:,o:]=False
  for h in range(7):v['causal_all_test_origins'] &= all(np.array_equal(x,y) for x,y in zip(e['features'](d,o,h),e['features'](aa,o,h)))
 rng=np.random.default_rng(42);shape=rng.uniform(.3,.9,(30,96));shape[0]=.5;shape[1,:20]=.95;shape[2]=0;mean=rng.uniform(.2,.6,30);peak=np.maximum(mean,rng.uniform(.6,1,30));out=e['project'](shape,mean,peak,shape)
 v['projection_bounds_and_maximum']=np.isfinite(out).all() and out.min()>=0 and out.max()<=1 and np.max(abs(out.max(-1)-peak))<1e-9
 v['minimax_chosen_on_early_dev']=s['candidate']['id']==min(read(Q/'dev_summary.json')['candidates'],key=lambda x:x['worst_metric_ratio'])['id']
 v['no_false_all_metric_claim']=not q['all_metric_test_candidate_dominance'] or q['test_worst_ratio']<=1+1e-10
 assert all(v.values()),v
 e['save'](Q/'independent_verification.json',dict(all_passed=True,checks={k:bool(x) for k,x in v.items()},checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),chosen_alpha=a,all_metric_test_candidate_dominance=q['all_metric_test_candidate_dominance'],independent_confirmation=False))
 print(json.dumps(dict(round=round_no,all_checks_passed=True,count=len(v),alpha=a,all_metric_candidate_dominance=q['all_metric_test_candidate_dominance']),ensure_ascii=False))
