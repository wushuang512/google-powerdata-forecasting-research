from pathlib import Path
import sys,runpy,pickle,json,datetime
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round18';Q=R/'results';p=R/'code/horizon_routed_experiment.py';assert (Q/'pipeline_complete.json').exists()
sys.argv=[str(p),str(T),'--stage','prepare'];e=runpy.run_path(str(p));read=e['read'];sha=e['sha'];d=e['common'].load();s=read(Q/'selection.json');q=read(Q/'summary.json');protocol=read(R/'docs/protocol.json');v={}
v['protocol_frozen']=sha(R/'docs/protocol.json')==s['protocol_sha256']==(R/'docs/protocol.sha256').read_text();v['selection_frozen']=sha(Q/'selection.json')==(Q/'selection.sha256').read_text();v['implementation_frozen']=sha(p)==protocol['script_sha256'];v['source_hashes_unchanged']=all(sha(B/f)==h for f,h in protocol['source_hashes'].items());v['dev_sources_unchanged']=sha(Q/'dev_summary.json')==s['dev_summary_sha256'] and sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256'] and sha(R/'models/dev_heads.pkl')==s['dev_heads_sha256']
with np.load(Q/'test_predictions.npz') as a:ps={k:a[k].copy() for k in a.files}
v['prediction_dimensions']=all(ps[k].shape==(57,3,672) for k in ['prior7','raw7','diagnostic7','selected7']) and ps['selected24'].shape==(57,9,96);v['finite_bounded_predictions']=all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in ps.values());v['unchanged24_branch']=np.array_equal(ps['selected24'],e['z']('round8b/results/test_predictions.npz','selected'))
v['all_scores_recomputed']=all(abs(e['score'](d,list(range(22,31)),ps['selected24'],1)[k]-q['scores24'][k])<1e-10 for k in e['METRICS'])
for kind in ['prior','diagnostic','selected']:
 got=e['score'](d,[22,23,24],ps[kind+'7'],7);v['all_scores_recomputed'] &= all(abs(got[k]-q['scores7'][kind][k])<1e-10 for k in got)
v['frozen_strengths_actual']=np.array_equal(ps['selected7'],(1-s['chosen_alpha'])*ps['prior7']+s['chosen_alpha']*ps['raw7']) and np.array_equal(ps['diagnostic7'],(1-s['diagnostic_alpha'])*ps['prior7']+s['diagnostic_alpha']*ps['raw7']);v['zero_strength_keeps_prior']=s['chosen_alpha']!=0 or np.array_equal(ps['selected7'],ps['prior7'])
v['future_labels_ignored_in_training']=True
for cutoff,name in [(14,'dev'),(22,'final')]:
 saved=pickle.load((R/f'models/{name}_heads.pkl').open('rb'));alt={k:x.copy() for k,x in d.items()};alt['y'][:,cutoff:]=.123;alt['valid'][:,cutoff:]=False;alt['prod'][:,cutoff:]=.987;alt['pvalid'][:,cutoff:]=False
 for ridge in [1000,10000,100000]:
  fit=e['fit_zero'](alt,cutoff,ridge);ref=saved['zero'][ridge];v['future_labels_ignored_in_training'] &= np.allclose(fit[-1].coef_,ref[-1].coef_,rtol=0,atol=1e-12) and fit[-1].fit_intercept is False
saved=pickle.load((R/'models/final_heads.pkl').open('rb'));v['actual_nonprior_predictions_causal_all_origins']=True
for j,o in enumerate([22,23,24]):
 aa={k:x.copy() for k,x in d.items()};aa['y'][:,o:]=.99;aa['valid'][:,o:]=False;aa['prod'][:,o:]=.99;aa['pvalid'][:,o:]=False
 got=e['candidate'](aa,[o],ps['prior7'][:,j:j+1],s['candidate'],saved);v['actual_nonprior_predictions_causal_all_origins'] &= np.array_equal(got,ps['raw7'][:,j:j+1])
old=read(B/'model_comparison_20261005/results/task_revision/summary.json');refs={a['model']:e['z'](a['source'],a['array']) for a in old['seven_day']}
for n in [15,16]:refs[f'round{n}']=e['z'](f'round{n}/results/test_predictions.npz','candidate7')
ev=e['envelope'](d,[22,23,24],refs,7);v['all20_controls_recomputed']=all(abs(ev[k]-q['control_best7'][k])<1e-10 for k in ev);diagnostic_best=e['leads'](q['scores7']['diagnostic'],ev);v['diagnostic_six_descriptive_bests_verified']=len(diagnostic_best)==6
with np.load(Q/'deployment7_origin31.npz') as a:dp={k:a[k].copy() for k in a.files}
dv=read(Q/'deployment7_verification.json');v['actual_native_deployment38304']=dv['rows']==38304 and dp['selected'].shape==(57,1,672) and np.isfinite(dp['selected']).all() and dp['selected'].min()>=0 and dp['selected'].max()<=1;v['original_backbone_reproduced']=dv['control96_max_abs_difference']<1e-5;v['deployment_uses_same_frozen_weight']=np.array_equal(dp['selected'],(1-s['chosen_alpha'])*dp['prior']+s['chosen_alpha']*dp['raw']);v['selection_before_test']=s['frozen_utc']<read(Q/'test_complete.json')['ended_utc'];v['no_test_based_upgrade']=s['chosen_alpha']==0 and not s['development_supported']
assert all(v.values()),v
e['save'](Q/'independent_verification.json',dict(all_passed=True,checks={k:bool(x) for k,x in v.items()},diagnostic_leading_metrics7=diagnostic_best,diagnostic_leading_count7=len(diagnostic_best),chosen_alpha=s['chosen_alpha'],diagnostic_alpha=s['diagnostic_alpha'],independent_confirmation=False,deployment_time_label_requires_canonical_PT_export=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print(json.dumps(dict(all_passed=True,checks=len(v),diagnostic_leading_metrics7=diagnostic_best,chosen_alpha=s['chosen_alpha']),ensure_ascii=False))
