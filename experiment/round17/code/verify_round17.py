from pathlib import Path
import sys,runpy,json,pickle,datetime
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round17';Q=R/'results';p=R/'code/multihorizon_residual_experiment.py'
assert (Q/'pipeline_complete.json').exists()
sys.argv=[str(p),str(T),'--stage','prepare'];e=runpy.run_path(str(p));sha=e['sha'];read=e['read'];d=e['common'].load();s=read(Q/'selection.json');q=read(Q/'summary.json');protocol=read(R/'docs/protocol.json');v={}
v['frozen_protocol']=sha(R/'docs/protocol.json')==s['protocol_sha256']==(R/'docs/protocol.sha256').read_text();v['frozen_selection']=sha(Q/'selection.json')==(Q/'selection.sha256').read_text();v['script_hash']=sha(p)==protocol['script_sha256'];v['sources_unchanged']=all(sha(B/f)==h for f,h in protocol['source_hashes'].items())
v['dev_predictions_unchanged']=sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256'];v['dev_summary_unchanged']=sha(Q/'dev_summary.json')==s['dev_summary_sha256'];v['dev_heads_unchanged']=sha(R/'models/dev_heads.pkl')==s['dev_heads_sha256']
cv=read(Q/'native_train7_verification.json');v['native_cache_consistent']=sha(Q/'native_train7.npz')==cv['aggregate_sha256']==s['train7_sha256'] and all(sha(R/f'cache/origin{o}.npz')==h for o,h in cv['source_hashes'].items());v['old_backbone_control_reproduced']=cv['control96_max_abs_difference']<1e-5
with np.load(Q/'test_predictions.npz') as z:arrays={k:z[k].copy() for k in z.files}
v['finite_bounded_arrays']=all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in arrays.values());v['actual_horizons']=all(arrays[k].shape==(57,9,96) for k in ['candidate24','selected24','prior24']) and all(arrays[k].shape==(57,3,672) for k in ['candidate7','selected7','prior7'])
v['all_scores_recomputed']=True
for H,orig,label in [(1,list(range(22,31)),24),(7,[22,23,24],7)]:
 for kind in ['prior','candidate','selected']:
  got=e['score'](d,orig,arrays[f'{kind}{label}'],H);v['all_scores_recomputed'] &= all(abs(got[k]-q[f'scores{label}'][kind][k])<1e-10 for k in got)
a=s['chosen_alpha'];v['shared_frozen_weight']=all(np.array_equal(arrays[f'selected{h}'],(1-a)*arrays[f'prior{h}']+a*arrays[f'candidate{h}']) for h in [24,7]);v['alpha0_preserves_both']=a!=0 or all(np.array_equal(arrays[f'selected{h}'],arrays[f'prior{h}']) for h in [24,7])
saved=pickle.load((R/'models/final_heads.pkl').open('rb'));v['final_training_ignores_future_labels']=True
alt={k:x.copy() for k,x in d.items()};alt['y'][:,22:]=.123;alt['valid'][:,22:]=False;alt['prod'][:,22:]=.987;alt['pvalid'][:,22:]=False
for H in [1,7]:
 for ridge in [100,1000]:
  fit=e['fit_heads'](alt,22,ridge,H);ref=saved[H][ridge];v['final_training_ignores_future_labels'] &= np.allclose(fit[-1].coef_,ref[-1].coef_,rtol=0,atol=1e-12) and np.allclose(fit[-1].intercept_,ref[-1].intercept_,rtol=0,atol=1e-12)
dev_saved=pickle.load((R/'models/dev_heads.pkl').open('rb'));v['dev_training_ignores_dev_and_test_labels']=True
alt14={k:x.copy() for k,x in d.items()};alt14['y'][:,14:]=.321;alt14['valid'][:,14:]=False;alt14['prod'][:,14:]=.876;alt14['pvalid'][:,14:]=False
for H in [1,7]:
 for ridge in [100,1000]:
  fit=e['fit_heads'](alt14,14,ridge,H);ref=dev_saved[H][ridge];v['dev_training_ignores_dev_and_test_labels'] &= np.allclose(fit[-1].coef_,ref[-1].coef_,rtol=0,atol=1e-12) and np.allclose(fit[-1].intercept_,ref[-1].intercept_,rtol=0,atol=1e-12)
v['causal_actual_candidate_all_test_origins']=True
for H,orig,label in [(1,list(range(22,31)),24),(7,[22,23,24],7)]:
 for j,o in enumerate(orig):
  aa={k:x.copy() for k,x in d.items()};aa['y'][:,o:]=.99;aa['valid'][:,o:]=False;aa['prod'][:,o:]=.99;aa['pvalid'][:,o:]=False
  p2=e['candidate'](aa,[o],arrays[f'prior{label}'][:,j:j+1],s['candidate'],saved[H],H);v['causal_actual_candidate_all_test_origins'] &= np.array_equal(p2,arrays[f'candidate{label}'][:,j:j+1])
v['minimax_chosen_on_early_dev']=s['candidate']['id']==min(read(Q/'dev_summary.json')['candidates'],key=lambda x:x['worst_metric_ratio'])['id'];v['no_false_all_metric_claim']=not q['all_metric_test_candidate_dominance'] or q['test_worst_ratio']<=1+1e-10
newcfg=next(c for c in e['CONFIGS'] if c['id']==min((r for r in read(Q/'dev_summary.json')['candidates'] if r['id']!='prior'),key=lambda r:r['worst_metric_ratio'])['id'])
v['nonprior_actual_candidate_causal']=True
for H,label in [(1,24),(7,7)]:
 aa={k:x.copy() for k,x in d.items()};aa['y'][:,22:]=.99;aa['valid'][:,22:]=False;aa['prod'][:,22:]=.99;aa['pvalid'][:,22:]=False;b=arrays[f'prior{label}'][:,:1]
 p1=e['candidate'](d,[22],b,newcfg,saved[H],H);p2=e['candidate'](aa,[22],b,newcfg,saved[H],H);v['nonprior_actual_candidate_causal'] &= np.array_equal(p1,p2)
assert all(v.values()),v
e['save'](Q/'independent_verification.json',dict(all_passed=True,checks={k:bool(x) for k,x in v.items()},checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),chosen_alpha=a,all_metric_test_candidate_dominance=q['all_metric_test_candidate_dominance'],independent_confirmation=False))
print(json.dumps(dict(round=17,all_checks_passed=True,count=len(v),alpha=a,all_metric_candidate_dominance=q['all_metric_test_candidate_dominance']),ensure_ascii=False))
