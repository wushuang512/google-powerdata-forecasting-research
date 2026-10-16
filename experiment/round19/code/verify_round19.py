from pathlib import Path
import sys,runpy,json,datetime
import numpy as np
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round19';Q=R/'results';p=R/'code/structured_projection_experiment.py'
sys.argv=[str(p),str(T),'--stage','prepare'];e=runpy.run_path(str(p));read=e['read'];sha=e['sha'];d=e['common'].load();s=read(Q/'selection.json');q=read(Q/'summary.json');dev=read(Q/'dev_summary.json');pr=read(R/'docs/protocol.json');v={}
v['pipeline_complete']=(Q/'pipeline_complete.json').exists() and read(Q/'resource_status.json')['stage']=='completed'
v['protocol_and_selection_hashes']=sha(R/'docs/protocol.json')==s['protocol_sha256']==(R/'docs/protocol.sha256').read_text() and sha(Q/'selection.json')==(Q/'selection.sha256').read_text() and sha(p)==pr['script_sha256']
v['all_bound_sources_unchanged']=all(sha(B/f)==h for f,h in pr['source_hashes'].items()) and read(B/'round18/results/independent_verification.json')['all_passed']
v['development_hashes']=sha(Q/'dev_summary.json')==s['dev_summary_sha256'] and sha(Q/'dev_predictions.npz')==s['dev_predictions_sha256']
pool=[x for x in dev['configs'] if x['config']['id']!='prior'];eligible=[x for x in pool if x['guard_passed']];best=min(eligible,key=lambda x:(-x['leading_count'],x['primary_loss'])) if eligible else min(pool,key=lambda x:x['primary_loss']);v['family_is_development_selected']=best['config']==s['candidate']
eligible=[x for x in dev['weights'] if x['eligible']]
if eligible:
 n=max(x['leading_count'] for x in eligible);pool=[x for x in eligible if x['leading_count']==n];win=min(pool,key=lambda x:x['primary_loss']);pick=min([x for x in pool if x['primary_loss']<=win['primary_loss']+win['paired_date_se']],key=lambda x:x['alpha'])
else:pick=dev['weights'][0]
v['strength_is_frozen_development_selected']=pick['alpha']==s['chosen_alpha'] and bool(eligible)==s['development_supported']
with np.load(Q/'test_predictions.npz') as a:ps={k:a[k].copy() for k in a.files}
v['actual_dimensions_and_bounds']=all(a.shape==(57,3,672) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for k,a in ps.items() if k!='selected24') and ps['selected24'].shape==(57,9,96)
v['unchanged24']=np.array_equal(ps['selected24'],e['z']('round8b/results/test_predictions.npz','selected'))
v['all_test_scores_recomputed']=all(abs(e['score'](d,[22,23,24],ps[k+'7'],7)[m]-q['scores7'][k][m])<1e-10 for k in ['prior','selected','diagnostic'] for m in e['METRICS'])
raw=e['z']('round18/results/test_predictions.npz','raw7');projected=e['project'](d,ps['prior7'],raw,s['candidate']);v['actual_projection_reproduced']=np.array_equal(projected,ps['projected7']);v['actual_frozen_strength']=np.array_equal(ps['selected7'],(1-s['chosen_alpha'])*ps['prior7']+s['chosen_alpha']*projected)
v['all_projection_variants_obey_declared_constraints']=True
for cfg in e['CONFIGS']:
 a=e['project'](d,ps['prior7'],raw,cfg)
 if cfg['neutrality']!='none' and cfg['id']!='prior':
  groups=[np.arange(57)] if cfg['neutrality']=='global' else [np.where(d['cells']==c)[0] for c in np.unique(d['cells'])] if cfg['neutrality']=='cell' else [np.array([i]) for i in range(57)]
  v['all_projection_variants_obey_declared_constraints'] &= all(np.max(abs(a[g].sum((0,2))-ps['prior7'][g].sum((0,2))))<1e-8 for g in groups)
 if cfg['peak']:v['all_projection_variants_obey_declared_constraints'] &= np.max(abs(a.reshape(57,3,7,96).max(-1)-ps['prior7'].reshape(57,3,7,96).max(-1)))<1e-12
alt={k:a.copy() for k,a in d.items()};alt['y'][:]=.999;alt['valid'][:]=False;alt['prod'][:]=.001;alt['pvalid'][:]=False
v['projection_has_no_truth_or_mask_dependency']=np.array_equal(e['project'](alt,ps['prior7'],raw,s['candidate']),projected)
v['forecast_total_not_physical_energy']=np.max(abs(ps['selected7'].sum((0,2))-ps['prior7'].sum((0,2))))<1e-8
old=read(B/'model_comparison_20261005/results/task_revision/summary.json');refs={a['model']:e['z'](a['source'],a['array']) for a in old['seven_day']}
for n in [15,16]:refs[f'round{n}']=e['z'](f'round{n}/results/test_predictions.npz','candidate7')
refs['round18']=e['z']('round18/results/test_predictions.npz','diagnostic7');ev=e['envelope'](d,[22,23,24],refs,7);wins=e['leads'](q['scores7']['selected'],ev);v['all21_comparators_and_seven_descriptive_bests']=len(refs)==21 and all(abs(ev[k]-q['control_best7'][k])<1e-10 for k in ev) and wins==q['leading_metrics7'] and len(wins)==7
with np.load(Q/'deployment7_origin31.npz') as a:dp={k:a[k].copy() for k in a.files}
v['actual_38304_deployment_and_same_parameters']=dp['selected'].shape==(57,1,672) and np.array_equal(dp['projected'],e['project'](d,dp['prior'],e['z']('round18/results/deployment7_origin31.npz','raw'),s['candidate'])) and np.array_equal(dp['selected'],(1-s['chosen_alpha'])*dp['prior']+s['chosen_alpha']*dp['projected'])
log=(Q/'pipeline.log').read_text(encoding='utf-8');start=log.split('STAGE test ')[1].splitlines()[0];v['selection_precedes_test_start']=s['frozen_utc']<start
v['no_false_independent_or_universal_best_claim']=not q['independent_confirmation'] and len(wins)<10
assert all(v.values()),v
e['save'](Q/'independent_verification.json',dict(all_passed=True,checks={k:bool(x) for k,x in v.items()},leading_metrics7=wins,chosen_alpha=s['chosen_alpha'],independent_confirmation=False,checked_utc=e['utc']()))
# Exploratory paired bootstrap by nine unique target dates, never independent overlapping windows.
y,m=e['common'].truth(d,[22,23,24],7);dates=np.array([[o+h for h in range(7)] for o in [22,23,24]]).reshape(-1);ya=y.reshape(57,21,96);ma=m.reshape(57,21,96);ud=np.unique(dates);stats={}
for kind in ['prior','selected']:
 pa=ps[kind+'7'].reshape(57,21,96);err=abs(pa-ya)*ma;cnt=ma.sum(-1);ay=(ya*ma).sum(-1)/cnt.clip(1);ap=(pa*ma).sum(-1)/cnt.clip(1);valid=cnt>0;pk=100*abs(np.where(ma,pa,-np.inf).max(-1)-np.where(ma,ya,-np.inf).max(-1))/np.maximum(np.where(ma,ya,-np.inf).max(-1),1e-8)
 vals=[]
 for day in ud:
  ix=dates==day;vals.append(dict(num=err[:,ix].sum((1,2)),den=(abs(ya[:,ix])*ma[:,ix]).sum((1,2)),dn=(abs(ap[:,ix]-ay[:,ix])*valid[:,ix]).sum(-1),dd=(abs(ay[:,ix])*valid[:,ix]).sum(-1),pn=np.where(valid[:,ix],pk[:,ix],0).sum(),pd=valid[:,ix].sum()))
 stats[kind]={k:np.stack([a[k] for a in vals],-1) for k in vals[0]}
rng=np.random.default_rng(20261005);counts=rng.multinomial(len(ud),np.full(len(ud),1/len(ud)),5000);bs={}
for kind,st in stats.items():bs[kind]=dict(wape=(100*(counts@st['num'].T)/(counts@st['den'].T)).mean(-1),daily_wape=(100*(counts@st['dn'].T)/(counts@st['dd'].T)).mean(-1),peak_mape=(counts@st['pn'])/(counts@st['pd']))
ci={k:dict(selected_minus_prior_pp=q['scores7']['selected'][k]-q['scores7']['prior'][k],exploratory_95_percentile_pp=np.quantile(bs['selected'][k]-bs['prior'][k],[.025,.975]).tolist()) for k in bs['prior']}
e['save'](Q/'exploratory_date_bootstrap.json',dict(target_dates=[int(x+1) for x in ud],samples=5000,seed=20261005,paired_unique_date_clusters=True,overlap_repetitions_preserved=True,metrics=ci,independent_confirmation=False,reason='Only9 unique dates, sequential dependence remains, May test repeatedly viewed. Not confirmatory significance.'))
print(json.dumps(dict(all_passed=True,checks=len(v),seven_best=wins,bootstrap=ci),ensure_ascii=False))
