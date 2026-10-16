"""Independent read-only scientific checks; output written only after all checks."""
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, json, sys
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args()
T=Path(args.root);B=T/'experiment';R=B/'round45';Q=R/'results';A=B/'round44'
sys.dont_write_bytecode=True
sys.path[:0]=[str(B/'code'),str(B/'round29/code'),str(B/'round28/code'),str(A/'reference/PowerData_Prediction')]
import common
from aligned_metrics import evaluate
from source_metrics import interval_metrics
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
checks=[]
def check(name,good):
    checks.append({'name':name,'passed':bool(good)})
    if not good:raise AssertionError(name)
p=read(R/'docs/protocol.json');s=read(Q/'selection.json');report=read(Q/'summary.json');done=read(Q/'complete.json')
check('protocol_script_digest',p['identity']['script']==sha(R/'code/hdsf_next_experiment.py'))
check('canonical_evaluation_unchanged',p['identity']['metrics']==sha(B/'round29/code/aligned_metrics.py'))
check('original_selection_unchanged',p['identity']['original_selection']==sha(A/'results/selection.json'))
check('original_predictions_unchanged',p['identity']['members']==sha(A/'results/predictions.npz'))
check('selection_frozen_digest',sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()==report['selection_sha256'])
check('freeze_before_final_models',s['created_utc']<read(Q/'complete.json')['created_utc'])
check('no_new_dataset_used',report['data']=='GooglePowerData2019 only' and p['foundation_members_frozen'])
check('no_TEST_hyperparameter_selection',report['test_used_for_selection'] is False and s['test_not_scored_in_this_phase'])
check('independence_limitation_disclosed',report['historical_test_informed_design'] and report['independent_confirmation'] is False)
for h,sel in s['selected'].items():
    fit=sel['config']['fit'];weights=np.asarray(fit['weights']);basis=np.asarray(fit['basis'])
    check('simplex_nonnegative_H'+h,(weights>=0).all());check('simplex_sum_one_H'+h,np.allclose(weights.sum(1),1))
    check('overlap_partition_unity_H'+h,np.allclose(basis.sum(0),1) and (basis>=0).all())
    for fold in sel['candidate']['folds']:
        check('chronological_target_purge_H'+h+'_'+fold['fold'],fold['fit_latest_day']<fold['validation_day'])
for info in report['training']:
    check(info['kind']+'_D1_excluded',info['D1_rows']==0)
    check(info['kind']+'_max_label22',info['label_max_day']==22)
    check(info['kind']+'_actual_fitting_rounds',info['rounds'] in [250,350] and len(info['training_residual_mae_per_seed'])==len(info['seeds']))

spec=importlib.util.spec_from_file_location('r45',R/'code/hdsf_next_experiment.py');x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);x.initialize(T)
arr=np.load(Q/'predictions.npz');d=common.load();verified={};bounds={}
for H,oo in [(1,p['test24']),(7,p['test7'])]:
    chosen=s['selected'][str(H)];name=chosen['config']['member'];old,methods=x.members(H,'test')
    if name=='original':P=old
    else:
        import joblib
        model=joblib.load(R/'models'/f'{name}_cut22.joblib')
        local=np.stack([x.local_predict(model,x.data,o,H) for o in oo],axis=1)
        P,mm=x.members(H,'test',name,local)
        before=x.signature(model)
        poisoned=x.poison_future(x.data,oo[0]);unseen=x.local_predict(model,poisoned,oo[0],H)
        check('selected_actual_local_future_poison_H'+str(H),np.array_equal(unseen,local[:,0]))
        check('selected_actual_local_parameter_frozen_H'+str(H),before==x.signature(model))
    restored=x.apply_geometry(P,chosen['config']['fit']).transpose(2,0,1)[x.canonical_index]
    candidate=arr[f'candidate_H{H}'];check('selected_prediction_replay_H'+str(H),np.array_equal(restored,candidate))
    check('complete_shape_H'+str(H),candidate.shape==(57,len(oo),H*96))
    check('all_finite_H'+str(H),np.isfinite(candidate).all())
    bounds[str(H)]={'min':float(candidate.min()),'max':float(candidate.max()),'outside_0_1':int(((candidate<0)|(candidate>1)).sum())}
    for role,key in [('candidate',f'candidate_H{H}'),('original',f'original_H{H}'),('adopted',f'adopted_H{H}')]:
        score=evaluate(common,d,oo,arr[key],H);expected=report['scores'][str(H)][role+'_score']
        check(role+'_all_canonical_metrics_H'+str(H),score==expected)
    y,m=common.truth(d,oo,H);iv=interval_metrics(y,candidate,m,arr[f'lower80_candidate_H{H}'],arr[f'upper80_candidate_H{H}'])
    check('candidate_interval_metrics_H'+str(H),iv==report['scores'][str(H)]['interval'])
    check('valid_interval_order_H'+str(H),(arr[f'lower80_candidate_H{H}']<=arr[f'upper80_candidate_H{H}']).all())
    verified[str(H)]=report['scores'][str(H)]['candidate_score']

# Characterize the same TRAIN prefix; no TEST information informs model structure here.
y=d['y'][:,:14];m=d['valid'][:,:14]
daily=y.mean(-1);within=y.std(-1);flat=y.reshape(57,-1);fm=m.reshape(57,-1);correlations={}
for lag in [96,7*96]:
    cc=[]
    for i in range(57):
        good=fm[i,lag:]&fm[i,:-lag]
        a=flat[i,lag:][good];b=flat[i,:-lag][good]
        if np.std(a)>0 and np.std(b)>0:cc.append(float(np.corrcoef(a,b)[0,1]))
    correlations[str(lag//96)+'day']={'median':float(np.median(cc)),'min':float(min(cc)),'max':float(max(cc)),'domains':len(cc)}
profile={'training_days':'D1—D14 only','target_unit':'normalized power utilization, notkW',
         'lag_correlations':correlations,'median_within_day_coefficient_of_variation_percent':float(np.median(within/y.mean(-1))*100),
         'median_daily_mean_coefficient_of_variation_percent':float(np.median(daily.std(1)/daily.mean(1))*100),
         'interpretation_limit':'Variation is heterogeneous acrossPDU andcalendar dates; aggregation doesnot establish independent sample count.'}
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_passed':all(c['passed'] for c in checks),
     'checks':checks,'check_count':len(checks),'candidate_bounds':bounds,'verified_scores':verified,
     'load_characteristics_train_only':profile,'selection_sha256':sha(Q/'selection.json'),
     'predictions_sha256':sha(Q/'predictions.npz'),'overfitting_absence_certified':False,'independent_confirmation':False}
(Q/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'all_passed':out['all_passed'],'check_count':len(checks),'bounds':bounds,'training_load_characteristics':profile},ensure_ascii=True))
