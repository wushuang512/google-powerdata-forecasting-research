"""Independent R46 chronological, training, cached prediction and metric audit."""
from pathlib import Path
import argparse, datetime, hashlib, json, sys
sys.dont_write_bytecode = True
import numpy as np

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument('root')
args = ap.parse_args()
t = Path(args.root)
b, r, a = t/'experiment', t/'experiment/round46', t/'experiment/round44'
q = r/'results'
sys.path[:0] = [str(r/'code'), str(b/'code'), str(b/'round29/code'), str(b/'round28/code')]
import hdsf_foundation_fusion as fusion
fusion.initialize(t)
import common
from aligned_metrics import evaluate
from source_metrics import interval_metrics
p, s, report = read(r/'docs/protocol.json'), read(q/'selection.json'), read(q/'summary.json')
checks = []
def check(name, good):
    checks.append(dict(name=name, passed=bool(good)))
    assert good, name

for field, path in [('script_sha256',r/'code/hdsf_foundation_stage.py'),
                    ('adapter_sha256',r/'code/hdsf_foundation_adapters.py'),
                    ('fusion_script_sha256',r/'code/hdsf_foundation_fusion.py'),
                    ('source_data_sha256',a/'results/source_dataset.joblib'),
                    ('source_predictions_sha256',a/'results/predictions.npz'),
                    ('source_selection_sha256',a/'results/selection.json'),
                    ('canonical_evaluation_sha256',b/'round29/code/aligned_metrics.py')]:
    check('unchanged_' + field, p[field] == sha(path))
check('historical_evaluator_unchanged', sha(b/'code/common.py') == '6569f5e40ea884081fedf64b335dd7e3b790d19a0c4d4baa2b8285e5cb014a91')
check('protocol_amendment_archived', sha(r/'docs/protocol_registered_v1.json') == p['selection_amendment']['previous_protocol_sha256'])
check('amendment_before_selection', p['selection_amendment']['UTC'] < s['created_utc'])
check('selection_after_DEV_completion', all(read(q/f'{f}_dev_complete.json')['completed_utc'] < s['created_utc'] for f in p['families']))
check('selection_digest_frozen', sha(q/'selection.json') == (q/'selection.sha256').read_text().strip() == report['selection_sha256'])
check('selection_bound_protocol', s['protocol_sha256'] == sha(r/'docs/protocol.json') == report['protocol_sha256'])
check('TEST_not_used_for_selection', s['DEV_only'] and p['freeze_before_TEST'].startswith('Freezeall'))
check('data_scope_only_original_Google', report['data'] == 'OriginalGooglePowerData2019 only')
check('limits_disclosed', report['historical_test_informed_design'] and not report['independent_confirmation'])
d = common.load()
all_models = []
runtime_count = 0
for family, fs in s['families'].items():
    check(family+'_family_choice_dates', all(max(case['calendar_dates']) <= 18 for case in fs['early_candidates']+fs['early_alpha_candidates']))
    check(family+'_early_selected_digest', fs['checkpoint_sha256'] == sha(r/'models'/f'{family}_{fs["config"]}_cut14.pt'))
    for cut, cases in [(14,[f'{scope}_s{step}' for scope in ['head','tail'] for step in [40,80]]), (22,[fs['config']])]:
        phase = 'dev' if cut == 14 else 'test'
        for cfg in cases:
            info = read(r/'models'/f'{family}_{cfg}_cut{cut}.json')
            history = read(q/f'{family}_{cfg}_cut{cut}_training.json')
            steps = int(cfg.split('_s')[1])
            check(f'{family}_{cfg}_cut{cut}_actual_weight_change', info['initial_trainable_sha256'] != info['updated_trainable_sha256'])
            check(f'{family}_{cfg}_cut{cut}_updates', info['real_optimizer_updates'] == steps == len(history) and all(x['max_gradient'] > 0 for x in history))
            check(f'{family}_{cfg}_cut{cut}_label_boundary', info['training_last_label_max_tick'] < cut*96 == info['training_cutoff_tick'])
            check(f'{family}_{cfg}_cut{cut}_model_sha', sha(r/'models'/f'{family}_{cfg}_cut{cut}.pt') == info['model_file_sha256'])
            if cfg.startswith('tail'):
                check(f'{family}_{cfg}_cut{cut}_last_block_gradient', info['tail_gradient_observed'])
            check(f'{family}_{cfg}_cut{cut}_device', info['CUDA'] == (family != 'tirex2'))
            if cut == 22:
                check(family+'_freeze_before_final_fit', s['created_utc'] < history[0]['UTC'] < info['utc'])
            npz = q/f'{family}_{phase}_{cfg}.npz'
            meta = read(npz.with_suffix('.json'))
            check(f'{family}_{cfg}_cut{cut}_prediction_sha', meta['npz_sha256'] == sha(npz))
            check(f'{family}_{cfg}_cut{cut}_loaded_weights', meta['trainable_weights_sha256'] == info['updated_trainable_sha256'])
            all_models.append(dict(family=family, config=cfg, cutoff=cut, updates=steps))
    audit = read(q/f'{family}_runtime_audit.json')
    check(family+'_actual_future_poison_runtime_passed', audit['all_passed'] and audit['selection_sha256'] == sha(q/'selection.json'))
    runtime_count += audit['check_count']
    for phase in ['dev','test']:
        check(family+'_'+phase+'_complete', read(q/f'{family}_{phase}_complete.json')['all_passed'])
        controlpath = q/f'{family}_{phase}_original_control.npz'
        check(family+'_'+phase+'_original_control_sha', read(controlpath.with_suffix('.json'))['npz_sha256'] == sha(controlpath))
        # Zero adaptation must preserve every original CPU prediction exactly.
        for h in [1,7]:
            bank = fusion.family_bank(family, fs['config'], phase, h, 0.)
            original = fusion.members_original(phase,h)
            check(family+'_'+phase+f'_alpha0_exact_H{h}', all(np.array_equal(v, original[k]) for k,v in bank.items()))

for h, ss in s['fusion'].items():
    for row in ss['candidates']:
        for fold in row['folds']:
            check(f'chronological_fusion_H{h}_{row["geometry"]}_{row["alpha"]}_{fold["fold"]}',
                  fold['fit_latest_day'] < fold['validation_day'] and fold['validation_day'] > 18)
    for key in ['selected_fit','diagnostic_fit']:
        fit=ss[key]
        weights,basis=np.asarray(fit['weights']),np.asarray(fit['basis'])
        check(f'simplex_nonnegative_H{h}_{key}', (weights>=0).all())
        check(f'simplex_sum_one_H{h}_{key}', np.allclose(weights.sum(1),1))
        check(f'overlap_partition_unity_H{h}_{key}', (basis>=0).all() and np.allclose(basis.sum(0),1))

for cut in [14,22]:
    loc=read(q/f'local24_cut{cut}.json')
    check(f'local24_cut{cut}_D1_excluded', loc['D1_rows'] == 0)
    check(f'local24_cut{cut}_label_end', loc['label_max_day'] == cut and loc['rounds'] == 350 and len(loc['seeds']) == 4)
    smoother=read(q/f'local7_cut{cut}.json')
    check(f'local7_cut{cut}_training_labels', smoother['last_label_day'] <= cut)
    check(f'local7_cut{cut}_actual_parameter_search', len(smoother['candidates']) == 9)

bounds={}
z=np.load(q/'predictions.npz')
with np.load(a/'results/predictions.npz') as old:
    for h in [1,7]:
        oo=fusion.origins('test',h)
        check(f'original_HDSF_exact_H{h}', np.array_equal(z[f'original_HDSF_H{h}'],old[f'hdsf_H{h}']))
        bank=fusion.candidate_bank('test',h,s)
        tensor=fusion.geometry_tensor(bank,h)
        ss=s['fusion'][str(h)]
        geometry=fusion.core.apply_geometry(tensor,ss['selected_fit']).transpose(2,0,1)[fusion.canon]
        alpha=ss['selected']['alpha']
        selected=(1-alpha)*old[f'hdsf_H{h}']+alpha*geometry
        check(f'DEV_frozen_selected_replay_H{h}', np.array_equal(selected,z[f'DEV_selected_H{h}']))
        check(f'fixed_weight_replay_H{h}', np.array_equal(fusion.frozen_original_weight_prediction(bank,h),z[f'family_gated_original_weights_H{h}']))
        rawbank=fusion.candidate_bank('test',h,s,True)
        check(f'all_finetuned_replay_H{h}', np.array_equal(fusion.frozen_original_weight_prediction(rawbank,h),z[f'all_finetuned_original_weights_diagnostic_H{h}']))
        for role, metrics in report['scores'][str(h)]['scores'].items():
            pred=z[f'{role}_H{h}']
            check(f'shape_{role}_H{h}', pred.shape==(57,len(oo),96*h))
            check(f'finite_{role}_H{h}', np.isfinite(pred).all())
            check(f'all_unmodified_metrics_{role}_H{h}', evaluate(common,d,oo,pred,h) == metrics)
        y,m=common.truth(d,oo,h)
        check(f'valid_points_H{h}', int(m.sum()) == {1:49243,7:114902}[h])
        lo,hi=z[f'lower80_H{h}'],z[f'upper80_H{h}']
        check(f'interval_exact_H{h}', interval_metrics(y,selected,m,lo,hi)==report['scores'][str(h)]['interval'])
        check(f'selected_bound_H{h}', (selected>=0).all() and (selected<=1).all())
        bounds[str(h)]=dict(min=float(selected.min()),max=float(selected.max()),outside_0_1=int(((selected<0)|(selected>1)).sum()))
for row in report['native_rows']:
    key=f'{row["kind"]}_{row["model"]}_H{row["horizon_days"]}'
    ev=evaluate(common,d,fusion.origins('test',row['horizon_days']),z[key],row['horizon_days'])
    check('native_member_'+key, all(row[k]==v for k,v in ev.items()))
check('frozen_selection_still_same', sha(q/'selection.json')==report['selection_sha256'])
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_passed=True,
         checks=checks,check_count=len(checks),runtime_check_count=runtime_count,
         all_six_families_really_finetuned=True,neural_model_checkpoints=len(all_models),
         training_updates=sum(x['updates'] for x in all_models if x['cutoff']==22)+6*2*80,
         selected_bounds=bounds,selection_sha256=sha(q/'selection.json'),
         predictions_sha256=sha(q/'predictions.npz'),independent_confirmation=False,
         overfitting_absence_certified=False,new_word_created=False)
(q/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
