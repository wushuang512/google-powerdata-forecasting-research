"""Independent real weight/input replay and canonical metric verification, R47."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--phase',choices=['runtime','verify'],required=True)
ap.add_argument('--family');ap.add_argument('--recipe');ap.add_argument('--split',choices=['dev','test']);args=ap.parse_args()
t=Path(args.root);r=t/'experiment/round47';q=r/'results';b=t/'experiment'
sys.path.insert(0,str(r/'code'))
checks=[]
def check(name,value,**details):
    checks.append(dict(name=name,passed=bool(value),**details));assert value,name

if args.phase=='runtime':
    import torch
    import hdsf_shape_training as train
    train.initialize(t);family,recipe,split=args.family,args.recipe,args.split
    cut=14 if split=='dev' else 22
    torch.manual_seed(20261012);adapter=train.Adapter(t,family,'cuda');stage=train.stage
    info=read(r/'models'/f'{family}_{recipe}_cut{cut}.json')
    adapter.set_scope('tail');check('original_weight_hash',adapter.hash_trainable()==info['initial_trainable_sha256'])
    with np.load(q/f'{family}_{split}_original_control.npz') as old:
        for member,h in train.tasks(family):
            original=stage.predict_member(adapter,member,cut,h)[stage.canon]
            check(f'original_native_control_H{h}',np.allclose(original,old[f'{member}_H{h}'][:,0],atol=2e-6,rtol=1e-6))
    state=torch.load(r/'models'/f'{family}_{recipe}_cut{cut}.pt',map_location='cpu',weights_only=True)
    adapter.load_trainable_state(state);del state
    check('updated_weight_hash',adapter.hash_trainable()==info['updated_trainable_sha256'])
    original=stage.data.target.copy();valid=stage.data.target_valid.copy()
    seq=train.batches(cut,family);poisoned=original.copy();poisoned[:,cut:]=.987654
    flat,pf=original.reshape(57,-1),poisoned.reshape(57,-1)
    for batch in seq:
        o,ctx,h,ix=batch['origin_tick'],batch['context_ticks'],batch['horizon_ticks'],batch['series_ids']
        assert o+h<=cut*96 and np.array_equal(flat[ix,o-ctx:o+h],pf[ix,o-ctx:o+h])
    check('all160_training_batches_future_poison',True)
    stage.data.target[:,cut:]=.987654;stage.data.target_valid[:,cut:]=False
    with np.load(q/f'{family}_{split}_{recipe}.npz') as cached:
        for member,h in train.tasks(family):
            point=stage.predict_member(adapter,member,cut,h)[stage.canon];old=cached[f'{member}_H{h}'][:,0]
            check(f'actual_updated_future_poison_H{h}',np.allclose(point,old,atol=2e-6,rtol=1e-6),max_difference=float(abs(point-old).max()))
    stage.data.target[:]=original;stage.data.target_valid[:]=valid
    check('inference_weights_fixed',adapter.hash_trainable()==info['updated_trainable_sha256'])
    out=dict(created_utc=utc(),all_passed=True,checks=checks,check_count=len(checks),family=family,recipe=recipe,split=split,
             model_sha256=sha(r/'models'/f'{family}_{recipe}_cut{cut}.pt'),protocol_sha256=sha(r/'docs/protocol.json'))
    (q/f'{family}_{recipe}_{split}_runtime_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(all_passed=True,family=family,recipe=recipe,split=split,checks=len(checks))))
else:
    import hdsf_shape_experiment as exp
    exp.initialize(t);p,s,report=exp.P,read(q/'selection.json'),read(q/'summary.json')
    check('registered_before_DEV',p['created_utc']<min(read(q/f'{family}_{recipe}_dev_training.json')[0]['UTC'] for family in p['new_families'] for recipe in p['new_recipes']))
    check('frozen_selection_digest',sha(q/'selection.json')==(q/'selection.sha256').read_text().strip()==report['selection_sha256'])
    check('protocol_bound',sha(r/'docs/protocol.json')==report['protocol_sha256']==s['protocol_sha256'])
    check('data_only_Google',report['dataset']=='OriginalGooglePowerData2019 only')
    check('limitations_disclosed',report['historical_test_informed_design'] and not report['independent_confirmation'])
    check('registered_metrics_unchanged',sha(b/'round29/code/aligned_metrics.py')=='5b34ab5c5c3e561d1626bb0e45a2f7fac9733661f28ce656c7e5c2ad2e2179da')
    for name,digest in p['bindings'].items():check('source_unchanged_'+name,sha(t/name)==digest)
    trained=[(f,recipe,'dev') for f in p['new_families'] for recipe in p['new_recipes']]+[(f,recipe,'test') for f,recipe in s['required_final_training']]
    runtimechecks=0
    for family,recipe,split in trained:
        cut=14 if split=='dev' else 22;tag=f'{family}_{recipe}_{split}'
        info=read(r/'models'/f'{family}_{recipe}_cut{cut}.json');history=read(q/f'{tag}_training.json')
        check(tag+'_real160updates',len(history)==info['steps']==160 and all(x['max_gradient']>0 for x in history))
        check(tag+'_encoder_and_head_updated',info['changed_head_tensors'] and info['changed_encoder_tensors'] and info['encoder_gradient_observed'])
        check(tag+'_different_weight_hash',info['initial_trainable_sha256']!=info['updated_trainable_sha256'])
        check(tag+'_weight_file_digest',info['model_sha256']==sha(r/'models'/f'{family}_{recipe}_cut{cut}.pt'))
        check(tag+'_training_label_boundary',info['last_training_label_tick']<cut*96)
        if split=='test':check(tag+'_freeze_before_training',s['created_utc']<history[0]['UTC'])
        else:check(tag+'_DEV_before_freeze',read(q/f'{tag}_complete.json')['completed_utc']<s['created_utc'])
        meta=read(q/f'{family}_{split}_{recipe}.json')
        check(tag+'_prediction_digest',meta['npz_sha256']==sha(q/f'{family}_{split}_{recipe}.npz'))
        check(tag+'_correct_weight_used',meta['trainable_sha256']==info['updated_trainable_sha256'])
        audit=read(q/f'{tag}_runtime_audit.json');check(tag+'_actual_runtime_passed',audit['all_passed'] and audit['model_sha256']==info['model_sha256'])
        runtimechecks+=audit['check_count']
    with np.load(q/'predictions.npz') as arrays:
        for h in [1,7]:
            cfg=s['tasks'][str(h)]['candidate'];prior=exp.base('test',h);delta=exp.route_delta(cfg['route'],'test',h)
            point=exp.adapt(prior,delta,cfg['level_alpha'],cfg['shape_alpha'])
            check(f'candidate_reconstructed_H{h}',np.array_equal(point,arrays[f'frozen_candidate_H{h}']))
            check(f'original_HDSF_exact_H{h}',np.array_equal(prior,arrays[f'original_HDSF_H{h}']))
            adopted=point if s['tasks'][str(h)]['adopt'] else prior
            check(f'adopted_unchanged_after_TEST_H{h}',np.array_equal(adopted,arrays[f'adopted_H{h}']))
            if cfg['level_alpha']==0:
                check(f'daily_mean_preserved_H{h}',np.allclose(point.reshape(57,-1,96).mean(-1),prior.reshape(57,-1,96).mean(-1),atol=1e-12,rtol=0))
            check(f'finite_bounded_H{h}',np.isfinite(point).all() and (point>=0).all() and (point<=1).all())
            chosen_early=exp.metric('dev',h,exp.adapt(exp.base('dev',h),exp.route_delta(cfg['route'],'dev',h),cfg['level_alpha'],cfg['shape_alpha']),'early')
            check(f'early_metrics_exact_H{h}',chosen_early==cfg['early_metrics'])
            y,m=exp.common.truth(exp.d,exp.origins('test',h),h)
            check(f'valid_point_count_H{h}',int(m.sum())=={1:49243,7:114902}[h])
            for role,val in report['scores'][str(h)].items():
                pp=arrays[f'{role}_H{h}'];expected=val['metrics'];actual=exp.metric('test',h,pp)
                check(f'all_metrics_recomputed_{role}_H{h}',actual==expected)
                macro=float(np.mean(100*((abs(y-pp)*m).sum((1,2)))/((abs(y)*m).sum((1,2)))))
                check(f'manual_macro57_{role}_H{h}',macro==expected['wape'])
                yd,pd,md=y.reshape(57,-1,96),pp.reshape(57,-1,96),m.reshape(57,-1,96);ok=md.all(-1)
                daily=float(np.mean(100*(abs(yd.mean(-1)-pd.mean(-1))*ok).sum(1)/(abs(yd.mean(-1))*ok).sum(1)))
                check(f'manual_complete_day_WAPE_{role}_H{h}',daily==expected['daily_wape'])
                lo,hi=arrays[f'{role}_lower80_H{h}'],arrays[f'{role}_upper80_H{h}']
                check(f'interval_metrics_{role}_H{h}',exp.interval_metrics(y,pp,m,lo,hi)==val['interval'])
    # The bounded projection must preserve requested daymeans even when clipping.
    toy=np.full((57,1,96),.7);perturb=np.tile(np.linspace(-.8,.8,96),(57,1,1))
    for level in p['coefficient_grid']:
        for shape in p['coefficient_grid']:
            pp=exp.adapt(toy,perturb,level,shape)
            check(f'projection_bracket_mean_{level}_{shape}',np.allclose(pp.mean(-1),toy.mean(-1)+level*perturb.mean(-1),atol=1e-12,rtol=0))
    check('retrospective_after_freeze',read(q/'retrospective_analysis.json')['created_utc']>s['created_utc'] and not read(q/'retrospective_analysis.json')['used_for_selection'])
    check('TEST_did_not_change_selection',sha(q/'selection.json')==report['selection_sha256'])
    out=dict(created_utc=utc(),all_passed=True,checks=checks,check_count=len(checks),runtime_check_count=runtimechecks,
             neural_training_runs=len(trained),real_optimizer_updates=len(trained)*160,selection_sha256=sha(q/'selection.json'),
             predictions_sha256=sha(q/'predictions.npz'),independent_confirmation=False,overfitting_absence_certified=False)
    (q/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
