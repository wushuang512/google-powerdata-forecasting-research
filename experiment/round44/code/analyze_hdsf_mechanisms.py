"""Prespecified retrospective diagnostics; no foundation inference or deployment edits."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
except ImportError:
    # BLAS environment limits were set before NumPy import; no new dependency is required.
    pass
sys.dont_write_bytecode = True

T = Path(sys.argv[1]); B = T/'experiment'; R = B/'round44'; Q = R/'results'
A = Q/'mechanism_analysis_20261008'; A.mkdir(exist_ok=True)
sys.path[:0] = [str(B/'code'), str(B/'round29/code')]
import common
from aligned_metrics import evaluate

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def table(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (A/name).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

selection_path = Q/'selection.json'; freeze_sha = sha(selection_path)
assert freeze_sha == (Q/'selection.sha256').read_text()
protocol = read(R/'docs/protocol.json')
assert sha(B/'round29/code/aligned_metrics.py') == protocol['evaluation_sha256']
assert read(Q/'science_verification.json')['all_passed']
source = R/'reference/PowerData_Prediction/power_forecast/attainment_fusion.py'
f = ast.parse(source.read_text(encoding='utf-8'))
allowed = {'weighted_breakpoint','fit_simplex','fit_segments','predict_segments'}
namespace = {'np': np}
exec(compile(ast.Module(body=[n for n in f.body if isinstance(n,ast.FunctionDef) and n.name in allowed],type_ignores=[]),str(source),'exec'),namespace)
fit = namespace['fit_segments']
selection = read(selection_path)['configs']; comparison = read(Q/'hdsf_comparison.json')
d = common.load(); archive = np.load(Q/'predictions.npz')
source_order = np.argsort(read(Q/'data_alignment.json')['reorder_index'])
analysis_protocol = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    retrospective=True, test_previously_seen=True, independent_confirmation=False,
    purpose='Mechanism attribution, not candidate selection; all registered diagnostics reported even when negative.',
    frozen_selection_sha256=freeze_sha, input_prediction_sha256=sha(Q/'predictions.npz'),
    canonical_evaluation_sha256=protocol['evaluation_sha256'], foundation_inference=False,
    deployed_models_unchanged=True, fitting_only_dev_origins=True,
    fixed_ablations=['source A only', 'source B only', 'uniform members',
                    'one segment, DEV refit A/B with original mix',
                    '7 daily segments, DEV refit A/B with original mix (H7 only)',
                    'each nonzero member removed and existing weights renormalized; no refitting'],
    limitations='Structural refits hold original mix fixed; original mix was selected for half-day segments. Drops do not refit remaining weights. Neither is an independently validated replacement. DEV fitted errors are resubstitution, not out-of-fold validation.',
    diagnostic_only=['signed residual correlations','triangle-inequality cancellation','MSE level/shape decomposition','PDU/origin/horizon contributions'])
assert not (A/'protocol.json').exists(), 'Do not overwrite a previous analysis protocol'
save(A/'protocol.json', analysis_protocol)

weights_rows=[]; ablation_rows=[]; domain_rows=[]; origin_rows=[]; segment_rows=[]; corr_rows=[]; findings={}
metrics = ['wape','daily_wape','mae','rmse','peak_mape','absolute_bias_percent','daily_mae','daily_rmse','hit2_percent','hit5_percent']

def reconstruct(bank, weights):
    steps = np.minimum(np.arange(bank.shape[2])*len(weights)//bank.shape[2],len(weights)-1)
    return np.einsum('iotm,tm->iot', bank, weights[steps])

def score_row(H, label, p, dp, oo, dev):
    ts=evaluate(common,d,oo,p,H); ds=evaluate(common,d,dev,dp,H)
    ablation_rows.append(dict(horizon_days=H,case=label,**{'test_'+k:ts[k] for k in metrics},**{'dev_fitted_'+k:ds[k] for k in metrics}))
    return ts

def mse_parts(y,p,m):
    y=y.reshape(57,-1,96); p=p.reshape(y.shape); m=m.reshape(y.shape)
    ok=m.all(-1); e=(p-y)[ok]; mu=e.mean(-1)
    level=float(np.mean(mu**2)); shape=float(np.mean((e-mu[:,None])**2))
    assert abs(level+shape-float(np.mean(e**2)))<1e-14
    return dict(complete_days=int(ok.sum()),mse=level+shape,level_mse=level,shape_mse=shape,
                daily_signed_bias=float(mu.mean()),daily_error_sd=float(mu.std()),
                level_rmse=float(np.sqrt(level)),shape_rmse=float(np.sqrt(shape)))

for H in (1,7):
    print('Analyzing cached predictions H'+str(H),flush=True)
    task='24h' if H==1 else '7d'; cfg=selection[task]; names=cfg['methods']; segments=cfg['segments'][0]
    oo=common.TEST24 if H==1 else common.TEST7; dev=common.DEV24 if H==1 else common.DEV7
    y,m=common.truth(d,oo,H); yd,md=common.truth(d,dev,H)
    bank=np.stack([archive[f'{name}_H{H}'] for name in names],axis=-1)
    devbank=np.stack([archive[f'dev_{name}_H{H}'] for name in names],axis=-1)
    wa=np.asarray(cfg['component_a'][str(segments)]); wb=np.asarray(cfg['component_b'][str(segments)])
    padded=np.pad(wa,((0,0),(0,1))); w=(1-cfg['mix'])*padded+cfg['mix']*wb
    p=archive[f'hdsf_H{H}']; dp=archive[f'dev_hdsf_H{H}']
    assert np.max(abs(reconstruct(bank,w)-p))<2e-12 and np.max(abs(reconstruct(devbank,w)-dp))<2e-12
    prior_row=next(r for r in comparison['unique_rows'] if r['horizon_days']==H and r['status']=='当前综合推荐')
    prior=np.load(B/prior_row['source']['path'])[prior_row['source']['array']]
    assert abs(evaluate(common,d,oo,prior,H)['wape']-prior_row['wape'])<1e-12
    score_row(H,'HDSF frozen',p,dp,oo,dev)
    score_row(H,'A foundation-only source weights',reconstruct(bank,padded),reconstruct(devbank,padded),oo,dev)
    score_row(H,'B local-included source weights',reconstruct(bank,wb),reconstruct(devbank,wb),oo,dev)
    uniform=np.full((1,len(names)),1/len(names))
    score_row(H,'All members uniform',reconstruct(bank,uniform),reconstruct(devbank,uniform),oo,dev)
    for nsegments in ([1] if H==1 else [1,7]):
        tensor=devbank[source_order].transpose(1,3,2,0); truth=yd[source_order].transpose(1,2,0); valid=md[source_order].transpose(1,2,0)
        aw=fit(tensor[:,:-1],truth,valid,nsegments); bw=fit(tensor,truth,valid,nsegments)
        fixed=(1-cfg['mix'])*np.pad(aw,((0,0),(0,1)))+cfg['mix']*bw
        score_row(H,f'DEV-refit {nsegments} segments, original mix fixed',reconstruct(bank,fixed),reconstruct(devbank,fixed),oo,dev)
        save(A/f'global_or_daily_weights_H{H}_segments{nsegments}.json',dict(methods=names,weights=fixed.tolist(),mix=cfg['mix'],fitted_origins=dev))
    for k,name in enumerate(names):
        if w[:,k].max()<1e-12: continue
        rw=w.copy(); rw[:,k]=0; sums=rw.sum(-1)
        if np.any(sums<1e-12):
            # Single-expert segment cannot be renormalized; fixed uniform remaining fallback is explicit.
            rw[sums<1e-12]=1.; rw[sums<1e-12,k]=0.
        rw/=rw.sum(-1)[:,None]
        score_row(H,'Drop '+name+'; renormalize; uniform fallback if needed',reconstruct(bank,rw),reconstruct(devbank,rw),oo,dev)
    for s in range(segments):
        for k,name in enumerate(names): weights_rows.append(dict(horizon_days=H,forecast_halfday=s+1,method=name,weight=float(w[s,k])))
    den=(abs(y)*m).sum((1,2)); e=abs(p-y); olde=abs(prior-y)
    pw=100*(e*m).sum((1,2))/den; ow=100*(olde*m).sum((1,2))/den
    train=d['y'][:,:14].mean(-1); trainok=d['valid'][:,:14].all(-1); train=np.where(trainok,train,np.nan)
    cv=100*np.nanstd(train,axis=1)/np.nanmean(train,axis=1)
    for i,name in enumerate(d['names']): domain_rows.append(dict(horizon_days=H,pdu=str(name),cell=str(d['cells'][i]),prior_wape=float(ow[i]),hdsf_wape=float(pw[i]),gain_pp=float(ow[i]-pw[i]),train_first14_daily_cv_percent=float(cv[i])))
    for j,o in enumerate(oo):
        dd=(abs(y[:,j])*m[:,j]).sum(-1)
        op=float(np.mean(100*(olde[:,j]*m[:,j]).sum(-1)/dd)); hp=float(np.mean(100*(e[:,j]*m[:,j]).sum(-1)/dd))
        origin_rows.append(dict(horizon_days=H,origin_day=o,prior_wape=op,hdsf_wape=hp,gain_pp=op-hp))
    for s in range(segments):
        sl=slice(s*48,(s+1)*48)
        gain=float(np.mean(100*((olde[:,:,sl]-e[:,:,sl])*m[:,:,sl]).sum((1,2))/den))
        segment_rows.append(dict(horizon_days=H,halfday=s+1,gain_contribution_pp= gain))
    assert abs(sum(r['gain_contribution_pp'] for r in segment_rows if r['horizon_days']==H)-float((ow-pw).mean()))<1e-12
    ids=np.arange(H*96)*segments//(H*96); pointw=w[ids]
    member_errors=bank-y[...,None]
    signed=member_errors*pointw[None,None,:,:]
    triangle=abs(signed).sum(-1)
    reference=float(np.mean(100*(triangle*m).sum((1,2))/den))
    active=pointw>1e-12
    low=np.where(active[None,None,:,:],member_errors,np.inf).min(-1)
    high=np.where(active[None,None,:,:],member_errors,-np.inf).max(-1)
    opposite=float(100*((low<0)&(high>0)&m).sum()/m.sum())
    assert np.max(abs(signed.sum(-1)-(p-y)))<2e-12
    for phase,bk,yy,mask in [('DEV',devbank,yd,md),('TEST',bank,y,m)]:
        err=bk-yy[...,None]
        for s in range(segments):
            sl=slice(s*48,(s+1)*48); active_ids=np.flatnonzero(w[s]>1e-12); errors=err[:,:,sl][mask[:,:,sl]]
            for ai,i in enumerate(active_ids):
                for j in active_ids[ai+1:]:
                    corr_rows.append(dict(horizon_days=H,phase=phase,halfday=s+1,member1=names[i],member2=names[j],signed_error_correlation=float(np.corrcoef(errors[:,i],errors[:,j])[0,1]),opposite_error_sign_percent=float(100*np.mean(errors[:,i]*errors[:,j]<0))))
    yy=y.reshape(57,-1,96); pp=p.reshape(yy.shape); mm=m.reshape(yy.shape); old=prior.reshape(yy.shape); validday=mm.all(-1)
    hdaily=abs(pp.mean(-1)-yy.mean(-1)); odaily=abs(old.mean(-1)-yy.mean(-1))
    oldpeak=np.where(mm,old,-np.inf).max(-1); hpeak=np.where(mm,pp,-np.inf).max(-1); truepeak=np.where(mm,yy,-np.inf).max(-1)
    valid_peak=mm.any(-1)
    group={}
    for label,sel in [('a5',d['cells']=='a'),('other52',d['cells']!='a')]:
        group[label]=dict(domains=int(sel.sum()),mean_gain_pp=float((ow-pw)[sel].mean()),contribution_to_57_macro_pp=float((ow-pw)[sel].sum()/57))
    findings[str(H)]=dict(prior=prior_row['model_name'],members=names,mean_effective_weights=w.mean(0).tolist(),
        improved_pdu_vs_prior=int((pw<ow-1e-12).sum()),worse_pdu_vs_prior=int((pw>ow+1e-12).sum()),
        improved_origins_vs_prior=sum(r['gain_pp']>0 for r in origin_rows if r['horizon_days']==H),
        daily_error_improved_count=int(((hdaily<odaily-1e-12)&validday).sum()),valid_daily_count=int(validday.sum()),
        pdu_gain_median_pp=float(np.median(ow-pw)),groups=group,
        median_training_daily_cv_percent=float(np.nanmedian(cv)),
        convex_abs_error_reference_wape=reference,actual_hdsf_wape=float(pw.mean()),
        cancellation_credit_pp=reference-float(pw.mean()),opposite_active_expert_sign_percent=opposite,
        cancellation_note='Retrospective triangle-inequality decomposition using actual truth; the reference is not an implementable forecast or independent model score.',
        prior_mse_decomposition=mse_parts(y,prior,m),hdsf_mse_decomposition=mse_parts(y,p,m),
        prior_peak_signed_percent=float(np.mean(100*(oldpeak-truepeak)[valid_peak]/truepeak[valid_peak])),
        hdsf_peak_signed_percent=float(np.mean(100*(hpeak-truepeak)[valid_peak]/truepeak[valid_peak])),
        prior_peak_underestimated_percent=float(100*(oldpeak[valid_peak]<truepeak[valid_peak]).mean()),
        hdsf_peak_underestimated_percent=float(100*(hpeak[valid_peak]<truepeak[valid_peak]).mean()))

assert sha(selection_path)==freeze_sha
assert sha(B/'round29/code/aligned_metrics.py')==protocol['evaluation_sha256']
table('frozen_weights.csv',weights_rows);table('fixed_ablations.csv',ablation_rows)
table('per_pdu_comparison.csv',domain_rows);table('per_origin_comparison.csv',origin_rows)
table('halfday_gain_contributions.csv',segment_rows);table('active_member_error_pairs.csv',corr_rows)
save(A/'analysis.json',dict(findings=findings,ablations=ablation_rows,bootstrap=comparison['bootstrap'],
    evaluator_unchanged=True,selection_unchanged=True,all_prediction_reconstructions_passed=True,
    retrospective=True,independent_confirmation=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print(json.dumps({'findings':findings,'ablations':[{k:r[k] for k in ['horizon_days','case','test_wape','test_daily_wape','test_peak_mape','dev_fitted_wape','dev_fitted_daily_wape']} for r in ablation_rows]},ensure_ascii=True),flush=True)
