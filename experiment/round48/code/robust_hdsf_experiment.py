"""R48 historical-volatility caps and same-cell/type hierarchical residual sharing."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def initialize(root,register=False):
    global T,B,R,Q,E,U,P,groups
    T=Path(root);B=T/'experiment';R=B/'round48';Q=R/'results'
    for name in ['code','docs','results']:(R/name).mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(B/'round47/code'))
    import predict_user_hdsf as user
    U=user;U.initialize(T);E=U.E
    meta=E.F.data.meta.iloc[E.F.canon]
    groups=[(str(c),bool(v)) for c,v in zip(meta['cell'],meta['is_mvpp'])]
    if register:
        assert not (R/'docs/protocol.json').exists()
        P=dict(round=48,created_utc=utc(),script_sha256=sha(__file__),dataset='OriginalGooglePowerData2019 only',
            source_user_configuration_sha256=sha(B/'round47/deployment/selected_configuration.json'),
            source_research_selection_sha256=sha(B/'round47/results/selection.json'),
            source_R47_predictions_sha256=sha(B/'round47/results/predictions.npz'),
            source_R47_summary_sha256=sha(B/'round47/results/summary.json'),
            new_neural_training_runs=0,method='Postprocessing actually trained powerforecasts; notadditionalbackprop or newfoundationarchitecture.',
            sources='Observed historicalpower andimmutable forecastcomponents; staticcell/type groups. No NLR orCPUClusterData.',
            historical_scale='Last7 observeddaymeans withfull96valid; robustMAD offirstdifferences, floor max(.0005,.001recentmean). Scale grows sqrt(lead_day) for H7.',
            group_shrink='Groups=samecell andis_mvpp; minimum3PDU. Medianrelativepredictedcorrection, shrink.25/.5; recenter shape aftersharing.',
            candidates1=['current','hard_cap_05','hard_cap_1','soft_cap_1','soft_cap_2','group_25','group_50','group_50_hard1'],
            candidates7=['current','level_hard_quarter','level_soft_half','level_group_half','level_group_hard_half','shape075','shape1','shape_group075'],
            coefficient_learning='No continuousparameterfitting; exactly8registeredstructural variantsperhorizon.',
            early='Select only targetdates≤D18; curveprimary, daily≤base+.002pp,peak≤+.05pp,P95APE≤+.02pp,absbias≤+.02pp. Prefercurrent within.0001pp.',
            late='Frozenearlycandidate onD19/D21 orD19–22: curve anddaily nonworse,meanunique-dategain>0,positive majority;secondarysameconstraints. No95%significanceclaim.',
            comparison_baseline='ExactR47candidate explicitly adoptedbyuser; do not compare headline against an older/easierbaseline.',
            TEST='FreezebeforefirstnewTESTmetrics; scorecurrent andonefixedcandidate perhorizon only. NoTESTreranking.',
            user_adopted_R47_after_results=True,historical_test_informed_design=True,independent_confirmation=False,
            metrics='Unchanged9.28macro57/full96validdaily plusallsupplementarypoint,peak,bias,domain,empirical80intervalfields.')
        save(R/'docs/protocol.json',P)
    else:P=read(R/'docs/protocol.json')
    assert P['script_sha256']==sha(__file__)
    for name,key in [('round47/deployment/selected_configuration.json','source_user_configuration_sha256'),
        ('round47/results/selection.json','source_research_selection_sha256'),('round47/results/predictions.npz','source_R47_predictions_sha256'),
        ('round47/results/summary.json','source_R47_summary_sha256')]:assert sha(B/name)==P[key]

def sigma(o):
    y,m=E.d['y'][:,max(0,o-7):o],E.d['valid'][:,max(0,o-7):o]
    means=y.mean(-1);valid=m.all(-1);out=[]
    for row,ok in zip(means,valid):
        values=np.diff(row)[ok[1:]&ok[:-1]]
        mad=1.4826*np.median(abs(values-np.median(values))) if len(values) else 0.
        recent=float(np.median(row[ok])) if ok.any() else float(np.median(row))
        out.append(max(float(mad),.0005,.001*abs(recent)))
    return np.asarray(out)
def share(values,scale,strength):
    result=values.copy();rel=values/np.maximum(scale,1e-8)
    for group in dict.fromkeys(groups):
        ix=np.array([i for i,g in enumerate(groups) if g==group])
        if len(ix)>=3:
            center=np.median(rel[ix],axis=0)
            result[ix]=(1-strength)*values[ix]+strength*center*scale[ix]
    return result
def cap(level,limit,soft=False):
    return level/(1+abs(level)/limit) if soft else np.clip(level,-limit,limit)
def predict(phase,h,variant):
    if variant=='current':return U.point(phase,h)
    old=E.base(phase,h);cfg=U.C['tasks'][str(h)]
    delta=E.route_delta(cfg['route'],phase,h)
    vd=delta.reshape(57,len(E.origins(phase,h)),h,96);mu=vd.mean(-1,keepdims=True);shape=vd-mu
    pd=old.reshape(57,len(E.origins(phase,h)),h,96);level=mu*cfg['level_alpha'];sc=shape*cfg['shape_alpha']
    limits=np.stack([sigma(o) for o in E.origins(phase,h)],axis=1)[:,:,None,None]*np.sqrt(np.arange(1,h+1))[None,None,:,None]
    scales=pd.mean(-1,keepdims=True)
    if h==1:
        if variant=='hard_cap_05':level=cap(mu,.5*limits)
        elif variant=='hard_cap_1':level=cap(mu,limits)
        elif variant=='soft_cap_1':level=cap(mu,limits,True)
        elif variant=='soft_cap_2':level=cap(mu,2*limits,True)
        elif variant=='group_25':level=share(mu,scales,.25)
        elif variant=='group_50':level=share(mu,scales,.5)
        elif variant=='group_50_hard1':level=cap(share(mu,scales,.5),limits)
        else:raise ValueError(variant)
    else:
        if variant=='level_hard_quarter':level=.25*cap(mu,limits)
        elif variant=='level_soft_half':level=.5*cap(mu,limits,True)
        elif variant=='level_group_half':level=.5*share(mu,scales,.5)
        elif variant=='level_group_hard_half':level=.5*cap(share(mu,scales,.5),limits)
        elif variant=='shape075':sc=.75*shape
        elif variant=='shape1':sc=shape
        elif variant=='shape_group075':
            sc=.75*share(shape,scales,.25);sc-=sc.mean(-1,keepdims=True)
        else:raise ValueError(variant)
    point=E.adapt(old,(level+sc).reshape(old.shape),1.,1.)
    assert np.isfinite(point).all() and (point>=0).all() and (point<=1).all()
    return point
def eligible(metrics,base,late=False):
    return metrics['wape']<=base['wape']+1e-12 and metrics['daily_wape']<=base['daily_wape']+(1e-12 if late else .002) and metrics['peak_mape']<=base['peak_mape']+.05 and metrics['p95_ape']<=base['p95_ape']+.02 and metrics['absolute_bias_percent']<=base['absolute_bias_percent']+.02

def freeze():
    assert not (Q/'selection.json').exists()
    result=dict(created_utc=utc(),protocol_sha256=sha(R/'docs/protocol.json'),DEV_only=True,tasks={});rows=[]
    for h in [1,7]:
        base=U.point('dev',h);bs=E.metric('dev',h,base,'early');cases=[]
        for variant in P['candidates'+str(h)]:
            point=predict('dev',h,variant);score=E.metric('dev',h,point,'early')
            cases.append(dict(variant=variant,early_metrics=score,early_eligible=eligible(score,bs)))
        good=[x for x in cases if x['early_eligible']];best=min(x['early_metrics']['wape'] for x in good)
        near=[x for x in good if x['early_metrics']['wape']<=best+.0001]
        chosen=next((x for x in near if x['variant']=='current'),min(near,key=lambda x:x['early_metrics']['wape']))
        point=predict('dev',h,chosen['variant']);score=E.metric('dev',h,point,'late');base_late=E.metric('dev',h,base,'late')
        gains=E.date_gains('dev',h,base,point,'late')
        supported=chosen['variant']=='current' or (eligible(score,base_late,True) and gains['mean_gain_pp']>0 and sum(x>0 for x in gains['gains_pp'])>len(gains['gains_pp'])/2)
        result['tasks'][str(h)]=dict(candidate=chosen,late_metrics=score,base_late_metrics=base_late,late_gains=gains,DEV_supported=bool(supported),early_candidates=cases)
        rows.extend(dict(horizon_days=h,**x) for x in cases)
    save(Q/'selection.json',result);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii');E.export(Q/'稳健校正开发期全部候选.csv',rows)
    print(json.dumps({h:dict(variant=x['candidate']['variant'],supported=x['DEV_supported'],late_gain=x['late_gains']) for h,x in result['tasks'].items()},ensure_ascii=True))
def score():
    s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()
    arrays={};summary={};rows=[]
    for hs,item in s['tasks'].items():
        h=int(hs);current=predict('test',h,'current');trial=predict('test',h,item['candidate']['variant'])
        dc,dt=predict('dev',h,'current'),predict('dev',h,item['candidate']['variant'])
        y,m=E.common.truth(E.d,E.origins('test',h),h);dy,dm=E.common.truth(E.d,E.origins('dev',h),h);summary[hs]={}
        for role,point,dev in [('user_current_R47',current,dc),('DEV_frozen_R48_candidate',trial,dt)]:
            metric=E.metric('test',h,point);low=np.array([np.quantile((dy-dev)[i][dm[i]],.1) for i in range(57)]);high=np.array([np.quantile((dy-dev)[i][dm[i]],.9) for i in range(57)])
            lo,hi=np.clip(point+low[:,None,None],0,1),np.clip(point+high[:,None,None],0,1);iv=E.interval_metrics(y,point,m,lo,hi)
            yd,pd,md=y.reshape(57,-1,96),point.reshape(57,-1,96),m.reshape(57,-1,96);ok=md.any(-1)
            peak_under=float(100*(np.where(md,pd,-np.inf).max(-1)[ok]<np.where(md,yd,-np.inf).max(-1)[ok]).mean())
            summary[hs][role]=dict(metrics=metric,interval=iv,peak_underestimate_percent=peak_under)
            arrays[f'{role}_H{h}']=point;arrays[f'dev_{role}_H{h}']=dev;arrays[f'{role}_lower80_H{h}']=lo;arrays[f'{role}_upper80_H{h}']=hi
            rows.append(dict(horizon_days=h,model=role,variant=item['candidate']['variant'],DEV_supported=item['DEV_supported'],**metric,peak_underestimate_percent=peak_under,**{'interval80_'+k:v for k,v in iv.items()}))
    np.savez_compressed(Q/'predictions.npz',**arrays);E.export(Q/'第48轮稳健校正全指标.csv',rows)
    save(Q/'summary.json',dict(created_utc=utc(),scores=summary,rows=rows,selection_sha256=sha(Q/'selection.json'),protocol_sha256=sha(R/'docs/protocol.json'),
        predictions_sha256=sha(Q/'predictions.npz'),new_neural_training_runs=0,independent_confirmation=False,historical_test_informed_design=True,
        current_user_adoption_unchanged=True))
    print(json.dumps({h:{role:dict(curve=x['metrics']['wape'],daily=x['metrics']['daily_wape'],peak=x['metrics']['peak_mape']) for role,x in v.items()} for h,v in summary.items()},ensure_ascii=True))
def verify():
    s=read(Q/'selection.json');report=read(Q/'summary.json');checks=[]
    def check(name,good):checks.append(dict(name=name,passed=bool(good)));assert good,name
    check('selection_frozen',sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()==report['selection_sha256'])
    check('registered_before_selection',P['created_utc']<s['created_utc']<report['created_utc'])
    check('source_user_adoption_unchanged',sha(B/'round47/deployment/selected_configuration.json')==P['source_user_configuration_sha256'])
    yoriginal,moriginal=E.d['y'].copy(),E.d['valid'].copy()
    for h in [1,7]:
        for variant in P['candidates'+str(h)]:
            original=predict('dev',h,variant)
            # Perturb only after firstorigin and compare that origin independently;
            # later origins may legitimately observe earlier forecast days.
            o=E.origins('dev',h)[0];E.d['y'][:,o:]=.987;E.d['valid'][:,o:]=False
            poison=predict('dev',h,variant)[:,0]
            check(f'future_input_poison_H{h}_{variant}',np.array_equal(poison,original[:,0]))
            E.d['y'][:]=yoriginal;E.d['valid'][:]=moriginal
    with np.load(Q/'predictions.npz') as z:
        for h in [1,7]:
            v=s['tasks'][str(h)]['candidate']['variant'];trial=predict('test',h,v)
            check(f'replay_H{h}',np.array_equal(trial,z[f'DEV_frozen_R48_candidate_H{h}']))
            check(f'current_matches_adopted_H{h}',np.array_equal(z[f'user_current_R47_H{h}'],U.point('test',h)))
            for role,val in report['scores'][str(h)].items():
                p=z[f'{role}_H{h}'];actual=E.metric('test',h,p)
                check(f'all_metrics_H{h}_{role}',actual==val['metrics'])
                y,m=E.common.truth(E.d,E.origins('test',h),h)
                check(f'interval_H{h}_{role}',E.interval_metrics(y,p,m,z[f'{role}_lower80_H{h}'],z[f'{role}_upper80_H{h}'])==val['interval'])
    check('no_new_gradient_training_claim',report['new_neural_training_runs']==0)
    save(Q/'verification.json',dict(created_utc=utc(),all_passed=True,checks=checks,check_count=len(checks),
        selection_sha256=sha(Q/'selection.json'),source_configuration_sha256=P['source_user_configuration_sha256'],independent_confirmation=False))
    print(json.dumps(dict(all_passed=True,checks=len(checks))))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--phase',choices=['register','freeze','score','verify'],required=True);a=ap.parse_args()
    initialize(a.root,register=a.phase=='register')
    if a.phase=='freeze':freeze()
    elif a.phase=='score':score()
    elif a.phase=='verify':verify()
    else:print('Registered16structuralcasesbeforeDEV')
