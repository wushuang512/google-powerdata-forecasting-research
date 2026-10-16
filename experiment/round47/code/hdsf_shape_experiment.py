"""R47 registered level/shape fine-tuning residual adaptation; unchanged metrics."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def export(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def initialize(root,register=False):
    global T,B,R,Q,F,P,d,common,evaluate,interval_metrics
    T=Path(root);B=T/'experiment';R=B/'round47';Q=R/'results'
    for sub in ['code','docs','models','results']:(R/sub).mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(B/'round46/code'))
    import hdsf_foundation_fusion as fusion
    F=fusion;F.initialize(T);d,common,evaluate,interval_metrics=F.d,F.common,F.evaluate,F.interval_metrics
    if register:
        assert not (R/'docs/protocol.json').exists()
        bindings={str(p.relative_to(T)):sha(p) for p in [B/'round44/results/selection.json',B/'round44/results/predictions.npz',
            B/'round44/results/source_dataset.joblib',B/'round46/results/selection.json',B/'round46/results/summary.json',
            B/'round46/code/hdsf_foundation_adapters.py',B/'round46/code/hdsf_foundation_stage.py',B/'round46/code/hdsf_foundation_fusion.py',
            B/'round29/code/aligned_metrics.py',B/'code/common.py']}
        for family in F.proto['families']:
            cfg=read(B/'round46/results/selection.json')['families'][family]['config']
            for phase in ['dev','test']:
                for label in ['original_control',cfg]:
                    path=B/'round46/results'/f'{family}_{phase}_{label}.npz';bindings[str(path.relative_to(T))]=sha(path)
        P=dict(round=47,created_utc=utc(),dataset='OriginalGooglePowerData2019 only',
            hypotheses='Separate fine-tuned neural deltas from fresh local-refit deltas; adapt daily level and within-day shape separately. Prior TESTobservations motivate hypothesis, never choose newparameters on TEST.',
            training_script_sha256=sha(R/'code/hdsf_shape_training.py'),analysis_script_sha256=sha(__file__),
            round46_selection_sha256=sha(B/'round46/results/selection.json'),bindings=bindings,
            new_families=['chronos2','patchtst'],new_recipes=['curve','structure'],steps=160,lr=1.5e-6,
            scope='Output head plus actual final encoder block, AdamW.01/normclip1/batch2; restart originalcheckpoint separately at cutoff14/22.',
            loss_recipes={'curve':dict(daily=.1,peak=.025,shape=0.,slope=.05),
                          'structure':dict(daily=.25,peak=.05,shape=.5,slope=.1)},
            common_loss='curveL1 +.05pinball +.15teacherpointMSE +.05relativeparameterprox +.1bounds; truefuture is losstarget only.',
            training_horizons={'chronos2':[96],'patchtst':[96,672]},seed=20261012,
            training_context_days=[3,7,14],training_cutoff_DEV=14,training_cutoff_final=22,
            dev24=[14,16,18,20],dev7=[14,15],test24=list(range(22,31)),test7=[22,23,24],
            early_selection_dates='D15–D18; H1 hasD15/D17',late_validation_dates='D19–D22; H1 hasD19/D21',
            routes_H1=['r46_neural_only','chronos2_curve_weighted','chronos2_structure_weighted','patchtst_curve_direct','patchtst_structure_direct'],
            routes_H7=['r46_neural_only','patchtst_curve_weighted','patchtst_structure_weighted','patchtst_curve_direct','patchtst_structure_direct'],
            coefficient_grid=[0.,.5,1.],
            projection='Point=originalHDSF + levelalpha*daymean(delta) + shapealpha*(delta-daymean(delta)). Allmeans useforecast values only. Bounded0/1projection preserves requested forecastdailymeans, notphysical energy conservation.',
            local_members='Keep exactoriginal local members andHDSFweights; do not silently replace with R46freshlocal fits.',
            early_choice='Filter dailyWAPE≤original+.002pp andpeakMAPE≤original+.1pp. MinimumcurveWAPE, preferlevel0/smallercoefficients within.0025pp. Freeze one nonzero diagnostic even if noearlyimprovement.',
            late_upgrade_gate='Same fixed candidate; mean pairedunique-date curve gain>max(.005pp,SE), positive majority dates; late/fullDEVcurve nonworse,daily≤+.002pp,peak≤+.1pp. Otherwise retainoriginal HDSF.',
            no_TEST_parameter_selection=True,old_HDSF_design_used_whole_DEV=True,historical_test_informed_design=True,
            independent_confirmation=False,new_Word=False,device='CUDA,onefamilyjob atatime,existing researchenvironment',
            retrospective_ablation='After newselection freezes, oldR46 per-family/neural-only/local-only substitutions; fixeddiagnostics only.',
            metrics='Unchanged9.28 macroPDU curveWAPE/equal57 andcomplete96pointdaily; allpreviouspoint/peak/error/hit/bias/domain/empirical80intervaldefinitions.',
            source_license='Original licensesretained; no newdata orweightsdownload; TimesFM3 remains noncommercialresearch.')
        save(R/'docs/protocol.json',P)
    else:P=read(R/'docs/protocol.json')
    assert P['analysis_script_sha256']==sha(__file__)
    assert P['training_script_sha256']==sha(R/'code/hdsf_shape_training.py')
    for name,digest in P['bindings'].items():assert sha(T/name)==digest,name

def origins(phase,h):return P[('dev' if phase=='dev' else'test')+('24' if h==1 else'7')]
def base(phase,h):return F.baseline(phase,h)
def member_weight(member,h):
    names=F.original_config['24h' if h==1 else'7d']['methods'];w=F.source_weights(h)
    return w[np.minimum(np.arange(h*96)//48,len(w)-1),names.index(member)]
def new_delta(family,recipe,phase,h,weighted):
    member='chronos2_long' if family=='chronos2' else'patchtstfm_long';key=f'{member}_H{h}'
    with np.load(Q/f'{family}_{phase}_{recipe}.npz') as updated,np.load(Q/f'{family}_{phase}_original_control.npz') as control:
        delta=updated[key]-control[key]
    return delta*member_weight(member,h)[None,None,:] if weighted else delta
def r46_deltas(phase,h):
    selection=read(B/'round46/results/selection.json');result={}
    for family,fs in selection['families'].items():
        old=F.members_original(phase,h);trained=F.family_bank(family,fs['config'],phase,h,1.)
        result[family]=sum(((point-old[member])*member_weight(member,h)[None,None,:] for member,point in trained.items()),start=np.zeros_like(base(phase,h)))
    return result
def route_delta(route,phase,h):
    if route=='r46_neural_only':return sum(r46_deltas(phase,h).values(),start=np.zeros_like(base(phase,h)))
    family,recipe,mode=route.split('_');return new_delta(family,recipe,phase,h,mode=='weighted')
def adapt(prior,delta,level,shape):
    shape0=prior.shape;p=prior.reshape(57,-1,96);v=delta.reshape(57,-1,96)
    mu=v.mean(-1,keepdims=True);raw=p+level*mu+shape*(v-mu)
    wanted=np.clip(p.mean(-1,keepdims=True)+level*mu,0,1)
    if (raw<0).any() or (raw>1).any():
        left=np.full_like(wanted,-2.);right=np.full_like(wanted,2.)
        for _ in range(60):
            mid=(left+right)/2;mean=np.clip(raw+mid,0,1).mean(-1,keepdims=True)
            left=np.where(mean<wanted,mid,left);right=np.where(mean>=wanted,mid,right)
        raw=np.clip(raw+(left+right)/2,0,1)
    assert np.isfinite(raw).all() and (raw>=0).all() and (raw<=1).all()
    assert np.allclose(raw.mean(-1,keepdims=True),wanted,atol=1e-12,rtol=0.)
    return raw.reshape(shape0)
def metric(phase,h,point,part='all'):
    mask=d['valid'].copy()
    if part=='early':mask[:,18:]=False
    elif part=='late':mask[:,:18]=False
    dm=dict(d);dm['valid']=mask
    with np.errstate(divide='ignore',invalid='ignore'):return evaluate(common,dm,origins(phase,h),point,h)
def date_gains(phase,h,old,point,part='late'):
    y,m=common.truth(d,origins(phase,h),h);dates=np.array([o+1+np.arange(h*96)//96 for o in origins(phase,h)])
    days=sorted(int(x) for x in np.unique(dates) if part=='all' or (x>18 if part=='late' else x<=18))
    gains=[]
    for day in days:
        keep=m&(dates[None]==day);den=(abs(y)*keep).sum((1,2));ok=den>0
        gain=100*((abs(y-old)-abs(y-point))*keep).sum((1,2))[ok]/den[ok]
        gains.append(float(gain.mean()))
    values=np.asarray(gains);return dict(days=days,gains_pp=gains,mean_gain_pp=float(values.mean()),paired_date_SE_pp=float(values.std(ddof=1)/np.sqrt(len(values))))

def freeze():
    path=Q/'selection.json'
    if path.exists():assert sha(path)==(Q/'selection.sha256').read_text().strip();return
    for family in P['new_families']:
        for recipe in P['new_recipes']:assert read(Q/f'{family}_{recipe}_dev_complete.json')['all_passed']
    selection=dict(created_utc=utc(),protocol_sha256=sha(R/'docs/protocol.json'),DEV_only=True,
        families_frozen_before_TEST=True,required_final_training=[],tasks={},historical_test_informed_design=True,independent_confirmation=False)
    allrows=[]
    for h in [1,7]:
        prior=base('dev',h);original=metric('dev',h,prior,'early');rows=[]
        for route in P[f'routes_H{h}']:
            delta=route_delta(route,'dev',h)
            for level in P['coefficient_grid']:
                for shape in P['coefficient_grid']:
                    if level==shape==0:continue
                    point=adapt(prior,delta,level,shape);score=metric('dev',h,point,'early')
                    eligible=score['daily_wape']<=original['daily_wape']+.002 and score['peak_mape']<=original['peak_mape']+.1
                    rows.append(dict(route=route,level_alpha=level,shape_alpha=shape,early_eligible=bool(eligible),early_metrics=score))
        good=[x for x in rows if x['early_eligible']] or rows;best=min(x['early_metrics']['wape'] for x in good)
        near=[x for x in good if x['early_metrics']['wape']<=best+.0025]
        chosen=min(near,key=lambda x:(x['level_alpha']>0,(x['level_alpha']>0)+(x['shape_alpha']>0),x['level_alpha']+x['shape_alpha'],x['early_metrics']['wape']))
        point=adapt(prior,route_delta(chosen['route'],'dev',h),chosen['level_alpha'],chosen['shape_alpha'])
        gains=date_gains('dev',h,prior,point);late=metric('dev',h,point,'late');oldlate=metric('dev',h,prior,'late');full=metric('dev',h,point);oldfull=metric('dev',h,prior)
        adopt=chosen['early_eligible'] and chosen['early_metrics']['wape']<original['wape'] and gains['mean_gain_pp']>max(.005,gains['paired_date_SE_pp']) and sum(v>0 for v in gains['gains_pp'])>len(gains['gains_pp'])/2
        adopt=adopt and late['wape']<=oldlate['wape'] and full['wape']<=oldfull['wape'] and late['daily_wape']<=oldlate['daily_wape']+.002 and full['daily_wape']<=oldfull['daily_wape']+.002 and late['peak_mape']<=oldlate['peak_mape']+.1 and full['peak_mape']<=oldfull['peak_mape']+.1
        selection['tasks'][str(h)]=dict(candidate=chosen,adopt=bool(adopt),late_date_gains=gains,late_metrics=late,
            original_late_metrics=oldlate,full_DEV_metrics=full,original_full_DEV_metrics=oldfull,early_candidates=rows)
        if chosen['route']!='r46_neural_only':
            family,recipe,_=chosen['route'].split('_')
            if [family,recipe] not in selection['required_final_training']:selection['required_final_training'].append([family,recipe])
        allrows.extend(dict(horizon_days=h,**x) for x in rows)
    save(path,selection);(Q/'selection.sha256').write_text(sha(path),encoding='ascii');export(Q/'开发期水平形状候选.csv',allrows)
    print(json.dumps(dict(frozen=True,tasks={h:dict(route=x['candidate']['route'],level=x['candidate']['level_alpha'],shape=x['candidate']['shape_alpha'],adopt=x['adopt'],late=x['late_date_gains']) for h,x in selection['tasks'].items()},required=selection['required_final_training']),ensure_ascii=True),flush=True)

def retrospective():
    s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()
    rows=[];domain_rows=[];day_rows=[];lead_rows=[]
    for h in [1,7]:
        prior=base('test',h);original=metric('test',h,prior);deltas=r46_deltas('test',h)
        cases={f'neural_family_only_{name}':prior+delta for name,delta in deltas.items()}
        cases['neural_only_all']=prior+sum(deltas.values(),start=np.zeros_like(prior))
        priorbank=F.members_original('test',h);locals_new=np.load(B/'round46/results/local_test.npz')
        member='local24' if h==1 else'local7';delta=(locals_new[f'{member}_H{h}']-priorbank[member])*member_weight(member,h)[None,None,:]
        cases['local_refit_only']=prior+delta
        with np.load(B/'round46/results/predictions.npz') as old:cases['all_neural_and_local_refit']=old[f'all_finetuned_original_weights_diagnostic_H{h}'].copy()
        y,m=common.truth(d,origins('test',h),h)
        for name,point in cases.items():
            score=metric('test',h,point);rows.append(dict(horizon_days=h,model=name,retrospective=True,used_for_selection=False,
                curve_gain_pp=original['wape']-score['wape'],daily_gain_pp=original['daily_wape']-score['daily_wape'],**score))
            num=((abs(y-prior)-abs(y-point))*m).sum((1,2));den=(abs(y)*m).sum((1,2))
            domain_rows.extend(dict(horizon_days=h,model=name,PDU=str(d['names'][i]),cell=str(d['cells'][i]),gain_pp=float(100*num[i]/den[i])) for i in range(57))
            dg=date_gains('test',h,prior,point,'all');day_rows.extend(dict(horizon_days=h,model=name,day=day,gain_pp=gain) for day,gain in zip(dg['days'],dg['gains_pp']))
            for lead in range(h):
                sl=slice(lead*96,(lead+1)*96);mm=m[:,:,sl];dd=(abs(y[:,:,sl])*mm).sum((1,2));nn=((abs(y[:,:,sl]-prior[:,:,sl])-abs(y[:,:,sl]-point[:,:,sl]))*mm).sum((1,2))
                lead_rows.append(dict(horizon_days=h,model=name,lead_day=lead+1,gain_pp=float(np.mean(100*nn/dd))))
    export(Q/'上一轮固定消融_仅事后解释.csv',rows);export(Q/'上一轮收益逐PDU.csv',domain_rows);export(Q/'上一轮收益逐日期.csv',day_rows);export(Q/'上一轮收益逐时距.csv',lead_rows)
    save(Q/'retrospective_analysis.json',dict(created_utc=utc(),rows=rows,used_for_selection=False,
        selection_sha256=sha(Q/'selection.json'),not_independent_confirmation=True))

def score():
    s=read(Q/'selection.json');assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()
    for family,recipe in s['required_final_training']:assert read(Q/f'{family}_{recipe}_test_complete.json')['all_passed']
    rows=[];summary={};arrays={}
    for h,choice in s['tasks'].items():
        h=int(h);cfg=choice['candidate'];prior=base('test',h);devprior=base('dev',h)
        point=adapt(prior,route_delta(cfg['route'],'test',h),cfg['level_alpha'],cfg['shape_alpha'])
        devpoint=adapt(devprior,route_delta(cfg['route'],'dev',h),cfg['level_alpha'],cfg['shape_alpha'])
        adopted=point if choice['adopt'] else prior;devadopted=devpoint if choice['adopt'] else devprior
        out={};y,m=common.truth(d,origins('test',h),h);dy,dm=common.truth(d,origins('dev',h),h)
        for role,p,dp in [('original_HDSF',prior,devprior),('frozen_candidate',point,devpoint),('adopted',adopted,devadopted)]:
            metrics=metric('test',h,p);low=np.array([np.quantile((dy-dp)[i][dm[i]],.1) for i in range(57)]);high=np.array([np.quantile((dy-dp)[i][dm[i]],.9) for i in range(57)])
            lo=np.clip(p+low[:,None,None],0,1);hi=np.clip(p+high[:,None,None],0,1);iv=interval_metrics(y,p,m,lo,hi)
            out[role]=dict(metrics=metrics,interval=iv);arrays[f'{role}_H{h}']=p;arrays[f'dev_{role}_H{h}']=dp;arrays[f'{role}_lower80_H{h}']=lo;arrays[f'{role}_upper80_H{h}']=hi
            rows.append(dict(horizon_days=h,model=role,route=cfg['route'],level_alpha=cfg['level_alpha'],shape_alpha=cfg['shape_alpha'],DEV_adopted=choice['adopt'],**metrics,**{'interval80_'+k:v for k,v in iv.items()}))
        summary[str(h)]=out
    np.savez_compressed(Q/'predictions.npz',**arrays);export(Q/'水平形状适配_统一全部指标.csv',rows)
    save(Q/'summary.json',dict(created_utc=utc(),scores=summary,rows=rows,selection_sha256=sha(Q/'selection.json'),
        predictions_sha256=sha(Q/'predictions.npz'),protocol_sha256=sha(R/'docs/protocol.json'),
        dataset='OriginalGooglePowerData2019 only',canonical_metrics_unchanged=True,
        historical_test_informed_design=True,independent_confirmation=False,new_Word=False))
    retrospective();save(Q/'complete.json',dict(completed_utc=utc(),all_passed=True,independent_verification_pending=True,selection_sha256=sha(Q/'selection.json')))
    print(json.dumps(dict(scores={h:{role:dict(curve=x['metrics']['wape'],daily=x['metrics']['daily_wape'],peak=x['metrics']['peak_mape']) for role,x in vals.items()} for h,vals in summary.items()}),ensure_ascii=True),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--phase',choices=['register','freeze','score'],required=True);a=ap.parse_args()
    initialize(a.root,register=a.phase=='register')
    if a.phase=='register':print('REGISTERED beforenewDEV')
    elif a.phase=='freeze':freeze()
    else:score()
