"""Round45: source-preserving HDSF audit, causal local refits and overlapping fusion.

No new data or foundation inference. TEST arrays are opened only after selection.
"""
from pathlib import Path
from dataclasses import replace
import argparse, csv, datetime, gc, hashlib, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import joblib
import numpy as np

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(p)
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def arrsha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
def export(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

LOCAL_CONFIGS = {
    'dropD1': {'kind':'local24', 'seeds':[101,202,303,404], 'params':{}},
    'regularized': {'kind':'local24', 'seeds':[101,202,303,404],
                    'params':{'num_leaves':7,'n_estimators':250,'min_child_samples':100,'reg_lambda':5.0,'max_depth':4}},
    'direct7': {'kind':'direct', 'seeds':[101,202],
                'params':{'num_leaves':15,'n_estimators':250,'min_child_samples':100,'reg_lambda':5.0,'max_depth':5}},
}
NAMES = {
    'original':'HDSF半日单纯形融合（原项目D14/D22固定训练）',
    'dropD1':'LightGBM四种子残差与滞后回归各50%（剔除首日）',
    'regularized':'LightGBM七叶强正则四种子残差与滞后回归各50%',
    'direct7':'LightGBM两种子共享时距头直接1—7天残差预测',
}
LAYOUTS = [{'hours':h,'overlap':ov,'shrink':s} for h in [3,6,12,24] for ov in [False,True] for s in [0.,.5]]

def initialize(root):
    global T,B,R,Q,A,repo,data,d,source_protocol,source_selection,canonical_index,natural_index
    T=Path(root); B=T/'experiment'; A=B/'round44'; R=B/'round45'; Q=R/'results'; repo=A/'reference/PowerData_Prediction'
    sys.path[:0]=[str(repo),str(B/'code'),str(B/'round29/code'),str(B/'round28/code')]
    global common, evaluate, interval_metrics, feature_cube, production_input, fit_lags, Local24, GBM_PARAMS
    import common as cm
    from aligned_metrics import evaluate as ev
    from source_metrics import interval_metrics as im
    from power_forecast.attainment_local import feature_cube as fc, production_input as pi, fit_lags as fl, Local24 as lc, GBM_PARAMS as gp
    common,evaluate,interval_metrics,feature_cube,production_input,fit_lags,Local24,GBM_PARAMS=cm,ev,im,fc,pi,fl,lc,gp
    source_protocol=read(A/'docs/protocol.json');source_selection=read(A/'results/selection.json')
    data=joblib.load(A/'results/source_dataset.joblib'); d=common.load()
    canonical_index=np.array([data.series_keys.index(str(n)) for n in d['names']])
    natural_index=np.array([list(map(str,d['names'])).index(n) for n in data.series_keys])
    assert np.array_equal(data.target_valid[canonical_index],d['valid'])
    assert np.max(abs(data.target[canonical_index]-d['y'])[d['valid']])<1e-12
    assert sha(B/'round29/code/aligned_metrics.py')==source_protocol['evaluation_sha256']
    assert sha(B/'code/common.py')==source_protocol['common_sha256']
    assert sha(A/'results/selection.json')==(A/'results/selection.sha256').read_text().strip()
    for f,h in source_protocol['source_sha256'].items():assert sha(repo/f)==h,f
    for sub in ['code','docs','models','results']: (R/sub).mkdir(parents=True,exist_ok=True)

def protocol():
    path=R/'docs/protocol.json'
    identity={'script':sha(__file__),'data':sha(A/'results/source_dataset.joblib'),
              'members':sha(A/'results/predictions.npz'),'original_selection':sha(A/'results/selection.json'),
              'metrics':sha(B/'round29/code/aligned_metrics.py'),'common':sha(B/'code/common.py'),
              'original_source':source_protocol['source_sha256']}
    if path.exists():
        p=read(path);assert p['identity']==identity,'Protocol/script/data identity changed; refuse incompatible resume';return p
    p={'round':45,'created_utc':utc(),'identity':identity,'dataset':'Google PowerData2019 only,57series,31days,96slots/day',
       'excluded_data':['NLR HPC PUE','Google ClusterData CPU/memory traces'],
       'foundation_members_frozen':True,'new_weight_downloads':False,'local_configs':LOCAL_CONFIGS,
       'dev_cutoff':14,'final_cutoff':22,'dev24':[14,16,18,20],'dev7':[14,15],
       'test24':list(range(22,31)),'test7':[22,23,24],
       'bottom_stage':'Three actual local refits. Select family using chronological-date fusion validation under original12h geometry; no TEST scores.',
       'fusion_stage':'16 width/overlap/shrink configurations for the DEV-selected member bank; original foundation pool retained. Full-member simplex, macro-normalized weighted absolute loss, no intercept.',
       'layouts':LAYOUTS,'optimizer_iterations':15,'optimizer_max_points':12000,
       'fusion_objective':'Each PDU total absolute truth denominator uses FIT labels only; duplicate calendar-date forecasts downweighted. Window taper multiplies FIT loss. Final headline is unchanged9.28 macro WAPE.',
       'shrink':'0 or50% convex shrink of window weights toward full-horizon weights; overlap uses strictly positive cosine taper normalized to partition-of-unity.',
       'cv24':'D15 fit→D17 validate; D15/17→D19; D15/17/19→D21; original member training stopsD14.',
       'cv7':'FIT targetdates≤D18→D19;≤D19→D20;≤D20→D21;≤D21→D22, for both DEV origins. Targetdate purged, not leave-one-overlapping-origin.',
       'unsupported_window':'Use global simplex if earlier FIT dates contain no samples of a lead window; exposes scarcity of late-horizon validation support.',
       'bottom_comparison7':'Direct7 is an added local expert alongside original local7; corrected one-day models are recursive7day diagnostics, never read future measured data.',
       'selection':'Choose best chronological CV WAPE, then lower dimensional layout within.005pp. For adoption require mean paired gain>max(.005pp,paired one-standard-error), positive majority folds, fullDEV curve anddaily WAPE nonworse, peak increase≤.10pp. Otherwise retain originalHDSF. Best DEV candidate remains frozen diagnostic even if gatefails.',
       'test_policy':'Freeze selection and SHA before any new final fit/test. Score selected candidate, original baseline and predeclared three12h bottom controls only; no TEST reranking.',
       'intervals':'Same perPDUDEV residual10/90 empirical80% and prior clipped0/1 interval metrics, no TEST calibration.',
       'causal_checks':'Training arrays and predictions unchanged after future-poison; uncached full-round first-seed training repeat of DEV-selected family and positive-control training-label change.',
       'historical_test_informed_design':True,'independent_confirmation':False,
       'device':'LightGBM CPU4threads; no large foundation model loaded. GPU foundation refit not claimed.',
       'no_new_word_report':True}
    save(path,p);return p

def training_arrays(dat,cut,cfg):
    assert cut in [14,22]
    if cfg['kind']=='local24':
        target=dat.target[:,:cut].transpose(1,2,0)
        prod=production_input(dat)[:,:cut].transpose(1,2,0)
        valid=dat.target_valid[:,:cut].transpose(1,2,0)
        F,names=feature_cube(target,prod,valid,dat.meta)
        residual=np.full_like(target,np.nan);residual[1:]=target[1:]-target[:-1]
        X=F.reshape(-1,F.shape[-1]);y=residual.reshape(-1);ok=valid.reshape(-1)&np.isfinite(y)&np.isfinite(X).any(1)
        dates=np.repeat(np.arange(1,cut+1),96*57)
        assert not ok[:96*57].any()
        info={'cutoff':cut,'label_min_day':int(dates[ok].min()),'label_max_day':int(dates[ok].max()),'D1_rows':0,
              'names':names,'kind':'local24','sample_count':int(ok.sum())}
        return np.ascontiguousarray(X[ok]),y[ok],np.ones(int(ok.sum())),info
    xs=[];ys=[];masks=[];dates=[];leads=[];series=[]
    for origin in range(7,cut):
        f,names=next_features(dat,origin)
        for ahead in range(1,min(7,cut-origin)+1):
            X=extend_features(f,origin,ahead,names)
            label=dat.target[:,origin+ahead-1].T
            ok=dat.target_valid[:,origin+ahead-1].T.reshape(-1)&np.isfinite(label.reshape(-1))
            xs.append(X[ok]);ys.append((label-dat.target[:,origin-1].T).reshape(-1)[ok])
            masks.append(label.reshape(-1)[ok]);dates.extend([origin+ahead]*int(ok.sum()));leads.extend([ahead]*int(ok.sum()))
            series.extend(np.tile(np.arange(57),96)[ok].tolist())
    X=np.concatenate(xs);y=np.concatenate(ys);truth=np.concatenate(masks);lead=np.asarray(leads);ser=np.asarray(series)
    # Equalize head/series counts in this pooled residual training task, not TEST.
    group=lead*57+ser;cnt=np.bincount(group,minlength=8*57);weight=1/np.maximum(cnt[group],1);weight/=weight.mean()
    info={'kind':'direct','cutoff':cut,'sample_count':len(y),'label_min_day':int(min(dates)),
          'label_max_day':int(max(dates)),'origin_min':7,'origin_max':cut-1,'D1_rows':0,
          'heads':[1,2,3,4,5,6,7],'names':names+['ahead_days','ahead_fraction','last_day_mean','mean7_day_level']}
    return X,y,weight,info

def next_features(dat,origin):
    # Prefix plus entirely unobserved next-day placeholder; no future target/prod.
    target=np.concatenate([dat.target[:,:origin],np.full((57,1,96),np.nan)],axis=1).transpose(1,2,0)
    prod=np.concatenate([production_input(dat)[:,:origin],np.full((57,1,96),np.nan)],axis=1).transpose(1,2,0)
    valid=np.concatenate([dat.target_valid[:,:origin],np.zeros((57,1,96),bool)],axis=1).transpose(1,2,0)
    f,names=feature_cube(target,prod,valid,dat.meta)
    return f[-1].reshape(-1,f.shape[-1]),names

def extend_features(f,origin,ahead,names):
    f=f.copy();phase=(origin+ahead-1+3)%7
    for n,v in [('dow_sin',np.sin(2*np.pi*phase/7)),('dow_cos',np.cos(2*np.pi*phase/7)),('is_weekend',float(phase in [5,6]))]:
        f[:,names.index(n)]=v
    last=np.asarray(f[:,names.index('lag1')]).reshape(96,57).mean(0)
    lvl=np.asarray(f[:,names.index('med7')]).reshape(96,57).mean(0)
    return np.column_stack([f,np.full(len(f),ahead),np.full(len(f),ahead/7),np.tile(last,96),np.tile(lvl,96)]).astype('float32')

def fit_local(cfgname,cut,dat=None,seeds=None):
    dat=data if dat is None else dat;cfg=LOCAL_CONFIGS[cfgname]
    X,y,w,info=training_arrays(dat,cut,cfg)
    import lightgbm as lgb
    params={**GBM_PARAMS,'n_jobs':4,**cfg['params']};rounds=params.pop('n_estimators');models=[];train=[]
    for seed in (cfg['seeds'] if seeds is None else seeds):
        print(f'fit {cfgname} cut{cut} seed{seed} samples{len(y)} rounds{rounds}',flush=True)
        ds=lgb.Dataset(X,label=y,weight=w,free_raw_data=True)
        model=lgb.train({**params,'seed':seed},ds,num_boost_round=rounds);models.append(model)
        train.append(float(np.average(abs(model.predict(X)-y),weights=w)));del ds;gc.collect()
    info.update(array_digest=arrsha(X,y,w),training_residual_mae_per_seed=train,params=params,rounds=rounds,seeds=cfg['seeds'] if seeds is None else seeds)
    if cfg['kind']=='local24':
        target=dat.target[:,:cut].transpose(1,2,0);valid=dat.target_valid[:,:cut].transpose(1,2,0)
        lag,coef=fit_lags(target,valid,cut);obj=Local24(cut,models,lag,coef,info['names'])
    else:obj={'models':models,'cutoff':cut,'names':info['names'][:-4]}
    del X,y,w;gc.collect();return obj,info

def signature(model):
    models=model.models if isinstance(model,Local24) else model['models']
    return [hashlib.sha256(m.model_to_string().encode()).hexdigest() for m in models]

def local_predict(model,dat,origin,H):
    if isinstance(model,Local24):
        # Multi-day diagnostic recursively uses own predicted power only; production future unknown.
        work=replace(dat,target=dat.target.copy(),production=dat.production.copy(),target_valid=dat.target_valid.copy(),
                     production_causal_median=production_input(dat).copy())
        out=[]
        for ahead in range(H):
            p=model.predict(work,origin+ahead).T;out.append(p)
            work.target[:,origin+ahead]=p;work.target_valid[:,origin+ahead]=True
            work.production[:,origin+ahead]=work.production[:,origin-1]
            work.production_causal_median[:,origin+ahead]=work.production_causal_median[:,origin-1]
        return np.stack(out,axis=1).reshape(57,H*96)
    f,names=next_features(dat,origin);assert names==model['names'];out=[]
    for ahead in range(1,H+1):
        X=extend_features(f,origin,ahead,names);res=np.mean([m.predict(X) for m in model['models']],axis=0).reshape(96,57).T
        out.append(np.clip(dat.target[:,origin-1]+res,0,1))
    return np.stack(out,axis=1).reshape(57,H*96)

def poison_future(dat,cut):
    return replace(dat,target=np.concatenate([dat.target[:,:cut],np.full_like(dat.target[:,cut:],9)],axis=1),
                   production=np.concatenate([dat.production[:,:cut],np.full_like(dat.production[:,cut:],9)],axis=1),
                   production_causal_median=np.concatenate([production_input(dat)[:,:cut],np.full_like(production_input(dat)[:,cut:],9)],axis=1),
                   target_valid=np.concatenate([dat.target_valid[:,:cut],np.zeros_like(dat.target_valid[:,cut:])],axis=1))

def members(H,phase,localname=None,localprediction=None):
    task='24h' if H==1 else '7d';methods=source_selection['configs'][task]['methods'].copy()
    with np.load(A/'results/predictions.npz') as z:
        pp=[z[('dev_' if phase=='dev' else '')+m+f'_H{H}'][natural_index] for m in methods]
    if localname:
        if H==1:methods[-1]=localname;pp[-1]=localprediction
        else:methods.append(localname);pp.append(localprediction)
    return np.stack(pp,axis=-1).transpose(1,2,0,3),methods

def truth(H,origins):
    y=np.stack([data.target[:,o:o+H].reshape(57,H*96) for o in origins],axis=1).transpose(1,2,0)
    m=np.stack([data.target_valid[:,o:o+H].reshape(57,H*96) for o in origins],axis=1).transpose(1,2,0)
    dates=np.array([o+1+np.arange(H*96)//96 for o in origins])
    return y,m,dates

def wape(y,p,m):
    num=(abs(p-y)*m).sum((0,1));den=(abs(y)*m).sum((0,1));assert (den>0).all()
    return float(np.mean(100*num/den))

def diagnostics(y,p,m):
    e=abs(p-y);curve=wape(y,p,m);O,L,S=y.shape
    yd=y.reshape(O,L//96,96,S);pd=p.reshape(O,L//96,96,S);md=m.reshape(O,L//96,96,S)
    ok=md.all(2);ya=yd.mean(2);pa=pd.mean(2);den=(abs(ya)*ok).sum((0,1))
    daily=float(np.mean(100*(abs(ya-pa)*ok).sum((0,1))/den))
    validday=md.any(2);ym=np.where(md,yd,-np.inf).max(2);pm=np.where(md,pd,-np.inf).max(2)
    return {'wape':curve,'daily_wape':daily,'peak_mape':float(100*np.mean((abs(ym-pm)/np.maximum(ym,1e-8))[validday])),
            'mae':float(e[m].mean()),'rmse':float(np.sqrt((e[m]**2).mean()))}

def weighted_simplex(P,y,w,iterations=15,max_points=12000):
    keep=np.isfinite(P).all(1)&np.isfinite(y)&np.isfinite(w)&(w>0);inds=np.flatnonzero(keep)
    assert len(inds)>0
    if len(inds)>max_points:
        # Deterministic bounded subsampling, never informed by TEST.
        inds=np.random.default_rng(45).choice(inds,max_points,replace=False)
    P=np.ascontiguousarray(P[inds]);y=y[inds];w=w[inds];M=P.shape[1];coef=np.full(M,1/M);current=P@coef
    for _ in range(iterations):
        moved=0.
        for i in range(M):
            for j in range(M):
                if i==j or coef[j]<=1e-12:continue
                direction=P[:,i]-P[:,j];good=direction!=0
                roots=-(y[good]-current[good])/direction[good];mass=abs(direction[good])*w[good]
                if not len(roots):continue
                order=np.argsort(roots);acc=np.cumsum(mass[order]);mid=min(np.searchsorted(acc,acc[-1]/2),len(order)-1)
                step=float(np.clip(roots[order[mid]],-coef[j],coef[i]))
                if abs(step)<=1e-12:continue
                coef[i]-=step;coef[j]+=step;current-=step*direction;moved+=abs(step)
        if moved<1e-9:break
    coef=np.maximum(coef,0);return coef/coef.sum()

def windows(L,layout):
    width=layout['hours']*4;stride=width//2 if layout['overlap'] else width
    if layout['overlap']:
        starts=list(range(-stride,L,stride));x=np.arange(width)
        taper=.5-.5*np.cos(2*np.pi*(x+.5)/width)
    else:starts=list(range(0,L,stride));taper=np.ones(width)
    basis=[];positions=np.arange(L)
    for start in starts:
        distance=positions-start;valid=(distance>=0)&(distance<width)
        b=np.zeros(L);b[valid]=taper[distance[valid]];basis.append(b)
    basis=np.array(basis);assert (basis.sum(0)>0).all()
    basis/=basis.sum(0);assert np.allclose(basis.sum(0),1)
    return basis,starts

def fitting_weights(y,mask,dates):
    den=(abs(y)*mask).sum((0,1));assert (den>0).all()
    unique,counts=np.unique(dates,return_counts=True);duplicate=counts[np.searchsorted(unique,dates)]
    # Every observed date/slot appears once or twice in DEV7; identical truth is not independent evidence.
    return mask/den[None,None,:]/duplicate[:,:,None]

def fit_geometry(P,y,mask,dates,layout,source_style=False):
    basis,starts=windows(y.shape[1],layout);weights=fitting_weights(y,mask,dates)
    def group_fit(V,ww):
        return weighted_simplex(V.reshape(-1,V.shape[-1]),y.reshape(-1),ww.reshape(-1),
                                iterations=25 if source_style else 15,max_points=20000 if source_style else 12000)
    if source_style:
        # Baseline validation control uses the source pooled objective and fixed original mix.
        weights=mask.astype(float)
    global_weight=group_fit(P,weights);local=[];unsupported=0
    for b in basis:
        ww=weights*b[None,:,None]
        if not (ww>0).any():coef=global_weight.copy();unsupported+=1
        else:coef=group_fit(P,ww)
        local.append((1-layout['shrink'])*coef+layout['shrink']*global_weight)
    return {'weights':np.array(local).tolist(),'global':global_weight.tolist(),'basis':basis.tolist(),
            'starts':starts,'unsupported_fit_windows':unsupported,'layout':layout,'source_style':source_style}

def apply_geometry(P,fit):
    coef=np.asarray(fit['basis']).T@np.asarray(fit['weights'])
    assert (coef>=-1e-12).all() and np.allclose(coef.sum(-1),1)
    return np.einsum('olsm,lm->ols',P,coef,optimize=True)

def original_style_fit(P,y,mask,dates,H):
    task='24h' if H==1 else '7d';mix=source_selection['configs'][task]['mix'];layout={'hours':12,'overlap':False,'shrink':0.}
    a=fit_geometry(P[...,:-1],y,mask,dates,layout,True);b=fit_geometry(P,y,mask,dates,layout,True)
    return {'a':a,'b':b,'mix':mix}
def original_style_apply(P,fit):
    return (1-fit['mix'])*apply_geometry(P[...,:-1],fit['a'])+fit['mix']*apply_geometry(P,fit['b'])

def folds(H,mask,dates):
    if H==1:
        return [(f'validateD{int(dates[i,0])}',mask&(np.arange(len(mask))[:,None,None]<i),
                 mask&(np.arange(len(mask))[:,None,None]==i)) for i in range(1,len(mask))]
    return [(f'validateD{day}',mask&(dates[:,:,None]<day),mask&(dates[:,:,None]==day)) for day in [19,20,21,22]]

def crossvalidate(P,y,mask,dates,H,layout=None,source_style=False):
    out=np.zeros_like(y);held=np.zeros_like(mask);scores=[];details=[]
    for name,fitmask,valmask in folds(H,mask,dates):
        assert not set(dates[fitmask.any(2)]).intersection(set(dates[valmask.any(2)]))
        assert dates[fitmask.any(2)].max()<dates[valmask.any(2)].min()
        if source_style:
            fit=original_style_fit(P,y,fitmask,dates,H);pred=original_style_apply(P,fit)
            unsupported=fit['a']['unsupported_fit_windows']+fit['b']['unsupported_fit_windows']
        else:
            fit=fit_geometry(P,y,fitmask,dates,layout);pred=apply_geometry(P,fit);unsupported=fit['unsupported_fit_windows']
        out[valmask]=pred[valmask];held|=valmask;scores.append(wape(y,pred,valmask))
        details.append({'fold':name,'fit_latest_day':int(dates[fitmask.any(2)].max()),'validation_day':int(dates[valmask.any(2)].min()),
                        'score':scores[-1],'unsupported_windows':unsupported})
    return {'prediction':out,'mask':held,'score':wape(y,out,held),'fold_scores':scores,'folds':details}

def audit(p):
    if (Q/'audit.json').exists():return
    findings=[];d1=[]
    for cut in [14,22]:
        target=data.target[:,:cut].transpose(1,2,0);prod=production_input(data)[:,:cut].transpose(1,2,0);v=data.target_valid[:,:cut].transpose(1,2,0)
        X,names=feature_cube(target,prod,v,data.meta)
        res=target-np.concatenate([np.full_like(target[:1],np.nan),target[:-1]])
        res[0]=target[0]-target.mean((0,1))[None,:]
        keep=np.isfinite(res)&v&np.isfinite(X).any(-1)
        delta=.123*(cut-1)/cut
        d1.append({'cutoff':cut,'original_D1_rows':int(keep[0].sum()),'original_rows':int(keep.sum()),
                   'D1_share_percent':float(100*keep[0].sum()/keep.sum()),'D1_label_change_if_later_train_days_plus_point123':delta,
                   'new_models_D1_rows':0,'cross_TEST_leakage_not_implied':True})
    s=read(A/'results/summary.json')
    findings=[{'issue':'D1 residual uses training-prefix mean while later days use previousday','status':'confirmed; corrected in new refits'},
              {'issue':'DEV7 windows overlapD16—D21','status':'confirmed; only8unique targetdates vs14 forecastdays; newfusionCV purges by targetdate'},
              {'issue':'Original fusion weights fit DEV labels; fitted DEV score is optimistic','status':'confirmed; forward-date scores provided'},
              {'issue':'All31days sameMay2019 TEST repeatedly viewed; historicaldesign informed byTEST','status':'confirmed; no independent confirmation possible in this run'},
              {'issue':'Future data leakage in original local fit/predict','status':'existing full-refit poison audit passed; source checksums and archival audit verified thisrun'},
              {'issue':'Point masks, daily full96 mask, macro57 vs pooled evaluation','status':'canonical evaluators and truth/masks verified unchanged'},
              {'issue':'Production utilization is not CPU workload; fixed calendar phase is not independently established actual weekday','status':'retain disclosed meanings; no physicalkW or energyconservation claim'},
              {'issue':'Foundation pretraining provenance could include similar series','status':'not certified; no claim all upstream training contamination excluded'}]
    archive=read(A/'results/science_verification.json');assert archive['all_passed']
    save(Q/'audit.json',{'created_utc':utc(),'findings':findings,'D1':d1,'source_archive_checks':archive['check_count'],
                        'source_archive_sha256':sha(A/'results/science_verification.json'),'source_unchanged':True,
                        'original_development':{h:s['development'][h]['hdsf'] for h in ['1','7']},
                        'overfitting_absence_proven':False,'independent_confirmation':False,'protocol_sha256':sha(R/'docs/protocol.json')})

def dev(p):
    if (Q/'selection.json').exists():assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip();return
    local_predictions={};training=[]
    for name,cfg in LOCAL_CONFIGS.items():
        modelpath=R/'models'/f'{name}_cut14.joblib';infopath=Q/f'{name}_cut14.json'
        if modelpath.exists():model=joblib.load(modelpath);info=read(infopath)
        else:
            model,info=fit_local(name,14);joblib.dump(model,modelpath);info['model_sha256']=sha(modelpath);save(infopath,info)
        assert sha(modelpath)==info['model_sha256']
        X,y,w,meta=training_arrays(data,14,cfg);Xp,yp,wp,mp=training_arrays(poison_future(data,14),14,cfg)
        assert arrsha(X,y,w)==arrsha(Xp,yp,wp)==info['array_digest']
        changed=replace(data,target=data.target.copy());changed.target[:,13]+=.08
        Xa,ya,wa,_=training_arrays(changed,14,cfg);assert arrsha(Xa,ya,wa)!=info['array_digest']
        del X,y,w,Xp,yp,wp,Xa,ya,wa;gc.collect()
        arrays={}
        for H,oo in [(1,p['dev24']),(7,p['dev7'])]:
            arr=np.stack([local_predict(model,data,o,H) for o in oo],axis=1)
            for j,o in enumerate(oo):assert np.array_equal(arr[:,j],local_predict(model,poison_future(data,o),o,H))
            arrays[f'H{H}']=arr;local_predictions[(name,H)]=arr
        np.savez_compressed(Q/f'{name}_development.npz',**arrays);training.append(info);del model;gc.collect()
    records=[];selected={};allfits={};CVbanks={};cvreferences={}
    for H,oo in [(1,p['dev24']),(7,p['dev7'])]:
        y,m,dates=truth(H,oo);original,methods=members(H,'dev');ref=crossvalidate(original,y,m,dates,H,source_style=True);cvreferences[str(H)]={k:v for k,v in ref.items() if k not in ['prediction','mask']}
        bottom=[]
        for name in LOCAL_CONFIGS:
            P,mm=members(H,'dev',name,local_predictions[(name,H)])
            cv=crossvalidate(P,y,m,dates,H,{'hours':12,'overlap':False,'shrink':.5})
            bottom.append({'member':name,'score':cv['score'],'fold_scores':cv['fold_scores']});CVbanks[(name,H)]=(P,mm)
            records.append({'stage':'bottom','horizon_days':H,'member':name,'model_name':NAMES[name],**{k:v for k,v in cv.items() if k not in ['prediction','mask']}})
        bestbottom=min(bottom,key=lambda c:c['score']);bankname=bestbottom['member'];P,mm=CVbanks[(bankname,H)]
        # Include original member bank under the new objective to isolate geometry/objective effects.
        bank_choices=[(bankname,P,mm),('original',original,methods)]
        candidates=[]
        for name,bank,mm in bank_choices:
            for layout in LAYOUTS:
                key=f"{name}_{layout['hours']}h_{'overlap' if layout['overlap'] else 'hard'}_shrink{layout['shrink']}"
                print(f'DEV H{H} {key}',flush=True)
                cv=crossvalidate(bank,y,m,dates,H,layout)
                fit=fit_geometry(bank,y,m,dates,layout);full=apply_geometry(bank,fit);fullmetrics=diagnostics(y,full,m)
                gains=np.asarray(ref['fold_scores'])-np.asarray(cv['fold_scores']);se=float(np.std(gains,ddof=1)/np.sqrt(len(gains)))
                baselinefull=read(A/'results/summary.json')['development'][str(H)]['hdsf']
                gate=(float(gains.mean())>max(.005,se) and int((gains>0).sum())>len(gains)/2
                      and fullmetrics['wape']<=baselinefull['wape']+1e-10
                      and fullmetrics['daily_wape']<=baselinefull['daily_wape']+1e-10
                      and fullmetrics['peak_mape']<=baselinefull['peak_mape']+.10)
                row={'stage':'geometry','horizon_days':H,'key':key,'member':name,'layout':layout,'score':cv['score'],
                     'fold_scores':cv['fold_scores'],'folds':cv['folds'],'paired_gain_mean_pp':float(gains.mean()),
                     'paired_gain_se_pp':se,'positive_folds':int((gains>0).sum()),'development_full':fullmetrics,'adoption_gate':bool(gate)}
                records.append(row);candidates.append(row);allfits[(H,key)]={'fit':fit,'methods':mm,'member':name,'key':key}
        best=min(c['score'] for c in candidates);plateau=[c for c in candidates if c['score']<=best+.005]
        # Within plateau prefer fewer windows, overlap smoothness, and stronger shrink.
        winner=min(plateau,key=lambda c:(len(allfits[(H,c['key'])]['fit']['weights']),-c['layout']['shrink'],c['score']))
        selected[str(H)]={'candidate':winner,'config':allfits[(H,winner['key'])],'adopt':winner['adoption_gate'],
                          'reference_cv':cvreferences[str(H)],'bottom_candidates':bottom,'bottom_selected':bankname,
                          'adopted_name':winner['key'] if winner['adoption_gate'] else 'original_HDSF'}
        save(Q/f'development_H{H}.json',{'records':[r for r in records if r['horizon_days']==H],'selected':selected[str(H)]})
    # Full uncached first-seed positive and negative controls of every selected trained family.
    checks=[]
    for name in sorted({c['config']['member'] for c in selected.values()}-{'original'}):
        original=joblib.load(R/'models'/f'{name}_cut14.joblib')
        duplicate,info=fit_local(name,14,poison_future(data,14),seeds=[LOCAL_CONFIGS[name]['seeds'][0]])
        assert signature(original)[0]==signature(duplicate)[0]
        changed=replace(data,target=data.target.copy());changed.target[:,13]+=.08
        positive,meta=fit_local(name,14,changed,seeds=[LOCAL_CONFIGS[name]['seeds'][0]])
        assert signature(original)[0]!=signature(positive)[0]
        checks.append({'member':name,'future_poison_uncached_full_rounds_first_seed':True,'positive_training_label_control':True})
        del original,duplicate,positive;gc.collect()
    selection={'created_utc':utc(),'protocol_sha256':sha(R/'docs/protocol.json'),'selected':selected,
               'training_checks':checks,'test_not_scored_in_this_phase':True,'dev_diagnostic_count':len(records),
               'historical_TEST_exposed':True,'independent_confirmation':False}
    save(Q/'development_summary.json',{'records':records,'reference_cv':cvreferences,'training':training})
    export(Q/'开发期全部试验.csv',records)
    save(Q/'selection.json',selection);(Q/'selection.sha256').write_text(sha(Q/'selection.json'),encoding='ascii')
    print('SELECTION FROZEN',json.dumps({h:{'key':c['config']['key'],'adopt':c['adopt']} for h,c in selected.items()}),flush=True)

def final(p):
    assert (Q/'selection.json').exists();assert sha(Q/'selection.json')==(Q/'selection.sha256').read_text().strip()
    s=read(Q/'selection.json');assert s['protocol_sha256']==sha(R/'docs/protocol.json')
    if (Q/'complete.json').exists():return
    models={};training=[];new_predictions={}
    # The three declared controls are scored after freeze, not used to rerank selection.
    for name in LOCAL_CONFIGS:
        path=R/'models'/f'{name}_cut22.joblib';info_path=Q/f'{name}_cut22.json'
        if path.exists():model=joblib.load(path);info=read(info_path);assert sha(path)==info['model_sha256']
        else:model,info=fit_local(name,22);joblib.dump(model,path);info['model_sha256']=sha(path);save(info_path,info)
        for H,oo in [(1,p['test24']),(7,p['test7'])]:
            arr=np.stack([local_predict(model,data,o,H) for o in oo],axis=1)
            for j,o in enumerate(oo):assert np.array_equal(arr[:,j],local_predict(model,poison_future(data,o),o,H))
            new_predictions[(name,H)]=arr
        X,y,w,_=training_arrays(data,22,LOCAL_CONFIGS[name]);Xp,yp,wp,_=training_arrays(poison_future(data,22),22,LOCAL_CONFIGS[name])
        assert arrsha(X,y,w)==arrsha(Xp,yp,wp)==info['array_digest'];del X,y,w,Xp,yp,wp
        training.append(info);del model;gc.collect()
    results=[];arrays={};summary={};reference=read(A/'results/summary.json')
    for H,oo in [(1,p['test24']),(7,p['test7'])]:
        chosen=s['selected'][str(H)];name=chosen['config']['member'];original,methods=members(H,'test')
        P,mm=(original,methods) if name=='original' else members(H,'test',name,new_predictions[(name,H)])
        assert mm==chosen['config']['methods'];pred=apply_geometry(P,chosen['config']['fit']).transpose(2,0,1)[canonical_index]
        with np.load(A/'results/predictions.npz') as z:prior=z[f'hdsf_H{H}'].copy()
        adopted=pred if chosen['adopt'] else prior
        ys,ms=common.truth(d,oo,H);do=p['dev24'] if H==1 else p['dev7'];yd,md=common.truth(d,do,H)
        devoriginal,dm=members(H,'dev');lp=None
        if name!='original':
            with np.load(Q/f'{name}_development.npz') as lz:lp=lz[f'H{H}']
        dpbank,dmm=(devoriginal,dm) if name=='original' else members(H,'dev',name,lp)
        dp=apply_geometry(dpbank,chosen['config']['fit']).transpose(2,0,1)[canonical_index]
        low=np.array([np.quantile((yd-dp)[i][md[i]],.1) for i in range(57)]);high=np.array([np.quantile((yd-dp)[i][md[i]],.9) for i in range(57)])
        lower=np.clip(pred+low[:,None,None],0,1);upper=np.clip(pred+high[:,None,None],0,1)
        iv=interval_metrics(ys,pred,ms,lower,upper)
        score=evaluate(common,d,oo,pred,H);prior_score=evaluate(common,d,oo,prior,H)
        assert prior_score==reference['scores'][str(H)]['hdsf']
        layout=chosen['candidate']['layout'];modelname=f"HDSF扩展：{layout['hours']}小时{'50%重叠' if layout['overlap'] else '不重叠'}，全局收缩{layout['shrink']*100:g}%，"+NAMES[name]
        results.append({'horizon_days':H,'role':'original','model_name':NAMES['original'],**prior_score})
        results.append({'horizon_days':H,'role':'DEV_selected_candidate','model_name':modelname,'adopted_from_DEV':chosen['adopt'],**score,**iv})
        arrays[f'candidate_H{H}']=pred;arrays[f'original_H{H}']=prior;arrays[f'adopted_H{H}']=adopted;arrays[f'dev_candidate_H{H}']=dp
        arrays[f'lower80_candidate_H{H}']=lower;arrays[f'upper80_candidate_H{H}']=upper
        for localname in LOCAL_CONFIGS:
            # Development-only fit of fixed12h geometry with same full-member objective and shrink50%.
            with np.load(Q/f'{localname}_development.npz') as lz:devlocal=lz[f'H{H}']
            db,dm=members(H,'dev',localname,devlocal);dy,mask,dates=truth(H,do)
            fit=fit_geometry(db,dy,mask,dates,{'hours':12,'overlap':False,'shrink':.5})
            tb,tm=members(H,'test',localname,new_predictions[(localname,H)])
            control=apply_geometry(tb,fit).transpose(2,0,1)[canonical_index]
            arrays[f'control_{localname}_H{H}']=control
            results.append({'horizon_days':H,'role':'preregistered_bottom_control_not_selected_by_TEST','model_name':f'HDSF扩展12小时，收缩50%＋{NAMES[localname]}',**evaluate(common,d,oo,control,H)})
        rng=np.random.default_rng(20261012);draw=rng.integers(0,len(oo),(3000,len(oo)))
        a=(abs(ys-prior)*ms).sum(-1);b=(abs(ys-pred)*ms).sum(-1);den=(abs(ys)*ms).sum(-1)
        gains=100*(a[:,draw].sum(-1)/den[:,draw].sum(-1)-b[:,draw].sum(-1)/den[:,draw].sum(-1)).mean(0)
        summary[str(H)]={'candidate_name':modelname,'selected_key':chosen['config']['key'],'candidate_score':score,'original_score':prior_score,
                         'adopted_by_DEV_gate':chosen['adopt'],'adopted_score':evaluate(common,d,oo,adopted,H),'interval':iv,
                         'gain_pp':prior_score['wape']-score['wape'],'relative_curve_reduction_percent':100*(1-score['wape']/prior_score['wape']),
                         'paired_origin_bootstrap95_gain_pp':np.quantile(gains,[.025,.975]).tolist(),'overlapping_test_windows':H==7,
                         'independent_confirmation':False,'future_poison_prediction_all_origins_passed':True}
    np.savez_compressed(Q/'predictions.npz',**arrays)
    export(Q/'统一评价结果.csv',results)
    save(Q/'summary.json',{'created_utc':utc(),'scores':summary,'results':results,'training':training,'metric_version':source_protocol['headline_metrics'],
                          'selection_sha256':sha(Q/'selection.json'),'protocol_sha256':sha(R/'docs/protocol.json'),'data':'GooglePowerData2019 only',
                          'test_used_for_selection':False,'historical_test_informed_design':True,'independent_confirmation':False})
    save(Q/'complete.json',{'created_utc':utc(),'all_passed':True,'primary_dev_final_local_fit_models':sum(len(c['seeds'])*2 for c in LOCAL_CONFIGS.values()),
                           'extra_uncached_audit_fit_models':len(s['training_checks'])*2,
                           'predictions_sha256':sha(Q/'predictions.npz'),'summary_sha256':sha(Q/'summary.json'),
                           'selection_sha256':sha(Q/'selection.json'),'metric_code_unchanged':True,'report_created':False,
                           'global_deployment_entry_changed':False,'user_requested_baseline':'HDSF','needs_independent_month':True})
    print('ROUND45 COMPLETE',json.dumps(summary,ensure_ascii=True),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--phase',choices=['audit','dev','final','all'],default='all');args=ap.parse_args()
    initialize(args.root);p=protocol();save(Q/'run_status.json',{'pid':os.getpid(),'phase':args.phase,'state':'running','started_utc':utc()})
    try:
        audit(p)
        if args.phase in ['all','dev']:dev(p)
        if args.phase in ['all','final']:final(p)
        save(Q/'run_status.json',{'pid':os.getpid(),'phase':args.phase,'state':'complete','completed_utc':utc()})
    except BaseException as e:
        import traceback
        save(Q/'run_status.json',{'pid':os.getpid(),'phase':args.phase,'state':'failed','error':repr(e),'traceback':traceback.format_exc(),'utc':utc()})
        raise
