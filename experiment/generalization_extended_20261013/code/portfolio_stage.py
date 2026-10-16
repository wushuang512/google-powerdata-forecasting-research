"""Development-only method fitting and frozen choices for broad comparisons."""
from pathlib import Path
import argparse, hashlib, json, sys
sys.dont_write_bytecode=True
import numpy as np,joblib
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('case');ap.add_argument('--stage',choices=['freeze','score'],default='freeze');ap.add_argument('--phase',default='test');a=ap.parse_args()
ROOT=Path(a.root);F=ROOT/'experiment/generalization_extended_20261013';OLD=ROOT/'experiment/generalization_validation_20261012'
sys.path[:0]=[str(OLD/'code'),str(ROOT/'experiment/round44/reference/PowerData_Prediction')]
from core import read,save,sha,utc,truth,baseline,metric,interval,export,np_save
from power_forecast.attainment_fusion import fit_simplex
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
proto=read(F/'docs/protocol.json');cfg=proto['cases'][a.case];z=np.load(F/'data'/f'{a.case}.npz');ppd=int(z['ppd']);sc=z['scales'];y=z['y'];x=z['x'];m=z['valid'];names=z['names'];dates=z['dates']
families=proto['neural_families']+['ttm1024'];selection=F/'results'/f'{a.case}_selection.json'
if ppd==96:families+=['moirai_rate4','moirai_rate12']
def origins(phase,H):
    if a.case.startswith('NLR'):return cfg['origins'][phase]
    if phase=='oof':return [7] if H==7 else list(range(7,14))
    if phase=='dev' and a.case=='Google2019':return list(range(14,22)) if H==1 else[14,15]
    return cfg['origins'][phase+str(H)]
def cutoff(phase):return 7 if phase=='oof' else cfg['cutoff'] if phase=='dev' or a.case.startswith('NLR') else cfg['final_cutoff']
def neural(f,label,phase,H):
    c=cutoff(phase);parts=[]
    for o in origins(phase,H):
        p=F/'cache'/f'{a.case}_{f}_cut{c}_{label}_{phase}_H{H}_o{o}.npz'
        with np.load(p) as q:
            assert int(q['origin'])==o and int(q['horizon'])==H and str(q['protocol_sha256'])==sha(F/'docs/protocol.json')
            parts.append(q['p'].copy()*sc[:,None])
    return np.stack(parts,1)
def stat(phase,H):
    c=cutoff(phase);p=F/'cache'/f'{a.case}_stat_cut{c}_{phase}_H{H}.npz'
    with np.load(p) as q:
        assert list(q['origins'])==origins(phase,H)
        return {k:q[k].copy()*sc[:,None,None] for k in q.files if k not in ['origins','script_sha256','protocol_sha256']}
def early_late(H):
    oo=np.array(origins('dev',H))
    if a.case.startswith('NLR'):
        split=len(oo)//2;late=np.arange(split,len(oo));early=np.where(oo+H<=oo[split])[0]
        assert len(early)>=5 and len(late)>=5
        return early,late
    return np.arange(len(origins('oof',H))),np.arange(len(oo))
def score_one(p,phase,H):
    oo=origins(phase,H);yy,mm=truth(y,m,oo,H);return metric(yy,p,mm,ppd)
def clip(p):return np.clip(p,0,1) if a.case=='Google2019' else np.maximum(0,p)
def calibrator_features(p,oo,H):
    last=baseline(x,oo,H,'last');mean3=baseline(x,oo,H,'mean3');lead=np.broadcast_to((np.arange(H*ppd)//ppd)[None,None],p.shape)
    pm=p.reshape(len(p),len(oo),H,ppd).mean(-1).repeat(ppd,axis=-1).reshape(p.shape)
    scale=sc[:,None,None]
    return np.stack([p/scale,last/scale,mean3/scale,pm/scale,(p-pm)/scale,lead/7],-1)
def fit_weights(P,yy,mm,segment):
    weights=[];den=(abs(yy)*mm).sum((1,2))/mm.sum((1,2)).clip(1);den=np.maximum(den,1e-8)
    for start in range(0,yy.shape[-1],segment):
        sl=slice(start,min(start+segment,yy.shape[-1]));A=P[:,:,sl]/den[:,None,None,None];b=yy[:,:,sl]/den[:,None,None]
        weights.append(fit_simplex(A.reshape(-1,P.shape[-1]),b.reshape(-1),mm[:,:,sl].reshape(-1),iterations=25,max_points=20000))
    return np.array(weights)
def blend(P,c):
    W=np.array(c['weights']);segment=c['segment'];steps=np.arange(P.shape[2])
    if c['smooth'] and len(W)>1:
        centers=np.arange(len(W))*segment+(segment-1)/2;w=np.stack([np.interp(steps,centers,W[:,k]) for k in range(W.shape[1])],1)
    else:w=W[np.minimum(steps//segment,len(W)-1)]
    assert np.max(abs(w.sum(-1)-1))<1e-7 and w.min()>=-1e-8
    return clip(np.einsum('solk,lk->sol',P,w,optimize=True))

def fourier_features(p,oo,H):
    pd=p.reshape(len(p),len(oo),H,ppd)/sc[:,None,None,None]
    last=baseline(x,oo,1,'last')/sc[:,None,None];recent=baseline(x,oo,1,'mean3')/sc[:,None,None]
    return np.concatenate([pd.mean(-1),pd.std(-1),last.mean(-1)[:,:,None],recent.mean(-1)[:,:,None],last.std(-1)[:,:,None]],-1)

def make_methods(bank, oo,H, configurations=None):
    out=bank.copy()
    ch=bank['raw_chronos2'];ti=bank['raw_tirex2'];tf=bank['raw_timesfm3']
    out['Chronos2_TiRex2_equal']=clip(.5*ch+.5*ti)
    out['Chronos2_TiRex2_TimesFM3_25_25_50']=clip(.25*ch+.25*ti+.5*tf)
    if configurations:
        for name,c in configurations.items():
            if c['kind']=='simplex':out[name]=blend(np.stack([bank[k] for k in c['members']],-1),c)
            elif c['kind']=='residual':
                base=out[c['base']];X=calibrator_features(base,oo,H);model=joblib.load(F/'models'/c['file']);delta=model.predict(X.reshape(-1,X.shape[-1])).reshape(base.shape)*sc[:,None,None]
                out[name]=clip(base+c['alpha']*delta)
            elif c['kind']=='fourier':
                base=out[c['base']];X=fourier_features(base,oo,H);model=joblib.load(F/'models'/c['file']);B=np.array(c['basis']);coef=model.predict(X.reshape(-1,X.shape[-1])).reshape(len(base),len(oo),H,len(B[0]))
                delta=np.einsum('sodk,tk->sodt',coef,B).reshape(base.shape)*sc[:,None,None]
                out[name]=clip(base+c['alpha']*delta)
    return out

def load_bank(phase,H,chosen):
    bank=stat(phase,H)
    for f in families:
        bank['raw_'+f]=neural(f,'s0',phase,H)
        label=chosen[f]
        bank['target_'+f]=neural(f,label,phase,H) if a.case!='Sea2022' else bank['raw_'+f]
    for kind in ['last','week','mean3']:bank['baseline_'+kind]=baseline(x,origins(phase,H),H,kind)
    if a.case.startswith('NLR'):
        control=F/'cache'/f'{a.case}_source_assemble_{phase}.npz'
        with np.load(control) as q:
            assert list(q['origins'])==origins(phase,H)
            bank['HDSF_original_source']=q[f'HDSF_original_source_H{H}'].copy()
            bank['R48_fixed_source']=q[f'R48_fixed_source_H{H}'].copy()
    return bank

def freeze():
    if selection.exists():assert sha(selection)==selection.with_suffix('.sha256').read_text().strip();return
    assert (F/'results/development_raw_bank_complete.json').exists(),'Finish all registered development families first'
    result=dict(created_utc=utc(),protocol_sha256=sha(F/'docs/protocol.json'),case=a.case,tasks={},test_scores_used=False)
    for H in [1,7]:
        early,late=early_late(H);oo=origins('dev',H);yy,mm=truth(y,m,oo,H);chosen={};checkpoint_scores={}
        for f in families:
            options=['s0'] if a.case=='Sea2022' else['s0','s16','s48'];scores=[]
            for label in options:
                p=neural(f,label,'dev',H);ix=early if a.case.startswith('NLR') else late
                v=metric(yy[:,ix],p[:,ix],mm[:,ix],ppd);per=[]
                for k in ix:per.append(metric(yy[:,k:k+1],p[:,k:k+1],mm[:,k:k+1],ppd)['wape'])
                scores.append(dict(label=label,wape=v['wape'],per_origin_wape=per))
            best=min(scores,key=lambda r:r['wape']);accepted=[]
            for row in scores:
                difference=np.array(row['per_origin_wape'])-np.array(best['per_origin_wape']);se=float(np.std(difference,ddof=1)/np.sqrt(len(difference))) if len(difference)>1 else 0.0
                row['paired_fluctuation_tolerance_pp']=max(.005,se);row['within_tolerance']=row['wape']<=best['wape']+row['paired_fluctuation_tolerance_pp']
                if row['within_tolerance']:accepted.append(row)
            chosen[f]=min(accepted,key=lambda r:options.index(r['label']))['label'];checkpoint_scores[f]=scores
        dev=load_bank('dev',H,chosen);fitphase='dev' if a.case.startswith('NLR') else'oof'
        fitbank=dev if fitphase=='dev' else load_bank('oof',H,chosen);fit_oo=origins(fitphase,H);fy,fm=truth(y,m,fit_oo,H)
        ix=early if fitphase=='dev' else np.arange(len(fit_oo));configurations={}
        # Sea per-family residual fits are actual low-dimensional target fits.
        bases=['Chronos2_TiRex2_TimesFM3_25_25_50']+(['raw_'+f for f in families] if a.case=='Sea2022' else[])
        fitmethods=make_methods(fitbank,fit_oo,H);devmethods=make_methods(dev,oo,H)
        for base_name in bases:
            for family in ['ridge','hgb'] if a.case!='Sea2022' else['ridge']:
                base=fitmethods[base_name];X=calibrator_features(base,fit_oo,H);valid=fm[:,ix].reshape(-1);XX=X[:,ix].reshape(-1,X.shape[-1])[valid];Y=((fy-base)/sc[:,None,None])[:,ix].reshape(-1)[valid]
                model=make_pipeline(StandardScaler(),Ridge(alpha=1000)) if family=='ridge' else HistGradientBoostingRegressor(loss='absolute_error',max_leaf_nodes=7,max_iter=100,l2_regularization=10,random_state=20261013)
                model.fit(XX,Y);fname=f'{a.case}_{base_name}_{family}_H{H}.joblib';joblib.dump(model,F/'models'/fname)
                name=base_name+'_target_'+family;c=dict(kind='residual',file=fname,base=base_name,alpha=1)
                candidates=[]
                for alpha in [0,.25,1]:
                    c['alpha']=alpha;pp=make_methods(dev,oo,H,{name:c})[name];v=metric(yy[:,late],pp[:,late],mm[:,late],ppd);candidates.append(dict(alpha=alpha,wape=v['wape']))
                minimum=min(r['wape'] for r in candidates);c['alpha']=min(r['alpha'] for r in candidates if r['wape']<=minimum+.005);c['validation_candidates']=candidates;configurations[name]=c
        if ppd==96:
            # Refit the previously promising TimesFM3 Fourier residual method.
            base_name='raw_timesfm3';base=fitbank[base_name];X=fourier_features(base,fit_oo,H)
            t=np.arange(ppd)/ppd;B=np.stack([np.ones(ppd)]+[fn(2*np.pi*k*t) for k in [1,2,3] for fn in [np.sin,np.cos]],1)
            residual=(fy-base)/sc[:,None,None];targets=[];features=[]
            for j in range(len(y)):
                for k in ix:
                    mask=fm[j,k].reshape(H,ppd);rr=residual[j,k].reshape(H,ppd)
                    if not mask.all(-1).all():continue
                    coef=np.stack([np.linalg.lstsq(B,rr[d],rcond=None)[0] for d in range(H)])
                    targets.append(coef.reshape(-1));features.append(X[j,k])
            if len(targets)>=3:
                model=make_pipeline(StandardScaler(),Ridge(alpha=1000,fit_intercept=False));model.fit(np.array(features),np.array(targets));fname=f'{a.case}_TimesFM3_FourierResidual_H{H}.joblib';joblib.dump(model,F/'models'/fname)
                name='TimesFM3_RidgeFourier_residual';c=dict(kind='fourier',file=fname,base=base_name,alpha=1,basis=B.tolist());candidates=[]
                for alpha in [0,.25,1]:
                    c['alpha']=alpha;p=make_methods(dev,oo,H,{name:c})[name];v=metric(yy[:,late],p[:,late],mm[:,late],ppd);candidates.append(dict(alpha=alpha,wape=v['wape']))
                minimum=min(r['wape'] for r in candidates);c['alpha']=min(r['alpha'] for r in candidates if r['wape']<=minimum+.005);c['validation_candidates']=candidates;configurations[name]=c
        members=['target_'+f for f in families]+[k for k in ['LightGBM_direct','Ridge_direct','HGB_residual','SparseMoE','JointFactorFourier'] if k in fitbank and k in dev]
        # Duplicate forecasts remain in the complete records but are not counted
        # as additional independent simplex experts.
        seen=set();unique=[]
        for k in members:
            digest=hashlib.sha256(np.ascontiguousarray(fitbank[k][:,ix]).tobytes()).hexdigest()
            if digest not in seen:seen.add(digest);unique.append(k)
        P=np.stack([fitbank[k][:,ix] for k in unique],-1)
        definitions=[('Simplex_global',H*ppd,False)] if ppd==1 else[
            ('Simplex_global',H*ppd,False),('HDSF_extended_halfday',48,False),
            ('HDSF_extended_overlap6h',24,True),('HDSF_extended_overlap12h',48,True),('HDSF_extended_overlap24h',96,True)]
        for name,segment,smooth in definitions:
            weights=fit_weights(P,fy[:,ix],fm[:,ix],segment);configurations[name]=dict(kind='simplex',members=unique,segment=segment,smooth=smooth,weights=weights.tolist())
        methods=make_methods(dev,oo,H,configurations);scores=[]
        reference=methods['baseline_last' if H==1 else'baseline_mean3'];b=metric(yy[:,late],reference[:,late],mm[:,late],ppd)
        for name,p in methods.items():
            s=metric(yy[:,late],p[:,late],mm[:,late],ppd);scores.append(dict(model=name,**s,relative_to_fixed_baseline=s['wape']/b['wape']))
        best=min(scores,key=lambda r:r['wape']);near=[r for r in scores if r['wape']<=best['wape']+.005]
        # Simpler methods break negligible ties; all ranking is development-only.
        selected=min(near,key=lambda r:(r['model'].startswith('HDSF'),r['model'].startswith('target_'),r['model'].startswith('Simplex'),r['wape'],r['model']))['model']
        calibration={}
        for name,p in methods.items():
            residual=yy[:,late]-p[:,late];calibration[name]=[np.quantile(residual[j][mm[j,late]],[.1,.9]).tolist() for j in range(len(y))]
        result['tasks'][str(H)]=dict(checkpoints=chosen,checkpoint_DEV_scores=checkpoint_scores,configurations=configurations,
            selected_method=selected,validation_scores=scores,interval_calibration=calibration,
            weights_fit_phase=fitphase,fit_origins=[fit_oo[int(k)] for k in ix],validation_origins=[oo[int(k)] for k in late],
            last_fit_target_date=str(dates[max(fit_oo[int(k)] for k in ix)+H-1]),last_validation_target_date=str(dates[max(oo[int(k)] for k in late)+H-1]))
        print('FREEZE',a.case,H,selected,'DEV',best['wape'],flush=True)
    save(selection,result);selection.with_suffix('.sha256').write_text(sha(selection),encoding='ascii')
    save(F/'results'/f'{a.case}_before_evaluation.json',dict(frozen_before_evaluation=True,selection_sha256=sha(selection),utc=utc(),exposed_old_periods_not_independent_blind_confirmation=True))

def score():
    s=read(selection);assert sha(selection)==selection.with_suffix('.sha256').read_text().strip();rows=[];arrays={};per_origin=[];leads=[]
    for H in [1,7]:
        task=s['tasks'][str(H)];oo=origins(a.phase,H);bank=load_bank(a.phase,H,task['checkpoints']);methods=make_methods(bank,oo,H,task['configurations']);yy,mm=truth(y,m,oo,H)
        base=methods['baseline_last' if H==1 else'baseline_mean3'];arrays[f'truth_H{H}']=yy;arrays[f'valid_H{H}']=mm;arrays[f'origins_H{H}']=np.array(oo)
        scopes=[(str(n),[i]) for i,n in enumerate(names)] if a.case.startswith('NLR') else [('all57',list(range(len(y))))] if a.case=='Google2019' else[('daily_energy',[0])]
        for name,p in methods.items():
            q=np.array(task['interval_calibration'][name]);iv,lo,hi=interval(yy,mm,p,q[:,0],q[:,1]);arrays[f'{name}_H{H}']=p;arrays[f'{name}_lower80_H{H}']=lo;arrays[f'{name}_upper80_H{H}']=hi
            for scope,ids in scopes:
                stats=metric(yy[ids],p[ids],mm[ids],ppd,base[ids]);ci=interval(yy[ids],mm[ids],p[ids],q[ids,0],q[ids,1])[0]
                row=dict(case=a.case,phase=a.phase,scope=scope,horizon_days=H,model=name,DEV_selected=name==task['selected_method'],unit=cfg['unit'],**stats,**{'interval80_'+k:v for k,v in ci.items()});rows.append(row)
                for i,o in enumerate(oo):per_origin.append(dict(case=a.case,phase=a.phase,scope=scope,horizon_days=H,model=name,origin_date=str(dates[o]),**metric(yy[ids,i:i+1],p[ids,i:i+1],mm[ids,i:i+1],ppd,base[ids,i:i+1])))
                for d in range(H):
                    sl=slice(d*ppd,(d+1)*ppd);leads.append(dict(case=a.case,phase=a.phase,scope=scope,horizon_days=H,model=name,lead_day=d+1,**metric(yy[ids,:,sl],p[ids,:,sl],mm[ids,:,sl],ppd)))
    np_save(F/'results'/f'{a.case}_{a.phase}_predictions.npz',**arrays);export(F/'results'/f'{a.case}_{a.phase}_all_metrics.csv',rows);export(F/'results'/f'{a.case}_{a.phase}_per_origin.csv',per_origin);export(F/'results'/f'{a.case}_{a.phase}_per_lead.csv',leads)
    save(F/'results'/f'{a.case}_{a.phase}_summary.json',dict(all_passed=True,utc=utc(),rows=rows,selection_sha256=sha(selection),prediction_sha256=sha(F/'results'/f'{a.case}_{a.phase}_predictions.npz'),old_test_exposure=True,pretraining_overlap_unknown=True))
    print('SCORED',a.case,a.phase,len(rows),flush=True)
if __name__=='__main__':freeze() if a.stage=='freeze' else score()
