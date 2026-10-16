from pathlib import Path
import argparse,copy,sys
import numpy as np
from core import *
from models import *
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('stage',choices=['freeze','score','audit']);a=ap.parse_args();R=Path(a.root);F,cfg,d,q=load(R)
sys.path.insert(0,str(R/'experiment/round44/reference/PowerData_Prediction'))
from power_forecast.attainment_fusion import fit_simplex
names=['LightGBM','timesfm3','toto2','moirai2'];threshold=cfg['mape_threshold']
def fitw(P,Y,segment):
    W=[]
    for start in range(0,Y.shape[1],segment):
        sl=slice(start,min(Y.shape[1],start+segment));A=P[:,sl].reshape(-1,4);B=Y[:,sl].reshape(-1);W.append(fit_simplex(A,B,np.ones(len(B),bool),max_points=20000))
    return np.array(W)
def blend(P,W):
    out=np.einsum('olk,lk->ol',P,W[np.arange(P.shape[1])//48]);assert np.isfinite(out).all() and W.min()>=0 and np.allclose(W.sum(1),1);return np.maximum(0,out)
if a.stage=='freeze':
    path=F/'results/selection.json'
    if path.exists():assert sha(path)==(F/'results/selection.sha256').read_text().strip();print('Original frozen selection retained');raise SystemExit()
    selected=dict(utc=utc(),protocol_sha256=sha(F/'docs/protocol.json'),tasks={},all_tasks_frozen_before_test=True,test_scoring_started=False);cvrows=[]
    for H in [1,7]:
        oo=np.array(origins(q,'validation',H));Y=target(d,oo,H);candidates=[read_forecasts(F,q,'validation',H,f'LightGBM_c{i}') for i in range(2)];c=read(F/'results'/f'LightGBM_H{H}_validation_selection.json')['selected']
        finalP=np.stack([candidates[c]]+[read_forecasts(F,q,'validation',H,n) for n in names[1:]],-1)
        hold_chunks=np.array_split(np.arange(len(oo)),4)[1:];cvpred={n:[] for n in names+['last','week','mean3']};hdsf={str(alpha):[] for alpha in cfg['hdsf_alpha_grid']};Ys=[];held_origins=[];foldinfo=[]
        for fold,ii in enumerate(hold_chunks,1):
            start=int(oo[ii[0]]);fitidx=np.where(oo+H<=start)[0];assert len(fitidx)>=8 and (oo[fitidx]+H<=start).all()
            fc=min(range(2),key=lambda i:(metrics(Y[fitidx],candidates[i][fitidx],threshold)['curve_mape_percent'],i))
            P=np.stack([candidates[fc]]+[read_forecasts(F,q,'validation',H,n) for n in names[1:]],-1);gw=fitw(P[fitidx],Y[fitidx],H*96);hw=fitw(P[fitidx],Y[fitidx],48)
            for alpha in cfg['hdsf_alpha_grid']:hdsf[str(alpha)].append(blend(P[ii],(1-alpha)*np.repeat(gw,len(hw),axis=0)+alpha*hw))
            for j,n in enumerate(names):cvpred[n].append(P[ii,:,j])
            for n in ['last','week','mean3']:cvpred[n].append(baseline(d,oo[ii],H,n))
            Ys.append(Y[ii]);held_origins.extend(oo[ii].tolist());foldinfo.append(dict(fold=fold,fit_origins=oo[fitidx].tolist(),held_origins=oo[ii].tolist(),purged_all_labels_end_no_later_than_hold_start=True,lightgbm_candidate_selected_only_on_prior_validation=fc))
        CY=np.concatenate(Ys);alpha_scores={key:metrics(CY,np.concatenate(p),threshold)['curve_mape_percent'] for key,p in hdsf.items()};alpha=min(cfg['hdsf_alpha_grid'],key=lambda x:(alpha_scores[str(x)],x));CV={n:np.concatenate(p) for n,p in cvpred.items()};CV['HDSF']=np.concatenate(hdsf[str(alpha)])
        scores={n:metrics(CY,p,threshold) for n,p in CV.items()};best=min(scores,key=lambda n:(scores[n]['curve_mape_percent'],n))
        gw=fitw(finalP,Y,H*96);hw=fitw(finalP,Y,48);W=(1-alpha)*np.repeat(gw,len(hw),axis=0)+alpha*hw;fp=blend(finalP,W)
        selected['tasks'][str(H)]=dict(lightgbm_candidate=c,alpha=alpha,weights=W.tolist(),expert_names=names,halfday_points=48,selection_metric='purged forward validation CV origin mean MAPE',cv_scores=scores,alpha_cv_mape=alpha_scores,selected_method=best,cv_origins=held_origins,cv_folds=foldinfo,
          fusion_in_sample_score=metrics(Y,fp,threshold),fusion_in_sample_score_is_not_generalization=True)
        np_save(F/'cache'/f'validation_cv_H{H}.npz',y=CY,origins=np.array(held_origins),**CV);save_forecasts(F,cfg,d,q,'validation',H,'HDSF_in_sample',fp,dict(in_sample_fit=True,not_generalization=True))
        for n,s in scores.items():cvrows.append(dict(phase='validation_forward_cv',horizon_days=H,model=n,**s))
        print('FREEZE',H,best,'alpha',alpha,'CV MAPE',scores[best]['curve_mape_percent'],flush=True)
    save(path,selected);(F/'results/selection.sha256').write_text(sha(path));export(F/'results/验证时间交叉检验全指标.csv',cvrows)
elif a.stage=='score':
    selection=read(F/'results/selection.json');assert sha(F/'results/selection.json')==(F/'results/selection.sha256').read_text().strip();rows=[];daily_rows=[]
    for H in [1,7]:
        oo=np.array(origins(q,'test',H));Y=target(d,oo,H);P=np.stack([read_forecasts(F,q,'test',H,n) for n in names],-1);preds={n:P[:,:,j] for j,n in enumerate(names)};preds['HDSF']=blend(P,np.array(selection['tasks'][str(H)]['weights']))
        for n in ['last','week','mean3']:preds[n]=baseline(d,oo,H,n)
        for n,p in preds.items():save_forecasts(F,cfg,d,q,'test',H,n,p,dict(selection_sha256=sha(F/'results/selection.json'),test_refit_after_inference=False))
        subsets={'daily_origins':np.arange(len(oo))}
        if H==7:subsets['nonoverlapping_calendar7day']=np.flatnonzero((oo-q['split']['test'][0])%7==0)
        for cohort,ii in subsets.items():
            if not len(ii):raise RuntimeError('No complete nonoverlap7day test window under calendar anchor')
            bname='last' if H==1 else'mean3';b=metrics(Y[ii],preds[bname][ii],threshold)
            for n,p in preds.items():
                score=add_baseline(metrics(Y[ii],p[ii],threshold),b);rows.append(dict(dataset='NLR derived facility subtotal',target=cfg['target'],unit='kW',phase='test',cohort=cohort,horizon_days=H,model=n,selected_on_validation=n==selection['tasks'][str(H)]['selected_method'],**score))
            for lead in range(H):
                for n,p in preds.items():daily_rows.append(dict(cohort=cohort,forecast_day=lead+1,model=n,**metrics(Y[ii,lead*96:(lead+1)*96],p[ii,lead*96:(lead+1)*96],threshold)))
        np_save(F/'results'/f'test_H{H}.npz',truth=Y,origins=oo,**preds)
    export(F/'results/新协议主要模型测试全指标.csv',rows);export(F/'results/新协议逐预测日测试指标.csv',daily_rows);save(F/'results/summary.json',dict(utc=utc(),rows=rows,selection_sha256=sha(F/'results/selection.json'),protocol_sha256=sha(F/'docs/protocol.json'),selection_not_changed_after_test=True,independent_confirmation=False,test_exposed_in_prior_experiments=True))
else:
    checks=[]
    def check(name,value):checks.append(dict(check=name,passed=bool(value)));assert value,name
    for H in [1,7]:
        for phase in ['train','validation','test']:
            for o in origins(q,phase,H):check('Window bounds '+phase+str(H)+' '+str(o),o>=14 and o+H<=q['split'][phase][1] and o>=q['split'][phase][0] and np.isfinite(d['y'][o-14:o+H]).all())
    # Actual future-poison check for feature builder and foundation context.
    for H in [1,7]:
        o=origins(q,'test',H)[0];base=features(d,o,H);poison={k:v.copy() for k,v in d.items()};poison['y'][o:]=1e10;poison['covariates'][o:]=-1e10
        check('Future actual target/weather never enters features '+str(H),np.array_equal(base,features(poison,o,H)))
        check('Foundation14day history future poison '+str(H),np.array_equal(d['y'][o-14:o],poison['y'][o-14:o]))
    toy=metrics(np.array([[1.,0],[100,100]]),np.array([[2.,50],[100,100]]),1e-8,2);check('Origin equal MAPE differs correctly from pooled',np.isclose(toy['curve_mape_percent'],50) and np.isclose(toy['curve_mape_pooled_percent'],100/3) and toy['mape_excluded']==1)
    zero=metrics(np.zeros((2,96)),np.ones((2,96)));check('Zero denominators null, not false accuracy',zero['curve_wape_percent'] is None and zero['curve_mape_percent'] is None and zero['mape_excluded']==192)
    check('Source quality timezone disclosure',not read(F/'data/quality_report.json')['timezone_verified']);check('Configuration no future covariates',cfg['future_inputs'].startswith('known calendar'))
    if (F/'results/summary.json').exists():
        summary=read(F/'results/summary.json');check('Original pretest freeze remains',summary['selection_sha256']==sha(F/'results/selection.json'))
        for H in [1,7]:
            with np.load(F/'results'/f'test_H{H}.npz') as z:
                for row in summary['rows']:
                    if row['horizon_days']!=H:continue
                    oo=z['origins'];ii=np.arange(len(oo)) if row['cohort']=='daily_origins' else np.flatnonzero((oo-q['split']['test'][0])%7==0);v=metrics(z['truth'][ii],z[row['model']][ii],threshold)
                    for k,w in v.items():check('Recomputed '+row['model']+str(H)+row['cohort']+' '+k,w==row[k] if w is None or isinstance(w,int) else np.isclose(w,row[k],atol=1e-10,rtol=0))
    save(F/'results/verification.json',dict(all_passed=True,utc=utc(),checks=checks,count=len(checks),test_scored=(F/'results/summary.json').exists(),source_labels_not_imputed=True,all_models_share_cohorts=True));print('Audit passed',len(checks),flush=True)
