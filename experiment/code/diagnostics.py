import json,warnings
import numpy as np
import pandas as pd
from common import *
from classical import fit_tree,predict_tree,features
from selection import peak_adjust

def interval_summary(y,p,m,lower,upper):
    return dict(coverage_percent=float(100*((y>=lower)&(y<=upper))[m].mean()),mean_width=float((upper-lower)[m].mean()),lower_violation_percent=float(100*(y<lower)[m].mean()),upper_violation_percent=float(100*(y>upper)[m].mean()))

def main():
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    d=load();s=json.loads((ROOT/'results/selection24.json').read_text());s7=json.loads((ROOT/'results/selection7.json').read_text())
    p24=np.load(ROOT/'results/test24_predictions.npz');p7=np.load(ROOT/'results/test7_predictions.npz')
    y,m=truth(d,TEST24);chosen=p24[s['primary24']];base=p24['last'];rows=[];dayrows=[]
    for k in p24.files:
        p=p24[k]
        per=100*(np.abs(y-p)*m).sum((1,2))/(np.abs(y)*m).sum((1,2))
        for i,v in enumerate(per):rows.append(dict(model=k,series=str(d['names'][i]),cell=str(d['cells'][i]),wape=float(v)))
        for j,o in enumerate(TEST24):dayrows.append(dict(model=k,target_day=o+1,**metrics(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1])))
    perdf=pd.DataFrame(rows);perdf.to_csv(ROOT/'results/per_domain_metrics.csv',index=False)
    pd.DataFrame(dayrows).to_csv(ROOT/'results/per_day_metrics.csv',index=False)
    perdf.groupby(['model','cell'],as_index=False).wape.mean().to_csv(ROOT/'results/per_cell_metrics.csv',index=False)
    pivot=perdf.pivot(index='series',columns='model',values='wape')
    selected=s['primary24'];wins=int((pivot[selected]<pivot['last']).sum())
    daydf=pd.DataFrame(dayrows).pivot(index='target_day',columns='model',values='wape')
    rng=np.random.default_rng(20260929);boot=[]
    # Moving three-day blocks keep neighbouring day effects together; inference
    # remains descriptive because there are only nine test days.
    for _ in range(2000):
        ix=np.concatenate([np.arange(a,a+3) for a in rng.integers(0,7,3)])
        den=(np.abs(y[:,ix])*m[:,ix]).sum((1,2))
        a=(np.abs(y[:,ix]-base[:,ix])*m[:,ix]).sum((1,2))/den
        b=(np.abs(y[:,ix]-chosen[:,ix])*m[:,ix]).sum((1,2))/den
        boot.append(100*np.mean(a-b))
    stability=dict(primary24=selected,domains_improved_vs_last=wins,domains_total=57,days_improved_vs_last=int((daydf[selected]<daydf['last']).sum()),days_total=9,three_day_block_bootstrap_difference_pp_95=np.percentile(boot,[2.5,97.5]).tolist(),bootstrap_replicates=2000,interpretation='Exploratory interval only; same previously examined month, nine dependent test days.')
    (ROOT/'results/stability.json').write_text(json.dumps(stability,indent=2))
    # Past-only causality test: overwrite every future value and verify feature inputs.
    altered={k:v.copy() for k,v in d.items()};altered['y'][:,14:]=.123;altered['prod'][:,14:]=.987;altered['pvalid'][:,14:]=False
    assert np.array_equal(features(d,14,True)[0],features(altered,14,True)[0])
    assert np.array_equal(baseline(d,[14]),baseline(altered,[14]))
    from deep_models import project_components
    a,b,c=project_components(np.array([-.2,.7,1.4]),np.array([.2,.8,.8]),np.array([-.1,.3,.9]))
    assert np.allclose(a,b+c) and (b>=0).all() and (c>=0).all() and (a<=1+1e-6).all()
    audit=dict(future_perturbation_features_unchanged=True,future_perturbation_baseline_unchanged=True,component_projection_constraints_passed=True,selection24_sha256=(ROOT/'results/selection24.sha256').read_text(),selection7_sha256=(ROOT/'results/selection7.sha256').read_text(),test_weights_frozen=True)
    (ROOT/'results/verification.json').write_text(json.dumps(audit,indent=2))
    # Intervals calibrated only on development residuals. No distribution-free
    # coverage claim is made for dependent and model-selected time-series data.
    intervals=[]
    for horizon,origins,devorig,preds,devfile,primary in [(1,TEST24,DEV24,p24,'dev_selected',selected),(7,TEST7,DEV7,p7,'dev7',s7['primary7'])]:
        yy,mm=truth(d,origins,horizon);dy,dm=truth(d,devorig,horizon);dp=np.load(ROOT/f'results/{devfile}.npz')[primary]
        radius=np.array([np.quantile(np.abs(dy[i]-dp[i])[dm[i]],.8,method='higher') for i in range(57)])[:,None,None]
        lower=np.clip(preds[primary]-radius,0,1);upper=np.clip(preds[primary]+radius,0,1)
        intervals.append(dict(horizon_days=horizon,model=primary,kind='development_absolute_residual_80',**interval_summary(yy,preds[primary],mm,lower,upper)))
        np.savez_compressed(ROOT/f'results/interval{horizon}_primary.npz',lower=lower,upper=upper,radius=radius)
        qfile=ROOT/f'results/test{24 if horizon==1 else 7}_quantiles.npz';q=np.load(qfile)
        dqfile=ROOT/'results/dev_chronos_quantiles.npz' if horizon==1 else ROOT/'results/dev7_quantiles.npz';dq=np.load(dqfile)
        for name in ['chronos_zero',s['families']['chronos_lora']]:
            lo=q[name][...,0];hi=q[name][...,2]
            intervals.append(dict(horizon_days=horizon,model=name,kind='raw_80',**interval_summary(yy,q[name][...,1],mm,lo,hi)))
            dlo=dq[name][...,0];dhi=dq[name][...,2]
            # Non-negative conformal-style expansion; not a coverage guarantee.
            scores=np.maximum(np.maximum(dlo-dy,dy-dhi),0)
            adjust=np.array([np.quantile(scores[i][dm[i]],.8,method='higher') for i in range(57)])[:,None,None]
            intervals.append(dict(horizon_days=horizon,model=name,kind='development_expanded_80',**interval_summary(yy,q[name][...,1],mm,np.clip(lo-adjust,0,1),np.clip(hi+adjust,0,1))))
    pd.DataFrame(intervals).to_csv(ROOT/'results/interval_metrics.csv',index=False)
    # Leave-one-cell-out sensitivity for the supervised load model. This does not
    # establish leave-one-cell-out performance of the complete foundation ensemble.
    cfgs=json.loads((ROOT/'results/classical_configs.json').read_text());name=s['families']['lgb_load'];cfg=cfgs[name];loco=np.empty_like(y)
    for cell in np.unique(d['cells']):
        model,_=fit_tree(d,22,cfg,exclude_cell=cell);pp=predict_tree(model,d,TEST24,cfg)
        ix=d['cells']==cell;loco[ix]=pp[ix];print('LOCO',cell,flush=True)
    np.savez_compressed(ROOT/'results/loco_predictions.npz',prediction=loco)
    result=metrics(y,loco,m);result.update(model=name,training_cells_exclude_target=True,target_cell_past_available=True)
    (ROOT/'results/loco_metrics.json').write_text(json.dumps(result,indent=2))
    # Human-readable long forecast files, only for primary forecasts.
    for horizon,origins,preds,primary in [(1,TEST24,p24,selected),(7,TEST7,p7,s7['primary7'])]:
        yy,mm=truth(d,origins,horizon);pp=preds[primary];recs=[]
        for i in range(57):
            for j,o in enumerate(origins):
                for step in range(horizon*96):
                    recs.append((str(d['names'][i]),str(d['cells'][i]),o,o+step//96+1,step%96,step+1,yy[i,j,step],pp[i,j,step],bool(mm[i,j,step])))
        pd.DataFrame(recs,columns=['series','cell','origin_end_day','target_day','quarter_hour_slot','horizon_step','actual_util','predicted_util','valid_target']).to_csv(ROOT/f'results/primary_forecasts_{horizon}day.csv',index=False)
    print(json.dumps(stability,indent=2));print(pd.DataFrame(intervals).to_string(index=False));print('LOCO',result)

if __name__=='__main__':main()
