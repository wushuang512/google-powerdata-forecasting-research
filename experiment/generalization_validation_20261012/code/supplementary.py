"""Real daily small-sample validation and matched same-source CPU ablation."""
from pathlib import Path
import argparse,sys
import numpy as np
from core import *
def sea(root,folder):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    import joblib
    sys.path.insert(0,str(root/'experiment/round44/reference/PowerData_Prediction'))
    from power_forecast.attainment_fusion import fit_simplex
    protocol_check(folder);z=np.load(folder/'data/sea_daily.npz');y=z['y'];mask=z['valid'];x=y[0,:,0]/z['scales'][0]
    origins=read(folder/'data/sea_preparation.json')['origins'];selection_path=folder/'results/sea_selection.json'
    def feature(history):
        return np.array([history[-1],history[-2],history[-3],history[-7],np.mean(history[-3:]),np.mean(history[-7:])])
    def train(cut):
        X=np.stack([feature(x[:o]) for o in range(7,cut)]);target=np.array([x[o]-x[o-1] for o in range(7,cut)])
        p=make_pipeline(StandardScaler(),Ridge(alpha=10));p.fit(X,target);return p
    def pred(model,oo,h):
        out=[]
        for o in oo:
            a=list(x[:o]);row=[]
            for k in range(h):
                v=max(0,float(model.predict(feature(np.array(a))[None])[0])+a[-1]);row.append(v);a.append(v)
            out.append(row)
        return np.array(out)[None]*z['scales'][0]
    def bank(model,oo,h):
        b={k:baseline(y,oo,h,k) for k in ['last','week','mean3']};b['Ridge_daily']=pred(model,oo,h);return b
    devmodel=train(14);choices={};devbanks={};calibrations={}
    for h in [1,7]:
        oo=origins['dev'+str(h)];b=bank(devmodel,oo,h);devbanks[h]=b;yy,mm=truth(y,mask,oo,h);P=np.stack(list(b.values()),-1)
        targets=np.array([o+np.arange(h) for o in oo]);early=mm&(targets[None]<18);late=mm&(targets[None]>=18)
        weights=fit_simplex(P.reshape(-1,len(b)),yy.reshape(-1),early.reshape(-1));globalpoint=np.einsum('solc,c->sol',P,weights)
        cases=[]
        for alpha in [0,.25,.5,1]:
            p=(1-alpha)*b['last']+alpha*globalpoint;score=metric(yy,p,late,1);cases.append(dict(alpha=alpha,late_wape=score['wape']))
        best=min(c['late_wape'] for c in cases);alpha=min(c['alpha'] for c in cases if c['late_wape']<=best+.005)
        b['daily_simplex']=(1-alpha)*b['last']+alpha*globalpoint
        cal={name:np.quantile((yy-point)[late],[.1,.9]).tolist() for name,point in b.items()}
        choices[str(h)]=dict(members=list(b)[:-1],weights=weights.tolist(),alpha=alpha,DEV_candidates=cases,calibration=cal)
    if selection_path.exists():
        old=read(selection_path);assert old['tasks']==choices
    else:
        save(selection_path,dict(utc=utc(),tasks=choices,protocol_sha256=sha(folder/'docs/protocol.json'),frozen_before_test=True));selection_path.with_suffix('.sha256').write_text(sha(selection_path),encoding='ascii')
    model=train(22);joblib.dump(model,folder/'models/sea_ridge_cut22.joblib');rows=[];arrays={}
    for h in [1,7]:
        oo=origins['test'+str(h)];b=bank(model,oo,h);yy,mm=truth(y,mask,oo,h);c=choices[str(h)];P=np.stack([b[n] for n in c['members']],-1)
        b['daily_simplex']=(1-c['alpha'])*b['last']+c['alpha']*np.einsum('solc,c->sol',P,np.array(c['weights']))
        for name,p in b.items():
            score=metric(yy,p,mm,1,b['last'] if h==1 else b['mean3']);qlo,qhi=c['calibration'][name];iv,lo,hi=interval(yy,mm,p,np.array([qlo]),np.array([qhi]))
            rows.append(dict(dataset='SeaGuang',unit='kWh',task='next_day_energy' if h==1 else'future7_daily_energy',horizon_days=h,model=name,**score,**{'interval80_'+k:v for k,v in iv.items()}))
            arrays[f'{name}_H{h}']=p;arrays[f'{name}_lower80_H{h}']=lo;arrays[f'{name}_upper80_H{h}']=hi
        arrays[f'truth_H{h}']=yy;arrays[f'valid_H{h}']=mm
    np_save(folder/'results/sea_predictions.npz',**arrays);export(folder/'results/海光寺_日电量全部统一指标.csv',rows)
    save(folder/'results/sea_summary.json',dict(utc=utc(),rows=rows,selection_sha256=sha(selection_path),days=30,independent_confirmation=False,
        full_HDSF_intraday_not_testable=True,method='daily-scale simplex transfer, not HDSF halfday curve validation',no_intraday_peak_metrics_fabricated=True))
    print('SeaGuang real daily validation complete',[(r['horizon_days'],r['model'],round(r['wape'],5)) for r in rows],flush=True)
def cluster(root,folder):
    import lightgbm as lgb,joblib
    protocol_check(folder);d=np.load(root/'experiment/results/data.npz');y=d['y'];m=d['valid'];names=d['names'];n=np.load(root/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz')
    assert bool(n['complete']) and int(n['source_files'])==int(n['expected_source_files'])==12085
    print('Cluster source keys',n.files,flush=True)
    # The source audit lists measured domains. Never infer availability from
    # finite zero placeholders in the other52domains.
    usage=n['aggregate'].copy()
    if usage.shape[-1]!=6:raise ValueError(usage.shape)
    covered=np.where((abs(usage).sum((1,2,3))>0))[0];assert len(covered)==5
    # Source aggregate may have different series order; align by actual IDs.
    source_names=n['domains'] if 'domains' in n.files else n['names']
    index=[list(map(str,source_names)).index(str(v)) for v in names];usage=usage[index];covered=np.where(abs(usage).sum((1,2,3))>0)[0]
    assert len(covered)==5 and all(str(names[i]).startswith('cella_') for i in covered)
    scales=np.maximum(np.mean(usage[covered,:14,:,:2],axis=(1,2)),1e-8)
    def features(o,h,j,with_cpu):
        steps=np.arange(h*96);slots=steps%96;power=y[j,max(0,o-7):o]
        pp=np.column_stack([np.full(len(steps),steps//96+1),np.sin(2*np.pi*slots/96),np.cos(2*np.pi*slots/96),
            power[-1,slots],np.mean(power[-3:],axis=0)[slots],np.mean(power,axis=0)[slots],np.std(power,axis=0)[slots],
            np.full(len(steps),power[-1].mean()),np.full(len(steps),power.mean())])
        if with_cpu:
            c=usage[j,max(0,o-7):o,:,:2]/scales[list(covered).index(j)][None,None,:]
            pp=np.column_stack([pp,c[-1,slots],c[-3:].mean(0)[slots],np.broadcast_to(c.mean((0,1)),(len(steps),2))])
        return pp
    def train(cut,h,cpu):
        X=[];Y=[]
        for o in range(3,cut-h+1):
            for j in covered:
                xx=features(o,h,j,cpu);yy=y[j,o:o+h].reshape(-1);mm=m[j,o:o+h].reshape(-1)
                X.append(xx[mm]);Y.append((yy-y[j,o-1,np.arange(h*96)%96])[mm])
        model=lgb.LGBMRegressor(objective='l1',num_leaves=7,n_estimators=120,learning_rate=.03,min_child_samples=80,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=20261012,deterministic=True,force_col_wise=True)
        model.fit(np.concatenate(X),np.concatenate(Y));return model
    def pred(model,oo,h,cpu):
        return np.stack([np.stack([np.maximum(0,y[j,o-1,np.arange(h*96)%96]+model.predict(features(o,h,j,cpu))) for o in oo]) for j in covered])
    rows=[];arrays={};selection_path=folder/'results/cluster_selection.json';s={};prior=np.load(root/'experiment/round48/results/predictions.npz')
    for h,dev,test in [(1,[14,16,18,20],list(range(22,31))),(7,[14,15],[22,23,24])]:
        a,b=train(14,h,False),train(14,h,True);da,db=pred(a,dev,h,False),pred(b,dev,h,True)
        # CPU correction chosen against the fixed current scheme using DEV only.
        original_dev=prior[f'dev_DEV_frozen_R48_candidate_H{h}'][covered]
        yy,mm=truth(y,m,dev,h);cases=[]
        for alpha in [0,.25,.5,1]:
            candidate=np.clip(original_dev+alpha*(db-da),0,1);score=metric(yy[covered],candidate,mm[covered])
            cases.append(dict(alpha=alpha,DEV_wape=score['wape'],DEV_daily_wape=score['daily_wape'],DEV_peak_mape=score['peak_mape']))
        best=min(c['DEV_wape'] for c in cases);alpha=min(c['alpha'] for c in cases if c['DEV_wape']<=best+.005)
        s[str(h)]=dict(alpha=alpha,candidates=cases,complete_domains=list(map(str,names[covered])),independent_confirmation=False)
        # Save frozen choices before any final model/test prediction for task.
        save(folder/'results'/f'cluster_H{h}_before_test.json',dict(utc=utc(),**s[str(h)],frozen_before_test=True))
        a,b=train(22,h,False),train(22,h,True);joblib.dump(a,folder/'models'/f'cluster_power_only_H{h}.joblib');joblib.dump(b,folder/'models'/f'cluster_cpu_memory_H{h}.joblib')
        pa,pb=pred(a,test,h,False),pred(b,test,h,True);full=prior[f'DEV_frozen_R48_candidate_H{h}'].copy();hybrid=full.copy();hybrid[covered]=np.clip(full[covered]+alpha*(pb-pa),0,1)
        yy,mm=truth(y,m,test,h)
        for scope,ix in [('cella5',covered),('all57',np.arange(57))]:
            for name,p in [('R48_fixed',full),('R48_CPU_DEV_frozen',hybrid)]:rows.append(dict(dataset='GoogleCluster_same_source',scope=scope,horizon_days=h,model=name,alpha=alpha,unit='normalized power',**metric(yy[ix],p[ix],mm[ix])))
        for name,p in [('LightGBM_power_only',pa),('LightGBM_power_CPU_memory',pb)]:rows.append(dict(dataset='GoogleCluster_same_source',scope='cella5',horizon_days=h,model=name,unit='normalized power',**metric(yy[covered],p,mm[covered])))
        assert np.array_equal(full[np.setdiff1d(np.arange(57),covered)],hybrid[np.setdiff1d(np.arange(57),covered)])
        # Future-poison actual feature audit at each declared origin.
        original_power=y.copy();original_usage=usage.copy()
        for o in test:
            ref=[features(o,h,int(j),True) for j in covered];y[:,o:]=.987;usage[:,o:]=999999
            assert all(np.array_equal(ref[k],features(o,h,int(j),True)) for k,j in enumerate(covered))
            y[:]=original_power;usage[:]=original_usage
        arrays[f'R48_fixed_H{h}']=full;arrays[f'R48_CPU_DEV_frozen_H{h}']=hybrid;arrays[f'power_only_H{h}']=pa;arrays[f'CPU_memory_H{h}']=pb
    save(selection_path,dict(utc=utc(),tasks=s,protocol_sha256=sha(folder/'docs/protocol.json')))
    np_save(folder/'results/cluster_predictions.npz',**arrays);export(folder/'results/ClusterData_同条件算力增益全部统一指标.csv',rows)
    save(folder/'results/cluster_summary.json',dict(utc=utc(),rows=rows,covered_domains=list(map(str,names[covered])),missing_placeholder_domains=52,
        scope='same previously viewed May2019; covariate ablation, not independent power generalization',all_source_files=12085,all_future_feature_poison_checks_passed=True))
    print('Cluster real matched ablation complete',[(r['scope'],r['horizon_days'],r['model'],round(r['wape'],5)) for r in rows],flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root');a.add_argument('stage',choices=['sea','cluster']);s=a.parse_args();root=Path(s.root);folder=root/'experiment/generalization_validation_20261012'
    (sea if s.stage=='sea' else cluster)(root,folder)
