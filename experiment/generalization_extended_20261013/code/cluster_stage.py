"""Matched target fits with real historical CPU/memory, never future workload."""
from pathlib import Path
import argparse, sys, hashlib
import numpy as np,joblib,lightgbm as lgb
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',choices=['freeze','score'],default='freeze');a=ap.parse_args();ROOT=Path(a.root);F=ROOT/'experiment/generalization_extended_20261013';OLD=ROOT/'experiment/generalization_validation_20261012'
sys.path.insert(0,str(OLD/'code'));from core import read,save,sha,utc,truth,metric,interval,np_save,export
z=np.load(F/'data/Google2019.npz');y=z['y'];x=z['x'];mask=z['valid'];names=z['names'];n=np.load(ROOT/'google-clusterdata-2019/aggregated/complete_usage_a_15min.npz')
assert bool(n['complete']) and int(n['source_files'])==int(n['expected_source_files'])==12085
usage=n['aggregate'];source_names=n['domains'] if 'domains' in n.files else n['names'];order=[list(map(str,source_names)).index(str(v)) for v in names];usage=usage[order,:,:,:2].copy();covered=np.where(abs(usage).sum((1,2,3))>0)[0]
assert len(covered)==5 and all(str(names[j]).startswith('cella_') for j in covered)
selected=read(F/'results/Google2019_selection.json');assert sha(F/'results/Google2019_selection.json')==(F/'results/Google2019_selection.sha256').read_text().strip()
def features(j,o,H,cpu,scale):
    step=np.arange(H*96);slot=step%96;hist=x[j,max(0,o-7):o]
    A=np.stack([hist[-1,slot],hist[-3:].mean(0)[slot],hist.mean(0)[slot],hist.std(0)[slot],np.full(len(step),hist[-1].mean()),np.full(len(step),hist.mean()),step//96/7,np.sin(2*np.pi*slot/96),np.cos(2*np.pi*slot/96)],1)
    if cpu:
        c=usage[j,max(0,o-7):o]/scale[None,None];A=np.column_stack([A,c[-1,slot],c[-3:].mean(0)[slot],c.mean(0)[slot],np.broadcast_to(c.mean((0,1)),(len(step),2))])
    return A,hist[-1,slot]
def build(cut,H,cpu,objective,mono):
    path=F/'models'/f'Cluster_{objective}_{"cpu" if cpu else "power"}_{"mono" if mono else "free"}_cut{cut}_H{H}.joblib'
    scales=np.maximum(abs(usage[covered,:cut]).mean((1,2)),1e-8)
    if path.exists():return joblib.load(path),scales
    X=[];Y=[];records=[]
    for o in range(3,cut-H+1):
        for k,j in enumerate(covered):
            A,last=features(int(j),o,H,cpu,scales[k]);keep=mask[j,o:o+H].reshape(-1);X.append(A[keep]);Y.append((y[j,o:o+H].reshape(-1)-last)[keep]);records.append(dict(origin=o,label_end_exclusive=o+H,series=int(j)))
    assert records and max(r['label_end_exclusive'] for r in records)<=cut
    X,Y=np.concatenate(X),np.concatenate(Y);constraints=[0]*9+([1,0]*4 if cpu and mono else [0]*8 if cpu else[])
    model=lgb.LGBMRegressor(objective=objective,num_leaves=7,n_estimators=150,learning_rate=.03,min_child_samples=80,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=20261013,deterministic=True,force_col_wise=True,
        monotone_constraints=constraints if mono else None)
    model.fit(X,Y);joblib.dump(model,path);save(path.with_suffix('.json'),dict(actual_fit=True,training_rows=len(X),cutoff=cut,training_records=records,model_sha256=sha(path),script_sha256=sha(Path(__file__)),CPU_units='NCU',memory_units='normalized source units',normalization_fit_prefix=cut,condition_monotonicity_is_inductive_prior_not_energy_conservation=mono))
    return model,scales
def predict(model,scales,oo,H,cpu):
    rows=[]
    for o in oo:
        p=[]
        for k,j in enumerate(covered):A,last=features(int(j),o,H,cpu,scales[k]);p.append(np.clip(last+model.predict(A),0,1))
        rows.append(np.array(p))
    return np.stack(rows,1)
def base(phase,H):
    import importlib.util
    # Load the frozen portfolio prediction bank without importing its CLI.
    path=F/'results'/f'Google2019_{phase}_portfolio_for_cluster.npz'
    with np.load(path) as q:return q[f'selected_H{H}'].copy()

selection_path=F/'results/Cluster_selection.json';rows=[];arrays={};configs={}
if a.stage=='freeze' and selection_path.exists():assert sha(selection_path)==selection_path.with_suffix('.sha256').read_text().strip();raise SystemExit(0)
if a.stage=='score':configs=read(selection_path)['tasks'];assert sha(selection_path)==selection_path.with_suffix('.sha256').read_text().strip()
for H in [1,7]:
    oo=list(range(14,22)) if a.stage=='freeze' and H==1 else[14,15] if a.stage=='freeze' else list(range(22,31)) if H==1 else[22,23,24]
    cut=14 if a.stage=='freeze' else 22;phase='dev' if a.stage=='freeze' else 'test';original=base(phase,H);yy,mm=truth(y,mask,oo,H);pairs={};models={}
    for label,obj,mono in [('CPU_L1','l1',False),('CPU_monotone_L2','regression',True)]:
        pmodel,psc=build(cut,H,False,obj,False);cmodel,csc=build(cut,H,True,obj,mono)
        p,c=predict(pmodel,psc,oo,H,False),predict(cmodel,csc,oo,H,True);pairs[label]=c-p;models[label+'_power_matched']=p;models[label]=c
    if a.stage=='freeze':
        candidates=[]
        for label,delta in pairs.items():
            for alpha in [0,.25,.5,1]:
                point=np.clip(original[covered]+alpha*delta,0,1);s=metric(yy[covered],point,mm[covered]);candidates.append(dict(method=label,alpha=alpha,DEV_a5_wape=s['wape'],DEV_a5_daily_wape=s['daily_wape']))
        best=min(r['DEV_a5_wape'] for r in candidates);chosen=min([r for r in candidates if r['DEV_a5_wape']<=best+.005],key=lambda r:(r['alpha'],r['DEV_a5_wape'],r['method']))
        adjusted=original.copy();adjusted[covered]=np.clip(original[covered]+chosen['alpha']*pairs[chosen['method']],0,1)
        calibration={}
        for name,point in dict(portfolio_DEV_selected=original,portfolio_CPU_DEV_frozen=adjusted,**models).items():
            actual=yy if len(point)==57 else yy[covered];keep=mm if len(point)==57 else mm[covered];residual=actual-point
            calibration[name]=[np.quantile(residual[j][keep[j]],[.1,.9]).tolist() for j in range(len(point))]
        configs[str(H)]=dict(selected=chosen,candidates=candidates,validation_origins=oo,interval_calibration=calibration)
    else:
        c=configs[str(H)]['selected'];point=original.copy();point[covered]=np.clip(point[covered]+c['alpha']*pairs[c['method']],0,1)
        other=np.setdiff1d(np.arange(57),covered);assert np.array_equal(point[other],original[other])
        for scope,ids in [('cella5',covered),('all57',np.arange(57))]:
            for label,p in [('portfolio_DEV_selected',original),('portfolio_CPU_DEV_frozen',point)]:
                q=np.array(configs[str(H)]['interval_calibration'][label]);iv=interval(yy[ids],mm[ids],p[ids],q[ids,0],q[ids,1])[0]
                rows.append(dict(case='GoogleCluster_same_source',scope=scope,horizon_days=H,model=label,alpha=c['alpha'],unit='normalized power',**metric(yy[ids],p[ids],mm[ids]),**{'interval80_'+k:v for k,v in iv.items()}))
        for label,p in models.items():
            q=np.array(configs[str(H)]['interval_calibration'][label]);iv=interval(yy[covered],mm[covered],p,q[:,0],q[:,1])[0]
            rows.append(dict(case='GoogleCluster_same_source',scope='cella5',horizon_days=H,model=label,unit='normalized power',**metric(yy[covered],p,mm[covered]),**{'interval80_'+k:v for k,v in iv.items()}))
        arrays[f'portfolio_H{H}']=original;arrays[f'portfolio_CPU_H{H}']=point
        for label,p in models.items():arrays[f'{label}_H{H}']=p
    # Actual future-poison feature audit on the frozen feature constructor.
    for o in oo:
        ref=[features(int(j),o,H,True,csc[k])[0] for k,j in enumerate(covered)];saved_x=x[:,o:].copy();saved_u=usage[:,o:].copy();x[:,o:]=999;usage[:,o:]=999999
        assert all(np.array_equal(ref[k],features(int(j),o,H,True,csc[k])[0]) for k,j in enumerate(covered));x[:,o:]=saved_x;usage[:,o:]=saved_u
if a.stage=='freeze':
    save(selection_path,dict(frozen_before_test=True,utc=utc(),tasks=configs,covered_domains=[str(names[i]) for i in covered],source_files=12085,independent_external_power_dataset=False));selection_path.with_suffix('.sha256').write_text(sha(selection_path),encoding='ascii')
else:
    np_save(F/'results/Cluster_predictions.npz',**arrays);export(F/'results/Cluster_all_metrics.csv',rows);save(F/'results/Cluster_summary.json',dict(all_passed=True,utc=utc(),rows=rows,source_files=12085,covered_domains=[str(names[i]) for i in covered],uncovered52_predictions_equal=True,future_poison_checks_passed=True,independent_external_power_dataset=False))
print('Actual matched CPU fits complete',a.stage,flush=True)
