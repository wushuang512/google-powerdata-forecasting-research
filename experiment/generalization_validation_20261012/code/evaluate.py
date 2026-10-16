"""Chronological target fitting, fixed source transfer, sealed test scoring."""
from pathlib import Path
import argparse,sys
import numpy as np
from core import *
from local_models import fit,collect
FAMILIES=['chronos2','tirex2','timesfm25','timesfm3','toto2','patchtst']
def load_bank(root,folder,phase,h):
    proto=protocol_check(folder);oo=proto['nlr_origins'][phase];names,W=original_weights(root,h);values={}
    families={'chronos2_long':'chronos2','tirex2':'tirex2','timesfm25':'timesfm25',
        'timesfm30':'timesfm3','timesfm30_long':'timesfm3','timesfm30_uni':'timesfm3',
        'toto2_joint':'toto2','toto2_single':'toto2','patchtstfm_long':'patchtst'}
    local=np.load(folder/'cache'/f'local_{phase}.npz')
    for name in names:
        if name=='local24':values[name]=local['google_local24_H1']
        elif name=='local7':values[name]=local['local7_H7']
        else:values[name]=np.stack([np.load(folder/'cache'/f'{phase}_{families[name]}_original_{name}_H{h}_o{o}.npz')['p'] for o in oo],1)
    tensor=np.stack([values[name] for name in names],-1);plain=simplex_tensor(tensor,W)
    if h==1:
        before=np.stack([np.load(folder/'cache'/f'{phase}_patchtst_original_patchtstfm_long_H1_o{o}.npz')['p'] for o in oo],1)
        after=np.stack([np.load(folder/'cache'/f'{phase}_patchtst_google_r47_curve_patchtstfm_long_H1_o{o}.npz')['p'] for o in oo],1)
        delta=(after-before).reshape(2,len(oo),1,96).mean(-1,keepdims=True).repeat(96,-1).reshape(plain.shape)
    else:
        delta=np.zeros_like(plain)
        for name in ['timesfm30_long','timesfm30_uni','toto2_joint','patchtstfm_long']:
            after=np.stack([np.load(folder/'cache'/f'{phase}_{families[name]}_google_r46_{name}_H7_o{o}.npz')['p'] for o in oo],1)
            w=W[np.arange(672)//48,names.index(name)];delta+=(after-values[name])*w[None,None]
        shaped=delta.reshape(2,len(oo),7,96);delta=(shaped-shaped.mean(-1,keepdims=True)).reshape(plain.shape)
    current=np.maximum(0,plain+delta)
    # Original local24 is unchanged for the transfer route. Target refit is a
    # separately named bank, not a silently modified transferred predictor.
    refit=tensor.copy()
    if h==1:refit[:,:,:,names.index('local24')]=local['nlr_local24_H1']
    return names,W,tensor,refit,np.maximum(0,plain),current
def fit_weights(P,y,m,segments):
    from power_forecast.attainment_fusion import fit_simplex
    weights=[];index=np.arange(P.shape[1])//(48 if segments>1 else P.shape[1])
    for s in range(segments):
        keep=index==s;members=P[:,keep].reshape(-1,P.shape[-1]);label=y[:,keep].reshape(-1);valid=m[:,keep].reshape(-1)
        weights.append(fit_simplex(members,label,valid,iterations=25,max_points=20000))
    return np.array(weights)
def apply_refit(P,c):
    a=np.einsum('olc,c->ol',P,np.array(c['global'])[0]);b=simplex_tensor(P[None],np.array(c['halfday']))[0]
    return np.maximum(0,(1-c['alpha'])*a+c['alpha']*b)
def freeze(root,folder):
    selection=folder/'results/nlr_selection.json'
    if selection.exists():
        assert sha(selection)==selection.with_suffix('.sha256').read_text().strip()
        if not all((folder/'models'/f'nlr_local_test_{j}.joblib').exists() and (folder/'models'/f'nlr_ridge_test_{j}.joblib').exists() for j in range(2)):fit(root,folder,'test')
        return
    proto=protocol_check(folder);sys.path.insert(0,str(root/'experiment/round44/reference/PowerData_Prediction'))
    fit(root,folder,'dev');collect(root,folder,'dev')
    z=np.load(folder/'data/nlr_15min.npz');oo=proto['nlr_origins']['dev'];dates=z['dates'];sc=z['scales'];result=dict(utc=utc(),protocol_sha256=sha(folder/'docs/protocol.json'),tasks={})
    early=np.array([str(dates[o])<'2023-07-01' for o in oo]);late=~early;assert early.sum()>=3 and late.sum()>=3
    for h in [1,7]:
        names,_,_,P,plain,current=load_bank(root,folder,'dev',h);y,m=truth(z['y']/sc[:,None,None],z['valid'],oo,h)
        result['tasks'][str(h)]={}
        for j in range(2):
            globalw=fit_weights(P[j,early],y[j,early],m[j,early],1);halfw=fit_weights(P[j,early],y[j,early],m[j,early],h*2)
            cases=[]
            for alpha in proto['simplex_candidates']:
                c={'global':globalw.tolist(),'halfday':halfw.tolist(),'alpha':alpha};p=apply_refit(P[j],c)
                score=metric(y[j:j+1,late],p[None,late],m[j:j+1,late])
                cases.append(dict(alpha=alpha,late_wape=score['wape'],late_daily_wape=score['daily_wape'],late_peak_mape=score['peak_mape']))
            best=min(c['late_wape'] for c in cases);chosen=min([c for c in cases if c['late_wape']<=best+.005],key=lambda c:c['alpha'])
            c=dict(halfday=halfw.tolist(),alpha=chosen['alpha'],members=names,DEV_candidates=cases,
                weights_fit_last_target_date=str(dates[max(np.array(oo)[early])+h-1]),selection_last_target_date=str(dates[max(np.array(oo)[late])+h-1]),
                neural_weights_updated_on_target=False,local_labels_through='2022-12-31');c['global']=globalw.tolist()
            # Each model gets its own late-DEV residual calibration. It does not
            # modify its point prediction and is explicitly target-calibrated.
            local=np.load(folder/'cache/local_dev.npz');preds=dict(HDSF_Google_frozen=plain[j],R48_Google_frozen=current[j],HDSF_NLR_refit=apply_refit(P[j],c))
            xn=z['x']/sc[:,None,None]
            for kind in ['last','week','mean3']:preds[kind]=baseline(xn[j:j+1],oo,h,kind)[0]
            if h==1:preds.update(LightGBM_NLR=local['nlr_local24_H1'][j],Ridge_NLR=local['nlr_ridge_H1'][j])
            else:preds['local7_smoother']=local['local7_H7'][j]
            intervals={}
            for name,p in preds.items():
                residual=(y[j]-p)[late][m[j,late]]*sc[j];intervals[name]=np.quantile(residual,[.1,.9]).tolist()
            c['intervals_late_DEV']=intervals;result['tasks'][str(h)][str(j)]=c
    save(selection,result);selection.with_suffix('.sha256').write_text(sha(selection),encoding='ascii')
    save(folder/'results/before_test_freeze.json',dict(utc=utc(),selection_sha256=sha(selection),protocol_sha256=sha(folder/'docs/protocol.json'),test_inference_started=False))
    print('NLR fusion/interval parameters frozen before TEST',sha(selection),flush=True)
    fit(root,folder,'test')
def paired_month_ci(y,m,p,b,dates):
    months=np.array([str(t)[:7] for t in dates]);groups=[np.where(months==g)[0] for g in sorted(set(months))]
    rng=np.random.default_rng(20261012);gain=[]
    for _ in range(2000):
        ix=np.concatenate([groups[k] for k in rng.integers(len(groups),size=len(groups))]);v=m[:,ix];den=(abs(y[:,ix])*v).sum()
        if den>0:gain.append(float(100*((abs(y[:,ix]-b[:,ix])-abs(y[:,ix]-p[:,ix]))*v).sum()/den))
    lo,hi=np.quantile(gain,[.025,.975]);return dict(lower95_gain_pp=float(lo),upper95_gain_pp=float(hi),resampling_unit='origin calendar month; all horizon points retained',month_clusters=len(groups),replications=2000,exploratory=True)
def score(root,folder,phase):
    proto=protocol_check(folder);s=read(folder/'results/nlr_selection.json');assert sha(folder/'results/nlr_selection.json')==(folder/'results/nlr_selection.sha256').read_text().strip()
    collect(root,folder,phase);z=np.load(folder/'data/nlr_15min.npz');oo=proto['nlr_origins'][phase];sc=z['scales'];xn=z['x']/sc[:,None,None]
    rows=[];leads=[];origins_rows=[];arrays={};names_series=['facility_total_kw','it_power_kw']
    trainend=int(np.searchsorted(z['dates'],'2023-01-01'));ya=z['y'][:,:trainend];ma=z['valid'][:,:trainend]
    for h in [1,7]:
        _,_,_,P,plain,current=load_bank(root,folder,phase,h);y,m=truth(z['y'],z['valid'],oo,h)
        base=baseline(z['x'],oo,h,'last' if h==1 else'mean3')
        arrays[f'truth_H{h}']=y;arrays[f'valid_H{h}']=m
        for j in range(2):
            c=s['tasks'][str(h)][str(j)];local=np.load(folder/'cache'/f'local_{phase}.npz')
            models=dict(HDSF_Google_frozen=plain[j],R48_Google_frozen=current[j],HDSF_NLR_refit=apply_refit(P[j],c))
            for kind in ['last','week','mean3']:models[kind]=baseline(xn[j:j+1],oo,h,kind)[0]
            if h==1:models.update(LightGBM_NLR=local['nlr_local24_H1'][j],Ridge_NLR=local['nlr_ridge_H1'][j])
            else:models['local7_smoother']=local['local7_H7'][j]
            for k,name in enumerate(c['members']):
                if name not in ['local24','local7']:models['expert_'+name]=P[j,:,:,k]
            pair=ma[j,1:]&ma[j,:-1];diff=ya[j,1:]-ya[j,:-1];masescale=float(abs(diff)[pair].mean());rmsescale=float(np.mean(diff[pair]**2))
            for name,p0 in models.items():
                p=np.maximum(0,p0)[None]*sc[j];yy,mm=y[j:j+1],m[j:j+1]
                a=metric(yy,p,mm,96,base[j:j+1]);a.update(MASE=a['mae']/masescale if masescale>0 else None,RMSSE=a['rmse']/np.sqrt(rmsescale) if rmsescale>0 else None)
                if name in c['intervals_late_DEV']:
                    qlo,qhi=c['intervals_late_DEV'][name];iv,lo,hi=interval(yy,mm,p,np.array([qlo]),np.array([qhi]));a.update({'interval80_'+k:v for k,v in iv.items()})
                    arrays[f'{names_series[j]}_{name}_lower80_H{h}']=lo;arrays[f'{names_series[j]}_{name}_upper80_H{h}']=hi
                a.update(paired_month_ci(yy,mm,p,base[j:j+1],z['dates'][oo]))
                row=dict(dataset='NLR',target=names_series[j],phase=phase,horizon_days=h,model=name,unit='kW',**a);rows.append(row)
                arrays[f'{names_series[j]}_{name}_H{h}']=p
                for lead in range(h):
                    sl=slice(lead*96,(lead+1)*96);score0=metric(yy[:,:,sl],p[:,:,sl],mm[:,:,sl],96)
                    leads.append(dict(target=names_series[j],phase=phase,horizon_days=h,model=name,lead_day=lead+1,**score0))
                for k,o in enumerate(oo):
                    score0=metric(yy[:,k:k+1],p[:,k:k+1],mm[:,k:k+1],96)
                    origins_rows.append(dict(target=names_series[j],phase=phase,horizon_days=h,model=name,origin_date=str(z['dates'][o]),**score0))
    np_save(folder/'results'/f'nlr_{phase}_predictions.npz',**arrays)
    export(folder/'results'/f'NLR_{phase}_全部统一指标.csv',rows);export(folder/'results'/f'NLR_{phase}_逐时距指标.csv',leads);export(folder/'results'/f'NLR_{phase}_逐起报指标.csv',origins_rows)
    save(folder/'results'/f'nlr_{phase}_summary.json',dict(utc=utc(),rows=rows,selection_sha256=sha(folder/'results/nlr_selection.json'),protocol_sha256=sha(folder/'docs/protocol.json'),
        prediction_sha256=sha(folder/'results'/f'nlr_{phase}_predictions.npz'),origin_count=len(oo),independent_target_dataset=True,all_calendar_days_scored=False,
        timezone_verified=False,pretrained_training_overlap_unknown=True))
    print('NLR actual scoring complete',phase,[(r['target'],r['horizon_days'],r['model'],round(r['wape'],5)) for r in rows if r['model'].startswith(('HDSF','R48'))],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--stage',choices=['freeze','score'],required=True);ap.add_argument('--phase',choices=['test','confirm'],default='test');a=ap.parse_args();root=Path(a.root);folder=root/'experiment/generalization_validation_20261012'
    if a.stage=='freeze':freeze(root,folder)
    else:score(root,folder,a.phase)
