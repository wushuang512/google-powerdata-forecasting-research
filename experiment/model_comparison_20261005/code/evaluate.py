"""Read frozen historical forecasts; no fit, hyperparameter selection or test reselection."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np,pandas as pd
TOP=Path(sys.argv[1]).resolve(); B=TOP/'experiment'; O=B/'model_comparison_20261005'
O.mkdir(exist_ok=True);(O/'results').mkdir(exist_ok=True);(O/'figures').mkdir(exist_ok=True)
sys.path.insert(0,str(B/'code'));import common
D=common.load();Y,M=common.truth(D,list(range(22,31)));BEST=np.load(B/'round8b/results/test_predictions.npz')['selected']
SOURCES={};CHECKS={}; CACHE={}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def readnp(rel,key):
    p=B/rel
    if str(p) not in CACHE:
        with np.load(p,allow_pickle=False) as z:CACHE[str(p)]={k:z[k] for k in z.files}
        SOURCES[str(p.relative_to(TOP))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return CACHE[str(p)][key]
def metric(y,p,m):
    a=common.metrics(y,p,m);e=p-y;n=int(m.sum());den=(abs(y)*m).sum()
    a.update(pooled_wape=float(100*(abs(e)*m).sum()/den),rmse=float(np.sqrt((e[m]**2).mean())),bias_percent=float(100*(e*m).sum()/den))
    yp=np.where(m,y,-np.inf).max(-1);pp=np.where(m,p,-np.inf).max(-1);dv=m.any(-1)
    a.update(peak_underestimate_frequency_percent=float(100*((pp<yp)&dv).sum()/dv.sum()),
        peak_underestimate_mean_percent=float(100*(np.maximum(yp[dv]-pp[dv],0)/yp[dv]).mean()),
        ramp_mae=float(abs(np.diff(p,axis=-1)-np.diff(y,axis=-1))[m[...,1:]&m[...,:-1]].mean()))
    per=100*(abs(e)*m).sum((1,2))/(abs(y)*m).sum((1,2))
    day=[common.metrics(y[:,j:j+1],p[:,j:j+1],m[:,j:j+1])['wape'] for j in range(y.shape[1])]
    a.update(domain_wape_median=float(np.median(per)),domain_wape_p90=float(np.percentile(per,90)),domain_wape_max=float(per.max()),
        daily_wape_std=float(np.std(day,ddof=1)),worst_forecast_day_wape=float(max(day)))
    return a
def paired(p):
    err=np.sum(abs(Y-p)*M,-1);be=np.sum(abs(Y-BEST)*M,-1);den=np.sum(abs(Y)*M,-1)
    rng=np.random.default_rng(20261005);idx=rng.integers(0,9,(5000,9));w=np.stack([np.bincount(row,minlength=9) for row in idx])
    scores=100*((err@w.T)/(den@w.T)).mean(0);best=100*((be@w.T)/(den@w.T)).mean(0)
    gain=best-scores
    return {'candidate_minus_best_pp':float(common.metrics(Y,p,M)['wape']-common.metrics(Y,BEST,M)['wape']),
        'best_gain_over_candidate_day_bootstrap95_pp':(-np.percentile(gain,[97.5,2.5])).tolist(),
        'best_wins_domains':int((err.sum(1)/den.sum(1)>be.sum(1)/den.sum(1)+1e-12).sum()),
        'ties_domains':int(np.isclose(err.sum(1)/den.sum(1),be.sum(1)/den.sum(1),rtol=0,atol=1e-12).sum()),
        'best_wins_dates':int(((100*err/den).mean(0)>(100*be/den).mean(0)+1e-10).sum()),
        'bootstrap_replicates':5000,'seed':20261005}
REG=[
 ('best','当前最佳 R8b','round8b/results/test_predictions.npz','selected','当前正式部署'),
 ('r5','第五轮融合组合','round5/results/test_predictions.npz','selected','此前正式方案'),
 ('r4','第四轮冻结组合','round4/results/test_predictions.npz','round4_selected','此前正式方案'),
 ('r3','第三轮主组合','round3/results/test_predictions.npz','new_curve','此前正式方案'),
 ('r2curve','第二轮曲线方案','round2/results/test24_predictions.npz','new_curve','此前正式方案'),
 ('r2peak','第二轮峰值方案','round2/results/test24_predictions.npz','new_peak','峰值侧重对照'),
 ('equal','Chronos 与 TiRex 等权组合','round2/results/test24_predictions.npz','old_equal_mix','初始强组合'),
 ('tirex','TiRex 2','round2/results/test24_predictions.npz','tirex_uni','单模型对照'),
 ('chronos','Chronos 2','round2/results/test24_predictions.npz','zero_uni','单模型对照'),
 ('chronoscov','Chronos 2 历史协变量','round2/results/test24_predictions.npz','zero_cov','单模型对照'),
 ('tf25','TimesFM 2.5','round4/results/test_predictions.npz','tf25_c1344_flip1','单模型对照'),
 ('tf3','TimesFM 3','round3/results/tf3_test.npz','tf3_uni_context2016','单模型对照'),
 ('ridge','Ridge','results/test24_predictions.npz','ridge_1000','传统模型对照'),
 ('lgb','LightGBM','results/test24_predictions.npz','lgb_p0_l15_n200','传统模型对照'),
 ('last','前一天同刻','results/test24_predictions.npz','last','强简单基线'),
 ('r13candidate','第十三轮未采用候选','round13/results/test_predictions.npz','candidate','负面或近似持平证据'),
 ('r9fusion','完整 CPU 稀疏 MoE','round9/results/test_predictions.npz','fusion_cpu_e10','未采用单独融合网络'),
 ('r14candidate','第十四轮直接网络','round14/results/test_predictions.npz','candidate','明显较差的未采用网络'),
]
preds={};rows=[];byday=[];bydomain=[];bycell=[];byhorizon=[];summary={}
for ident,name,rel,key,status in REG:
    p=readnp(rel,key).astype('float64');assert p.shape==Y.shape and np.isfinite(p).all(),(ident,p.shape)
    preds[ident]=p
    a=metric(Y,p,M);a.update(model=ident,name=name,status=status,source=rel,array=key,**paired(p))
    a['cella5_wape']=common.metrics(Y[D['cells']=='a'],p[D['cells']=='a'],M[D['cells']=='a'])['wape']
    a['other52_wape']=common.metrics(Y[D['cells']!='a'],p[D['cells']!='a'],M[D['cells']!='a'])['wape']
    a['bounded_0_1']=bool(((p>=0)&(p<=1)).all());rows.append({k:v for k,v in a.items() if not isinstance(v,list)})
    summary[ident]=a
    for j in range(9):byday.append(dict(model=ident,name=name,forecast_date=f'2019-05-{j+23:02}',**common.metrics(Y[:,j:j+1],p[:,j:j+1],M[:,j:j+1])))
    for i in range(57):bydomain.append(dict(model=ident,name=name,domain=str(D['names'][i]),cell=str(D['cells'][i]),**common.metrics(Y[i:i+1],p[i:i+1],M[i:i+1])))
    for cell in sorted(set(D['cells'])):
        ix=D['cells']==cell;bycell.append(dict(model=ident,name=name,cell=str(cell),domains=int(ix.sum()),**common.metrics(Y[ix],p[ix],M[ix])))
    for start in [0,24,48,72]:
        mm=M.copy();mm[...,:start]=False;mm[...,start+24:]=False
        byhorizon.append(dict(model=ident,name=name,forecast_hours=f'{start/4:g}至{start/4+6:g}',wape=common.metrics(Y,p,mm)['wape']))
CHECKS.update(shape_all_same=True,truth_valid_points=int(M.sum())==49243,names_unique=len(set(D['names']))==57,
    best_matches_original_summary=abs(summary['best']['wape']-json.loads((B/'round8b/results/summary.json').read_text())['scores']['selected']['wape'])<1e-12,
    persistence_exact_original=np.array_equal(preds['last'],common.baseline(D,list(range(22,31)),kind='last')),
    macro_not_pooled=True,no_training=True,no_test_reselection=True)
duplicates={}
for r in ['round6','round7','round8','round9','round11','round12','round13','round14']:
    p=readnp(f'{r}/results/test_predictions.npz','selected');duplicates[r]=bool(np.array_equal(p,BEST))
# Evaluate all historical arrays for screening and preserve a compact catalogue.
catalog=[]
for rd in [B]+sorted(B.glob('round*')):
    files=list((rd/'results').glob('test*predictions.npz'))
    if rd.name=='round10':files=list((rd/'results/repair1').glob('test*predictions.npz'))
    for f in files:
        with np.load(f,allow_pickle=False) as z:
            for k in z.files:
                p=z[k]
                if p.shape!=Y.shape or k.startswith('delta') or not np.isfinite(p).all():continue
                catalog.append(dict(round=rd.name,array=k,source=str(f.relative_to(B)),**common.metrics(Y,p,M)))
# Seven-day comparison is separate and has overlapping windows.
seven=[];y7,m7=common.truth(D,[22,23,24],7)
for rd,fn in [(B,'test7_predictions.npz'),(B/'round2','test7_predictions.npz'),(B/'round4','test7.npz')]:
    f=rd/'results'/fn
    with np.load(f) as z:
        for k in z.files:
            if k not in ['last','week','ridge_1000','lgb_p0_l15_n200','chronos_zero','zero_uni','tirex_uni','old_equal_mix','new_curve','new_peak','best_mix','ensemble7']:continue
            p=z[k];assert p.shape==y7.shape
            seven.append(dict(round=rd.name,model=k,source=str(f.relative_to(B)),**common.metrics(y7,p,m7)))
# Existing intervals only: recenter R5's frozen residual width at R8b, without any test calibration.
intervals=[]
for label,center,lower,upper in [
    ('当前最佳 R8b',preds['best'],readnp('round9/results/selected_intervals_test.npz','lower'),readnp('round9/results/selected_intervals_test.npz','upper')),
    ('第一轮正式方案',readnp('results/test24_predictions.npz',json.loads((B/'results/selection24.json').read_text())['primary24']),readnp('results/interval1_primary.npz','lower'),readnp('results/interval1_primary.npz','upper'))]:
    assert lower.shape==Y.shape and (lower<=upper).all()
    intervals.append(dict(model=label,nominal_percent=80,coverage_percent=float(100*((Y>=lower)&(Y<=upper))[M].mean()),mean_width_util=float((upper-lower)[M].mean()),source='现有开发残差经验区间 不是重新拟合'))
for fn,rr in [('overall_metrics.csv',rows),('per_day.csv',byday),('per_domain.csv',bydomain),('per_cell.csv',bycell),('forecast_horizon.csv',byhorizon),('all_historical_scores.csv',catalog),('seven_day_comparison.csv',seven),('interval_coverage.csv',intervals)]:
    pd.DataFrame(rr).to_csv(O/'results'/fn,index=False,encoding='utf-8-sig')
out=dict(created_utc=utc(),title='Google PowerData 2019 多维度模型评估与文件整理',models=summary,intervals=intervals,duplicates=duplicates,
    seven_day=seven,checks=CHECKS,all_passed=bool(all(CHECKS.values())),sources=SOURCES,
    protocol=dict(test_origins=list(range(22,31)),forecast_dates=['2019-05-23','2019-05-31'],domains=57,valid_points=49243,horizon=96,
        primary='Mean of 57 PDU WAPEs; each PDU equally weighted',test_repeatedly_seen=True,independent_confirmation=False,
        bootstrap='5000 common date block resamples; 57 PDU retained together; exploratory pointwise intervals, no multiple-comparison correction',
        no_new_model_selection=True,seven_day_origins=[22,23,24],seven_day_overlap=True))
(O/'results/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
assert out['all_passed']
print(json.dumps({i:{k:v for k,v in a.items() if k in ['name','wape','daily_wape','peak_mape','p95_ape','cella5_wape','best_gain_over_candidate_day_bootstrap95_pp']} for i,a in summary.items()},ensure_ascii=False,indent=2))
print('DUPLICATES',duplicates);print('INTERVALS',intervals)
