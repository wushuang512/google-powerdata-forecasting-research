"""Recompute actual scores, audit frozen parameters, summarize period robustness."""
from pathlib import Path
import sys,json,hashlib,datetime,csv
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(sys.argv[1]);F=ROOT/'experiment/generalization_extended_20261013';OLD=ROOT/'experiment/generalization_validation_20261012'
sys.path.insert(0,str(OLD/'code'));from core import read,save,sha,utc,metric,interval,baseline,truth,export,parity
proto=read(F/'docs/protocol.json');checks=[];rows=[]
def check(label,value):checks.append(dict(check=label,passed=bool(value)));assert value,label
def equal(a,b):
    if a is None or b is None:return a is b
    if isinstance(a,(str,bool,int)):return a==b
    return bool(np.isclose(a,b,rtol=0,atol=1e-9))
for case,cfg in proto['cases'].items():
    sel=F/'results'/f'{case}_selection.json';check(case+' frozen selection SHA',sha(sel)==sel.with_suffix('.sha256').read_text().strip())
    s=read(sel);check(case+' no test selection',not s['test_scores_used'])
    phases=['test','confirm'] if case=='NLR_2024_2025' else['test'];z=np.load(F/'data'/f'{case}.npz');ppd=int(z['ppd'])
    for phase in phases:
        summary=read(F/'results'/f'{case}_{phase}_summary.json');path=F/'results'/f'{case}_{phase}_predictions.npz';check(case+phase+' prediction SHA',sha(path)==summary['prediction_sha256']);q=np.load(path)
        for r in summary['rows']:
            H=r['horizon_days'];oo=list(map(int,q[f'origins_H{H}']));yy,mm=truth(z['y'],z['valid'],oo,H);check(case+phase+' truth'+str(H),np.array_equal(yy,q[f'truth_H{H}']) and np.array_equal(mm,q[f'valid_H{H}']))
            ids=[list(map(str,z['names'])).index(r['scope'])] if case.startswith('NLR') else list(range(len(z['y'])))
            p=q[f'{r["model"]}_H{H}'];check(case+phase+r['model']+' finite',np.isfinite(p).all() and p.shape==yy.shape and p.min()>=0)
            b=baseline(z['x'],oo,H,'last' if H==1 else'mean3');actual=metric(yy[ids],p[ids],mm[ids],ppd,b[ids])
            for k,v in actual.items():check(case+phase+r['model']+str(H)+k,equal(v,r.get(k)))
            rows.append(r.copy())
for path in (F/'models').glob('*.json'):
    info=read(path);weight=path.with_suffix('.pt') if 'weight_file_sha256' in info else path.with_suffix('.joblib') if path.with_suffix('.joblib').exists() else path.with_suffix('.pt')
    expected=info.get('weight_file_sha256',info.get('model_sha256'))
    if expected and weight.exists():check('model byte SHA '+path.name,sha(weight)==expected)
    for w in info.get('training_windows',[]):
        if 'label_end_exclusive' in w:check('training label bound '+path.name,w['label_end_exclusive']<=info['training_cutoff_exclusive'])
        if 'label_end_fine_exclusive' in w:check('coarse training label bound '+path.name,w['label_end_fine_exclusive']<=w['cutoff_fine'])
    for w in info.get('training_records',info.get('training_origins',[])):
        check('statistical training label bound '+path.name,w['label_end_exclusive']<=info.get('cutoff',info.get('training_end_exclusive',10**9)))
    if 'real_optimizer_steps' in info:check('finite real neural updates '+path.name,info['real_optimizer_steps'] in ([200] if 'SparseMoE' in path.name else [16,48]))
    if 'real_optimizer_updates' in info:check('finite real coarse neural updates '+path.name,info['real_optimizer_updates'] in [16,48])

# Recompute old adopted controls on exactly the same Google masks and origins.
reference=read(F/'results/Google_archived_controls.json');q=np.load(F/'results/Google_archived_controls.npz');z=np.load(F/'data/Google2019.npz')
for r in reference['records']:
    H=r['horizon_days'];oo=list(range(22,31)) if H==1 else[22,23,24];yy,mm=truth(z['y'],z['valid'],oo,H);p=q[f'{r["model"]}_H{H}'];b=baseline(z['x'],oo,H,'last' if H==1 else'mean3')
    stats=metric(yy,p,mm,96,b);row=dict(case='Google2019',phase='test',scope='all57',horizon_days=H,model=r['model'],model_name=r['model_name'],DEV_selected=False,previous_user_adopted=r.get('previous_user_adopted',False),unit='normalized power',**stats)
    lower=f'{r["model"]}_lower80_H{H}';upper=f'{r["model"]}_upper80_H{H}'
    if lower in q.files:
        lo,hi=q[lower],q[upper];width=hi-lo;row.update(interval80_nominal_percent=80,interval80_coverage_percent=float(100*((yy>=lo)&(yy<=hi))[mm].mean()),interval80_mean_width=float(width[mm].mean()),interval80_winkler80=float((width+10*np.maximum(lo-yy,0)+10*np.maximum(yy-hi,0))[mm].mean()),interval80_below_percent=float(100*(yy<lo)[mm].mean()),interval80_above_percent=float(100*(yy>hi)[mm].mean()))
    rows.append(row)
cluster=read(F/'results/Cluster_summary.json');check('complete real cell-a usage',cluster['source_files']==12085 and len(cluster['covered_domains'])==5);check('unavailable52 unchanged',cluster['uncovered52_predictions_equal']);check('actual CPU future poison',cluster['future_poison_checks_passed']);rows+=cluster['rows']

FAMILY_NAMES={'chronos2':'Amazon Chronos-2','tirex2':'NX-AI TiRex-2','timesfm25':'Google TimesFM 2.5 200M','timesfm3':'Google TimesFM3.0','toto2':'Datadog Toto2.0 313M','patchtst':'IBM Granite PatchTST-FM-r2','moirai2':'Salesforce Moirai2.0 R Small 15分钟单变量','ttm':'IBM TinyTimeMixer R2 512历史 720输出','ttm1024':'IBM TinyTimeMixer R2 1024历史 720输出','moirai_rate4':'Salesforce Moirai2.0 R Small 小时同场站联合及15分钟曲线回填','moirai_rate12':'Salesforce Moirai2.0 R Small 三小时同场站联合及15分钟曲线回填'}
METHOD_NAMES={'baseline_last':'前一天同刻或前一日电量','baseline_week':'上周同刻或上周同日电量','baseline_mean3':'前三天同刻或前三日均值','Ridge_direct':'Ridge历史多尺度与日历直接残差预测','LightGBM_direct':'LightGBM历史多尺度与日历直接残差预测','HGB_residual':'HistGradientBoosting 7叶直接残差预测','SparseMoE':'多尺度统计特征四专家Top2稀疏MoE','JointFactorFourier':'训练期PCA共同因子与Ridge傅里叶直接预测','Chronos2_TiRex2_equal':'Chronos-2与TiRex-2各50%固定组合','Chronos2_TiRex2_TimesFM3_25_25_50':'Chronos-2 25% TiRex-2 25% TimesFM3 50%固定组合','Chronos2_TiRex2_TimesFM3_25_25_50_target_ridge':'Chronos-2 TiRex-2 TimesFM3固定组合加目标Ridge残差校准','Chronos2_TiRex2_TimesFM3_25_25_50_target_hgb':'Chronos-2 TiRex-2 TimesFM3固定组合加目标HGB残差校准','TimesFM3_RidgeFourier_residual':'TimesFM3加目标Ridge傅里叶残差校准','Simplex_global':'扩展专家库全局单纯形目标重拟合','HDSF_extended_halfday':'HDSF扩展专家库12小时分段单纯形目标重拟合','HDSF_extended_overlap6h':'HDSF扩展专家库6小时平滑重叠单纯形目标重拟合','HDSF_extended_overlap12h':'HDSF扩展专家库12小时平滑重叠单纯形目标重拟合','HDSF_extended_overlap24h':'HDSF扩展专家库24小时平滑重叠单纯形目标重拟合','HDSF_original_source':'HDSF半日单纯形既有Google权重固定迁移','R48_fixed_source':'第48轮HDSF加原Google底层微调校正固定迁移','portfolio_DEV_selected':'开发期冻结的目标重拟合方案','portfolio_CPU_DEV_frozen':'开发期冻结方案加历史CPU内存差分校正','CPU_L1':'LightGBM历史功率CPU内存 L1','CPU_L1_power_matched':'LightGBM仅历史功率 L1同条件控制','CPU_monotone_L2':'LightGBM历史功率CPU内存 条件单调平方损失','CPU_monotone_L2_power_matched':'LightGBM仅历史功率 平方损失同条件控制'}
for r in rows:
    if 'model_name' not in r:
        name=r['model']
        if name.startswith('raw_') and name.endswith('_target_ridge'):
            f=name[4:-len('_target_ridge')];r['model_name']=FAMILY_NAMES[f]+' 日尺度目标Ridge残差拟合'
        elif name.startswith('raw_'):r['model_name']=FAMILY_NAMES[name[4:]]+' 原权重推理'
        elif name.startswith('target_'):
            f=name[7:];sel=read(F/'results'/f'{r["case"]}_selection.json')['tasks'][str(r['horizon_days'])];label=sel['checkpoints'][f]
            r['model_name']=FAMILY_NAMES[f]+(' 开发筛选保留原权重' if label=='s0' else f' 目标参数更新{label[1:]}步')
        else:r['model_name']=METHOD_NAMES.get(name,name)
export(F/'results/全部数据集与方法_统一指标完整结果.csv',rows)

# Descriptive external stability is computed within one facility; two nested
# targets are not counted as two independent external facilities.
external=[r for r in rows if r['case'].startswith('NLR') and r['scope']=='facility_total_kw'];stability=[]
for model in sorted(set(r['model'] for r in external)):
    rr=[r for r in external if r['model']==model];curve=[r['wape']/r['baseline_wape'] for r in rr];daily=[r['daily_wape']/r['baseline_daily_wape'] for r in rr if r['horizon_days']==7]
    if len(curve)!=6:continue
    stability.append(dict(model=model,model_name=rr[0]['model_name'],external_facilities=1,periods=3,curve_tasks=6,
        geometric_mean_curve_error_ratio=float(np.exp(np.mean(np.log(curve)))),worst_period_curve_error_ratio=max(curve),
        curve_period_tasks_better_than_fixed_baseline=sum(v<1 for v in curve),
        geometric_mean_7day_daily_error_ratio=float(np.exp(np.mean(np.log(daily)))),
        fixed_weights_transfer=model.startswith('raw_') or model in ['HDSF_original_source','R48_fixed_source']))
stability.sort(key=lambda r:(r['geometric_mean_curve_error_ratio'],r['worst_period_curve_error_ratio']))
export(F/'results/NLR三时期_曲线精度与稳定性比较.csv',stability)

diagnostics=[]
for case,phase in [('NLR_2021','test'),('NLR_2024_2025','test'),('NLR_2024_2025','confirm')]:
    q=np.load(F/'results'/f'{case}_{phase}_predictions.npz');z=np.load(F/'data'/f'{case}.npz');sel=read(F/'results'/f'{case}_selection.json')
    for H in [1,7]:
        name=sel['tasks'][str(H)]['selected_method'];oo=q[f'origins_H{H}'].astype(int);yy=q[f'truth_H{H}'][0];mm=q[f'valid_H{H}'][0];p=q[f'{name}_H{H}'][0];b=baseline(z['x'],oo,H,'last' if H==1 else'mean3')[0]
        numerator=((abs(yy-b)-abs(yy-p))*mm).sum(1);denominator=(abs(yy)*mm).sum(1);origin_dates=z['dates'][oo].astype('datetime64[D]');start=origin_dates.min().astype('datetime64[Y]').astype('datetime64[D]');end=(start.astype('datetime64[Y]')+1).astype('datetime64[D]')
        for length in [7,14,28]:
            buckets=((origin_dates-start).astype(int)//length);count=int(np.ceil((end-start).astype(int)/length));num=np.bincount(buckets,weights=numerator,minlength=count);den=np.bincount(buckets,weights=denominator,minlength=count);rng=np.random.default_rng(20261013);gains=[]
            for _ in range(1000):
                sampled=rng.integers(count,size=count);d=den[sampled].sum()
                if d>0:gains.append(100*num[sampled].sum()/d)
            lo,hi=np.quantile(gains,[.025,.975]);diagnostics.append(dict(case=case,phase=phase,horizon_days=H,method=name,calendar_block_days=length,
                observed_gain_pp=float(100*numerator.sum()/denominator.sum()),exploratory_lower95_quantile_gain_pp=float(lo),exploratory_upper95_quantile_gain_pp=float(hi),
                block_count=count,nonempty_blocks=int((den>0).sum()),replications=1000,calibrated_confidence_guarantee=False,
                resampling='Whole origin forecast trajectories in aligned nonoverlapping calendar blocks; empty calendar blocks retained; between-block dependence not proven absent'))
export(F/'results/NLR冻结方案_时间块敏感性诊断.csv',diagnostics)
parity(ROOT,F/'results/9月28日原评价方法数值一致性核对.json')
check('original metric numeric parity',read(F/'results/9月28日原评价方法数值一致性核对.json')['all_passed'])
summary=dict(all_passed=True,created_utc=utc(),all_metric_rows=len(rows),rows=rows,external_stability=stability,
    best_observed_external_curve_family=stability[0] if stability else None,external_facilities=1,
    temporal_evaluation_periods=[2021,2024,2025],source_old_tests_exposed=True,strict_zero_shot_not_established=True,
    scientific_conclusion='Expanded target-fit and conditional-transfer comparison with chronological boundaries and unchanged metrics; exploratory, not broad independent multi-facility confirmation',
    source_selection_sha256={c:sha(F/'results'/f'{c}_selection.json') for c in proto['cases']})
save(F/'results/summary.json',summary);save(F/'results/final_verification.json',dict(all_passed=True,checks=checks,count=len(checks),utc=utc(),metrics_recomputed_from_actual_predictions=True))
print(json.dumps(dict(all_passed=True,rows=len(rows),checks=len(checks),best_external_observed=summary['best_observed_external_curve_family']),ensure_ascii=True),flush=True)
