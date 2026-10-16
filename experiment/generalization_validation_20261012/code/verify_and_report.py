"""Independent saved-output scoring, source preservation and concise report."""
from pathlib import Path
import argparse,datetime,sys
import numpy as np
from core import *
def main(root):
    folder=root/'experiment/generalization_validation_20261012';q=folder/'results';p=protocol_check(folder);checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)));assert value,name
    check('portable_canonical_metric_parity',read(q/'metric_parity.json')['all_passed'])
    for f,digest in p['source_bindings'].items():check('source_preserved '+f,sha(root/f)==digest)
    for f,digest in p['prepared_sha256'].items():check('prepared_cache_bound '+f,sha(folder/'data'/f)==digest)
    check('original_NLR_raw_preserved',sha(root/'nlr-hpc-pue/raw/esif.influx.buildingData.PUE.combined.parquet')==read(folder/'data/nlr_preparation.json')['source_sha256'])
    check('SeaGuang_original_workbook_preserved',sha(root/'海光寺-能耗能效报表_9.1-9.30电能统计V2.xlsx')==read(folder/'data/sea_preparation.json')['source_sha256'])
    selection=q/'nlr_selection.json';check('NLR_selection_digest_frozen',sha(selection)==selection.with_suffix('.sha256').read_text().strip())
    selected=read(selection)
    check('chronological_protocol_then_selection',p['created_utc']<selected['utc'])
    for h,tasks in selected['tasks'].items():
        for j,c in tasks.items():
            for kind in ['global','halfday']:
                w=np.array(c[kind]);check(f'simplex_H{h}_target{j}_{kind}',bool((w>=0).all() and np.allclose(w.sum(1),1,atol=1e-10)))
            check(f'fusion_DEV_only_H{h}_target{j}',c['selection_last_target_date']<'2024-01-01')
    z=np.load(folder/'data/nlr_15min.npz');allrows=[]
    for phase in ['test','confirm']:
        info=read(q/f'nlr_{phase}_summary.json');check('NLR_'+phase+'_selection_bound',info['selection_sha256']==sha(selection))
        predictions=np.load(q/f'nlr_{phase}_predictions.npz');oo=p['nlr_origins'][phase]
        for f in (folder/'cache').glob(phase+'_*.npz'):
            check('post_freeze_prediction '+f.name,datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat()>=selected['utc'])
        for row in info['rows']:
            j=['facility_total_kw','it_power_kw'].index(row['target']);h=row['horizon_days'];name=row['model'];point=predictions[f'{row["target"]}_{name}_H{h}']
            yy,mm=truth(z['y'][j:j+1],z['valid'][j:j+1],oo,h);base=baseline(z['x'][j:j+1],oo,h,'last' if h==1 else'mean3')
            check(f'finite_nonnegative_{phase}_{j}_{h}_{name}',np.isfinite(point).all() and (point>=0).all())
            sc=metric(yy,point,mm,96,base)
            for key,value in sc.items():
                actual=row.get(key);ok=actual==value if value is None or isinstance(value,(str,int)) else np.isclose(value,actual,rtol=0,atol=1e-10)
                check(f'rescore_{phase}_{j}_{h}_{name}_{key}',ok)
            allrows.append(row)
    # Independently inspect only the history slices used by external adapters.
    from local_models import setup,fields
    setup(root);original=z['x'];future=original.copy();o=p['nlr_origins']['test'][0];future[:,o:]=999999
    for j in range(2):
        a,_=fields(original[j,o-28:o]/z['scales'][j],z['dates'][o]);b,_=fields(future[j,o-28:o]/z['scales'][j],z['dates'][o])
        check(f'external_history_feature_suffix_poison_{j}',np.array_equal(a,b,equal_nan=True))
    check('Cluster_matched_future_poison',read(q/'cluster_summary.json')['all_future_feature_poison_checks_passed'])
    check('Sea_did_not_fabricate_intraday',read(q/'sea_summary.json')['full_HDSF_intraday_not_testable'])
    export(q/'NLR_测试与独立确认全部指标.csv',allrows)
    report=['# HDSF科学泛化验证：实际结果','',
        'Google当前推荐仍为固定第48轮。此次外部结果独立列示，不根据外部测试重新调整参数或改写Google已公布分数。',
        'NLR主目标为设施总功率（五个已测功率分量之和），IT功率为单独的次要目标；不同目标未混合平均。',
        '采用源文件无时区时间坐标下的15分钟和24小时窗口，不声称当地自然日。每月1/15日起报，缺测筛选原因已保存；不是全年逐日起报。',
        '原评价公式已逐项与原Google程序重算对齐。NLR kW/海光寺kWh与Google归一化功率不跨数据集混排。','',
        '|目标/阶段|模型|24小时曲线WAPE|7天曲线WAPE|7天日均WAPE|24小时峰值MAPE|7天峰值MAPE|',
        '|---|---|---:|---:|---:|---:|---:|']
    display={'last':'前一天同刻','week':'上周同刻','mean3':'前三天同刻均值','HDSF_Google_frozen':'HDSF：Google半日权重及局部模型固定迁移',
        'R48_Google_frozen':'第48轮HDSF：Google参数固定＋日均/零均值形状校正迁移','HDSF_NLR_refit':'HDSF：NLR局部LightGBM重新训练＋验证期半日单纯形权重适配'}
    for phase in ['test','confirm']:
        for target in ['facility_total_kw','it_power_kw']:
            for name in display:
                a=next(r for r in allrows if r['phase']==phase and r['target']==target and r['model']==name and r['horizon_days']==1)
                b=next(r for r in allrows if r['phase']==phase and r['target']==target and r['model']==name and r['horizon_days']==7)
                report.append(f'|{target}/{phase}|{display[name]}|{a["wape"]:.6f}%|{b["wape"]:.6f}%|{b["daily_wape"]:.6f}%|{a["peak_mape"]:.6f}%|{b["peak_mape"]:.6f}%|')
    report+=['','固定参数迁移保留Google权重，但存在必要的数据接口适配：单场站联合模式退化为单变量；缺失生产/其他域/静态ID字段走缺失分支；历史分位数缩放只用2022年及以前；物理功率仅非负限制，不套用[0,1]容量上限。少于三个可比场站时50%同组收缩不启用。',
        'NLR重拟合仅更新局部LightGBM/Ridge及验证期组合权重，没有声称在NLR上微调全部神经基础模型。两种路线分别报告。80%区间使用目标场站后段开发残差校准，不是零样本不确定性保证。',
        '收益区间以起报月份为配对重采样块，包含该月所有起报及各时距点；月份数有限，不能把15分钟点视为独立样本。预训练基础模型的原始训练语料重叠情况未知。',
        '', '## 海光寺日电量补充验证','只有30天日电量，当前月此前已暴露；不能支持原96点/672点曲线或日内峰值验证。日尺度组合为Ridge及三个历史基线的单纯形组合，不冒充完整HDSF半日模型。',
        '|任务|模型|WAPE|MAE(kWh)|RMSE(kWh)|','|---|---|---:|---:|---:|']
    sea=read(q/'sea_summary.json')
    for row in sea['rows']:report.append(f'|{row["horizon_days"]}天每日电量|{row["model"]}|{row["wape"]:.6f}%|{row["mae"]:.6f}|{row["rmse"]:.6f}|')
    report+=['','## ClusterData真实算力增益对照','仅cell-a五域有完整真实算力列，另外52域是不可用占位。所有CPU/内存特征只使用起报前历史。功率真值是原Google同月，结果只能证明同源增益，不能证明独立泛化。',
        '|范围|时长|模型|曲线WAPE|日均WAPE|峰值MAPE|','|---|---|---|---:|---:|---:|']
    for row in read(q/'cluster_summary.json')['rows']:report.append(f'|{row["scope"]}|{row["horizon_days"]}天|{row["model"]}|{row["wape"]:.6f}%|{row["daily_wape"]:.6f}%|{row["peak_mape"]:.6f}%|')
    report+=['','全部指标、逐时距和逐起报表均位于本实验results目录。保留所有不改善的模型和真实结果。TimesFM3仍只限非商业研究。']
    (root/'HDSF科学泛化验证_实际结果.md').write_text('\n'.join(report),encoding='utf-8')
    save(q/'final_verification.json',dict(all_passed=True,utc=utc(),check_count=len(checks),checks=checks,actual_external_scoring_completed=True,
        source_data_preserved=True,metrics_unchanged=True,independent_validation_scope='NLR held-out2024 anduntuned2025 only',
        source_current_recommendation_unchanged=True,report_sha256=sha(root/'HDSF科学泛化验证_实际结果.md')))
    print(json.dumps(dict(all_passed=True,checks=len(checks),report=str(root/'HDSF科学泛化验证_实际结果.md')),ensure_ascii=True),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();main(Path(a.root))
