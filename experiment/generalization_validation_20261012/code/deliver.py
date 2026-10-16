"""Presentation and additional independent input-cache audit; never reselects."""
from pathlib import Path
import argparse,datetime,sys
import numpy as np
from core import *
def main(root):
    folder=root/'experiment/generalization_validation_20261012';q=folder/'results';p=protocol_check(folder)
    assert read(q/'final_verification.json')['all_passed'] and read(q/'pipeline_complete.json')['all_passed']
    names={'last':'前一天同刻（海光寺为前一日电量）','week':'上周同刻（海光寺为上周同日电量）','mean3':'前三天同刻均值（海光寺为前三日电量均值）',
        'HDSF_Google_frozen':'HDSF半日单纯形：Google原组合参数固定迁移',
        'R48_Google_frozen':'第48轮HDSF：24小时PatchTST-FM日均残差／7天TimesFM3、Toto2、PatchTST-FM零均值形状校正，Google参数固定迁移',
        'HDSF_NLR_refit':'HDSF场站适配：NLR局部LightGBM训练＋全局/半日单纯形验证选择（神经权重固定）',
        'LightGBM_NLR':'NLR局部LightGBM：四随机种子、350棵树、原64特征词表＋训练期滞后混合',
        'Ridge_NLR':'NLR Ridge：原64特征词表、训练期插补与标准化、固定正则强度10',
        'local7_smoother':'24天历史水平/形状指数平滑：水平半衰期3/5天，形状半衰期10天',
        'expert_chronos2_long':'Amazon Chronos-2：28天历史、单场站',
        'expert_tirex2':'NX-AI TiRex-2：14天历史、CPU原生推理',
        'expert_timesfm25':'Google TimesFM 2.5 200M：14天历史、单场站',
        'expert_timesfm30':'Google TimesFM 3.0：14天历史；单场站联合模式退化为单变量',
        'expert_timesfm30_long':'Google TimesFM 3.0：21天历史；单场站联合模式退化为单变量',
        'expert_timesfm30_uni':'Google TimesFM 3.0：21天历史、原单变量模式（与上一行同一预测，不能算独立专家）',
        'expert_toto2_joint':'Datadog Toto2 313M：14天历史；单场站同组模式退化为单变量',
        'expert_toto2_single':'Datadog Toto2 313M：14天历史、原单变量模式（单场站时与同组模式相同）',
        'expert_patchtstfm_long':'IBM Granite PatchTST-FM-r2：21天历史、原预训练权重',
        'Ridge_daily':'海光寺日尺度Ridge：1/2/3/7日滞后及近期均值、固定正则强度10',
        'daily_simplex':'海光寺日尺度单纯形组合：验证期选择后回退到前一日电量，非完整HDSF半日模型',
        'R48_fixed':'Google第48轮HDSF固定当前推荐',
        'R48_CPU_DEV_frozen':'Google第48轮HDSF＋历史CPU/内存LightGBM差分校正：开发期冻结权重25%，未采纳',
        'LightGBM_power_only':'LightGBM同条件对照：仅历史功率，7叶/120树/正则10',
        'LightGBM_power_CPU_memory':'LightGBM算电对照：历史功率＋历史CPU(NCU)/归一化内存，7叶/120树/正则10'}
    rows=[]
    for file in ['nlr_test_summary.json','nlr_confirm_summary.json','sea_summary.json','cluster_summary.json']:
        for original in read(q/file)['rows']:
            row=dict(original);row['model_name']=names[row['model']];rows.append(row)
    export(q/'泛化验证_全部统一指标_模型全名.csv',rows);save(q/'model_names.json',names)
    z=np.load(folder/'data/nlr_15min.npz');xn=z['x']/z['scales'][:,None,None];input_checks=[];days={'chronos2_long':28,'timesfm30_long':21,'timesfm30_uni':21,'patchtstfm_long':21}
    for f in (folder/'cache').glob('*.npz'):
        if f.name.startswith('local_'):continue
        record=np.load(f);o=int(record['origin']);member=str(record['member']);d=days.get(member,14)
        context=np.ascontiguousarray(xn[:,o-d:o].reshape(2,-1),dtype=np.float32)
        good=str(record['input_digest'])==hashlib.sha256(context.tobytes()).hexdigest() and str(record['protocol'])==sha(folder/'docs/protocol.json')
        assert good,f.name;input_checks.append(dict(file=f.name,passed=True,last_input_date=str(z['dates'][o-1]),origin_date=str(z['dates'][o])))
    save(q/'independent_input_cache_audit.json',dict(all_passed=True,utc=utc(),cache_count=len(input_checks),checks=input_checks,all_inputs_strictly_before_origin=True))
    s=read(q/'nlr_selection.json');nlr=[r for r in rows if r['dataset']=='NLR'];lines=['# HDSF科学泛化验证：结论与全部指标','',
        '## 先看结论','',
        '1. 第48轮HDSF具有部分外部迁移能力，但本轮没有证明其在所有场景、所有指标上最优。Google当前推荐参数保持原冻结记录，没有用外部测试名次重新选模型。',
        '2. NLR设施总功率：固定参数迁移在2024/2025的24小时曲线均优于前一天基线；7天曲线在2024优于前三天均值，在2025略差，外部性能存在任务与时间段差异。',
        '3. NLR场站适配方案的7天曲线点估计在两期均改善；2025相对前三天均值的收益区间跨零，峰值未全面领先。24小时适配方案没有超过固定迁移。',
        '4. 海光寺只有30天日电量；日尺度组合经开发期选择回退至前一日电量，未得到新增收益。不能用它验证原96/672点曲线和日内峰值。',
        '5. ClusterData CPU/内存帮助了单独LightGBM，但其差分叠加到第48轮HDSF后曲线与日均退步。保留结果，不采用该修正。它仍是Google同月同源消融，不是独立场站泛化。','',
        '## NLR设施总功率核心结果（百分数，越低越好）','',
        '|时期|具体方案|24小时曲线WAPE|7天曲线WAPE|7天日均WAPE|24小时峰值MAPE|7天峰值MAPE|',
        '|---|---|---:|---:|---:|---:|---:|']
    short={'last':'前一天同刻','mean3':'前三天同刻均值','HDSF_Google_frozen':'HDSF原版：Google固定参数迁移',
        'R48_Google_frozen':'第48轮HDSF：Google固定参数迁移','HDSF_NLR_refit':'NLR局部LightGBM＋验证期单纯形权重适配'}
    for phase in ['test','confirm']:
        for name,display in short.items():
            a=next(r for r in nlr if r['phase']==phase and r['target']=='facility_total_kw' and r['horizon_days']==1 and r['model']==name)
            b=next(r for r in nlr if r['phase']==phase and r['target']=='facility_total_kw' and r['horizon_days']==7 and r['model']==name)
            lines.append(f'|{"2024测试" if phase=="test" else"2025确认"}|{display}|{a["wape"]:.4f}|{b["wape"]:.4f}|{b["daily_wape"]:.4f}|{a["peak_mape"]:.4f}|{b["peak_mape"]:.4f}|')
    lines+=['','NLR覆盖为固定日历规则下筛选后的16个2024起报、13个2025起报，分别来自9/7个有效月份；不是全年逐日结果。起报剔除仅依据历史数据覆盖率，不按预测误差或模型表现选择。',
        '24小时任务的原对应基线为前一天同刻；7天任务为前三天同刻均值。表中两种基线均列出，不能只选择更弱基线宣传。','',
        '第48轮固定迁移设施总功率曲线WAPE相对原对应基线的误差降低：2024为19.2871%/16.5682%，2025为22.2343%/-4.5142%（24小时/7天）。',
        '2025第48轮7天曲线相对前三天均值的收益95%探索区间为[-0.901322,+0.256375]个百分点；场站适配版为[-0.083477,+0.410108]个百分点。两者都跨零，不能声称独立显著改善。',
        '80%区间也存在覆盖不足：2024第48轮7天覆盖率69.3223%，适配版58.9282%，不能宣称可靠的80%概率保证。','',
        '单模型实际测试曲线也可能领先：2024设施24小时Google TimesFM3（14天历史）为3.142708%，7天TimesFM3（21天）为8.026728%；2025设施24小时TimesFM3（21天）为3.093019%，7天Datadog Toto2为4.014587%。这些为预登记对照的已观测名次，未据此重新部署或选参。','',
        '## 具体模型及选择','',
        '固定迁移的24小时方案：HDSF半日单纯形＋PatchTST-FM日均残差；7天方案：HDSF半日单纯形＋TimesFM3/Toto2/PatchTST-FM零均值形状校正。神经权重和Google原组合权重均不重新拟合。',
        '单场站没有57个并行PDU；原联合模式按单场站通道执行，不引入虚构域。两个目标（设施/IT）分别推理、分别评价。少于三个可比场站时50%同组收缩不启用。',
        '场站适配没有声称重新训练全部神经权重：只实际重新训练NLR局部LightGBM/Ridge，并拟合、验证选择单纯形权重。以下是测试前已冻结的半日分支比例：']
    for h,tasks in s['tasks'].items():
        for j,c in tasks.items():lines.append(f'- {"设施总功率" if j=="0" else"IT功率"}，{h}天：半日分支α={c["alpha"]}；α=0表示采用全局单纯形。')
    lines+=['','## 数据处理与科学边界','',
        '- NLR原Parquet不改动。设施功率为IT、制冷、HVAC、插座照明、泵五项之和，不额外重复加2.67kW。按源文件无时区时间坐标聚合15分钟均值：每格至少12个有效样本、观测跨度至少600秒、格内间隔不超过180秒。没有确认当地时区，不声称当地自然日。',
        '- 功率组成需有限且非负、IT为正。缺失真值保持无效；模型输入只向前延续历史，不使用未来插值或回填。起报前28天每目标有效覆盖至少95%，最后输入格有效。',
        '- 仅使用2022年及以前的95%分位数进行量纲缩放，预测反变换为kW；不套用Google归一化[0,1]物理容量上限。原局部模型缺失生产、静态ID及其他序列特征走缺失路径，没有伪造算力或容量。',
        '- 训练2016—2022；2023验证期前半年拟合权重、后半年选比例并校准区间；2024测试及2025确认均不参与参数选择。局部模型测试前可按同一冻结配方重拟合至2023年底，2025继续用同一模型，没有根据2024误差再训练。',
        '- WAPE继续同一序列有效点误差绝对和除以真值绝对和，再按真实同类域数等权平均；本数据每个目标N=1。Google保持57域原口径。每日平均负荷主评价只采用完整96点。峰值仍用原有效格/首次最大值口径。零真值APE为不可定义并计数。',
        '- 所有模型同一起报、同一真值掩码、同一评价公式。MAE/RMSE、P95APE、2%/5%命中率、绝对偏差、峰值幅度/时间/低估率及80%区间覆盖/宽度/Winkler均输出；MASE/RMSSE仅作补充，不替换原指标。',
        '- 收益区间以起报月份为配对重采样块（2000次），同月各起报及完整各时距一起保留；不能把所有15分钟点当独立样本。月份有限，区间只作探索性证据。基础模型预训练语料与目标数据的重叠无法核实。',
        '- 海光寺只取每日汇总行，另外两条分项不作为独立样本；不生成虚假的日内功率。ClusterData只用官方cell-a12085个Parquet必要列及五域真实历史CPU/内存，52域缺失占位不作真实零算力。',
        '- 环境恢复只切换到此前已验证的CPU/GPU解释器，冻结的科学计算代码和参数不变。未设置3GiB启动门槛，没有关闭用户程序；本轮没有新增权重或数据下载。所有失败和负面结果保留。TimesFM3仍限非商业研究。','',
        '## 全部指标文件','',
        '- experiment/generalization_validation_20261012/results/泛化验证_全部统一指标_模型全名.csv：各数据、目标、任务、模型全部指标，含明确中文全名。',
        '- NLR_test/confirm_逐时距指标.csv及逐起报指标.csv：D1—D7及各起报拆分。',
        '- 预测NPZ包含所有对照与固定推荐预测、有效掩码和经验区间；data目录含建模数组与质量覆盖表；models目录含实际新训练的局部模型。',
        '- final_verification.json：3366项独立核对通过；independent_input_cache_audit.json：所有神经输入缓存的历史切片摘要重新核对。']
    report=root/'HDSF科学泛化验证_结论与指标.md';report.write_text('\n'.join(lines),encoding='utf-8')
    # Scientific figure: fixed comparisons, no smoothing or invented values.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,3,figsize=(13,7.5));models=['baseline','HDSF_Google_frozen','R48_Google_frozen','HDSF_NLR_refit'];labels=['原对应\n历史基线','HDSF原版\n固定迁移','第48轮HDSF\n固定迁移','NLR权重\n适配']
    for i,phase in enumerate(['test','confirm']):
        for k,(h,key,title) in enumerate([(1,'wape','24小时曲线WAPE'),(7,'wape','7天曲线WAPE'),(7,'daily_wape','7天每日平均负荷WAPE')]):
            values=[]
            for model in models:
                name=('last' if h==1 else'mean3') if model=='baseline' else model
                values.append(next(r[key] for r in nlr if r['phase']==phase and r['target']=='facility_total_kw' and r['horizon_days']==h and r['model']==name))
            ax=axes[i,k];bars=ax.bar(np.arange(4),values,color=['#8795a5','#e5ad65','#397abc','#5aa394']);ax.set_xticks(np.arange(4),labels,fontsize=9);ax.set_ylabel('误差(%)，越低越好');ax.set_title(('2024测试' if phase=='test' else'2025确认')+' · '+title,fontsize=11);ax.set_ylim(0,max(values)*1.23);ax.spines[['top','right']].set_visible(False)
            for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+max(values)*.025,f'{value:.4f}',ha='center',fontsize=9)
    fig.suptitle('NLR设施总功率：冻结参数后的外部验证',fontsize=16);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(q/'NLR设施总功率_三项任务比较.png',dpi=160);plt.close(fig)
    progress=root/'当前进展.md';progress.write_text('# 当前进展\n\n2026年10月12日，科学泛化验证已实际完成：NLR2024测试及2025确认、海光寺日电量补充、ClusterData同源算力消融。固定第48轮HDSF仍为Google当前推荐，未根据外部测试修改其参数。\n\nNLR24小时曲线有持续基线收益，7天固定迁移在2025未超过原对应基线；NLR场站适配7天点估计改善但收益区间跨零。不能声称全场景全指标最优。\n\n先看[HDSF科学泛化验证_结论与指标.md](HDSF科学泛化验证_结论与指标.md)。全部指标、预测、模型、协议、负面结果和独立核对证据在experiment/generalization_validation_20261012。原Word保留为此前阶段历史快照，本轮按用户要求直接汇报，未新建Word。\n',encoding='utf-8')
    readme=root/'开始阅读.md'
    if readme.exists():
        text=readme.read_text(encoding='utf-8-sig');intro='最新科学泛化验证已完成：[结论与全部指标](HDSF科学泛化验证_结论与指标.md)。NLR24小时有持续迁移收益，7天效果有时间段差异；海光寺/Cluster结果和限制分别列出。Google当前推荐仍为第48轮。\n\n'
        if intro not in text:readme.write_text(intro+text,encoding='utf-8')
    save(q/'delivery_verification.json',dict(all_passed=True,utc=utc(),model_names_complete=True,input_cache_checks=len(input_checks),
        scientific_checks=read(q/'final_verification.json')['check_count'],report=str(report),report_sha256=sha(report),
        full_metric_csv_sha256=sha(q/'泛化验证_全部统一指标_模型全名.csv'),current_recommendation_sha256=sha(root/'当前推荐方案.json'),actual_external_model_scores=True))
    # Last mutation is the manifest inventory. An independent existing verifier
    # will run afterwards; the manifest deliberately excludes itself.
    manifest=root/'交付文件清单.json';v=read(manifest);v.update(generalization_model_scores_computed=True,generalization_validation=str(folder),
        generalization_report=str(report),generalization_completed_utc=utc(),inventory_verified=False)
    files=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size) for f in root.rglob('*') if f.is_file() and f!=manifest];files.sort(key=lambda r:r['path'])
    v['all_files']=files;v['files']=len(files);v['bytes']=sum(r['bytes'] for r in files);save(manifest,v)
    print(json.dumps(dict(all_passed=True,metric_rows=len(rows),input_cache_checks=len(input_checks),files=len(files),report=str(report)),ensure_ascii=True),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();main(Path(a.root))
