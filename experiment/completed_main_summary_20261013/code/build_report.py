"""Results-first report for the reduced, actually completed comparison scope."""
from pathlib import Path
import argparse, json, sys, datetime, hashlib, importlib.util
from docx.shared import Pt
from docx.oxml.ns import qn
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();R=Path(a.root)
F=R/'experiment/completed_main_summary_20261013'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
s=read(F/'results/summary.json');audit=read(F/'results/source_verification.json');assert audit['all_passed']
for rel,h in audit['source_sha256'].items():assert sha(R/rel)==h,rel
rows=s['completed_rows'];dev=s['development_rows'];assert not s['new_target_fit_test_scored']
if a.validate_only:print(json.dumps(dict(all_passed=True,completed_rows=len(rows),dev_rows=len(dev))));raise SystemExit()
spec=importlib.util.spec_from_file_location('style',F/'code/report_style.py');style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
rep=style.Report();doc=rep.doc;bound=[]
original_table=rep.table
def compact_table(*args,**kwargs):
    table=original_table(*args,**kwargs)
    for node in table._tbl.tblPr.findall('.//'+qn('w:top'))+table._tbl.tblPr.findall('.//'+qn('w:bottom')):
        if node.get(qn('w:w')) is not None:node.set(qn('w:w'),'45')
    blank=doc.paragraphs[-1];blank.paragraph_format.line_spacing=Pt(1);blank.paragraph_format.space_after=Pt(4);blank.paragraph_format.space_before=Pt(0)
    return table
rep.table=compact_table
def start_section(title):
    p=rep.h(title);p.paragraph_format.page_break_before=True
rep.new=start_section
for nm in ['Normal','Heading 1','Heading 2']:doc.styles[nm].paragraph_format.line_spacing=1.05
doc.core_properties.title='主要负荷预测方法精度与泛化评估汇报'
doc.core_properties.subject='已完成测试结果与新增训练开发期结果分开汇总'
LABEL={'baseline':'历史基线','hdsf':'HDSF原版','R48':'HDSF第48轮','timesfm30_long':'TimesFM3 21天','toto2_joint':'Toto2 14天',
 'HDSF_Google_frozen':'HDSF原版迁移','R48_Google_frozen':'HDSF第48轮迁移','HDSF_NLR_refit':'NLR单纯形适配','expert_timesfm30':'TimesFM3 14天','expert_timesfm30_long':'TimesFM3 21天','expert_toto2_joint':'Toto2 14天',
 'last':'前一天同刻','mean3':'前三天同刻均值','Ridge_daily':'日电量Ridge','daily_simplex':'日电量单纯形',
 'R48_fixed':'HDSF第48轮','R48_CPU_DEV_frozen':'HDSF加CPU差分','LightGBM_power_only':'LightGBM功率','LightGBM_power_CPU_memory':'LightGBM功率CPU内存'}
def val(x,d=4):return '—' if x is None else str(x) if isinstance(x,int) else f'{x:.{d}f}'
def label(row):return LABEL.get(row['model'],row['model'])
def numtable(rr,fields,size=8.8):
    data=[[label(r)]+[val(r.get(k),5 if r.get('unit')=='normalized power' and k in ['mae','rmse','daily_mae','daily_rmse','interval80_mean_width','interval80_winkler80'] else digits) for k,title,digits in fields] for r in rr]
    rep.table(['明确模型']+[title for k,title,d in fields],data,[1.58]+[5.52/len(fields)]*len(fields),size)
    for r in rr:
        for k,t,d in fields:bound.append(dict(dataset=r['dataset'],target=r.get('target'),phase=r.get('phase'),horizon_days=r['horizon_days'],scope=r.get('scope'),model=r['model'],key=k,value=r.get(k),display=val(r.get(k),5 if r.get('unit')=='normalized power' and k in ['mae','rmse','daily_mae','daily_rmse','interval80_mean_width','interval80_winkler80'] else d)))
POINT=[('wape','曲线\nWAPE %',4),('mae','MAE',3),('rmse','RMSE',3),('p95_ape','P95\nAPE %',3),('hit2_percent','命中2%\n%',2),('hit5_percent','命中5%\n%',2),('absolute_bias_percent','绝对\n偏差 %',4)]
DAY=[('daily_wape','日均\nWAPE %',4),('daily_mae','日均\nMAE',3),('daily_rmse','日均\nRMSE',3),('daily_hit2_percent','日均命中\n2% %',2),('daily_hit5_percent','日均命中\n5% %',2),('peak_mape','峰值\nMAPE %',4),('peak_time_mae_minutes','峰时误差\n分钟',1)]
IV=[('peak_underestimate_percent','峰值\n低估 %',2),('interval80_coverage_percent','区间\n覆盖 %',2),('interval80_mean_width','区间\n宽度',3),('interval80_winkler80','Winkler80',3),('interval80_below_percent','低于区间\n%',2),('interval80_above_percent','高于区间\n%',2)]
def get(dataset,phase,target,h,model):return next(r for r in rows if r['dataset']==dataset and r.get('phase')==phase and r.get('target')==target and r['horizon_days']==h and r['model']==model)
def group(dataset,phase,target,h):return [r for r in rows if r['dataset']==dataset and r.get('phase')==phase and r.get('target')==target and r['horizon_days']==h]
def overview_row(dataset,period,target,m1,m7):
    u=get(dataset,period,target,1,m1);v=get(dataset,period,target,7,m7)
    for r,k in [(u,'wape'),(v,'wape'),(v,'daily_wape')]:bound.append(dict(dataset=dataset,target=target,phase=period,horizon_days=r['horizon_days'],model=r['model'],key=k,value=r[k],display=val(r[k])))
    return [('Google2019' if dataset=='GooglePower2019' else'NLR2024' if period=='test' else'NLR2025'),label(u) if m1==m7 else label(u)+'\n7天 '+label(v),val(u['wape']),val(v['wape']),val(v['daily_wape'])]
rep.p('主要负荷预测方法精度与泛化评估汇报',True,18,'Title');rep.p('已完成结果汇总  2026年10月13日',style='Subtitle')
rep.p('Google继续推荐第48轮HDSF稳健校正。NLR没有一个方案在两个年份的三项任务全部领先：TimesFM3在24小时曲线表现较好，Toto2与NLR单纯形适配在7天曲线上较稳定。海光寺日电量仍以简单前一日预测最合适。',True)
rep.p('本汇报只保留主要且已经完成的对照。以下是已完成测试的描述性结果；新增权重训练结果另列为开发期，不算作新的泛化证明。所有百分比指标越低越好，命中率和覆盖率除外。',size=10)
main=[]
for dataset,phase,target in [('GooglePower2019','test','57PDU'),('NLR','test','facility_total_kw'),('NLR','confirm','facility_total_kw')]:
    if dataset=='GooglePower2019':models=[('baseline','baseline'),('R48','R48'),('timesfm30_long','timesfm30_long')]
    else:models=[('last','mean3'),('R48_Google_frozen','R48_Google_frozen'),('expert_timesfm30' if phase=='test' else'expert_timesfm30_long','expert_timesfm30_long'),('expert_toto2_joint','expert_toto2_joint'),('HDSF_NLR_refit','HDSF_NLR_refit')]
    for m1,m7 in models:main.append(overview_row(dataset,phase,target,m1,m7))
t=rep.table(['数据时期','具体模型','24小时曲线\nWAPE %','7天曲线\nWAPE %','7天日均\nWAPE %'],main,[.8,2.28,1.34,1.34,1.34],9)
for period in ['Google2019','NLR2024','NLR2025']:
    ii=[i for i,r in enumerate(main) if r[0]==period]
    for j in [2,3,4]:
        best=min(float(main[i][j]) for i in ii)
        for i in ii:
            if abs(float(main[i][j])-best)<1e-9:
                for run in t.rows[i+1].cells[j].paragraphs[0].runs:run.font.bold=True
rep.p('同一时期同一列的最低值已加粗；是已完成主要对照的描述性名次，不是根据测试集新选出的部署方案。',size=9)
rep.p('Google为57个PDU等权；NLR这里为设施总功率。NLR2024测试16次起报、2025确认13次起报，属于稀疏代表日期回测，不是全年逐日回测。“确认”是原实验阶段名称，数据现已查看，不能称为本次新盲测。',size=9.5)

rep.new('模型定义与评价口径')
rep.p('表格短名只为节省空间，完整模型名称如下。每个任务使用自己的固定配置；不把24小时和7天两个分支伪装成一个通用神经网络。')
rep.table(['表格短名','完整定义'],[
 ['HDSF原版','HDSF半日单纯形融合，每48个15分钟点使用一组非负且和为1的专家权重。原预训练专家及局部预测器构成专家库，按开发期拟合权重。'],
 ['HDSF第48轮','24小时为HDSF加PatchTST-FM日均残差校正和同组50%收缩；7天为HDSF加TimesFM3、Toto2、PatchTST-FM零均值形状全量校正。它是用户已采纳方案，原开发期升级门槛没有通过。'],
 ['HDSF固定迁移','将上述Google权重固定应用NLR。缺少资源字段按缺失输入处理，少于3个可比场站时不启用同组收缩。点预测权重没有用2024或2025重新拟合。'],
 ['NLR单纯形适配','神经基础模型原权重不变；局部模型及单纯形权重由NLR训练段、2023开发段适配。设施总功率两任务开发期选全局权重，半日比例0；IT七天半日比例25%。'],
 ['TimesFM3与Toto2','已完成外部测试使用原预训练权重。TimesFM3分别明确14天或21天历史；Toto2使用14天。一个目标的联合模式退化为单变量，不当作独立模型重复排名。'],
 ['Moirai2目标微调','只在新增训练开发期比较中出现。实际更新末层与输出头16或48次，不声称全模型重训练已完成外部测试。']], [1.35,5.75],10)
rep.sub('统一计算方式')
rep.p('曲线WAPE = 每个域的有效点绝对误差和 ÷ 同域有效真值绝对和 × 100%，再对各域等权平均。Google为57域；NLR设施总功率与IT分别评价，不能与kWh或归一化功率混合成一个总分。')
rep.p('每天96个真值点都有效，才计入每日平均负荷的主指标。另存“仅有效格日均WAPE”供诊断，它不能替换主指标。7天日均由同一条672点预测曲线按96点分日求均值，不另换目标。')
rep.p('峰值幅值与峰时沿原有效点掩码评价；多个最大值取首个，峰时误差按15分钟格差计算。峰值低估率、P95 APE、2%和5%命中率、绝对偏差、MAE、RMSE及日均对应指标均保留。真值为0时APE不定义并计数。')
rep.p('80%预测区间是目标开发期残差校准的经验区间。覆盖率、平均宽度、Winkler80以及落在上下界外的比例同时列出；并非理论上保证80%覆盖。未经统一残差校准的单模型区间标“—”。')
rep.p('本次重算与原9月28日对齐评价程序核查一致。完整指标CSV还保留样本数、有效格日均、相对基线收益、MASE及RMSSE等已有补充字段。原始结果和失败证据均保留。',size=10)

rep.new('Google原数据已完成主要方法比较')
for h in [1,7]:
    rr=group('GooglePower2019','test','57PDU',h);rep.sub('未来24小时完整曲线' if h==1 else'未来7天完整曲线及每日平均负荷')
    numtable(rr,POINT);numtable(rr,DAY)
rep.p('误差单位是原数据归一化功率，不能解释为kW。24小时共有49243个有效点和508个完整日；7天共有114902个有效点和1187个完整日。第48轮两项曲线WAPE都低于本表主要对照；七天日均与HDSF原版相同，峰值并不全面领先。',size=10)
rep.p('相对原版，当前推荐24小时曲线只改善约0.00551个百分点，七天约0.00044个百分点。原同月测试多轮已查看，且用户采纳不等于开发期稳定升级通过；不能称为本次证明了大幅改善。',size=10)

for phase in ['test','confirm']:
    for target in ['facility_total_kw','it_power_kw']:
        title=('NLR2024测试' if phase=='test' else'NLR2025确认')+('设施总功率' if target=='facility_total_kw' else'IT功率')
        rep.new(title+'主要方法全部指标')
        for h in [1,7]:
            rr=[r for r in group('NLR',phase,target,h) if r['model']!='HDSF_Google_frozen'];rr.sort(key=lambda r:['last','mean3','R48_Google_frozen','HDSF_NLR_refit','expert_timesfm30','expert_timesfm30_long','expert_toto2_joint'].index(r['model']))
            cov=next(x for x in s['methodological_audit']['actual_evaluation_coverage'] if x['phase']==phase and x['target']==target and x['horizon_days']==h)
            rep.sub('未来24小时完整曲线' if h==1 else'未来7天完整曲线及每日平均负荷')
            rep.p(f'单位kW；{cov["origins"]}次起报，{cov["valid_points"]}个有效点，{cov["valid_complete_days"]}个完整日均目标。',size=9)
            numtable(rr,POINT,8.4);numtable(rr,DAY,8.4)
        rep.p('预测区间见后面的集中附表。同族HDSF原版固定迁移及单模型峰值低估率保留在全指标CSV，正文只保留第48轮迁移以减少重复。',size=9)

rep.new('主要方案预测区间集中比较')
rep.p('标称覆盖率80%，区间由目标开发期残差校准。以下Google区间宽度与Winkler为归一化功率；NLR为kW。原单模型未做统一经验区间校准，不能凭空补值。峰值低估率越低通常越少低估，但也应结合幅值误差判断。',size=10)
ivrows=[]
for r in rows:
    if r.get('interval80_coverage_percent') is None or r['dataset'] not in ['NLR','GooglePower2019']:continue
    if r['dataset']=='NLR' and r['model']=='HDSF_Google_frozen':continue
    ctx='Google' if r['dataset']=='GooglePower2019' else ('24' if r['phase']=='test' else'25')+('总功率' if r['target']=='facility_total_kw' else'IT')
    vals=[ctx+' '+str(r['horizon_days'])+'天',label(r)]+[val(r.get(k),5 if r['unit']=='normalized power' and k in ['interval80_mean_width','interval80_winkler80'] else d) for k,t,d in IV]
    ivrows.append(vals)
    for k,t,d in IV:bound.append(dict(dataset=r['dataset'],target=r.get('target'),phase=r.get('phase'),horizon_days=r['horizon_days'],model=r['model'],key=k,value=r.get(k)))
rep.table(['时期对象任务','明确模型']+[t for k,t,d in IV],ivrows,[1.2,1.58]+[4.32/6]*6,8.5)

rep.new('海光寺日电量与同源算力特征对照')
rep.sub('海光寺仅能评价日电量任务')
sea=[r for r in rows if r['dataset']=='SeaGuang']
for h in [1,7]:
    rr=[r for r in sea if r['horizon_days']==h]
    rep.p('未来1日总电量' if h==1 else'未来7日每日电量',True,10)
    numtable(rr,POINT,8.7);numtable(rr,[('interval80_coverage_percent','覆盖率\n%',2),('interval80_mean_width','区间宽度\nkWh',2),('interval80_winkler80','Winkler80\nkWh',2),('interval80_below_percent','低于区间\n%',2),('interval80_above_percent','高于区间\n%',2)],8.7)
rep.p('源表只有2022年9月30个日电量。次日测试为9月23至30日8个点，七日为2个重叠起报。冻结单纯形回退前一日。无日内真值，不能验证完整HDSF曲线或峰值。',size=10)
rep.sub('ClusterData只做原Google五域的历史特征消融')
cr=[r for r in rows if r['dataset']=='GoogleCluster_same_source']
table=[]
for h in [1,7]:
    for rr in [r for r in cr if r['horizon_days']==h and r['scope']=='cella5']:
        table.append([str(h)+'天',label(rr),val(rr['wape']),val(rr['daily_wape']),val(rr['peak_mape'])])
rep.table(['任务','明确模型','曲线WAPE %','日均WAPE %','峰值MAPE %'],table,[.55,2.8,1.25,1.25,1.25],9)
rep.p('历史CPU内存帮助LightGBM七天预测，但五域曲线仍不如当前HDSF。追加25%差分的HDSF退步，未采用。这是同月同标签消融，不是外部功率泛化。',size=10)

rep.new('新增目标权重训练完成情况')
rep.p('TimesFM3、Toto2与Moirai2在两个NLR训练边界及Google训练边界均已有真实16步和48步更新，并完成对应开发期推理。用户要求加快汇总后，未完成的次要队列与后续新测试已取消；本页所有数值仅是开发期，不与前面2024或2025测试分数混排。',size=10)
for case,title in [('NLR_2021','NLR开发2020 训练截止2019'),('NLR_2024_2025','NLR开发2023 训练截止2022'),('Google2019','Google开发训练截止第14天')]:
    rep.sub(title)
    table=[]
    for h in [1,7]:
        for family in ['timesfm3','toto2','moirai2']:
            rr=[r for r in dev if r['dataset']==case and r['horizon_days']==h and r['model']==family and r['target'] in ['facility_total_kw','57PDU']];dd={r['checkpoint']:r for r in rr}
            table.append([str(h)+'天',{'timesfm3':'TimesFM3','toto2':'Toto2','moirai2':'Moirai2'}[family],val(dd['s0']['wape']),val(dd['s16']['wape']),val(dd['s48']['wape']),val(dd['s48']['daily_wape'])])
            for r in rr:
                for k in ['wape','daily_wape']:bound.append(dict(dataset=case,target=r['target'],phase='dev',horizon_days=h,model=family,checkpoint=r['checkpoint'],key=k,value=r[k],display=val(r[k])))
    rep.table(['任务','模型','原权重\n曲线WAPE %','16步\n曲线WAPE %','48步\n曲线WAPE %','48步日均\nWAPE %'],table,[.5,1.2,1.35,1.35,1.35,1.35],9)
rep.p('这里NLR只展示设施总功率，完整IT开发期指标另存CSV。NLR2020为46个按周开发起报、2023为31个；Google24小时为8个起报，七天为2个。它们与旧泛化报告的2023稀疏开发起报及旧Google四次起报不同，因此不能直接横向比较开发分数。',size=9.5)
rep.p('可见微调大多只产生很小变化。Moirai2在NLR2023的24小时16步略好但48步回退；两个NLR开发期的七天48步有所改善，Google七天却变差。没有证据支持“大幅全面提高”，也尚无新测试证明这些更新提高了泛化。',True,10)
rep.p('训练标签均结束于训练边界内，缓存输入摘要已逐项核对历史片段；TimesFM3某训练分支出现非有限梯度批次，已跳过该批次并保留证据，48次有限更新实际完成。Moirai2训练使用原生64点预测头，完整96或672点推理的误差已按原网格计算，不能称为直接训练了完整七天输出头。',size=9.5)

rep.new('泛化判断与数据处理说明')
rep.sub('哪些方案准确且较稳定')
rep.p('Google当前仍用第48轮HDSF稳健校正。NLR24小时可把TimesFM3作为强单模型对照，14天版本在2024总功率曲线更好，21天版本在2025更好；不能看过年份结果后逐年切换并宣传成冻结方案。')
rep.p('HDSF固定迁移的24小时曲线在两个年份均超过前一天基线，删去任一季度的配对曲线收益仍为正，支持在该设施观察到较稳定的短期迁移。七天固定HDSF在2025落后前三天均值；目标数据单纯形适配与Toto2的七天曲线在两年均超过该基线，更适合作为长时距重点候选。')
rob=s['methodological_audit']['robustness'];q=[]
for phase in ['test','confirm']:
    for model in ['R48_Google_frozen','HDSF_NLR_refit']:
        for h in [1,7]:
            z=next(x for x in rob if x['phase']==phase and x['target']=='facility_total_kw' and x['horizon_days']==h and x['model']==model)
            q.append(['2024' if phase=='test' else'2025',label(dict(model=model)),str(h)+'天',val(z['full_gain_pp']),f'{z["positive_months"]}/{z["evaluated_months"]}',val(z['drop_one_quarter_gain_min_pp'])])
rep.table(['时期','方案','任务','基线收益\n百分点','改善月数\n已评月数','删一季度收益\n最小百分点'],q,[.6,1.8,.5,1.3,1.3,1.6],9)
rep.p('这只是同一外部设施的事后稳定性审查，季度不是随机独立实验；不当作正式显著性检验。开发期拟合和测试期评分边界核查通过，不等于已排除所有过拟合、语料重叠或样本选择问题。',size=10)
rep.sub('原始字段与实际处理')
rep.p('Google功率标签为原始PDU归一化功率利用率，结合时间戳和PDU标识整理为57域、31天、每日96点。保留原有效值掩码，缺失真值不填作评分标签；预测始终只输入起报以前的历史。')
rep.p('NLR时间列是源文件采样时刻；五类功率字段为IT、冷却、HVAC、插座照明及泵功率，单位kW。设施总功率取五类之和，IT单列评价，二者来自同一设施且相互依赖。按15分钟整理，输入采用因果向前填充，标签有效掩码不变；缩放只用训练段有效值95%分位数除以0.8，反变换回kW，不使用Google的0到1容量上界。源时区未确认，日均是源时间坐标下96格窗口。')
rep.p('海光寺读取日期和每日总电量，单位kWh，保留日尺度，不造15分钟值。ClusterData按对象去重并做时间对齐，将历史CPU、内存及资源信息按15分钟聚合；CPU为NCU、内存为归一化量。真实算力只覆盖cell-a五域，其他52域不可用占位不能当真实零算力，也不把CPU转换为物理kW。')
rep.p('读取范围为官方cell-a的12085个Parquet必要列，已经完整且剩余下载为零；不是全部八个集群或全部字段。原文件、保留列字节、逐对象聚合和审计均在D盘；桌面同名路径为兼容链接。原数据许可及TimesFM3仅非商业研究用途限制继续适用。')
rep.sub('交付范围')
rep.p('本次已完成结果汇总含主要方法的三项任务和全部原评价指标；完整精度、区间、样本数及已有补充指标见“已完成主要方法统一全指标.csv”，新增训练另见“新增训练开发期全指标.csv”。原先全部模型结果仍保留，不因本次缩小正文而删除或重新选择。')
rep.p('现有证据仍属探索性：NLR只有一个外部设施，Google测试曾多轮查看，基础模型语料重叠未知。全面泛化需要新的未见月份或外部场站验证。')
out=R/'主要负荷预测方法精度与泛化评估汇报_已完成结果.docx';doc.save(out)
check_doc=style.Document(out);assert len(check_doc.tables)==len(doc.tables)
# Compare every generated table cell to its in-memory bound content after save.
table_cells=[[c.text for row in t.rows for c in row.cells] for t in doc.tables]
reopened=[[c.text for row in t.rows for c in row.cells] for t in check_doc.tables];assert table_cells==reopened
for p in check_doc.paragraphs:
    if p.style.name=='Title':assert p.text=='主要负荷预测方法精度与泛化评估汇报' and p._p.find('.//'+qn('w:pBdr')) is None
save(F/'docs/document_content_verification.json',dict(all_passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),document_sha256=sha(out),completed_rows=len(rows),development_rows=len(dev),numerical_records=bound,table_count=len(doc.tables),numeric_fields=len(bound),source_verification_sha256=sha(F/'results/source_verification.json'),new_target_fit_not_presented_as_test=True,current_deployment_unchanged=True))
print(json.dumps(dict(all_passed=True,document=str(out),numeric_fields=len(bound),tables=len(doc.tables)),ensure_ascii=False))
