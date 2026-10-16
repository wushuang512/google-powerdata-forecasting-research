"""Results-first Word report generated exclusively from verified real outputs."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
SHORT={
 'HDSF_Google_frozen':'HDSF原版 固定迁移','R48_Google_frozen':'第48轮HDSF 固定迁移','HDSF_NLR_refit':'NLR单纯形权重适配',
 'last':'前一天同刻','week':'上周同刻','mean3':'前三天同刻均值','LightGBM_NLR':'NLR局部LightGBM','Ridge_NLR':'NLR Ridge',
 'local7_smoother':'水平形状指数平滑','expert_chronos2_long':'Chronos-2 28天',
 'expert_tirex2':'TiRex-2 14天','expert_timesfm25':'TimesFM 2.5 14天','expert_timesfm30':'TimesFM 3 14天',
 'expert_timesfm30_long':'TimesFM 3 21天 原联合','expert_timesfm30_uni':'TimesFM 3 21天 单变量',
 'expert_toto2_joint':'Toto2 14天 原同组','expert_toto2_single':'Toto2 14天 单变量',
 'expert_patchtstfm_long':'PatchTST-FM-r2 21天',
 'Ridge_daily':'日尺度Ridge','daily_simplex':'日尺度单纯形 回退前一日',
 'R48_fixed':'第48轮HDSF 固定推荐','R48_CPU_DEV_frozen':'第48轮HDSF 加历史算力差分',
 'LightGBM_power_only':'LightGBM 仅历史功率','LightGBM_power_CPU_memory':'LightGBM 功率加CPU内存'}
ORDER=['last','week','mean3','HDSF_Google_frozen','R48_Google_frozen','HDSF_NLR_refit','LightGBM_NLR','Ridge_NLR','local7_smoother',
 'expert_chronos2_long','expert_tirex2','expert_timesfm25','expert_timesfm30','expert_timesfm30_long','expert_timesfm30_uni','expert_toto2_joint','expert_toto2_single','expert_patchtstfm_long']
def value(x,digits=4):
    if x is None:return '不适用'
    if isinstance(x,bool):return '是' if x else'否'
    if isinstance(x,(int,)):return str(x)
    if isinstance(x,float):return f'{x:.{digits}f}'
    return str(x)
def set_font(run,size=11,bold=False,color='000000'):
    run.font.name='Calibri';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor.from_string(color)
    r=run._element.get_or_add_rPr();f=r.find(qn('w:rFonts'))
    if f is None:f=OxmlElement('w:rFonts');r.insert(0,f)
    f.set(qn('w:eastAsia'),'宋体');f.set(qn('w:ascii'),'Calibri');f.set(qn('w:hAnsi'),'Calibri')
class Report:
    def __init__(self):
        self.doc=Document();self.records=[];self.sections=[]
        s=self.doc.sections[0];s.page_width=Inches(8.5);s.page_height=Inches(11)
        s.top_margin=Inches(.65);s.bottom_margin=Inches(.65);s.left_margin=Inches(.65);s.right_margin=Inches(.65)
        s.footer_distance=Inches(.28)
        for st in self.doc.styles:
            for border in list(st._element.iter(qn('w:pBdr'))):
                border.getparent().remove(border)
        for name,size in [('Normal',11),('Title',18),('Subtitle',11.5),('Heading 1',14),('Heading 2',12)]:
            st=self.doc.styles[name];st.font.name='Calibri';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
            if name in ('Title','Subtitle'):st.font.italic=False
            st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体' if name=='Normal' else'黑体')
            st.paragraph_format.space_after=Pt(6);st.paragraph_format.line_spacing=1.15
            if name.startswith('Heading'):st.paragraph_format.space_before=Pt(10);st.paragraph_format.keep_with_next=True
        footer=s.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
        run=footer.add_run('第 ');set_font(run,9)
        field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
        set_font(footer.add_run(' 页'),9)
        self.doc.core_properties.title='HDSF负荷预测泛化能力评估与科学性审查汇报'
        self.doc.core_properties.subject='NLR外部时间留出 海光寺日电量 ClusterData同源消融'
        self.doc.core_properties.author='负荷预测研究'
    def p(self,text,bold=False,size=11,style=None):
        p=self.doc.add_paragraph(style=style);set_font(p.add_run(text),size,bold);return p
    def h(self,text):self.sections.append(text);return self.p(text,True,14,'Heading 1')
    def sub(self,text):return self.p(text,True,12,'Heading 2')
    def new(self,title):self.doc.add_page_break();self.h(title)
    def table(self,headers,rows,widths=None,size=9.5,source_rows=None,keys=None):
        table=self.doc.add_table(rows=1,cols=len(headers));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
        widths=widths or [7.1/len(headers)]*len(headers)
        for c,w in zip(table.columns,widths):c.width=Inches(w)
        pr=table._tbl.tblPr
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
        pr.append(borders)
        margins=OxmlElement('w:tblCellMar')
        for edge,v in [('top',65),('bottom',65),('left',65),('right',65)]:
            e=OxmlElement('w:'+edge);e.set(qn('w:w'),str(v));e.set(qn('w:type'),'dxa');margins.append(e)
        pr.append(margins)
        allrows=[headers]+rows
        for i,vals in enumerate(allrows):
            row=table.rows[0] if i==0 else table.add_row();trpr=row._tr.get_or_add_trPr()
            keep=OxmlElement('w:cantSplit');trpr.append(keep)
            if i==0:repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
            for j,(cell,txt,w) in enumerate(zip(row.cells,vals,widths)):
                cell.width=Inches(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                para=cell.paragraphs[0];para.paragraph_format.space_after=Pt(0);para.paragraph_format.space_before=Pt(0);para.paragraph_format.line_spacing=1.05
                para.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                set_font(para.add_run(str(txt)),size,i==0,'FFFFFF' if i==0 else'000000')
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'274663' if i==0 else('F3F3F3' if i%2==0 else'FFFFFF'));cell._tc.get_or_add_tcPr().append(shade)
        self.doc.add_paragraph().paragraph_format.space_after=Pt(2)
        if source_rows is not None:
            for row in source_rows:
                for key in keys or []:
                    if row.get(key) is not None:self.records.append(dict(dataset=row.get('dataset'),phase=row.get('phase'),target=row.get('target'),horizon_days=row['horizon_days'],model=row['model'],key=key,value=row[key]))
        return table
    def metric_table(self,rows,fields,widths,size=9.2):
        vals=[[SHORT[r['model']]]+[value(r.get(k),d) for k,label,d in fields] for r in rows]
        return self.table(['具体模型']+[label for k,label,d in fields],vals,widths,size,rows,[k for k,label,d in fields])
    def link(self,p,title,url):
        part=p.part;rel=part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True)
        h=OxmlElement('w:hyperlink');h.set(qn('r:id'),rel);r=OxmlElement('w:r');prop=OxmlElement('w:rPr');color=OxmlElement('w:color');color.set(qn('w:val'),'205080');prop.append(color);r.append(prop);t=OxmlElement('w:t');t.text=title;r.append(t);h.append(r);p._p.append(h)
def load(root,audit):
    R=root/'experiment/generalization_validation_20261012';Q=R/'results'
    files=['nlr_test_summary.json','nlr_confirm_summary.json','sea_summary.json','cluster_summary.json'];rows=[]
    for name in files:rows.extend(read(Q/name)['rows'])
    assert len(rows)==130 and read(Q/'final_verification.json')['all_passed'] and read(Q/'independent_input_cache_audit.json')['all_passed']
    a=read(audit/'methodological_audit.json')
    for name,d in a['reference_source_sha256'].items():assert sha(root/name)==d,name
    assert read(root/'当前推荐方案.json')['scheme_id']=='R48_ROBUST_HDSF_USER_ADOPTED'
    return R,Q,rows,a
def create(root,audit,out):
    R,Q,rows,a=load(root,audit);rep=Report();doc=rep.doc;nlr=[r for r in rows if r['dataset']=='NLR']
    def get(phase,target,h,model):return next(r for r in nlr if r['phase']==phase and r['target']==target and r['horizon_days']==h and r['model']==model)
    rep.p('HDSF负荷预测泛化能力评估与科学性审查汇报',True,18,'Title')
    rep.p('2026年10月12日',False,11.5,'Subtitle')
    rep.p('科学性判断  本轮属于可复核的探索性外部时间留出验证。24小时迁移收益较稳定；7天效果依赖时间段，尚不能证明多场站普适泛化或所有指标最优。',True)
    rep.sub('NLR设施总功率主要结果')
    main=[]
    for phase in ['test','confirm']:
        for name in ['baseline','R48_Google_frozen','HDSF_NLR_refit']:
            n1='last' if name=='baseline' else name;n7='mean3' if name=='baseline' else name
            u,v=get(phase,'facility_total_kw',1,n1),get(phase,'facility_total_kw',7,n7)
            main.append(['2024测试' if phase=='test' else'2025确认','原对应历史基线' if name=='baseline' else SHORT[name],value(u['wape']),value(v['wape']),value(v['daily_wape'])])
    rep.table(['时期','具体方案','24小时曲线\nWAPE %','7天曲线\nWAPE %','7天日均\nWAPE %'],main,[.8,2.25,1.35,1.35,1.35],10)
    rep.p('基线固定为24小时前一天同刻、7天前三天同刻均值。上述数值越低越好；设施功率由五个功率分量相加得到，使用源文件时间坐标下的15分钟窗口。',size=10)
    rep.sub('结果对推荐方案的含义')
    for text in [
        '第48轮HDSF固定参数迁移的24小时曲线误差，在2024、2025分别比原基线减少19.29%、22.23%。季度删除审查仍保留正收益。',
        '7天固定迁移在2024优于基线，在2025的曲线与日均点估计未超过基线。场站权重适配改善了7天点估计，但原月簇探索区间跨零，不能称为正式显著改善。',
        '海光寺日尺度组合未带来新增收益。ClusterData历史CPU和内存改善了单独LightGBM，却未改善叠加后的HDSF；保留负面结果，不采用该修正。',
        'Google当前推荐仍为已固定的第48轮。外部单模型在部分任务更优，峰值与区间覆盖也未全面领先；本轮没有根据外部测试名次再次选择或部署新模型。']:
        rep.p(text)
    rep.new('主要结果图与单模型对照')
    doc.add_picture(str(Q/'NLR设施总功率_三项任务比较.png'),width=Inches(7.1))
    drawing=doc.inline_shapes[-1]._inline.docPr;drawing.set('descr','NLR设施总功率2024测试和2025确认的24小时曲线、7天曲线和7天每日均值WAPE四方案比较')
    rep.p('图1  固定规则下的外部结果。2024保留16个起报、覆盖9个月；2025保留13个起报、覆盖7个月。未评价全年每一天。',size=10)
    winner=[]
    for phase in ['test','confirm']:
        for h in [1,7]:
            source=[r for r in nlr if r['phase']==phase and r['target']=='facility_total_kw' and r['horizon_days']==h]
            best=min(source,key=lambda r:r['wape'])
            winner.append(['2024测试' if phase=='test' else'2025确认',str(h)+'天',SHORT[best['model']],value(best['wape'],6)])
    rep.table(['时期','任务','实测曲线WAPE最低的预登记对照','WAPE %'],winner,[.8,.55,4.4,1.35],10)
    rep.p('表中的最低值是测试后的描述性名次，不是新的选模依据。单场站时TimesFM3两种21天模式、Toto2原同组与单变量模式预测相同，不能据重复行增加“独立专家”数量。',size=10)
    rep.new('科学性复审结论与证据强度')
    rep.table(['审查项目','实际情况','判断'],[
        ['时间与参数边界','训练止于2022 验证为2023\n测试2024 确认2025不再调参','支持时间留出比较'],
        ['统一评价','原Google指标重算对齐\n同一起报 同一真值掩码','支持模型间公平比较'],
        ['外部对象','NLR只有一个独立设施\n设施与IT属于同一场站','不能推断多场站总体'],
        ['零样本定义','目标历史用于缩放\n区间使用目标开发残差','固定参数迁移有条件'],
        ['统计与区间','有效月份9个与7个\n原重采样未验证月间独立','仅探索性区间'],
        ['其他两份数据','海光寺为30天日电量\nCluster功率仍为原同月','不算两个新增曲线泛化集']],
        [1.3,4.05,1.75],10)
    rep.p('训练、验证、测试按时间推进，符合预测任务只能使用起报前信息的基本要求。[1] 固定迁移与目标场站拟合分别列示，避免把目标数据训练后的性能称为无适配的直接迁移。')
    rep.p('外部测试的“独立”指本项目没有用这些未来段选择参数。基础模型预训练语料未公开完整清单，不能证明NLR没有进入其预训练语料。[4][5] Google原2019年测试月已被多轮查看；用户接受第48轮候选的部署决定也不等于其开发期稳定升级通过。')
    rep.p('已有3366项工程核查及774份历史输入缓存核对支持代码、文件与计算一致性；这些不是3366个独立统计实验，也不能单独证明没有过拟合。')
    rep.sub('需要准确更正的表述')
    rep.p('起报资格代码同时检查了未来7天是否有可评分真值。本次重新审查发现，该条件实际没有改变任何候选日期，影响数为0；因此原成绩不受影响。今后的实际部署应仅按历史确定起报资格，再单独记录未来真值缺失，不使用未来可用性决定是否预测。')
    rep.p('原“95%收益区间”统一称为月簇重采样的探索性95%分位区间。它保留同月样本，却不等于已验证的时间序列移动块或平稳Bootstrap，也不具有已证明的95%覆盖保证。[3]')
    rep.new('新增稳定性审查与区间可靠性')
    robust=[]
    for r in a['robustness']:
        if r['target']=='facility_total_kw' and r['model'] in ['R48_Google_frozen','HDSF_NLR_refit']:
            robust.append(['2024' if r['phase']=='test' else'2025',str(r['horizon_days'])+'天',SHORT[r['model']],
                f'{r["positive_months"]}/{r["evaluated_months"]}',f'{r["positive_quarters"]}/{r["evaluated_quarters"]}',
                f'{r["drop_one_quarter_gain_min_pp"]:+.4f} 至 {r["drop_one_quarter_gain_max_pp"]:+.4f}'])
    rep.table(['时期','任务','具体方案','正收益\n月份','正收益\n季度','删除任一季度后收益\n百分点'],robust,[.55,.5,1.95,.75,.75,2.6],9.6)
    rep.p('收益均相对原对应基线，正数代表误差减少。此处是对既有测试结果的事后稳定性诊断，没有新预测、重新训练、改权重或再选模型，不是新的盲测。')
    rep.p('第48轮设施24小时在2024有8/9个月正收益，2025为7/7个月；删除任一季度仍为正。7天固定迁移在2025虽有5/7个月正收益，整体WAPE仍退步，并且删除季度后结论可变；月份数多的一方并不必然获得更低总WAPE，实际功率和有效点数影响分母权重。')
    interval_rows=[]
    for phase in ['test','confirm']:
        for name in ['R48_Google_frozen','HDSF_NLR_refit']:
            r=get(phase,'facility_total_kw',7,name)
            interval_rows.append(['2024' if phase=='test' else'2025',SHORT[name],value(r['interval80_coverage_percent'],2),
                f'{r["lower95_gain_pp"]:+.4f} 至 {r["upper95_gain_pp"]:+.4f}'])
    rep.table(['时期','7天具体方案','标称80%区间\n实际覆盖 %','曲线收益探索区间\n百分点'],interval_rows,[.65,2.4,1.4,2.65],10)
    rep.p('经验区间由2023后段开发残差的10%和90%分位数得到。未建立交换性或平稳性保证，覆盖率必须以实测为准；2024覆盖不足不能通过保留“80%”标签掩盖。区间宽度和越界惩罚Winkler分数也已列入附表。')
    rep.new('三份数据的任务与处理边界')
    rep.table(['数据','有效验证内容','不能从本轮推出'],[
        ['NLR设施总功率与IT功率','各自按15分钟评价三项功率任务','多个独立场站总体表现'],
        ['海光寺报表','次日电量 未来7天每日电量','96点或672点功率曲线 日内峰值'],
        ['Google ClusterData2019','五域历史CPU内存的同条件增益','新增独立功率场站泛化']],
        [1.55,3.25,2.3],10)
    rep.sub('NLR字段与建模处理')
    rep.p('ts为无时区时间戳。it_power_kw、cooling_kw、hvac_kw、plug_and_light_kw、pump_kw分别表示IT、制冷、HVAC、插座照明和泵功率，单位kW。五项相加是本轮派生设施总功率标签；2.67kW过滤泵已包含在制冷统计，不再重复加入。pue及能量再利用字段未作为未来输入，也没有用PUE冒充功率。外部天气未输入本轮模型。')
    rep.p('原Parquet保持不变。有效15分钟格至少有12个有效样本，观测跨度至少600秒，格内原时间间隔不超过180秒；功率组成需有限、非负，IT需为正。此处是样本算术均值，并非连续功率时间积分或电量计量。缺失真值只标无效，历史模型输入向前延续已知值，不向后填补或使用未来插值。')
    rep.p('缩放量只由2022年底及以前的有效功率95%分位数除以0.8得到，预测反变换为kW。这不是设施额定容量，不使用Google归一化功率的[0,1]上界。时区未确认，日均指源文件时间坐标下96格窗口，不能擅称当地自然日。')
    rep.sub('海光寺与ClusterData处理')
    rep.p('海光寺为2022年9月30个日期。每天汇总加两条分项，只保留每日总电量作为目标，单位kWh；90行不是90个日期，也不插值伪造日内曲线。开发与测试仍按日期留出，但月份此前已暴露，样本很小。')
    rep.p('ClusterData只读取官方cell-a的12085个Parquet必要列，真实CPU覆盖五域。其CPU为NCU、内存为归一化量，并非kW。job_window_equivalent、job_window_count及采样率加权时长、时长合计是聚合描述，不是CPU方差。此次增益对照只输入功率、CPU和内存的历史统计；其他52域不可用占位不作真实零算力。功率标签来自原Google同月。')
    rep.new('统一指标与实际模型定义')
    rep.p('曲线WAPE先在每个真实域累计有效点的绝对误差，再除以该域有效真值绝对和，各域等权平均后乘100。Google仍按57域等权；NLR两个目标分别计算且每项N为1。不同单位和不同预测对象不混合排序。')
    rep.p('每日平均负荷主指标只用真值完整96格的日期，预测日均从同一条96点曲线计算；有效格日均WAPE仅作为补充。MAE、RMSE使用原目标单位；P95APE、2%/5%命中率和绝对偏差保留原定义。峰值幅度、首个最大值时间、低估率使用原有效格规则；海光寺没有日内峰值，标为不适用。')
    rep.p('MASE、RMSSE仅作为补充，尺度取训练期相隔一个日周期的有效真实差分，不替代WAPE。[2] 零真值APE不作强行百分比计算并计数。相对基线误差下降为本模型误差相对固定基线的减少比例，负值表示退步。')
    rep.table(['方案或模型','清晰定义'],[
        ['第48轮HDSF 24小时','HDSF半日单纯形加PatchTST-FM日均残差校正；Google权重固定。少于3个可比场站时，同组50%收缩不启用。'],
        ['第48轮HDSF 7天','HDSF半日单纯形加TimesFM3 Toto2 PatchTST-FM零均值形状全量校正；Google权重固定。'],
        ['NLR权重适配','实际训练局部LightGBM及Ridge对照；神经预测器原权重固定。2023前半年拟合全局及半日单纯形，后半年从0 25% 50% 100%选半日分支比例。'],
        ['Chronos-2 TiRex-2 TimesFM 2.5','各为独立基础模型家族；历史分别为28天 14天 14天。TiRex使用CPU原生，其余主要使用GPU。'],
        ['TimesFM 3 Toto2 PatchTST-FM-r2','历史14或21天 14天 21天；单场站联合模式退化为单变量，预测相同行保留但不作独立模型证据。']],
        [1.6,5.5],10)
    rep.p('设施总功率的两项NLR适配均由开发期选为全局单纯形，半日分支比例为0；IT功率7天比例为25%。因此不能把设施适配收益归因于保留了半日专门化。2024与2025使用同一冻结选择；本轮没有在NLR上微调全部神经权重。')
    rep.new('更强泛化能力评估的后续设计与方法依据')
    rep.p('当前结果足以比较已冻结方案在一个新设施和多个时间段的表现。若要形成广泛泛化、严格零样本或正式显著性结论，应在下一轮数据评分前冻结更强设计，不重复利用已查看的2024和2025选择参数。以下为建议设计，尚未执行。')
    for t in [
        '扩大外部场站覆盖，并保留至少一个完全未参与建模与选择的独立场站；多场站可采用留一场站验证，模型选择层与最终测试层分开。[4]',
        '在基础模型训练或发布之后收集新的实际功率，记录模型版本与数据时间；无法核实预训练语料时，避免宣称已排除语料污染。[5]',
        '在新确认期进行所有合格日期的滚动多步预测，同时报告全部日期、缺失分组和各时距；起报资格只依赖历史，7天标签不得越过训练或验证边界。[1]',
        '在足够密集、足够长且依赖条件适合的误差序列上，预先规定7/14/28天时间块敏感性分析，保留跨日相关性；不能把当前9或7个月的抽样直接当作已校准的大样本显著性证明。[3]',
        '提前规定曲线WAPE为主指标，日均、峰值与区间覆盖为约束，并比较全部既定历史和强单模型基线。只有新开发期选定的唯一配置才能进入新确认期。']:
        rep.p(t)
    rep.sub('参考依据')
    refs=[
        ('[1] Hyndman RJ Athanasopoulos G Forecasting Principles and Practice 第5章 时间留出 多步滚动评价与区间分数','https://otexts.com/fpp3/toolbox.html'),
        ('[2] Hyndman RJ WAPE Weighted Absolute Percentage Error 2025年8月8日','https://robjhyndman.com/hyndsight/wape.html'),
        ('[3] Politis DN Romano JP The Stationary Bootstrap JASA 1994 89 428 1303至1313','https://doi.org/10.1080/01621459.1994.10476870'),
        ('[4] Aksu T 等 GIFT Eval A Benchmark For General Time Series Forecasting Model Evaluation 2024','https://arxiv.org/abs/2410.10393'),
        ('[5] Meyer M 等 Rethinking Evaluation in the Era of Time Series Foundation Models 2026年2月25日版本','https://arxiv.org/abs/2510.13654')]
    for text,url in refs:
        pp=rep.p(text,size=10);rep.link(pp,'  查看原文',url)
    rep.p('文献用于说明评价原则，并不替本项目证明精度或统计显著性。项目结果均取自实际预测与重算记录，补充稳定性诊断属于事后审查。',size=10)
    point=[('wape','曲线\nWAPE %',4),('mae','MAE',2),('rmse','RMSE',2),('p95_ape','P95\nAPE %',3),('hit2_percent','命中\n2% %',2),('hit5_percent','命中\n5% %',2),('absolute_bias_percent','绝对\n偏差 %',4),('relative_curve_error_reduction_percent','曲线误差\n下降 %',3)]
    daily=[('daily_wape','日均\nWAPE %',4),('daily_mae','日均\nMAE',2),('daily_rmse','日均\nRMSE',2),('daily_hit2_percent','日均命中\n2% %',2),('daily_hit5_percent','日均命中\n5% %',2),('peak_mape','峰值\nMAPE %',4),('peak_time_mae_minutes','峰时误差\n分钟',1),('peak_underestimate_percent','峰值\n低估 %',2),('relative_daily_error_reduction_percent','日均误差\n下降 %',3)]
    extras=[('daily_wape_valid_slots_supplement','有效格日均\nWAPE %',4),('MASE','MASE',4),('RMSSE','RMSSE',4),('lower95_gain_pp','收益区间\n下限 pp',4),('upper95_gain_pp','收益区间\n上限 pp',4)]
    ivfields=[('interval80_coverage_percent','覆盖\n%',2),('interval80_mean_width','平均\n宽度',2),('interval80_winkler80','Winkler',2),('interval80_below_percent','低于区间\n%',2),('interval80_above_percent','高于区间\n%',2)]
    cases=0
    for phase in ['test','confirm']:
        for target in ['facility_total_kw','it_power_kw']:
            for h in [1,7]:
                cases+=1;case=[r for r in nlr if r['phase']==phase and r['target']==target and r['horizon_days']==h];case.sort(key=lambda r:ORDER.index(r['model']))
                stem=('2024测试' if phase=='test' else'2025确认')+' '+('设施总功率' if target=='facility_total_kw' else'IT功率')+' '+str(h)+'天'
                cov=next(c for c in a['actual_evaluation_coverage'] if c['phase']==phase and c['target']==target and c['horizon_days']==h)
                rep.new('附表'+str(cases)+' '+stem+' 曲线指标')
                rep.p(f'单位为kW；有效点{cov["valid_points"]}，完整日均目标{cov["valid_complete_days"]}，起报{cov["origins"]}。基线为'+('前一天同刻。' if h==1 else'前三天同刻均值。'),size=10)
                rep.metric_table(case,point,[1.62]+[.685]*8)
                rep.p('曲线WAPE MAE RMSE P95和绝对偏差越低越好；命中率越高越好。收益为相对本任务固定基线的误差降低，负值代表退步。表中单模型实测名次不用于本轮重新选参。',size=10)
                rep.new('附表'+str(cases)+' '+stem+' 日均与峰值指标')
                rep.p('日均只采用完整96点日期；MAE RMSE为kW，百分比列单位为%，峰时误差为分钟。峰值使用原有效格与首个最大值定义。',size=10)
                rep.metric_table(case,daily,[1.62]+[.61]*9,9)
                rep.p('日均峰值任务与曲线任务分别判断，不能只凭曲线WAPE最低断言所有指标领先。',size=10)
                rep.new('附表'+str(cases)+' '+stem+' 补充统计与经验区间')
                rep.metric_table(case,extras,[2.4,.94,.78,.78,1.10,1.10],9.4)
                iv=[r for r in case if 'interval80_coverage_percent' in r]
                rep.sub('80%经验区间实际表现')
                rep.metric_table(iv,ivfields,[2.4,.90,.95,.95,.95,.95],9.4)
                rep.p('原重采样区间是月簇探索性分位区间，pp为百分点，不具有已验证的95%总体覆盖保证。只有已校准的主方案及历史对照具有本表区间；其他单模型未计算区间，不伪造数值。所有本表零真值APE点数为0。',size=9.5)
    for h in [1,7]:
        case=[r for r in rows if r['dataset']=='SeaGuang' and r['horizon_days']==h]
        rep.new('海光寺附表 '+str(h)+'天每日电量')
        rep.p('目标单位kWh。'+('测试日期9月23日至30日，共8个次日目标。' if h==1 else'两个7天起报，目标日期重叠，不是14个独立日期。')+' 不存在日内峰值，不能与NLR或Google曲线混排。',size=10)
        for r in case:
            if r['model'] in ['last','week','mean3']:SHORT[r['model']]={'last':'前一日电量','week':'上周同日电量','mean3':'前三日电量均值'}[r['model']]
        rep.metric_table(case,point,[1.62]+[.685]*8)
        rep.sub('日均统计与区间')
        sea_fields=[('daily_mae','每日\nMAE',2),('daily_rmse','每日\nRMSE',2),('daily_hit2_percent','每日命中\n2% %',2),('daily_hit5_percent','每日命中\n5% %',2),('interval80_coverage_percent','区间覆盖\n%',2),('interval80_mean_width','区间宽度\nkWh',2),('interval80_winkler80','Winkler\nkWh',2)]
        rep.metric_table(case,sea_fields,[1.8]+[.757]*7,9.3)
        rep.p('日电量本身就是每日目标，其日均WAPE与任务WAPE相同，不重复当作两种独立成绩。日尺度单纯形α=0，实际退回前一日电量；峰值幅度 时间和低估率不适用。月份此前已暴露，不能声称独立确认。',size=10)
    # Reset shared historical baseline short names for the original Google scope.
    SHORT.update(last='前一天同刻',week='上周同刻',mean3='前三天同刻均值')
    for h in [1,7]:
        case=[r for r in rows if r['dataset']=='GoogleCluster_same_source' and r['horizon_days']==h]
        rep.new('ClusterData附表 '+str(h)+'天历史算力增益对照')
        rep.p('单位为归一化功率利用率，并非kW；只有cell-a五域有真实CPU覆盖。all57包括其余52域原推荐不变的结果，缺失算力不作为真实零值。',size=10)
        vals=[]
        for r in case:vals.append([r['scope']+' '+SHORT[r['model']]]+[value(r.get(k),d) for k,label,d in point[:-1]])
        rep.table(['范围与具体模型']+[label for k,label,d in point[:-1]],vals,[2.05]+[.72]*7,9.2,case,[k for k,label,d in point[:-1]])
        rep.sub('日均与峰值')
        vals=[]
        for r in case:vals.append([r['scope']+' '+SHORT[r['model']]]+[value(r.get(k),d) for k,label,d in daily[:8]])
        rep.table(['范围与具体模型']+[label for k,label,d in daily[:8]],vals,[2.0]+[.635]*8,9.0,case,[k for k,label,d in daily[:8]])
        rep.p('历史算力差分系数25%由开发期选择并冻结，测试退步后未改成其他系数宣传最优结果。该方案未采纳，Google仍保留第48轮推荐。同源测试此前已暴露，没有独立场站意义。',size=10)
    rep.new('三项主要方案的未来七天逐时距结果')
    import csv
    leadrows=[]
    for phase in ['test','confirm']:
        with (Q/f'NLR_{phase}_逐时距指标.csv').open(encoding='utf-8-sig',newline='') as f:
            source=list(csv.DictReader(f))
        for target in ['facility_total_kw','it_power_kw']:
            rep.sub(('2024测试' if phase=='test' else'2025确认')+' '+('设施总功率' if target=='facility_total_kw' else'IT功率'))
            vals=[]
            for name in ['mean3','HDSF_Google_frozen','R48_Google_frozen','HDSF_NLR_refit']:
                vals.append([SHORT[name]]+[value(float(next(r['wape'] for r in source if r['target']==target and r['model']==name and r['horizon_days']=='7' and r['lead_day']==str(k))),4) for k in range(1,8)])
            rep.table(['具体模型']+['D'+str(k)+' WAPE %' for k in range(1,8)],vals,[1.62]+[.78]*7,9.4)
    rep.p('D1至D7分别是起报后第1至第7个24小时窗口，不是新增独立预测实验。完整逐起报、逐时距数据保留在实验结果目录。',size=10)
    out.parent.mkdir(parents=True,exist_ok=True);doc.save(out)
    # Structural numeric QA uses each original row as a canonical source;
    # formatting rounds only for display. It never edits or rescales source rows.
    reopened=Document(out);tables=reopened.tables;assert len(tables)>20
    text='\n'.join(p.text for p in reopened.paragraphs)+'\n'+'\n'.join(c.text for t in tables for row in t.rows for c in row.cells)
    for r in rows:
        assert value(r['wape']) in text
    save(audit/'document_content_verification.json',dict(all_passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        document_sha256=sha(out),data_rows=130,numeric_values_bound=len(rep.records),table_count=len(tables),
        title_at_front=True,result_table_precedes_methods=True,original_predictions_and_selection_unchanged=True,
        numerical_records=rep.records,source_sha256=a['reference_source_sha256'],sections=rep.sections))
    print(json.dumps(dict(all_passed=True,rows=130,numeric_values=len(rep.records),tables=len(tables),document=str(out)),ensure_ascii=True))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('audit');ap.add_argument('output');ap.add_argument('--validate-only',action='store_true');a=ap.parse_args()
    if a.validate_only:
        R,Q,rows,audit=load(Path(a.root),Path(a.audit));print(json.dumps(dict(all_passed=True,rows=len(rows),verdict=audit['verdict']),ensure_ascii=False))
    else:create(Path(a.root),Path(a.audit),Path(a.output))
