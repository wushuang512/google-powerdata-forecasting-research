from pathlib import Path
import json,csv,sys
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'Google PowerData 2019 精度改进实验汇总.docx'
def js(name):return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
def rows(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def f(row,key):return float(row[key])

def main():
    s=js('selection24.json');s7=js('selection7.json');audit=js('data_audit.json');st=js('stability.json');loco=js('loco_metrics.json')
    t={r['model']:r for r in rows('test24_metrics.csv')};t7={r['model']:r for r in rows('test7_metrics.csv')};primary=s['primary24'];p=t[primary];b=t['last'];fam=s['families']
    labels={'last':'前一天同刻','week':'上周同刻','mean3':'前三天同刻均值','mean7':'前七天同刻均值','chronos_zero':'Chronos 2 零样本','chronos_cov':'Chronos 2 历史生产特征','chronos_joint_raw':'Chronos 2 分量联合预测','chronos_joint_physical':'Chronos 2 物理一致性修正','ensemble':'验证集加权组合','ensemble_peak':'组合与峰值先验','ensemble7':'七天验证集加权组合',fam['history']:'平滑历史与水平修正',fam['ridge']:'Ridge 残差回归',fam['lgb_load']:'LightGBM 历史负荷',fam['lgb_prod']:'LightGBM 历史生产特征',fam['patch_plain']:'PatchTST 类无约束模型',fam['patch_physical']:'PatchTST 类约束模型',fam['chronos_lora']:'Chronos 2 领域 LoRA'}
    label=lambda name:labels.get(name,name)
    doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=Cm(1.8);sec.bottom_margin=Cm(1.7);sec.left_margin=Cm(2);sec.right_margin=Cm(2)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
        style=doc.styles[name];style.font.name='Calibri';style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑');style.font.color.rgb=RGBColor(0,0,0)
    normal=doc.styles['Normal'];normal.font.size=Pt(10);normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.line_spacing=1.15
    doc.styles['Title'].font.size=Pt(22);doc.styles['Heading 1'].font.size=Pt(15);doc.styles['Heading 2'].font.size=Pt(11.5)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    rr=footer.add_run();fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');rr._r.append(fld);rr.font.size=Pt(9)
    def ptext(text,style=None):return doc.add_paragraph(text,style)
    def heading(text):doc.add_heading(text,level=1)
    def page(text):doc.add_page_break();heading(text)
    def table(headers,data,widths):
        tb=doc.add_table(rows=1,cols=len(headers));tb.alignment=WD_TABLE_ALIGNMENT.CENTER;tb.autofit=False
        for col,w in zip(tb.columns,widths):col.width=Cm(w)
        for cell,txt,w in zip(tb.rows[0].cells,headers,widths):cell.text=txt;cell.width=Cm(w)
        repeat=OxmlElement('w:tblHeader');tb.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in data:
            for cell,txt,w in zip(tb.add_row().cells,row,widths):cell.text=str(txt);cell.width=Cm(w)
        for i,row in enumerate(tb.rows):
            cant=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cant)
            for j,cell in enumerate(row.cells):
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;pr=cell._tc.get_or_add_tcPr()
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'DDEBF7' if i==0 else ('F4F7FA' if i%2==0 else 'FFFFFF'));pr.append(shade)
                margins=OxmlElement('w:tcMar')
                for side in ['top','left','bottom','right']:
                    a=OxmlElement('w:'+side);a.set(qn('w:w'),'85');a.set(qn('w:type'),'dxa');margins.append(a)
                pr.append(margins)
                borders=OxmlElement('w:tcBorders')
                for side in ['top','left','bottom','right']:
                    a=OxmlElement('w:'+side);a.set(qn('w:val'),'single');a.set(qn('w:sz'),'4');a.set(qn('w:color'),'D9D9D9');borders.append(a)
                pr.append(borders)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after=Pt(0);para.paragraph_format.space_before=Pt(0);para.paragraph_format.line_spacing=1.08
                    para.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in para.runs:run.font.size=Pt(9);run.bold=i==0
        doc.add_paragraph().paragraph_format.space_after=Pt(0)
    def fig(name,caption):
        doc.add_picture(str(ROOT/'figures'/name),width=Cm(16.6));pp=ptext(caption);pp.paragraph_format.space_after=Pt(8)
        for r in pp.runs:r.font.size=Pt(9)
    improvement=(f(b,'wape')-f(p,'wape'))/f(b,'wape')*100
    doc.add_heading('Google PowerData 2019\n精度改进实验汇总',0)
    ptext('2026年9月29日   本地实测   15分钟粒度   57个供电域')
    ptext(f'本轮以第23—31天为次日测试期，通过开发集选定的主模型为“{label(primary)}”。测试宏平均 WAPE 为 {f(p,"wape"):.4f}%，相较复现的前一天同刻基线 {f(b,"wape"):.4f}%，误差减少 {improvement:.2f}%，但没有超过历史报告的2.416%。本轮取得的是相对简单基线的改善和峰值权衡证据，尚未实现整体曲线精度的新纪录。全部结果来自本机实际训练和预测。')
    keys=list(dict.fromkeys(['last','chronos_zero',fam['chronos_lora'],'chronos_joint_physical','ensemble','ensemble_peak',primary]))
    table(['方法','曲线 WAPE','日均 WAPE','峰值 MAPE'],[[label(k),f'{f(t[k],"wape"):.3f}%',f'{f(t[k],"daily_wape"):.3f}%',f'{f(t[k],"peak_mape"):.3f}%'] for k in keys],[7.1,3.1,3.1,3.1])
    ptext(f'主模型在 {st["domains_improved_vs_last"]}/57 个供电域和 {st["days_improved_vs_last"]}/9 个测试日上优于前一天同刻基线。其时点绝对百分比误差的95分位为 {f(p,"p95_ape"):.3f}%，基线为 {f(b,"p95_ape"):.3f}%。峰值必须单独比较：整体曲线改善不等于峰值同时改善。')
    ptext('创新尝试包括：基于可靠历史生产负荷的协变量预测，Chronos 2 的低秩领域适配，总量与分量的物理一致性投影，带容量和多尺度损失的 PatchTST 类模型，以及保持日均值的峰值先验修正。每个模块都保留对照结果；没有收益的模块不强行作为最终方案。')
    ptext('结论边界：同一31天公开数据此前已被研究使用，本轮虽隔离了开发与测试过程，仍不属于全新月份的独立验证。历史报告中的2.416%为既有记录，本次未重跑 TiRex 2 等权组合，不能据此声称已完成同环境的全面超越验证。')

    page('数据质量与实验规则')
    ptext('数据来自 Google 官方 PowerData 2019，包含10个 cell、57个供电域、2019年5月31天的5分钟平均功率利用率。原始508896条记录先按时间排序，再连续三个点取均值，形成169632个15分钟点。原始压缩文件通过MD5校验，解压CSV与原始内容逐字节对应。')
    table(['项目','数量或规则'],[['总功率异常原始点',str(audit['bad_measured_5min'])],['生产负荷低可信原始点',f'{audit["bad_production_5min"]} 约22.90%'],['次日测试有效目标',str(audit['valid_test24'])],['异常处理','历史输入仅向前填补 目标不计分'],['生产信息缺失','零占位加质量标记 或 NaN 掩码'],['量纲','利用率 不等同于 kW MW 或 kWh']],[7.2,9.2])
    ptext('官方PDF对质量标记的文字存在歧义，官方分析笔记使用 NOT bad_measurement_data 和 NOT bad_production_power_data 过滤。因此本轮将 true 视为低可信。15分钟总功率目标仅在三个原始点全部有效时评分；生产负荷质量不会额外删除总功率评估目标。')
    table(['阶段','训练上限','预测起点 日末','目标'],[['开发次日','第14天','14 16 18 20','次日96点'],['测试次日','第22天','22至30','第23至31天'],['开发七天','第14天','14 15','未来672点'],['测试七天','第22天','22 23 24','未来672点']],[3.1,3.1,5.5,4.7])
    ptext('所有超参数、训练轮数与组合权重均根据开发集确定。最终监督模型仅用第22天及以前的目标训练，测试期间权重固定；后续起点可以使用届时已经观测到的历史。未来实际总负荷、生产负荷、CPU利用率或天气均未作为已知输入。七天递推仅加入自身预测。')
    ptext('主指标先分别计算各供电域的 WAPE，再对57条序列等权平均。日均值和峰值只在同一组有效时点上比较。峰值时间使用首次最大值，平台型曲线可能使该指标不稳定。训练与测试点之间相关，不能将五万余测试点视为独立样本。')

    page('模型与约束设计')
    ptext('Chronos 2 使用 Amazon 官方固定版本权重，SHA256 已核对。上下文上限为过去14天。领域适配采用 LoRA，学习率0.0001，批大小16，训练上下文672点，预测长度96点；在100步与300步中仅按开发结果选择，再用前22天重训。')
    ptext('PatchTST 类实验采用本地实现的轻量残差模型：672点上下文、24点分块、步长12、两层编码器、64维表示、4个注意力头。它借鉴分块与可逆归一化思想，并非官方 PatchTST 的严格复现。比较无约束版本及加入容量惩罚、日均和峰值辅助损失的版本；候选训练轮数为4、8、16、24。')
    doc.add_heading('可验证的物理一致性',level=2)
    ptext('在可靠记录中，生产负荷利用率不高于总功率利用率。将剩余分量定义为总量减生产分量，对总量、生产分量、剩余分量进行联合预测，再求解与原预测最接近的可行结果：')
    ptext('总量 = 生产分量 + 剩余分量；生产分量 ≥ 0；剩余分量 ≥ 0；总量 ≤ 1。')
    ptext('投影最小化三个输出相对原始预测的平方偏差。对历史可靠比例不足20%的供电域，退回总量单变量预测。剩余分量是数学差值，并不等于制冷用电；这不是已知设备参数下的热力学模型。数据不含真实设备容量、气象和任务排期，未虚构这些输入。')
    doc.add_heading('组合与峰值先验',level=2)
    weights=[(label(k),f'{w*100:.2f}%') for k,w in s['weights'].items() if w>.00001]
    table(['次日组合成员','验证集选定权重'],weights,[11.2,5.2])
    pk=s['peak_adjustment'];ptext(f'组合系数非负且和为1，在开发集上最小化宏平均绝对误差。峰值模块采用 {pk["anchor"]} 历史峰值先验，强度为 {pk["beta"]:.2f}，对曲线顶部进行零均值修正；它是工程先验，不是硬性物理定律。约束选择允许开发曲线WAPE最多增加0.01个百分点，并优先减少峰值误差。')

    page('次日曲线与逐日表现')
    fig('model_comparison.png','图1 次日测试误差比较 绿色为开发集选定的主模型 数值越低越好')
    fig('daily_error.png','图2 九个次日测试目标日的宏平均 WAPE')
    ptext(f'主模型 MAE 为 {f(p,"mae"):.6f}，单位为利用率。峰值幅值 MAPE 为 {f(p,"peak_mape"):.3f}%，峰值时间平均绝对偏差为 {f(p,"peak_time_mae_minutes"):.1f} 分钟。所有候选最终结果可在 test24_metrics.csv 中查看，所有开发配置均保留在 development_all.csv。')

    page('消融与精度权衡')
    pairs=[('历史生产信息','chronos_zero','chronos_cov'),('领域适配','chronos_zero',fam['chronos_lora']),('分量一致性投影','chronos_joint_raw','chronos_joint_physical'),('Patch模型约束损失',fam['patch_plain'],fam['patch_physical']),('峰值先验','ensemble','ensemble_peak')]
    data=[]
    for title,a,c in pairs:
        data.append([title,f'{f(t[a],"wape"):.3f} → {f(t[c],"wape"):.3f}',f'{f(t[c],"wape")-f(t[a],"wape"):+.3f}',f'{f(t[a],"peak_mape"):.3f} → {f(t[c],"peak_mape"):.3f}'])
    table(['改动','测试 WAPE 前后','差值 百分点','峰值 MAPE 前后'],data,[4.5,4.3,3.3,4.3])
    ptext('表中差值为改动后减改动前，负值表示曲线误差降低。这些对照评估的是本次配置，不代表对应模型家族的最优能力。Patch两种损失各自选定了开发最优轮数，因此对照同时含训练轮数选择的影响。')
    gain_phys=f(t['chronos_joint_physical'],'wape')-f(t['chronos_joint_raw'],'wape')
    ptext(f'物理投影使联合预测满足分量和容量约束，但测试曲线 WAPE 的变化为 {gain_phys:+.4f} 个百分点。应将“约束满足”与“预测精度改善”分开论证。若测量估计误差或领域分布变化使分量预测不可靠，硬性一致性修正也可能放大偏差。')
    ptext('本轮峰值修正保留曲线日均值（容量截断前），属于结构一致的后处理。它不保证峰值出现时间正确，也不应因峰值误差下降而掩盖完整曲线误差增加。最终同时交付主精度模型与峰值组合方案。')
    ptext(f'历史报告最佳等权组合为2.416%。本次主模型数值为 {f(p,"wape"):.4f}%，两者差值为 {f(p,"wape")-2.416:+.4f} 个百分点。基线与有效样本数已复现，但旧组合权重、软件版本及完整预测文件未在本轮重跑，该对比仅作历史参照。')
    ptext('创新定位建议：将工作表述为“质量感知的领域适配与物理一致性集成预测”，重点展示质量掩码、域内适配、分量约束及峰值权衡的实证效果。当前结果不足以独立支撑一种全新通用算法或长期跨季节适应能力的结论。')

    page('七天预测与区间校准')
    k7=list(dict.fromkeys(['mean3','week','chronos_zero',fam['chronos_lora'],'chronos_joint_physical','ensemble7',s7['primary7']]))
    table(['七天方法','曲线 WAPE','日均 WAPE','峰值 MAPE'],[[label(k),f'{f(t7[k],"wape"):.3f}%',f'{f(t7[k],"daily_wape"):.3f}%',f'{f(t7[k],"peak_mape"):.3f}%'] for k in k7],[7.1,3.1,3.1,3.1])
    ptext(f'七天模型单独依据第14、15天结束后的开发预测选择，主模型为“{label(s7["primary7"])}”。第22、23、24天三个测试起点彼此重叠，不是三个独立周样本。只据现有31天数据，不开展15—30天预测的泛化承诺。')
    ir=rows('interval_metrics.csv');picked=[r for r in ir if r['kind']=='development_absolute_residual_80' or (r['model']=='chronos_zero' and r['horizon_days']=='7')]
    table(['对象与区间','覆盖率','平均区间宽度'],[[f'{r["horizon_days"]}天 {"主模型残差校准" if r["kind"]=="development_absolute_residual_80" else ("Chronos原始" if r["kind"]=="raw_80" else "Chronos扩展校准")}',f'{float(r["coverage_percent"]):.2f}%',f'{float(r["mean_width"]):.5f}'] for r in picked],[9.4,3.3,3.7])
    ptext('主模型区间按每个供电域的开发绝对残差80分位构建，并截断至0—1。Chronos原始10%—90%分位区间另作开发残差扩展。区间覆盖率在测试集实测，宽度越大并不必然越好，必须结合覆盖率比较。')
    ptext('上述方法属于经验校准。开发数据同时用于模型选择且时间点相关，七天开发窗口还重叠，因此不宣称具有严格的80%有限样本覆盖保证。没有使用测试残差来调整区间。')

    page('跨供电域表现与稳定性')
    fig('cell_error.png','图3 各 cell 内供电域 WAPE 的平均值 此图为同一训练模型的分组测试')
    fig('example_curve.png','图4 代表曲线 选取相对基线改善位于中位数的供电域及第一个测试日')
    ci=st['three_day_block_bootstrap_difference_pp_95']
    ptext(f'按连续三天组成的时间块重采样2000次，基线减主模型的WAPE差值探索性95%区间为 [{ci[0]:.3f}, {ci[1]:.3f}] 个百分点。仅九天样本，不能将该区间当作跨月份的可靠性保证。')
    ptext(f'额外对负荷特征 LightGBM 进行10次逐 cell 留出训练，测试 WAPE 为 {loco["wape"]:.3f}%。预测时可使用目标 cell 的过去负荷，但其训练目标被排除。此检查只验证该监督基准的跨 cell 敏感性，不代表整套前沿组合已完成留一 cell 验证。')

    page('本地部署与复现')
    ptext('桌面项目文件夹为 GooglePowerData2019_预测研究_20260929。原始数据副本在 google-powerdata-2019；原始任务文档在参考材料；experiment 内包含独立环境、模型、代码和全部产出。原 E 盘数据仍保留，桌面副本作为后续实验输入。')
    table(['目录或文件','用途'],[['experiment/.venv','独立 Python 环境 含 GPU 依赖'],['experiment/models','官方权重 开发检查点 最终训练模型'],['experiment/code','清洗 训练 选模 测试 校准 推理脚本'],['experiment/results','逐点预测 指标 选模记录 验证及运行日志'],['experiment/figures','本报告使用的图表'],['experiment/docs','官方接口及模型版本资料'],['预测未来一天.cmd 与 预测未来七天.cmd','调用保存的冻结模型 不重新训练']],[7.1,9.3])
    ptext('运行方式：在桌面文件夹双击对应预测入口。默认用数据集末日2019年5月31日作为起报时间，预测2019年6月1日开始的曲线，结果保存至 experiment/results/deployment_forecast_origin31_1day.csv 或对应7day文件。这是历史数据末端的外推示例，不是对当前日期的实时电力预测。')
    ptext('复现实验依次运行 prepare.py、classical.py、deep_models.py all、selection.py、evaluate.py、diagnostics.py 和 make_figures.py。使用本目录 .venv/Scripts/python.exe；推理入口 predict_next.py 支持 --days 1 或7、--origin 14至31。requirements-lock.txt 保存版本。环境已在本机 RTX 4060 Laptop GPU 上执行验证。')
    ptext('目录已包含独立 Python 基础运行时和训练依赖，显卡运行仍需要电脑上的 NVIDIA 驱动。迁移到另一台电脑或更改目录时，应检查解释器路径、LoRA基础权重路径与驱动兼容性，必要时按锁定依赖重建环境。新增月份或供电域时需扩展数据读取与切分逻辑，并重新完成独立验证。')
    doc.add_heading('主要限制与下一步',level=2)
    ptext('当前数据仅31天，不能证明跨季节、长期规划或真实园区绝对功率预测能力；生产分量为估计利用率，并非完整分项电表。预训练模型与公开评估数据是否重合尚未排除。后续优先补充未参与选模的新月份，加入可提前获得的任务排期与设备状态，再测试真实功率、峰值容量及区间校准。')
    doc.add_heading('主要来源',level=2)
    for text in ['Google 官方数据与许可 https://github.com/google/cluster-data/blob/master/PowerData2019.md','Chronos 2 官方实现 https://github.com/amazon-science/chronos-forecasting','Chronos 2 固定模型版本 29ec3766d36d6f73f0696f85560a422f50e8498c https://huggingface.co/amazon/chronos-2','PatchTST 思想参考 https://github.com/yuqinie98/PatchTST','本地参考 9.28算力中心需求预测实验.docx 与 算力负荷特性分析与建模.docx']:
        pp=ptext(text);pp.paragraph_format.space_after=Pt(3)
        for r in pp.runs:r.font.size=Pt(8)
    doc.core_properties.title='Google PowerData 2019 精度改进实验汇总';doc.core_properties.author='';doc.core_properties.subject='严格时间隔离下的预测精度与物理约束实测'
    doc.save(OUT)
    print(OUT)

if __name__=='__main__':main()
