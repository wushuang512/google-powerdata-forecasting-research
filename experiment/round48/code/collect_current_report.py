from pathlib import Path
import csv,json,hashlib,sys,datetime,zipfile
import numpy as np
from docx import Document
from lxml import etree
T=Path(sys.argv[1]); R=T/'experiment/round48'; D=R/'docs/final_report'; D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
reference=T/'Google PowerData 2019 三项预测任务与模型比较汇报_加入HDSF.docx'
q=read(T/'experiment/round44/docs/integrated_report_qa/verification.json');assert sha(reference)==q['document_sha256']
doc=Document(reference);s=doc.sections[0]
parts=[]
with zipfile.ZipFile(reference) as z:
    for n in z.namelist():parts.append(dict(path=n,sha256=hashlib.sha256(z.read(n)).hexdigest(),bytes=len(z.read(n)),scope='editable' if n in ['word/document.xml','word/footer1.xml','word/settings.xml'] else 'preserve_only'))
ev=dict(reference=str(reference),reference_sha256=sha(reference),reference_QA=str(T/'experiment/round44/docs/integrated_report_qa/verification.json'),reference_pages_reviewed=list(range(1,10)),section_xml=etree.tostring(s._sectPr).decode(),geometry={k:getattr(s,k).inches for k in ['page_width','page_height','left_margin','right_margin','top_margin','bottom_margin','header_distance','footer_distance']},parts=parts,table_patterns=[dict(index=i,rows=len(x.rows),columns=len(x.columns),headers=[c.text for c in x.rows[0].cells]) for i,x in enumerate(doc.tables)])
save(D/'template_evidence.json',ev)
(D/'artifact.md').write_text('''# 本轮详细汇报模板约定
参考为 D盘顶层 Google PowerData 2019 三项预测任务与模型比较汇报_加入HDSF.docx。SHA、部件、节XML及版式证据见 template_evidence.json。原文档不变；本次实际查看其全部9张已有渲染，哈希相等。
页面沿用单节横向11.6903×8.2701英寸，左右0.65，上0.55，下0.50英寸。Normal Calibri/微软雅黑10.5pt、1.1倍行距、段后7pt；Title21pt，Heading1 15pt、黑色。表格克隆原第4表tblPr，DCEAF2浅蓝表头、F5F7F9隔行，D9D9D9边框、单元格内距、重复表头、数值居中，禁止固定行高。主要正文沿用原字号，所有数字最多保留6位小数，未保存值标记—。源表格组件允许按新增指标重新分配列宽，不能通过缩小字号塞入。
正文word/document.xml全部为用户要求的新的详细报告内容插槽，保留节XML。依次结论与三任务、模型明确名称、完整指标、区间、开发与测试、训练及失败原因、原字段处理、复现说明、全部历史指标附表与模型字典。用户要求所有指标，允许扩展原分页模式；不硬限9页。页脚文字改为当前报告短名与日期，原PAGE字段保留，Word只读导出时更新显示值。settings允许仅设置updateFields以在打开时刷新；其他ZIP部件须逐字节保留。无新增文本框、控件、脚注或外链关系。
最终每页实际查看，允许正文授权重排与新增页，不允许页面尺寸、保留样式、关系、opaque部件漂移。内部PDF和页图不作交付。
''',encoding='utf-8')
data={}
for key,path in [('history',T/'experiment/round44/results/hdsf_comparison.json'),('r46',T/'experiment/round46/results/summary.json'),('r47',T/'experiment/round47/results/summary.json'),('r48',R/'results/summary.json'),('selection48',R/'results/selection.json'),('config',T/'experiment/round47/deployment/selected_configuration.json'),('audit',T/'experiment/round44/docs/causes_data_report_20261009/data_audit.json'),('reference_metric',T/'experiment/round44/results/metric_reference_audit.json')]:data[key]=read(path)
data['historical_rows']=data.pop('history')['unique_rows'];assert len(data['historical_rows'])==114
data['native']=rows(T/'experiment/round46/results/基础模型微调前后全部指标.csv')
data['native47']=rows(T/'experiment/round47/results/新训练基础模型_同设备前后指标.csv')
data['ablation']=rows(T/'experiment/round47/results/上一轮固定消融_仅事后解释.csv')
data['dev48']=rows(R/'results/稳健校正开发期全部候选.csv')
data['intervals']=rows(T/'模型预测区间结果.csv')
data['clear47']=rows(T/'experiment/round47/results/清晰模型名称_全指标比较.csv')
data['r45']=read(T/'experiment/round45/results/summary.json')
data['r45rows']=rows(T/'experiment/round45/results/统一评价结果.csv')
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
for rnd in [46,47,48]:
    v=read(T/f'experiment/round{rnd}/results/verification.json');ck(f'R{rnd}_scientific_verification',v['all_passed'])
ck('R47_selection_unchanged',sha(T/'experiment/round47/results/selection.json')==data['config']['frozen_research_selection_sha256'])
with np.load(T/'experiment/round47/results/predictions.npz') as z:
    for h in [1,7]:
        ex=read(T/f'experiment/round47/deployment/export_H{h}.json');p=z[f'frozen_candidate_H{h}'];lo=z[f'frozen_candidate_lower80_H{h}'];hi=z[f'frozen_candidate_upper80_H{h}']
        rr=rows(Path(ex['curve_csv']));dd=rows(Path(ex['daily_csv']));pred=np.array([float(x['prediction_normalized']) for x in rr]);flags=np.array([int(x['evaluation_valid']) for x in rr],bool)
        ck(f'H{h}_curve_export_hash',sha(Path(ex['curve_csv']))==ex['curve_sha256']);ck(f'H{h}_daily_export_hash',sha(Path(ex['daily_csv']))==ex['daily_sha256'])
        ck(f'H{h}_row_counts',len(rr)==p.size==ex['curve_rows'] and len(dd)==ex['daily_rows'])
        ck(f'H{h}_exact_prediction',np.array_equal(pred,p.ravel()))
        ck(f'H{h}_intervals',np.array_equal(np.array([float(x['lower80']) for x in rr]),lo.ravel()) and np.array_equal(np.array([float(x['upper80']) for x in rr]),hi.ravel()) and (lo<=hi).all())
        ck(f'H{h}_bounds_finite',np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all())
        ck(f'H{h}_57_domains',len({(x['cell'],x['PDU']) for x in rr})==57)
        ck(f'H{h}_valid_counts',int(flags.sum())==ex['metrics']['valid_points'] and sum(int(x['complete96_valid']) for x in dd)==ex['metrics']['valid_daily_targets'])
        ck(f'H{h}_daily_from_same_curve',np.array_equal(np.array([float(x['predicted_daily_mean_normalized']) for x in dd]),p.reshape(57,-1,96).mean(-1).ravel()))
        truth=np.array([float(x['actual_normalized']) if x['evaluation_valid']=='1' else np.nan for x in rr]);yp=truth.reshape(p.shape);mm=flags.reshape(p.shape)
        cw=float(np.mean([100*np.abs(yp[i][mm[i]]-p[i][mm[i]]).sum()/np.abs(yp[i][mm[i]]).sum() for i in range(57)]))
        ck(f'H{h}_macro57_WAPE_independent_recompute',abs(cw-ex['metrics']['wape'])<1e-12)
        dtruth=np.array([float(x['actual_daily_mean_normalized']) if x['complete96_valid']=='1' else np.nan for x in dd]).reshape(57,-1);dp=p.reshape(57,-1,96).mean(-1);dm=np.isfinite(dtruth)
        dw=float(np.mean([100*np.abs(dtruth[i][dm[i]]-dp[i][dm[i]]).sum()/np.abs(dtruth[i][dm[i]]).sum() for i in range(57)]))
        ck(f'H{h}_daily_WAPE_independent_recompute',abs(dw-ex['metrics']['daily_wape'])<1e-12)
        ck(f'H{h}_origin_time_bounds',{int(x['origin_day']) for x in rr}==set(range(22,31) if h==1 else range(22,25)) and all(int(x['target_day'])==int(x['origin_day'])+1+k//96 for k,x in enumerate(rr[:96*h])) and all(600000000<=float(x['raw_time'])<=2678700000000 for x in rr))
        data[f'export{h}']=ex
data['evidence']=ev
save(D/'report_data.json',data);save(D/'pre_authoring_verification.json',dict(all_passed=True,checks=checks,check_count=len(checks),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print(json.dumps(dict(all_passed=True,checks=len(checks),historical_unique_models=114,native=len(data['native']),native47=len(data['native47']),interval_records=len(data['intervals']),document_not_yet_created=True)))
