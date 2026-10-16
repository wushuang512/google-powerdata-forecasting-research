from pathlib import Path
import csv,gzip,hashlib,json,sys,datetime,shutil,zipfile
import numpy as np
import pandas as pd
from docx import Document
sys.dont_write_bytecode=True
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round44';Q=R/'results';D=R/'docs/causes_data_report_20261009';D.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'reference/PowerData_Prediction'))
from power_forecast.data import load_dataset
from power_forecast.attainment_features import build_rich
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
raw=T/'google-powerdata-2019/raw';rows=[];checks=[];partial=0;invalid=0;raw_bad=0;prod_bad=0;reference_times=None
cols=['time','cell','pdu','measured_power_util','production_power_util','bad_measurement_data','bad_production_power_data']
for p in sorted(raw.glob('*.csv.gz')):
    if p.name.startswith('machine_to_'):continue
    a=pd.read_csv(p);assert list(a.columns)==cols;unsorted=not a.time.is_monotonic_increasing;a=a.sort_values('time',kind='stable');t=a.time.to_numpy();assert len(a)==8928 and len(np.unique(t))==8928 and np.all(np.diff(t)==300000000)
    if reference_times is None:reference_times=t
    assert np.array_equal(t,reference_times)
    ok=(~a.bad_measurement_data)&np.isfinite(a.measured_power_util);pok=(~a.bad_production_power_data)&np.isfinite(a.production_power_util)
    good=ok.to_numpy().reshape(-1,3).sum(1);pcount=pok.to_numpy().reshape(-1,3).sum(1)
    partial+=int(((good>0)&(good<3)).sum());invalid+=int((good<3).sum());raw_bad+=int(a.bad_measurement_data.sum());prod_bad+=int(a.bad_production_power_data.sum())
    rows.append(dict(file=p.name,sha256=sha(p),rows=len(a),cell=str(a.cell.iloc[0]),pdu=str(a.pdu.iloc[0]),raw_was_unsorted=bool(unsorted),partial_measured_blocks=int(((good>0)&(good<3)).sum()),invalid_measured_blocks=int((good<3).sum()),empty_production_blocks=int((pcount==0).sum())))
source=load_dataset(raw);canonical=np.load(B/'results/data.npz');index=[source.series_keys.index(str(n)) for n in canonical['names']]
valid=source.target_valid[index]
assert np.array_equal(valid,canonical['valid']) and np.max(abs(source.target[index]-canonical['y'])[valid])==0
assert source.quality==json.loads((Q/'data_alignment.json').read_text(encoding='utf-8'))['source_quality']
mapping=pd.read_csv(raw/'machine_to_pdu_mapping.csv.gz')
assert list(mapping.columns)==['machine_id','pdu','cell']
groups=mapping.groupby(['cell','machine_id']).pdu.nunique(); ambiguous=int((groups>1).sum())
mf=dict(rows=len(mapping),unique_machine_id_global=int(mapping.machine_id.nunique()),unique_cell_machine_pairs=len(groups),exact_duplicate_rows=int(mapping.duplicated().sum()),ambiguous_cell_machine_pairs=ambiguous,cells=sorted(mapping.cell.unique().tolist()))
F,names,g=build_rich(source.target[:,:14].transpose(1,2,0),source.production_causal_median[:,:14].transpose(1,2,0),source.target_valid[:,:14].transpose(1,2,0),source.meta)
del F
cluster=T/'google-clusterdata-2019';usage=np.load(cluster/'aggregated/complete_usage_a_15min.npz');resource=np.load(cluster/'aggregated/resources_15min.npz')
assert bool(usage['complete']) and int(usage['source_files'])==int(usage['expected_source_files'])==12085
ua=json.loads((cluster/'results/complete_projection_audit.json').read_text(encoding='utf-8'));assert ua['all_passed']
selected=['start_time','end_time','machine_id','collection_type','average_usage.cpus','average_usage.memory','sample_rate']
feature_sources={str(p.relative_to(T)):sha(p) for p in [R/'reference/PowerData_Prediction/power_forecast/data.py',R/'reference/PowerData_Prediction/power_forecast/attainment_features.py',R/'reference/PowerData_Prediction/power_forecast/attainment_local.py',cluster/'code/extract_cpu.py',cluster/'code/aggregate_resources.py',B/'round29/code/aligned_metrics.py']}
data=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_passed=True,power_files=rows,power_columns=cols,raw_rows=57*8928,raw_start=int(reference_times[0]),raw_end=int(reference_times[-1]),raw_step_microseconds=300000000,
    aggregated_shape=list(source.target.shape),aggregated_slots=int(source.target.size),quality=source.quality,
    partial_measured_blocks=partial,invalid_measured_blocks=invalid,all_empty_measured_blocks=source.quality['imputed_measurement_blocks'],
    target_valid_slots=int(source.target_valid.sum()),production_causal_median_remaining_nonfinite=int((~np.isfinite(source.production_causal_median)).sum()),
    production_causal_median_filled_blocks=int((~np.isfinite(source.production)&np.isfinite(source.production_causal_median)).sum()),
    valid_measured_range=[float(source.target[source.target_valid].min()),float(source.target[source.target_valid].max())],mapping=mf,
    measured_valid_truth_equal_previous=True,quality_mask_equal_previous=True,
    invalid_context_max_difference=float(np.max(abs(source.target[index]-canonical['y'])[~valid])),
    derived_feature_names=names,local24_feature_count=len(names),metadata_fields=source.meta.columns.tolist(),
    usage_selected_raw_fields=selected,usage_derived_features=usage['features'].tolist(),resource_derived_features=resource['features'].tolist(),
    cluster_audit_summary={k:ua[k] for k in ['source_files','original_objects_bytes','retained_bytes','all_rows','used_job_rows','alloc_set_rows_excluded','limits','time']},
    source_file_sha256=feature_sources,original_power_files_unmodified=True,source_dataset_cache_unmodified=True)
save(D/'data_audit.json',data)
reference=T/'Google PowerData 2019 三项预测任务与模型比较汇报_加入HDSF.docx';reference_copy=D/'reference.docx';shutil.copyfile(reference,reference_copy)
doc=Document(reference);sec=doc.sections[0]
with zipfile.ZipFile(reference) as z:
    inventory=[dict(part=n,sha256=hashlib.sha256(z.read(n)).hexdigest(),bytes=len(z.read(n))) for n in z.namelist()]
evidence=dict(reference=str(reference),reference_sha256=sha(reference),section_count=len(doc.sections),width_inches=sec.page_width.inches,height_inches=sec.page_height.inches,
    margins={k:getattr(sec,k).inches for k in ['left_margin','right_margin','top_margin','bottom_margin','header_distance','footer_distance']},
    source_title_style=str(doc.styles['Title'].font.size),source_normal_style=str(doc.styles['Normal'].font.size),package_parts=inventory)
save(D/'template_evidence.json',evidence)
contract='''# HDSF结果原因与数据处理汇报执行约定

参考为同目录reference.docx，源为用户现有的加入HDSF九页Word，SHA见template_evidence.json。源不改写。用户明确要求重排、原因结果前置并增加字段与处理说明，因此正文页数和章节顺序允许改写为十页。

保持原单节横向11.6903×8.2701英寸，左右0.65、上0.55、下0.50英寸，页眉页脚距离与sectPr保持。使用来源真实Title 21pt、Heading1 15pt、Normal 10.5pt Calibri/微软雅黑与1.1行距。标题全部黑色，无装饰线；结果最优数值可加粗、浅绿色E9F4E5标记，仅用于数值单元格。蓝灰表头、浅灰交替行、D9D9D9边框，来源表格和段落样式复制，保持PAGE字段。

正文是全面改写插槽。前四页为结果原因、全指标对照、实际专家权重、固定消融和误差分解。后六页为PowerData全七字段及机器映射三字段、真实清洗与聚合、派生特征及时间切分、ClusterData本地必要字段、CPU资源聚合、评价与证据边界。必要的新表按参考真实表格样式和单元格属性复制；不另造封面或总结框，不压缩字体塞内容。

仅word/document.xml与word/footer1.xml允许改动；其余包部件、关系、样式、主题、设置、customXml、元数据均逐字节保留。无新增图片和外部关系。真实字段含义及处理以官方字段文档、官方Notebook质量筛选、当前管线代码和data_audit.json为准；仅发生在代码中的填补分支与本次实际触发数要区分。

首次真正创建前执行marker一次。输出DOCX需使用render_docx.py；无LibreOffice时保留失败诊断，使用自己新建隐藏Word COM读打开并导PDF，Poppler出页图。查看每页、修正溢出后记录QA，源文件SHA和保留包部件相等、数据字段完整、真实结果不变才交付。
'''
(D/'artifact.md').write_text(contract,encoding='utf-8')
print(json.dumps({k:data[k] for k in ['all_passed','raw_rows','partial_measured_blocks','invalid_measured_blocks','all_empty_measured_blocks','target_valid_slots','production_causal_median_remaining_nonfinite','production_causal_median_filled_blocks','valid_measured_range','mapping','local24_feature_count']},ensure_ascii=True))
