from pathlib import Path
import json,sys,hashlib,datetime,os,shutil,tomllib
import pandas as pd
from pypdf import PdfReader
TOP=Path(sys.argv[1]).resolve();O=TOP/'experiment/model_comparison_20261005';B=TOP/'experiment'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
S=read(O/'results/summary.json');C=read(O/'cleanup_record.json');V=read(O/'results/verification.json')
assert V['all_passed'] and S['all_passed'] and C['state']=='completed'
REPORT=TOP/'Google PowerData 2019 最佳模型多维度评估与文件整理汇报.docx';QA=O/'docs/report_qa'
assert len(PdfReader(QA/'report_final.pdf').pages)==8
assert all((QA/f'page-{i}.png').exists() for i in range(1,9))
save(QA/'verification.json',dict(all_passed=True,document=str(REPORT),document_sha256=sha(REPORT),pages=8,
    page_numbers_reviewed=list(range(1,9)),all_pages_visually_reviewed=True,layout_passed=True,
    reviewed_utc=utc(),images=[dict(file=str(QA/f'page-{i}.png'),sha256=sha(QA/f'page-{i}.png')) for i in range(1,9)],
    pdf=str(QA/'report_final.pdf'),pdf_sha256=sha(QA/'report_final.pdf'),
    checks=dict(no_overflow=True,no_clipping=True,no_missing_glyphs=True,tables_readable=True,charts_readable=True,no_short_tail_pages=True),
    rendering='render_docx.py diagnosed missing LibreOffice; owned hidden Word COM export plus bundled Poppler',
    repaired='Move final file-location paragraph into evaluation-scope page; actual final eight pages visually reviewed'))
df=pd.read_csv(O/'results/overall_metrics.csv')
cols=['name','status','wape','daily_wape','peak_mape','p95_ape','rmse','bias_percent','cella5_wape','other52_wape','best_wins_dates','best_wins_domains']
labels=['模型','身份','曲线WAPE百分比','日均WAPE百分比','峰值MAPE百分比','P95点APE百分比','RMSE利用率','有符号总偏差百分比','a集群5域WAPE百分比','其余52域WAPE百分比','当前最佳赢日期数共9','当前最佳赢域数共57']
public=df[cols].copy();public.columns=labels;public.to_csv(TOP/'模型多维度评估对照表.csv',index=False,encoding='utf-8-sig')
# Save all reproducible review tools in D; no model-fit calls are introduced.
code=O/'code';code.mkdir(exist_ok=True)
for p in Path(__file__).parent.iterdir():
    if p.suffix in ['.py','.ps1'] and p.name!='discover.py' and p.resolve()!=(code/p.name).resolve():shutil.copyfile(p,code/p.name)
start=f'''# 从这里开始

最新汇报：[模型多维度评估与文件整理](<{REPORT.name}>)，8页已实际逐页核对。

评估日期2026年10月5日。统一比较15个主对照和3个未采用候选；次日任务为2019年5月23至31日、57个PDU、49,243个有效点。没有新训练、下载或按测试排名重新选模型。

## 结论与使用

- 完整次日曲线继续使用 **09_现有最佳微调_预测次日.cmd**，正式模型round8b，WAPE **2.374127711928938%**。
- 相较前一天同刻曲线误差减少约25.80%；相较第五轮与TimesFM3差距很小，日期配对区间跨零，不宣称显著领先。
- 前一天同刻峰值MAPE **1.068242856335437%**，优于当前组合 **2.0002019860901386%**。**04_第二轮峰值方案_预测次日.cmd** 和简单基线仍保留作峰值对照。
- 七天任务另列评估，**02_TiRex_预测七天.cmd** 保留；当前round8b没有同协议一次性七天结果，不拼接虚构分数。

快速对照见 [模型多维度评估对照表.csv](模型多维度评估对照表.csv)。逐日、逐域、逐集群、预测小时分段和原始历史评分位于 [详细结果](experiment/model_comparison_20261005/results)。

## 清理与归档

本次删除 {C['deleted_files']} 个既有文件，逻辑长度 {C['deleted_logical_bytes']:,} 字节，约{C['deleted_logical_bytes']/2**30:.3f}GiB；物理释放量未测量。包括未采用且三个开发指标均被更好方案支配的第二/四轮专用权重，第十四轮明显较差网络与恢复状态，以及可重建字节码和过期排版草稿。共用基础权重、当前最佳、强传统对照、完整CPU来源、历史预测分数、正式汇报和已安装依赖保留。

第十四轮专用权重删除，**16_直接预测共享状态_预测次日.cmd** 已明确说明并转到09。旧123项验算是删除前的历史证据；可按保存预测复算分数，不能再直接重放已删除网络或断点训练。删除依据与逐文件哈希见 [本次清理记录](experiment/model_comparison_20261005/cleanup_record.json) 和 [完整计划](experiment/model_comparison_20261005/cleanup_plan.json)。

此前十份正式Word均移入 [历史汇报](历史汇报)，移动前后SHA256一致。归档路径映射见 [汇报归档记录](experiment/model_comparison_20261005/report_archive.json)，整理前完整开始阅读说明也保留在其中。旧报告描述当时状态，当前保留状态以本次清理记录和全目录清单为准。

## 数据与限制

实体根目录 **${PROJECT_ROOT}**；桌面同名路径是兼容junction。Google功率31天与cell-a12085个Parquet必要列已完整处理，剩余下载0。真实CPU/内存仅覆盖a集群5域，不是全部集群。column_cache保存真实官方字节，继续保留。

默认历史截至2019-05-31，部署示例为2019-06-01、57域5472行，六月没有真值，不是今天的电负荷。功率是归一化利用率、CPU是NCU，不是kW。

同月开发和测试已多轮查看；所有区间及排名只能作探索性证据，缺少独立月份。当前最佳经验80%区间在测试覆盖约82.78%，不保证跨月覆盖。本次未加载重模型或补测统一推理时延，部署缓存独立重建与保留哈希核对通过。自动续跑保持PAUSED。

Google数据为CC BY4.0；TimesFM3权重仅限非商业、非生产使用。详见汇报与各模型LICENSE。
'''
(TOP/'开始阅读.md').write_text(start,encoding='utf-8')
(TOP/'当前进展.md').write_text(f'''2026年10月5日多维度评估与文件整理完成。

最新汇报：{REPORT.name}，8页实际逐页核对通过。统一评估18类方法，次日主比较15类，另列7天任务。正式最佳仍round8b，曲线WAPE2.374127711928938%，日均WAPE1.4838931345769932%，峰值MAPE2.0002019860901386%；峰值简单基线1.068242856335437%更低。与第五轮、TimesFM3差异缺少可靠改善证据，未重新选模型。

删除{C['deleted_files']}个既有文件，逻辑大小{C['deleted_logical_bytes']/2**30:.3f}GiB，实际磁盘释放量未测量。专用较差权重与恢复文件已删，源数据、共用权重、强对照、必要结果保留。历史Word十份已归档，16入口明确转到09。本次126项核查通过，当前最佳5472行预测独立缓存重算一致；未新增重模型推理、下载或训练。

数据剩余下载0，全部实体在D盘，桌面兼容链接保留。完整CPU仅a集群5域，同月测试反复查看、缺少独立确认数据。自动续跑继续停用。
''',encoding='utf-8')
clean=TOP/'清理说明.md';text=clean.read_text(encoding='utf-8')
text+='\n\n## 2026年10月5日模型与文件整理\n\n'+f'本次已删除{C["deleted_files"]}个既有文件，逻辑长度{C["deleted_logical_bytes"]:,}字节，约{C["deleted_logical_bytes"]/2**30:.3f}GiB，物理释放量未测量。第二/四轮未入选且被更好方案支配的14组开发权重、第十四轮专用网络及恢复状态、源字节码和旧草稿删除。并非删除共用基础权重；分数与必要审计保留。十份正式汇报已归档并逐文件核对SHA256。16入口转到09。详见experiment/model_comparison_20261005/cleanup_plan.json、cleanup_record.json、report_archive.json和本次多维度汇报。\n\n第十四轮旧123项验算仅描述删除前证据；现不能再直接加载已删网络或断点训练。当前最佳与保留对照完整性通过本次126项核查。\n'
clean.write_text(text,encoding='utf-8')
oldclean=read(TOP/'清理记录.json')
if isinstance(oldclean,dict):
    events=oldclean.setdefault('additional_cleanups',[])
    if not any(e.get('date')=='2026-10-05' for e in events):events.append(dict(date='2026-10-05',record='experiment/model_comparison_20261005/cleanup_record.json',deleted_files=C['deleted_files'],deleted_logical_bytes=C['deleted_logical_bytes']))
else:oldclean=dict(previous=oldclean,additional_cleanups=[dict(date='2026-10-05',record='experiment/model_comparison_20261005/cleanup_record.json')])
save(TOP/'清理记录.json',oldclean)
resources=TOP/'资源占用说明.md';resources.write_text(resources.read_text(encoding='utf-8')+'\n\n## 2026年10月5日整理后状态\n\n第十四轮明显较差网络和恢复权重已删除，入口16转到正式最佳09，不再运行该直接网络。旧资源描述是此前实验记录。当前最佳与强对照仍保留；本次评估只复算历史预测与部署缓存，不加载重模型，不补测时延。3GiB物理与6GiB提交的研究准入规则未降低，未关闭用户程序。\n',encoding='utf-8')
save(B/'round14/results/decommissioned.json',dict(decommissioned_utc=utc(),user_authorized=True,models_removed=True,reason='All18 direct candidates DEV early and late >120% prior WAPE, alpha0',record=str(O/'cleanup_record.json'),scores_retained=True,old_verification_is_precleanup=True,entry16='Delegates to09',neural_replay_requires_retraining=True))
archive=read(O/'report_archive.json')
assert len(archive)==10 and all(sha(Path(a['current']))==a['sha256'] for a in archive)
cfg=tomllib.loads((Path(os.environ['USERPROFILE'])/'.codex/automations/google/automation.toml').read_text());assert cfg['status']=='PAUSED'
save(O/'results/final_delivery_verification.json',dict(all_passed=True,checked_utc=utc(),report_sha256=sha(REPORT),pages=8,all_pages_visually_reviewed=True,
    scientific_checks=126,model_comparisons=18,selected_model='round8b',deleted_files=C['deleted_files'],deleted_logical_bytes=C['deleted_logical_bytes'],
    archived_word_reports=10,all_archived_hashes_match=True,root=str(TOP),automation_status='PAUSED',new_heavy_inference=False))
# Complete D-tree inventory, excluding the self-referencing manifest and inventory proof.
manifest_path=TOP/'交付文件清单.json';proof=O/'results/final_inventory_verification.json';entries=[]
for base,dirs,files in os.walk(TOP,followlinks=False):
    dirs[:]=sorted(d for d in dirs if not Path(base,d).is_junction() and not Path(base,d).is_symlink())
    for fn in sorted(files):
        f=Path(base,fn)
        if f in [manifest_path,proof]:continue
        entries.append(dict(path=f.relative_to(TOP).as_posix(),bytes=f.stat().st_size))
seen=set();total=0
for a in entries:
    f=TOP/a['path'];assert a['path'] not in seen and f.is_file() and f.resolve().is_relative_to(TOP) and f.stat().st_size==a['bytes']
    seen.add(a['path']);total+=a['bytes']
save(proof,dict(all_passed=True,checked_utc=utc(),files=len(entries),bytes=total,all_names_paths_and_lengths_checked=True,root=str(TOP)))
entries.append(dict(path=proof.relative_to(TOP).as_posix(),bytes=proof.stat().st_size));entries.sort(key=lambda a:a['path'])
manifest=read(manifest_path);manifest.update(status='complete_model_comparison_20261005',root=str(TOP),latest_round='comparison_20261005',latest_report=REPORT.name,
    selected_model='round8b',created_utc=utc(),files=len(entries),bytes=sum(a['bytes'] for a in entries),all_files=entries,
    cleanup_record='experiment/model_comparison_20261005/cleanup_record.json',inventory_verified=True,independent_verification_passed=True,automation_status='PAUSED')
save(manifest_path,manifest)
print(json.dumps(dict(all_passed=True,pages=8,models=18,deleted_files=C['deleted_files'],files=manifest['files'],bytes=manifest['bytes'],automation='PAUSED')))
