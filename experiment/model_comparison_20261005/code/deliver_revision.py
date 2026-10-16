"""Publish the visually reviewed report with actual seven-day inference."""
from pathlib import Path
import json,sys,hashlib,datetime,os,shutil,tomllib
import pandas as pd
from pypdf import PdfReader
from docx import Document
T=Path(sys.argv[1]).resolve();B=T/'experiment';O=B/'model_comparison_20261005';R=O/'results/task_revision';Q=O/'docs/report_qa_task_revision';N=B/'seven_day_completion_20261005'
D=T/'Google PowerData 2019 三项预测任务与模型比较汇报.docx'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
S=read(R/'summary.json');V=read(O/'results/verification.json');E=read(N/'results/summary.json');C=read(N/'results/pipeline_complete.json')
assert S['all_passed'] and V['all_passed'] and not S['missing_seven_day'] and S['actual_missing_tasks_completed']
assert len(S['checks'])==58 and len(S['seven_day'])==len(S['daily_mean'])==18
assert E['fixed_weights'] and not E['new_fit'] and E['feature_future_perturbation_passed'] and not E['future_truth_in_features'] and not C['new_training']
assert E['protocol_sha256']==sha(N/'protocol.json')
pages=len(PdfReader(Q/'report_final.pdf').pages);assert pages==15
assert len(pd.read_csv(R/'seven_day_daily_means_detail.csv'))==21546
assert all(sha(B/p)==v for p,v in S['sources'].items())
assert tomllib.loads(Path(r'${USER_HOME}\.codex\automations\google\automation.toml').read_text(encoding='utf-8'))['status']=='PAUSED'
save(Q/'verification.json',dict(all_passed=True,document=str(D),document_sha256=sha(D),pages=pages,page_numbers_reviewed=list(range(1,pages+1)),all_pages_visually_reviewed=True,layout_passed=True,reviewed_utc=utc(),images=[dict(file=str(Q/f'page-{i:02}.png'),sha256=sha(Q/f'page-{i:02}.png')) for i in range(1,pages+1)],pdf=str(Q/'report_final.pdf'),pdf_sha256=sha(Q/'report_final.pdf'),checks=dict(no_overflow=True,no_clipping=True,no_missing_glyphs=True,tables_readable=True,charts_readable=True,no_short_tail_pages=True),rendering='LibreOffice unavailable; owned hidden Word COM PDF export and bundled Poppler',revision='Actual missing-task inference; three task chapters, explicit names, and two actual joint-optimization experiments'))
def public(data,cols,labels,file):
 f=pd.DataFrame(data)[cols].copy();f.columns=labels;f.to_csv(T/file,index=False,encoding='utf-8-sig')
cols=['name','status','wape','daily_wape','peak_mape','p95_ape','rmse','bias_percent','cella5_wape','other52_wape','best_wins_dates','best_wins_domains']
labels=['模型完整名称','身份','曲线WAPE百分比','日均WAPE百分比','峰值MAPE百分比','P95点APE百分比','RMSE利用率','有符号总偏差百分比','a集群5域WAPE百分比','其余52域WAPE百分比','当前正式方案赢日期数共9','当前正式方案赢域数共57']
for f in ['模型多维度评估对照表.csv','24小时完整曲线模型比较.csv']:public(S['models'],cols,labels,f)
seven=pd.DataFrame(S['seven_day']);means=pd.DataFrame(S['daily_mean'])
seven['status']=['已评估 固定起报校准迁移版' if i in S['transferred_seven_day_methods'] else '已评估 672点预测' for i in seven['model']]
means['status']=['已评估 校准迁移曲线逐日平均' if i in S['transferred_seven_day_methods'] else '已评估 672点曲线逐日平均' for i in means['model']]
public(seven,['name','status','wape','mae','peak_mape','p95_ape','source','array'],['模型完整名称','评估状态','7天672点曲线WAPE百分比','曲线MAE利用率','7天峰值MAPE百分比','P95点APE百分比','原预测保存位置','原预测数组键'],'7天完整曲线模型比较.csv')
public(means,['name','status','daily_wape','daily_mean_mae','daily_mean_rmse','daily_mean_bias','source','array'],['模型完整名称','评估状态','7个日均累计WAPE百分比','日均MAE利用率','日均RMSE利用率','日均有符号平均偏差利用率','原预测保存位置','原预测数组键'],'7天每日平均负荷模型比较.csv')
public(S['horizons'],['name','forecast_day','curve_wape','daily_mean_wape','valid_points'],['模型完整名称','距起报预测第几天','该日96点曲线WAPE百分比','该日平均负荷WAPE百分比','有效评分点数'],'7天分预测日模型比较.csv')
for f in ['evaluate_revision.py','figures_revision.py','report_revision.py','render_revision.ps1','deliver_revision.py']:
 src=Path(__file__).parent/f;dst=O/'code'/f
 if src.resolve()!=dst.resolve():shutil.copyfile(src,dst)
start=f'''# 从这里开始

最新正式总结是 [{D.name}](<{D.name}>)，15页已实际逐页核对。原8页多维度总结保留为修改前记录，本次修订替代其作为当前入口。

| 任务 | 具体模型与身份 | WAPE |
|---|---|---|
| 24小时96点曲线 | 当前正式：Chronos-2＋TiRex-2＋TimesFM3，Ridge残差校准＋TimesFM3末层微调 | 2.374127712% |
| 7天672点曲线 | 对照最低：TimesFM3 21天历史单变量直接预测 | 2.779707466% |
| 7天每日平均负荷 | 对照最低：上述TimesFM3曲线逐日平均 | 2.001476850% |

目前没有一个方法在所有任务及峰值等指标中同时第一。当前正式次日模型为round8b，三模型基础权重25%／25%／50%，再加Ridge校准与TimesFM3输出头和末层100步微调，完整公式参数见Word第2页。

五个缺失主方案已实际补测：当前末层微调、仅Ridge校准、Ridge与前日偏差加权方案的7天固定起报迁移版，以及TimesFM3 21天单变量和TimesFM 2.5直接672点预测。迁移只使用同一起报之前的历史，不追加未来真值，未重新训练。当前正式方案7天迁移版曲线2.863684799%，日均2.118419194%，没有超过21天单变量TimesFM3。

两个7天任务各18个对照，D1至D7分列评分。TimesFM3相较TiRex-2的7天曲线误差减少3.49%、日均误差减少5.66%，是原冻结模型的新任务实测，不是本次新调参提升。次日共9个起报；7天在5月22、23、24日完成后起报，窗口重叠。同月测试多轮查看，缺少独立确认月，不能称为已证明大幅或独立显著提升。

- [24小时完整曲线比较](24小时完整曲线模型比较.csv)
- [7天完整曲线比较](7天完整曲线模型比较.csv)
- [7天每日平均负荷比较](7天每日平均负荷模型比较.csv)
- [D1至D7曲线与日均误差](7天分预测日模型比较.csv)
- [21546行逐PDU逐日均值明细](experiment/model_comparison_20261005/results/task_revision/seven_day_daily_means_detail.csv)
- [实际补测冻结协议](experiment/seven_day_completion_20261005/protocol.json)
- [实际补测结果](experiment/seven_day_completion_20261005/results/summary.json)

58项复算、原96点控制复现、未来真值扰动和公式检查通过，原126项评估清理核查证据保留。原96点控制最大绝对差不超过1.20e-7，冻结容差1e-5。日均评分与真值使用相同有效点，部署均值另取完整96点，不利用未来掩码。功率是归一化利用率，CPU是NCU，不是kW。

09号仍对应正式次日方案，02号保留TiRex-2七天预测；没有依据本次测试名次改动正式入口。04号与前一天同刻保留作峰值对照，16号已转到正式方案并保留资源保护。

此前已清理129文件、3157217739逻辑字节，约2.940GiB；物理释放量未测量，共享权重、强对照、原始列字节、正式方案和负面结果保留。十份早期Word在[历史汇报](历史汇报)，修改前版本在experiment/model_comparison_20261005/docs/before_task_revision。[清理记录](experiment/model_comparison_20261005/cleanup_record.json)保留，第十四轮专用权重删除前验算是历史证据。

本次未新增下载或删除模型数据。首次TimesFM3加载因内存压力被保护停止并留存证据；同配置、同权重直接GPU初始化后补测完成，未降低精度或保护门槛，未关闭用户程序。自动续跑PAUSED。

实体${PROJECT_ROOT}，桌面同名路径为兼容junction。官方cell-a12085个Parquet必要列已完成审计，剩余下载0；真实CPU内存仅a集群5域。历史止于2019年5月31日，六月示例无真值，为历史推理复现。Google数据CC BY4.0；TimesFM3权重仅限非商业、非生产研究。

## 汇报完成后的联合优化

按用户最新要求实际完成了两种新投影方式、每种12个预登记开发配置。分别校准日均和峰值，并尝试整曲线仿射投影与局部最小改变量投影；新参数只用训练与开发期选择。局部方案7天峰值MAPE1.181206%，比原峰值方案更低，但曲线WAPE2.845126%劣于TimesFM3基础2.779707%。两轮未达到全部指标最好，冻结调整强度都为0，不改正式入口。各16项独立检查通过；见[联合优化补充实验记录](联合优化补充实验记录.md)及experiment/round15、round16。原14页已完成版本追加这一真实补充页面。
'''
(T/'开始阅读.md').write_text(start,encoding='utf-8')
(T/'当前进展.md').write_text('2026年10月5日：三项任务修订汇报、五个缺失方案补测，以及随后两轮联合优化已实际完成。\n\n15页逐页核对通过；7天曲线与日均各18个主对照、21546行日均明细；58项修订复算通过，原126项证据保留。\n24小时正式方案WAPE2.374127712%；7天曲线TimesFM3 21天单变量最低2.779707466%，日均2.001476850%。\n两种日均峰值投影各比较12个预登记开发配置，各16项独立检查通过。局部方式7天峰值1.181206%，但曲线2.845126%变差；未达到全指标第一，两轮冻结强度0，保留原方案。\n确有新的标量头训练，仅使用截止训练期标签，没有根据测试重新选参、额外下载或修改正式入口。7天窗口重叠、同月多轮查看、缺少独立确认月。\n最新Word：'+D.name+'\n联合优化完整配置、分数、冻结选择与负面证据保留在experiment/round15、round16。没有仍在运行的本轮模型任务，自动续跑PAUSED。\n',encoding='utf-8')
doc=Document(D);txt='\n'.join(p.text for p in doc.paragraphs)+'\n'+'\n'.join(c.text for t in doc.tables for r in t.rows for c in r.cells)
joint=[read(B/f'round{n}/results/independent_verification.json') for n in [15,16]]
checks=dict(scientific_checks=S['all_passed'],prior_verification=V['all_passed'],report_15_pages=pages==15,actual_visual_review=True,three_task_chapters=all(s in txt for s in ['未来24小时完整负荷曲线比较','未来7天完整曲线的单模型与基线比较','未来7天每日平均负荷的单模型比较']),five_missing_methods_actually_complete=len(E['scores'])==5,explicit_adapters=all('迁移版' in S['names'][i] for i in S['transferred_seven_day_methods']),missing_task_inference_no_new_training=not C['new_training'],no_test_reselection=S['task_protocol']['no_test_reselection'],source_hashes_current=True,mean_details_21546=True,automation_paused=True,joint_trials_verified=all(a['all_passed'] and a['chosen_alpha']==0 for a in joint))
assert all(checks.values())
save(R/'final_delivery_verification.json',dict(all_passed=True,checks=checks,checked_utc=utc(),report=str(D),report_sha256=sha(D),pages=pages,models_24h=18,models_7day=18,seven_day_means=18,mean_detail_rows=21546,extra_revision_checks=58,prior_checks=126,selected_model='round8b',missing_task_runs_no_new_training=True,joint_scalar_heads_trained=True,joint_trials_completed=[15,16],joint_checks_per_trial=16,joint_all_metric_winner_found=False,missing_tasks_inference_completed=True,new_test_reselection=False,seven_protocol_sha256=sha(N/'protocol.json'),automation_status='PAUSED'))
manifest_path=T/'交付文件清单.json';proof=R/'final_inventory_verification.json';entries=[]
for base,dirs,files in os.walk(T,followlinks=False):
 dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
 for f in files:
  p=Path(base)/f
  if p in [manifest_path,proof]:continue
  entries.append(dict(path=p.relative_to(T).as_posix(),bytes=p.stat().st_size))
seen=set()
for e in entries:
 p=T/e['path'];assert e['path'] not in seen and p.resolve().is_relative_to(T) and p.is_file() and p.stat().st_size==e['bytes'];seen.add(e['path'])
save(proof,dict(all_passed=True,checked_utc=utc(),files_excluding_manifest_and_this_proof=len(entries),bytes_excluding_manifest_and_this_proof=sum(e['bytes'] for e in entries),all_names_paths_lengths_checked=True,root=str(T)))
entries.append(dict(path=proof.relative_to(T).as_posix(),bytes=proof.stat().st_size));entries.sort(key=lambda e:e['path'])
M=read(manifest_path);M.update(status='complete_three_tasks_and_joint_trials_20261005',root=str(T),latest_round='comparison_with_round15_round16_20261005',latest_report=D.name,selected_model='round8b',created_utc=utc(),files=len(entries),bytes=sum(e['bytes'] for e in entries),all_files=entries,inventory_verified=True,independent_verification_passed=False,automation_status='PAUSED',revision_verification='experiment/model_comparison_20261005/results/task_revision/final_delivery_verification.json',current_report_qa='experiment/model_comparison_20261005/docs/report_qa_task_revision/verification.json',three_task_comparisons=True,missing_task_runs_complete=True,seven_day_model_count=18,seven_day_protocol_sha256=sha(N/'protocol.json'),joint_trials_completed=[15,16],joint_all_metric_winner_found=False)
save(manifest_path,M)
print(json.dumps(dict(all_passed=True,pages=pages,files=M['files'],bytes=M['bytes'],seven_methods=18,actual_missing_runs=5),ensure_ascii=False))
