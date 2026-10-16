"""Record explicit user adoption separately from immutable research selection."""
from pathlib import Path
import datetime,hashlib,json,sys
sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
t=Path(sys.argv[1]);b=t/'experiment';r=b/'round47';q=r/'deployment';q.mkdir(exist_ok=True)
frozen=r/'results/selection.json';s=read(frozen);report=read(r/'results/summary.json')
assert sha(frozen)==(r/'results/selection.sha256').read_text().strip()==report['selection_sha256']
assert read(r/'results/verification.json')['all_passed']
configpath=q/'selected_configuration.json'
assert not configpath.exists(),'Adoption already recorded; do not silently overwrite'
names={'1':'HDSF半日单纯形加PatchTST FM日均残差校正',
       '7':'HDSF半日单纯形加TimesFM3 Toto2 PatchTST FM零均值形状校正'}
bindings={str(p.relative_to(t)):sha(p) for p in [frozen,r/'results/predictions.npz',r/'results/summary.json',r/'code/hdsf_shape_experiment.py',r/'code/predict_user_hdsf.py']}
models=[r/'models/patchtst_curve_cut22.pt',b/'round46/models/timesfm3_tail_s40_cut22.pt',
        b/'round46/models/toto2_tail_s40_cut22.pt',b/'round46/models/patchtst_tail_s40_cut22.pt']
for p in models:bindings[str(p.relative_to(t))]=sha(p)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
config=dict(scheme_id='R47_HDSF_LEVEL_SHAPE_USER_ADOPTED',created_utc=now,
    decision_type='explicit_user_adoption_after_results',
    direct_user_instruction='这个不是多项改善了吗，然后只有微退步，就用这个',
    adoption_reason='用户接受主要指标改善与少数指标微小退步的取舍，明确采用已公布的固定候选。',
    frozen_research_selection_sha256=sha(frozen),research_validation_gate_originally_passed=False,
    original_research_selection_modified=False,hyperparameters_changed_after_TEST=False,
    adoption_itself_is_post_result_user_decision=True,independent_confirmation=False,
    scientific_limits='开发期跨日期收益不稳，探索性置信区间跨零，同月测试已多轮查看；不声称独立显著或大幅提高。',
    tasks={h:dict(model_name=names[h],**{k:s['tasks'][h]['candidate'][k] for k in ['route','level_alpha','shape_alpha']},
            metrics=report['scores'][h]['frozen_candidate']['metrics'],interval=report['scores'][h]['frozen_candidate']['interval']) for h in ['1','7']},
    weights=[str(p) for p in models],bindings=bindings,old_HDSF_retained=True,
    usage='已核对历史滚动预测复现，使用原GooglePowerData2019数据；不是今天实时电负荷预测。')
save(configpath,config);save(r/'results/user_adoption.json',dict(created_utc=now,all_passed=True,configuration=str(configpath),
    configuration_sha256=sha(configpath),user_request=config['direct_user_instruction'],research_selection_still_sha256=sha(frozen),
    previous_validation_flags_retained=True,independent_confirmation=False))
save(t/'当前推荐方案.json',dict(scheme_id=config['scheme_id'],configuration=str(configpath),
    configuration_sha256=sha(configpath),tasks=config['tasks'],user_adopted=True,research_gate_passed=False,
    prediction_entry_1day='33_1天当前推荐HDSF校正历史复现.cmd',prediction_entry_7day='33_7天当前推荐HDSF校正历史复现.cmd'))
backup=r/'docs/before_user_adoption';backup.mkdir(exist_ok=True)
for name in ['开始阅读.md','当前进展.md','19_1天当前推荐模型历史复现.cmd','19_7天当前推荐模型历史复现.cmd']:
    p=t/name
    if p.exists():(backup/name).write_bytes(p.read_bytes())
for h in [1,7]:
    entry=f'33_{h}天当前推荐HDSF校正历史复现.cmd'
    text='@echo off\r\nchcp 65001 >nul\r\n"%~dp0experiment\\round44\\.venv\\Scripts\\python.exe" -X utf8 "%~dp0experiment\\round47\\code\\predict_user_hdsf.py" "%~dp0." '+str(h)+'\r\npause\r\n'
    (t/entry).write_bytes(text.encode('utf-8'))
    (t/f'19_{h}天当前推荐模型历史复现.cmd').write_bytes(text.encode('utf-8'))
note='''# 当前推荐方案采用说明

2026年10月12日，用户明确接受多项主要指标改善与少数指标微小退步，采用第47轮已冻结候选。

24小时使用原HDSF半日单纯形预测，加PatchTST-FM新微调相对原权重的每日均值差分，水平系数1，形状系数0。
7天使用原HDSF预测，加TimesFM3、Toto-2、PatchTST-FM微调差分的日内零均值形状部分，水平系数0，形状系数0.5。原本地成员保留。

24小时曲线WAPE为2.329622%，日均WAPE为1.436128%，峰值MAPE为1.965495%。7天曲线WAPE为2.701341%，日均WAPE为1.868830%，峰值MAPE为1.946208%。24小时95分位APE由6.777904%变为6.795068%，属于已接受的微小退步。

这是用户在查看公布结果后作出的采用决定。原开发期升级门槛未通过的记录继续保留，原selection.json及SHA未修改，不把采用决定改写成开发期验证成功。当前方案仍缺少独立确认数据，不声称已证明大幅或显著提高。

运行根目录入口33可重新构建已核对的历史滚动预测CSV及每日平均负荷CSV，入口19也已指向同一当前方案。它们使用已核对的模型缓存复现原2019年实验，不预测今天实时负荷。实际基础模型权重更新和原生推理已在第46与47轮核查。

参数、权重位置和文件摘要见selected_configuration.json，原HDSF及全部负面实验保留。新的探索以当前采用方案为比较基准，评价沿用9.28统一口径。
'''
(q/'采用说明.md').write_text(note,encoding='utf-8')
(t/'开始阅读.md').write_text('''# 开始阅读

当前推荐已按用户明确指示采用第47轮HDSF负荷水平与形状残差校正方案。主要指标改善，少数指标微小退步已接受；完整采用说明位于[采用说明](experiment/round47/deployment/采用说明.md)。

使用33_1天当前推荐HDSF校正历史复现.cmd或33_7天当前推荐HDSF校正历史复现.cmd。入口19也已同步。原数据和模型都保存在D盘实体目录，桌面旧目录仍为兼容入口。

三项任务分别是未来24小时完整曲线、未来7天完整曲线和未来7天每日平均负荷。当前曲线WAPE分别为2.329622%、2.701341%，7天日均WAPE为1.868830%。统一评价逐PDU计算后57域等权平均；日均使用完整96个有效点。

最新逐项比较见[第47轮全指标](experiment/round47/results/清晰模型名称_全指标比较.csv)。第48轮继续研究稳健校正和跨PDU信息共享，完成后生成完整Word汇报。

此前Word均为保留的历史报告，尚未反映本次用户采用决定和第48轮实验。原HDSF、Moirai等历史入口和研究证据继续保留。开发期波动、已多轮查看的同月测试和缺少独立确认数据等限制仍成立。
''',encoding='utf-8')
(t/'当前进展.md').write_text('''# 当前进展

2026年10月12日，第47轮固定候选已由用户明确采用为当前推荐。配置见当前推荐方案.json，详细依据见experiment/round47/deployment/采用说明.md。原研究冻结记录未改写。

第47轮五次实际训练与800次参数更新已完成，统一指标、未来输入扰动和权重核对通过。第48轮正在准备以当前采用方案为基准的稳健残差校正与跨PDU信息共享实验，所有新产出继续放D盘。

本次结束后生成详细Word，结论与模型比较前置，列全三项任务、9.28指标及补充和区间指标，完成逐页核对。旧Word暂保留为历史文件。现有官方必要列读取完成，剩余下载为零，不使用其他数据集。
''',encoding='utf-8')
print(json.dumps(dict(all_passed=True,user_adoption_recorded=True,configuration=str(configpath),selection_unchanged_sha256=sha(frozen)),ensure_ascii=True))
