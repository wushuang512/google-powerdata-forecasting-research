from pathlib import Path
import json,hashlib,sys,datetime
T=Path(sys.argv[1]);R=T/'experiment/round48';D=R/'deployment';D.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
source=R/'results/selection.json';s=read(source);z=read(R/'results/summary.json');assert sha(source)==z['selection_sha256']==(R/'results/selection.sha256').read_text().strip()
assert read(R/'results/verification.json')['all_passed'];assert not (D/'selected_configuration.json').exists()
prev=T/'experiment/round47/deployment/selected_configuration.json';backup=R/'docs/before_R48_user_adoption';backup.mkdir(exist_ok=True)
for name in ['当前推荐方案.json','开始阅读.md','当前进展.md','19_1天当前推荐模型历史复现.cmd','19_7天当前推荐模型历史复现.cmd','33_1天当前推荐HDSF校正历史复现.cmd','33_7天当前推荐HDSF校正历史复现.cmd']:
    if (T/name).exists():(backup/name).write_bytes((T/name).read_bytes())
bindings={str(p.relative_to(T)):sha(p) for p in [source,R/'docs/protocol.json',R/'results/summary.json',R/'results/predictions.npz',R/'code/robust_hdsf_experiment.py',R/'code/predict_recommended_r48.py',prev]}
names={'1':'HDSF半日单纯形加PatchTST FM日均残差校正及同组50%收缩','7':'HDSF半日单纯形加TimesFM3 Toto2 PatchTST FM零均值形状全量校正'}
tasks={h:dict(model_name=names[h],variant=s['tasks'][h]['candidate']['variant'],metrics=z['scores'][h]['DEV_frozen_R48_candidate']['metrics'],interval=z['scores'][h]['DEV_frozen_R48_candidate']['interval'],DEV_supported_originally=s['tasks'][h]['DEV_supported']) for h in ['1','7']}
now=datetime.datetime.now(datetime.timezone.utc).isoformat();cfg=dict(scheme_id='R48_ROBUST_HDSF_USER_ADOPTED',created_utc=now,decision_type='explicit_user_adoption_after_results',direct_user_instruction='这个候选不是在几个主要评价指标上超过了推荐吗，就把它作为新推荐吧',previous_configuration=str(prev),previous_configuration_sha256=sha(prev),bindings=bindings,tasks=tasks,original_research_selection_modified=False,hyperparameters_changed_after_TEST=False,research_gate_originally_passed=False,independent_confirmation=False,limits='用户接受已公布固定候选的取舍；后段开发不稳定，同月TEST多次暴露；不是大幅或独立显著提升。历史复现接口尚非外部数据通用预测器。')
save(D/'selected_configuration.json',cfg);save(R/'results/user_adoption.json',dict(all_passed=True,created_utc=now,user_instruction=cfg['direct_user_instruction'],configuration=str(D/'selected_configuration.json'),configuration_sha256=sha(D/'selected_configuration.json'),original_frozen_selection_sha256=sha(source),previous_R47_configuration_unchanged_sha256=sha(prev)))
save(T/'当前推荐方案.json',dict(scheme_id=cfg['scheme_id'],configuration=str(D/'selected_configuration.json'),configuration_sha256=sha(D/'selected_configuration.json'),tasks=tasks,user_adopted=True,research_gate_passed=False,prediction_entry_1day='34_1天当前推荐HDSF稳健校正历史复现.cmd',prediction_entry_7day='34_7天当前推荐HDSF稳健校正历史复现.cmd'))
for h in [1,7]:
    cmd='@echo off\r\nchcp 65001 >nul\r\n"%~dp0experiment\\round44\\.venv\\Scripts\\python.exe" -X utf8 "%~dp0experiment\\round48\\code\\predict_recommended_r48.py" "%~dp0." '+str(h)+'\r\npause\r\n'
    (T/f'34_{h}天当前推荐HDSF稳健校正历史复现.cmd').write_bytes(cmd.encode('utf-8'))
    (T/f'19_{h}天当前推荐模型历史复现.cmd').write_bytes(cmd.encode('utf-8'))
    (T/f'33_{h}天当前推荐HDSF校正历史复现.cmd').write_bytes(cmd.encode('utf-8'))
(D/'采用说明.md').write_text('''# 当前推荐采用说明

用户明确要求采用第48轮已经公布的固定候选。24小时使用PatchTST FM日均差分及同场站同类型50%相对校正收缩；7天使用TimesFM3 Toto2 PatchTST FM零均值形状权重1。原研究selection、协议及R47配置不变。

24小时曲线WAPE2.329603065559927%，日均1.4358730219729698%，峰值MAPE1.9656829707881618%；7天曲线2.7011842083735202%，日均1.86882999522989%，峰值1.9440720139726937%。部分尾部或命中指标略退步。后段开发门槛未通过，用户采用不改写为验证成功，不声称大幅或独立显著提升。

34和当前19/33入口复现原Google2019历史缓存，尚不是任意外部数据的通用预测接口。此前58页Word反映此次采用决定之前的状态，保留为历史快照；新的评价适用性汇报在当前聊天。
''',encoding='utf-8')
print(json.dumps(dict(all_passed=True,scheme_id=cfg['scheme_id'],frozen_selection_unchanged=sha(source)==z['selection_sha256'])))
