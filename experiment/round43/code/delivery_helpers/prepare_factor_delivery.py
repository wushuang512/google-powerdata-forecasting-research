from pathlib import Path
import sys,json
T=Path(sys.argv[1]);B=T/'experiment';R=B/'round43';a=json.loads((R/'results/factor_comparison.json').read_text(encoding='utf-8'));assert not any(s['adopt_shared_scheme'] for s in a['factor_rounds'].values())
s=(B/'round40/code/delivery_continued_moirai.py').read_text(encoding='utf-8').replace("R=B/'round40'","R=B/'round43'").replace("Q/'continued_comparison.json'","Q/'factor_comparison.json'")
s=s.replace("entry=[];titles={37:'多分辨率Moirai2',38:'季节残差Moirai2',39:'相关域Moirai2',40:'CPU内存Moirai2'}", "entry=[];titles={41:'共同因子Moirai2',42:'注意力训练Moirai2',43:'同条件水平Moirai2控制'}").replace('for nr in range(37,41):','for nr in [41,42,43]:').replace('round40\\\\code\\\\read_continued_research.py','round43\\\\code\\\\read_factor_research.py')
start=s.index("(T/'开始阅读.md').write_text(");end=s.index("(Q/'delivery_note.json')",start)
block="""(T/'开始阅读.md').write_text('''# 从这里开始

最新九页Word已实际逐页核对，先列当前推荐与全部实测最低，再给统一口径、三个任务模型结果、三轮新训练证据及区间。126条记录、98组不同预测，未按TEST重新选模型。

- [最新Word汇报](<Google PowerData 2019 三项预测任务与模型比较汇报.docx>)
- [126条完整指标](模型各项指标完整结果.csv)
- [各任务单项最优](各任务单项最优结果.csv)
- [区间结果](模型预测区间结果.csv)
- [口径审查](9月28日评价口径审查.md)
- [清理记录](清理说明.md)

|任务|当前推荐WAPE|全部实测最低WAPE|
|---|---:|---:|
|24小时完整曲线|2.373898591%|2.367483113%|
|7天完整曲线|2.706500896%|2.694095946%|
|7天每日平均负荷|1.904237896%|1.889162799%|

三个最低值仍来自此前不同研究配置，未达到一个方案全面或大幅领跑。第41至43轮共同因子、注意力及全参数训练、每日变化量匹配对照，共35组设置，6组开发训练与1组最终重训实际完成；两跨度共享配置均未通过开发升级条件。第42轮训练候选测试为24小时2.376932340%、7天2.704295335%、7天日均1.899620528%，有局部改善但不能整体升级。第43轮选中的是水平控制，不能把其24小时小幅改善称为差分方法改善。

当前推荐保留第32轮24小时与第36轮7天，19号入口不变。29/30/31号分别是共同因子、注意力训练、同条件水平控制的冻结研究历史复现，每个提供24小时与7天输出。它们核验已保存的2019年预测；不是今天实时推理服务。实际神经网络和训练程序、派生权重、冻结选择、原始缓存及审计都在experiment/round41至round43。

新增126项独立复核通过，包含注意力实际更新与其余应冻结参数等值、数据边界、缓存、预测及指标重放。每日差分额外一天历史纳入缓存签名，旧缓存输入通过既有PCA训练前缀及历史签名核验后恢复；预测、参数和原冻结选择未改变。修复前后证据保留。

全部实体仍在D盘，桌面为兼容junction。cell-a12085文件必要列完整读取，剩余下载0，未扩到其他集群。三轮实验全部退出，自动续跑保持PAUSED；本轮没有新下载、没有关闭其他程序，也没有删除研究数据或失败证据。
''',encoding='utf-8')
(T/'当前进展.md').write_text('''# 当前进展

2026年10月7日，第41至43轮完整完成，共35组设置、7次实际训练。新增126项科学复核通过。最终Word九页已逐页核对，全部比较表126条记录、98组不同预测。

本轮无共同升级：共同因子、注意力及全参数训练、变化量匹配对照未达到两个曲线共同领先。当前推荐仍第32/36轮，原最低研究记录未刷新，不能声称大幅提升。第43轮水平控制的24小时改善伴随7天退步，差分分支本身未被开发选中。

训练及缓存校验重放均已结束，没有等待内存或剩余下载。自动续跑保持PAUSED。当前研究目标尚未实现，负面结果和权重已保留，独立确认仍需新的确认数据。
''',encoding='utf-8')
(T/'资源占用说明.md').write_text('''# 资源占用说明

第41至43轮及缓存校验重放都已结束，无本轮残留训练/下载进程。官方cell-a必要列读取完成，剩余下载0；用户其他程序未关闭。

沿用已授权的小模型保护：物理准入0.75GiB、提交准入2.875GiB，实验Job聚合提交上限4GiB；4GiB并非需要预留的物理内存。运行物理低于0.75GiB或提交低于2GiB连续两次，只停止所属Job。没有修改其他大模型保护。

第42轮实际训练最后1层及输出层236万参数、最后2层及输出层413万参数和全参数1139万。6组开发、1组最终冻结重训，各80步。最大训练Job提交约2.96GB，完整训练结束，没有因内存等待而宣布科学失败。派生权重与官方原权重均保留；没有新下载，没有删研究证据。
''',encoding='utf-8')
"""
s=s[:start]+block+s[end:];s=s.replace('rows=114,unique_predictions=92','rows=126,unique_predictions=98').replace('science_check_count=140','science_check_count=126');compile(s,'delivery_factor_final.py','exec');(R/'code/delivery_factor_final.py').write_text(s,encoding='utf-8')
reader=(B/'round40/code/read_continued_research.py').read_text(encoding='utf-8').replace('round40/results/research_entry_verification.json','round43/results/research_entry_verification.json');(R/'code/read_factor_research.py').write_text(reader,encoding='utf-8')
v=(B/'round40/code/verify_continued_delivery.py').read_text(encoding='utf-8').replace("Q=B/'round40/results'","Q=B/'round43/results'").replace("Q/'continued_comparison.json'","Q/'factor_comparison.json'").replace("B/'round40/docs/report_qa/verification.json'","B/'round43/docs/report_qa/verification.json'").replace("len(a['rows'])==114 and len(a['unique_rows'])==92","len(a['rows'])==126 and len(a['unique_rows'])==98").replace('114 unified records 92 unique forecasts','126 unified records 98 unique forecasts').replace('140 scientific checks passed','126 scientific checks passed').replace("==140","==126").replace("len(rows)==114","len(rows)==126").replace('two current and eight research entries','two current and six research entries').replace('len(entries)==10','len(entries)==8').replace("B/'round40/docs/report_qa/layout_cleanup.json'","B/'round43/docs/report_qa/layout_cleanup.json'").replace('rows=114,unique_predictions=92','rows=126,unique_predictions=98').replace('research_entries=8','research_entries=6')
v=v.replace("'未找到同时稳健领跑','0.010492'","'尚未达到单一','共同因子'")
v=v.replace("'没有达到大幅提升'","'大幅领跑'")
v=v.replace("u=read(Q/f'usage_source_cut{cut}.json')", "u=read(B/'round40/results'/f'usage_source_cut{cut}.json')")
v=v.replace("check('one common frozen method not falsely declared winner'", "check('one common frozen method not falsely declared winner'")
v=v.replace("entries=read(B/'round36", "check('three factor rounds not adopted',all(not s['adopt_shared_scheme'] for s in a['factor_rounds'].values()))\nentries=read(B/'round36")
v=v.replace("print(json.dumps(dict(all_passed=True,checks=len(checks),report_sha256=sha(doc),current_entries=2,research_entries=6)))","print(json.dumps(dict(all_passed=True,checks=len(checks),report_sha256=sha(doc),current_entries=2,research_entries=6)))")
compile(v,'verify_factor_delivery.py','exec');(R/'code/verify_factor_delivery.py').write_text(v,encoding='utf-8')
print('Prepared new QA, six research entries, report-aware independent delivery checks')
