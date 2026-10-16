"""Describe actual development-only results without selecting a final model."""
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve()
round9=root.parent/'experiment/round9'
ledger=json.loads((round9/'results/partial_dev_ledger.json').read_text())
assert ledger['all_six_completed'] and ledger['completed_models']==6 and ledger['test_scored'] is False
rows=['| 正则强度 | 无 CPU：全部57域 | 加 CPU历史：全部57域 | 无 CPU：a五域 | 加 CPU历史：a五域 |',
      '|---:|---:|---:|---:|---:|']
for alpha in [10000,100000,1000000]:
    control=ledger['models'][f'control_ridge{alpha}']['score']
    history=ledger['models'][f'history_ridge{alpha}']['score']
    rows.append(f"| {alpha} | {control['all57']['wape']:.6f}% | {history['all57']['wape']:.6f}% | {control['cell_a_5']['wape']:.6f}% | {history['cell_a_5']['wape']:.6f}% |")
text='''# 六组轻量对照阶段

六组已实际完成，仅在训练期（预测标签5月8—14日）拟合，在验证期（5月15—22日）评分；本阶段没有读取测试真值选参或评分。

复用已有 TimesFM3、Chronos-2、TiRex-2 的冻结功率预测与现有微调组合，比较有/无完整 CPU 历史特征的相同 Ridge 正则强度。实际 CPU 数据覆盖 cell-a 的五个供电域，其他52域保持原冻结预测。

下表全部为**验证期 WAPE**，不能与先前2.374128%的测试期WAPE直接横向比较。

'''+ '\n'.join(rows)+'''

当前结论：收益小且随正则强度变化，没有显示大幅提高。仍需完成未来工作量预测特征、条件单调回归与多尺度稀疏 MoE 的已登记对照，再依协议冻结候选和组合权重，最后评估测试集。本阶段不能确定最终最佳模型。

逐项模型、特征、预测文件与协议/CPU/输入哈希已保存在 results/partial_dev_ledger.json；正式实验验证后直接复用，避免重复训练。

同一个May2019测试集以前多次被查看，本轮严格冻结筛选并不能恢复真正独立性，最终仍需跨月或新场站确认。
'''
(round9/'docs/轻量对照阶段.md').write_text(text,encoding='utf-8')
print('Six completed development models documented; no final selection or test scoring')
