# 算力中心负荷预测研究

Google PowerData 2019、Google ClusterData 2019辅助特征、NLR HPC设施负荷及海光寺日电量的阶段研究。包含模型代码、实验协议、验证冻结记录、评分表、机制分析和报告文字。

**本仓库不包含数据集、处理后标签、逐点预测、大型预训练权重、运行环境或本地缓存。** 公开数据与预训练权重需从官方来源另外取得；海光寺私有原表不提供下载。小规模融合系数作为选模记录保留。参考项目通过固定提交的Git子模块引用，没有复制其源码或历史到本仓库。

## 复现入口（2026-10-18补充）

请使用根目录`reproduce.py`，按[完整复现指南](docs/REPRODUCTION.md)准备固定版本、官方数据与权重，在Git目录之外串行重训及评分。原`experiment/`是历史档案，不能直接当成统一启动入口。

本次已验证官方下载/预处理、真实局部模型重训及指标重建；完整神经训练和NLR十四模型全量重跑的验证范围单列于[实际验证记录](docs/REPRODUCTION_VALIDATION.md)，不保证不同机器逐位相同。数据和TimesFM微调权重仍不上传。

## 先看结果

Google当前采用第48轮HDSF方案，24小时采用PatchTST-FM日均残差及同组50%收缩，7天采用TimesFM3/Toto2/PatchTST-FM零均值形状校正。

| Google任务 | 基线WAPE | 当前采用WAPE |
| --- | ---: | ---: |
| 24小时完整曲线 | 3.199450% | 2.329603% |
| 7天完整曲线 | 3.211919% | 2.701184% |
| 7天每日平均负荷 | 2.262403% | 1.868830% |

Google曲线先逐供电域计算WAPE，再对57域等权平均；日均仅使用96点全部有效的目标日。24小时主基线为前一天同刻，7天为前三天同刻均值。Google功率是归一化利用率，不是kW。

当前采用属于已暴露测试月份上的研究取舍，原开发升级门槛未通过的事实保留。它不是所有曾评分候选的全部指标最优，也不是新的独立盲测。7天窗口重叠，不将每个点或每个目标日记录当作独立样本。

NLR最新严格协议比较了十四种方案。按验证MAPE预先选择的是24小时Toto2、7天第48轮HDSF迁移；测试描述性最低为24小时Chronos-2/TiRex-2等权3.901513%，7天第8轮三专家校准扩展7.423535%，对应日均6.855010%。测试最低不用于回改选择，Google最优不代表跨数据集最优。

## 阅读入口

- [阶段性全面总结](算力中心负荷预测研究阶段性总结_20261015.md)
- [当前Google推荐报告](docs/reports/google-current-recommendation.md)
- [NLR严格协议报告](docs/reports/nlr-strict-rerun.md)
- [泛化科学性审查](docs/reports/generalization-review.md)
- [HDSF原因及数据处理](docs/reports/hdsf-mechanism-and-preprocessing.md)
- [实验目录索引](docs/EXPERIMENT_INDEX.md)
- [数据来源与标签口径](docs/DATA_SOURCES.md)
- [复现范围及公开副本说明](docs/REPRODUCTION.md)
- [第三方来源与许可说明](THIRD_PARTY_NOTICES.md)

## 无数据查看评分

只需Python标准库：

```console
python scripts/review_results.py
```

该命令读取已公布的聚合结果，不加载数据集或模型，也不重新训练。Google完整指标见`模型各项指标完整结果.csv`及`本轮方案全部指标对照.csv`；NLR主表为`NLR严格重跑_新协议主要模型测试全指标.csv`，包含42条结果；逐预测日表包含210条结果。

## 目录

```text
nlr-hpc-pue/code/                NLR官方下载与原始文件质量审计
experiment/                     研究源码、各轮协议、选择及聚合指标
  code/                         初始功率模型、预处理和统一评分
  round2/ ... round48/           后续结构、微调、HDSF及校正探索
  nlr_strict_protocol_20261013/  最新NLR严格协议研究
google-clusterdata-2019/code/    算力特征读取、聚合及匹配对照代码
google-powerdata-2019/           数据来源和许可说明，不含数据
docs/reports/                   既有Word的文字表格提取，不含附件
scripts/                        评分查看工具
```

本地Word、原始数据、权重和完整审计材料仍保存在研究工作区，未因发布而移动或删除。公开副本去除了个人路径和本地运维记录，原始冻结哈希与公开副本哈希的关系记录于`docs/PUBLICATION_PROVENANCE.json`。

报告快照截至2026年10月13日完成的评估，阶段总结日期为2026年10月15日，公开整理日期为10月16日。没有将本次上传当作新实验。
