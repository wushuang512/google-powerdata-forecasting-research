# 复现入口与研究档案索引

完整流程使用根目录`reproduce.py`，参见[复现指南](REPRODUCTION.md)和[实际验证记录](REPRODUCTION_VALIDATION.md)。

| 当前入口 | 作用 |
| --- | --- |
| `repro/support.py` | 独立工作目录、源码和官方输入/权重绑定 |
| `repro/google.py` | Google预处理、HDSF成员、所选实际梯度训练、R48校正与统一指标 |
| `repro/legacy.py` | NLR历史迁移比较所必需的Google R5/R8源模型 |
| `repro/nlr.py` | 不删减候选的原严格十四模型流程 |
| `repro/pipeline.py` | 串行执行、固定环境预检与输出哈希续跑 |
| `scripts/check_reproduced_results.py` | 新计算指标与原表逐项比较 |
| `tests/test_reproduction.py` | 统一评价公式和工作目录隔离测试 |

以下是保留的历史研究源码/协议/聚合结果；早期分支没有统一新启动器。34个仅排版或本机交付脚本已从当前公开快照移除，原研究文件未删除。

| 历史目录 | Python文件数 | 聚合CSV表数 |
| --- | ---: | ---: |
| `experiment/code` | 9 | 0 |
| `experiment/completed_main_summary_20261013` | 1 | 2 |
| `experiment/generalization_extended_20261013` | 16 | 0 |
| `experiment/generalization_review_20261012` | 2 | 0 |
| `experiment/generalization_validation_20261012` | 10 | 10 |
| `experiment/model_comparison_20261005` | 8 | 10 |
| `experiment/nlr_strict_protocol_20261013` | 16 | 3 |
| `experiment/nlr_user_protocol_20261013` | 17 | 4 |
| `experiment/results` | 0 | 13 |
| `experiment/round10` | 0 | 2 |
| `experiment/round11` | 0 | 2 |
| `experiment/round12` | 0 | 2 |
| `experiment/round13` | 0 | 2 |
| `experiment/round14` | 0 | 2 |
| `experiment/round15` | 2 | 0 |
| `experiment/round16` | 3 | 0 |
| `experiment/round17` | 4 | 0 |
| `experiment/round18` | 2 | 0 |
| `experiment/round19` | 6 | 0 |
| `experiment/round2` | 11 | 11 |
| `experiment/round20` | 2 | 2 |
| `experiment/round21` | 2 | 2 |
| `experiment/round22` | 2 | 0 |
| `experiment/round23` | 2 | 0 |
| `experiment/round24` | 3 | 0 |
| `experiment/round25` | 2 | 0 |
| `experiment/round26` | 16 | 0 |
| `experiment/round27` | 2 | 0 |
| `experiment/round28` | 9 | 0 |
| `experiment/round29` | 5 | 0 |
| `experiment/round3` | 7 | 20 |
| `experiment/round30` | 4 | 0 |
| `experiment/round31` | 2 | 0 |
| `experiment/round32` | 5 | 0 |
| `experiment/round33` | 2 | 0 |
| `experiment/round34` | 2 | 0 |
| `experiment/round35` | 8 | 0 |
| `experiment/round36` | 2 | 0 |
| `experiment/round37` | 2 | 0 |
| `experiment/round38` | 2 | 0 |
| `experiment/round39` | 2 | 0 |
| `experiment/round4` | 8 | 8 |
| `experiment/round40` | 11 | 1 |
| `experiment/round41` | 1 | 0 |
| `experiment/round42` | 1 | 0 |
| `experiment/round43` | 12 | 0 |
| `experiment/round44` | 12 | 10 |
| `experiment/round45` | 3 | 3 |
| `experiment/round46` | 9 | 5 |
| `experiment/round47` | 5 | 8 |
| `experiment/round48` | 4 | 2 |
| `experiment/round5` | 14 | 10 |
| `experiment/round6` | 0 | 2 |
| `experiment/round7` | 0 | 2 |
| `experiment/round8` | 0 | 1 |
| `experiment/round8b` | 0 | 1 |
| `experiment/round8c` | 0 | 1 |
| `experiment/seven_day_completion_20261005` | 2 | 1 |
