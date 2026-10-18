# 从原始文件复现当前结果

2026-10-18补充。原2026-10-16版本主要是源码和指标档案，不能据此保证独立完整复现；本版增加根目录`reproduce.py`、固定资产锁、隔离工作目录及重训练流程。**查看原表、使用旧预测重新计分、重新运行模型，是三个不同层次。** 实际验证到哪一步见[验证记录](REPRODUCTION_VALIDATION.md)，不能把未执行的全量流程写成已通过。

## 支持范围与边界

- Google：当前采用的第48轮HDSF。24小时为“半日单纯形加PatchTST-FM日均残差校正及同组50%收缩”；7天为“半日单纯形加TimesFM3/Toto2/PatchTST-FM零均值形状全量校正”。固定采用原已公布参数，重新生成底层预测、按原方法重训所需权重，再计算指标；不按本次测试名次选参。
- NLR：严格协议的十四种方案、24小时/7天曲线/每日均值、逐预测日及非重叠检验。其六个Google迁移对照需要先生成Google来源的局部模型、Ridge和R8/R46/R47微调状态；程序缺少这些文件会明确报错，**不会悄悄换成八模型表**。
- 更早各轮保留为研究档案，不声称每个探索分支都有新的统一一键重跑入口。海光寺原始Excel未公开，外部人员仅凭本仓库无法独立复现该案例。
- TimesFM 3.0官方许可禁止转发模型及衍生权重；本仓库提供官方获取方式及本地训练代码，不提供该微调检查点。见[官方许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE)。其他模型也各按原条款取得。

## 1. 获取源码与固定第三方版本

```console
git clone https://github.com/wushuang512/google-powerdata-forecasting-research.git
cd google-powerdata-forecasting-research
git submodule update --init --recursive
```

HDSF参考、uni2ts、Granite均固定提交，具体值在`.gitmodules`、`docs/UPSTREAM_SOURCES.json`及`repro/upstream_sources.lock.json`。新入口检查实际使用的143份上游Python文件哈希，按LF统一核对以兼容Git的CRLF检出。不要用这些项目的最新分支替代指定版本。

## 2. 分开准备三个环境

已验证的宿主为Windows、Python 3.12。Google原HDSF推理保持CPU Torch 2.8.0；所选神经权重训练保持CUDA Torch 2.11.0/cu128；NLR统计处理使用单独版本组合。需要支持cu128的NVIDIA驱动。Linux/macOS全模型流程没有在本次发布中验证；公式测试不需要GPU。

以下为PowerShell命令，环境和数据工作目录均建在Git目录之外：

```powershell
py -3.12 -m venv ..\.power-cpu
py -3.12 -m venv ..\.power-gpu
py -3.12 -m venv ..\.power-nlr
$cpu = (Resolve-Path ..\.power-cpu\Scripts\python.exe).Path
$gpu = (Resolve-Path ..\.power-gpu\Scripts\python.exe).Path
$nlr = (Resolve-Path ..\.power-nlr\Scripts\python.exe).Path
$work = Join-Path (Split-Path (Get-Location) -Parent) 'power-repro-work'

& $cpu -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
& $gpu -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
& $cpu -m pip install -r repro/requirements-common.txt
& $gpu -m pip install -r repro/requirements-common.txt
& $nlr -m pip install -r repro/requirements-nlr-cpu.txt
```

这里的Granite只使用固定上游源码中的PatchTST-FM模块，新入口将该源码加入模块搜索路径，**不安装其整套工具包**。整包声明的Torch>=2.10与原CPU Torch2.8不一致，不能把它们硬塞进同一安装清单。模型包的实际版本快照在`repro/runtime-cpu.json`、`runtime-gpu.json`；NLR版本组合已通过本次独立预处理与统计重拟合验证。没有声称本次在空机器上重新安装验证了所有依赖。

## 3. 建立空工作目录并取得官方输入

```powershell
& $cpu reproduce.py --work $work init
& $cpu reproduce.py --work $work fetch google
& $cpu reproduce.py --work $work fetch models --accept-timesfm-license
& $cpu reproduce.py --work $work doctor
```

`--accept-timesfm-license`表示使用者已阅读模型原条款并仅作允许的本地研究，不赋予商业或转发权利。模型和原数据均从官方地址获取；`repro/assets.lock.json`固定七个模型提交和逐文件SHA256，Google 58个对象还固定对象generation。发现既有文件与锁不符即停止。首次下载模型需要数GiB空间。**复现R48不需要下载数百GiB的ClusterData列文件。** ClusterData辅助建模的旧研究仍保留，但不是R48输入。

正常复现不需要原作者电脑路径。`local-assets`和`--source-root`仅供已有官方文件/固定源码镜像的本地核验使用；可建立硬链接，不计作重新下载验证。训练状态和预测输出写到独立目录，不写回官方权重。禁止把工作目录放进Git仓库。

## 4. 重跑Google当前推荐

```powershell
& $cpu reproduce.py --work $work google-all --cpu-python $cpu --gpu-python $gpu --include-nlr-source-models
& $cpu scripts/check_reproduced_results.py --work $work --task google --output ..\google-reproduction-check.json
```

该串行流程包括：

1. 从原始58文件重建57×31×96输入、质量掩码与生产功率代理。Google原始`time`保留为序列时间，不擅自当UTC绝对时间。
2. 重训HDSF局部LightGBM：开发截止D14、最终截止D22，四个固定随机种子。保留原实现已披露的D1残差基线不一致，以复现原结果；修复它属于新实验。
3. 在CPU分别重新运行11个HDSF成员的开发/测试任务；原14/21/28天历史和同组/单变量方式保持不变。PatchTST按原内存回退使用batch8，不以换设备产生的数值差别宣称提分。
4. 对TimesFM3、Toto2、PatchTST按固定tail/40步/学习率3e-6重新训练第46轮所需状态，学习率计划分母仍为80；对PatchTST按curve/160步/1.5e-6重训第47轮状态。开发与最终分别截止14/22天，随机种子20261012。**不是从随机权重预训练整个基础模型。**
5. 每次计算微调增量时，用同设备原始权重作匹配控制，再加入原HDSF。24小时采用日均校正和组内50%收缩；7天使用去日均的形状校正。用原投影处理0至1边界，原开发残差生成经验80%区间。
6. 导出全部主指标，并逐项与原公开表比较。数值容差明确写在比较脚本中，比较不改变参数、不把容差通过称为逐位相同。
7. `--include-nlr-source-models`额外重新生成Google R5/R8迁移所需文件：三模型历史预测、Ridge拟合、R8固定100步训练。没有该选项时不准备这条仅供NLR历史迁移对照的支路。

不需要一次跑完；流水记录每个串行任务、退出码及输出哈希，失败后用同一命令续跑。已完成文件被改动会停止，不能混用另一份配置的缓存。完整模型流程可能耗时数小时；日志在工作目录`reproduction/`。如需先检查所选训练接口，可分别运行：

```powershell
& $gpu reproduce.py --work $work neural-smoke timesfm3
& $gpu reproduce.py --work $work neural-smoke toto2
& $gpu reproduce.py --work $work neural-smoke patchtst
```

这三个命令各做一次真实参数更新，用后即释放，仅验证前向、梯度及参数更新；不输出新的精度结论，不替代正式40/160步训练。

## 5. 重跑NLR严格十四模型评估

```powershell
& $cpu reproduce.py --work $work fetch nlr
& $nlr reproduce.py --work $work nlr-prepare
& $gpu reproduce.py --work $work nlr-run --cpu-python $nlr
& $nlr scripts/check_reproduced_results.py --work $work --task nlr --output ..\nlr-reproduction-check.json
```

`nlr-run`要求步骤4已准备六个历史迁移方案所需的Google源模型文件；它不依赖原作者的预测缓存。流程先验证、统一冻结两个时距和全部十四候选，之后重拟合训练加验证数据并做测试推理。三个原始基础模型没有新增NLR梯度微调，LightGBM与目标域单纯形参数会真实拟合。

NLR目标保持五项各自15分钟均值之和，任一分项缺失即缺失；温度华氏转摄氏，时间暂按UTC解释且`timezone_verified=false`。目标及历史目标不填充，历史协变量只前向填4格；14天历史；60/20/20连续日划分；未来仅日历。原始数据可用记录均值不代表15条分钟观测全部齐全。检验口径保持用户协议，不为了通过复现而改清洗、窗口、阈值或测试日期。

## 6. 查看输出和判定结果

- Google：工作目录`reproduction/results/google_R48_all_metrics.csv`、两项`google_R48_H*_comparison.json`。包括曲线/日均WAPE、MAE/RMSE、P95 APE、命中率、偏差、峰值及峰时、域组/日期改善、经验区间等。
- NLR：`experiment/nlr_strict_protocol_20261013/results/新协议主要模型测试全指标.csv`应为42行，逐预测日表应为210行，并提供验证向前CV、非重叠七天检验和实际标签覆盖。
- `scripts/check_reproduced_results.py`只对比新生成的指标表；`scripts/review_results.py`只读已发布的旧表，两者都不能替代训练/推理。
- Google WAPE按PDU计算后等权平均57域，日均主指标只取完整96点目标日；NLR按其冻结协议计算起报平均MAPE、池化WAPE、日均和计数。**不要改为全域池化Google WAPE再与旧表比较。**

## 冻结、已知问题与复现证据

原`experiment/`冻结记录仍描述历史原件；公开脱敏可能改变字节。新入口使用单独工作目录及新输入/源码绑定，不伪造原SHA、不改历史选择、不删除旧脚本的断言以冒充原件。复现采用的是已公开的固定参数，不重新搜索几十轮开发网格，也不能使多轮暴露的May2019/NLR测试重新成为盲测。

保留以下限制：Google仅一个月、NLR单设施；预训练语料重合不明；7天窗口重叠；NLR时区尚未核实；经验80%区间无跨设施覆盖保证。重跑出现差异应检查版本、输入哈希、数据掩码、数值设备和检查点，保留失败，不按测试分数选另一个“更接近”的参数。
