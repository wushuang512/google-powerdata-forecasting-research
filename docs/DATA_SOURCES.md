# 数据来源与标签口径

本仓库只提供来源、处理代码和实验结果，不附数据文件。

| 数据 | 官方来源或获取方式 | 本项目目标 |
| --- | --- | --- |
| Google PowerData 2019 | https://github.com/google/cluster-data/blob/master/PowerData2019.md ；公开桶`gs://powerdata_2019` | 2019年5月57个供电域实测功率利用率；5分钟聚合到15分钟 |
| Google ClusterData 2019 | https://github.com/google/cluster-data ；本研究完整读取cell-a的12,085个对象必要列 | 历史CPU NCU、归一化内存及资源协变量；只覆盖五个PDU，不是独立功率标签 |
| NLR HPC PUE | https://data.nlr.gov/submissions/300 | 15分钟五项功率均值之和，kW；此前另评估IT功率 |
| 海光寺日电量 | 本地项目提供的电能统计Excel，不随仓库发布 | 30天每日总电量，kWh；不能验证日内96点或672点曲线 |

NLR功率文件：
https://data.nlr.gov/system/files/300/1757103411-esif.influx.buildingData.PUE.combined.parquet

NLR天气文件：
https://data.nlr.gov/system/files/300/1757105566-esif.influx.buildingData.outside.combined_2.parquet

NLR原始时间字段为`ts`，温湿度字段为`outdoor_air_temp`和`outdoor_air_humidity`。温度由华氏度转摄氏度，湿度为百分数。默认`facility_power_kw`是IT、制冷、HVAC、照明插座和泵五项各自区间均值的汇总，任一缺失则标签缺失；不是直接设施总表。时区暂按UTC解释，`timezone_verified=false`。

Google质量标记按官方分析示例解释：`bad_* = true`为异常。目标评分掩码和历史输入补值分开处理。7天日均评价使用完整96点日。

NLR严格实验的目标及历史目标不填充，仅历史协变量最多前向填充4格。未来只允许已知日历，不使用真实未来天气或分项负荷。不同阶段标签、缺测覆盖和测试窗口有变化，分数不能跨协议直接混排。

Google数据按CC BY 4.0来源说明署名。NLR按其官方DOE/NREL/ALLIANCE数据使用声明处理，不将它误标为CC BY。具体说明见对应官方页面。

NLR官方获取及质量审计源码位于`nlr-hpc-pue/code/`。按需手动执行以下命令，数据会写到指定本地研究目录，仍被Git忽略；本次发布未运行下载：

```console
python nlr-hpc-pue/code/download_nlr_pue.py <本地研究根目录> --phase metadata
python nlr-hpc-pue/code/download_nlr_pue.py <本地研究根目录> --phase data
python nlr-hpc-pue/code/audit_nlr_pue.py <本地研究根目录>
```

原始文件审计需要NumPy及PyArrow；模型评估依赖另按各轮环境准备。
