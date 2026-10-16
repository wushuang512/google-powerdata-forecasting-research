# Google PowerData 2019 下载说明

下载日期：2026-09-29。官方名称：PowerData 2019 traces，数据集标识 powerdata-2019。

## 官方来源与年份

- 官方项目：https://github.com/google/cluster-data/blob/master/PowerData2019.md
- 官方说明：https://github.com/google/cluster-data/blob/master/power_trace_documentation.pdf
- 数据存储桶：gs://powerdata_2019
- HTTPS 文件入口：https://storage.googleapis.com/powerdata_2019/<文件名>
- 官方发布公告：https://groups.google.com/g/googleclusterdata-discuss/c/VzC1BONPvlE
- 数据观测期间：2019 年 5 月，共 31 天；官方文档日期为 2024-02-15，发布公告日期为 2024-02-27。2019 是数据年份。
- 覆盖 10 个 cell、57 个供电域（55 个传统 PDU、2 个 MVPP）。它与 ClusterData2019 工作负载数据配套，但不是同一个数据集。

## 本地文件

```text
google-powerdata-2019/
  raw/                         # 官方原始文件，58 个 .csv.gz
    cella_pdu6.csv.gz           # 示例；每个供电域一个文件
    ...
    celli_mvpp1.csv.gz
    cellj_mvpp2.csv.gz
    machine_to_pdu_mapping.csv.gz
  csv/                         # 对应的 58 个解压 CSV，未改写内容或行顺序
  docs/
    PowerData2019.md
    power_trace_documentation.pdf
    power_trace_analysis_colab.ipynb
    gcs_objects.json            # 下载时的官方文件元数据
    download.curl               # 原始下载地址与本地保存位置
  manifest.json                # 逐文件大小、来源、MD5、SHA-256、行数和字段
  LICENSE-NOTICE.md
  README_zh.md
```

官方按文件分别使用 GZIP 压缩，并非提供一个总 ZIP 包。
58 个压缩数据文件合计 3,254,733 字节（3.255 MB / 3.104 MiB）。
解压后数据合计 24,763,748 字节（24.764 MB / 23.617 MiB）。上述大小不含说明文档。
所有 58 个文件的字节数和 MD5 均与官方存储桶元数据一致；所有文件均通过 GZIP 解压完整性检查。SHA-256 是本次本地计算值，供后续复核。

## 字段与验证结果

每个供电域文件具有 8,928 行数据（不含表头），57 个文件共 508,896 行。
所有供电域时间戳唯一；排序后相邻时间戳相差 300,000,000 微秒，即 5 分钟。
原始 CSV 行顺序并非按时间升序排列，开展时序划分、重采样、训练前应先按 time 排序。

功率数据字段：
- time：相对时间戳，单位微秒。600,000,000 对应 2019-05-01 00:00 PT；末点为 2019-05-31 23:55 PT。时间转换公式为起点加 (time - 600000000) 微秒。
- cell、pdu：匿名集群和供电域标识。
- measured_power_util：该 5 分钟窗口的平均总功率利用率。
- production_power_util：估计的生产工作负载功率利用率。
- bad_measurement_data、bad_production_power_data：质量标记，保留原始值。

利用率是功率与容量之比，不能直接视为 kW/MW。官方描述包含机房冷却设备等非 IT 设备功率。
原始时间粒度为 5 分钟；15 分钟数据需要另行聚合。
质量标记的官方文档文字将 false 描述为低置信度，与 bad_* 字段命名存在语义疑点；此下载任务不据此清洗或筛除任何记录，使用前宜结合官方示例核实。

机器映射表包含 96,616 行，字段为 machine_id、pdu、cell；官方文档说明映射覆盖传统架构的 cell a 至 h。

## 许可和引用

数据与官方说明文档采用 CC BY 4.0：https://creativecommons.org/licenses/by/4.0/
使用和分享时应适当署名，附许可链接，并注明所做修改。
本目录 raw/ 原始字节未修改；csv/ 仅解压，内容与原文件一致。
官方请求研究者引用背景论文：
Data Center Power Oversubscription with a Medium Voltage Power Plane and Priority-Aware Capping (ASPLOS 2020).
https://research.google/pubs/data-center-power-oversubscription-with-a-medium-voltage-power-plane-and-priority-aware-capping/

完整逐文件清单见 manifest.json。
