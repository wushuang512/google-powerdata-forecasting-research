-- Replace YOUR_PROJECT.YOUR_DATASET with an authorized project/dataset.
-- Upload the official machine_to_pdu_mapping.csv to power_mapping first.
-- Run a dry run and inspect totalBytesProcessed BEFORE any billable execution.
-- maximumBytesBilled must be set to the user-approved ceiling in job config.
-- CPU and memory are normalized units, not cores or watts.
WITH unambiguous_mapping AS (
  SELECT cell,machine_id,ANY_VALUE(pdu) AS pdu
  FROM `YOUR_PROJECT.YOUR_DATASET.power_mapping`
  GROUP BY cell,machine_id HAVING COUNT(DISTINCT pdu)=1
), usage_rows AS (
SELECT 'a' AS cell, machine_id, start_time, end_time,
  average_usage.cpus AS cpu, average_usage.memory AS memory,
  assigned_memory, sample_rate
FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
WHERE collection_type = 0 AND machine_id > 0
  AND end_time > start_time AND end_time-start_time <= 300000001
  AND end_time > 600000000 AND end_time <= 2679000000000
UNION ALL
SELECT 'b' AS cell, machine_id, start_time, end_time,
  average_usage.cpus AS cpu, average_usage.memory AS memory,
  assigned_memory, sample_rate
FROM `google.com:google-cluster-data.clusterdata_2019_b.instance_usage`
WHERE collection_type = 0 AND machine_id > 0
  AND end_time > start_time AND end_time-start_time <= 300000001
  AND end_time > 600000000 AND end_time <= 2679000000000
UNION ALL
SELECT 'c' AS cell, machine_id, start_time, end_time,
  average_usage.cpus AS cpu, average_usage.memory AS memory,
  assigned_memory, sample_rate
FROM `google.com:google-cluster-data.clusterdata_2019_c.instance_usage`
WHERE collection_type = 0 AND machine_id > 0
  AND end_time > start_time AND end_time-start_time <= 300000001
  AND end_time > 600000000 AND end_time <= 2679000000000
UNION ALL
SELECT 'd' AS cell, machine_id, start_time, end_time,
  average_usage.cpus AS cpu, average_usage.memory AS memory,
  assigned_memory, sample_rate
FROM `google.com:google-cluster-data.clusterdata_2019_d.instance_usage`
WHERE collection_type = 0 AND machine_id > 0
  AND end_time > start_time AND end_time-start_time <= 300000001
  AND end_time > 600000000 AND end_time <= 2679000000000
UNION ALL
SELECT 'e' AS cell, machine_id, start_time, end_time,
  average_usage.cpus AS cpu, average_usage.memory AS memory,
  assigned_memory, sample_rate
FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage`
WHERE collection_type = 0 AND machine_id > 0
  AND end_time > start_time AND end_time-start_time <= 300000001
  AND end_time > 600000000 AND end_time <= 2679000000000
UNION ALL
SELECT 'f' AS cell, machine_id, start_time, end_time,
  average_usage.cpus AS cpu, average_usage.memory AS memory,
  assigned_memory, sample_rate
FROM `google.com:google-cluster-data.clusterdata_2019_f.instance_usage`
WHERE collection_type = 0 AND machine_id > 0
  AND end_time > start_time AND end_time-start_time <= 300000001
  AND end_time > 600000000 AND end_time <= 2679000000000
UNION ALL
SELECT 'g' AS cell, machine_id, start_time, end_time,
  average_usage.cpus AS cpu, average_usage.memory AS memory,
  assigned_memory, sample_rate
FROM `google.com:google-cluster-data.clusterdata_2019_g.instance_usage`
WHERE collection_type = 0 AND machine_id > 0
  AND end_time > start_time AND end_time-start_time <= 300000001
  AND end_time > 600000000 AND end_time <= 2679000000000
UNION ALL
SELECT 'h' AS cell, machine_id, start_time, end_time,
  average_usage.cpus AS cpu, average_usage.memory AS memory,
  assigned_memory, sample_rate
FROM `google.com:google-cluster-data.clusterdata_2019_h.instance_usage`
WHERE collection_type = 0 AND machine_id > 0
  AND end_time > start_time AND end_time-start_time <= 300000001
  AND end_time > 600000000 AND end_time <= 2679000000000
), completed AS (
  SELECT m.cell,m.pdu,u.* EXCEPT(cell),
    CAST(CEIL((end_time-600000000)/900000000.0)-1 AS INT64) AS slot,
    (end_time-start_time)/1000000.0 AS duration_seconds
  FROM usage_rows u JOIN unambiguous_mapping m
    ON u.cell=m.cell AND u.machine_id=m.machine_id
)
SELECT cell,pdu,slot,
  SUM(cpu*duration_seconds)/900.0 AS completed_job_cpu_time_NCU_per_second,
  SUM(memory*duration_seconds)/900.0 AS completed_job_memory_time_per_second,
  SUM(assigned_memory*duration_seconds)/900.0 AS assigned_memory_time_per_second,
  COUNT(*) AS observed_job_windows,
  COUNT(DISTINCT machine_id) AS observed_machines,
  COUNTIF(cpu IS NULL) AS missing_cpu_windows,
  COUNTIF(memory IS NULL) AS missing_memory_windows,
  MIN(end_time) AS min_record_end_us,MAX(end_time) AS max_record_end_us
FROM completed WHERE slot BETWEEN 0 AND 2975
GROUP BY cell,pdu,slot ORDER BY cell,pdu,slot;
-- Entire windows are assigned to completion bins so every record is available
-- by the bin end. This creates a delay of at most 300 seconds. It is not an
-- exact overlap integral of instantaneous load in that 15-minute bin.
-- Empty/partly monitored bins remain missing; do not silently treat as zero.
