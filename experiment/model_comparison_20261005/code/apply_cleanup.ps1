param([Parameter(Mandatory=$true)][string]$ResearchRoot)
$ErrorActionPreference='Stop'
$taskResolvedRoot=[System.IO.Path]::GetFullPath($ResearchRoot).TrimEnd('\')
if ($taskResolvedRoot -ne '${PROJECT_ROOT}') { throw 'Unexpected root' }
$taskPlanPath=Join-Path $taskResolvedRoot 'experiment\model_comparison_20261005\cleanup_plan.json'
$taskRecordPath=Join-Path $taskResolvedRoot 'experiment\model_comparison_20261005\cleanup_record.json'
if(Test-Path -LiteralPath $taskRecordPath) { throw 'Cleanup already recorded; inspect before retry' }
$taskPlan=Get-Content -LiteralPath $taskPlanPath -Raw | ConvertFrom-Json
if($taskPlan.root -ne $taskResolvedRoot) { throw 'Plan root mismatch' }
$taskProcesses=Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'GooglePowerData' -and $_.CommandLine -match 'direct_shared_experiment|native_covariate_experiment|complete_usage_experiment|extract_cpu.py|finetune'}
if($taskProcesses) { throw 'A research training or prediction process is active' }
# All recursive targets must resolve inside D, match the plan, and contain no reparse points.
foreach($taskTarget in $taskPlan.targets) {
    $taskAbsolute=[System.IO.Path]::GetFullPath($taskTarget.path)
    $taskFromRelative=[System.IO.Path]::GetFullPath((Join-Path $taskResolvedRoot $taskTarget.relative_path))
    if($taskAbsolute -ne $taskFromRelative -or -not $taskAbsolute.StartsWith($taskResolvedRoot+'\',[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Target escapes root' }
    $taskItem=Get-Item -LiteralPath $taskAbsolute -Force
    if($taskItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'Target is a link' }
    $taskItems=@(Get-ChildItem -LiteralPath $taskAbsolute -Recurse -Force)
    if($taskItems | Where-Object {$_.Attributes -band [System.IO.FileAttributes]::ReparsePoint}) { throw 'Target contains a link' }
    $taskFiles=@($taskItems | Where-Object {-not $_.PSIsContainer})
    if($taskFiles.Count -ne $taskTarget.files_count) { throw 'File count changed' }
    foreach($taskFile in $taskTarget.files) {
        $taskFilePath=Join-Path $taskResolvedRoot $taskFile.path
        $taskFileItem=Get-Item -LiteralPath $taskFilePath
        if($taskFileItem.Length -ne $taskFile.bytes) { throw 'File length changed' }
        if((Get-FileHash -LiteralPath $taskFilePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskFile.sha256) { throw 'File content changed' }
    }
}
foreach($taskProp in $taskPlan.protected_files.PSObject.Properties) {
    $taskProtected=Join-Path $taskResolvedRoot $taskProp.Name
    if((Get-FileHash -LiteralPath $taskProtected -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProp.Value.sha256) { throw 'Protected file changed' }
}
$taskEntry=Join-Path $taskResolvedRoot $taskPlan.deprecated_entry
$taskOriginalEntry=Get-Content -LiteralPath $taskEntry -Raw
$taskBegin=(Get-Date).ToUniversalTime().ToString('o')
$taskLog=@()
foreach($taskTarget in $taskPlan.targets) {
    Remove-Item -LiteralPath $taskTarget.path -Recurse -Force
    if(Test-Path -LiteralPath $taskTarget.path) { throw 'Removal did not complete' }
    $taskLog+=[ordered]@{path=$taskTarget.relative_path;files=$taskTarget.files_count;logical_bytes=$taskTarget.bytes;reason=$taskTarget.reason;deleted=$true}
}
# Disable only the decommissioned neural entry. Route a user who opens it to the actual retained best entry.
$taskNewEntry="@echo off`r`nchcp 65001 >nul`r`necho 第十四轮直接网络未采用，专用权重已按要求删除。`r`necho 原分数和汇报保留；以下运行当前正式最佳模型。`r`ncall `"%~dp009_现有最佳微调_预测次日.cmd`"`r`n"
Set-Content -LiteralPath $taskEntry -Value $taskNewEntry -Encoding utf8
$taskRecord=[ordered]@{state='completed';started_utc=$taskBegin;completed_utc=(Get-Date).ToUniversalTime().ToString('o');root=$taskResolvedRoot;targets=$taskLog;deleted_files=$taskPlan.total_files;deleted_logical_bytes=$taskPlan.total_logical_bytes;physical_space_release_not_measured=$true;plan_sha256=(Get-FileHash -LiteralPath $taskPlanPath).Hash.ToLowerInvariant();original_entry_text=$taskOriginalEntry;entry_action='Entry16 explicitly states decommissioning and delegates to09';research_scores_and_source_data_retained=$true}
$taskRecord | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $taskRecordPath -Encoding utf8
$taskRecord | Select-Object state,deleted_files,deleted_logical_bytes,completed_utc | ConvertTo-Json
