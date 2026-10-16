$ErrorActionPreference='Stop'
$taskQa=[System.IO.Path]::GetFullPath('${PROJECT_ROOT}\experiment\round28\docs\report_qa')
$taskFiles=@();$taskDirs=@()
foreach($taskPath in @($taskQa,(Join-Path $taskQa 'final'),(Join-Path $taskQa 'checked'))){
 $taskResolved=[System.IO.Path]::GetFullPath($taskPath)
 if(-not($taskResolved.Equals($taskQa,[System.StringComparison]::OrdinalIgnoreCase) -or $taskResolved.StartsWith($taskQa+'\',[System.StringComparison]::OrdinalIgnoreCase))){throw 'Target outside intended QA directory'}
 if(Test-Path -LiteralPath $taskResolved){
  $taskFiles+=@(Get-ChildItem -LiteralPath $taskResolved -File | Where-Object {$_.Name -match '^page-\d+\.png$' -or $_.Name -eq 'report_final.pdf'})
  if($taskResolved -ne $taskQa){$taskDirs+=$taskResolved}
 }
}
foreach($taskFile in $taskFiles){if(-not($taskFile.FullName.StartsWith($taskQa+'\',[System.StringComparison]::OrdinalIgnoreCase)) -or $taskFile.FullName.StartsWith((Join-Path $taskQa 'verified')+'\',[System.StringComparison]::OrdinalIgnoreCase)){throw 'Unexpected target'}}
$taskRecord=[pscustomobject]@{Scope='Only unused layout-attempt PNG/PDF files created this turn';Files=$taskFiles.Count;LogicalBytes=($taskFiles | Measure-Object Length -Sum).Sum;Paths=@($taskFiles.FullName);FinalVerifiedPagesPreserved=$true;RemovedUtc=[DateTime]::UtcNow.ToString('o')}
foreach($taskFile in $taskFiles){Remove-Item -LiteralPath $taskFile.FullName -Force}
foreach($taskDir in $taskDirs){if(@(Get-ChildItem -LiteralPath $taskDir -Force).Count -eq 0){Remove-Item -LiteralPath $taskDir}}
$taskRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath '${PROJECT_ROOT}\experiment\round28\results\layout_intermediate_cleanup.json' -Encoding utf8
$taskRecord | Select-Object Scope,Files,LogicalBytes,FinalVerifiedPagesPreserved
