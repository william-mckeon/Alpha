param([Parameter(Mandatory=$true)][DateTimeOffset]$StopAt,
      [Parameter(Mandatory=$true)][string]$Root,
      [ValidateSet("probe","baseline","application","preflight","train","conversion","expanded-preflight","specialization","verify-depth","package","verify-package","teacher-qualification","adaptation-qualification","adaptation")][string]$Mode="probe",
      [string]$RequestsFile="", [switch]$PersistMemory, [switch]$Phase8Initialization,
      [string]$DataRoot="", [string]$PreflightReport="", [string]$AdapterPath="", [string]$ResumePath="", [string]$ConvertedPath="", [string]$ExpandedPath="", [string]$PackagePath="", [string]$TeacherPath="", [string]$EvaluationRoot="", [ValidateSet('light','developmental','full')][string]$EvaluationTier='full')
$ErrorActionPreference='Stop'
$workspace=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $workspace
$project=Get-Content configs/arcus3/project.json -Raw | ConvertFrom-Json
$runtime=Get-Content configs/arcus3/local_runtime.json -Raw | ConvertFrom-Json
if ($project.authorization.inference -ne $true -or $project.authorization.cloud -or $project.authorization.publication) { throw 'Local inference scope required' }
if ($Mode -in @('preflight','train') -and ($project.authorization.training -ne $true -or $project.training_scope -ne 'dense-control-v1')) { throw 'Bounded dense training scope required' }
if ($project.donor.revision -ne '31b70e2e869a7173562077fd711b654946d38674') { throw 'Donor pin mismatch' }
$maxSessionSeconds=if ($Mode -eq 'adaptation') {86400} else {1800}
if ($StopAt -le [DateTimeOffset]::Now -or ($StopAt-[DateTimeOffset]::Now).TotalSeconds -gt $maxSessionSeconds) { throw 'Future deadline within allowed session bound required' }
if ($Root -notmatch '^runs/arcus3/(donor-probe|baseline|application|preflight|dense-control|conversion|expanded-preflight|release|teacher|adaptation)-[a-z0-9-]+$') { throw 'Isolated donor-probe root required' }
if ($Mode -eq 'adaptation') {
 $adaptation=Get-Content configs/arcus3/backbone_adaptation.json -Raw | ConvertFrom-Json
 $windows=Get-Content configs/arcus3/training_windows.json -Raw | ConvertFrom-Json
 if (!$adaptation.campaign_enabled -or !$windows.enabled) { throw 'Campaign and training windows are not enabled' }
}
if ($Mode -in @('adaptation','adaptation-qualification') -and (!$ConvertedPath -or !$DataRoot -or !$TeacherPath)) { throw 'Adaptation requires parent, data and teacher cache' }
if ($Mode -eq 'conversion' -and (!$project.authorization.conversion -or $project.authorization.training -or $project.conversion_scope -ne 'selective-experts-parity-v1')) { throw 'Construction-only scope required' }
if ($AdapterPath -and $ConvertedPath) { throw 'Choose one model variant' }
if ($Mode -eq 'expanded-preflight' -and (!$project.authorization.expanded_preflight -or $project.expanded_scope -ne 'qualification-v1' -or !$ConvertedPath -or !$DataRoot)) { throw 'Bounded expanded qualification scope, parent and data required' }
if ($Mode -eq 'specialization' -and (!$project.authorization.specialization -or $project.specialization_scope -ne 'matched-64-v1' -or !$ConvertedPath -or !$DataRoot)) { throw 'Matched campaign scope, parent and data required' }
$rootPath=Join-Path $workspace $Root
if (Test-Path -LiteralPath $rootPath) { throw 'Use a fresh probe root; never clear existing pauses' }
if ($runtime.image_id -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Verified image ID required' }
docker image inspect $runtime.image_id | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Pinned image unavailable; verify the rebuilt image and update local_runtime.json before launching' }
$ids=@(docker ps -q)
if ($LASTEXITCODE -ne 0) { throw 'Docker unavailable' }
foreach ($id in $ids) {
 $info=(docker inspect $id | ConvertFrom-Json)[0]
 if ($info.HostConfig.DeviceRequests -or $info.HostConfig.Devices) { throw 'Another GPU container is active' }
}
$donorPath=(Resolve-Path ('artifacts/arcus3/donor/'+$project.donor.revision)).Path
New-Item -ItemType Directory -Path $rootPath | Out-Null
$name='arcus3-donor-'+(Get-Date -Format 'yyyyMMdd-HHmmss')
$deadlineUtc=$StopAt.ToUniversalTime().ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'")
$entryScript=if ($Mode -eq 'verify-package') {'scripts/verify_alpha_3.py'} elseif ($Mode -eq 'package') {'scripts/package_alpha_3.py'} elseif ($Mode -eq 'expanded-preflight') {'scripts/qualify_arcus3_training.py'} elseif ($Mode -eq 'conversion') {'scripts/convert_arcus3.py'} elseif ($Mode -eq 'preflight') {'scripts/benchmark_arcus3.py'} elseif ($Mode -eq 'train') {'scripts/train_arcus3.py'} elseif ($Mode -eq 'application') {'scripts/chat_arcus3.py'} elseif ($Mode -eq 'baseline') {'scripts/evaluate_arcus3.py'} else {'scripts/probe_arcus3_donor.py'}
if ($Mode -eq 'specialization') { $entryScript='scripts/train_arcus3_specialization.py' }
if ($Mode -eq 'verify-depth') { $entryScript='scripts/verify_arcus3_depth.py' }
if ($Mode -in @('adaptation','adaptation-qualification')) { $entryScript='scripts/train_arcus3_backbone_adaptation.py' }
if ($Mode -eq 'teacher-qualification') { $entryScript='scripts/prepare_arcus3_teacher_targets.py' }
$argsDocker=@('run','-d','--name',$name,'--gpus','all','--network','none','--memory',$runtime.memory,'--memory-swap',$runtime.memory,'--cpus',"$($runtime.cpus)",'--pids-limit',"$($runtime.pids)",'--cap-drop','ALL','--security-opt','no-new-privileges',
 '-e','ARCUS3_CONTROLLED_DOCKER=1','-e','ALPHA_JOB_CONTROL=/job-control',
 '--mount',"type=volume,source=$($runtime.gpu_lock_volume),target=/job-control",
 '--mount',"type=bind,source=$donorPath,target=/donor,readonly",
 '--mount',"type=bind,source=$rootPath,target=/output",
 $runtime.image_id,$entryScript,'--deadline',$deadlineUtc,'--max-new-tokens',"$($runtime.max_new_tokens)")
if ($RequestsFile -or $PersistMemory) {
 if ($Mode -ne 'application') { throw 'Application options require application mode' }
 if ($RequestsFile) {
  $requestPath=(Resolve-Path -LiteralPath $RequestsFile).Path
  $imageIndex=[Array]::IndexOf($argsDocker,$runtime.image_id)
  $argsDocker=$argsDocker[0..($imageIndex-1)]+@('--mount',"type=bind,source=$requestPath,target=/requests.json,readonly")+$argsDocker[$imageIndex..($argsDocker.Length-1)]+@('--requests','/requests.json')
 }
 if ($PersistMemory) { $argsDocker+=@('--persist-memory') }
}
if ($Mode -in @('preflight','train') -and !$DataRoot) { throw 'Reviewed data root required' }
if ($Mode -eq 'train' -and !$PreflightReport) { throw 'Measured preflight report required' }
foreach ($mountSpec in @(@{source=$TeacherPath;target='/teacher';option='--teacher'},@{source=$PackagePath;target='/package';option='--package'},@{source=$ExpandedPath;target='/expanded';option='--expanded'},@{source=$ConvertedPath;target='/converted';option='--converted'},@{source=$DataRoot;target='/data';option=''},@{source=$PreflightReport;target='/preflight.json';option=$(if ($Mode -eq 'adaptation') {'--qualification-report'} else {'--preflight-report'})},@{source=$AdapterPath;target='/adapter';option='--adapter'},@{source=$ResumePath;target='/resume';option='--resume'})) {
 if ($mountSpec.source) {
  $mountPath=(Resolve-Path -LiteralPath $mountSpec.source).Path
  $imageIndex=[Array]::IndexOf($argsDocker,$runtime.image_id)
  $argsDocker=$argsDocker[0..($imageIndex-1)]+@('--mount',"type=bind,source=$mountPath,target=$($mountSpec.target),readonly")+$argsDocker[$imageIndex..($argsDocker.Length-1)]
  if ($mountSpec.option) { $argsDocker+=@($mountSpec.option,$mountSpec.target) }
 }
}
if ($Mode -in @('adaptation-qualification','teacher-qualification')) { $argsDocker+='--qualification' }
if ($Mode -eq 'baseline') { $argsDocker+=@('--tier',$EvaluationTier) }
if ($Phase8Initialization) {
 if ($Mode -ne 'baseline' -or !$ConvertedPath -or $ExpandedPath) { throw 'Initial Phase 8 evaluation requires pristine conversion' }
 $argsDocker+='--phase8-initialization'
}
if ($EvaluationRoot) {
 if ($Mode -ne 'adaptation') { throw 'Evaluation receipt is only for an adaptation resume' }
 $evaluationPath=(Resolve-Path -LiteralPath $EvaluationRoot).Path
 $imageIndex=[Array]::IndexOf($argsDocker,$runtime.image_id)
 $argsDocker=$argsDocker[0..($imageIndex-1)]+@('--mount',"type=bind,source=$evaluationPath,target=/evaluation-receipt,readonly")+$argsDocker[$imageIndex..($argsDocker.Length-1)]+@('--evaluation-root','/evaluation-receipt')
}
docker @argsDocker | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Container creation failed' }
@{container=$name;image_id=$runtime.image_id;deadline=$deadlineUtc;memory_watchdog=$false;mode=$Mode} | ConvertTo-Json | Set-Content (Join-Path $rootPath 'runtime.json')
try {
 while ($true) {
  $running=docker inspect $name --format '{{.State.Running}}'
  if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect own container' }
  if ($running -ne 'true') { break }
  if ($Mode -in @('adaptation','adaptation-qualification') -and [DateTimeOffset]::Now -ge $StopAt.AddSeconds(-300)) {
   if (!(Test-Path (Join-Path $rootPath 'pause-training'))) { Set-Content (Join-Path $rootPath 'pause-training') 'Graceful deadline pause' }
  }
  if ([DateTimeOffset]::Now -ge $StopAt -or (Test-Path (Join-Path $rootPath 'pause-inference'))) { docker kill $name | Out-Null; break }
  Start-Sleep -Seconds 1
 }
} finally {
 if ((docker inspect $name --format '{{.State.Running}}') -eq 'true') { docker kill $name | Out-Null }
 docker logs $name 2>&1 | Set-Content (Join-Path $rootPath 'worker.log')
 $state=(docker inspect $name | ConvertFrom-Json)[0].State
 $state | ConvertTo-Json | Set-Content (Join-Path $rootPath 'container-state.json')
}
if ($state.ExitCode -ne 0) { throw "Probe exited $($state.ExitCode); inspect logs before retry" }

if ($Mode -eq "baseline") {
 & .venv/Scripts/python.exe scripts/report_arcus3.py --root $Root --deadline $deadlineUtc
 if ($LASTEXITCODE -ne 0) { throw "Baseline executor/report failed; preserve partial evidence" }
}
