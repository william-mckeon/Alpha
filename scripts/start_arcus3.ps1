param([Parameter(Mandatory=$true)][DateTimeOffset]$StopAt,
      [Parameter(Mandatory=$true)][string]$Root,
      [ValidateSet("probe","baseline")][string]$Mode="probe")
$ErrorActionPreference='Stop'
$workspace=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $workspace
$project=Get-Content configs/arcus3/project.json -Raw | ConvertFrom-Json
$runtime=Get-Content configs/arcus3/local_runtime.json -Raw | ConvertFrom-Json
if ($project.authorization.inference -ne $true -or $project.authorization.training -or $project.authorization.cloud -or $project.authorization.publication) { throw 'Phase 1 inference scope required' }
if ($project.donor.revision -ne '31b70e2e869a7173562077fd711b654946d38674') { throw 'Donor pin mismatch' }
if ($StopAt -le [DateTimeOffset]::Now -or ($StopAt-[DateTimeOffset]::Now).TotalSeconds -gt 1800) { throw 'Future deadline within 30 minutes required' }
if ($Root -notmatch '^runs/arcus3/(donor-probe|baseline)-[a-z0-9-]+$') { throw 'Isolated donor-probe root required' }
$rootPath=Join-Path $workspace $Root
if (Test-Path -LiteralPath $rootPath) { throw 'Use a fresh probe root; never clear existing pauses' }
if ($runtime.image_id -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Verified image ID required' }
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
$entryScript=if ($Mode -eq 'baseline') {'scripts/evaluate_arcus3.py'} else {'scripts/probe_arcus3_donor.py'}
$argsDocker=@('run','-d','--name',$name,'--gpus','all','--network','none','--memory',$runtime.memory,'--memory-swap',$runtime.memory,'--cpus',"$($runtime.cpus)",'--pids-limit',"$($runtime.pids)",'--cap-drop','ALL','--security-opt','no-new-privileges',
 '-e','ARCUS3_CONTROLLED_DOCKER=1','-e','ALPHA_JOB_CONTROL=/job-control',
 '--mount',"type=volume,source=$($runtime.gpu_lock_volume),target=/job-control",
 '--mount',"type=bind,source=$donorPath,target=/donor,readonly",
 '--mount',"type=bind,source=$rootPath,target=/output",
 $runtime.image_id,$entryScript,'--deadline',$deadlineUtc,'--max-new-tokens',"$($runtime.max_new_tokens)")
docker @argsDocker | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Container creation failed' }
@{container=$name;image_id=$runtime.image_id;deadline=$deadlineUtc;memory_watchdog=$false;mode=$Mode} | ConvertTo-Json | Set-Content (Join-Path $rootPath 'runtime.json')
try {
 while ($true) {
  $running=docker inspect $name --format '{{.State.Running}}'
  if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect own container' }
  if ($running -ne 'true') { break }
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
