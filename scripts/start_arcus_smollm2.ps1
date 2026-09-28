param(
 [Parameter(Mandatory=$true)][string]$Corpus,
 [Parameter(Mandatory=$true)][string]$Root,
 [Parameter(Mandatory=$true)][DateTimeOffset]$StopAt,
 [string]$Image='arcus-foundation:nanotron-v1',
 [int]$Updates=1,
 [switch]$Resume,
 [string]$CampaignAuthorization=''
)
$ErrorActionPreference='Stop'
$workspace=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $workspace
if ($StopAt -le [DateTimeOffset]::Now) { throw 'A future deadline is required' }
if (-not $CampaignAuthorization -and ($Updates -lt 1 -or $Updates -gt 2 -or ($StopAt-[DateTimeOffset]::Now).TotalSeconds -gt 1800)) { throw 'Pilot is limited to two updates and 30 minutes' }
if ($Root -notmatch '^runs/foundation/[a-z0-9-]+$') { throw 'Use a new isolated foundation root' }
$running=@(docker ps -q)
if ($LASTEXITCODE -ne 0) { throw 'Docker unavailable' }
foreach ($id in $running) {
 $info=(docker inspect $id | ConvertFrom-Json)[0]
 if ($info.HostConfig.DeviceRequests -or $info.HostConfig.Devices) { throw 'Another GPU container is active' }
}
$imageId=docker image inspect $Image --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Build the pinned image first' }
$corpusPath=(Resolve-Path -LiteralPath $Corpus).Path
$rootPath=Join-Path $workspace $Root
if (Test-Path (Join-Path $rootPath 'pause-training')) { throw 'Explicit pause exists; do not remove it silently' }
if (-not $Resume -and (Test-Path $rootPath)) { throw 'Existing run requires Resume' }
New-Item -ItemType Directory -Path $rootPath -Force | Out-Null
$sha=[Security.Cryptography.SHA256]::Create()
$fingerprint=[BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($env:COMPUTERNAME))).Replace('-','').ToLower()
$name='arcus-foundation-'+(Get-Date -Format 'yyyyMMdd-HHmmss')
$arguments=@('run','-d','--name',$name,'--gpus','all','--network','none','--memory','8g','--memory-swap','8g','--cpus','2','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',
 '-e','ALPHA_RUNTIME_PROFILE=alpha-container-v1','-e','ALPHA_GPU_MODE=controlled-docker','-e',"ALPHA_RUNTIME_IMAGE=$imageId",'-e',"ALPHA_HOST_FINGERPRINT=$fingerprint",'-e','ALPHA_JOB_CONTROL=/job-control',
 '--mount','type=volume,source=arcus-alpha-three-stage_alpha-job-control,target=/job-control','--mount',"type=bind,source=$corpusPath,target=/corpus.sqlite,readonly",'--mount',"type=bind,source=$rootPath,target=/app/$Root")
if ($CampaignAuthorization) {
 $authorizationPath=(Resolve-Path -LiteralPath $CampaignAuthorization).Path
 $arguments+=@('--mount',"type=bind,source=$authorizationPath,target=/authorization.json,readonly")
}
$deadlineUtc=$StopAt.ToUniversalTime().ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'")
$arguments+=@($imageId,'scripts/train_arcus_smollm2.py','--corpus','/corpus.sqlite','--root',$Root,'--deadline',$deadlineUtc,'--max-updates',"$Updates")
if ($Resume) { $arguments+='--resume' }
if ($CampaignAuthorization) { $arguments+=@('--campaign-authorization','/authorization.json') }
docker @arguments | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Container start failed' }
@{container=$name;image_id=$imageId;deadline=$StopAt.ToString('o');memory_watchdog=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $rootPath 'runtime.json')
try {
 while ((docker inspect $name --format '{{.State.Running}}') -eq 'true') {
  if ([DateTimeOffset]::Now -ge $StopAt.AddMinutes(-1)) { New-Item -ItemType File -Path (Join-Path $rootPath 'pause-training') -Force | Out-Null }
  if ([DateTimeOffset]::Now -ge $StopAt) { docker kill $name | Out-Null; break }
  Start-Sleep -Seconds 5
 }
} finally {
 docker logs $name 2>&1 | Set-Content -LiteralPath (Join-Path $rootPath 'worker.log')
}
$state=(docker inspect $name | ConvertFrom-Json)[0].State
$state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $rootPath 'container-state.json')
if ($state.ExitCode -ne 0) { throw "Worker exited $($state.ExitCode); inspect durable checkpoint and logs before retry" }
