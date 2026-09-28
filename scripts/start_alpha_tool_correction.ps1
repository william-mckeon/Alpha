param(
 [string]$Name = ('alpha-tool-correction-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
 [string]$RunConfig = 'runs/test2/alpha-tool-correction-run-config-v3',
 [string]$DataRoot = 'runs/test2/tool-correction-data-v4',
 [string]$Image = 'arcus-alpha-tool-correction:experimental',
 [string]$ExecutorImage = 'arcus-alpha-fresh-executor:local',
 [switch]$PrepareOnly,
 [switch]$EvaluateOnly,
 [switch]$WithCoding,
 [switch]$NoMemoryWatchdog,
 [string]$StopAt = '',
 [int]$MaxSeconds = 0
)
$ErrorActionPreference='Stop'
if ($NoMemoryWatchdog) {
 if (-not $StopAt -or [DateTimeOffset]::Parse($StopAt) -le [DateTimeOffset]::Now) { throw 'A future absolute StopAt is required without the memory watchdog' }
}
if ($MaxSeconds -eq 0) { $MaxSeconds=if ($EvaluateOnly) { 1800 } else { 172800 } }
if ($MaxSeconds -lt 1 -or $MaxSeconds -gt 172800) { throw 'Watchdog duration must be 1..172800 seconds' }
$workspace=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $workspace
if ($Name -notmatch '^alpha-tool-correction-[a-z0-9-]+$') { throw 'Invalid container name' }
$running=@(docker ps -q)
if ($LASTEXITCODE -ne 0) { throw 'Docker is unavailable' }
foreach ($container in $running) {
 $existing=(docker inspect $container | ConvertFrom-Json)[0]
 if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect existing container' }
 if ($existing.HostConfig.DeviceRequests -or $existing.HostConfig.Devices -or $existing.Name -match '^/alpha-tool-correction-') { throw 'Another GPU or fresh training container is running' }
}
$configPath=(Resolve-Path -LiteralPath $RunConfig).Path
$learner=Get-Content -LiteralPath (Join-Path $configPath 'learner.json') -Raw | ConvertFrom-Json
$plan=Get-Content -LiteralPath (Join-Path $configPath 'training.json') -Raw | ConvertFrom-Json
$runRelative=$learner.root.Replace('\','/')
if ($runRelative -notmatch '^runs/test2/alpha-tool-correction-[a-z0-9-]+$') { throw 'Invalid isolated run root' }
$sourceRelative=(Split-Path $plan.source_checkpoint -Parent).Replace('\','/')
if ($sourceRelative -notmatch '^runs/test2/alpha-[a-z0-9-]+$' -or $sourceRelative -eq $runRelative) { throw 'Invalid source checkpoint root' }
$runPath=Join-Path $workspace $runRelative
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$dataPath=(Resolve-Path -LiteralPath $DataRoot).Path
$datasetEnv=Join-Path $configPath 'dataset.env'
'ALPHA_DATASET_MOUNTS={"datasetforge":"/dataset"}' | Set-Content -LiteralPath $datasetEnv -Encoding ascii
$corpus=(Resolve-Path 'alpha dataset').Path
$image=docker image inspect $Image --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Image unavailable' }
$reference=(docker inspect alpha-memory-before | ConvertFrom-Json)[0]
$fingerprint=$reference.Config.Env | Where-Object { $_.StartsWith('ALPHA_HOST_FINGERPRINT=') }
if (-not $fingerprint) { throw 'Host identity unavailable' }
if (-not $NoMemoryWatchdog) {
 python -c "import ctypes,sys; sys.path.insert(0,'scripts'); from watch_alpha_training import Memory,host_memory_low; m=Memory();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));assert not host_memory_low(m.free,m.total)"
 if ($LASTEXITCODE -ne 0) { throw 'Host free memory at or below user-selected 0.5 GiB guard' }
}
$mounts=@('--mount',"type=bind,source=$workspace/$sourceRelative,target=/app/$sourceRelative,readonly", '--mount',"type=bind,source=$runPath,target=/app/$runRelative",'--mount',"type=bind,source=$configPath,target=/run-config,readonly",'--mount',"type=bind,source=$dataPath,target=/review,readonly",'--mount',"type=bind,source=$corpus,target=/dataset,readonly")
if (-not (Test-Path (Join-Path $runPath 'three-stage-continuation.json'))) {
 docker run --rm --network none --memory 3g --cpus 2 @mounts $image scripts/prepare_alpha_tool_correction.py --config /run-config/learner.json
 if ($LASTEXITCODE -ne 0) { throw 'Fresh preparation failed' }
}
if ($PrepareOnly) { Write-Output 'Prepared; training remains paused'; exit }
$evidence=Join-Path $runPath $Name
New-Item -ItemType Directory -Path $evidence -ErrorAction Stop | Out-Null
$pause=Join-Path $runPath 'pause-training'
if (-not $EvaluateOnly) {
 $plan=Get-Content -LiteralPath (Join-Path $configPath 'training.json') -Raw | ConvertFrom-Json
 $gates=Get-Content -LiteralPath (Join-Path $configPath 'gates.json') -Raw | ConvertFrom-Json
 if (-not $plan.training_enabled -or -not $gates.training_authorized) { throw 'Corrective dataset and gates remain pending review' }
 if (Test-Path -LiteralPath $pause) { Remove-Item -LiteralPath $pause }
}
$arguments=@('run','-d','--name',$Name,'--gpus','all','--network','none','--memory','8g','--memory-swap','8g','--cpus','2','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges','--log-opt','max-size=10m','--log-opt','max-file=3',
 '-e','ALPHA_RUNTIME_PROFILE=alpha-container-v1','-e','ALPHA_GPU_MODE=controlled-docker','-e',"ALPHA_RUNTIME_IMAGE=$image",'-e',$fingerprint,
 '-e','ALPHA_JOB_CONTROL=/control','--env-file',$datasetEnv,
 '--mount','type=volume,source=arcus-alpha-three-stage_alpha-job-control,target=/control')+$mounts
$executorName=$null
try {
if ($WithCoding -or -not $EvaluateOnly) {
 $executorImage=docker image inspect $ExecutorImage --format '{{.Id}}'
 if ($LASTEXITCODE -ne 0) { throw 'Build the fresh coding executor image first' }
 $network='alpha-tool-correction-internal'
 $networkState=docker network inspect $network --format '{{.Internal}}' 2>$null
 if ($LASTEXITCODE -ne 0) { docker network create --internal $network | Out-Null }
 elseif ($networkState -ne 'true') { throw 'Evaluation network must be internal' }
 $credential=Join-Path $configPath 'executor.env'
 if (-not (Test-Path -LiteralPath $credential)) {
  ('ALPHA_EXECUTOR_TOKEN='+[guid]::NewGuid().ToString('N')+[guid]::NewGuid().ToString('N')) | Set-Content -LiteralPath $credential
 }
 $executorName=$Name+'-executor'
 docker run -d --name $executorName --network $network --network-alias executor --memory 256m --cpus 0.5 --pids-limit 64 --env-file $credential --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock $executorImage | Out-Null
 if ($LASTEXITCODE -ne 0) { throw 'Coding executor launch failed' }
 $ready=$false
 for ($attempt=0; $attempt -lt 10; $attempt++) {
  docker exec $executorName python3 -c "import os; from baby_arcus.transport import Client; assert Client('http://127.0.0.1:8933',os.environ['ALPHA_EXECUTOR_TOKEN'],attempts=1).request('GET','/health')['ready']" 2>$null
  if ($LASTEXITCODE -eq 0) { $ready=$true; break }
  Start-Sleep -Seconds 1
 }
 if (-not $ready) { throw 'Executor readiness failed' }
 $networkIndex=$arguments.IndexOf('--network')
 $arguments[$networkIndex+1]=$network
 $arguments+=@('--env-file',$credential,'-e','ALPHA_EXECUTOR_URL=http://executor:8933')
}
if ($EvaluateOnly) {
 $arguments+=@($image,'scripts/evaluate_alpha_fresh.py','--config','/run-config/learner.json','--output',"/app/$runRelative/$Name/evaluation.json")
 if ($WithCoding) { $arguments+=@('--coding') }
} else {
 $arguments+=@($image,'scripts/run_alpha_tool_correction.py','--config','/run-config/learner.json')
}
 docker @arguments
 if ($LASTEXITCODE -ne 0) { throw 'Launch failed' }
if ($NoMemoryWatchdog) {
 @{memory_watchdog=$false;stop_at=$StopAt;authorized_by='User explicitly requested watchdog removal and pause by tomorrow 4pm'} | ConvertTo-Json | Set-Content (Join-Path $evidence 'runtime-override.json')
 python scripts/wait_alpha_deadline.py $Name $runPath $evidence $StopAt
} else {
 python scripts/watch_alpha_training.py $Name $evidence --max-seconds $MaxSeconds
}
 $state=docker inspect $Name --format '{{json .State}}' | ConvertFrom-Json
 $state | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $evidence 'container-state.json')
 if ($state.Running -or $state.ExitCode -ne 0) { throw 'Worker did not exit cleanly; inspect durable run status' }
} catch {
 New-Item -ItemType File -Path $pause -Force | Out-Null
 throw
} finally {
 if ($executorName) { docker stop $executorName | Out-Null }
}
