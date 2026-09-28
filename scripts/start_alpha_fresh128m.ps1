param(
 [string]$Name = ('alpha-fresh128m-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
 [string]$RunConfig = 'runs/test2/alpha-fresh128m-run-config-v2',
 [string]$DataRoot = 'runs/test2/fresh128m-dataset-v1',
 [switch]$PrepareOnly,
 [switch]$EvaluateOnly,
 [switch]$WithCoding
)
$ErrorActionPreference='Stop'
$workspace=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $workspace
if ($Name -notmatch '^alpha-fresh128m-[a-z0-9-]+$') { throw 'Invalid container name' }
$running=@(docker ps -q)
if ($LASTEXITCODE -ne 0) { throw 'Docker is unavailable' }
foreach ($container in $running) {
 $existing=(docker inspect $container | ConvertFrom-Json)[0]
 if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect existing container' }
 if ($existing.HostConfig.DeviceRequests -or $existing.HostConfig.Devices -or $existing.Name -match '^/alpha-fresh128m-') { throw 'Another GPU or fresh training container is running' }
}
$runPath=(Resolve-Path 'runs/test2/alpha-fresh128m-16k-seed-2101').Path
$configPath=(Resolve-Path -LiteralPath $RunConfig).Path
$dataPath=(Resolve-Path -LiteralPath $DataRoot).Path
$corpus=(Resolve-Path 'alpha dataset').Path
$image=docker image inspect arcus-alpha-efficiency:experimental --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Image unavailable' }
$reference=(docker inspect alpha-memory-before | ConvertFrom-Json)[0]
$fingerprint=$reference.Config.Env | Where-Object { $_.StartsWith('ALPHA_HOST_FINGERPRINT=') }
if (-not $fingerprint) { throw 'Host identity unavailable' }
python -c "import ctypes,sys; sys.path.insert(0,'scripts'); from watch_alpha_training import Memory,host_memory_low; m=Memory();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));assert not host_memory_low(m.free,m.total)"
if ($LASTEXITCODE -ne 0) { throw 'Host free memory at or below user-selected 0.5 GiB guard' }
$mounts=@('--mount',"type=bind,source=$runPath,target=/app/runs/test2/alpha-fresh128m-16k-seed-2101",'--mount',"type=bind,source=$configPath,target=/run-config,readonly",'--mount',"type=bind,source=$dataPath,target=/review,readonly",'--mount',"type=bind,source=$corpus,target=/dataset,readonly")
if (-not (Test-Path (Join-Path $runPath 'three-stage-continuation.json'))) {
 docker run --rm --network none --memory 3g --cpus 2 @mounts $image scripts/prepare_alpha_fresh_training.py --config /run-config/learner.json
 if ($LASTEXITCODE -ne 0) { throw 'Fresh preparation failed' }
}
if ($PrepareOnly) { Write-Output 'Prepared; training remains paused'; exit }
$evidence=Join-Path $runPath $Name
New-Item -ItemType Directory -Path $evidence -ErrorAction Stop | Out-Null
$pause=Join-Path $runPath 'pause-training'
if (-not $EvaluateOnly -and (Test-Path -LiteralPath $pause)) { Remove-Item -LiteralPath $pause }
$arguments=@('run','-d','--name',$Name,'--gpus','all','--network','none','--memory','8g','--memory-swap','8g','--cpus','2','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges','--log-opt','max-size=10m','--log-opt','max-file=3',
 '-e','ALPHA_RUNTIME_PROFILE=alpha-container-v1','-e','ALPHA_GPU_MODE=controlled-docker','-e',"ALPHA_RUNTIME_IMAGE=$image",'-e',$fingerprint,
 '-e','ALPHA_JOB_CONTROL=/control','-e','ALPHA_DATASET_MOUNTS={"datasetforge":"/dataset"}',
 '--mount','type=volume,source=arcus-alpha-three-stage_alpha-job-control,target=/control')+$mounts
$executorName=$null
try {
if ($WithCoding -or -not $EvaluateOnly) {
 $executorImage=docker image inspect arcus-alpha-fresh-executor:local --format '{{.Id}}'
 if ($LASTEXITCODE -ne 0) { throw 'Build the fresh coding executor image first' }
 $network='alpha-fresh128m-internal'
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
 $arguments+=@($image,'scripts/evaluate_alpha_fresh.py','--config','/run-config/learner.json','--output',"/app/runs/test2/alpha-fresh128m-16k-seed-2101/$Name/evaluation.json")
 if ($WithCoding) { $arguments+=@('--coding') }
} else {
 $arguments+=@($image,'scripts/run_alpha_fresh_40000.py','--config','/run-config/learner.json')
}
 docker @arguments
 if ($LASTEXITCODE -ne 0) { throw 'Launch failed' }
 python scripts/watch_alpha_training.py $Name $evidence
 $state=docker inspect $Name --format '{{json .State}}' | ConvertFrom-Json
 $state | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $evidence 'container-state.json')
 if ($state.Running -or $state.ExitCode -ne 0) { throw 'Worker did not exit cleanly; inspect durable run status' }
} catch {
 New-Item -ItemType File -Path $pause -Force | Out-Null
 throw
} finally {
 if ($executorName) { docker stop $executorName | Out-Null }
}

