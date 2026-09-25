param(
 [string]$Name = ('alpha-efficiency-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
 [string]$Image = 'arcus-alpha-efficiency:experimental',
 [string[]]$PythonArgs = @('-m','unittest','tests.test_efficiency_repairs','tests.test_context_efficiency','tests.baby_arcus.test_packed_training_store','-v')
)
$ErrorActionPreference = 'Stop'
$testRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $testRoot
if ($Name -notmatch '^alpha-efficiency-[a-z0-9-]+$') { throw 'Invalid diagnostic name' }
$running = @(docker ps -q)
if ($LASTEXITCODE -ne 0 -or $running.Count -gt 0) { throw 'Docker unavailable or other containers running' }
python -c "import ctypes; from scripts.watch_alpha_memory_minute import Memory; m=Memory(); m.length=ctypes.sizeof(m); assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); print('Free RAM GiB:',round(m.free/2**30,2)); assert m.free>=2*2**30"
if ($LASTEXITCODE -ne 0) { throw 'Insufficient RAM' }
$imageId = docker image inspect $Image --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Image missing' }
$reference = (docker inspect alpha-memory-before | ConvertFrom-Json)[0]
$hostIdentity = $reference.Config.Env | Where-Object { $_.StartsWith('ALPHA_HOST_FINGERPRINT=') }
if (-not $hostIdentity) { throw 'Missing host identity' }
$evidence = Join-Path $testRoot ('runs/diagnostics/' + $Name)
if (Test-Path -LiteralPath $evidence) { throw 'Use a fresh evidence directory' }
New-Item -ItemType Directory -Path $evidence | Out-Null
$parent = Join-Path $testRoot 'runs/test2/alpha-phase2-attempt-001'
@{image=$imageId;command=$PythonArgs;checkpoint=(Get-Content (Join-Path $parent 'candidate.json') -Raw | ConvertFrom-Json);host_minimum_free_gib=2} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $evidence 'run-contract.json')
$argsList = @('run','-d','--name',$Name,'--gpus','all','--network','none','--memory','8g','--memory-swap','8g','--cpus','2','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',
 '-e','ALPHA_RUNTIME_PROFILE=alpha-container-v1','-e','ALPHA_GPU_MODE=controlled-docker','-e',"ALPHA_RUNTIME_IMAGE=$imageId",'-e',$hostIdentity,'-e','ALPHA_JOB_CONTROL=/control',
 '-e','ALPHA_PHASE2B_CUDA_TEST=1','--mount','type=volume,source=arcus-alpha-three-stage_alpha-job-control,target=/control',
 '--mount',"type=bind,source=$parent,target=/parent,readonly",'--mount',"type=bind,source=$evidence,target=/evidence",$imageId) + $PythonArgs
docker @argsList
if ($LASTEXITCODE -ne 0) { throw 'Container start failed' }
python scripts/watch_alpha_memory_minute.py $Name $evidence
docker logs $Name 2>&1 | Tee-Object -FilePath (Join-Path $evidence 'container.log')
$state = docker inspect $Name --format '{{json .State}}' | ConvertFrom-Json
$state | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $evidence 'container-state.json')
Write-Output "Evidence: $evidence"
if ($state.Running -or $state.ExitCode -ne 0) { throw 'Diagnostic did not complete successfully' }
