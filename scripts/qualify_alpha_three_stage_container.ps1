param([string]$Image = 'arcus-test2:qualification')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$name = 'three-stage-cpu-services-' + [Guid]::NewGuid().ToString('N')
$output = Join-Path $projectRoot "runs/test2/$name"
New-Item -ItemType Directory -Path $output | Out-Null
# No GPU, no network, no writable production mounts. A new tiny fixture is created.
& docker run --rm --network none --memory 2g --cpus 1 --pids-limit 128 `
    -e CUDA_VISIBLE_DEVICES= -e ALPHA_RUNTIME_PROFILE=alpha-container-v1 -e ALPHA_JOB_CONTROL=/tmp/alpha-control `
    --mount "type=bind,source=$projectRoot,target=/app,readonly" `
    --mount "type=bind,source=$output,target=/app/runs/test2/$name" `
    --entrypoint python3 $Image scripts/qualify_alpha_cpu_services.py --root "/app/runs/test2/$name"
if ($LASTEXITCODE -ne 0) { throw "CPU service qualification failed; inspect $output" }
Write-Output "CPU fixture report: $output/report.json. Production GPU qualification remains separate."
