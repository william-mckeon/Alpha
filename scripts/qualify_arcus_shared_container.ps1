param([string]$Config)
$ErrorActionPreference = 'Stop'
$arcusProject = Split-Path -Parent $PSScriptRoot
$arcusImage = docker image inspect arcus-visual:qualification --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Existing qualification image required.' }
$arcusPlatform = docker run --rm --network none --read-only --entrypoint cat $arcusImage /etc/os-release
if ($LASTEXITCODE -ne 0 -or -not ($arcusPlatform -match 'VERSION_ID="22.04"')) { throw 'Ubuntu 22.04 required.' }
docker run --rm --network none --read-only --tmpfs /tmp -e PYTHONDONTWRITEBYTECODE=1 -v "${arcusProject}:/workspace:ro" -w /workspace --entrypoint python3 $arcusImage -m unittest tests.baby_arcus.test_shared_learning tests.baby_arcus.test_shared_checkpoint tests.baby_arcus.test_shared_runtime tests.baby_arcus.test_shared_replay tests.baby_arcus.test_shared_curriculum tests.baby_arcus.test_shared_qualification tests.baby_arcus.test_shared_retention tests.baby_arcus.test_shared_temporal tests.baby_arcus.test_shared_pooling tests.baby_arcus.test_shared_memory tests.baby_arcus.test_shared_causal tests.baby_arcus.test_interaction_integration
if ($LASTEXITCODE -ne 0) { throw 'Shared foundation tests failed.' }
Write-Output "Shared CPU/HTTP contracts passed on $arcusImage; not a learning or promotion gate."
if ($Config) {
    $arcusConfigPath = [IO.Path]::GetFullPath((Join-Path $arcusProject $Config))
    $arcusConfigData = Get-Content -Raw -LiteralPath $arcusConfigPath | ConvertFrom-Json
    $arcusRun = [IO.Path]::GetFullPath((Join-Path $arcusProject $arcusConfigData.root))
    $arcusPrefix = $arcusProject.TrimEnd('\') + '\'
    if (-not $arcusConfigPath.StartsWith($arcusPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $arcusRun.StartsWith($arcusPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Portable qualification requires the config and run inside this project.'
    }
    $arcusRelativeRun = [IO.Path]::GetRelativePath($arcusProject,$arcusRun).Replace('\','/')
    $arcusRelativeConfig = [IO.Path]::GetRelativePath($arcusProject,$arcusConfigPath).Replace('\','/')
    docker run --rm --gpus all --network none --read-only --tmpfs /tmp -e PYTHONDONTWRITEBYTECODE=1 -e TIKTOKEN_CACHE_DIR=/workspace/runs/arcus_shared_v3/tokenizer-cache -v "${arcusProject}:/workspace:ro" -v "${arcusRun}:/workspace/${arcusRelativeRun}" -w /workspace --entrypoint python3 $arcusImage scripts/qualify_arcus_shared_recovery.py --config $arcusRelativeConfig --report recovery-ubuntu.json
    if ($LASTEXITCODE -ne 0) { throw 'Full-size Linux GPU recovery failed.' }
}

