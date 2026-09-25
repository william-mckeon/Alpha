param([switch]$Build)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$plan = Get-Content configs/baby_arcus/alpha_three_stage.container.json -Raw | ConvertFrom-Json
if ($plan.training_enabled) { throw 'Startup requires training_enabled=false. Review data and run settings separately.' }
$compose = 'docker/baby-arcus/compose.alpha-three-stage.yaml'
# Compose loads the project .env; never print resolved configuration or credentials.
& docker compose --env-file .env -f $compose config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Invalid Docker configuration. Check required mount paths and credentials.' }
& python -X utf8 scripts/inspect_alpha_resources.py
if ($LASTEXITCODE -ne 0) { throw 'Insufficient Docker capacity or competing workload. No services launched.' }
if ($Build) {
    & docker compose --env-file .env -f $compose build learner executor
    if ($LASTEXITCODE -ne 0) { throw 'Container build failed.' }
}
& "$PSScriptRoot/inspect_alpha_runtime.ps1" -RequireQualified
& docker compose --env-file .env -f $compose up -d --wait --wait-timeout 90 learner review playroom executor
if ($LASTEXITCODE -ne 0) { throw 'Container startup failed; inspect service logs. No native fallback was attempted.' }
Write-Output 'Docker services ready: playroom http://127.0.0.1:8930; review http://127.0.0.1:8932. Training remains paused.'
