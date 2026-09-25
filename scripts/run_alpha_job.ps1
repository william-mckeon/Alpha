param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('train_alpha_three_stage','evaluate_alpha_three_stage','evaluate_alpha_coding','evaluate_alpha_tool_discovery','qualify_alpha_three_stage','prepare_alpha_three_stage','prepare_alpha_training_data','prepare_alpha_source_lessons','qualify_alpha_gpu_runtime')]
    [string]$Job,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$JobArguments
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$compose = 'docker/baby-arcus/compose.alpha-three-stage.yaml'
& docker compose --env-file .env -f $compose config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Docker configuration failed validation.' }
$service = if ($Job -in @('prepare_alpha_three_stage','prepare_alpha_training_data','prepare_alpha_source_lessons','evaluate_alpha_tool_discovery')) { 'prepare' } else { 'learner' }
if ($Job -eq 'qualify_alpha_gpu_runtime') { $service = 'gpu-probe' }
& python -X utf8 scripts/inspect_alpha_resources.py --mode job --service $service
if ($LASTEXITCODE -ne 0) { throw 'Insufficient Docker capacity or competing workload. No job launched.' }
if ($service -in @('learner','gpu-probe')) {
    & "$PSScriptRoot/inspect_alpha_runtime.ps1" -RequireQualified:($Job -ne 'qualify_alpha_gpu_runtime')
}
# CPU preparation has no GPU request; the shared control volume serializes GPU work.
& docker compose --env-file .env -f $compose run --rm --no-deps $service "scripts/$Job.py" @JobArguments
if ($LASTEXITCODE -ne 0) { throw "Container job failed: $Job (exit $LASTEXITCODE). No native fallback." }
