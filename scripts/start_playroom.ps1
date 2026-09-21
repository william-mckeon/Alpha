param([int]$Port = 8890, [int]$SimulationPort = 8891, [switch]$Desktop)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$playroomPython = Join-Path $projectRoot '.venv-baby/Scripts/python.exe'
if ($Desktop) {
    $playroomPython = Join-Path $projectRoot '.venv-desktop/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $playroomPython)) {
        throw 'Install requirements-desktop.lock into .venv-desktop first.'
    }
}
if (-not (Test-Path -LiteralPath $playroomPython)) {
    $playroomPython = (Get-Command python -ErrorAction Stop).Source
}
Push-Location -LiteralPath $projectRoot
try {
    $playroomModule = if ($Desktop) { 'baby_arcus.desktop' } else { 'baby_arcus.services.playroom' }
    $modelOptions = if ($Desktop) { @() } else { @('--connect-model') }
    & $playroomPython -m $playroomModule --port $Port --simulation-port $SimulationPort @modelOptions
    if ($LASTEXITCODE -ne 0) { throw "Playroom exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
