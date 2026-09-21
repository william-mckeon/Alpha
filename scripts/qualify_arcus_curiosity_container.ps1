param([switch]$GpuPerception)
$ErrorActionPreference = 'Stop'
$arcusProject = Split-Path -Parent $PSScriptRoot
$arcusImage = docker image inspect arcus-visual:qualification --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Existing Ubuntu qualification image required.' }
$arcusPlatform = docker run --rm --network none --read-only --entrypoint cat $arcusImage /etc/os-release
if ($LASTEXITCODE -ne 0 -or -not ($arcusPlatform -match 'VERSION_ID="22.04"')) { throw 'Ubuntu 22.04 required.' }
docker run --rm --network none --read-only --tmpfs /tmp -e PYTHONDONTWRITEBYTECODE=1 -v "${arcusProject}:/workspace:ro" -w /workspace --entrypoint python3 $arcusImage -m unittest tests.baby_arcus.test_curiosity tests.baby_arcus.test_curiosity_perception tests.baby_arcus.test_playroom tests.baby_arcus.test_object_perception_learning tests.baby_arcus.test_object_perception_runtime
if ($LASTEXITCODE -ne 0) { throw 'Curiosity container tests failed.' }
if ($GpuPerception) {
    docker run --rm --gpus all --network none --read-only --tmpfs /tmp -e PYTHONDONTWRITEBYTECODE=1 -v "${arcusProject}:/workspace:ro" -w /workspace --entrypoint python3 $arcusImage -m unittest tests.baby_arcus.test_object_perception_learning.GpuQualificationTests
    if ($LASTEXITCODE -ne 0) { throw 'GPU perception parity failed.' }
}
$arcusOutput = Join-Path $arcusProject 'runs/arcus_object_pixels_v1'
New-Item -ItemType Directory -Force -Path $arcusOutput | Out-Null
@{passed=$true;image=$arcusImage;platform=$arcusPlatform;gpu_perception_parity=[bool]$GpuPerception;scope='Unit and local HTTP persistence tests; optional GPU/CPU candidate numerical parity. This does not establish learned accuracy or GPU movement.'} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $arcusOutput 'container-qualification.json') -Encoding utf8
