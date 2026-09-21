$ErrorActionPreference = 'Stop'
$arcusProject = Split-Path -Parent $PSScriptRoot
$arcusImage = docker image inspect arcus-visual:qualification --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Existing pinned Ubuntu qualification image is required.' }
$arcusPlatform = docker run --rm --network none --read-only --entrypoint cat $arcusImage /etc/os-release
if ($LASTEXITCODE -ne 0 -or -not ($arcusPlatform -match 'VERSION_ID="22.04"')) { throw 'Ubuntu 22.04 required.' }
$arcusTests = docker run --rm --network none --read-only --tmpfs /tmp -v "${arcusProject}:/workspace:ro" -w /workspace --entrypoint python3 $arcusImage -m unittest tests.baby_arcus.test_rest_learning
if ($LASTEXITCODE -ne 0) { throw 'Rest container tests failed.' }
$arcusRaw = docker run --rm --network none --read-only --tmpfs /tmp -v "${arcusProject}:/workspace:ro" -w /workspace --entrypoint python3 $arcusImage scripts/evaluate_arcus_rest_endurance.py --output /tmp/endurance.json
if ($LASTEXITCODE -ne 0) { throw 'Rest container simulation failed.' }
$arcusResult = $arcusRaw | ConvertFrom-Json
$arcusConfig = Get-Content (Join-Path $arcusProject 'configs/baby_arcus/rest.json') -Raw | ConvertFrom-Json
$arcusOutput = Join-Path $arcusProject $arcusConfig.output
@{passed=$arcusResult.passed;image=$arcusImage;platform=$arcusPlatform;read_only_workspace=$true;network_disabled=$true;simulation=$arcusResult;scope='Rest-head and state/tool tests only; not full GPU learned-lying HTTP qualification.'} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $arcusOutput 'container-qualification.json') -Encoding utf8
if (-not $arcusResult.passed) { throw 'Rest container qualification failed.' }
Write-Output 'Ubuntu 22.04 rest tests and accelerated simulation passed.'
