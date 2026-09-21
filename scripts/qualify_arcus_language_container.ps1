param([ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$OutputName = 'arcus_language_container')
$ErrorActionPreference = 'Stop'
$arcusProject = Split-Path -Parent $PSScriptRoot
$arcusOutput = Join-Path $arcusProject "runs/$OutputName"
if (Test-Path -LiteralPath $arcusOutput) { throw 'Qualification directory already exists; preserve it and choose a new run.' }
New-Item -ItemType Directory -Path $arcusOutput | Out-Null
foreach ($arcusName in @('bootstrap.pt','qualification.json','dataset-manifest.json')) {
    Copy-Item -LiteralPath (Join-Path $arcusProject "runs/arcus_language/$arcusName") -Destination $arcusOutput
}
$env:ARCUS_LANGUAGE_TOKEN = [guid]::NewGuid().ToString('N')
$arcusContainer = 'arcus-language-live-check'
$arcusHeaders = @{Authorization="Bearer $env:ARCUS_LANGUAGE_TOKEN"}
try {
    docker run --rm -d --name $arcusContainer --gpus all -e ARCUS_LANGUAGE_TOKEN -p 127.0.0.1:8895:8895 --read-only --tmpfs /tmp -v "${arcusProject}/alpha dataset:/dataset:ro" -v "${arcusProject}/runs/arcus_approach_v2/postures.pt:/model/postures.pt:ro" -v "${arcusOutput}:/state" arcus-language:qualification | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Container launch failed' }
    $arcusReady = $false
    for ($arcusAttempt=0; $arcusAttempt -lt 60; $arcusAttempt++) {
        try { $arcusBefore=Invoke-RestMethod 'http://127.0.0.1:8895/health' -Headers $arcusHeaders -TimeoutSec 2; $arcusReady=$true; break } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $arcusReady) { docker logs $arcusContainer; throw 'Language service did not become ready' }
    $arcusBody=@{op='step';messages=@(@{request_id='linux-language-test';sender='you';text='Hello Arcus. This is a language service test.'})} | ConvertTo-Json -Depth 5
    $arcusAfter=Invoke-RestMethod 'http://127.0.0.1:8895/v1/language' -Method Post -Headers $arcusHeaders -ContentType application/json -Body $arcusBody -TimeoutSec 120
    $arcusPassed=$arcusAfter.message_receipts.'linux-language-test'.trained_tokens -gt 0
    @{gate_passed=$arcusPassed;before=$arcusBefore;after=$arcusAfter;platform='Ubuntu 22.04 / CUDA 12.8 / Python 3.10';scope='Authenticated HTTP service, readonly dataset mount, GPU update and checkpoint write'} | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $arcusOutput 'container-report.json') -Encoding utf8
    if (-not $arcusPassed) { throw 'Container language update did not pass' }
    Write-Output 'Container HTTP language learning passed.'
} finally {
    docker logs $arcusContainer 2>&1 | Set-Content -LiteralPath (Join-Path $arcusOutput 'container.log') -Encoding utf8
    docker stop $arcusContainer | Out-Null
    Remove-Item Env:ARCUS_LANGUAGE_TOKEN -ErrorAction SilentlyContinue
}
