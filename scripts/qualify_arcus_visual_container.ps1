param([switch]$Navigation)
$ErrorActionPreference = 'Stop'
$arcusProject = Split-Path -Parent $PSScriptRoot
$arcusConfigPath = if ($Navigation) { 'configs/baby_arcus/visual_navigation.json' } else { 'configs/baby_arcus/visual.json' }
$arcusConfig = Get-Content (Join-Path $arcusProject $arcusConfigPath) -Raw | ConvertFrom-Json
$arcusOutput = Join-Path $arcusProject $arcusConfig.output
$arcusLive = Get-Content (Join-Path $arcusOutput 'live-qualification.json') -Raw | ConvertFrom-Json
$arcusObservations = @(Get-Content (Join-Path $arcusLive.session 'experiences.jsonl') | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object observation | Select-Object -First 2)
if ($arcusObservations.Count -ne 2) { throw 'Need two qualified visual observations.' }
$env:ARCUS_VISUAL_TOKEN = [guid]::NewGuid().ToString('N')
$arcusContainer = 'arcus-visual-check-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$arcusHeaders = @{Authorization="Bearer $env:ARCUS_VISUAL_TOKEN"}
$arcusParent = Join-Path $arcusProject $arcusConfig.parent
$arcusContainerConfig = Join-Path $arcusProject $(if ($Navigation) { 'configs/baby_arcus/visual_navigation.container.json' } else { 'configs/baby_arcus/visual.container.json' })
try {
    docker run --rm -d --name $arcusContainer --gpus all -e ARCUS_VISUAL_TOKEN -p 127.0.0.1:18896:8896 --read-only --tmpfs /tmp -v "${arcusParent}:/model:ro" -v "${arcusOutput}:/visual:ro" -v "${arcusContainerConfig}:/app/config.json:ro" arcus-visual:qualification | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Visual container launch failed.' }
    $arcusReady = $false
    for ($arcusAttempt=0; $arcusAttempt -lt 60; $arcusAttempt++) {
        try { $arcusStatus=Invoke-RestMethod 'http://127.0.0.1:18896/ready' -Headers $arcusHeaders -TimeoutSec 2; $arcusReady=$true; break } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $arcusReady) { throw 'Visual container did not become ready.' }
    $arcusUnauthorized = $false
    try { Invoke-RestMethod 'http://127.0.0.1:18896/ready' -TimeoutSec 2 | Out-Null } catch { $arcusUnauthorized = [int]$_.Exception.Response.StatusCode -eq 401 }
    $arcusResults = @()
    foreach ($arcusRow in $arcusObservations) {
        $arcusBody = $arcusRow.observation | ConvertTo-Json -Depth 40 -Compress
        $arcusResults += Invoke-RestMethod 'http://127.0.0.1:18896/v1/visual' -Method Post -Headers $arcusHeaders -ContentType 'application/json' -Body $arcusBody -TimeoutSec 30
    }
    $arcusPlatform = docker exec $arcusContainer cat /etc/os-release
    $arcusPassed = $arcusUnauthorized -and $arcusResults[0].action_name -eq 'open' -and $arcusResults[1].action_name -eq 'right'
    $arcusPassed = $arcusPassed -and [bool]$arcusStatus.learned_budget -eq [bool]$arcusConfig.learned_budget
    @{passed=$arcusPassed;authenticated=$arcusUnauthorized;status=$arcusStatus;results=$arcusResults;platform=$arcusPlatform;read_only_model_mounts=$true} | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $arcusOutput 'container-qualification.json') -Encoding utf8
    if (-not $arcusPassed) { throw 'Visual container action or authentication gate failed.' }
    Write-Output 'Visual Ubuntu HTTP inference qualification passed.'
} finally {
    docker logs $arcusContainer 2>&1 | Set-Content -LiteralPath (Join-Path $arcusOutput 'container.log') -Encoding utf8
    docker stop $arcusContainer | Out-Null
    Remove-Item Env:ARCUS_VISUAL_TOKEN -ErrorAction SilentlyContinue
}
