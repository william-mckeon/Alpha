param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference = 'Stop'
$arcusBase = 'http://127.0.0.1:8890'
$arcusBefore = Invoke-RestMethod "$arcusBase/api/state"
if ($arcusBefore.shared.enabled) { throw 'Stop the existing shared session before this bounded smoke test.' }
if ($arcusBefore.shared.qualification.qualified -ne $true) { throw 'Native host has no qualified shared generation.' }
if ($arcusBefore.shared.depth_capacity -ne 0.25) { throw 'Native host must use the required 25% depth capacity.' }
$arcusBefore | ConvertTo-Json -Depth 60 | Set-Content (Join-Path $Root 'native-smoke-before.json') -Encoding utf8
$arcusMessage = @{ request_id = [guid]::NewGuid().ToString(); sender = 'you'; text = 'wait quietly' } | ConvertTo-Json
$arcusObserved = $null
$arcusStartedAt = [datetime]::UtcNow
$arcusStartupBudget = 60
$arcusConfigFile = Join-Path $Root 'config.json'
if (Test-Path -LiteralPath $arcusConfigFile) {
    $arcusRunConfig = Get-Content -Raw -LiteralPath $arcusConfigFile | ConvertFrom-Json
    if ($arcusRunConfig.startup_timeout_seconds) { $arcusStartupBudget = [math]::Min(300,[math]::Max(1,$arcusRunConfig.startup_timeout_seconds)) }
}
try {
    $arcusStart = @{ request_id = [guid]::NewGuid().ToString(); action = 'start' } | ConvertTo-Json
    Invoke-RestMethod "$arcusBase/api/shared/control" -Method Post -ContentType 'application/json' -Body $arcusStart | Out-Null
    Invoke-RestMethod "$arcusBase/api/messages" -Method Post -ContentType 'application/json' -Body $arcusMessage | Out-Null
    $arcusDeadline = [datetime]::UtcNow.AddSeconds([math]::Max(90,$arcusStartupBudget+30))
    do {
        Start-Sleep -Milliseconds 250
        $arcusObserved = Invoke-RestMethod "$arcusBase/api/state"
        if ($arcusObserved.shared.status -eq 'error') { throw $arcusObserved.shared.reason }
        if ($arcusObserved.shared.observations -ge 3) { break }
    } while ([datetime]::UtcNow -lt $arcusDeadline)
    if ($arcusObserved.shared.observations -lt 3) { throw 'Native model did not produce three observations before the deadline.' }
} finally {
    $arcusStop = @{ request_id = [guid]::NewGuid().ToString(); action = 'stop' } | ConvertTo-Json
    Invoke-RestMethod "$arcusBase/api/shared/control" -Method Post -ContentType 'application/json' -Body $arcusStop | Out-Null
    Start-Sleep -Seconds 1
}
$arcusAfter = Invoke-RestMethod "$arcusBase/api/state"
$arcusAfter | ConvertTo-Json -Depth 60 | Set-Content (Join-Path $Root 'native-smoke-after.json') -Encoding utf8
$arcusMessageId = ($arcusMessage | ConvertFrom-Json).request_id
$arcusSession = Get-ChildItem (Join-Path $Root 'sessions') -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$arcusProposals = @(Get-Content (Join-Path $arcusSession.FullName 'experiences.jsonl') | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object phase -eq 'proposed')
$arcusHeard = @($arcusProposals | Where-Object { $_.experience.hearing.request_id -contains $arcusMessageId }).Count -gt 0
$arcusDepthVerified = $arcusObserved.shared.last_prediction.depth.capacity -eq 0.25
$arcusContinuityVerified = $arcusObserved.shared.qualification.continuity -ne $true -or $null -ne $arcusObserved.shared.last_prediction.continuity
$arcusReport = @{
    passed = $arcusHeard -and $arcusDepthVerified -and $arcusContinuityVerified -and $arcusAfter.arcus.entity_id -eq $arcusBefore.arcus.entity_id -and $arcusAfter.environment.environment_id -eq $arcusBefore.environment.environment_id -and -not $arcusAfter.shared.enabled -and $arcusAfter.shared.status -eq 'stopped'
    depth_capacity_verified = $arcusDepthVerified
    continuity_service_connected = $arcusContinuityVerified
    elapsed_seconds = ([datetime]::UtcNow-$arcusStartedAt).TotalSeconds
    simulated_hearing_delivered = $arcusHeard
    observations = $arcusObserved.shared.observations
    generation = $arcusObserved.shared.generation
    entity_preserved = $arcusAfter.arcus.entity_id -eq $arcusBefore.arcus.entity_id
    room_preserved = $arcusAfter.environment.environment_id -eq $arcusBefore.environment.environment_id
    status = $arcusAfter.shared.status
    scope = 'Qualified native subprocess, simulated hearing delivery and bounded observation; movement is qualified separately in disposable playpens.'
}
$arcusReport | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $Root 'native-smoke.json') -Encoding utf8
$arcusReport | ConvertTo-Json -Depth 10
if (-not $arcusReport.passed) { throw 'Native smoke verification failed.' }

