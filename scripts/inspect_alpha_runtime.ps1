param([switch]$RequireQualified)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$compose = 'docker/baby-arcus/compose.alpha-three-stage.yaml'
# Resolved credentials stay in memory; never print the full Compose configuration.
$raw = & docker compose --env-file .env -f $compose config --format json
if ($LASTEXITCODE -ne 0) { throw 'Compose configuration unavailable.' }
$resolved = ($raw -join "`n") | ConvertFrom-Json
$learner = $resolved.services.learner
$actualImage = & docker image inspect $learner.image --format '{{.Id}}'
if ($LASTEXITCODE -ne 0 -or $actualImage -ne $learner.environment.ALPHA_RUNTIME_IMAGE) { throw 'Runtime image differs from qualification scope. Rebuild and qualify the exact image.' }
$diagnostic = Join-Path (Get-Location) ('runs/diagnostics/preflight-' + [Guid]::NewGuid().ToString('N') + '.json')
& "$PSScriptRoot/collect_alpha_host_diagnostics.ps1" -Output $diagnostic | Out-Null
$hostReport = Get-Content -LiteralPath $diagnostic -Raw | ConvertFrom-Json
if ($hostReport.host_fingerprint -ne $learner.environment.ALPHA_HOST_FINGERPRINT) { throw 'Host configuration changed since qualification.' }
$mount = @($learner.volumes | Where-Object target -eq '/qualification')
if ($mount.Count -ne 1 -or -not $mount[0].read_only) { throw 'Read-only qualification mount required.' }
$files = @('host.json'); if ($RequireQualified) { $files += 'gpu.json' }
if ($learner.environment.ALPHA_GPU_MODE -eq 'controlled-docker') {
    if ($learner.mem_limit -le 0 -or $learner.mem_limit -gt 10737418240 -or
        [double]$learner.cpus -le 0 -or [double]$learner.cpus -gt 2 -or
        $learner.pids_limit -le 0 -or $learner.pids_limit -gt 256 -or
        $learner.logging.driver -ne 'json-file') { throw 'Controlled Docker resource/logging settings required.' }
    $files = @()
    Write-Output 'Controlled Docker operating mode selected; host hardware stability is unverified. Data approval remains separate.'
}
foreach ($file in $files) {
    $report = Get-Content -LiteralPath (Join-Path $mount[0].source $file) -Raw | ConvertFrom-Json
    $age = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $report.created_at
    $schema = if ($file -eq 'host.json') { 'alpha-host-qualification-v1' } else { 'alpha-gpu-qualification-v1' }
    if ($report.schema -ne $schema -or $report.complete -ne $true -or $age -lt 0 -or $age -gt 86400 -or
        $report.scope.image_id -ne $actualImage -or $report.scope.host_fingerprint -ne $hostReport.host_fingerprint -or
        -not $report.checks -or @($report.checks.PSObject.Properties | Where-Object Value -ne $true).Count) {
        throw "Missing, stale or failed qualification: $file"
    }
}
$parent = @($learner.volumes | Where-Object target -eq '/app/runs/test2/alpha-idle-fresh-release-20260923')
if ($parent.Count -ne 1 -or -not $parent[0].read_only) { throw 'Read-only release parent mount required.' }
$candidate = Get-Content -LiteralPath (Join-Path $parent[0].source 'candidate.json') -Raw | ConvertFrom-Json
$expected = '9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520'
if ($candidate.generation -notmatch '^[a-f0-9]{32}$' -or $candidate.sha256 -ne $expected -or $candidate.updates -ne 37000) { throw 'Wrong release baseline.' }
if ((Get-FileHash -LiteralPath (Join-Path $parent[0].source ($candidate.generation+'.pt'))).Hash -ne $expected) { throw 'Release hash mismatch.' }
Write-Output 'Image, current host scope, selected operating mode and immutable release parent verified.'
