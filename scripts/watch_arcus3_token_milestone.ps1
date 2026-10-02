param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$CheckpointRoot,
    [Parameter(Mandatory = $true)][long]$InputTokens,
    [int]$PollSeconds = 15,
    [int]$PauseTimeoutSeconds = 900
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ownedRuns = [System.IO.Path]::GetFullPath((Join-Path $workspace 'runs\arcus3'))
$runPath = if ([System.IO.Path]::IsPathRooted($Root)) { $Root } else { Join-Path $workspace $Root }
$runRoot = [System.IO.Path]::GetFullPath($runPath)
$storage = Get-Content -LiteralPath (Join-Path $workspace 'configs\arcus3\phase8_storage.json') -Raw | ConvertFrom-Json
$ownedCheckpoints = [System.IO.Path]::GetFullPath((Join-Path $storage.external_root 'alpha3.2.2\checkpoints'))
$checkpointRoot = [System.IO.Path]::GetFullPath($CheckpointRoot)
if (-not $runRoot.StartsWith($ownedRuns + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
    -not $checkpointRoot.Equals($ownedCheckpoints, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Milestone watcher requires the owned Alpha 3.2.2 coordinator and checkpoint root'
}
if ($InputTokens -lt 1000000 -or $InputTokens -ge 7000000 -or $InputTokens % 1000000 -ne 0 -or
    $PollSeconds -lt 1 -or $PauseTimeoutSeconds -lt 30) {
    throw 'Only the 1M through 6M interim boundaries are supported'
}

function Read-Json([string]$Path) { Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
function Write-Json([string]$Path, $Value) {
    $temp = $Path + '.tmp'
    [System.IO.File]::WriteAllText($temp, ($Value | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}
$receiptPath = Join-Path $runRoot "interim-$InputTokens-pause.json"
$errorPath = Join-Path $runRoot "interim-$InputTokens-error.json"
if (Test-Path -LiteralPath $receiptPath) { throw 'Interim milestone receipt already exists' }

try {
    while ($true) {
        if (Test-Path -LiteralPath (Join-Path $runRoot 'session-result.json')) {
            $terminal = Read-Json (Join-Path $runRoot 'session-result.json')
            throw "Coordinator stopped before interim boundary: $($terminal.reason)"
        }
        $sessionPath = Join-Path $runRoot 'session.json'
        $pointerPath = Join-Path $checkpointRoot 'latest.json'
        if ((Test-Path -LiteralPath $sessionPath) -and (Test-Path -LiteralPath $pointerPath)) {
            $session = Read-Json $sessionPath
            $pointer = Read-Json $pointerPath
            $checkpoint = [System.IO.Path]::GetFullPath((Join-Path $checkpointRoot $pointer.generation))
            if (-not $checkpoint.StartsWith($checkpointRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'Checkpoint pointer escaped its owned root'
            }
            $manifestPath = Join-Path $checkpoint 'manifest.json'
            if (Test-Path -LiteralPath $manifestPath) {
                $manifest = Read-Json $manifestPath
                if ($manifest.model_label -ne 'alpha3.2.2' -or $manifest.lineage_id -ne 'alpha3.2.2-wsd-001') {
                    throw 'Checkpoint pointer belongs to a different lineage'
                }
                if ($session.mode -eq 'adaptation' -and [long]$manifest.learning_rate_input_tokens -ge $InputTokens) {
                    $actualHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
                    if ($actualHash -ne $pointer.manifest_sha256) { throw 'Checkpoint pointer hash mismatch' }
                    $receipt = [ordered]@{
                        schema = 'arcus3-alpha322-interim-pause-v1'
                        status = 'boundary-observed'
                        requested_input_tokens = $InputTokens
                        observed_input_tokens = [long]$manifest.learning_rate_input_tokens
                        observed_updates = [long]$manifest.updates
                        observed_checkpoint_manifest_sha256 = $actualHash
                        observed_at = (Get-Date).ToUniversalTime().ToString('o')
                        coordinator = $runRoot
                        active_run = $session.active_run
                        verified_stopped = $false
                    }
                    Write-Json $receiptPath $receipt
                    $python = 'C:\Python314\python.exe'
                    if (-not (Test-Path -LiteralPath $python)) { throw 'Pinned Python executable is unavailable' }
                    $pauseOutput = @(& $python (Join-Path $workspace 'scripts\pause_arcus3_training.py') --root $runRoot 2>&1)
                    if ($LASTEXITCODE -ne 0) { throw ('Pause request failed: ' + ($pauseOutput -join "`n")) }
                    $pause = ($pauseOutput -join "`n") | ConvertFrom-Json
                    if (-not $pause.pause_requested) { throw 'Pause helper did not accept request' }
                    $receipt.status = 'pause-requested'
                    Write-Json $receiptPath $receipt
                    $timeout = (Get-Date).ToUniversalTime().AddSeconds($PauseTimeoutSeconds)
                    while ((Get-Date).ToUniversalTime() -lt $timeout) {
                        $resultPath = Join-Path $runRoot 'session-result.json'
                        if (Test-Path -LiteralPath $resultPath) {
                            $terminal = Read-Json $resultPath
                            if (-not $terminal.stopped -or [long]$terminal.input_tokens -lt $InputTokens) {
                                throw 'Coordinator did not stop after the requested boundary'
                            }
                            $runtime = Read-Json (Join-Path $session.active_run 'runtime.json')
                            $inspection = @(docker inspect $runtime.container 2>$null | ConvertFrom-Json)[0]
                            if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect owned training container' }
                            if (-not $inspection.State.Running) {
                                if ([int]$inspection.State.ExitCode -ne 0) { throw 'Training container exited with failure' }
                                $committed = Read-Json (Join-Path $runRoot 'controller-state.json')
                                $finalCheckpoint = [System.IO.Path]::GetFullPath($committed.checkpoint)
                                if (-not $finalCheckpoint.StartsWith($checkpointRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
                                    throw 'Coordinator selected a checkpoint outside owned storage'
                                }
                                if ([long]$committed.state.input_tokens -ne [long]$terminal.input_tokens -or
                                    [long]$committed.state.input_tokens -lt $InputTokens) {
                                    throw 'Terminal exposure and committed checkpoint disagree'
                                }
                                $finalManifest = Join-Path $finalCheckpoint 'manifest.json'
                                $finalHash = (Get-FileHash -LiteralPath $finalManifest -Algorithm SHA256).Hash.ToLowerInvariant()
                                $files = (Read-Json $finalManifest).files
                                foreach ($name in @('delta.safetensors','state.pt')) {
                                    if ((Get-FileHash -LiteralPath (Join-Path $finalCheckpoint $name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $files.$name) {
                                        throw "Checkpoint payload hash mismatch: $name"
                                    }
                                }
                                $receipt.status = 'verified-stopped'
                                $receipt.verified_stopped = $true
                                $receipt.verified_at = (Get-Date).ToUniversalTime().ToString('o')
                                $receipt.actual_input_tokens = [long]$terminal.input_tokens
                                $receipt.actual_target_tokens = [long]$committed.state.target_tokens
                                $receipt.actual_updates = [long]$committed.state.updates
                                $receipt.checkpoint = $finalCheckpoint
                                $receipt.checkpoint_manifest_sha256 = $finalHash
                                $receipt.payload_hashes_verified = $true
                                $receipt.container = $runtime.container
                                Write-Json $receiptPath $receipt
                                exit 0
                            }
                        }
                        Start-Sleep -Seconds ([Math]::Min($PollSeconds, 5))
                    }
                    throw 'Timed out waiting for verified checkpoint and container exit'
                }
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
} catch {
    Write-Json $errorPath ([ordered]@{
        schema = 'arcus3-alpha322-interim-pause-error-v1'
        requested_input_tokens = $InputTokens
        failed_at = (Get-Date).ToUniversalTime().ToString('o')
        type = $_.Exception.GetType().Name
        detail = $_.Exception.Message
    })
    throw
}
