param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][long]$MinimumInputTokens,
    [Parameter(Mandatory = $true)][long]$MinimumEvaluationUpdate,
    [int]$PollSeconds = 15,
    [int]$PauseTimeoutSeconds = 900,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace $Root))
$ownedRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'runs\arcus3'))
if (-not $runRoot.StartsWith($ownedRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Root must be an owned runs/arcus3 directory'
}
if ($MinimumInputTokens -le 0 -or $MinimumEvaluationUpdate -lt 0 -or $PollSeconds -lt 1 -or $PauseTimeoutSeconds -lt 30) {
    throw 'Invalid comparison watcher boundary or timeout'
}

$receiptPath = Join-Path $runRoot 'comparison-pause-receipt.json'
$errorPath = Join-Path $runRoot 'comparison-pause-error.json'
function Write-Json([string]$Path, $Value) {
    $temp = $Path + '.tmp'
    [System.IO.File]::WriteAllText($temp, ($Value | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}
function Read-Json([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}
function Latest-Evaluation-Update($State) {
    $updates = @($State.evaluation_completed | ForEach-Object { [long]$_.update })
    if ($updates.Count -eq 0) { return [long]-1 }
    return [long]($updates | Measure-Object -Maximum).Maximum
}
function Verify-Container-Stopped([string]$ActiveRun) {
    $runtimePath = Join-Path $ActiveRun 'runtime.json'
    if (-not (Test-Path -LiteralPath $runtimePath)) { throw 'Active worker runtime receipt is missing' }
    $runtime = Read-Json $runtimePath
    $raw = docker inspect $runtime.container 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect the owned training container after pause' }
    $inspection = ($raw | ConvertFrom-Json)[0]
    if ($inspection.State.Running) { return $false }
    return [ordered]@{ container = $runtime.container; running = $false; exit_code = [int]$inspection.State.ExitCode }
}

try {
    while ($true) {
        if (Test-Path (Join-Path $runRoot 'session-result.json')) {
            $result = Read-Json (Join-Path $runRoot 'session-result.json')
            if ($result.stopped) { throw "Coordinator stopped before comparison boundary: $($result.reason)" }
        }
        $sessionPath = Join-Path $runRoot 'session.json'
        if (Test-Path $sessionPath) {
            $session = Read-Json $sessionPath
            $reportPath = Join-Path $session.active_run 'report.json'
            if ($session.mode -eq 'adaptation' -and (Test-Path $reportPath)) {
                $report = Read-Json $reportPath
                $state = $report.state
                $latestEvaluationUpdate = Latest-Evaluation-Update $state
                if ([long]$state.input_tokens -ge $MinimumInputTokens -and
                    [long]$latestEvaluationUpdate -gt $MinimumEvaluationUpdate) {
                    $receipt = [ordered]@{
                        schema = 'arcus3-comparison-pause-v2'
                        status = 'boundary-observed'
                        observed_at = (Get-Date).ToUniversalTime().ToString('o')
                        requested_input_tokens = $MinimumInputTokens
                        observed_input_tokens = [long]$state.input_tokens
                        boundary_exact = ([long]$state.input_tokens -eq $MinimumInputTokens)
                        observed_updates = [long]$state.updates
                        latest_evaluation_update = [long]$latestEvaluationUpdate
                        active_run = $session.active_run
                        dry_run = [bool]$DryRun
                        verified_stopped = $false
                    }
                    Write-Json $receiptPath $receipt
                    if ($DryRun) { exit 0 }

                    $python = 'C:\Python314\python.exe'
                    if (-not (Test-Path -LiteralPath $python)) { throw 'Pinned Python executable is unavailable' }
                    $pauseScript = Join-Path $workspace 'scripts\pause_arcus3_training.py'
                    $pauseOutput = @(& $python $pauseScript --root $runRoot 2>&1)
                    if ($LASTEXITCODE -ne 0) { throw ('Pause request failed: ' + ($pauseOutput -join "`n")) }
                    $pause = ($pauseOutput -join "`n") | ConvertFrom-Json
                    if (-not $pause.pause_requested) { throw 'Pause helper did not confirm the pause request' }
                    $receipt.pause = $pause
                    $receipt.status = 'pause-requested'
                    Write-Json $receiptPath $receipt

                    $timeout = (Get-Date).ToUniversalTime().AddSeconds($PauseTimeoutSeconds)
                    while ((Get-Date).ToUniversalTime() -lt $timeout) {
                        $resultPath = Join-Path $runRoot 'session-result.json'
                        if (Test-Path -LiteralPath $resultPath) {
                            $result = Read-Json $resultPath
                            if (-not $result.stopped) { throw 'Coordinator result did not confirm a stop' }
                            if ([long]$result.input_tokens -lt [long]$receipt.observed_input_tokens) {
                                throw 'Terminal exposure moved backwards from the observed boundary'
                            }
                            $container = Verify-Container-Stopped $session.active_run
                            if ($container) {
                                $receipt.status = 'verified-stopped'
                                $receipt.verified_stopped = $true
                                $receipt.verified_at = (Get-Date).ToUniversalTime().ToString('o')
                                $receipt.terminal = $result
                                $receipt.container = $container
                                $receipt.actual_input_tokens = [long]$result.input_tokens
                                $receipt.actual_boundary_exact = ([long]$result.input_tokens -eq $MinimumInputTokens)
                                Write-Json $receiptPath $receipt
                                exit 0
                            }
                        }
                        Start-Sleep -Seconds ([Math]::Min($PollSeconds, 5))
                    }
                    throw 'Pause request timed out before coordinator and container stop were verified'
                }
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
} catch {
    Write-Json $errorPath ([ordered]@{
        schema = 'arcus3-comparison-pause-error-v1'
        failed_at = (Get-Date).ToUniversalTime().ToString('o')
        requested_input_tokens = $MinimumInputTokens
        minimum_evaluation_update = $MinimumEvaluationUpdate
        type = $_.Exception.GetType().Name
        detail = $_.Exception.Message
    })
    throw
}
