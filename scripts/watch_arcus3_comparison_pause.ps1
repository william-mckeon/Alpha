param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][long]$MinimumInputTokens,
    [Parameter(Mandatory = $true)][long]$MinimumEvaluationUpdate,
    [int]$PollSeconds = 15,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace $Root))
$ownedRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'runs\arcus3'))
if (-not $runRoot.StartsWith($ownedRoot + [System.IO.Path]::DirectorySeparatorChar)) {
    throw 'Root must be an owned runs/arcus3 directory'
}

$receiptPath = Join-Path $runRoot 'comparison-pause-receipt.json'
while ($true) {
    if (Test-Path (Join-Path $runRoot 'session-result.json')) {
        $result = Get-Content (Join-Path $runRoot 'session-result.json') -Raw | ConvertFrom-Json
        if ($result.stopped) { throw "Coordinator stopped before comparison boundary: $($result.reason)" }
    }
    $sessionPath = Join-Path $runRoot 'session.json'
    if (Test-Path $sessionPath) {
        $session = Get-Content $sessionPath -Raw | ConvertFrom-Json
        $reportPath = Join-Path $session.active_run 'report.json'
        if ($session.mode -eq 'adaptation' -and (Test-Path $reportPath)) {
            $report = Get-Content $reportPath -Raw | ConvertFrom-Json
            $state = $report.state
            $latestEvaluationUpdate = @($state.evaluation_completed | ForEach-Object { [long]$_.update } |
                Measure-Object -Maximum).Maximum
            if ([long]$state.input_tokens -ge $MinimumInputTokens -and
                [long]$latestEvaluationUpdate -gt $MinimumEvaluationUpdate) {
                $receipt = [ordered]@{
                    observed_at = (Get-Date).ToUniversalTime().ToString('o')
                    input_tokens = [long]$state.input_tokens
                    updates = [long]$state.updates
                    latest_evaluation_update = [long]$latestEvaluationUpdate
                    active_run = $session.active_run
                    dry_run = [bool]$DryRun
                }
                if (-not $DryRun) {
                    $dependencyRoot = (Resolve-Path (Join-Path $workspace '.python-deps')).Path
                    $env:PYTHONPATH = $dependencyRoot
                    $python = 'C:\Python314\python.exe'
                    $pauseScript = Join-Path $workspace 'scripts\pause_arcus3_training.py'
                    $pause = & $python $pauseScript --root $runRoot | ConvertFrom-Json
                    $receipt.pause = $pause
                }
                $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
                exit 0
            }
        }
    }
    Start-Sleep -Seconds $PollSeconds
}
