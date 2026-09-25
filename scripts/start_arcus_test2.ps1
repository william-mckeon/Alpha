param([string]$Config = 'configs/baby_arcus/test2.json')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$settings = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($settings.preset -ne 'tiny') { throw 'Production Alpha runs in Docker. Use start_alpha_three_stage.ps1.' }
$runRoot = Join-Path $projectRoot $settings.root
if (!(Test-Path -LiteralPath (Join-Path $runRoot 'candidate.json'))) { throw 'Initialize Test 2 first.' }
$viewerPort = if ($settings.port) { [int]$settings.port } else { 8900 }
$learnerPort = if ($settings.learner_port) { [int]$settings.learner_port } else { 8901 }
foreach ($port in @($viewerPort,$learnerPort)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "Port $port is already in use." }
}
$env:ARCUS_TEST2_TOKEN = [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
$env:LANGSMITH_TRACING = 'false'
if ($settings.idle_learning) { $env:CUDA_LAUNCH_BLOCKING = '1' }
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
$worker = Start-Process -FilePath $python -ArgumentList @('-m','baby_arcus.services.shared_trainer','--config',('"'+$Config+'"'),'--port',$learnerPort) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runRoot 'learner.stdout.log') -RedirectStandardError (Join-Path $runRoot 'learner.stderr.log')
$viewer = Start-Process -FilePath $python -ArgumentList @('-m','baby_arcus.services.test2_playroom','--config',('"'+$Config+'"'),'--port',$viewerPort,'--learner-url',"http://127.0.0.1:$learnerPort") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runRoot 'viewer.stdout.log') -RedirectStandardError (Join-Path $runRoot 'viewer.stderr.log')
@{learner_pid=$worker.Id;viewer_pid=$viewer.Id;url="http://127.0.0.1:$viewerPort";config=$Config} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'services.json')
$deadline = [DateTime]::UtcNow.AddSeconds(120)
$ready = $false
while ([DateTime]::UtcNow -lt $deadline) {
    $worker.Refresh()
    $viewer.Refresh()
    if ($worker.HasExited -or $viewer.HasExited) { throw 'A Test 2 service exited; inspect its stderr log.' }
    try {
        $learnerStatus = Invoke-RestMethod "http://127.0.0.1:$learnerPort/ready" -Headers @{Authorization="Bearer $env:ARCUS_TEST2_TOKEN"} -TimeoutSec 2
        $viewerStatus = Invoke-RestMethod "http://127.0.0.1:$viewerPort/ready" -TimeoutSec 2
        if ($learnerStatus.ready -and $viewerStatus.ready) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if (!$ready) { throw 'Readiness timed out; inspect Test 2 service logs before retrying.' }
Write-Output "Test 2 ready at http://127.0.0.1:$viewerPort; sessions begin paused."
