param([switch]$MigrateRoom)
$ErrorActionPreference = 'Stop'
$arcusProject = Split-Path -Parent $PSScriptRoot
$arcusEvidence = Join-Path $arcusProject 'runs/arcus_language'
$arcusStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$arcusExport = Invoke-RestMethod 'http://127.0.0.1:8890/api/session'
$arcusExportPath = Join-Path $arcusEvidence "session-before-language-$arcusStamp.json"
$arcusExport | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $arcusExportPath -Encoding utf8
if ($MigrateRoom) {
    & (Join-Path $arcusProject '.venv-desktop/Scripts/python.exe') (Join-Path $arcusProject 'scripts/migrate_arcus_room.py') $arcusExportPath (Join-Path $arcusProject 'runs/arcus_playroom/entity-state') --check
    if ($LASTEXITCODE -ne 0) { throw 'Room migration validation failed; host was not stopped.' }
}
$arcusProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like "*$arcusProject*" -and $_.CommandLine -match '-m baby_arcus\.desktop(?:\s|$)'
})
if (-not $arcusProcesses) { throw 'No matching Arcus desktop host found; refusing an ambiguous restart.' }
foreach ($arcusProcess in ($arcusProcesses | Sort-Object ParentProcessId -Descending)) {
    if (Get-Process -Id $arcusProcess.ProcessId -ErrorAction SilentlyContinue) { Stop-Process -Id $arcusProcess.ProcessId }
}
try {
    if ($MigrateRoom) {
        & (Join-Path $arcusProject '.venv-desktop/Scripts/python.exe') (Join-Path $arcusProject 'scripts/migrate_arcus_room.py') $arcusExportPath (Join-Path $arcusProject 'runs/arcus_playroom/entity-state')
        if ($LASTEXITCODE -ne 0) { throw 'Room migration failed.' }
    }
} finally {
    $arcusHost = Start-Process -FilePath (Join-Path $arcusProject '.venv-desktop/Scripts/pythonw.exe') -ArgumentList '-m','baby_arcus.desktop' -WorkingDirectory $arcusProject -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $arcusEvidence "desktop-$arcusStamp.stdout.log") -RedirectStandardError (Join-Path $arcusEvidence "desktop-$arcusStamp.stderr.log")
}
Write-Output "Arcus desktop restarted: $($arcusHost.Id)"
