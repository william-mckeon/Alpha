# Run once from an Administrator PowerShell. Copies evidence only; no ACL changes.
param([string[]]$DumpNames = @('092426-14812-01.dmp'), [switch]$IncludeMemory)
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open PowerShell as Administrator, then run this script again.'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $projectRoot 'runs/diagnostics/crash-analysis'
New-Item -ItemType Directory -Path $destination -Force | Out-Null
foreach ($name in $DumpNames) {
    if ($name -notmatch '^\d{6}-\d+-\d{2}\.dmp$') { throw 'Expected a minidump filename, not a path.' }
    $source = Join-Path 'C:\Windows\Minidump' $name
    $target = Join-Path $destination $name
    if (-not (Test-Path -LiteralPath $target)) {
        Copy-Item -LiteralPath $source -Destination $target
    }
    if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash) {
        throw "Copy verification failed: $name"
    }
    Write-Output "Verified crash dump copy: $target"
}
if ($IncludeMemory) {
    $source = 'C:\Windows\MEMORY.DMP'
    $stamp = (Get-Item -LiteralPath $source).LastWriteTime.ToString('yyyyMMdd-HHmmss')
    $target = Join-Path $destination ('MEMORY-' + $stamp + '.dmp')
    if (-not (Test-Path -LiteralPath $target)) { Copy-Item -LiteralPath $source -Destination $target }
    if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash) {
        throw 'Full memory dump copy verification failed.'
    }
    Write-Output "Verified full memory dump copy: $target"
}
