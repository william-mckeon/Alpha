param([string]$Output)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Output) { $Output = Join-Path $projectRoot ('runs/diagnostics/host-' + [Guid]::NewGuid().ToString('N') + '.json') }
if (Test-Path -LiteralPath $Output) { throw 'Use a new diagnostics output file.' }
$facts = [ordered]@{
    machine = @(Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory,NumberOfLogicalProcessors)
    bios = @(Get-CimInstance Win32_BIOS | Select-Object SMBIOSBIOSVersion,ReleaseDate)
    cpu = @(Get-CimInstance Win32_Processor | Select-Object Name)
    os = @(Get-CimInstance Win32_OperatingSystem | Select-Object Version,BuildNumber)
    os_update_revision = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').UBR
    graphics = @(Get-CimInstance Win32_VideoController | Sort-Object Name | Select-Object Name,DriverVersion,DriverDate)
}
$wslVersion = (& wsl.exe --version 2>$null) -join "`n"
$facts.wsl_version = $wslVersion.Replace([string][char]0,'').Trim()
$dockerInfo = & docker info --format '{{json .}}' 2>$null
if ($LASTEXITCODE -eq 0) {
    $engine = $dockerInfo | ConvertFrom-Json
    $facts.docker = [ordered]@{version=$engine.ServerVersion; kernel=$engine.KernelVersion; memory_bytes=$engine.MemTotal; cpus=$engine.NCPU}
} else { $facts.docker = 'unavailable' }
$bytes = [Text.Encoding]::UTF8.GetBytes(($facts | ConvertTo-Json -Depth 6 -Compress))
$sha = [Security.Cryptography.SHA256]::Create()
try { $fingerprint = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLowerInvariant() }
finally { $sha.Dispose() }
$events = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; StartTime=(Get-Date).AddDays(-7)} -MaxEvents 30 -ErrorAction SilentlyContinue | Select-Object TimeCreated,Id,Message)
$memoryTests = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-MemoryDiagnostics-Results'} -MaxEvents 5 -ErrorAction SilentlyContinue | Select-Object TimeCreated,Id,Message)
$crashes = @(Get-WinEvent -FilterHashtable @{LogName='System'; Id=41,1001; StartTime=(Get-Date).AddDays(-7)} -MaxEvents 12 -ErrorAction SilentlyContinue | Select-Object TimeCreated,Id,ProviderName,Message)
$report = [ordered]@{schema='alpha-host-diagnostics-v2'; created_at=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); host_fingerprint=$fingerprint; facts=$facts; whea=$events; memory_diagnostics=$memoryTests; recent_crashes=$crashes; training_authorized=$false; host_qualified=$false}
New-Item -ItemType Directory -Path (Split-Path -Parent ([IO.Path]::GetFullPath($Output))) -Force | Out-Null
$report | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $Output -Encoding utf8
Write-Output "Read-only diagnostics saved: $Output"
Write-Output "Host fingerprint: $fingerprint. This report does not clear GPU execution."
