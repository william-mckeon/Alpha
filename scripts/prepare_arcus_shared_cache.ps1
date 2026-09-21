param([string]$Config = 'configs/baby_arcus/shared.json')
$ErrorActionPreference = 'Stop'
$arcusProject = Split-Path -Parent $PSScriptRoot
$arcusConfig = Get-Content -LiteralPath (Join-Path $arcusProject $Config) -Raw | ConvertFrom-Json
if ($arcusConfig.encoding -ne 'o200k_base') { throw 'This cache preparation is pinned to o200k_base.' }
$arcusName = 'fb374d419588a4632f3f557e76b4b70aebbca790'
$arcusHash = '446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d'
$arcusDestination = Join-Path (Join-Path $arcusProject $arcusConfig.root) 'tokenizer-cache'
New-Item -ItemType Directory -Force -Path $arcusDestination | Out-Null
$arcusSource = Join-Path (Join-Path $env:TEMP 'data-gym-cache') $arcusName
if (-not (Test-Path -LiteralPath $arcusSource)) { throw 'Run the Windows tokenizer bootstrap first to populate its verified cache.' }
if ((Get-FileHash -LiteralPath $arcusSource -Algorithm SHA256).Hash.ToLower() -ne $arcusHash) { throw 'Tokenizer cache checksum mismatch.' }
Copy-Item -LiteralPath $arcusSource -Destination (Join-Path $arcusDestination $arcusName)
Write-Output "Verified offline tokenizer cache: $arcusDestination"
