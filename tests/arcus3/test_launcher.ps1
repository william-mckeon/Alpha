param([ValidateSet("probe","baseline","application","preflight","train","conversion","expanded-preflight","specialization","package","verify-package")][string]$Mode="probe")
# Integration fixture for actual launcher control flow; Docker is mocked, no GPU job.
$ErrorActionPreference='Stop'
$global:Arcus3FixtureCalls=[Collections.Generic.List[string]]::new()
$global:Arcus3FixtureStopped=$false
function global:Get-Content {
 param([string]$Path,[switch]$Raw)
 $text=Microsoft.PowerShell.Management\Get-Content -LiteralPath $Path -Raw
 if ($Path -eq 'configs/arcus3/project.json') {
  $fixture=$text | ConvertFrom-Json
  $fixture.authorization.training=($Mode -in @('preflight','train'))
  $fixture.authorization | Add-Member -NotePropertyName conversion -NotePropertyValue $true -Force
  $fixture | Add-Member -NotePropertyName conversion_scope -NotePropertyValue 'selective-experts-parity-v1' -Force
  $fixture.training_scope='dense-control-v1'
  $fixture.authorization | Add-Member -NotePropertyName expanded_preflight -NotePropertyValue $true -Force
  $fixture | Add-Member -NotePropertyName expanded_scope -NotePropertyValue 'qualification-v1' -Force
  $fixture.authorization | Add-Member -NotePropertyName specialization -NotePropertyValue $true -Force
  $fixture | Add-Member -NotePropertyName specialization_scope -NotePropertyValue 'matched-64-v1' -Force
  return ($fixture | ConvertTo-Json -Depth 12)
 }
 return $text
}
function global:docker {
 $global:LASTEXITCODE=0
 $global:Arcus3FixtureCalls.Add(($args -join ' '))
 switch ($args[0]) {
  'image' { return }
  'ps' { return }
  'run' { return 'fixture-id' }
  'kill' { $global:Arcus3FixtureStopped=$true; return 'fixture-id' }
  'logs' { return 'fixture deadline termination' }
  'inspect' {
   if ($args -contains '--format') { return (-not $global:Arcus3FixtureStopped).ToString().ToLower() }
   return '[{"State":{"Running":false,"ExitCode":137,"OOMKilled":false}}]'
  }
  default { throw "Unexpected fixture Docker operation: $args" }
 }
}
$root='runs/arcus3/donor-probe-deadline-fixture-'+(Get-Date -Format yyyyMMddHHmmss)
try {
 try {
  & (Join-Path $PSScriptRoot '../../scripts/start_arcus3.ps1') -StopAt ([DateTimeOffset]::Now.AddSeconds(2)) -Root $root -Mode $Mode -DataRoot artifacts/arcus3/data/dense-control-v2 -PreflightReport artifacts/arcus3/data/dense-control-v2/manifest.json -ConvertedPath runs/arcus3/conversion-phase5-001/converted
  throw 'Expected deadline termination failure receipt'
 } catch { if ($_.Exception.Message -notmatch 'Probe exited 137') { throw } }
 $kills=@($global:Arcus3FixtureCalls | Where-Object { $_ -like 'kill *' })
 if ($kills.Count -ne 1 -or $kills[0] -notmatch '^kill arcus3-donor-\d{8}-\d{6}$') { throw 'Cleanup must kill exactly its own container' }
 if (!(Test-Path "$root/container-state.json")) { throw 'Missing failure state receipt' }
 if (!(Test-Path 'runs/test2/alpha-tool-correction-60k-001/pause-training')) { throw 'Historical pause missing' }
 @{status='passed';fixture_only=$true;calls=$global:Arcus3FixtureCalls} | ConvertTo-Json -Depth 4 | Set-Content "$root/fixture-result.json"
 Write-Output "PASS: actual launcher deadline/own-container cleanup fixture; evidence $root"
} finally { Remove-Item Function:/docker; Remove-Item Function:/Get-Content }
