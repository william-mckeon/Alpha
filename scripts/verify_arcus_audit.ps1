$ErrorActionPreference='Stop'
$auditPorts=@{artifacts=8766;simulation=8765;inference=8767;training=8768;evaluator=8770;dashboard=8771;controller=8769}
$auditChecks=@()
foreach ($auditStack in @('baby-arcus-restored','baby-arcus-qualification')) {
 foreach ($auditRole in $auditPorts.Keys) {
  $auditPort=$auditPorts[$auditRole]
  $auditProbe='import os,json; from baby_arcus.transport import Client; r=Client("http://127.0.0.1:'+$auditPort+'",os.environ["BABY_ARCUS_TOKEN"]).request("GET","/health"); print(json.dumps(r.get("audit",{})))'
  $auditStatus=docker exec "$auditStack-$auditRole-1" python3 -c $auditProbe
  if ($LASTEXITCODE -ne 0) {throw "Health query failed: $auditStack-$auditRole"}
  $auditParsed=$auditStatus | ConvertFrom-Json
  if (-not $auditParsed.healthy) {throw "Audit unhealthy: $auditStack-$auditRole"}
  $auditChecks+=@{stack=$auditStack;service=$auditRole;audit=$auditParsed}
 }
}
$auditChecks | ConvertTo-Json -Depth 8 | Set-Content runs/arcus_desktop/audit-live-services.json
Write-Output "$($auditChecks.Count) services report healthy audit logging."

