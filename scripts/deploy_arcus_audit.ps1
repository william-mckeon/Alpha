$ErrorActionPreference = 'Stop'
$loggingStacks = @('baby-arcus-restored','baby-arcus-qualification')
$loggingRoles = @('artifacts','simulation','inference','training','evaluator','dashboard','controller')
$loggingProbe = 'import os,json; from baby_arcus.transport import Client; s=Client("http://127.0.0.1:8769",os.environ["BABY_ARCUS_TOKEN"]).request("GET","/v1/status"); print(json.dumps({"live":s["live"],"lease":s["resource"]["lease"],"run_id":s["run"]["run_id"],"status":s["run"]["status"],"checkpoint":s["run"]["checkpoint_id"],"cycles":s["run"]["cycles"]}))'
$loggingBefore = @{}
foreach ($loggingStack in $loggingStacks) {
    $loggingResult = docker exec "$loggingStack-controller-1" python3 -c $loggingProbe
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Arcus controller' }
    $loggingState = $loggingResult | ConvertFrom-Json
    if ($null -ne $loggingState.live -or $null -ne $loggingState.lease) { throw 'Arcus is active; deployment stopped' }
    $loggingBefore[$loggingStack] = $loggingState
}
$loggingBefore | ConvertTo-Json -Depth 8 | Set-Content runs/arcus_desktop/audit-stacks-before.json
foreach ($loggingStack in $loggingStacks) {
    foreach ($loggingRole in $loggingRoles) {
        $loggingContainer = "$loggingStack-$loggingRole-1"
        foreach ($loggingFile in @('audit.py','transport.py','cli.py','services/worker.py')) {
            docker cp "baby_arcus/$loggingFile" "${loggingContainer}:/app/baby_arcus/$loggingFile"
            if ($LASTEXITCODE -ne 0) { throw "Copy failed: $loggingContainer" }
        }
        docker restart $loggingContainer
        if ($LASTEXITCODE -ne 0) { throw "Restart failed: $loggingContainer" }
    }
}
$loggingAfter = @{}
foreach ($loggingStack in $loggingStacks) {
    $loggingResult = docker exec "$loggingStack-controller-1" python3 -c $loggingProbe
    if ($LASTEXITCODE -ne 0) { throw 'Cannot verify controller after restart' }
    $loggingState = $loggingResult | ConvertFrom-Json
    $loggingPrevious = $loggingBefore[$loggingStack]
    if ($loggingState.checkpoint -ne $loggingPrevious.checkpoint -or $loggingState.cycles -ne $loggingPrevious.cycles -or $loggingState.run_id -ne $loggingPrevious.run_id) { throw 'Experiment identity changed' }
    $loggingAfter[$loggingStack] = $loggingState
}
$loggingAfter | ConvertTo-Json -Depth 8 | Set-Content runs/arcus_desktop/audit-stacks-after.json
Write-Output 'Both Arcus stacks updated; checkpoint IDs and update counts preserved.'

