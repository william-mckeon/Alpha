param([switch]$FullSize)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$testRoot = Join-Path $projectRoot 'runs/test2'
$datasetRoot = Join-Path $projectRoot 'alpha dataset'
$image = 'arcus-test2:qualification'
docker run --rm --network none --mount "type=bind,source=$projectRoot,target=/app,readonly" $image -m unittest tests.baby_arcus.test_test2_integration -v
if ($LASTEXITCODE -ne 0) { throw 'Linux contracts failed.' }
if ($FullSize) {
    $mounts = @('--mount',"type=bind,source=$projectRoot,target=/app,readonly",'--mount',"type=bind,source=$testRoot,target=/app/runs/test2",'--mount',"type=bind,source=$datasetRoot,target=/dataset,readonly")
    $argsCommon = @('run','--rm','--gpus','all','--network','none','-e','TIKTOKEN_CACHE_DIR=/app/runs/arcus_shared_v3/tokenizer-cache') + $mounts + @($image)
    & docker @argsCommon scripts/train_arcus_test2.py --config configs/baby_arcus/test2.container.json --updates 11
    if ($LASTEXITCODE -ne 0) { throw 'Linux full-size learning failed.' }
    & docker @argsCommon scripts/qualify_arcus_test2.py --config configs/baby_arcus/test2.container.json
    if ($LASTEXITCODE -ne 0) { throw 'Linux live qualification failed.' }
}
