$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runName = "c2_tiered_vd100pct_s0_120e_b4_w0"
$runDir = Join-Path $repoRoot "A2OR\runs\$runName"

if (Test-Path -LiteralPath $runDir) {
    throw "Refusing to start because the output directory already exists: $runDir"
}

Set-Location -LiteralPath $repoRoot
$env:YOLO_CONFIG_DIR = Join-Path $repoRoot "A2OR\.ultralytics_config"
$env:TQDM_ASCII = "1"

conda run --no-capture-output -n yolo_master python A2OR/train_explicit.py `
    --coverage-tiered `
    --coverage-min-candidates 3 `
    --coverage-expand-target 20 `
    --coverage-long-side 32 `
    --name $runName `
    --model ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml `
    --data A2OR/visdrone_full.yaml `
    --epochs 120 `
    --patience 0 `
    --batch 4 `
    --nbs 64 `
    --workers 0 `
    --imgsz 800 `
    --device 0 `
    --seed 0 `
    --tal-topk 10 `
    --tal-alpha 0.5 `
    --tal-beta 6.0 `
    --save-period 1 `
    --assignment-stats `
    --no-pretrained

if ($LASTEXITCODE -ne 0) {
    throw "C2 tiered experiment exited with code $LASTEXITCODE"
}
