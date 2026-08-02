#!/usr/bin/env bash
set -euo pipefail

# Reproducible COCO128 pilot for the Issue #54 MoE/MoT/MoA comparison.
# Environment variables may override defaults without editing this file.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DATA="${DATA:-$ROOT/ultralytics/cfg/datasets/coco128.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT/runs/mot_ablation_coco128_3seed}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-0}"
DEVICE="${DEVICE:-0}"
SEEDS="${SEEDS:-42 123 3407}"
MODELS="${MODELS:-v10 v10_mot v10_moa v10_moa_mot}"
BENCHMARK="${BENCHMARK:-1}"
WARMUP="${WARMUP:-100}"
REPS="${REPS:-1000}"

read -r -a seed_args <<<"$SEEDS"
read -r -a model_args <<<"$MODELS"

mkdir -p "$PROJECT_ROOT"
for seed in "${seed_args[@]}"; do
  echo "[$(date --iso-8601=seconds)] starting seed=$seed models=$MODELS"
  "$PYTHON" "$ROOT/scripts/compare_mot_ablation.py" \
    --train \
    --models "${model_args[@]}" \
    --data "$DATA" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --seed "$seed" \
    --project "$PROJECT_ROOT/seed_$seed" \
    --exist-ok
  echo "[$(date --iso-8601=seconds)] completed seed=$seed"
done

aggregate_args=(
  --root "$PROJECT_ROOT"
  --expected-seeds "${seed_args[@]}"
  --title "COCO128 MoE/MoT/MoA 3-seed pilot"
  --note "COCO128 is a smoke benchmark whose train and validation images overlap; do not present these metrics as full-COCO generalization results."
)

if [[ "$BENCHMARK" == "1" ]]; then
  profile_dir="$PROJECT_ROOT/profile"
  "$PYTHON" "$ROOT/scripts/compare_mot_ablation.py" \
    --benchmark \
    --actual-flops \
    --models "${model_args[@]}" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --warmup "$WARMUP" \
    --reps "$REPS" \
    --project "$profile_dir"
  aggregate_args+=(--latency-csv "$profile_dir/latency_${DEVICE}_${IMGSZ}.csv")
fi

"$PYTHON" "$ROOT/scripts/aggregate_mot_ablation_seeds.py" "${aggregate_args[@]}"
echo "[$(date --iso-8601=seconds)] all COCO128 runs and aggregation completed"
