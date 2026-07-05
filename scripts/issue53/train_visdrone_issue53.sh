#!/usr/bin/env bash
set -euo pipefail

ROOT="${YOLO_ISSUE53_ROOT:-$HOME/yolo-master-issue53}"
PROJECT="$ROOT/YOLO-Master"
VENV="$ROOT/.venv"
DATA="$ROOT/datasets/VisDrone/visdrone.yaml"
PROJECT_OUT="${PROJECT_OUT:-$ROOT/runs/issue53_visdrone}"

EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-0}"
MODELS="${MODELS:-v10 v10_moa}"
CACHE="${CACHE:-0}"
AMP="${AMP:-1}"

EXTRA_ARGS=()
if [[ "$CACHE" == "1" || "$CACHE" == "true" || "$CACHE" == "True" ]]; then
  EXTRA_ARGS+=(--cache)
fi
case "$AMP" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    EXTRA_ARGS+=(--no-amp)
    ;;
  *)
    EXTRA_ARGS+=(--amp)
    ;;
esac

. "$VENV/bin/activate"
cd "$PROJECT"

python scripts/compare_moa_ablation.py \
  --train \
  --models $MODELS \
  --data "$DATA" \
  --project "$PROJECT_OUT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --plots \
  --exist-ok \
  "${EXTRA_ARGS[@]}"
