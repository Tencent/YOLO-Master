#!/usr/bin/env bash
set -euo pipefail

ROOT="${YOLO_ISSUE53_ROOT:-$HOME/yolo-master-issue53}"
PROJECT="$ROOT/YOLO-Master"
BATCH="${BATCH:-30}"
WORKERS="${WORKERS:-8}"
LOG="$ROOT/logs/issue53_train_v10_then_moa_3090_noamp.log"
PID_FILE="$ROOT/logs/issue53_train_v10_then_moa_3090_noamp.pid"
PROJECT_OUT="${PROJECT_OUT:-$ROOT/runs/issue53_visdrone_3090_noamp}"

mkdir -p "$ROOT/logs"
cd "$PROJECT"

nohup env \
  MODELS="v10 v10_moa" \
  EPOCHS=50 \
  BATCH="$BATCH" \
  IMGSZ=640 \
  WORKERS="$WORKERS" \
  DEVICE=0 \
  CACHE=1 \
  AMP=0 \
  PROJECT_OUT="$PROJECT_OUT" \
  scripts/issue53/train_visdrone_issue53.sh > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "started PID $(cat "$PID_FILE")"
echo "batch: $BATCH"
echo "amp: 0"
echo "project: $PROJECT_OUT"
echo "log: $LOG"
