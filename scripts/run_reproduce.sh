#!/usr/bin/env bash
set -euo pipefail

# 命令
# 只跑 EsMoE-N
# bash scripts/run_reproduce.sh --dataset visdrone --model esmoe_n

# 只跑 v0.1-N
# bash scripts/run_reproduce.sh --dataset visdrone --model v0_1_n

# 两个模型都跑
# bash scripts/run_reproduce.sh --dataset visdrone --model all

# Default hyperparameters. Override them with CLI flags below.
DATASETS_DIR="/home/czdata3/腾讯犀牛鸟/data"
DATASET="visdrone"          # visdrone | sku110k
MODEL="esmoe_n"             # esmoe_n | v0_1_n | all
EPOCHS=100
IMGSZ=640
BATCH=16
DEVICE=0
WORKERS=8
SEED=42
RUN_SUFFIX=""
WANDB_MODE="online"         # online | offline | disabled
LOG_FILE=""

DRY_RUN=0
CHECK_BUILD=0
SUMMARY_ONLY=0
RESUME_EXISTING=0
SKIP_EXISTING=0
NO_LOG=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_reproduce.sh [options] [-- extra python args]

Common examples:
  # 1-epoch smoke test on VisDrone EsMoE-N
  bash scripts/run_reproduce.sh --smoke --dataset visdrone --model esmoe_n

  # Full VisDrone training for one model
  bash scripts/run_reproduce.sh --dataset visdrone --model esmoe_n --epochs 100 --batch 16
  bash scripts/run_reproduce.sh --dataset visdrone --model v0_1_n --epochs 100 --batch 16

  # Run both models sequentially
  bash scripts/run_reproduce.sh --dataset visdrone --model all --epochs 100

  # SKU-110K after it is downloaded
  bash scripts/run_reproduce.sh --dataset sku110k --model esmoe_n --epochs 100

Options:
  --dataset VALUE        visdrone | sku110k
  --model VALUE          esmoe_n | v0_1_n | all
  --datasets-dir PATH    Dataset root containing VisDrone/ and SKU-110K/
  --epochs N
  --imgsz N
  --batch N
  --device VALUE
  --workers N
  --seed N
  --suffix VALUE         Run suffix. Default: e${EPOCHS}; smoke uses smoke.
  --wandb-mode VALUE     online | offline | disabled
  --log-file PATH        Save complete stdout/stderr to this log file
  --no-log               Do not create a raw terminal log
  --smoke                Use epochs=1, batch=8, wandb offline, suffix=smoke
  --dry-run              Print resolved Python command only
  --check-build          Instantiate model configs without training
  --summary-only         Refresh summary.csv without training
  --resume-existing      Resume from weights/last.pt when available
  --skip-existing        Skip runs with existing results.csv
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --datasets-dir)
      DATASETS_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --imgsz)
      IMGSZ="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --suffix)
      RUN_SUFFIX="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --no-log)
      NO_LOG=1
      shift
      ;;
    --smoke)
      EPOCHS=1
      BATCH=8
      RUN_SUFFIX="smoke"
      WANDB_MODE="offline"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --check-build)
      CHECK_BUILD=1
      shift
      ;;
    --summary-only)
      SUMMARY_ONLY=1
      shift
      ;;
    --resume-existing)
      RESUME_EXISTING=1
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_SUFFIX" ]]; then
  RUN_SUFFIX="e${EPOCHS}"
fi

case "$DATASET" in
  visdrone)
    REPRO_SCRIPT="scripts/reproduce/reproduce_visdrone.py"
    ;;
  sku110k|sku-110k|SKU-110K)
    DATASET="sku110k"
    REPRO_SCRIPT="scripts/reproduce/reproduce_sku110k.py"
    ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    exit 2
    ;;
esac

case "$MODEL" in
  esmoe_n|v0_1_n|all)
    ;;
  *)
    echo "Unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

CMD=(
  python "$REPRO_SCRIPT"
  --datasets-dir "$DATASETS_DIR"
  --models "$MODEL"
  --epochs "$EPOCHS"
  --imgsz "$IMGSZ"
  --batch "$BATCH"
  --device "$DEVICE"
  --workers "$WORKERS"
  --seed "$SEED"
  --run-suffix "$RUN_SUFFIX"
  --wandb-mode "$WANDB_MODE"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry-run)
fi
if [[ "$CHECK_BUILD" -eq 1 ]]; then
  CMD+=(--check-build)
fi
if [[ "$SUMMARY_ONLY" -eq 1 ]]; then
  CMD+=(--summary-only)
fi
if [[ "$RESUME_EXISTING" -eq 1 ]]; then
  CMD+=(--resume-existing)
fi
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  CMD+=(--skip-existing)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "$NO_LOG" -eq 0 ]]; then
  if [[ -z "$LOG_FILE" ]]; then
    LOG_DIR="runs/issue49_${DATASET}/logs"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    SAFE_MODEL="${MODEL//[^[:alnum:]_.-]/_}"
    SAFE_SUFFIX="${RUN_SUFFIX//[^[:alnum:]_.-]/_}"
    LOG_FILE="${LOG_DIR}/${DATASET}_${SAFE_MODEL}_${SAFE_SUFFIX}_${TIMESTAMP}.log"
  fi
  mkdir -p "$(dirname "$LOG_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "[run_reproduce] dataset=$DATASET model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH device=$DEVICE workers=$WORKERS seed=$SEED suffix=$RUN_SUFFIX wandb=$WANDB_MODE"
if [[ "$NO_LOG" -eq 0 ]]; then
  echo "[run_reproduce] log=$LOG_FILE"
else
  echo "[run_reproduce] log=disabled"
fi
printf '[run_reproduce] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
