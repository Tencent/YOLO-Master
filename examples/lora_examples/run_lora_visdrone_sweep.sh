#!/usr/bin/env bash
# ==============================================================================
# YOLO-Master-EsMoE-N LoRA Rank Sweep — VisDrone (Dense Aerial Detection)
#
# Usage:
#   bash examples/lora_examples/run_lora_visdrone_sweep.sh
#
# Prerequisites:
#   - yolo command available (pip install ultralytics or editable install)
#   - VisDrone dataset downloaded (auto-downloads via VisDrone.yaml on first run)
#   - NVIDIA A40 (48 GB) or equivalent GPU
#
# Sweep matrix: r ∈ {4, 8, 16, 32} × alpha = 2×r
# Total experiments: 4 (serial execution for accurate VRAM measurement)
# ==============================================================================
set -euo pipefail

CFG="examples/lora_examples/yolo_master_visdrone_lora.yaml"
PROJECT="runs/lora_examples"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="logs/visdrone_sweep"

# Rank sweep: r:alpha:run_name
EXPERIMENTS=(
  "4:8:visdrone_r4"
  "8:16:visdrone_r8"
  "16:32:visdrone_r16"
  "32:64:visdrone_r32"
)

mkdir -p "${LOG_DIR}"

echo "==========================================================================="
echo "  VisDrone LoRA Rank Sweep — YOLO-Master-EsMoE-N"
echo "  GPU: ${GPU_ID}  |  Config: ${CFG}  |  Project: ${PROJECT}"
echo "  Ranks: 4, 8, 16, 32 (alpha = 2×r)"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================================================="

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0
FAILED=0

for EXP in "${EXPERIMENTS[@]}"; do
  IFS=':' read -r R ALPHA NAME <<< "${EXP}"
  CURRENT=$((CURRENT + 1))
  LOG_FILE="${LOG_DIR}/${NAME}.log"

  echo ""
  echo "── [${CURRENT}/${TOTAL}] ${NAME} (r=${R}, α=${ALPHA}) ──"
  echo "    Log: ${LOG_FILE}"

  START_TS=$(date +%s)

  if CUDA_VISIBLE_DEVICES=${GPU_ID} yolo train \
    cfg="${CFG}" \
    device=0 \
    lora_r="${R}" \
    lora_alpha="${ALPHA}" \
    name="${NAME}" \
    project="${PROJECT}" \
    > "${LOG_FILE}" 2>&1; then

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    MIN=$((ELAPSED / 60))
    SEC=$((ELAPSED % 60))
    echo "    ✅ Done in ${MIN}m ${SEC}s"

    # Extract key metrics from log for quick preview
    BEST_MAP=$(grep -oP 'mAP50-95\(B\)=\K[0-9.]+' "${LOG_FILE}" | tail -1 || echo "N/A")
    PEAK_VRAM=$(grep -oP '\d+\.?\d*G' "${LOG_FILE}" | tail -1 || echo "N/A")
    echo "    Best mAP50-95: ${BEST_MAP}  |  Peak VRAM: ${PEAK_VRAM}"

  else
    EXIT_CODE=$?
    echo "    ❌ FAILED (exit ${EXIT_CODE})"
    FAILED=$((FAILED + 1))
    continue
  fi
done

echo ""
echo "==========================================================================="
echo "  VisDrone Sweep Complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total: ${TOTAL} | Passed: $((TOTAL - FAILED)) | Failed: ${FAILED}"
echo "==========================================================================="

[ "${FAILED}" -eq 0 ] || exit 1
