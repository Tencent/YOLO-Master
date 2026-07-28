#!/usr/bin/env bash
# ==============================================================================
# YOLO-Master-EsMoE-N LoRA Rank Sweep — Brain Tumor (Sparse Medical Detection)
#
# Usage:
#   bash examples/lora_examples/run_lora_brain_tumor_sweep.sh
#
# Prerequisites:
#   - yolo command available (pip install ultralytics or editable install)
#   - Brain Tumor dataset downloaded (auto-downloads via brain-tumor.yaml on first run)
#   - NVIDIA A40 (48 GB) or equivalent GPU
#
# Sweep matrix: r ∈ {4, 8, 16, 32} × alpha = 2×r
# Total experiments: 4 (serial execution for accurate timing)
# ==============================================================================
set -euo pipefail

CFG="examples/lora_examples/yolo_master_brain_tumor_lora.yaml"
PROJECT="runs/lora_examples"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="logs/brain_tumor_sweep"

# Rank sweep: r:alpha:run_name
EXPERIMENTS=(
  "4:8:brain_tumor_r4"
  "8:16:brain_tumor_r8"
  "16:32:brain_tumor_r16"
  "32:64:brain_tumor_r32"
)

mkdir -p "${LOG_DIR}"

echo "==========================================================================="
echo "  Brain Tumor LoRA Rank Sweep — YOLO-Master-EsMoE-N"
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
echo "  Brain Tumor Sweep Complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total: ${TOTAL} | Passed: $((TOTAL - FAILED)) | Failed: ${FAILED}"
echo "==========================================================================="

[ "${FAILED}" -eq 0 ] || exit 1
