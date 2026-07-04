#!/usr/bin/env bash
set -euo pipefail

# ==================== 配置区 ====================
CONDA_ENV="yolo_master"
CFG="examples/lora_examples/yolo_master_brain_tumor_lora.yaml"
PROJECT="runs/lora_examples"
GPU_ID=0
LOG_DIR="logs"

# 实验参数矩阵 (r:alpha:name)
EXPERIMENTS=(
  "4:8:brain_tumor_r4"
  "8:16:brain_tumor_r8"
  "16:32:brain_tumor_r16"
)
# ================================================

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "🚀 Brain Tumor LoRA Sweep Starting"
echo "   GPU: ${GPU_ID}"
echo "   Experiments: ${#EXPERIMENTS[@]}"
echo "   Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0
FAILED=0

for EXP in "${EXPERIMENTS[@]}"; do
  IFS=':' read -r R ALPHA NAME <<< "${EXP}"
  CURRENT=$((CURRENT + 1))
  LOG_FILE="${LOG_DIR}/${NAME}.log"

  echo ""
  echo "[${CURRENT}/${TOTAL}] 🏋️ Training: ${NAME} (r=${R}, alpha=${ALPHA})"
  echo "   Log: ${LOG_FILE}"
  echo "   Started: $(date '+%H:%M:%S')"

  if CUDA_VISIBLE_DEVICES=${GPU_ID} yolo train \
    cfg="${CFG}" \
    device=0 \
    lora_r="${R}" \
    lora_alpha="${ALPHA}" \
    name="${NAME}" \
    project="${PROJECT}" \
    > "${LOG_FILE}" 2>&1; then
    echo "   ✅ Completed: $(date '+%H:%M:%S')"
  else
    EXIT_CODE=$?
    echo "   ❌ FAILED (exit code ${EXIT_CODE}): $(date '+%H:%M:%S')"
    FAILED=$((FAILED + 1))
    continue
  fi
done

echo ""
echo "=========================================="
echo "🏁 Brain Tumor Sweep Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   Total: ${TOTAL} | Success: $((TOTAL - FAILED)) | Failed: ${FAILED}"
echo "=========================================="

[ "${FAILED}" -eq 0 ] || exit 1