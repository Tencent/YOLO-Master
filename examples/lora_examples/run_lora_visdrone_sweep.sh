#!/usr/bin/env bash
set -euo pipefail

# ==================== 配置区 ====================
CONDA_ENV="yolo_master"
CFG="examples/lora_examples/yolo_master_visdrone_lora.yaml"
PROJECT="runs/lora_examples"
GPU_ID=0
LOG_DIR="logs"

# 实验参数矩阵 (r:alpha:name)
EXPERIMENTS=(
  "4:8:visdrone_r4"
  "8:16:visdrone_r8"
  "16:32:visdrone_r16"
)
# ================================================

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "🚀 LoRA Sweep Starting"
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

  # ✅ 关键修复：去掉 &，串行执行，避免 GPU 争抢
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
    # set -e 下失败会退出，这里手动捕获以便继续下一个实验
    # 如果希望失败即停，删除下面这行即可
    continue
  fi
done

echo ""
echo "=========================================="
echo "🏁 Sweep Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   Total: ${TOTAL} | Success: $((TOTAL - FAILED)) | Failed: ${FAILED}"
echo "=========================================="

# 如果有失败的实验，以非零状态退出（方便 CI/调度系统感知）
[ "${FAILED}" -eq 0 ] || exit 1