#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/jpfs/huangyidan3/Rhinoceros——Bird/YOLO-Master-main}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROJECT="${PROJECT:-${PROJECT_DIR}/runs/mot_ablation_k8s/${RUN_ID}}"
DATA="${DATA:-/jpfs/huangyidan3/datasets/VisDrone.yaml}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
DOWNLOAD_DATA_SOURCE="${DOWNLOAD_DATA_SOURCE:-official}"
HF_DATASET_REPO="${HF_DATASET_REPO:-banu4prasad/VisDrone-Dataset}"
HF_SNAPSHOT_DIR="${HF_SNAPSHOT_DIR:-/jpfs/huangyidan3/datasets/hf/VisDrone-Dataset}"
HF_ALLOW_PATTERNS="${HF_ALLOW_PATTERNS:-README.md visdrone.yaml VisDrone2019-DET-train/** VisDrone2019-DET-val/**}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
HF_RETRIES="${HF_RETRIES:-8}"
DATASET_ROOT="${DATASET_ROOT:-/jpfs/huangyidan3/datasets/VisDrone}"
DATASET_YAML="${DATASET_YAML:-/jpfs/huangyidan3/datasets/VisDrone.yaml}"
VISDRONE_DOWNLOAD_DIR="${VISDRONE_DOWNLOAD_DIR:-/jpfs/huangyidan3/datasets/downloads/VisDrone}"
VISDRONE_RETRIES="${VISDRONE_RETRIES:-10}"
MODELS="${MODELS:-v010_esmoe v010_mot v010_moa v010_moa_mot}"
SUMMARY_MODELS="${SUMMARY_MODELS:-${MODELS}}"
DEVICE="${DEVICE:-0,1,2,3,4,5,6,7}"
IMGSZ="${IMGSZ:-640}"
WARMUP="${WARMUP:-10}"
REPS="${REPS:-100}"
FLOPS_METHOD="${FLOPS_METHOD:-thop}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
AMP="${AMP:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_BUILD_BENCHMARK="${RUN_BUILD_BENCHMARK:-1}"
RUN_ROUTE_ANALYSIS="${RUN_ROUTE_ANALYSIS:-0}"
ROUTE_MODEL="${ROUTE_MODEL:-${PROJECT}/v010_mot/weights/best.pt}"
ROUTE_LIMIT="${ROUTE_LIMIT:-512}"
VISDRONE_ANN_DIR="${VISDRONE_ANN_DIR:-}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SETUP_SCRIPT="${SETUP_SCRIPT:-${PROJECT_DIR}/scripts/setup_k8s_env.sh}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT}/logs"

echo "=========================================="
echo "YOLO-Master MoT Ablation"
echo "project=${PROJECT}"
echo "data=${DATA}"
echo "models=${MODELS}"
echo "summary_models=${SUMMARY_MODELS}"
echo "device=${DEVICE}"
echo "run_train=${RUN_TRAIN}"
echo "run_build_benchmark=${RUN_BUILD_BENCHMARK}"
echo "=========================================="

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  bash "${SETUP_SCRIPT}" 2>&1 | tee "${PROJECT}/logs/setup_env.log"
fi

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  if [[ "${DOWNLOAD_DATA_SOURCE}" == "hf" ]]; then
    read -r -a hf_allow_patterns <<< "${HF_ALLOW_PATTERNS}"
    python scripts/download_hf_dataset.py \
      --repo-id "${HF_DATASET_REPO}" \
      --local-dir "${HF_SNAPSHOT_DIR}" \
      --allow-patterns "${hf_allow_patterns[@]}" \
      --max-workers "${HF_MAX_WORKERS}" \
      --retries "${HF_RETRIES}" \
      --no-force
    python scripts/prepare_visdrone_hf.py \
      --snapshot-dir "${HF_SNAPSHOT_DIR}" \
      --out-dir "${DATASET_ROOT}" \
      --yaml-out "${DATASET_YAML}"
  else
    python scripts/download_visdrone.py \
      --root "${DATASET_ROOT}" \
      --download-dir "${VISDRONE_DOWNLOAD_DIR}" \
      --yaml-out "${DATASET_YAML}" \
      --retries "${VISDRONE_RETRIES}"
  fi
  DATA="${DATASET_YAML}"
fi

python -m pytest tests/test_mot.py tests/test_moa.py -q 2>&1 | tee "${PROJECT}/logs/pytest_mot_moa.log"

if [[ "${RUN_BUILD_BENCHMARK}" == "1" ]]; then
  python scripts/compare_mot_ablation.py \
    --models ${MODELS} \
    --project "${PROJECT}" \
    --data "${DATA}" \
    --device "${DEVICE}" \
    --imgsz "${IMGSZ}" \
    --check-build \
    --benchmark \
    --warmup "${WARMUP}" \
    --reps "${REPS}" \
    --flops-method "${FLOPS_METHOD}" \
    --exist-ok 2>&1 | tee "${PROJECT}/logs/build_benchmark.log"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_amp_arg="--amp"
  if [[ "${AMP}" == "0" || "${AMP}" == "false" || "${AMP}" == "False" ]]; then
    train_amp_arg="--no-amp"
  fi
  python scripts/compare_mot_ablation.py \
    --models ${MODELS} \
    --project "${PROJECT}" \
    --data "${DATA}" \
    --device "${DEVICE}" \
    --imgsz "${IMGSZ}" \
    --train \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --seed "${SEED}" \
    "${train_amp_arg}" \
    --plots \
    --exist-ok 2>&1 | tee "${PROJECT}/logs/train.log"
fi

if [[ "${RUN_ROUTE_ANALYSIS}" == "1" ]]; then
  route_args=(
    --model "${ROUTE_MODEL}"
    --data "${DATA}"
    --split val
    --out "${PROJECT}/routing"
    --device "${DEVICE}"
    --imgsz "${IMGSZ}"
    --limit "${ROUTE_LIMIT}"
  )
  if [[ -n "${VISDRONE_ANN_DIR}" ]]; then
    route_args+=(--visdrone-ann-dir "${VISDRONE_ANN_DIR}")
  fi
  python scripts/analyze_mot_routing.py "${route_args[@]}" 2>&1 | tee "${PROJECT}/logs/routing.log"
fi

python scripts/compare_mot_ablation.py \
  --models ${SUMMARY_MODELS} \
  --project "${PROJECT}" \
  --summary-only 2>&1 | tee "${PROJECT}/logs/summary.log"

echo "Artifacts written under ${PROJECT}"
