#!/usr/bin/env bash
set -euo pipefail

# Remote training entrypoint for the MoT hybrid architecture issue.
# Override any variable from the command line, for example:
#   DEVICE=1 BATCH=8 EPOCHS=100 bash examples/mot_hybrid_architecture/run_visdrone_mot_ablation.sh
#   RESUME=1 BATCH=8 bash examples/mot_hybrid_architecture/run_visdrone_mot_ablation.sh
# AMP is disabled by default for stability. Use AMP=1 to re-enable it.

CONDA_ENV="${CONDA_ENV:-yolo_master}"
DATA="${DATA:-ultralytics/cfg/datasets/VisDrone.yaml}"
PROJECT="${PROJECT:-runs/mot_ablation/visdrone_v10_mot_hybrid_50ep}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
WARMUP="${WARMUP:-20}"
REPS="${REPS:-200}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_BUILD_CHECK="${RUN_BUILD_CHECK:-1}"
RUN_REQUIRED="${RUN_REQUIRED:-0}"
RUN_HYBRID="${RUN_HYBRID:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_PREPARE_ROUTING="${RUN_PREPARE_ROUTING:-1}"
RUN_ROUTING="${RUN_ROUTING:-1}"
AMP="${AMP:-0}"
RESUME="${RESUME:-0}"
DETERMINISTIC="${DETERMINISTIC:-0}"
BENCHMARK_MODELS="${BENCHMARK_MODELS:-v10 v10_mot v10_moa v10_moa_mot}"
SUMMARY_MODELS="${SUMMARY_MODELS:-v10 v10_mot v10_moa v10_moa_mot}"
ROUTING_DATASET="${ROUTING_DATASET:-datasets/VisDrone}"
ROUTING_SPLIT="${ROUTING_SPLIT:-val}"
ROUTING_SCENES="${ROUTING_SCENES:-datasets/VisDrone/routing_scenes}"
ROUTING_BATCH="${ROUTING_BATCH:-8}"
ROUTING_MAX_IMAGES="${ROUTING_MAX_IMAGES:-128}"
ROUTING_PERMUTATIONS="${ROUTING_PERMUTATIONS:-5000}"
ROUTING_BOOTSTRAP_SAMPLES="${ROUTING_BOOTSTRAP_SAMPLES:-5000}"
ROUTING_ALPHA="${ROUTING_ALPHA:-0.05}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

echo "[1/9] Environment"
python - <<'PY'
import sys
import torch
print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
PY

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[2/9] Boundary tests"
  MPLCONFIGDIR=/tmp/yolo_master_matplotlib \
    python -m pytest tests/test_validator_helpers.py tests/test_mot.py -q
else
  echo "[2/9] Boundary tests skipped"
fi

AMP_ARGS=(--no-amp)
if [[ "$AMP" == "1" ]]; then
  AMP_ARGS=(--amp)
fi

RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

DETERMINISTIC_ARGS=(--no-deterministic)
if [[ "$DETERMINISTIC" == "1" ]]; then
  DETERMINISTIC_ARGS=(--deterministic)
fi
read -r -a BENCHMARK_MODEL_ARGS <<< "$BENCHMARK_MODELS"
read -r -a SUMMARY_MODEL_ARGS <<< "$SUMMARY_MODELS"

if [[ "$RUN_BUILD_CHECK" == "1" ]]; then
  echo "[3/9] Build check"
  python scripts/compare_mot_ablation.py \
    --check-build \
    --models v10 v10_mot v10_moa v10_moa_mot \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --project "$PROJECT"
else
  echo "[3/9] Build check skipped"
fi

if [[ "$RUN_REQUIRED" == "1" ]]; then
  echo "[4/9] Train required variants: v10, v10_mot, v10_moa"
  python scripts/compare_mot_ablation.py \
    --train \
    --models v10 v10_mot v10_moa \
    --data "$DATA" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --project "$PROJECT" \
    --plots \
    --exist-ok \
    "${AMP_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${DETERMINISTIC_ARGS[@]}"
else
  echo "[4/9] Required variants skipped"
fi

if [[ "$RUN_HYBRID" == "1" ]]; then
  echo "[5/9] Train hybrid variant: v10_moa_mot"
  python scripts/compare_mot_ablation.py \
    --train \
    --models v10_moa_mot \
    --data "$DATA" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --project "$PROJECT" \
    --plots \
    --exist-ok \
    "${AMP_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${DETERMINISTIC_ARGS[@]}"
else
  echo "[5/9] Hybrid skipped"
fi

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  echo "[6/9] Latency and actual FLOPs benchmark: $BENCHMARK_MODELS"
  python scripts/compare_mot_ablation.py \
    --benchmark \
    --models "${BENCHMARK_MODEL_ARGS[@]}" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --warmup "$WARMUP" \
    --reps "$REPS" \
    --actual-flops \
    --project "$PROJECT"
else
  echo "[6/9] Benchmark skipped"
fi

if [[ "$RUN_SUMMARY" == "1" ]]; then
  echo "[7/9] Summary table: $SUMMARY_MODELS"
  python scripts/compare_mot_ablation.py \
    --summary-only \
    --models "${SUMMARY_MODEL_ARGS[@]}" \
    --project "$PROJECT"
else
  echo "[7/9] Summary skipped"
fi

if [[ "$RUN_PREPARE_ROUTING" == "1" ]]; then
  echo "[8/9] Prepare routing scene folders"
  python scripts/prepare_mot_routing_scenes.py \
    --dataset "$ROUTING_DATASET" \
    --split "$ROUTING_SPLIT" \
    --output "$ROUTING_SCENES" \
    --max-images-per-scene "$ROUTING_MAX_IMAGES"
else
  echo "[8/9] Routing scene preparation skipped"
fi

if [[ "$RUN_ROUTING" == "1" ]]; then
  if [[ ! -d "$ROUTING_SCENES" ]]; then
    echo "[routing] missing scene directory: $ROUTING_SCENES" >&2
    exit 1
  fi
  if [[ ! -f "$PROJECT/v10_mot/weights/best.pt" ]]; then
    echo "[routing] missing MoT checkpoint: $PROJECT/v10_mot/weights/best.pt" >&2
    echo "[routing] set PROJECT to the directory containing v10_mot, or set RUN_ROUTING=0" >&2
    exit 1
  fi
  echo "[9/9] Diagnose trained v10_mot router and test Deformable activation"
  python scripts/diagnose_mot_routing.py \
    --model "$PROJECT/v10_mot/weights/best.pt" \
    --image-dir "$ROUTING_SCENES" \
    --device "$DEVICE" \
    --imgsz "$IMGSZ" \
    --batch "$ROUTING_BATCH" \
    --max-images "$ROUTING_MAX_IMAGES" \
    --project "$PROJECT/routing" \
    --permutations "$ROUTING_PERMUTATIONS" \
    --bootstrap-samples "$ROUTING_BOOTSTRAP_SAMPLES" \
    --alpha "$ROUTING_ALPHA"
else
  echo "[9/9] Routing skipped"
fi

echo "Done."
echo "Summary: $PROJECT/summary.csv"
echo "Routing: $PROJECT/routing/"
echo "Runs:    $PROJECT/{v10,v10_mot,v10_moa,v10_moa_mot}/"
