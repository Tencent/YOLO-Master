#!/usr/bin/env bash
# Launch YOLOE-26-Master-N runs on 8x A100-80GB via torchrun.
#
# Path-independent: TASK_ROOT is derived from this script's location, so the same
# file works whether the share is mounted at /comp_robot (event-engine-1) or
# /vepfs_comp_robot (event-engine-2).
#
# ARMS
#   arm (default) : MODEL_CFG=yoloe-26-master-n-seg.yaml  TRAIN_SCOPE=preserve50
#   baseline      : MODEL_CFG=yoloe-26-seg.yaml           TRAIN_SCOPE=full
# The baseline has no `.moe_adapter.` / `.moe_scale` params, so preserve50 would
# abort with "No MoE adapter parameters found".
#
# BATCH SEMANTICS: Ultralytics treats `batch` as the GLOBAL batch and divides it by
# world_size inside _build_train_pipeline(). 128 / 8 ranks = 16 per GPU, and with
# nbs=128 the gradient accumulation collapses to 1. Effective batch (128) and the
# weight-decay scaling (wd * 128 * 1 / 128) therefore match the original 4090 run.
#
# DDP LAUNCH: we drive torchrun ourselves instead of letting model.train() spawn its
# own DDP subprocess. Ultralytics writes a temp file that does
# `from __main__ import <TrainerClass>`, which cannot work for a trainer defined in
# a script, and the child would also resolve `ultralytics` to whatever the venv has
# installed rather than this checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOME_BASE="$(cd "$TASK_ROOT/../.." && pwd)"
PYTHON_BIN="$HOME_BASE/codes/.artifacts/yolo_master_coco_venv/bin/python"
O365_IMAGES="$HOME_BASE/public_data/objects365_v1/objects365/train"

SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-yoloe26_master_n_preserve50_main_30ep_a100x8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
TOTAL_BATCH="${TOTAL_BATCH:-128}"
WORKERS="${WORKERS:-5}"
EPOCHS="${EPOCHS:-30}"
# Short probes use FRACTION to cut one epoch down to ~1.5k steps, and must set
# WARMUP_EPOCHS=0 LRF=1.0 or the whole run sits in warmup / decays to nothing.
FRACTION="${FRACTION:-1.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3.0}"
LRF="${LRF:-0.01}"
# AMP=false trains in FP32; used to isolate fp16 overflow in the MoE routing path.
AMP="${AMP:-true}"
AMP_FLAG=""
[ "$AMP" = "false" ] && AMP_FLAG="--no-amp"

# Feature-preservation distillation: charge only for per-block drift beyond TOL.
# Measured operating point is ~0.3% relative drift; 3% costs ~3.4 AP.
DISTILL_COEFF="${DISTILL_COEFF:-0}"
DISTILL_TOL="${DISTILL_TOL:-0.003}"
# "ram" would cache DECODED images (~595 GiB/rank for Objects365), so it is off by
# default: 8 ranks would need ~4.8 TiB against 1.88 TiB of RAM.
CACHE="${CACHE:-false}"

MODEL_CFG="${MODEL_CFG:-ultralytics/cfg/models/26/yoloe-26-master-n-seg.yaml}"
TRAIN_SCOPE="${TRAIN_SCOPE:-preserve50}"
SCRATCH="${SCRATCH:-0}"
SCRATCH_FLAG=""
[ "$SCRATCH" = "1" ] && SCRATCH_FLAG="--scratch"
INIT_FROM="${INIT_FROM:-$TASK_ROOT/weights/yoloe-26n-seg.pt}"
LR0="${LR0:-0.002}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
PRESERVE_COEFF="${PRESERVE_COEFF:-50.0}"
# Ceiling on tanh(moe_scale). 1.0 reproduces the unconstrained behaviour; small
# values stop the residual branch from overwriting the frozen core.
GATE_CAP="${GATE_CAP:-1.0}"

cd "$TASK_ROOT"
mkdir -p data/yoloe_main/runs
# gate_cap is a runtime-only knob and is NOT part of the state_dict, so any later
# inference must replay the same value. Persist it next to the run instead of
# relying on whoever launches the eval remembering the env var.
mkdir -p "data/yoloe_main/runs/$RUN_NAME"
printf '%s\n' "$GATE_CAP" > "data/yoloe_main/runs/$RUN_NAME/gate_cap.txt"
# TSMoE's residual cap and the query-text ablation are also runtime-only knobs.
# Persist them so the eval watcher can replay exactly what training used.
TSMOE_GATE_CAP="${TSMOE_GATE_CAP:-}"
if [ -n "$TSMOE_GATE_CAP" ]; then
  printf '%s\n' "$TSMOE_GATE_CAP" > "data/yoloe_main/runs/$RUN_NAME/tsmoe_gate_cap.txt"
  export YOLO_TSMOE_GATE_CAP="$TSMOE_GATE_CAP"
fi
TEXT_ABLATION="${TEXT_ABLATION:-}"
if [ -n "$TEXT_ABLATION" ]; then
  printf '%s\n' "$TEXT_ABLATION" > "data/yoloe_main/runs/$RUN_NAME/text_ablation.txt"
  export YOLO_TSMOE_TEXT_ABLATION="$TEXT_ABLATION"
fi
ROUTE_CONTEXT="${ROUTE_CONTEXT:-}"
if [ -n "$ROUTE_CONTEXT" ]; then
  printf '%s\n' "$ROUTE_CONTEXT" > "data/yoloe_main/runs/$RUN_NAME/route_context.txt"
  export YOLO_TSMOE_ROUTE_CONTEXT="$ROUTE_CONTEXT"
fi
# Opt-in: publish real class text embeddings to TSMoE routers during training.
# Legacy runs leave this unset and stay byte-for-byte reproducible.
TRAIN_TEXT_CONTEXT="${TRAIN_TEXT_CONTEXT:-}"
if [ "$TRAIN_TEXT_CONTEXT" = "1" ]; then
  printf '1\n' > "data/yoloe_main/runs/$RUN_NAME/train_text_context.txt"
  export YOLO_TSMOE_TRAIN_TEXT_CONTEXT=1
fi

case "$DISTILL_COEFF" in
  0 | 0.0 | "") ;;
  *)
    export YOLO_MOE_DISTILL_COEFF="$DISTILL_COEFF"
    export YOLO_MOE_DISTILL_TOL="$DISTILL_TOL"
    printf 'coeff=%s tol=%s\n' "$DISTILL_COEFF" "$DISTILL_TOL" > "data/yoloe_main/runs/$RUN_NAME/distill.txt"
    ;;
esac

# One intra-op thread per process: the dataloader workers already fill the node,
# and default torch threading oversubscribes 112 cores many times over.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# The label cache hash verification stats 1.2M paths per rank and costs hours on
# this network filesystem. The dataset is static for this run, so trust the cache.
export YOLO_SKIP_CACHE_HASH=1
export YOLO_MOE_GATE_CAP="$GATE_CAP"

# A fraction<1 probe sees only part of the annotations, so the grounding dataset's
# hardcoded full-dataset instance count no longer holds.
case "$FRACTION" in
  1 | 1.0) ;;
  *) export YOLO_SKIP_LABEL_COUNT_CHECK=1 ;;
esac

# SwanLab credentials live outside the repo (chmod 600) so keys never enter git.
if [ -f "$HOME_BASE/.swanlab_env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$HOME_BASE/.swanlab_env"
  set +a
fi
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-YOLOE-Master}"

exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  scripts/ov_distill/train_yoloe_main.py \
  --device "$GPUS" \
  --batch "$TOTAL_BATCH" \
  --workers "$WORKERS" \
  --cache "$CACHE" \
  --epochs "$EPOCHS" \
  --fraction "$FRACTION" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --lrf "$LRF" \
  $AMP_FLAG \
  --imgsz 640 \
  --seed "$SEED" \
  --lr0 "$LR0" \
  --optimizer "$OPTIMIZER" \
  --preserve-coeff "$PRESERVE_COEFF" \
  --model-cfg "$MODEL_CFG" \
  --init-from "$INIT_FROM" \
  --train-scope "$TRAIN_SCOPE" \
  $SCRATCH_FLAG \
  --objects365-images "$O365_IMAGES" \
  --name "$RUN_NAME"
