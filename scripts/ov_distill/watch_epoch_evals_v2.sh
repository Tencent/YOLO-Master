#!/usr/bin/env bash
# Watch the master run and evaluate epoch 1/3/5 with the LVIS federated protocol.
#
# Runs on a separate GPU box (gpu4090-3) so it never competes with the 8x A100
# training job. Each target epoch's `weights/last.pt` is snapshotted to
# `weights/epoch{N}.pt` first, otherwise later epochs would overwrite it and the
# evaluation would silently report the wrong checkpoint.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOME_BASE="$(cd "$ROOT/../.." && pwd)"
PY="$HOME_BASE/codes/.artifacts/yolo_master_coco_venv/bin/python"

RUN_NAME="${RUN_NAME:-yoloe26_master_n_preserve50_main_30ep_a100x8}"
RUN_DIR="$ROOT/data/yoloe_main/runs/$RUN_NAME"
TRAIN_LOG="$RUN_DIR.train.log"
CFG="${CFG:-ultralytics/cfg/models/26/yoloe-26-master-n-seg.yaml}"
DATA="$HOME_BASE/workspaces/datasets/lvis_ov_v1/full80_seed42_v1"
VAL="$HOME_BASE/public_data/lvis/lvis_v1_val.json"

TARGETS="${TARGETS:-1 3 5}"
POLL="${POLL:-120}"
IDLE_LIMIT="${IDLE_LIMIT:-3600}"
LOG="$RUN_DIR/epoch_eval_watch.log"

# gate_cap is a runtime-only knob, not part of the state_dict. Without replaying the
# training value here the evaluation silently runs a different (much stronger)
# residual branch and reports a collapsed model.
#
# Priority (low -> high): legacy file < TSMoE recorded file < explicit env override.
# The old ordering put the recorded file LAST, which silently discarded an explicit
# TSMOE_GATE_CAP passed on the command line (e.g. for gate x route factorials).
if [ -f "$RUN_DIR/gate_cap.txt" ]; then
  export YOLO_MOE_GATE_CAP="$(cat "$RUN_DIR/gate_cap.txt")"
fi
if [ -f "$RUN_DIR/tsmoe_gate_cap.txt" ]; then
  export YOLO_TSMOE_GATE_CAP="$(cat "$RUN_DIR/tsmoe_gate_cap.txt")"
fi
if [ -n "${TSMOE_GATE_CAP:-}" ]; then
  export YOLO_TSMOE_GATE_CAP="$TSMOE_GATE_CAP"
fi
if [ -f "$RUN_DIR/text_ablation.txt" ]; then
  export YOLO_TSMOE_TEXT_ABLATION="$(cat "$RUN_DIR/text_ablation.txt")"
fi
if [ -f "$RUN_DIR/route_context.txt" ]; then
  export YOLO_TSMOE_ROUTE_CONTEXT="$(cat "$RUN_DIR/route_context.txt")"
fi

cd "$ROOT" || exit 1
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "watcher start | run=$RUN_NAME | targets=[$TARGETS] | poll=${POLL}s"

last_size=0
idle=0

while :; do
  # Give up if the training log has stopped growing: training died or finished.
  size=$(stat -c %s "$TRAIN_LOG" 2>/dev/null || echo 0)
  if [ "$size" -le "$last_size" ]; then
    idle=$((idle + POLL))
  else
    idle=0
  fi
  last_size=$size
  if [ "$idle" -ge "$IDLE_LIMIT" ]; then
    log "training log idle for ${idle}s, stopping watcher"
    break
  fi

  done_epochs=0
  if [ -f "$RUN_DIR/results.csv" ]; then
    lines=$(wc -l < "$RUN_DIR/results.csv")
    done_epochs=$((lines - 1))
    [ "$done_epochs" -lt 0 ] && done_epochs=0
  fi

  for e in $TARGETS; do
    out="$RUN_DIR/lvis_eval_ep$e"
    [ -f "$out/summary.json" ] && continue
    [ "$done_epochs" -ge "$e" ] || continue

    snap="$RUN_DIR/weights/epoch$e.pt"
    if [ ! -f "$snap" ]; then
      # results.csv is written after the checkpoint, but keep a small guard so we
      # never snapshot a previous epoch's file.
      sleep 20
      if [ ! -f "$RUN_DIR/weights/last.pt" ]; then
        log "epoch $e done but weights/last.pt missing, retrying next round"
        continue
      fi
      cp -f "$RUN_DIR/weights/last.pt" "$snap" || { log "snapshot failed for epoch $e"; continue; }
      log "epoch $e checkpoint snapshotted -> $snap"
    fi

    log "evaluating epoch $e"
    mkdir -p "$out"
    flock /tmp/lvis_eval_4090.lock "$PY" -u scripts/ov_distill/eval_lvis_ov.py \
      --checkpoint "$snap" \
      --model-cfg "$CFG" \
      --dataset "$DATA" \
      --annotations "$VAL" \
      --output "$out" \
      --imgsz 512 --batch 8 --device 0 --seed 42 \
      >> "$out/eval.log" 2>&1
    rc=$?
    log "epoch $e eval finished rc=$rc -> $out"
  done

  all_done=1
  for e in $TARGETS; do
    [ -f "$RUN_DIR/lvis_eval_ep$e/summary.json" ] || all_done=0
  done
  if [ "$all_done" -eq 1 ]; then
    log "all target epochs evaluated, stopping watcher"
    break
  fi

  sleep "$POLL"
done
