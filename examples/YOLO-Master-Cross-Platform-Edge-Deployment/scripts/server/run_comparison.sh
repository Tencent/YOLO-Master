#!/usr/bin/env bash
# API server vs bare C++ runtime on the full COCO val2017 (5000 images), all backends.
#   bash scripts/server/run_comparison.sh [OUT=/data/results/api_bench]
# Cells: model {v01n-fp32, v01n-fp16, v01n-pruned-fp32, esmoen-fp32, esmoen-fp16}
#      x backend {ort-cuda, trt, ncnn-cpu, mnn-cpu [, mnn-cuda]}
#      x path {cli (bare runtime, sequential), api-c1 (closed loop, 1 in flight), api-c8 (8 in flight)}
# Every cell dumps 'class conf x1 y1 x2 y2' txts; mAP50-95 comes from scripts/eval_map.py on those dumps.
# Detection params for scoring: conf 0.001, iou 0.7 (ultralytics val defaults), max_det 300.
set -uo pipefail
REPO="${REPO:-/data/yolo-master-edge}"
BIN="${BIN:-$REPO/cpp/build_l40s}"
MODELS="${MODELS:-/data/models_api}"
IMAGES="${IMAGES:-/data/datasets/coco/val2017}"
LABELS="${LABELS:-/data/datasets/coco/labels/val2017}"
OUT="${1:-${OUT:-/data/results/api_bench}}"
PORT="${PORT:-18090}"
# The RunPod L40S container shows 128 vCPUs but is capped by a cgroup quota of 13.6 cores
# (/sys/fs/cgroup/cpu.max = 1360000/100000): CPU cells use 6 threads per worker and at most
# 2 workers (12 threads) so the c=8 cells measure the server, not quota thrashing.
CPU_THREADS="${CPU_THREADS:-6}"
API_WORKERS_CPU="${API_WORKERS_CPU:-2}"
API_WORKERS_GPU="${API_WORKERS_GPU:-1}"     # c=8 GPU cells: >1 lets JPEG decode/pre/post overlap across workers (one TRT/ORT context each)
LIMIT="${LIMIT:-0}"                         # 0 = full val (5000)
REPEATS="${REPEATS:-1}"
PY="${PY:-python3}"
CELLS="${CELLS:-}"                          # optional "model:backend model:backend ..." subset
PATHS="${PATHS:-cli c1 c8}"                 # which paths to run per cell (e.g. PATHS=c8 for a scaling run)
export PYTHONPATH="${PYTHONPATH:-/data/YOLO-Master}"
mkdir -p "$OUT"
log() { echo "[cmp $(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
CLI="$BIN/yolomaster_edge"; SERVER="$BIN/server/yolomaster_server"
CONF=0.001; IOU=0.7; MAXDET=300

# model id -> file per backend (+ precision flag for TRT / ncnn)
model_path() {   # $1 model-cell, $2 backend -> echo "path|backend|device|precision|threads"
  local m=$1 b=$2 dir prec
  case $m in
    v01n-fp32) dir=$MODELS/v01n; prec=fp32;; v01n-fp16) dir=$MODELS/v01n; prec=fp16;;
    v01n-pruned-fp32) dir=$MODELS/v01n-pruned; prec=fp32;;
    esmoen-fp32) dir=$MODELS/esmoen; prec=fp32;; esmoen-fp16) dir=$MODELS/esmoen; prec=fp16;;
    *) return 1;;
  esac
  case $b in
    ort-cuda) [ $prec = fp16 ] && echo "$dir/model-fp16.onnx|onnx|cuda|auto|$CPU_THREADS" || echo "$dir/model.onnx|onnx|cuda|auto|$CPU_THREADS";;
    trt)      echo "$dir/model.onnx|trt|cuda|$prec|$CPU_THREADS";;
    ncnn-cpu) [ $prec = fp16 ] && return 2 || echo "$dir/ncnn|ncnn|cpu|fp32|$CPU_THREADS";;   # x86 ncnn has no fp16 arithmetic
    mnn-cpu)  [ $prec = fp16 ] && echo "$dir/model-fp16.mnn|mnn|cpu|auto|$CPU_THREADS" || echo "$dir/model.mnn|mnn|cpu|auto|$CPU_THREADS";;
    mnn-cuda) [ $prec = fp16 ] && echo "$dir/model-fp16.mnn|mnn|cuda|fp16|$CPU_THREADS" || echo "$dir/model.mnn|mnn|cuda|fp32|$CPU_THREADS";;
    *) return 1;;
  esac
}
is_gpu() { case $1 in ort-cuda|trt|mnn-cuda) return 0;; *) return 1;; esac; }

score() {   # $1 preds dir -> prints "map50 map5095 n" (eval_map.py uses ultralytics DetMetrics)
  $PY "$REPO/scripts/eval_map.py" --preds "$1" --images "$IMAGES" --labels "$LABELS" --limit "$LIMIT" 2>/dev/null | grep -E "^images=" | tail -1
}
gpu_sample() { nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | head -1; }

run_cell() {   # $1 model $2 backend
  local m=$1 b=$2 spec; spec=$(model_path "$m" "$b") || { log "skip $m/$b (n/a)"; return; }
  IFS='|' read -r path be dev prec thr <<< "$spec"
  [ -e "$path" ] || { log "skip $m/$b (missing $path)"; return; }
  local cell="$OUT/${m}__${b}"; mkdir -p "$cell"
  local limit_arg=(); [ "$LIMIT" -gt 0 ] && limit_arg=(--limit "$LIMIT")
  # ---- bare CLI (sequential, same stb decoder, txt dump) ----
  if [[ " $PATHS " == *" cli "* ]] && [ ! -f "$cell/cli.summary" ]; then
    log "CLI  $m/$b"
    local t0=$(date +%s.%N)
    "$CLI" -m "$path" -b "$be" -d "$dev" --precision "$prec" --threads "$thr" -s "$IMAGES" "${limit_arg[@]}" --warmup 20 \
      --conf $CONF --iou $IOU --no-save --quiet --save-txt "$cell/cli_txt" > "$cell/cli.log" 2>&1
    local rc=$? t1=$(date +%s.%N)
    grep -E "^\[model\]|^\[summary\]" "$cell/cli.log" > "$cell/cli.summary"
    echo "wall_s=$(echo "$t1 - $t0" | bc) rc=$rc" >> "$cell/cli.summary"
    [ $rc -ne 0 ] && { log "CLI failed rc=$rc: $(tail -2 "$cell/cli.log")"; }
  fi
  # ---- API server: c1 latency, c8 throughput ----
  for c in 1 8; do
    [[ " $PATHS " == *" c$c "* ]] || continue
    [ -f "$cell/api_c$c.json" ] && continue
    local workers=$API_WORKERS_GPU; is_gpu "$b" || workers=$API_WORKERS_CPU; [ $c -eq 1 ] && workers=1
    log "API  $m/$b c=$c workers=$workers"
    "$SERVER" -p "$PORT" --loop-threads 2 --max-queue 64 --timeout-ms 60000 \
      -m "m=$path,backend=$be,device=$dev,precision=$prec,threads=$thr,workers=$workers" > "$cell/server_c$c.log" 2>&1 &
    local spid=$!
    for i in $(seq 1 600); do curl -sf "localhost:$PORT/readyz" >/dev/null && break; sleep 0.5; done
    curl -sf "localhost:$PORT/readyz" >/dev/null || { log "server not ready: $(tail -3 "$cell/server_c$c.log")"; kill $spid; wait $spid; continue; }
    gpu_sample > "$cell/gpu_before_c$c.txt"
    local dump=(); [ $c -eq 1 ] && dump=(--dump-txt "$cell/api_txt")
    $PY "$REPO/clients/python/bench_client.py" --url "http://localhost:$PORT" --model m --images "$IMAGES" ${LIMIT:+--limit $LIMIT} \
      --concurrency $c --warmup 20 --conf $CONF --iou $IOU --max-det $MAXDET "${dump[@]}" --summary "$cell/api_c$c.json" > "$cell/bench_c$c.log" 2>&1 &
    local bpid=$!
    # sample GPU/CPU utilisation mid-run
    sleep 5; gpu_sample > "$cell/gpu_during_c$c.txt"; top -bn1 | head -3 | tail -1 > "$cell/cpu_during_c$c.txt"
    wait $bpid
    curl -s "localhost:$PORT/v1/stats" > "$cell/stats_c$c.json"
    kill -INT $spid; wait $spid
  done
  # ---- scoring + parity ----
  if [[ " $PATHS " == *" cli "* ]] && [ ! -f "$cell/score.txt" ]; then
    log "score $m/$b"
    { echo "cli: $(score "$cell/cli_txt")"; echo "api: $(score "$cell/api_txt")"; } > "$cell/score.txt"
    echo "parity: $($PY "$REPO/scripts/server/parity_txt.py" "$cell/cli_txt" "$cell/api_txt" --tol 0.01)" >> "$cell/score.txt"
    cat "$cell/score.txt" | tee -a "$OUT/run.log"
  fi
}

MODELS_LIST="v01n-fp32 v01n-fp16 v01n-pruned-fp32 esmoen-fp32 esmoen-fp16"
BACKENDS="ort-cuda trt ncnn-cpu mnn-cpu"
[ -f "$REPO/third_party/mnn-src/build_cuda/libMNN.so" ] && [ -d "$REPO/cpp/build_l40s_cuda" ] && BACKENDS="$BACKENDS mnn-cuda"
for r in $(seq 1 $REPEATS); do
  if [ -n "$CELLS" ]; then
    for mb in $CELLS; do run_cell "${mb%%:*}" "${mb##*:}"; done
  else
    for b in $BACKENDS; do for m in $MODELS_LIST; do run_cell "$m" "$b"; done; done
  fi
done
log "all cells done -> $OUT"
$PY "$REPO/scripts/server/summarize_comparison.py" "$OUT" 2>/dev/null || true
