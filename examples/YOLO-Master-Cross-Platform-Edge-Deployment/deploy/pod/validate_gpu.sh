#!/usr/bin/env bash
# v1.2.0 GPU validation on a pod after deploy/pod/build_gpu.sh: preprocessing parity, TensorRT and
# ORT-CUDA txt parity between the CPU and the CUDA preprocessing paths, CUDA-graph parity, the
# three-column timing table, the MNN CUDA fp16 candidates, and the server suites with GPU
# preprocessing on. Everything lands under $OUT (default /root/validate_gpu).
#   OUT=/root/validate_gpu MODELS=/data/yolo-master-edge/models/api bash deploy/pod/validate_gpu.sh
set -uo pipefail
REPO="${REPO:-/data/yolo-master-edge}"
BIN="${BIN:-$REPO/cpp/build_l40s}"
MODELS="${MODELS:-$REPO/models/api}"
OUT="${OUT:-/root/validate_gpu}"
CACHE="${CACHE:-/root/trt_cache}"
DIR="$REPO/visdrone50/images/val"
LABELS="$REPO/visdrone50/labels/val"
PAR="python3 $REPO/scripts/server/parity_txt.py"
mkdir -p "$OUT" "$CACHE"
log() { echo "[validate $(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }
run() { timeout "${RUN_TIMEOUT:-900}" "$BIN/yolomaster_edge" "$@" 2>&1; }   # a hung backend (MNN CUDA fp16 did) must not stall the chain
summary() { grep -E "^\[summary\]" | sed -E 's/.*avg\/frame: (pre=[^ ]+ infer=[^ ]+ post=[^ ]+ total=[^ ]+).*/\1/'; }

if [ "${ONLY_SERVER:-0}" != 1 ]; then
log "== 1. CUDA preprocessing parity (max |diff| <= 1/255) =="
"$BIN/preproc_parity" "$DIR" | tail -1 | tee -a "$OUT/log.txt"
"$BIN/preproc_parity" "$DIR" 640 --stretch | tail -1 | tee -a "$OUT/log.txt"

for id in esmoen v01n; do
  ONNX="$MODELS/$id/model.onnx"
  [ -f "$ONNX" ] || { log "skip $id (no $ONNX)"; continue; }
  log "== 2. TensorRT fp16 $id: cpu preproc vs cuda preproc vs cuda preproc + graph =="
  for v in cpu cuda graph; do
    case $v in cpu) F="--cpu-preproc";; cuda) F="";; graph) F="--cuda-graph";; esac
    rm -rf "$OUT/trt_${id}_$v"
    run -m "$ONNX" -b trt --precision fp16 $F -s "$DIR" --warmup 5 --save-txt "$OUT/trt_${id}_$v" --no-save --quiet \
        --conf 0.001 --iou 0.7 --multi-label 2>&1 | tee "$OUT/trt_${id}_$v.log" | grep -E "^\[model\]|^\[summary\]" | cut -c1-200 | tee -a "$OUT/log.txt"
  done
  log "parity cpu vs cuda (tol 0.01 px):"; $PAR "$OUT/trt_${id}_cpu" "$OUT/trt_${id}_cuda" --tol 0.01 | tail -1 | tee -a "$OUT/log.txt"
  log "parity cuda vs graph (tol 0):";      $PAR "$OUT/trt_${id}_cuda" "$OUT/trt_${id}_graph" --tol 0 | tail -1 | tee -a "$OUT/log.txt"
  for v in cpu cuda graph; do echo -n "mAP $id trt $v: "; "$BIN/yolomaster_score" "$OUT/trt_${id}_$v" "$DIR" "$LABELS" | tee -a "$OUT/log.txt"; done

  log "== 3. ORT-CUDA $id: cpu preproc vs cuda preproc (IoBinding) =="
  for v in cpu cuda; do
    F=""; [ $v = cpu ] && F="--cpu-preproc"
    rm -rf "$OUT/ort_${id}_$v"
    run -m "$ONNX" -b onnx -d cuda $F -s "$DIR" --warmup 5 --save-txt "$OUT/ort_${id}_$v" --no-save --quiet \
        --conf 0.001 --iou 0.7 --multi-label 2>&1 | tee "$OUT/ort_${id}_$v.log" | grep -E "^\[model\]|^\[summary\]" | cut -c1-200 | tee -a "$OUT/log.txt"
  done
  log "parity ort cpu vs cuda (tol 0.01 px):"; $PAR "$OUT/ort_${id}_cpu" "$OUT/ort_${id}_cuda" --tol 0.01 | tail -1 | tee -a "$OUT/log.txt"

  log "== 4. bench JSON on TensorRT (cold run, probe_mode, device name) =="
  run -m "$ONNX" -b trt --precision fp16 -s "$DIR" --limit 10 --bench cold --bench-iters 50 --bench-warmup 10 \
      --bench-json "$OUT/bench_trt_$id.json" --no-save --quiet | grep -E "^\[bench\]" | tee -a "$OUT/log.txt"
  python3 -c "import json;j=json.load(open('$OUT/bench_trt_$id.json'));print('  gpu:',j['environment']['gpu_name'],'| probe:',j['cold']['probe_mode'],'| ep:',j['model']['execution_provider'])" | tee -a "$OUT/log.txt"

  log "== 5. MNN CUDA $id: fp32, fp16 (downgrade expected), fp16-routed (C2), multipath (C1, experimental) =="
  M="$MODELS/$id"
  for v in fp32 fp16; do
    rm -rf "$OUT/mnn_${id}_$v"
    run -m "$M/model.mnn" -b mnn -d cuda --precision $v -s "$DIR" --save-txt "$OUT/mnn_${id}_$v" --no-save --quiet \
        --conf 0.001 --iou 0.7 --multi-label 2>&1 | grep -E "^\[model\]|^\[summary\]" | cut -c1-220 | tee -a "$OUT/log.txt"
    echo -n "mAP $id mnn-cuda $v: "; "$BIN/yolomaster_score" "$OUT/mnn_${id}_$v" "$DIR" "$LABELS" | tee -a "$OUT/log.txt"
  done
  if [ -f "$M/model-fp16-routed.mnn" ]; then
    rm -rf "$OUT/mnn_${id}_c2"
    run -m "$M/model-fp16-routed.mnn" -b mnn -d cuda --precision fp16 -s "$DIR" --save-txt "$OUT/mnn_${id}_c2" --no-save --quiet \
        --conf 0.001 --iou 0.7 --multi-label 2>&1 | grep -E "^\[model\]|^\[summary\]" | cut -c1-220 | tee -a "$OUT/log.txt"
    echo -n "mAP $id mnn-cuda C2 fp16-routed: "; "$BIN/yolomaster_score" "$OUT/mnn_${id}_c2" "$DIR" "$LABELS" | tee -a "$OUT/log.txt"
  fi
  if [ -f "$M/model.paths.json" ]; then
    rm -rf "$OUT/mnn_${id}_c1"
    YOLOMASTER_MNN_MULTIPATH=1 run -m "$M/model.mnn" -b mnn -d cuda --precision fp16 -s "$DIR" --save-txt "$OUT/mnn_${id}_c1" --no-save --quiet \
        --conf 0.001 --iou 0.7 --multi-label 2>&1 | grep -E "^\[model\]|^\[summary\]|error" | cut -c1-220 | tee -a "$OUT/log.txt"
    echo -n "mAP $id mnn-cuda C1 multipath: "; "$BIN/yolomaster_score" "$OUT/mnn_${id}_c1" "$DIR" "$LABELS" | tee -a "$OUT/log.txt"
  fi
done
fi   # ONLY_SERVER

log "== 6. server suites with GPU preprocessing (TensorRT model + ORT-CUDA model) =="
PORT=$(( 20000 + RANDOM % 20000 ))
# no --engine-cache: the server then builds / reuses the engine next to the .onnx, the same file the
# CLI parity test loads (a separately built engine can pick different fp16 tactics)
"$BIN/server/yolomaster_server" -p "$PORT" --loop-threads 2 --max-queue 8 --max-body-mb 4 \
  -m "esmoe=$MODELS/esmoen/model.onnx,backend=trt,device=cuda,precision=fp16,workers=1" \
  -m "esmoe-ort=$MODELS/esmoen/model.onnx,backend=onnx,device=cuda,workers=1" > "$OUT/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 240); do curl -sf "localhost:$PORT/readyz" >/dev/null && break; sleep 0.5; done
curl -s "localhost:$PORT/v1/models" | python3 -c "import json,sys;[print('  ',m['id'],m.get('ep'),m.get('ready')) for m in json.load(sys.stdin)['models']]" | tee -a "$OUT/log.txt"
export YM_SERVER_URL="http://localhost:$PORT" YM_TEST_MODEL=esmoe YM_TEST_MODEL2=esmoe-ort YM_TEST_IMAGES="$DIR" YM_CLI="$BIN/yolomaster_edge" YM_CLI_MODEL="$MODELS/esmoen/model.onnx"
export YM_CLI_ARGS="-b trt --precision fp16"   # the CLI parity test must run the same engine as the server's esmoe model
python3 -m pytest "$REPO/tests/server/test_api.py" -q -p no:cacheprovider 2>&1 | tail -3 | tee -a "$OUT/log.txt"
kill -INT $SPID; wait $SPID
log "done -> $OUT/log.txt"
