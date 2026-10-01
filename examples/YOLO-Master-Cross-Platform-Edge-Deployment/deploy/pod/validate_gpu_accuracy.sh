#!/usr/bin/env bash
# Follow-up to validate_gpu.sh: the CPU-vs-CUDA preprocessing question answered the way it matters.
# The two paths differ by up to 1/255 per input value (float vs fixed-point bilinear), which an fp16
# engine turns into flipped marginal detections at conf 0.001, so txt parity at 0.01 px is the wrong
# gate. This script compares (a) txt parity at conf 0.25 with a 1 px tolerance and (b) in-process
# mAP50-95 on coco500 for the COCO models (esmoen, v01n) on TensorRT fp16 and ORT-CUDA, cpu vs cuda
# preprocessing, plus the CUDA-graph run, and (c) the MNN C2 candidate with fp16 really on.
#   MODELS=/root/models_api OUT=/root/validate_gpu2 bash deploy/pod/validate_gpu_accuracy.sh
set -uo pipefail
REPO="${REPO:-/data/yolo-master-edge}"
BIN="${BIN:-$REPO/cpp/build_l40s}"
MODELS="${MODELS:-$REPO/models/api}"
OUT="${OUT:-/root/validate_gpu2}"
DIR="$REPO/visdrone50/images/val"
COCO="$REPO/datasets/coco500/coco500.yaml"
PAR="python3 $REPO/scripts/server/parity_txt.py"
mkdir -p "$OUT"
log() { echo "[validate2 $(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }
run() { timeout "${RUN_TIMEOUT:-900}" "$BIN/yolomaster_edge" "$@" 2>&1; }   # a hung backend (MNN CUDA fp16 did) must not stall the chain
acc() {  # <tag> <args...>: accuracy pass on coco500, prints the [accuracy] line
  local tag="$1"; shift
  run "$@" -s "$COCO" --accuracy auto --bench-iters 20 --bench-warmup 5 --bench-json "$OUT/$tag.json" --no-save --quiet \
    | grep -E "^\[(accuracy|bench)\]" | sed "s/^/  $tag: /" | tee -a "$OUT/log.txt"
}
for id in esmoen v01n; do
  ONNX="$MODELS/$id/model.onnx"
  [ -f "$ONNX" ] || continue
  log "== $id: txt parity at conf 0.25, tol 1 px (TensorRT fp16 cpu vs cuda preproc) =="
  for v in cpu cuda; do F=""; [ $v = cpu ] && F="--cpu-preproc"; rm -rf "$OUT/trt25_${id}_$v"
    run -m "$ONNX" -b trt --precision fp16 $F -s "$DIR" --warmup 3 --save-txt "$OUT/trt25_${id}_$v" --no-save --quiet --conf 0.25 >/dev/null; done
  $PAR "$OUT/trt25_${id}_cpu" "$OUT/trt25_${id}_cuda" --tol 1.0 | tail -1 | tee -a "$OUT/log.txt"
  log "== $id: coco500 mAP, TensorRT fp16 =="
  acc "trt_${id}_cpu"   -m "$ONNX" -b trt --precision fp16 --cpu-preproc
  acc "trt_${id}_cuda"  -m "$ONNX" -b trt --precision fp16
  acc "trt_${id}_graph" -m "$ONNX" -b trt --precision fp16 --cuda-graph
  log "== $id: coco500 mAP, ORT-CUDA =="
  acc "ort_${id}_cpu"   -m "$ONNX" -b onnx -d cuda --cpu-preproc
  acc "ort_${id}_cuda"  -m "$ONNX" -b onnx -d cuda
  M="$MODELS/$id"
  if [ -f "$M/model-fp16-routed.mnn" ]; then
    log "== $id: MNN CUDA C2 (routed fp16 conversion, fp16 on) vs fp32 on coco500 =="
    grep -v "^fp16_safe:" "$M/metadata.yaml" > "$M/model-fp16-routed.metadata.yaml"; echo "fp16_safe: true" >> "$M/model-fp16-routed.metadata.yaml"
    acc "mnn_${id}_fp32" -m "$M/model.mnn" -b mnn -d cuda --precision fp32
    acc "mnn_${id}_c2"   -m "$M/model-fp16-routed.mnn" -b mnn -d cuda --precision fp16
    run -m "$M/model-fp16-routed.mnn" -b mnn -d cuda --precision fp16 -s "$DIR" --limit 3 --no-save --quiet | grep "^\[model\]" | cut -c1-200 | tee -a "$OUT/log.txt"
  fi
done
log "done -> $OUT/log.txt"
