#!/usr/bin/env bash
# Prebuilt-bundle acceptance on a plain GPU box (no CUDA / cuDNN / TensorRT installed, only a driver).
# Every bundle: checksum, unpack, clean-env CLI (env -i: nothing from this host's environment), one
# inference per bundled model on visdrone50, --bench cold JSON (+ schema check), --accuracy on coco500,
# tracking on the pan clip. GPU bundles: --device cuda / -b trt (engine built HERE for this GPU), CUDA
# graph, CPU-preproc parity (mAP), a short sustained run. Log: /root/test_bundles.log
set -uo pipefail
B=/root/bundles; W=/root/work; L=/root/test_bundles.log; V=/root/visdrone50/images/val
mkdir -p $W; : > $L
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $L; }
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); log "  PASS  $*"; }
bad()  { FAIL=$((FAIL+1)); log "  FAIL  $*"; }
# run the bundle CLI with a scrubbed environment (no PATH to toolkits, no LD_LIBRARY_PATH)
run() { local dir="$1"; shift; env -i HOME=$W PATH=/usr/bin:/bin timeout "${T:-900}" "$dir/yolomaster_edge" "$@" 2>&1; }
summary_ms() { grep -oE "infer=[0-9.]+" | head -1 | cut -d= -f2; }
acc() { grep -oE "mAP50-95=[0-9.]+" | head -1 | cut -d= -f2; }
ep() { grep -oE "ep=[^ ]+" | head -1; }

log "host: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) | $(grep PRETTY /etc/os-release | cut -d= -f2) | glibc $(ldd --version | head -1 | awk '{print $NF}')"
log "installed cuDNN/TensorRT packages: $(dpkg -l 2>/dev/null | grep -cE 'libcudnn|libnvinfer') | libcudnn/libnvinfer in /usr/lib: $(ls /usr/lib/x86_64-linux-gnu 2>/dev/null | grep -cE 'libcudnn|libnvinfer')"

# ---- eval sets ----
if [ ! -d /root/coco500/coco500/images ]; then mkdir -p /root/coco500 && tar xzf $B/eval-set-coco500.tar.gz -C /root/coco500; fi
C=/root/coco500/coco500/images
log "eval sets: visdrone50 $(ls $V | wc -l) images, coco500 $(ls $C | wc -l) images, clip $B/pan.mp4"

# ---- checksums ----
log "== checksums"
( cd $B && sha256sum -c --ignore-missing SHA256SUMS-linux-1.2.0.txt ) 2>&1 | tee -a $L | grep -q FAILED && bad "checksums" || ok "checksums (6 bundles)"

test_bundle() {   # $1 variant  $2 list of "label|model|extra args"
  local v="$1"; shift
  local name="yolomaster-edge-linux-x64-$v-1.2.0" dir="$W/$v/yolomaster-edge-linux-x64-$v-1.2.0"
  log "================ $v"
  rm -rf "$W/$v"; mkdir -p "$W/$v"; tar xzf "$B/$name.tar.gz" -C "$W/$v" || { bad "$v unpack"; return; }
  run "$dir" --help >/dev/null && ok "$v --help (clean env)" || bad "$v --help"
  ldd "$dir/yolomaster_edge" | grep -q "not found" && bad "$v unresolved libs: $(ldd "$dir/yolomaster_edge" | grep 'not found' | tr '\n' ' ')" || ok "$v ldd closure"
  for spec in "$@"; do
    IFS='|' read -r label model extra <<< "$spec"
    local m="$dir/$model"
    # 1. inference on visdrone50 (writes annotated outputs: the full path incl. the encoder)
    out="$(T=1200 run "$dir" -m "$m" -s "$V" $extra --out "$W/$v/out_$label" --quiet)"
    if echo "$out" | grep -q "^\[summary\]" && ! echo "$out" | grep -q "\[skip\]"; then
      ok "$v $label inference visdrone50: infer=$(echo "$out" | summary_ms) ms  $(echo "$out" | ep)  outputs=$(ls "$W/$v/out_$label" 2>/dev/null | wc -l)"
    else bad "$v $label inference: $(echo "$out" | grep -E 'error|skip|failed|cannot' | head -2 | cut -c1-200)"; fi
    # 2. bench cold JSON + schema
    out="$(T=1200 run "$dir" -m "$m" -s "$V" $extra --bench cold --bench-iters 50 --bench-warmup 10 --bench-json "$W/$v/bench_$label.json" --no-save --quiet)"
    if python3 $B/bench_schema_check.py "$W/$v/bench_$label.json" --iters 50 >/dev/null 2>&1; then
      ok "$v $label bench cold: $(echo "$out" | grep -oE 'median=[0-9.]+ms' | head -1) schema OK"
    else bad "$v $label bench: $(echo "$out" | tail -1 | cut -c1-160)"; fi
    # 3. accuracy on coco500 (val protocol)
    out="$(T=2400 run "$dir" -m "$m" -s "$C" $extra --accuracy auto --no-save --quiet)"
    a="$(echo "$out" | acc)"
    if [ -n "$a" ]; then ok "$v $label accuracy coco500: mAP50-95=$a"; echo "$v,$label,$a" >> $W/accuracy.csv
    else bad "$v $label accuracy: $(echo "$out" | grep -E 'error|skip|failed' | head -1 | cut -c1-160)"; fi
    # 4. tracking on the clip (video decode + encode + tracker)
    out="$(T=1200 run "$dir" -m "$m" -s "$B/pan.mp4" $extra --track botsort --out "$W/$v/track_$label" --save-txt "$W/$v/trk_$label" --quiet)"
    n="$(ls "$W/$v/trk_$label" 2>/dev/null | wc -l)"; cols="$(head -1 "$W/$v/trk_$label"/$(ls "$W/$v/trk_$label" 2>/dev/null | head -1) 2>/dev/null | awk '{print NF}')"
    if [ "$n" -ge 80 ] && [ "${cols:-0}" = 7 ]; then ok "$v $label tracking botsort: $n frames, 7-column txt, video $(ls "$W/$v/track_$label"/*.mp4 2>/dev/null | wc -l)"
    else bad "$v $label tracking: frames=$n cols=${cols:-?} $(echo "$out" | grep -E 'error|failed' | head -1 | cut -c1-160)"; fi
  done
}

# ---- the six bundles ----
test_bundle ncnn      "ncnn-cpu|models/v0.1-seg-n_ncnn|-b ncnn"
test_bundle mnn       "mnn-cpu|models/v0.1-seg-n.mnn|-b mnn"
test_bundle onnx-cpu  "ort-cpu|models/v0.1-seg-n.onnx|-b onnx"
test_bundle onnx-cuda12 "ort-cuda|models/v0.1-seg-n.onnx|-b onnx -d cuda" "ort-cuda-cpupre|models/v0.1-seg-n.onnx|-b onnx -d cuda --cpu-preproc" "ort-cpu|models/v0.1-seg-n.onnx|-b onnx -d cpu"
test_bundle trt10-cuda12 "trt-fp16|models/v0.1-seg-n.onnx|-b trt --precision fp16" "trt-fp32-graph|models/v0.1-seg-n.onnx|-b trt --cuda-graph"
test_bundle all-cuda12 "trt-fp16|models/v0.1-seg-n.onnx|-b trt --precision fp16" "ort-cuda|models/v0.1-seg-n.onnx|-b onnx -d cuda" "ncnn-cpu|models/v0.1-seg-n_ncnn|-b ncnn" "mnn-cpu|models/v0.1-seg-n.mnn|-b mnn"

# ---- extras on the GPU bundles ----
D=$W/trt10-cuda12/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0
log "================ extras"
out="$(T=600 run "$D" -m "$D/models/v0.1-seg-n.onnx" -s "$V" -b trt --precision fp16 --bench sustained --bench-minutes 0.5 --bench-json "$W/sustained_trt.json" --no-save --quiet)"
python3 $B/bench_schema_check.py "$W/sustained_trt.json" >/dev/null 2>&1 && ok "trt sustained 30 s: $(echo "$out" | grep -oE 'throttle[^ ]* *[-+0-9.%]+' | head -1)" || bad "trt sustained: $(echo "$out" | tail -1 | cut -c1-160)"
out="$(T=600 run "$D" -m "$D/models/v0.1-seg-n.onnx" -s "$V" -b trt --precision fp16 --slicing dense --no-save --quiet --limit 5)"
echo "$out" | grep -q "^\[summary\]" && ok "trt dense slicing (5 images)" || bad "trt slicing: $(echo "$out" | tail -1 | cut -c1-160)"
D2=$W/onnx-cuda12/yolomaster-edge-linux-x64-onnx-cuda12-1.2.0
out="$(T=600 run "$D2" -m "$D2/models/v0.1-seg-n.onnx" -s "$V" -b onnx -d cuda --export-labels "$W/labels_coco" --label-format coco --no-save --quiet --limit 5)"
[ -s "$W/labels_coco/annotations.json" ] || [ "$(ls "$W/labels_coco" 2>/dev/null | wc -l)" -gt 0 ] && ok "ort-cuda --export-labels coco (5 images)" || bad "export-labels: $(echo "$out" | tail -1 | cut -c1-160)"
# engine cache: the second TRT run must not rebuild
out="$(T=600 run "$D" -m "$D/models/v0.1-seg-n.onnx" -s "$V" -b trt --precision fp16 --no-save --quiet --limit 3)"
echo "$out" | grep -q "cached engine" && ok "trt engine cache reused ($(ls "$D/models" | grep -c '\.engine$') engine files)" || bad "trt engine cache: $(echo "$out" | grep -i engine | head -1 | cut -c1-160)"

log "================ accuracy table (coco500, v0.1-seg-N boxes, val protocol)"
sort $W/accuracy.csv | tee -a $L
log "RESULT: $PASS passed, $FAIL failed"
