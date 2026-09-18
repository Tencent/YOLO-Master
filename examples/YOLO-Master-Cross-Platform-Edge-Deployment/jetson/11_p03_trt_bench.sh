#!/usr/bin/env bash
# Project03 engine building + inference on the Orin: pruned v0.1 COCO models
# across scales, every precision that matters.
#
#   bash jetson/11_p03_trt_bench.sh            # v0.1-N (default)
#   bash jetson/11_p03_trt_bench.sh n s        # multiple scales
#   bash jetson/11_p03_trt_bench.sh s m l      # the scaling set (see memory note)
#
# Per scale, four engines:
#   fp32       honest baseline (no TF32 on Orin GPU)
#   fp16       THE deployment engine (criterion holder: -41.9% at N)
#   int8-naive speed ceiling probe (uncalibrated implicit; accuracy meaningless,
#              answers "can int8 beat fp16 at this width on Orin" - datacenter
#              result: yes at S/M by 5-8%, no at N)
#   int8-qdq   the accuracy artifact (explicit quantization, scales in-graph, no
#              calibrator needed; per-scale sensitive sets baked into the ONNX:
#              N/S stem-pair, M modules 4-6, L modules 7-9)
#
# Models fetched from the mdb drop (override with MDB=user@host:path).
# 4GB Orin Nano: the L build will likely OOM the builder - run headless
# (sudo systemctl isolate multi-user.target) and/or WORKSPACE=256.
# Latency = trtexec "GPU Compute Time" median (model-only). End-to-end belongs
# to the runner; commands are printed at the end.
set -e
cd "$(dirname "$0")"
MDB="${MDB:-root@185.76.11.54:/yolotmp/jetson-p03}"
SCALES=("${@:-n}")

TRTEXEC="${TRTEXEC:-}"
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
  for c in "$(command -v trtexec 2>/dev/null)" \
           /usr/src/tensorrt/bin/trtexec \
           "$(find /usr -name trtexec -type f 2>/dev/null | head -1)"; do
    [ -n "$c" ] && [ -x "$c" ] && { TRTEXEC="$c"; break; }
  done
fi
[ -n "$TRTEXEC" ] && [ -x "$TRTEXEC" ] || {
  echo "trtexec not found. sudo apt install -y tensorrt  (then re-run)"; exit 1; }
echo "using trtexec: $TRTEXEC"

WORKSPACE="${WORKSPACE:-512}"
OPT="${OPT:-3}"

# the 4GB Nano's builder OOMs on the M/L QDQ graphs - back it with a swapfile.
# SWAP_GB=0 skips; the file persists across runs (delete /swapfile-p03 to reclaim).
SWAP_GB="${SWAP_GB:-8}"
if [ "$SWAP_GB" -gt 0 ] && [ "$(free -m | awk '/Swap:/{print $2}')" -lt $((SWAP_GB * 900)) ]; then
  echo "==================== enabling ${SWAP_GB}G build swap ===================="
  if [ ! -f /swapfile-p03 ]; then
    sudo fallocate -l "${SWAP_GB}G" /swapfile-p03 ||       sudo dd if=/dev/zero of=/swapfile-p03 bs=1M count=$((SWAP_GB * 1024))
    sudo chmod 600 /swapfile-p03
    sudo mkswap /swapfile-p03
  fi
  sudo swapon /swapfile-p03 2>/dev/null || true
  free -h | sed 's/^/  /'
fi

mkdir -p models engines
[ -f models/p03-metadata.yaml ] || scp "$MDB/p03-metadata.yaml" models/p03-metadata.yaml

fetch() { [ -f "models/$1" ] || scp "$MDB/$1" "models/$1"; }

bench() {  # tag  onnx  extra-flags...
  local tag="$1" onnx="$2"; shift 2
  local log="engines/trtexec_${tag}.log"
  echo "==================== $tag ===================="
  "$TRTEXEC" --onnx="$onnx" --saveEngine="engines/${tag}.engine" \
             --memPoolSize=workspace:"${WORKSPACE}" \
             --builderOptimizationLevel="${OPT}" "$@" 2>&1 | tee "$log" | \
             grep -E "Engine built|error|Error" || true
  cp models/p03-metadata.yaml "engines/${tag}.metadata.yaml"
  # the stats line, not "Total GPU Compute Time" (and never the D2H section)
  grep -E "GPU Compute Time: min" "$log" | tail -1 | sed 's/^/  /'
  echo
}

for SC in "${SCALES[@]}"; do
  case "$SC" in
    n) PLAIN=yolo-master-v01n-coco-pruned.onnx; QDQ=yolo-master-v01n-coco-qdq.onnx ;;
    s) PLAIN=yolo-master-v01s-coco-pruned.onnx; QDQ=yolo-master-v01s-coco-qdq.onnx ;;
    m) PLAIN=yolo-master-v01m-coco-pruned.onnx; QDQ=yolo-master-v01m-coco-qdq.onnx ;;
    l) PLAIN=yolo-master-v01l-coco-pruned.onnx; QDQ=yolo-master-v01l-coco-qdq.onnx ;;
    *) echo "unknown scale '$SC' (use n/s/m/l)"; exit 1 ;;
  esac
  fetch "$PLAIN"; fetch "$QDQ"
  bench "p03_${SC}_fp32"       "models/$PLAIN"
  bench "p03_${SC}_fp16"       "models/$PLAIN" --fp16
  bench "p03_${SC}_int8naive"  "models/$PLAIN" --int8 --fp16
  bench "p03_${SC}_int8qdq"    "models/$QDQ"   --int8 --fp16
done

echo "==================== summary (GPU compute median, ms) ===================="
for f in engines/trtexec_p03_*.log; do
  name=$(basename "$f" .log | sed 's/trtexec_//')
  med=$(grep -E "GPU Compute Time: min" "$f" | tail -1 | grep -oE "median = [0-9.]+" | awk '{print $3}')
  printf "  %-22s %s\n" "$name" "${med:-build failed}"
done
echo
echo "End-to-end via the runner (sidecars already placed next to each engine):"
echo "  ./build/yolomaster --engine jetson/engines/p03_<scale>_fp16.engine --input <img-or-dir> --bench"
echo "fp16 is the deployment engine; int8-qdq is the accuracy artifact;"
echo "int8-naive numbers are speed ceilings only (accuracy is garbage by design)."
