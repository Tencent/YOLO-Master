#!/usr/bin/env bash
# Orin CPU (aarch64) mixed-INT8 vs fp16 latency: rewrite of the 9d0f537 tempo-ncnn/run_orin.sh
# for cpp/tools/ncnn_bench. Builds the bench against the aarch64 ncnn install
# (third_party/ncnn-aarch64-20260526, from jetson/24_build_ncnn.sh), renders the probe blob
# from a real VisDrone image with the runtime letterbox, then runs the p03 pair
# (models/p03_v01n_ncnn + models/p03_v01n-int8_ncnn) and the EsMoE pair
# (models/esmoe_n_visdrone_ncnn + models/esmoe_n_visdrone-int8_ncnn) at 1/2/6 threads,
# 3 interleaved rounds, all four CPU variants (fp32, fp16, int8+fp32, int8+fp16) plus the Vulkan GPU
# variant (jetson/24_build_ncnn.sh builds ncnn with NCNN_VULKAN=ON; runs once per model, threads=gpu).
# Everything is tee'd to results/int8_bench/<hostname>.log (+ one JSON per model).
#
# Usage:  bash jetson/26_ncnn_int8_bench.sh            # from any cwd
#   env:  NCNN=<ncnn install prefix>  PROBE_IMAGE=<jpg>  THREADS=1,2,6  ITERS=100  ROUNDS=3
#         MODELS="models/p03_v01n_ncnn models/esmoe_n_visdrone_ncnn"  LABEL=<device label>
#
# Read the verdict per (model, threads) from the [ncnn_bench] device=... lines: "INT8 wins at T"
# iff int8 valid=1 with dets within +/-10% of the float variant, median(int8) < 0.97*median(float)
# and p90(int8) < median(float); |delta| < 3% is a tie (bandwidth-bound, as on Orin before).
# Orin cores are homogeneous, so "big" == all cores and --powersave 2 is a no-op there.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

VER=20260526
NCNN="${NCNN:-$ROOT/third_party/ncnn-aarch64-$VER}"
THREADS="${THREADS:-1,2,6}"
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-20}"
ROUNDS="${ROUNDS:-3}"
MODELS="${MODELS:-models/p03_v01n_ncnn models/esmoe_n_visdrone_ncnn}"
LABEL="${LABEL:-$(hostname)}"
OUT="$ROOT/results/int8_bench"
LOG="$OUT/$(hostname).log"
PROBE="$OUT/probe_640.f32"
BUILD="$ROOT/cpp/tools/build-aarch64"
PY="${PY:-python3}"

mkdir -p "$OUT"

[ -f "$NCNN/include/ncnn/net.h" ] && [ -e "$NCNN/lib/libncnn.so" ] || {
  echo "ncnn aarch64 install missing: $NCNN  (run: bash jetson/24_build_ncnn.sh)"; exit 1; }

echo "==================== build cpp/tools/ncnn_bench ($NCNN) ===================="
cmake -S "$ROOT/cpp/tools" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DNCNN_INSTALL="$NCNN" >/dev/null
cmake --build "$BUILD" -j"$(nproc)" >/dev/null
BENCH="$BUILD/ncnn_bench"
[ -x "$BENCH" ] || { echo "build failed: $BENCH missing"; exit 1; }

echo "==================== probe blob (runtime letterbox of a real image) ===================="
if [ ! -f "$PROBE" ]; then
  PROBE_IMAGE="${PROBE_IMAGE:-$(ls "$ROOT"/visdrone50/images/val/*.jpg 2>/dev/null | head -1)}"
  [ -n "$PROBE_IMAGE" ] && [ -f "$PROBE_IMAGE" ] || {
    echo "no probe image: set PROBE_IMAGE=<jpg> (needs $PY with numpy + cv2)"; exit 1; }
  "$PY" "$ROOT/scripts/make_probe_f32.py" --image "$PROBE_IMAGE" --imgsz 640 \
    --out "$PROBE" --meta "$OUT/probe_640.json"
fi
echo "  probe: $PROBE"

{
  echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) label=$LABEL ncnn=$NCNN"
  echo "== kernel=$(uname -r) nvpmodel=$(nvpmodel -q 2>/dev/null | tr '\n' ' ' || true)"
  echo "== threads=$THREADS iters=$ITERS warmup=$WARMUP rounds=$ROUNDS"
  for M in $MODELS; do
    NAME="$(basename "$M")"
    INT8="${M%_ncnn}-int8_ncnn"
    echo
    echo "==================== $NAME  (+ ${INT8#models/}) ===================="
    [ -f "$M/model.ncnn.param" ] || { echo "  missing float model $M - skipped"; continue; }
    [ -f "$INT8/model.ncnn.param" ] || echo "  note: int8 sibling $INT8 absent - int8 variants will be skipped"
    "$BENCH" "$M" --input "$PROBE" --shape 3,640,640 \
      --threads "$THREADS" --iters "$ITERS" --warmup "$WARMUP" --rounds "$ROUNDS" \
      --variants fp32,fp16,int8+fp32,int8+fp16,vulkan --powersave 2 \
      --label "$LABEL" --json "$OUT/$LABEL-$NAME.json" --conf 0.25
  done
  echo
  echo "== done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$LOG"

echo
echo "log: $LOG"
echo "summary (median-of-round-medians):"
grep -h '^\[ncnn_bench\] device=' "$LOG" | tail -n 40 | \
  sed -E 's/.*model=([^ ]+) variant=([^ ]+) threads=([0-9]+) .*median_ms=([0-9.]+) min_median_ms=([0-9.]+) p90_ms=([0-9.]+) dets=([0-9]+) .*valid=([01]).*/  \1  \2  T=\3  median=\4ms  min=\5ms  p90=\6ms  dets=\7  valid=\8/'
