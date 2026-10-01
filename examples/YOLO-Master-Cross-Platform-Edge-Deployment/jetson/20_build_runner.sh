#!/usr/bin/env bash
# Build the C++ edge runner on aarch64 (ONNX backend, CPU) and run it.
# The GPU ceiling is measured by trtexec (10_trt_bench.sh); this proves the portable
# runner builds+runs unchanged on the Jetson (same source as Linux/Windows).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"                      # edge repo root (cpp/ lives here)
ORT_VER=1.20.1
ORT_DIR="$ROOT/third_party/onnxruntime-linux-aarch64-$ORT_VER"

echo "==================== ONNXRuntime aarch64 SDK ===================="
if [ ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ]; then
  mkdir -p "$ROOT/third_party"
  wget -qO /tmp/ort-aarch64.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-aarch64-${ORT_VER}.tgz"
  tar xzf /tmp/ort-aarch64.tgz -C "$ROOT/third_party"
fi
echo "  $ORT_DIR"

echo "==================== build (aarch64, ORT backend, PORTABLE) ===================="
cd "$ROOT/cpp"
rm -rf build_jetson && mkdir build_jetson && cd build_jetson
# PORTABLE=ON strips videoio (image-only) to avoid the ffmpeg/GStreamer closure.
# When the lean ffmpeg OpenCV from 21_build_trt_runner.sh is cached, that concern
# is gone: link it and build the FULL runner (video sources included). Without it,
# fall back to the original image-only portable build against system OpenCV.
# ncnn backend: on when the on-device aarch64 build exists (jetson/24_build_ncnn.sh)
NCNN_AARCH64="$ROOT/third_party/ncnn-aarch64-20260526"
if [ -f "$NCNN_AARCH64/include/ncnn/net.h" ]; then
  NCNN_ARGS="-DUSE_NCNN=ON -DNCNN_ROOT=$NCNN_AARCH64"
  echo "  ncnn backend: ON (aarch64 build at $NCNN_AARCH64)"
else
  NCNN_ARGS="-DUSE_NCNN=OFF"
fi
# MNN backend: on when the on-device aarch64 build exists (jetson/25_build_mnn.sh)
MNN_AARCH64="$ROOT/third_party/mnn-src-aarch64"
if [ -f "$MNN_AARCH64/build/libMNN.so" ]; then
  MNN_ARGS="-DUSE_MNN=ON -DMNN_ROOT=$MNN_AARCH64"
  echo "  MNN backend: ON (aarch64 build at $MNN_AARCH64)"
else
  MNN_ARGS="-DUSE_MNN=OFF"
fi
OCV_LEAN="$ROOT/third_party/opencv-lean"
if [ -f "$OCV_LEAN/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
  MODE_ARGS="-DPORTABLE=OFF -DOpenCV_DIR=$OCV_LEAN/lib/cmake/opencv4"
  echo "  full build (video on): lean ffmpeg OpenCV at $OCV_LEAN"
else
  MODE_ARGS="-DPORTABLE=ON"
  echo "  portable build (image-only): no cached lean OpenCV"
fi
# full logs land in build_jetson/{configure,build}.log; the console shows the backend lines only
cmake .. -DCMAKE_BUILD_TYPE=Release $MODE_ARGS $NCNN_ARGS $MNN_ARGS \
         -DONNXRUNTIME_ROOT="$ORT_DIR" 2>&1 | tee configure.log | grep -iE "backend:|error" || true
cmake --build . --parallel "$(nproc)" 2>&1 | tee build.log | grep -iE "error|Built target" | tail -1 || true
BIN="$ROOT/cpp/build_jetson/yolomaster_edge"
[ -x "$BIN" ] || { echo "build completed without an executable: $BIN (see build_jetson/build.log)" >&2; exit 1; }
echo "  binary: $BIN"

echo "==================== run ===================="
M="$ROOT/jetson/models/esmoe_n_visdrone_sim.onnx"
IMG="${1:-}"     # optional: pass a test image path as arg 1
if [ -n "$IMG" ] && [ -f "$IMG" ]; then
  "$BIN" --model "$M" --source "$IMG" --device cpu --conf 0.25 --out out
else
  echo "  built OK. Run on an image:"
  echo "    $BIN --model $M --source <image_or_dir> --device cpu --out out"
  echo "  (this is the CPU path; the GPU FPS ceiling is trtexec / 10_trt_bench.sh)"
fi
