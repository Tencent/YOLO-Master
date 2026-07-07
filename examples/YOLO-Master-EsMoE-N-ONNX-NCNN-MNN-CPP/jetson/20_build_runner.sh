#!/usr/bin/env bash
# Build the C++ edge runner on aarch64 (ONNX backend, CPU) and run it.
# The GPU ceiling is measured by trtexec (10_trt_bench.sh); this proves the portable
# runner builds+runs unchanged on the Jetson (same source as Linux/Windows).
set -e
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
cmake .. -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON -DUSE_NCNN=OFF \
         -DONNXRUNTIME_ROOT="$ORT_DIR" 2>&1 | grep -iE "backend:|error" || true
make -j"$(nproc)" 2>&1 | grep -iE "error|Built target" | tail -1
BIN="$ROOT/cpp/build_jetson/yolomaster_edge"
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
