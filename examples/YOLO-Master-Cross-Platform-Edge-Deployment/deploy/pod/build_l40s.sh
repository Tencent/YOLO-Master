#!/usr/bin/env bash
# Build MNN (with the converter, CUDA when it compiles), the runtime core + CLI + API server with
# all four backends, and run the CLI + server test suites. Run after bootstrap_l40s.sh.
#   bash deploy/pod/build_l40s.sh [REPO=/data/yolo-master-edge]
set -euo pipefail
REPO="${1:-/data/yolo-master-edge}"
TP="$REPO/third_party"
J="$(nproc)"; [ "$J" -gt 32 ] && J=32
log() { echo "[build $(date +%H:%M:%S)] $*"; }

# ---- SDK roots (rsynced from the workstation) ----
ORT_ROOT="$TP/onnxruntime-linux-x64-gpu-1.20.1"
NCNN_ROOT="$TP/ncnn-20260526-ubuntu-2204-shared"
MNN_ROOT="$TP/mnn-src"
OPENCV_DIR="$TP/opencv-lean/lib/cmake/opencv4"
# CUDA that matches the TensorRT build (+cuda12.9), not the pod image's default /usr/local/cuda (12.4)
CUDA_HOME="${CUDA_HOME:-$(ls -d /usr/local/cuda-12.9 2>/dev/null || echo /usr/local/cuda)}"
for d in "$ORT_ROOT/include/onnxruntime_cxx_api.h" "$NCNN_ROOT/include/ncnn/net.h" "$MNN_ROOT/CMakeLists.txt" "$OPENCV_DIR"; do
  [ -e "$d" ] || { log "missing: $d"; exit 2; }
done
[ -f "$TP/uWebSockets/src/App.h" ] || bash "$REPO/scripts/server/fetch_uws.sh"

# ---- MNN: CPU (+AVX512) with the ONNX converter; CUDA backend is a best-effort second build ----
if [ ! -f "$MNN_ROOT/build/libMNN.so" ] || [ ! -x "$MNN_ROOT/build/MNNConvert" ]; then
  log "MNN cpu build (+converter)"
  cmake -S "$MNN_ROOT" -B "$MNN_ROOT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_TEST=OFF -DMNN_BUILD_DEMO=OFF \
    -DMNN_AVX512=ON -DMNN_OPENMP=OFF -DMNN_USE_THREAD_POOL=ON -DMNN_SEP_BUILD=OFF -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_BUILD_OPENCV=OFF -DMNN_BUILD_LLM=OFF -DMNN_LOW_MEMORY=OFF >/dev/null
  cmake --build "$MNN_ROOT/build" -j"$J" --target MNN MNNConvert >/dev/null
  log "MNN cpu build done: $(ls -la "$MNN_ROOT/build/libMNN.so" | awk '{print $5}') bytes"
fi
if [ ! -f "$MNN_ROOT/build_cuda/libMNN.so" ]; then
  log "MNN cuda build (best effort)"
  if cmake -S "$MNN_ROOT" -B "$MNN_ROOT/build_cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DMNN_CUDA=ON -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_TEST=OFF -DMNN_BUILD_DEMO=OFF \
       -DMNN_AVX512=ON -DMNN_OPENMP=OFF -DMNN_USE_THREAD_POOL=ON -DMNN_SEP_BUILD=OFF -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_OPENCV=OFF -DMNN_BUILD_LLM=OFF \
       -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" > "$MNN_ROOT/build_cuda.log" 2>&1 \
     && cmake --build "$MNN_ROOT/build_cuda" -j"$J" --target MNN >> "$MNN_ROOT/build_cuda.log" 2>&1; then
    log "MNN cuda build done"
  else
    log "MNN cuda build FAILED (see $MNN_ROOT/build_cuda.log); MNN rows stay CPU-only"
    rm -rf "$MNN_ROOT/build_cuda"
  fi
fi

# ---- runtime core + CLI + server (four backends) ----
for variant in cpu cuda; do
  B="$REPO/cpp/build_l40s_$variant"
  MROOT="$MNN_ROOT"; MLIB="$MNN_ROOT/build/libMNN.so"
  if [ "$variant" = cuda ]; then
    [ -f "$MNN_ROOT/build_cuda/libMNN.so" ] || { log "skip cuda variant (no MNN cuda lib)"; continue; }
    # separate root so the rpath resolves the CUDA-enabled libMNN.so (same soname as the CPU one)
    MROOT="$TP/mnn-cuda"; mkdir -p "$MROOT/lib"
    ln -sfn "$MNN_ROOT/include" "$MROOT/include"; ln -sf "$MNN_ROOT/build_cuda/libMNN.so" "$MROOT/lib/libMNN.so"
    MLIB="$MROOT/lib/libMNN.so"
  fi
  log "configure $variant -> $B"
  cmake -S "$REPO/cpp" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ORT=ON -DUSE_NCNN=ON -DUSE_MNN=ON -DUSE_TRT=ON -DBUILD_SERVER=ON \
    -DONNXRUNTIME_ROOT="$ORT_ROOT" -DNCNN_ROOT="$NCNN_ROOT" -DMNN_ROOT="$MROOT" -DMNN_LIB="$MLIB" \
    -DOpenCV_DIR="$OPENCV_DIR" -DTENSORRT_ROOT=/usr -DCUDA_INC="$CUDA_HOME/include" -DCUDART_LIB="$CUDA_HOME/lib64/libcudart.so" \
    > "$B.configure.log" 2>&1 || { tail -30 "$B.configure.log"; exit 3; }
  grep -E "backend:|parser:" "$B.configure.log"
  cmake --build "$B" -j"$J" > "$B.build.log" 2>&1 || { grep -E "error" "$B.build.log" | head -20; exit 3; }
  ls -la "$B/yolomaster_edge" "$B/server/yolomaster_server" | awk '{print $5, $9}'
done
BIN_DIR="$REPO/cpp/build_l40s_cpu"; [ -d "$REPO/cpp/build_l40s_cuda" ] && BIN_DIR="$REPO/cpp/build_l40s_cuda"
ln -sfn "$BIN_DIR" "$REPO/cpp/build_l40s"

# ---- tests ----
log "CLI regression tests"
BIN="$REPO/cpp/build_l40s/yolomaster_edge" ONNX="$REPO/models/esmoe_n_visdrone_sim.onnx" NCNN="$REPO/models/esmoe_n_visdrone_ncnn" \
  DIR="$REPO/visdrone50/images/val" YAML="$REPO/visdrone50/visdrone50.yaml" bash "$REPO/tests/run_tests.sh" | tail -3
log "server integration tests"
pip install -q pytest websocket-client opencv-python-headless 2>/dev/null || true
BIN="$REPO/cpp/build_l40s/server/yolomaster_server" CLI="$REPO/cpp/build_l40s/yolomaster_edge" \
  ONNX="$REPO/models/esmoe_n_visdrone_sim.onnx" NCNN="$REPO/models/esmoe_n_visdrone_ncnn" DIR="$REPO/visdrone50/images/val" \
  bash "$REPO/tests/run_server_tests.sh" | tail -4
log "done"
