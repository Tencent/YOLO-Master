#!/usr/bin/env bash
# Build ONNXRuntime from source WITH the TensorRT EP on the Jetson (aarch64 / CUDA / TRT).
# For bleeding-edge CUDA (e.g. 13.x) where no prebuilt onnxruntime-gpu wheel exists.
# SLOW: hours on a 4GB Nano. Needs swap (see 21/OOM notes) + low --parallel (nvcc is memory-hungry).
# Produces $ORT_OUT/{lib,include} -> then: ORT_ROOT=$ORT_OUT bash jetson/22_build_ort_trt.sh
#
#   ORT_REF=main JOBS=2 bash 23_build_ort_source.sh
set -e
ORT_REF="${ORT_REF:-main}"                    # pin a release tag if main breaks (e.g. v1.20.1)
JOBS="${JOBS:-2}"                             # KEEP LOW on 4GB — each nvcc wants ~1-2GB (swap helps)
SRC="${SRC:-$HOME/onnxruntime}"
ORT_OUT="${ORT_OUT:-$HOME/ort-jetson}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "==== prereqs ===="
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "  nvcc missing. Install the CUDA toolkit:  sudo apt install -y cuda-toolkit"
  echo "  then re-run.  (nvidia-jetpack is runtime-only; the compiler is in cuda-toolkit.)"
  exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version | tail -1
cmake --version | head -1                     # ORT wants a recent cmake; build.sh bootstraps if too old
[ -f /usr/include/aarch64-linux-gnu/NvInfer.h ] || echo "  [warn] TRT headers not at /usr/include/aarch64-linux-gnu"
swapon --show | grep -q . || echo "  [warn] NO SWAP — add 8G+ or the compile gets OOM-killed"
echo "  free disk on $HOME:"; df -h "$HOME" | tail -1
echo "  apt build deps: sudo apt install -y build-essential python3-dev python3-numpy libpython3-dev git"
echo

echo "==== clone ONNXRuntime ($ORT_REF) ===="
[ -d "$SRC/.git" ] || git clone --recursive --branch "$ORT_REF" --depth 1 \
  https://github.com/microsoft/onnxruntime "$SRC"
cd "$SRC"

echo "==== build: Release, shared lib, CUDA + TensorRT, sm87 (HOURS) ===="
./build.sh --config Release --build_shared_lib --parallel "$JOBS" --skip_tests --allow_running_as_root \
  --use_cuda     --cuda_home "$CUDA_HOME" --cudnn_home /usr \
  --use_tensorrt --tensorrt_home /usr \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF

echo "==== assemble ORT_ROOT ($ORT_OUT) ===="
mkdir -p "$ORT_OUT/lib" "$ORT_OUT/include"
cp -av build/Linux/Release/libonnxruntime.so*        "$ORT_OUT/lib/"
find include -name 'onnxruntime_*_api.h' -exec cp -v {} "$ORT_OUT/include/" \;
find include -name 'onnxruntime_*.h'     -exec cp -v {} "$ORT_OUT/include/" \; 2>/dev/null || true
echo
echo "done. next:"
echo "  ORT_ROOT=$ORT_OUT bash jetson/22_build_ort_trt.sh"
echo "  then run the .onnx on the GPU:  --device trt   (first run builds+caches the TRT engine)"
