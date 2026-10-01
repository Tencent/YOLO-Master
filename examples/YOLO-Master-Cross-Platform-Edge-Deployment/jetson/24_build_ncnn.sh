#!/usr/bin/env bash
# Build ncnn (CPU + Vulkan) from source on the Jetson and install it into
# third_party/ncnn-aarch64-<ver>, matching the layout cpp/CMakeLists.txt expects
# (include/ncnn/net.h + lib/libncnn.so). There is no aarch64 prebuilt release; the
# x86 SDK in third_party/ncnn is useless here. One-time (~10-15 min on an Orin),
# cached afterward. 20_build_runner.sh picks the install up automatically.
#
# The tag matches the x86 SDK (1.0.20260526) so converted .param/.bin files behave
# identically across platforms. Conversion itself always happens on x64 (pnnx is
# x86-only); the model files are portable.
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

VER=20260526
SRC="$ROOT/third_party/ncnn-src"
DST="$ROOT/third_party/ncnn-aarch64-$VER"

if [ -f "$DST/include/ncnn/net.h" ] && [ -e "$DST/lib/libncnn.so" ]; then
  echo "ncnn aarch64 already built: $DST"
  exit 0
fi

echo "==================== deps (Vulkan headers + glslang) ===================="
sudo apt-get install -y -qq libvulkan-dev glslang-dev glslang-tools libomp-dev 2>/dev/null || \
  apt-get install -y -qq libvulkan-dev glslang-dev glslang-tools libomp-dev

echo "==================== ncnn source (tag $VER) ===================="
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$VER" --recurse-submodules --shallow-submodules \
    https://github.com/Tencent/ncnn.git "$SRC"
fi

echo "==================== build (CPU + Vulkan, shared) ===================="
cmake -S "$SRC" -B "$SRC/build-aarch64" -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=ON \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_PYTHON=OFF \
  -DCMAKE_INSTALL_PREFIX="$DST"
cmake --build "$SRC/build-aarch64" -j"$(nproc)"
cmake --install "$SRC/build-aarch64"

[ -f "$DST/include/ncnn/net.h" ] || { echo "ERROR: install layout unexpected"; exit 1; }
echo "  installed: $DST"
echo "  next: bash jetson/20_build_runner.sh   (auto-detects this install)"
